import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


# Noise Addition (Zhu et al., 2023)
def noise_addition(x_hat_0t, x_t, sigma_t_minus_1, xi=0.5):
    """
    Add noise to update latent state in HQS (Zhu et al., 2023).
    
    Args:
        x_hat_0t (torch.Tensor): INR-reconstructed image (batch, 1, height, width).
        x_t (torch.Tensor): Current latent state (batch, 1, height, width).
        sigma_t_minus_1 (torch.Tensor): Noise level at t-1.
        xi (float): Noise balance parameter (default: 0.5).
    
    Returns:
        torch.Tensor: Updated latent state x_{t-1}.
    """
    epsilon = torch.randn_like(x_hat_0t)
    term1 = torch.sqrt(torch.tensor(xi, device=x_hat_0t.device)) * epsilon
    term2 = torch.sqrt(torch.tensor(1 - xi, device=x_hat_0t.device)) * (x_t - x_hat_0t)
    x_t_minus_1 = x_hat_0t + sigma_t_minus_1.view(-1, 1, 1, 1) * (term1 + term2)
    return x_t_minus_1

# HQS Framework
class HQSFramework:
    def __init__(self, sampler, inr, degradation_model, lambda_mc=1.0, alpha_ssim=0.1, 
                 lambda_pc=1.0, T=1000, num_iterations=10, inr_steps=100, inr_lr=1e-3):
        """
        Half Quadratic Splitting (HQS) framework for LF MRI enhancement.
        
        Args:
            sampler (Sampler): VE-SDE sampler for generating priors.
            inr (INR): Implicit Neural Representation for coordinate-based reconstruction.
            degradation_model (DegradationModel): LF MRI degradation (blur, downsample, noise).
            lambda_mc (float): Weight for measurement consistency loss (default: 1.0).
            alpha_ssim (float): Weight for SSIM loss (default: 0.1).
            lambda_pc (float): Weight for prior consistency loss (default: 1.0).
            T (int): Number of diffusion timesteps (default: 1000).
            num_iterations (int): Number of HQS iterations (default: 10).
            inr_steps (int): Number of INR optimization steps per iteration (default: 100).
            inr_lr (float): Learning rate for INR optimization (default: 1e-3).
        """
        self.sampler = sampler
        self.inr = inr
        self.degradation_model = degradation_model
        self.lambda_mc = lambda_mc
        self.alpha_ssim = alpha_ssim
        self.lambda_pc = lambda_pc
        self.T = T
        self.num_iterations = num_iterations
        self.inr_steps = inr_steps
        self.inr_lr = inr_lr

    def compute_ssim(self, x, y):
        """
        Compute SSIM between two images.
        
        Args:
            x, y (torch.Tensor): Images (batch, 1, height, width).
        
        Returns:
            torch.Tensor: SSIM value.
        """
        x_np = x.cpu().numpy().squeeze()
        y_np = y.cpu().numpy().squeeze()
        return torch.tensor(ssim(x_np, y_np, data_range=2.0), device=x.device)

    def inr_optimization(self, y, x_0t, coords, device):
        """
        Optimize INR to minimize MC and PC losses.
        
        Args:
            y (torch.Tensor): LF MRI input (batch, 1, 128, 128).
            x_0t (torch.Tensor): VE-SDE prior (batch, 1, 256, 256).
            coords (torch.Tensor): Coordinates (batch, num_points, 2).
            device (torch.device): Device for computation.
        
        Returns:
            torch.Tensor: Optimized INR output (batch, 1, 256, 256).
        """
        optimizer = torch.optim.Adam(self.inr.parameters(), lr=self.inr_lr)
        for _ in range(self.inr_steps):
            optimizer.zero_grad()
            x_hat_0t = self.inr(coords).reshape(-1, 1, 256, 256)
            y_hat = self.degradation_model(x_hat_0t)
            l2_mc = F.mse_loss(y_hat, y)
            ssim_loss = 1 - self.compute_ssim(y_hat, y)
            mc_loss = self.lambda_mc * l2_mc + self.alpha_ssim * ssim_loss
            pc_loss = self.lambda_pc * F.mse_loss(x_hat_0t, x_0t)
            loss = mc_loss + pc_loss
            loss.backward()
            optimizer.step()
        return x_hat_0t.detach()

    def get_vesde_prior(self, shape, t, device):
        """
        Generate VE-SDE prior using Sampler's predictor-corrector sampling.
        
        Args:
            shape (tuple): Shape of output (batch, 1, 256, 256).
            t (torch.Tensor): Timestep (scalar in [0, 1]).
            device (torch.device): Device for computation.
        
        Returns:
            torch.Tensor: Prior x_{0|t} (batch, 1, 256, 256).
        """
        with torch.no_grad():
            with self.sampler.ema.average_parameters():
                rx_k = self.sampler.sde.prior_sampling(shape).to(device)
                timesteps = torch.linspace(self.sampler.sde.T, self.sampler.config.sampling.eps, 
                                        self.sampler.config.sampling.discretization_steps, device=device)
                rt = torch.ones(shape[0], device=device) * timesteps[int(t * (self.T - 1))]
                rx_kp1, rx_kp1_no_noise = self.sampler.corrector(self.sampler.model, rx_k, rt)
                rx_kp1, rx_kp1_no_noise = self.sampler.predictor(self.sampler.model, rx_kp1, rt)
                x_0t = self.sampler.data_inverse_scaler(rx_kp1_no_noise if self.sampler.config.sampling.noise_removal else rx_kp1)
        return x_0t

    def forward(self, y, device):
        """
        Enhance LF MRI using HQS.
        
        Args:
            y (torch.Tensor): LF MRI input (batch, 1, 128, 128).
            device (torch.device): Device for computation.
        
        Returns:
            torch.Tensor: Enhanced HF MRI (batch, 1, 256, 256).
        """
        assert y.dim() == 4 and y.shape[1] == 1 and y.shape[2:] == (128, 128), f"Expected y shape (batch, 1, 128, 128), got {y.shape}"
        batch_size = y.shape[0]
        height, width = 256, 256
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=-1).reshape(1, -1, 2).repeat(batch_size, 1, 1)
        x_t = self.sampler.sde.prior_sampling((batch_size, 1, height, width)).to(device)
        for i in range(self.num_iterations):
            t = torch.tensor([(self.T - i) / self.T], device=device)
            x_0t = self.get_vesde_prior((batch_size, 1, height, width), t, device)
            x_hat_0t = self.inr_optimization(y, x_0t, coords, device)
            _, sigma_t_minus_1 = self.sampler.sde.get_sigma(t)
            x_t = noise_addition(x_hat_0t, x_t, sigma_t_minus_1)
        x_final = self.inr(coords).reshape(batch_size, 1, height, width)
        return self.sampler.data_inverse_scaler(x_final)
