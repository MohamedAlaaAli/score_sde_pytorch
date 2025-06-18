import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Degradation Model (Eq. 2: Gaussian blur, downsampling, noise)
class DegradationModel:
    def __init__(self, blur_sigma=1.5, downsample_factor=2, noise_std=0.05):
        self.blur_sigma = blur_sigma
        self.downsample_factor = downsample_factor
        self.noise_std = noise_std

    def gaussian_blur(self, x):
        # Implement Gaussian blur using a 2D Gaussian kernel
        kernel_size = int(6 * self.blur_sigma + 1)  # Ensure odd size
        if kernel_size % 2 == 0:
            kernel_size += 1
        x_range = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=x.device)
        x_grid, y_grid = torch.meshgrid(x_range, x_range, indexing='ij')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * self.blur_sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        x_padded = F.pad(x, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
        blurred = F.conv2d(x_padded, kernel, padding=0)
        return blurred.view_as(x)

    def downsample(self, x):
        return F.interpolate(x, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=False)

    def add_noise(self, x):
        return x + torch.randn_like(x) * self.noise_std

    def __call__(self, x):
        x = self.gaussian_blur(x)
        x = self.downsample(x)
        x = self.add_noise(x)
        return x

def noise_addition(x_hat_0t, x_t, sigma_t_minus_1, xi=0.5):
    """
    Add noise to refined prior x_hat_0t to produce x_{t-1}.
    Formula: x_{t-1} = x_hat_0t + sigma_{t-1} * (sqrt(xi) * epsilon + sqrt(1-xi) * (x_t - x_hat_0t))
    """
    epsilon = torch.randn_like(x_hat_0t)
    term1 = torch.sqrt(torch.tensor(xi)) * epsilon
    term2 = torch.sqrt(torch.tensor(1 - xi)) * (x_t - x_hat_0t)
    x_t_minus_1 = x_hat_0t + sigma_t_minus_1.view(-1, 1, 1, 1) * (term1 + term2)
    return x_t_minus_1

# HQS Framework
class HQSFramework:
    def __init__(self, vesde, score_net, inr, degradation_model, lambda_mc=1.0, alpha_ssim=0.1, lambda_pc=1.0,
                 T=1000, num_iterations=10, inr_steps=100, inr_lr=1e-3):
        self.vesde = vesde
        self.score_net = score_net
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
        """Compute SSIM between two images."""
        x_np = x.cpu().numpy().squeeze()
        y_np = y.cpu().numpy().squeeze()
        return torch.tensor(ssim(x_np, y_np, data_range=2.0), device=x.device)

    def inr_optimization(self, y, x_0t, coords, device):
        """Optimize INR to minimize MC and PC losses."""
        optimizer = torch.optim.Adam(self.inr.parameters(), lr=self.inr_lr)
        for _ in range(self.inr_steps):
            optimizer.zero_grad()
            x_hat_0t = self.inr(coords).reshape(-1, 1, 256, 256)  # (batch, 1, 256, 256)
            
            # Measurement Consistency (MC) Loss: L2 + SSIM
            y_hat = self.degradation_model(x_hat_0t)
            l2_mc = F.mse_loss(y_hat, y)
            ssim_loss = 1 - self.compute_ssim(y_hat, y)
            mc_loss = self.lambda_mc * l2_mc + self.alpha_ssim * ssim_loss
            
            # Prior Consistency (PC) Loss: L2
            pc_loss = self.lambda_pc * F.mse_loss(x_hat_0t, x_0t)
            
            # Total loss
            loss = mc_loss + pc_loss
            loss.backward()
            optimizer.step()
        return x_hat_0t.detach()

    def forward(self, y, device):
        """
        Run HQS framework to enhance LF MRI image y.
        Input: y (batch, 1, height/downsample_factor, width/downsample_factor)
        Output: enhanced image (batch, 1, 256, 256)
        """
        batch_size = y.shape[0]
        height, width = 256, 256
        # Generate coordinate grid
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=-1).reshape(1, -1, 2).repeat(batch_size, 1, 1)
        
        # Initialize x_t as noisy image
        x_t = torch.randn(batch_size, 1, height, width, device=device) * self.vesde.sigma_max
        
        # HQS iterations
        for i in range(self.num_iterations):
            t = torch.tensor([(self.T - i) / self.T], device=device)  # Linear schedule: t from 1 to 0
            # Step 1: Diffusion model (Tweedie denoising)
            with torch.no_grad():
                score = self.score_net(x_t, t)
                sigma_t, sigma_t_minus_1 = self.vesde.get_sigma(t)
                x_0t = x_t + sigma_t.view(-1, 1, 1, 1) ** 2 * score
            
            # Step 2: INR optimization
            x_hat_0t = self.inr_optimization(y, x_0t, coords, device)
            
            # Step 3: Noise addition
            x_t = noise_addition(x_hat_0t, x_t, sigma_t_minus_1, xi=0.5)
        
        # Final INR output
        x_final = self.inr(coords).reshape(batch_size, 1, height, width)
        return x_final