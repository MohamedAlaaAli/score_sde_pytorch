import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

class DiffDeuR(nn.Module):
    def __init__(
        self,
        sampler,
        inr,
        degradation_model,
        num_steps=20,
        xi=0.5,
        lambda_df=1.0,
        image_size=256,
    ):
        super().__init__()
        self.sampler = sampler  # contains .sde and pretrained score model s_theta*
        self.inr = inr  # coordinate-based MLP
        self.degradation_model = degradation_model
        self.num_steps = num_steps
        self.xi = xi
        self.lambda_df = lambda_df
        self.image_size = image_size

        coords = self._make_coordinates(image_size)
        self.register_buffer("coords", coords.cpu())  # (1, H*W, 2)

    def _make_coordinates(self, size):
        x_coords = torch.linspace(-1, 1, size)
        y_coords = torch.linspace(-1, 1, size)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="ij")
        coords = torch.stack([x_grid, y_grid], dim=-1).reshape(1, -1, 2)
        return coords

    def compute_ssim(self, x, y):
        x_np = x.detach().cpu().numpy().squeeze()
        y_np = y.detach().cpu().numpy().squeeze()
        return torch.tensor(ssim(x_np, y_np, data_range=2.0), device="cpu")

    # def forward(self, y):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     B, _, H_lr, W_lr = y.shape
    #     H_hr = W_hr = self.image_size

    #     # Step 1: Initialize x_T from prior on GPU
    #     with torch.no_grad():
    #         x_t = self.sampler.sde.prior_sampling((B, 1, H_hr, W_hr)).to(device)

        
    #     for step in tqdm(range(self.num_steps), desc="Diffusion Steps", leave=True):
    #         with torch.no_grad():
    #             # Convert step to t in [0, 1]
    #             t = torch.tensor([(self.num_steps - step) / self.num_steps], device=device)

    #             # Get sigma_t and sigma_{t-1}
    #             _, sigma_t = self.sampler.sde.get_sigma(t)
    #             t_prev = torch.tensor([(self.num_steps - step - 1) / self.num_steps], device=device)
    #             _, sigma_t_minus_1 = self.sampler.sde.get_sigma(t_prev)

    #             # 1. Data-prior subproblem: Denoising with Tweedie's formula
    #             score = self.sampler.model(x_t, sigma_t)
    #             x_0t = x_t + sigma_t.view(-1, 1, 1, 1) ** 2 * score
    #             x_0t = x_0t.detach()  # ✨ important line ✨


    #         # 2. Data-fidelity subproblem: Optimize INR on CPU
    #         inr_opt = torch.optim.Adam(self.inr.parameters(), lr=1e-3)
    #         for _ in range(100):
    #             inr_opt.zero_grad()
    #             x_hat_0t = self.inr(self.coords).reshape(B, 1, H_hr, W_hr)
    #             y_hat = self.degradation_model(x_hat_0t)
    #             loss_df = F.mse_loss(y_hat, y.cpu())
    #             loss_pc = F.mse_loss(x_hat_0t, x_0t.cpu())
    #             loss = self.lambda_df * loss_df + loss_pc
    #             loss.backward()
    #             inr_opt.step()

    #         x_hat_0t = self.inr(self.coords).reshape(B, 1, H_hr, W_hr).detach()

    #         # 3. Update x_{t-1} on GPU
    #         eps = torch.randn_like(x_hat_0t).to(device)
    #         term1 = torch.sqrt(torch.tensor(self.xi, device=device)) * eps
    #         term2 = torch.sqrt(torch.tensor(1 - self.xi, device=device)) * (x_t - x_hat_0t.to(device))
    #         x_t = x_hat_0t.to(device) + sigma_t_minus_1.view(-1, 1, 1, 1) * (term1 + term2)
    #         del score, x_0t, eps, term1, term2
    #         torch.cuda.empty_cache()
    #         if step%100==0:
    #         # Optional visualization
    #             plt.imshow(x_t[0, 0].detach().cpu().numpy(), cmap="gray")
    #             plt.title(f"x_t, step {step}")
    #             plt.axis("off")
    #             plt.show()

    #     with torch.no_grad():
    #         # Final output: CPU INR evaluation
    #         x_out = self.inr(self.coords).reshape(B, 1, H_hr, W_hr)
    #     return x_out
    

    def forward(self, y):
        diffusion_device = torch.device("cuda")
        hqs_device = torch.device("cpu")

        B, _, H_lr, W_lr = y.shape
        H_hr = W_hr = self.image_size

        # Move models to correct devices
        self.sampler.model.to(diffusion_device)
        self.inr.to(hqs_device)
        self.degradation_model.to(hqs_device)

        # Step 1: Initialize x_T from prior
        with torch.no_grad():
            x_t = self.sampler.sde.prior_sampling((B, 1, H_hr, W_hr)).to(diffusion_device)

        for step in tqdm(range(self.num_steps), desc="Diffusion"):
            with torch.no_grad():
                t = torch.tensor([(self.num_steps - step) / self.num_steps], device=diffusion_device)
                t_prev = torch.tensor([(self.num_steps - step - 1) / self.num_steps], device=diffusion_device)
                _, sigma_t = self.sampler.sde.get_sigma(t)
                _, sigma_t_minus_1 = self.sampler.sde.get_sigma(t_prev)

                score = self.sampler.model(x_t, t)
                x_0t = x_t + sigma_t.view(-1, 1, 1, 1) ** 2 * score
                x_0t = x_0t.detach()

            # INR optimization on CPU
            inr_opt = torch.optim.Adam(self.inr.parameters(), lr=1e-3)
            patience = 5
            best_loss = float('inf')
            no_improve_count = 0

            for i in range(500):
                inr_opt.zero_grad()
                coords = self.coords.to(hqs_device)
                x_hat_0t = self.inr(coords).reshape(B, 1, H_hr, W_hr)
                y_hat = self.degradation_model(x_hat_0t)
                loss_df = F.mse_loss(y_hat, y.to(hqs_device))
                loss_pc = F.mse_loss(x_hat_0t, x_0t.to(hqs_device))
                loss = self.lambda_df * loss_df + loss_pc
                print(loss)
                if not torch.isfinite(loss):
                    print(f"[Step {step}] ⚠️ Loss is not finite. Reinitializing INR.")
                    for layer in self.inr.parameters():
                        if layer.requires_grad and layer.dim() > 1:
                            nn.init.kaiming_normal_(layer)
                    break

                loss.backward()
                inr_opt.step()

                # Early stopping
                if loss.item() < best_loss - 1e-6:
                    best_loss = loss.item()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > patience:
                    print(f"[Step {step}] ✅ Early stopping at iter {i} (loss: {loss.item():.6f})")
                    break

            # Get final INR output
            x_hat_0t = self.inr(self.coords.to(hqs_device)).reshape(B, 1, H_hr, W_hr).detach()

            if step % 5 == 0 or step == self.num_steps - 1:
                img = x_hat_0t[0, 0].cpu().numpy()
                plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
                plt.title(f"Step {step} - INR Output")
                plt.axis('off')
                plt.show()

            # Diffusion update
            eps = torch.randn_like(x_hat_0t).to(diffusion_device)
            term1 = torch.sqrt(torch.tensor(self.xi, device=diffusion_device)) * eps
            term2 = torch.sqrt(torch.tensor(1 - self.xi, device=diffusion_device)) * (
                x_t - x_hat_0t.to(diffusion_device)
            )
            x_t = x_hat_0t.to(diffusion_device) + sigma_t_minus_1.view(-1, 1, 1, 1) * (term1 + term2)

            del score, x_0t, eps, term1, term2
            torch.cuda.empty_cache()

        # Final output
        with torch.no_grad():
            x_out = self.inr(self.coords.to(hqs_device)).reshape(B, 1, H_hr, W_hr)
        return x_out
