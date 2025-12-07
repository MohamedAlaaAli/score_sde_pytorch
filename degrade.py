import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationModel(nn.Module):
    def __init__(self, blur_sigma=1.5, downsample_factor=2, noise_std=0.05, downsample_mode="avg"):
        """
        Degradation model for LF-MRI simulation with CPU-only operations.

        Args:
            blur_sigma (float): Standard deviation of the Gaussian blur.
            downsample_factor (int): Downsampling factor.
            noise_std (float): Gaussian noise std.
            downsample_mode (str): "avg" or "bilinear".
        """
        super().__init__()
        self.blur_sigma = blur_sigma
        self.downsample_factor = downsample_factor
        self.noise_std = noise_std
        self.downsample_mode = downsample_mode

    def gaussian_blur(self, x_cpu):
        kernel_size = int(6 * self.blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x_range = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device="cpu")
        x_grid, y_grid = torch.meshgrid(x_range, x_range, indexing='ij')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * self.blur_sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)  # [1,1,k,k]

        B, C, H, W = x_cpu.shape
        x_padded = F.pad(x_cpu, (kernel_size // 2,) * 4, mode='reflect')
        blurred = F.conv2d(x_padded, kernel, padding=0, groups=1)
        return blurred

    def downsample(self, x_cpu):
        if self.downsample_mode == "avg":
            return F.avg_pool2d(x_cpu, kernel_size=self.downsample_factor)
        elif self.downsample_mode == "bilinear":
            return F.interpolate(x_cpu, scale_factor=1 / self.downsample_factor, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported downsample mode: {self.downsample_mode}")

    def add_noise(self, x_cpu):
        return x_cpu + torch.randn_like(x_cpu, device="cpu") * self.noise_std

    def forward(self, x):
        """
        Degrades an HF MRI image into LF MRI using only CPU ops.
        
        Args:
            x (torch.Tensor): Input HF MRI [B, 1, H, W] (can be on GPU or CPU).
        
        Returns:
            torch.Tensor: LF MRI degraded image [B, 1, H/2, W/2] on CPU.
        """
        x_cpu = x.detach().cpu()  # move input to CPU
        x_cpu = self.gaussian_blur(x_cpu)
        x_cpu = self.downsample(x_cpu)
        x_cpu = self.add_noise(x_cpu)
        return x_cpu
