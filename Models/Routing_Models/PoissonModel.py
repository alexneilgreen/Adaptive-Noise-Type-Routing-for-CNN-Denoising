import torch
import torch.nn as nn

class VarianceStabilization(nn.Module):
    """Variance stabilization layer for Poisson noise (Anscombe transform)."""
    
    def __init__(self):
        super(VarianceStabilization, self).__init__()
        self.epsilon = 1e-6
    
    def forward(self, x):
        # Anscombe transform: 2 * sqrt(x + 3/8)
        return 2.0 * torch.sqrt(x + 3.0/8.0 + self.epsilon)
    
    def inverse(self, y):
        # Inverse Anscombe: (y/2)^2 - 3/8
        return torch.clamp((y / 2.0) ** 2 - 3.0/8.0, 0.0, 1.0)


class PoissonDenoiser(nn.Module):
    """
    Specialized denoising CNN for Poisson noise.
    Uses variance stabilization and instance normalization for signal-dependent noise.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the Poisson noise denoising network.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            out_channels (int): Number of output channels (1 for grayscale, 3 for RGB)
        """
        super(PoissonDenoiser, self).__init__()
        
        # Variance stabilization
        self.variance_stab = VarianceStabilization()
        
        # Encoder Block 1 with Instance Norm (better for signal-dependent noise)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck with parallel branches for different intensity levels
        self.bottleneck_main = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Additional branch for high-intensity regions
        self.bottleneck_high = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Combine branches
        self.bottleneck_combine = nn.Conv2d(384, 256, kernel_size=1)
        
        # Decoder Block 1
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 2
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input noisy image [B, C, H, W]
        
        Returns:
            torch.Tensor: Denoised image [B, C, H, W]
        """
        # Apply variance stabilization
        x_stabilized = self.variance_stab(x)
        
        # Encoder with skip connections
        enc1_out = self.enc1(x_stabilized)
        pool1_out = self.pool1(enc1_out)
        
        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)
        
        # Dual-branch bottleneck
        main_branch = self.bottleneck_main(pool2_out)
        high_branch = self.bottleneck_high(pool2_out)
        combined = torch.cat([main_branch, high_branch], dim=1)
        bottleneck_out = self.bottleneck_combine(combined)
        
        # Decoder with skip connections
        up1 = self.upconv1(bottleneck_out)
        dec1_in = torch.cat([up1, enc2_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        
        up2 = self.upconv2(dec1_out)
        dec2_in = torch.cat([up2, enc1_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        
        # Output in stabilized space
        output_stabilized = self.out(dec2_out)
        
        # Inverse variance stabilization
        denoised = self.variance_stab.inverse(output_stabilized)
        
        return torch.clamp(denoised, 0.0, 1.0)


def get_model(in_channels=3, out_channels=3):
    """
    Factory function to get the Poisson noise denoising model.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        nn.Module: Poisson noise denoising model
    """
    return PoissonDenoiser(in_channels, out_channels)