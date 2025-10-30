import torch
import torch.nn as nn

class GaussianDenoiser(nn.Module):
    """
    Specialized denoising CNN for Gaussian noise.
    Uses encoder-decoder with dilated convolutions for multi-scale context.
    Optimized for smooth, additive noise removal.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the Gaussian noise denoising network.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            out_channels (int): Number of output channels (1 for grayscale, 3 for RGB)
        """
        super(GaussianDenoiser, self).__init__()
        
        # Encoder Block 1 - Smooth feature extraction
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck with dilated convolutions for multi-scale context
        self.bottleneck = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=2, dilation=2),  # Dilated
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=4, dilation=4),  # Dilated
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 1
        self.upconv1 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),  # 192 from skip connection
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 2
        self.upconv2 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=3, padding=1),  # 96 from skip connection
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.out = nn.Conv2d(48, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input noisy image [B, C, H, W]
        
        Returns:
            torch.Tensor: Denoised image [B, C, H, W]
        """
        # Encoder with skip connections
        enc1_out = self.enc1(x)
        pool1_out = self.pool1(enc1_out)
        
        enc2_out = self.enc2(pool1_out)
        pool2_out = self.pool2(enc2_out)
        
        # Bottleneck with multi-scale context
        bottleneck_out = self.bottleneck(pool2_out)
        
        # Decoder with skip connections
        up1 = self.upconv1(bottleneck_out)
        dec1_in = torch.cat([up1, enc2_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        
        up2 = self.upconv2(dec1_out)
        dec2_in = torch.cat([up2, enc1_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        
        # Output - predict noise and subtract
        noise = self.out(dec2_out)
        denoised = x - noise
        
        return torch.clamp(denoised, 0.0, 1.0)


def get_model(in_channels=3, out_channels=3):
    """
    Factory function to get the Gaussian denoising model.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        nn.Module: Gaussian denoising model
    """
    return GaussianDenoiser(in_channels, out_channels)