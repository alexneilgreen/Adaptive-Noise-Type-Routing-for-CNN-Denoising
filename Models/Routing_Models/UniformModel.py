import torch
import torch.nn as nn

class UniformDenoiser(nn.Module):
    """
    Specialized denoising CNN for Uniform noise.
    Uses wider filters and dropout for robustness against uniformly distributed noise.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the Uniform noise denoising network.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            out_channels (int): Number of output channels (1 for grayscale, 3 for RGB)
        """
        super(UniformDenoiser, self).__init__()
        
        # Encoder Block 1 - Wider channels for better averaging
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(80, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(80, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck with dropout for regularization
        self.bottleneck = nn.Sequential(
            nn.Conv2d(160, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 1
        self.upconv1 = nn.ConvTranspose2d(256, 160, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(320, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 2
        self.upconv2 = nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(160, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(80, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.out = nn.Conv2d(80, out_channels, kernel_size=1)
    
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
        
        # Bottleneck
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
    Factory function to get the Uniform noise denoising model.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        nn.Module: Uniform noise denoising model
    """
    return UniformDenoiser(in_channels, out_channels)