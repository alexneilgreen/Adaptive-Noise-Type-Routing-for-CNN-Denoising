import torch
import torch.nn as nn
import torch.nn.functional as F


class CorruptionDetection(nn.Module):
    """Module to detect corrupted pixel locations."""
    
    def __init__(self, in_channels):
        super(CorruptionDetection, self).__init__()
        self.detect = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.detect(x)


class ImpulseDenoiser(nn.Module):
    """
    Specialized denoising CNN for Impulse noise.
    Uses corruption detection and larger receptive fields for pixel replacement.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the Impulse noise denoising network.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            out_channels (int): Number of output channels (1 for grayscale, 3 for RGB)
        """
        super(ImpulseDenoiser, self).__init__()
        
        # Corruption detection branch
        self.corruption_detector = CorruptionDetection(in_channels)
        
        # Initial feature extraction with larger kernel
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Block 1 - replaced non-local with dilated convs
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),  # Dilated for larger context
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 2 - replaced non-local with dilated convs
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),  # Dilated for larger context
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck with larger receptive field
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 1
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 2
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final refinement with larger kernel
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
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
        # Detect corrupted pixels
        corruption_mask = self.corruption_detector(x)
        
        # Initial processing
        init_out = self.initial(x)
        
        # Encoder
        enc1_out = self.enc1(init_out)
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
        
        # Final refinement
        final_out = self.final(dec2_out)
        
        # Predict clean image
        predicted = self.out(final_out)
        
        # Blend based on corruption mask
        # Use prediction where corrupted, keep original where clean
        output = corruption_mask * predicted + (1 - corruption_mask) * x
        
        return torch.clamp(output, 0.0, 1.0)


def get_model(in_channels=3, out_channels=3):
    """
    Factory function to get the Impulse noise denoising model.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        nn.Module: Impulse noise denoising model
    """
    return ImpulseDenoiser(in_channels, out_channels)