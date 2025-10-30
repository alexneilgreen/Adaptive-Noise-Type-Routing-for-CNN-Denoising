import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    """Non-local self-attention block for context aggregation."""
    
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.inter_channels = in_channels // 2
        
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Compute attention
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # [B, HW, C/2]
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C/2, HW]
        
        attention = torch.matmul(theta_x, phi_x)  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [B, C/2, HW]
        g_x = g_x.permute(0, 2, 1)  # [B, HW, C/2]
        
        y = torch.matmul(attention, g_x)  # [B, HW, C/2]
        y = y.permute(0, 2, 1).contiguous()  # [B, C/2, HW]
        y = y.view(batch_size, self.inter_channels, H, W)
        
        out = self.out_conv(y)
        return out + x  # Residual connection


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
    Uses corruption detection and non-local attention for pixel replacement.
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
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Block 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.nonlocal1 = NonLocalBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.nonlocal2 = NonLocalBlock(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck with context aggregation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
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
        
        # Final context aggregation
        self.final_nonlocal = NonLocalBlock(64)
        
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
        
        # Encoder with non-local attention
        enc1_out = self.enc1(init_out)
        enc1_out = self.nonlocal1(enc1_out)
        pool1_out = self.pool1(enc1_out)
        
        enc2_out = self.enc2(pool1_out)
        enc2_out = self.nonlocal2(enc2_out)
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
        
        # Final non-local context
        final_out = self.final_nonlocal(dec2_out)
        
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