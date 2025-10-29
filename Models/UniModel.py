import torch
import torch.nn as nn

class ComprehensiveDenoiser(nn.Module):
    """
    Comprehensive denoising CNN that handles all noise types with a single unified model.
    Uses an encoder-decoder architecture with skip connections.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the comprehensive denoising network.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            out_channels (int): Number of output channels (1 for grayscale, 3 for RGB)
        """
        super(ComprehensiveDenoiser, self).__init__()
        
        # Encoder Block 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
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
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 1
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 because of skip connection
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder Block 2
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 because of skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
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
        
        # Output
        output = self.out(dec2_out)
        
        # Use residual learning: predict noise and subtract from input
        denoised = x - output
        
        return torch.clamp(denoised, 0.0, 1.0)


class AdaptiveRoutingDenoiser(nn.Module):
    """
    Adaptive routing denoiser with noise-type classification.
    This is a placeholder for future implementation.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the adaptive routing denoising network.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(AdaptiveRoutingDenoiser, self).__init__()
        raise NotImplementedError("Adaptive routing model will be implemented in Phase 2")
    
    def forward(self, x):
        raise NotImplementedError("Adaptive routing model will be implemented in Phase 2")


def get_model(model_type='comp', in_channels=3, out_channels=3):
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type (str): Type of model ('comp' or 'routing')
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        nn.Module: The requested model
    """
    if model_type == 'comp':
        return ComprehensiveDenoiser(in_channels, out_channels)
    elif model_type == 'routing':
        return AdaptiveRoutingDenoiser(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'comp' or 'routing'.")