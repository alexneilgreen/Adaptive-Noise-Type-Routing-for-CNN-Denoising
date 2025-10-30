import torch
import torch.nn as nn

class DeblockingLayer(nn.Module):
    """Specialized layer to remove 8x8 block artifacts."""
    
    def __init__(self, channels):
        super(DeblockingLayer, self).__init__()
        # Larger kernel to span block boundaries
        self.deblock = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=9, padding=4),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.deblock(x)


class EdgePreservingBlock(nn.Module):
    """Block that preserves edges while removing artifacts."""
    
    def __init__(self, in_channels, out_channels):
        super(EdgePreservingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection if dimensions match
        self.residual = nn.Identity() if in_channels == out_channels else \
                       nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class JPEGDenoiser(nn.Module):
    """
    Specialized denoising CNN for JPEG compression artifacts.
    Focuses on removing blocking artifacts and ringing while preserving edges.
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        """
        Initialize the JPEG artifact removal network.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            out_channels (int): Number of output channels (1 for grayscale, 3 for RGB)
        """
        super(JPEGDenoiser, self).__init__()
        
        # Initial deblocking
        self.initial_deblock = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder Block 1 with edge preservation
        self.enc1 = nn.Sequential(
            EdgePreservingBlock(64, 64),
            EdgePreservingBlock(64, 64)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 2
        self.enc2 = nn.Sequential(
            EdgePreservingBlock(64, 128),
            EdgePreservingBlock(128, 128)
        )
        self.deblock2 = DeblockingLayer(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck with multiple deblocking layers
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DeblockingLayer(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DeblockingLayer(256)
        )
        
        # Decoder Block 1
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            EdgePreservingBlock(256, 128),
            EdgePreservingBlock(128, 128)
        )
        
        # Decoder Block 2
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            EdgePreservingBlock(128, 64),
            EdgePreservingBlock(64, 64)
        )
        
        # Final deblocking and output
        self.final_deblock = DeblockingLayer(64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input JPEG compressed image [B, C, H, W]
        
        Returns:
            torch.Tensor: Deblocked image [B, C, H, W]
        """
        # Initial deblocking
        init_out = self.initial_deblock(x)
        
        # Encoder with skip connections
        enc1_out = self.enc1(init_out)
        pool1_out = self.pool1(enc1_out)
        
        enc2_out = self.enc2(pool1_out)
        enc2_out = self.deblock2(enc2_out)
        pool2_out = self.pool2(enc2_out)
        
        # Bottleneck with deblocking
        bottleneck_out = self.bottleneck(pool2_out)
        
        # Decoder with skip connections
        up1 = self.upconv1(bottleneck_out)
        dec1_in = torch.cat([up1, enc2_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        
        up2 = self.upconv2(dec1_out)
        dec2_in = torch.cat([up2, enc1_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        
        # Final deblocking and output
        final_out = self.final_deblock(dec2_out)
        output = self.out(final_out)
        
        # Direct prediction (not residual for JPEG artifacts)
        return torch.clamp(output, 0.0, 1.0)


def get_model(in_channels=3, out_channels=3):
    """
    Factory function to get the JPEG artifact removal model.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        nn.Module: JPEG artifact removal model
    """
    return JPEGDenoiser(in_channels, out_channels)