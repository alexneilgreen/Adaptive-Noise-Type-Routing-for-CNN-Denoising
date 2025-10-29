import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False
    print("Warning: pytorch_msssim not available. Install with: pip install pytorch-msssim")


class CompositeLoss(nn.Module):
    """
    Composite loss function combining MSE, MAE, SSIM, and Gradient losses.
    Weights: MSE (35%) + MAE (35%) + SSIM (20%) + Gradient (10%)
    """
    
    def __init__(self, mse_weight=0.35, mae_weight=0.35, ssim_weight=0.20, 
                 gradient_weight=0.10, data_range=1.0):
        """
        Initialize composite loss.
        
        Args:
            mse_weight: Weight for MSE loss (default: 0.35)
            mae_weight: Weight for MAE loss (default: 0.35)
            ssim_weight: Weight for SSIM loss (default: 0.20)
            gradient_weight: Weight for gradient loss (default: 0.10)
            data_range: Range of image values (default: 1.0 for [0,1])
        """
        super(CompositeLoss, self).__init__()
        
        # Validate weights sum to 1.0
        total_weight = mse_weight + mae_weight + ssim_weight + gradient_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.data_range = data_range
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Check if MSSSIM is available
        if not MSSSIM_AVAILABLE:
            print("Warning: Using fallback SSIM implementation. Install pytorch-msssim for better results.")
    
    def ssim_loss(self, pred, target):
        """
        Calculate SSIM loss.
        Returns 1 - SSIM to convert to minimization objective.
        """
        if MSSSIM_AVAILABLE:
            # Use pytorch-msssim implementation
            ssim_val = ssim(pred, target, data_range=self.data_range, size_average=True)
        else:
            # Fallback to simplified SSIM
            ssim_val = self._simplified_ssim(pred, target)
        
        # Convert to loss (1 - SSIM)
        return 1.0 - ssim_val
    
    def _simplified_ssim(self, pred, target):
        """
        Simplified SSIM calculation (fallback if pytorch-msssim not available).
        """
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_val
    
    def gradient_loss(self, pred, target):
        """
        Calculate gradient loss using Sobel filters for edge preservation.
        Computes L1 loss on image gradients.
        """
        # Sobel filters for x and y gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=pred.device)
        
        # Reshape for conv2d: (1, 1, 3, 3)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Get number of channels
        channels = pred.shape[1]
        
        # Expand filters for all channels
        sobel_x = sobel_x.repeat(channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(channels, 1, 1, 1)
        
        # Calculate gradients for prediction
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=channels)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=channels)
        
        # Calculate gradients for target
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
        
        # L1 loss on gradients
        loss_x = torch.mean(torch.abs(pred_grad_x - target_grad_x))
        loss_y = torch.mean(torch.abs(pred_grad_y - target_grad_y))
        
        return (loss_x + loss_y) / 2.0
    
    def forward(self, pred, target, return_components=False):
        """
        Calculate composite loss.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target clean images [B, C, H, W]
            return_components: If True, return dict with individual components
        
        Returns:
            loss: Total weighted loss
            components (optional): Dict with individual loss components
        """
        # Calculate individual losses
        mse = self.mse_loss(pred, target)
        mae = self.mae_loss(pred, target)
        ssim_l = self.ssim_loss(pred, target)
        grad = self.gradient_loss(pred, target)
        
        # Weighted combination
        total_loss = (self.mse_weight * mse + 
                     self.mae_weight * mae + 
                     self.ssim_weight * ssim_l + 
                     self.gradient_weight * grad)
        
        if return_components:
            components = {
                'mse': mse.item(),
                'mae': mae.item(),
                'ssim_loss': ssim_l.item(),
                'gradient': grad.item(),
                'total': total_loss.item()
            }
            return total_loss, components
        
        return total_loss