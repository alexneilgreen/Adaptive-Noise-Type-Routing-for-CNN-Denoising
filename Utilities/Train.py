import os
import csv
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging

from Utilities.Composite_Loss import CompositeLoss

# Setup logging
logger = logging.getLogger(__name__)


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: First image tensor
        img2: Second image tensor
        max_val: Maximum possible pixel value (default: 1.0)
    
    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Simplified implementation for batch processing.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        window_size: Size of the sliding window (default: 11)
        max_val: Maximum possible pixel value (default: 1.0)
    
    Returns:
        float: SSIM value
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()


class Trainer:
    """Trainer class for denoising models."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
             learning_rate=0.001, results_dir='./results', 
             early_stopping_patience=10, device='cuda'):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            learning_rate: Learning rate for optimizer
            results_dir: Directory to save results
            early_stopping_patience: Epochs to wait before early stopping
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.results_dir = results_dir
        self.early_stopping_patience = early_stopping_patience
        
        # Set device
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)
        
        # Import and initialize composite loss
        self.criterion = CompositeLoss(
            mse_weight=0.35,
            mae_weight=0.35,
            ssim_weight=0.20,
            gradient_weight=0.10
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.best_model_state = None
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_psnrs = []
        self.val_psnrs = []
        self.train_ssims = []
        self.val_ssims = []
        
        # Track loss components
        self.loss_components = {'train': [], 'val': []}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info("Using Composite Loss: MSE(35%) + MAE(35%) + SSIM(20%) + Gradient(10%)")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        num_batches = 0
        
        # Track loss components
        epoch_components = {'mse': 0.0, 'mae': 0.0, 'ssim_loss': 0.0, 'gradient': 0.0}
        
        for noisy, clean, _ in self.train_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(noisy)
            
            # Calculate composite loss with components
            loss, components = self.criterion(output, clean, return_components=True)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            for key in epoch_components:
                epoch_components[key] += components[key]
            
            with torch.no_grad():
                batch_psnr = calculate_psnr(output, clean)
                batch_ssim = calculate_ssim(output, clean)
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim
            
            num_batches += 1
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        avg_ssim = epoch_ssim / num_batches
        
        # Average loss components
        avg_components = {k: v / num_batches for k, v in epoch_components.items()}
        self.loss_components['train'].append(avg_components)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        num_batches = 0
        
        # Track loss components
        epoch_components = {'mse': 0.0, 'mae': 0.0, 'ssim_loss': 0.0, 'gradient': 0.0}
        
        with torch.no_grad():
            for noisy, clean, _ in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Forward pass
                output = self.model(noisy)
                
                # Calculate composite loss with components
                loss, components = self.criterion(output, clean, return_components=True)
                
                # Accumulate metrics
                epoch_loss += loss.item()
                for key in epoch_components:
                    epoch_components[key] += components[key]
                
                batch_psnr = calculate_psnr(output, clean)
                batch_ssim = calculate_ssim(output, clean)
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim
                
                num_batches += 1
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        avg_ssim = epoch_ssim / num_batches
        
        # Average loss components
        avg_components = {k: v / num_batches for k, v in epoch_components.items()}
        self.loss_components['val'].append(avg_components)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def test(self):
        """Test the model on test set."""
        self.model.eval()
        test_loss = 0.0
        test_psnr = 0.0
        test_ssim = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for noisy, clean, _ in self.test_loader:  # Unpack 3 values
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # Forward pass
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                
                # Calculate metrics
                test_loss += loss.item()
                batch_psnr = calculate_psnr(output, clean)
                batch_ssim = calculate_ssim(output, clean)
                test_psnr += batch_psnr
                test_ssim += batch_ssim
                
                num_batches += 1
        
        # Average metrics
        avg_loss = test_loss / num_batches
        avg_psnr = test_psnr / num_batches
        avg_ssim = test_ssim / num_batches
        
        return avg_loss, avg_psnr, avg_ssim
    
    def save_epoch_data(self):
        """Save epoch-level training data to CSV."""
        epoch_data = {
            'Epoch': list(range(1, len(self.train_losses) + 1)),
            'Train_Loss': self.train_losses,
            'Val_Loss': self.val_losses,
            'Train_PSNR': self.train_psnrs,
            'Val_PSNR': self.val_psnrs,
            'Train_SSIM': self.train_ssims,
            'Val_SSIM': self.val_ssims
        }
        
        csv_path = os.path.join(self.results_dir, 'epoch_data.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_data.keys())
            writer.writerows(zip(*epoch_data.values()))
        
        logger.info(f"Epoch data saved to {csv_path}")
        
        # Save loss components separately
        components_path = os.path.join(self.results_dir, 'loss_components.csv')
        with open(components_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Split', 'MSE', 'MAE', 'SSIM_Loss', 'Gradient', 'Total'])
            
            for epoch in range(len(self.loss_components['train'])):
                train_comp = self.loss_components['train'][epoch]
                writer.writerow([epoch + 1, 'train', 
                            train_comp['mse'], train_comp['mae'], 
                            train_comp['ssim_loss'], train_comp['gradient'],
                            train_comp['mse'] + train_comp['mae'] + 
                            train_comp['ssim_loss'] + train_comp['gradient']])
                
                if epoch < len(self.loss_components['val']):
                    val_comp = self.loss_components['val'][epoch]
                    writer.writerow([epoch + 1, 'val',
                                val_comp['mse'], val_comp['mae'],
                                val_comp['ssim_loss'], val_comp['gradient'],
                                val_comp['mse'] + val_comp['mae'] + 
                                val_comp['ssim_loss'] + val_comp['gradient']])
        
        logger.info(f"Loss components saved to {components_path}")
    
    def visualize_results(self, num_samples=5):
        """Generate visual comparisons of denoising results."""
        save_dir = os.path.join(self.results_dir, 'visual_comparisons')
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (noisy, clean, _) in enumerate(self.test_loader):
                if batch_idx >= 1:  # Only first batch
                    break
                
                for i in range(min(num_samples, noisy.size(0))):
                    noisy_img = noisy[i].to(self.device)
                    clean_img = clean[i].to(self.device)
                    
                    # Denoise
                    denoised = self.model(noisy_img.unsqueeze(0)).squeeze(0)
                    
                    # Move to CPU and convert to numpy
                    clean_np = clean_img.cpu().numpy()
                    noisy_np = noisy_img.cpu().numpy()
                    denoised_np = denoised.cpu().numpy()
                    
                    # Create figure
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    if clean_np.shape[0] == 1:  # Grayscale
                        clean_np = clean_np.squeeze(0)
                        noisy_np = noisy_np.squeeze(0)
                        denoised_np = denoised_np.squeeze(0)
                        cmap = 'gray'
                    else:  # RGB
                        clean_np = np.transpose(clean_np, (1, 2, 0))
                        noisy_np = np.transpose(noisy_np, (1, 2, 0))
                        denoised_np = np.transpose(denoised_np, (1, 2, 0))
                        cmap = None
                    
                    axes[0].imshow(clean_np, cmap=cmap)
                    axes[0].set_title('Clean')
                    axes[0].axis('off')
                    
                    axes[1].imshow(noisy_np, cmap=cmap)
                    axes[1].set_title('Noisy')
                    axes[1].axis('off')
                    
                    axes[2].imshow(denoised_np, cmap=cmap)
                    axes[2].set_title('Denoised')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, f'comparison_{i}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
        
        logger.info(f"Visual comparisons saved to {save_dir}")
    
    def save_result_images(self, num_samples=5):
        """Save individual images for figure creation."""
        save_dir = os.path.join(self.results_dir, 'images')
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (noisy, clean, _) in enumerate(self.test_loader):
                if batch_idx >= 1:
                    break
                
                for i in range(min(num_samples, noisy.size(0))):
                    noisy_img = noisy[i].to(self.device)
                    clean_img = clean[i].to(self.device)
                    
                    # Denoise
                    denoised = self.model(noisy_img.unsqueeze(0)).squeeze(0)
                    
                    # Convert to numpy
                    clean_np = clean_img.cpu().numpy()
                    denoised_np = denoised.cpu().numpy()
                    
                    # Save images
                    if clean_np.shape[0] == 1:  # Grayscale
                        clean_np = (clean_np.squeeze(0) * 255).astype(np.uint8)
                        denoised_np = (denoised_np.squeeze(0) * 255).astype(np.uint8)
                        Image.fromarray(clean_np).save(os.path.join(save_dir, f'{i+1}_original.png'))
                        Image.fromarray(denoised_np).save(os.path.join(save_dir, f'{i+1}_generated.png'))
                    else:  # RGB
                        clean_np = (np.transpose(clean_np, (1, 2, 0)) * 255).astype(np.uint8)
                        denoised_np = (np.transpose(denoised_np, (1, 2, 0)) * 255).astype(np.uint8)
                        Image.fromarray(clean_np).save(os.path.join(save_dir, f'{i+1}_original.png'))
                        Image.fromarray(denoised_np).save(os.path.join(save_dir, f'{i+1}_generated.png'))
        
        logger.info(f"Result images saved to {save_dir}")
    
    def train(self, epochs=50):
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            dict: Training results
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Results directory: {self.results_dir}")
        
        # Open log file
        log_path = os.path.join(self.results_dir, 'training_log.txt')
        log_file = open(log_path, 'w')
        
        def log_message(msg):
            """Write message to both console and file."""
            logger.info(msg)
            log_file.write(msg + '\n')
            log_file.flush()
        
        start_time = time.time()
        
        log_message(f"Starting training on device: {self.device}")
        log_message(f"Total epochs: {epochs}")
        log_message("="*60)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_psnr, train_ssim = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_psnrs.append(train_psnr)
            self.train_ssims.append(train_ssim)
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_psnrs.append(val_psnr)
            self.val_ssims.append(val_ssim)
            
            # Log progress
            log_message(f"Epoch {epoch}/{epochs}")
            log_message(f"  Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}dB, SSIM: {train_ssim:.4f}")
            log_message(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}dB, SSIM: {val_ssim:.4f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # Save best model
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                log_message(f"  New best validation loss: {val_loss:.4f}")
            else:
                self.early_stopping_counter += 1
                log_message(f"  Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    log_message(f"\nEarly stopping triggered at epoch {epoch}")
                    # Restore best model
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        log_message("Best model restored")
                    break
            
            log_message("-"*60)
        
        # Test phase
        log_message("\nRunning final test evaluation...")
        test_loss, test_psnr, test_ssim = self.test()
        
        total_time = time.time() - start_time
        
        log_message("\n" + "="*60)
        log_message("FINAL TEST RESULTS")
        log_message("="*60)
        log_message(f"Test Loss: {test_loss:.4f}")
        log_message(f"Test PSNR: {test_psnr:.2f} dB")
        log_message(f"Test SSIM: {test_ssim:.4f}")
        log_message(f"Total Computation Time: {total_time:.2f} seconds")
        log_message("="*60)
        
        # Close log file
        log_file.close()
        
        # Save visualizations
        logger.info("Generating visualizations...")
        self.save_result_images()
        self.visualize_results()
        
        # Save epoch data
        self.save_epoch_data()
        
        # Prepare results dictionary
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_psnrs': self.train_psnrs,
            'val_psnrs': self.val_psnrs,
            'train_ssims': self.train_ssims,
            'val_ssims': self.val_ssims,
            'test_loss': test_loss,
            'test_psnr': test_psnr,
            'test_ssim': test_ssim,
            'computation_time': total_time,
            'epochs_trained': len(self.train_losses)
        }
        
        # Save results to JSON
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Training complete. Results saved to {self.results_dir}")
        
        return results