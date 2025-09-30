import torch
import numpy as np
from PIL import Image
import io
import os
import argparse
from typing import Union, Dict, List, Tuple
import matplotlib.pyplot as plt

class NoiseGenerator:
    """
    Generator for various types of image noise.
    Supports on-the-fly noise generation and example generation.
    """
    
    def __init__(self):
        """Initialize noise generator with default parameters."""
        self.noise_params = {
            'gaussian': {'sigma': 25.0},
            'salt_pepper': {'noise_ratio': 0.1},
            'uniform': {'low': -25.0, 'high': 25.0},
            'poisson': {'lam': 30.0},
            'jpeg': {'quality': 30},
            'impulse': {'noise_ratio': 0.1}  # Random-valued impulse noise
        }
    
    def add_noise(self, image: torch.Tensor, noise_type: str) -> torch.Tensor:
        """
        Add specified noise type to an image.
        
        Args:
            image: Input image tensor (C, H, W) with values in [0, 1]
            noise_type: Type of noise to add
            
        Returns:
            Noisy image tensor with same shape and range [0, 1]
        """
        if noise_type == 'gaussian':
            return self._add_gaussian_noise(image)
        elif noise_type == 'salt_pepper':
            return self._add_salt_pepper_noise(image)
        elif noise_type == 'uniform':
            return self._add_uniform_noise(image)
        elif noise_type == 'poisson':
            return self._add_poisson_noise(image)
        elif noise_type == 'jpeg':
            return self._add_jpeg_artifacts(image)
        elif noise_type == 'impulse':
            return self._add_impulse_noise(image)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def _add_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to image."""
        sigma = self.noise_params['gaussian']['sigma'] / 255.0  # Scale to [0,1]
        noise = torch.randn_like(image) * sigma
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)
    
    def _add_salt_pepper_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add salt and pepper noise to image."""
        noise_ratio = self.noise_params['salt_pepper']['noise_ratio']
        noisy_image = image.clone()
        
        # Create random mask for noise locations
        mask = torch.rand_like(image)
        
        # Salt noise (white pixels)
        salt_mask = mask < noise_ratio / 2
        noisy_image[salt_mask] = 1.0
        
        # Pepper noise (black pixels)
        pepper_mask = (mask >= noise_ratio / 2) & (mask < noise_ratio)
        noisy_image[pepper_mask] = 0.0
        
        return noisy_image
    
    def _add_uniform_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add uniform noise to image."""
        low = self.noise_params['uniform']['low'] / 255.0  # Scale to [0,1]
        high = self.noise_params['uniform']['high'] / 255.0
        
        noise = torch.rand_like(image) * (high - low) + low
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)
    
    def _add_poisson_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Poisson noise to image."""
        # Scale image to appropriate range for Poisson
        lam = self.noise_params['poisson']['lam']
        
        # Convert to numpy for Poisson sampling
        img_np = image.numpy()
        
        # Scale image and apply Poisson noise
        scaled_img = img_np * lam
        noisy_scaled = np.random.poisson(scaled_img)
        noisy_image = noisy_scaled / lam
        
        # Convert back to tensor and clamp
        noisy_tensor = torch.from_numpy(noisy_image).float()
        return torch.clamp(noisy_tensor, 0.0, 1.0)
    
    def _add_jpeg_artifacts(self, image: torch.Tensor) -> torch.Tensor:
        """Add JPEG compression artifacts to image."""
        quality = self.noise_params['jpeg']['quality']
        
        # Convert tensor to PIL Image
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
            # Convert from (C, H, W) to (H, W, C)
            img_np = image.permute(1, 2, 0).numpy()
        else:
            img_np = image.numpy()
        
        # Handle grayscale vs RGB
        if img_np.shape[-1] == 1:  # Grayscale
            img_np = img_np.squeeze(-1)
            pil_image = Image.fromarray((img_np * 255).astype(np.uint8), mode='L')
        else:  # RGB
            pil_image = Image.fromarray((img_np * 255).astype(np.uint8), mode='RGB')
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to tensor
        compressed_np = np.array(compressed_image).astype(np.float32) / 255.0
        
        # Restore original tensor format
        if len(compressed_np.shape) == 2:  # Grayscale
            compressed_np = compressed_np[..., None]
        
        # Convert from (H, W, C) back to (C, H, W)
        if compressed_np.shape[-1] in [1, 3]:
            compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1)
        else:
            compressed_tensor = torch.from_numpy(compressed_np)
        
        return compressed_tensor.float()
    
    def _add_impulse_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add random-valued impulse noise to image."""
        noise_ratio = self.noise_params['impulse']['noise_ratio']
        noisy_image = image.clone()
        
        # Create random mask for noise locations
        mask = torch.rand_like(image) < noise_ratio
        
        # Replace selected pixels with random values
        random_values = torch.rand_like(image)
        noisy_image[mask] = random_values[mask]
        
        return noisy_image
    
    def update_noise_params(self, noise_type: str, **kwargs):
        """Update noise parameters for a specific noise type."""
        if noise_type in self.noise_params:
            self.noise_params[noise_type].update(kwargs)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def get_noise_params(self, noise_type: str = None) -> Dict:
        """Get current noise parameters."""
        if noise_type:
            return self.noise_params.get(noise_type, {})
        return self.noise_params


def generate_examples(dataset_name: str = 'cifar10', 
                     num_examples: int = 10,
                     output_dir: str = './dataset/examples') -> None:
    """
    Generate example images with different noise types for visualization.
    
    Args:
        dataset_name: Name of dataset to use
        num_examples: Number of example images to generate
        output_dir: Directory to save examples
    """
    from data_loader import get_dataloader
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize noise generator
    noise_gen = NoiseGenerator()
    noise_types = ['gaussian', 'salt_pepper', 'uniform', 'poisson', 'jpeg', 'impulse']
    
    # Get clean data loader (no noise)
    clean_loader = get_dataloader(
        dataset_name=dataset_name,
        split='train',
        batch_size=num_examples,
        shuffle=False
    )
    
    # Get first batch
    clean_images, labels = next(iter(clean_loader))
    
    print(f"\nGenerating {num_examples} example images for {dataset_name}...")
    
    for i in range(num_examples):
        clean_img = clean_images[i]
        label = labels[i].item()
        
        # Create subdirectory for this example
        example_dir = os.path.join(output_dir, f"{dataset_name}_example_{i+1}")
        os.makedirs(example_dir, exist_ok=True)
        
        # Save clean image
        save_image_tensor(clean_img, 
                         os.path.join(example_dir, f"{i+1}_original.png"))
        
        # Generate and save noisy versions
        for noise_type in noise_types:
            try:
                noisy_img = noise_gen.add_noise(clean_img, noise_type)
                save_image_tensor(noisy_img, 
                                 os.path.join(example_dir, f"{i+1}_{noise_type}.png"))
            except Exception as e:
                print(f"\tWarning: Failed to generate {noise_type} noise for image {i+1}: {e}")
        
        # Generate comparison grid for this image
        create_comparison_grid(clean_img, noise_gen, 
                              os.path.join(example_dir, f"{i+1}_comparison_grid.png"))
    
    print(f"\tExamples saved to: {output_dir}")


def save_image_tensor(img_tensor: torch.Tensor, filepath: str) -> None:
    """Save a tensor image to file."""
    # Convert from (C, H, W) to (H, W, C) if needed
    if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3]:
        img_np = img_tensor.permute(1, 2, 0).numpy()
    else:
        img_np = img_tensor.numpy()
    
    # Handle grayscale
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
        cmap = 'gray'
    else:
        cmap = None
    
    # Save using matplotlib
    plt.figure(figsize=(4, 4))
    plt.imshow(img_np, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=150, pad_inches=0)
    plt.close()


def create_comparison_grid(clean_image: torch.Tensor, 
                          noise_gen: NoiseGenerator,
                          output_path: str) -> None:
    """Create a comparison grid showing clean image and all noise types."""
    noise_types = ['gaussian', 'salt_pepper', 'uniform', 'poisson', 'jpeg', 'impulse']
    
    fig = plt.figure(figsize=(16, 8))
    
    # Create clean image subplot that spans both rows in the first column
    ax_clean = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
    
    img_np = clean_image.permute(1, 2, 0).numpy()
    if img_np.shape[-1] == 1:
        img_np = img_np.squeeze(-1)
        cmap = 'gray'
    else:
        cmap = None
    
    ax_clean.imshow(img_np, cmap=cmap)
    ax_clean.set_title('Original (Clean)', fontsize=12)
    ax_clean.axis('off')
    
    # Create noise subplot positions (right 3 columns, 2 rows = 6 positions)
    noise_positions = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    
    for i, noise_type in enumerate(noise_types):
        try:
            noisy_img = noise_gen.add_noise(clean_image, noise_type)
            noisy_np = noisy_img.permute(1, 2, 0).numpy()
            if noisy_np.shape[-1] == 1:
                noisy_np = noisy_np.squeeze(-1)
            
            ax_noise = plt.subplot2grid((2, 4), noise_positions[i])
            ax_noise.imshow(noisy_np, cmap=cmap)
            ax_noise.set_title(f'{noise_type.replace("_", " ").title()}', fontsize=12)
            ax_noise.axis('off')
        except Exception as e:
            ax_noise = plt.subplot2grid((2, 4), noise_positions[i])
            ax_noise.text(0.5, 0.5, f'Error\n{noise_type}', 
                        ha='center', va='center', transform=ax_noise.transAxes)
            ax_noise.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Noise Generator')
    parser.add_argument('--generate_examples', action='store_true',
                       help='Generate example images with different noise types')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'all'],
                       help='Dataset to use for examples. Defaults to all.')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of example images to generate')
    parser.add_argument('--output_dir', type=str, default='./noise_examples',
                       help='Directory to save examples')
    parser.add_argument('--test_noise', type=str, 
                       choices=['gaussian', 'salt_pepper', 'uniform', 'poisson', 'jpeg', 'impulse'],
                       help='Test a specific noise type')
    
    args = parser.parse_args()
    
    if args.generate_examples:
        if args.dataset == "all":
            datasets = ["mnist", "cifar10", "cifar100", "stl10"]
            for d in datasets:
                generate_examples(
                    dataset_name=d,
                    num_examples=args.num_examples,
                    output_dir=os.path.join(args.output_dir, d)  # separate dirs
                )
        else:
            generate_examples(
                dataset_name=args.dataset,
                num_examples=args.num_examples,
                output_dir=args.output_dir
            )
    
    elif args.test_noise:
        # Test specific noise type
        print(f"\nTesting {args.test_noise} noise...")
        
        # Create a simple test image
        test_image = torch.rand(3, 64, 64)  # Random RGB image
        
        # Apply noise
        noise_gen = NoiseGenerator()
        noisy_image = noise_gen.add_noise(test_image, args.test_noise)
        
        print(f"\tOriginal image shape: {test_image.shape}")
        print(f"\tOriginal image range: [{test_image.min():.3f}, {test_image.max():.3f}]")
        print(f"\tNoisy image shape: {noisy_image.shape}")
        print(f"\tNoisy image range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
        print("\tTest completed successfully")
    
    else:
        # Default: show noise parameters
        noise_gen = NoiseGenerator()
        print("\nAvailable noise types and parameters:")
        for noise_type, params in noise_gen.get_noise_params().items():
            print(f"\t{noise_type}: {params}")