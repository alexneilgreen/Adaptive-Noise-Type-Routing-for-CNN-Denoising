import os
import argparse
import torch
import csv
from torch.utils.data import Dataset, DataLoader, random_split
import logging

from Utilities.Data_Loader import AdaptiveDataset, get_available_datasets
from Utilities.Noise_Generator import NoiseGenerator

from Utilities.Comp_Train import CompTrainer
from Models.CompModel import get_model

from Utilities.Class_Train import ClassTrainer
from Models.NoiseClassifier import NoiseTypeClassifier

from Models.Routing_Models.GaussianModel import GaussianDenoiser
from Models.Routing_Models.PoissonModel import PoissonDenoiser
from Models.Routing_Models.SaltPepperModel import SaltPepperDenoiser
from Models.Routing_Models.UniformModel import UniformDenoiser
from Models.Routing_Models.JPEGModel import JPEGDenoiser
from Models.Routing_Models.ImpulseModel import ImpulseDenoiser

# ========== Setup logging ==========

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== General Functions ==========

class NoisyDataset(Dataset):
    """
    Dataset wrapper that applies noise to clean images on-the-fly.
    """
    
    def __init__(self, clean_dataset, noise_generator, noise_types='all'):
        """
        Initialize noisy dataset.
        
        Args:
            clean_dataset: Base dataset with clean images
            noise_generator: NoiseGenerator instance
            noise_types: List of noise types to use, or 'all' for all types
        """
        self.clean_dataset = clean_dataset
        self.noise_generator = noise_generator
        
        # Define available noise types
        available_noise_types = ['gaussian', 'salt_pepper', 'uniform', 
                                'poisson', 'jpeg', 'impulse']
        
        if noise_types == 'all':
            self.noise_types = available_noise_types
        elif isinstance(noise_types, list):
            self.noise_types = noise_types
        else:
            self.noise_types = [noise_types]
        
        logger.info(f"Dataset initialized with noise types: {self.noise_types}")
    
    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self, idx):
        """
        Get item with applied noise.
        
        Returns:
            tuple: (noisy_image, clean_image, noise_type_idx)
        """
        clean_image, label = self.clean_dataset[idx]
        
        # Randomly select a noise type
        noise_type_idx = torch.randint(0, len(self.noise_types), (1,)).item()
        noise_type = self.noise_types[noise_type_idx]
        
        # Apply noise
        noisy_image = self.noise_generator.add_noise(clean_image, noise_type)
        
        return noisy_image, clean_image, noise_type_idx


def get_batch_size_for_dataset(dataset_name):
    """Auto-adjust batch size based on dataset image size."""
    batch_sizes = {
        'mnist': 64,
        'cifar10': 64,
        'cifar100': 64,
        'stl10': 32  # Larger images (96x96)
    }
    return batch_sizes.get(dataset_name.lower(), 32)


def prepare_datasets(dataset_name, noise_types='all', batch_size=None, 
                    root='./Data', num_workers=4, val_split=0.2):
    """
    Prepare train, validation, and test datasets with noise.
    
    Args:
        dataset_name (str): Name of dataset ('mnist', 'cifar10', 'cifar100', 'stl10')
        noise_types: List of noise types or 'all'
        batch_size (int): Batch size for dataloaders (None for auto-adjust)
        root (str): Root directory for datasets
        num_workers (int): Number of worker processes
        val_split (float): Validation split ratio (default: 0.2)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, num_channels, actual_batch_size)
    """
    logger.info(f"Preparing {dataset_name} dataset...")
    
    # Auto-adjust batch size if not specified
    if batch_size is None:
        batch_size = get_batch_size_for_dataset(dataset_name)
        logger.info(f"Auto-adjusted batch size: {batch_size}")
    
    # Initialize noise generator
    noise_gen = NoiseGenerator()
    
    # Load clean datasets
    train_dataset_clean = AdaptiveDataset(
        dataset_name=dataset_name,
        split='train',
        root=root
    )
    
    test_dataset_clean = AdaptiveDataset(
        dataset_name=dataset_name,
        split='test',
        root=root
    )
    
    # Split train into train and validation
    train_size = int((1 - val_split) * len(train_dataset_clean))
    val_size = len(train_dataset_clean) - train_size
    train_subset, val_subset = random_split(
        train_dataset_clean, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Train size: {train_size}, Val size: {val_size}, Test size: {len(test_dataset_clean)}")
    
    # Wrap with noise generation
    train_dataset = NoisyDataset(train_subset, noise_gen, noise_types)
    val_dataset = NoisyDataset(val_subset, noise_gen, noise_types)
    test_dataset = NoisyDataset(test_dataset_clean, noise_gen, noise_types)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Determine number of channels
    sample_image, _, _ = train_dataset[0]
    num_channels = sample_image.shape[0]
    logger.info(f"Image channels: {num_channels}")
    
    return train_loader, val_loader, test_loader, num_channels, batch_size


def save_results_to_csv(results_dict, output_dir='./Results', model_type='comp'):
    """
    Append results to a CSV file for aggregation across runs.
    
    Args:
        results_dict (dict): Dictionary with keys: dataset, model, test_loss, 
                           test_psnr, test_ssim, computation_time, total_params
        output_dir (str): Base directory for saving results
        model_type (str): 'comp' or 'routing' to determine subdirectory
    """
    # Determine subdirectory based on model type
    subdir = 'Comprehensive' if model_type == 'comp' else 'Routing'
    csv_dir = os.path.join(output_dir, subdir)
    os.makedirs(csv_dir, exist_ok=True)
    
    csv_path = os.path.join(csv_dir, 'results_summary.csv')
    file_exists = os.path.exists(csv_path)
    
    # Convert computation_time from seconds to H:M:S format
    total_seconds = int(results_dict['computation_time'])
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Create new dict with formatted time
    csv_dict = results_dict.copy()
    csv_dict['computation_time'] = time_formatted
    
    # Define field names
    fieldnames = ['dataset', 'model', 'test_loss', 'test_psnr', 'test_ssim', 
                  'computation_time', 'total_params', 'batch_size']
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(csv_dict)
    
    logger.info(f"Results appended to {csv_path}")

# ========== Comprehensive Model Functions ==========

def train_single_dataset(dataset_name, args):
    """
    Train model on a single dataset.
    
    Args:
        dataset_name (str): Name of dataset
        args: Command line arguments
        
    Returns:
        dict: Training results
    """
    logger.info("="*60)
    logger.info(f"Training on {dataset_name.upper()}")
    logger.info("="*60)
    
    # Prepare datasets
    train_loader, val_loader, test_loader, num_channels, batch_size = prepare_datasets(
        dataset_name=dataset_name.lower(),
        noise_types='all',
        batch_size=args.batch_size if args.batch_size > 0 else None,
        root=args.data_root,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info(f"Initializing {args.model} model...")
    model = get_model(
        model_type=args.model,
        in_channels=num_channels,
        out_channels=num_channels
    )
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # # Create output directory
    # output_path = os.path.join(args.output_dir, f"{args.model}_{dataset_name.lower()}")
    # os.makedirs(output_path, exist_ok=True)
    # logger.info(f"Results will be saved to: {output_path}")

    # Determine subdirectory based on model type
    subdir = 'Comprehensive' if args.model == 'comp' else 'Routing'
    output_path = os.path.join(args.output_dir, subdir, dataset_name.lower())
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Results will be saved to: {output_path}")
    
    
    trainer = CompTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        results_dir=output_path,
        early_stopping_patience=args.ESP,
        device=args.device
    )
    
    # Train the model
    logger.info("Starting training...")
    results = trainer.train(epochs=args.epochs)
    
    # Prepare results for CSV
    results_dict = {
        'dataset': dataset_name.lower(),
        'model': args.model,
        'test_loss': results['test_loss'],
        'test_psnr': results['test_psnr'],
        'test_ssim': results['test_ssim'],
        'computation_time': results['computation_time'],
        'total_params': total_params,
        'batch_size': batch_size
    }
    
    # Save to aggregated CSV
    save_results_to_csv(results_dict, args.output_dir, model_type=args.model)
    
    logger.info(f"Completed training on {dataset_name.upper()}")
    logger.info("="*60)
    
    return results

# ========== Routing Model Functions ==========

def check_classifier_exists(dataset_name, model_dir='./Models/Saved'):
    """
    Check if a trained classifier exists for the dataset.
    
    Args:
        dataset_name (str): Name of dataset
        model_dir (str): Directory where models are saved
    
    Returns:
        tuple: (exists, model_path)
    """
    model_path = os.path.join(model_dir, f'classifier_{dataset_name.lower()}.pth')
    return os.path.exists(model_path), model_path


def get_noise_model(noise_type, in_channels, out_channels):
    """
    Get the appropriate noise-specific denoiser model.
    
    Args:
        noise_type (str): Type of noise ('gaussian', 'salt_pepper', etc.)
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Returns:
        nn.Module: Noise-specific denoiser model
    """
    if noise_type == 'gaussian':
        return GaussianDenoiser(in_channels, out_channels)
    elif noise_type == 'salt_pepper':
        return SaltPepperDenoiser(in_channels, out_channels)
    elif noise_type == 'uniform':
        return UniformDenoiser(in_channels, out_channels)
    elif noise_type == 'poisson':
        return PoissonDenoiser(in_channels, out_channels)
    elif noise_type == 'jpeg':
        return JPEGDenoiser(in_channels, out_channels)
    elif noise_type == 'impulse':
        return ImpulseDenoiser(in_channels, out_channels)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def check_noise_model_exists(noise_type, dataset_name, model_dir='./Models/Saved'):
    """
    Check if a trained noise-specific model exists.
    
    Args:
        noise_type (str): Type of noise
        dataset_name (str): Name of dataset
        model_dir (str): Directory where models are saved
    
    Returns:
        tuple: (exists, model_path)
    """
    model_path = os.path.join(model_dir, f'{noise_type}_{dataset_name.lower()}.pth')
    return os.path.exists(model_path), model_path


def train_noise_specific_model(noise_type, dataset_name, args):
    """
    Train a noise-specific denoiser model.
    
    Args:
        noise_type (str): Type of noise to train for
        dataset_name (str): Name of dataset
        args: Command line arguments
        
    Returns:
        str: Path to saved model
    """
    logger.info("="*60)
    logger.info(f"Training {noise_type.upper()} Denoiser on {dataset_name.upper()}")
    logger.info("="*60)
    
    # Prepare datasets with only this noise type
    train_loader, val_loader, test_loader, num_channels, batch_size = prepare_datasets(
        dataset_name=dataset_name.lower(),
        noise_types=[noise_type],  # Single noise type
        batch_size=args.batch_size if args.batch_size > 0 else None,
        root=args.data_root,
        num_workers=args.num_workers
    )
    
    # Initialize noise-specific model
    logger.info(f"Initializing {noise_type} denoiser model...")
    model = get_noise_model(noise_type, num_channels, num_channels)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create output directory
    output_path = os.path.join(args.output_dir, 'Routing', dataset_name.lower(), noise_type)
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Results will be saved to: {output_path}")
    
    # Initialize trainer
    trainer = CompTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        results_dir=output_path,
        early_stopping_patience=args.ESP,
        device=args.device
    )
    
    # Train the model
    logger.info("Starting training...")
    results = trainer.train(epochs=args.epochs)
    
    # Save model to Models/Saved directory
    model_save_dir = os.path.join('./Models', 'Saved')
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_path = os.path.join(model_save_dir, f'{noise_type}_{dataset_name.lower()}.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info(f"Completed {noise_type} denoiser training on {dataset_name.upper()}")
    logger.info(f"Test PSNR: {results['test_psnr']:.2f} dB")
    logger.info("="*60)
    
    return model_path


def evaluate_routing_system(dataset_name, args):
    """
    Evaluate the full routing system: classifier + noise-specific denoisers.
    
    Args:
        dataset_name (str): Name of dataset
        args: Command line arguments
    """
    logger.info("="*60)
    logger.info(f"Evaluating Routing System on {dataset_name.upper()}")
    logger.info("="*60)
    
    # Prepare test dataset with all noise types
    _, _, test_loader, num_channels, batch_size = prepare_datasets(
        dataset_name=dataset_name.lower(),
        noise_types='all',
        batch_size=args.batch_size if args.batch_size > 0 else None,
        root=args.data_root,
        num_workers=args.num_workers
    )
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load classifier
    classifier_path = os.path.join('./Models/Saved', f'classifier_{dataset_name.lower()}.pth')
    classifier = NoiseTypeClassifier(in_channels=num_channels, num_classes=6)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    logger.info(f"Loaded classifier from {classifier_path}")
    
    # Load all noise-specific models
    noise_types = ['gaussian', 'salt_pepper', 'uniform', 'poisson', 'jpeg', 'impulse']
    denoisers = {}
    
    for noise_type in noise_types:
        model_path = os.path.join('./Models/Saved', f'{noise_type}_{dataset_name.lower()}.pth')
        model = get_noise_model(noise_type, num_channels, num_channels)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        denoisers[noise_type] = model
        logger.info(f"Loaded {noise_type} denoiser from {model_path}")
    
    logger.info("\nStarting routing evaluation...")
    
    # Add total_loss to metrics
    from Utilities.Composite_Loss import CompositeLoss
    criterion = CompositeLoss(
        mse_weight=0.35,
        mae_weight=0.35,
        ssim_weight=0.20,
        gradient_weight=0.10
    ).to(device)
    
    # Evaluation metrics
    from Utilities.Comp_Train import calculate_psnr, calculate_ssim
    
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    correct_routes = 0
    total_images = 0
    
    per_noise_metrics = {nt: {'loss': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'count': 0} for nt in noise_types}
    
    with torch.no_grad():
        for noisy, clean, true_noise_idx in test_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            true_noise_idx = true_noise_idx.to(device)
            
            # Classify noise type
            predicted_noise_idx, _ = classifier.predict(noisy)
            
            # Route to appropriate denoiser
            batch_size_actual = noisy.size(0)
            denoised_batch = torch.zeros_like(noisy)
            
            for i in range(batch_size_actual):
                pred_idx = predicted_noise_idx[i].item()
                true_idx = true_noise_idx[i].item()
                noise_type = noise_types[pred_idx]
                true_noise_type = noise_types[true_idx]
                
                # Denoise with predicted model
                denoised_batch[i] = denoisers[noise_type](noisy[i:i+1]).squeeze(0)
                
                # Calculate metrics
                loss = criterion(denoised_batch[i:i+1], clean[i:i+1]).item()
                psnr = calculate_psnr(denoised_batch[i:i+1], clean[i:i+1])
                ssim = calculate_ssim(denoised_batch[i:i+1], clean[i:i+1])
                
                total_loss += loss
                total_psnr += psnr
                total_ssim += ssim
                
                # Track per-noise-type metrics
                per_noise_metrics[true_noise_type]['loss'] += loss
                per_noise_metrics[true_noise_type]['psnr'] += psnr
                per_noise_metrics[true_noise_type]['ssim'] += ssim
                per_noise_metrics[true_noise_type]['count'] += 1
                
                if pred_idx == true_idx:
                    correct_routes += 1
                
                total_images += 1
    
    # Calculate averages
    avg_loss = total_loss / total_images
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    routing_accuracy = 100 * correct_routes / total_images
    
    # Calculate per-noise averages
    for noise_type in noise_types:
        if per_noise_metrics[noise_type]['count'] > 0:
            per_noise_metrics[noise_type]['loss'] /= per_noise_metrics[noise_type]['count']
            per_noise_metrics[noise_type]['psnr'] /= per_noise_metrics[noise_type]['count']
            per_noise_metrics[noise_type]['ssim'] /= per_noise_metrics[noise_type]['count']
    
    # Create output directory (without nested routing_results folder)
    output_path = os.path.join(args.output_dir, 'Routing', dataset_name.lower())
    os.makedirs(output_path, exist_ok=True)
    
    # Save results
    results = {
        'dataset': dataset_name.lower(),
        'routing_accuracy': routing_accuracy,
        'avg_loss': avg_loss,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'per_noise_metrics': per_noise_metrics,
        'total_images': total_images
    }
    
    import json
    with open(os.path.join(output_path, 'routing_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Log results
    logger.info("\n")
    logger.info("="*60)
    logger.info("ROUTING SYSTEM RESULTS")
    logger.info("="*60)
    logger.info(f"Routing Accuracy: {routing_accuracy:.2f}%")
    logger.info(f"Average Loss: {avg_loss:.4f}")  # Now saved to routing_results.json
    logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info("\nPer-Noise-Type Performance:")
    for noise_type in noise_types:
        if per_noise_metrics[noise_type]['count'] > 0:
            logger.info(f"  {noise_type}:")
            logger.info(f"    Loss: {per_noise_metrics[noise_type]['loss']:.4f}")
            logger.info(f"    PSNR: {per_noise_metrics[noise_type]['psnr']:.2f} dB")
            logger.info(f"    SSIM: {per_noise_metrics[noise_type]['ssim']:.4f}")
            logger.info(f"    Count: {per_noise_metrics[noise_type]['count']}")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_path}")


def train_noise_classifier(dataset_name, args):
    """
    Train the noise type classifier.
    
    Args:
        dataset_name (str): Name of dataset
        args: Command line arguments
        
    Returns:
        str: Path to saved classifier model
    """
    logger.info("="*60)
    logger.info(f"Training Noise Type Classifier on {dataset_name.upper()}")
    logger.info("="*60)
    
    # Prepare datasets
    train_loader, val_loader, test_loader, num_channels, batch_size = prepare_datasets(
        dataset_name=dataset_name.lower(),
        noise_types='all',
        batch_size=args.batch_size if args.batch_size > 0 else None,
        root=args.data_root,
        num_workers=args.num_workers
    )
    
    # Initialize classifier model
    logger.info("Initializing noise type classifier...")
    classifier = NoiseTypeClassifier(in_channels=num_channels, num_classes=6)
    
    # Log model parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create output directory under Routing subdirectory
    output_path = os.path.join(args.output_dir, 'Routing', dataset_name.lower())
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Results will be saved to: {output_path}")
    
    # Initialize trainer
    trainer = ClassTrainer(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        results_dir=output_path,
        early_stopping_patience=args.ESP,
        device=args.device
    )
    
    # Train the classifier
    logger.info("Starting classifier training...")
    results = trainer.train(epochs=args.classifier_epochs)
    
    # Move saved model to Models/Saved directory
    model_save_dir = os.path.join('./Models', 'Saved')
    os.makedirs(model_save_dir, exist_ok=True)
    
    final_model_path = os.path.join(model_save_dir, f'classifier_{dataset_name.lower()}.pth')
        
    logger.info(f"Completed classifier training on {dataset_name.upper()}")
    logger.info(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    logger.info("="*60)
    
    return final_model_path

# ========== Main ==========

def main(args):
    """
    Main function to run training.
    
    Args:
        args: Command line arguments
    """
    logger.info("="*60)
    logger.info("Adaptive Noise-Type Routing for CNN Denoising")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset(s): {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Batch Size: {'Auto-adjust' if args.batch_size <= 0 else args.batch_size}")
    logger.info(f"Early Stopping Patience: {args.ESP}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("="*60)
    
    # Determine which datasets to train on
    if args.dataset == 'all':
        datasets = ['mnist', 'cifar10', 'cifar100', 'stl10']
        logger.info(f"Training on all datasets: {datasets}")
    else:
        datasets = [args.dataset.lower()]

    # Comprehensive Model
    if args.model == 'comp':
        # Train on each dataset
        all_results = {}
        for dataset_name in datasets:
            try:
                results = train_single_dataset(dataset_name, args)
                all_results[dataset_name] = results
            except Exception as e:
                logger.error(f"Error training on {dataset_name}: {e}")
                raise
        
        logger.info("\n")
        logger.info("="*60)
        logger.info("All training completed successfully!")
        logger.info("="*60)
        logger.info(f"Results summary saved to: {os.path.join(args.output_dir, 'Comprehensive', 'results_summary.csv')}")
        
        return all_results
    
    # Routing Model
    else:
        logger.info("\n")
        logger.info("="*60)
        logger.info("ROUTING MODE: Checking dependencies")
        logger.info("="*60)
        
        noise_types = ['gaussian', 'salt_pepper', 'uniform', 'poisson', 'jpeg', 'impulse']
        
        for dataset_name in datasets:
            logger.info(f"\n--- Processing {dataset_name.upper()} ---")
            
            # 1. Check and train classifier if needed
            classifier_exists, classifier_path = check_classifier_exists(dataset_name)
            if classifier_exists:
                logger.info(f"✓ Classifier found: {classifier_path}")
            else:
                logger.info(f"✗ Classifier not found, training...")
                classifier_path = train_noise_classifier(dataset_name, args)
                logger.info(f"✓ Classifier trained: {classifier_path}")
            
            # 2. Check and train each noise-specific model if needed
            logger.info(f"\nChecking noise-specific models for {dataset_name}...")
            for noise_type in noise_types:
                model_exists, model_path = check_noise_model_exists(noise_type, dataset_name)
                if model_exists:
                    logger.info(f"✓ {noise_type} model found: {model_path}")
                else:
                    logger.info(f"✗ {noise_type} model not found, training...")
                    model_path = train_noise_specific_model(noise_type, dataset_name, args)
                    logger.info(f"✓ {noise_type} model trained: {model_path}")
            
            # 3. Evaluate routing system
            logger.info(f"\n✓ All models ready for {dataset_name}, evaluating routing system...")
            evaluate_routing_system(dataset_name, args)
        
        logger.info("\n")
        logger.info("="*60)
        logger.info("All routing training and evaluation completed!")
        logger.info("="*60)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Adaptive Noise-Type Routing for CNN Denoising',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and experiment settings
    parser.add_argument('--model', type=str, default='comp',
                       choices=['comp', 'routing'],
                       help='Model type: comp (Comprehensive) or routing (Adaptive Routing)')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'all'],
                       help='Dataset to use (or "all" for all datasets)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    
    parser.add_argument('--classifier_epochs', type=int, default=30,
                       help='Number of training epochs for noise classifier')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    
    parser.add_argument('--batch_size', type=int, default=0,
                       help='Batch size for training (0 for auto-adjust based on dataset)')
    
    parser.add_argument('--ESP', type=int, default=10,
                       help='Early Stopping Patience: epochs to wait before stopping')
    
    # Directory settings
    parser.add_argument('--output_dir', type=str, default='./Results',
                       help='Base directory for saving results')
    
    parser.add_argument('--data_root', type=str, default='./Data',
                       help='Root directory for datasets')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda/cpu)')
    
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Run main
    try:
        main(args)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise