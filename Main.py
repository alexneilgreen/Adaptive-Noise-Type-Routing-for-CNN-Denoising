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
        
        logger.info("\n" + "="*60)
        logger.info("All training completed successfully!")
        logger.info("="*60)
        logger.info(f"Results summary saved to: {os.path.join(args.output_dir, 'Comprehensive', 'results_summary.csv')}")
        
        return all_results
    
    # Routing Model
    else:
        logger.info("\n" + "="*60)
        logger.info("ROUTING MODE: Checking for trained classifiers")
        logger.info("="*60)
        
        for dataset_name in datasets:
            classifier_exists, classifier_path = check_classifier_exists(dataset_name)
            
            if classifier_exists:
                logger.info(f"Classifier found for {dataset_name}: {classifier_path}")
            else:
                logger.info(f"Classifier not found for {dataset_name}")
                logger.info(f"Training classifier for {dataset_name}...")
                classifier_path = train_noise_classifier(dataset_name, args)
                logger.info(f"Classifier trained and saved: {classifier_path}")
        
        logger.info("="*60)
        logger.info("All classifiers ready. Proceeding with routing model training...")
        logger.info("="*60 + "\n")


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