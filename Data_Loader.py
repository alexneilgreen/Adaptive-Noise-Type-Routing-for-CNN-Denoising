import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import argparse
from typing import Dict, List, Tuple, Optional

class AdaptiveDataset(Dataset):
    """
    Adaptive dataset that loads clean images.
    Noise generation is handled separately by noise_generator.py
    """
    
    def __init__(self, 
                 dataset_name: str, 
                 split: str = 'train',
                 root: str = './Data',
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize the adaptive dataset.
        
        Args:
            dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar100', 'stl10')
            split: Dataset split ('train', 'test', 'val')
            root: Root directory for datasets
            transform: Additional transforms to apply
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.root = root
        self.transform = transform
        
        # Load the appropriate dataset
        self.dataset = self._load_dataset()
    
    def _load_dataset(self):
        """Load the specified dataset."""
        # Base transforms - convert to tensor and normalize to [0,1]
        base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if self.dataset_name == 'mnist':
            # Check if MNIST is already downloaded
            mnist_path = os.path.join(self.root, 'MNIST')
            download_needed = not os.path.exists(mnist_path)
            
            if self.split == 'train':
                return torchvision.datasets.MNIST(
                    root=self.root, train=True, download=download_needed, transform=base_transform
                )
            else:  # test
                return torchvision.datasets.MNIST(
                    root=self.root, train=False, download=download_needed, transform=base_transform
                )
        
        elif self.dataset_name == 'cifar10':
            # Check if CIFAR-10 is already downloaded
            cifar10_path = os.path.join(self.root, 'cifar-10-batches-py')
            download_needed = not os.path.exists(cifar10_path)
            
            if self.split == 'train':
                return torchvision.datasets.CIFAR10(
                    root=self.root, train=True, download=download_needed, transform=base_transform
                )
            else:  # test
                return torchvision.datasets.CIFAR10(
                    root=self.root, train=False, download=download_needed, transform=base_transform
                )
        
        elif self.dataset_name == 'cifar100':
            # Check if CIFAR-100 is already downloaded
            cifar100_path = os.path.join(self.root, 'cifar-100-python')
            download_needed = not os.path.exists(cifar100_path)
            
            if self.split == 'train':
                return torchvision.datasets.CIFAR100(
                    root=self.root, train=True, download=download_needed, transform=base_transform
                )
            else:  # test
                return torchvision.datasets.CIFAR100(
                    root=self.root, train=False, download=download_needed, transform=base_transform
                )
        
        elif self.dataset_name == 'stl10':
            # Check if STL-10 is already downloaded
            stl10_path = os.path.join(self.root, 'stl10_binary')
            download_needed = not os.path.exists(stl10_path)
            
            # STL-10 has train/test/unlabeled splits
            split_map = {'train': 'train', 'test': 'test', 'val': 'test'}
            return torchvision.datasets.STL10(
                root=self.root, split=split_map[self.split], download=download_needed, transform=base_transform
            )
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Returns:
            tuple: (image, label) - clean image and its label
        """
        # Get original image and label
        if hasattr(self.dataset, 'data'):
            # For datasets like CIFAR that have .data attribute
            image, label = self.dataset[idx]
        else:
            # For other datasets
            image, label = self.dataset[idx]
        
        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloader(dataset_name: str, 
                  split: str = 'train',
                  batch_size: int = 32,
                  shuffle: bool = True,
                  root: str = './Data',
                  num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        dataset_name: Name of dataset
        split: Dataset split
        batch_size: Batch size
        shuffle: Whether to shuffle data
        root: Root directory for datasets
        num_workers: Number of worker processes
    
    Returns:
        DataLoader object
    """
    dataset = AdaptiveDataset(
        dataset_name=dataset_name,
        split=split,
        root=root
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def get_available_datasets() -> List[str]:
    """Return list of available datasets."""
    return ['mnist', 'cifar10', 'cifar100', 'stl10']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test data loader')
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['all'] + get_available_datasets(),
                       help='Dataset to load (default: all)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if args.dataset == 'all':
        datasets_to_test = get_available_datasets()
    else:
        datasets_to_test = [args.dataset]
    
    for dataset_name in datasets_to_test:
        print(f"\nTesting data loader with {dataset_name} dataset...")
        
        try:
            # Create dataloader
            dataloader = get_dataloader(
                dataset_name=dataset_name,
                split='train',
                batch_size=args.batch_size
            )
            
            # Test a few batches
            for i, (images, labels) in enumerate(dataloader):
                print(f"\tBatch {i+1}:")
                print(f"\t\tImages shape: {images.shape}")
                print(f"\t\tLabels shape: {labels.shape}")
                
                if i >= 1:  # Just test 2 batches per dataset
                    break
            
            print(f"\t✅ {dataset_name} loaded successfully")
            
        except Exception as e:
            print(f"\t❌ Error loading {dataset_name}: {e}")
    
    print("\nData loader test completed")