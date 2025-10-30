import os
import csv
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Setup logging
logger = logging.getLogger(__name__)


class ClassTrainer:
    """Trainer class for noise type classifier."""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 learning_rate=0.001, results_dir='./results', 
                 early_stopping_patience=10, device='cuda'):
        """
        Initialize classifier trainer.
        
        Args:
            model: Noise classifier model
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Early stopping variables
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.best_model_state = None
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Noise type names
        self.noise_types = ['gaussian', 'salt_pepper', 'uniform', 'poisson', 'jpeg', 'impulse']
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Classifier Trainer initialized on device: {self.device}")
        logger.info(f"Using Cross-Entropy Loss")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for noisy, _, noise_type_idx in self.train_loader:
            noisy = noisy.to(self.device)
            noise_type_idx = noise_type_idx.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(noisy)
            loss = self.criterion(outputs, noise_type_idx)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += noise_type_idx.size(0)
            correct += (predicted == noise_type_idx).sum().item()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for noisy, _, noise_type_idx in self.val_loader:
                noisy = noisy.to(self.device)
                noise_type_idx = noise_type_idx.to(self.device)
                
                # Forward pass
                outputs = self.model(noisy)
                loss = self.criterion(outputs, noise_type_idx)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += noise_type_idx.size(0)
                correct += (predicted == noise_type_idx).sum().item()
                
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def test(self):
        """Test the model on test set and generate confusion matrix."""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        per_class_correct = {i: 0 for i in range(len(self.noise_types))}
        per_class_total = {i: 0 for i in range(len(self.noise_types))}
        
        with torch.no_grad():
            for noisy, _, noise_type_idx in self.test_loader:
                noisy = noisy.to(self.device)
                noise_type_idx = noise_type_idx.to(self.device)
                
                # Forward pass
                outputs = self.model(noisy)
                loss = self.criterion(outputs, noise_type_idx)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += noise_type_idx.size(0)
                correct += (predicted == noise_type_idx).sum().item()
                
                # Per-class accuracy
                for i in range(noise_type_idx.size(0)):
                    label = noise_type_idx[i].item()
                    per_class_total[label] += 1
                    if predicted[i] == label:
                        per_class_correct[label] += 1
                
                test_loss += loss.item()
                
                # Store for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(noise_type_idx.cpu().numpy())
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for i in range(len(self.noise_types)):
            if per_class_total[i] > 0:
                per_class_accuracy[self.noise_types[i]] = 100 * per_class_correct[i] / per_class_total[i]
            else:
                per_class_accuracy[self.noise_types[i]] = 0.0
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return avg_loss, accuracy, per_class_accuracy, cm
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.noise_types,
                    yticklabels=self.noise_types)
        plt.title('Confusion Matrix - Noise Type Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def save_epoch_data(self):
        """Save epoch-level training data to CSV."""
        epoch_data = {
            'Epoch': list(range(1, len(self.train_losses) + 1)),
            'Train_Loss': self.train_losses,
            'Val_Loss': self.val_losses,
            'Train_Accuracy': self.train_accuracies,
            'Val_Accuracy': self.val_accuracies
        }
        
        csv_path = os.path.join(self.results_dir, 'epoch_data.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_data.keys())
            writer.writerows(zip(*epoch_data.values()))
        
        logger.info(f"Epoch data saved to {csv_path}")
    
    def train(self, epochs=50):
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            dict: Training results including test_accuracy, test_loss, 
                  per_class_accuracy, confusion_matrix, computation_time, model_path
        """
        logger.info(f"Starting classifier training for {epochs} epochs")
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
        
        log_message(f"Starting classifier training on device: {self.device}")
        log_message(f"Total epochs: {epochs}")
        log_message("="*60)
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Log progress
            log_message(f"Epoch {epoch}/{epochs}")
            log_message(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            log_message(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            
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
        test_loss, test_accuracy, per_class_accuracy, confusion_mat = self.test()
        
        total_time = time.time() - start_time
        
        log_message("\n" + "="*60)
        log_message("FINAL TEST RESULTS")
        log_message("="*60)
        log_message(f"Test Loss: {test_loss:.4f}")
        log_message(f"Test Accuracy: {test_accuracy:.2f}%")
        log_message("\nPer-Class Accuracy:")
        for noise_type, acc in per_class_accuracy.items():
            log_message(f"  {noise_type}: {acc:.2f}%")
        log_message(f"\nTotal Computation Time: {total_time:.2f} seconds")
        log_message("="*60)
        
        # Close log file
        log_file.close()
        
        # Save confusion matrix
        cm_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(confusion_mat, cm_path)
        
        # Save epoch data
        self.save_epoch_data()
        
        # Save model to Models/Saved directory
        model_save_dir = os.path.join('./Models', 'Saved')
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Extract dataset name from results_dir path
        # results_dir format: ./Results/classifier_<dataset>
        dataset_name = os.path.basename(self.results_dir).replace('classifier_', '')
        model_path = os.path.join(model_save_dir, f'classifier_{dataset_name}.pth')
        
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Prepare results dictionary
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'per_class_accuracy': per_class_accuracy,
            'confusion_matrix': confusion_mat.tolist(),
            'computation_time': total_time,
            'epochs_trained': len(self.train_losses),
            'model_path': model_path
        }
        
        # Save results to JSON
        with open(os.path.join(self.results_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Classifier training complete. Results saved to {self.results_dir}")
        
        return results