import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseTypeClassifier(nn.Module):
    """
    CNN classifier to identify the type of noise present in an image.
    Outputs probabilities for 6 noise types: gaussian, salt_pepper, uniform, 
    poisson, jpeg, impulse.
    """
    
    def __init__(self, in_channels=3, num_classes=6):
        """
        Initialize the noise type classifier.
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
            num_classes (int): Number of noise types to classify (default: 6)
        """
        super(NoiseTypeClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the classifier.
        
        Args:
            x (torch.Tensor): Input noisy image [B, C, H, W]
        
        Returns:
            torch.Tensor: Logits for each noise class [B, num_classes]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x
    
    def predict(self, x):
        """
        Predict noise type class.
        
        Args:
            x (torch.Tensor): Input noisy image [B, C, H, W]
        
        Returns:
            tuple: (predicted_classes, probabilities)
                - predicted_classes: Tensor of predicted class indices [B]
                - probabilities: Tensor of class probabilities [B, num_classes]
        """
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes, probabilities


def get_noise_type_names(): #! Should probablt just be coded wherever it is called
    """
    Get the mapping of noise type indices to names.
    
    Returns:
        list: List of noise type names in order
    """
    return ['gaussian', 'salt_pepper', 'uniform', 'poisson', 'jpeg', 'impulse']


def get_model(in_channels=3, num_classes=6): #! Probably shouldn't exist
    """
    Factory function to get the noise type classifier.
    
    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of noise types to classify
    
    Returns:
        nn.Module: Noise type classifier model
    """
    return NoiseTypeClassifier(in_channels, num_classes)