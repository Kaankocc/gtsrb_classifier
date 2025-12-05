import torch
import torch.nn as nn
from torchvision import models

class GTSRBNet(nn.Module):
    """
    A ResNet18-based Convolutional Neural Network adapted for the GTSRB dataset.
    
    This architecture leverages Transfer Learning by initializing with weights pre-trained 
    on ImageNet. Crucially, it modifies the initial layers to handle the low resolution 
    (32x32) of traffic sign images, avoiding the aggressive downsampling found in 
    standard ResNet implementations designed for 224x224 inputs.

    Args:
        num_classes (int): The number of output classes. Defaults to 43 for GTSRB.
    """

    def __init__(self, num_classes=43):
        super(GTSRBNet, self).__init__()
        
        # 1. Load Pre-trained Weights (Transfer Learning)
        # We use the current 'DEFAULT' weights (best available ImageNet weights)
        print("Loading pre-trained ResNet18 weights...")
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        
        # 2. Adapt Input Layer for Low-Resolution Images (The "Eye")
        # Standard ResNet starts with a 7x7 convolution (stride 2) followed by a MaxPool.
        # For 224x224 images, this is fine. For 32x32 images, this aggressive downsampling
        # would shrink the feature map to 8x8 too quickly, destroying vital spatial details.
        
        # We replace the first layer with a 3x3 convolution with stride 1 to preserve dimensions.
        self.model.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3, # Smaller kernel (was 7)
            stride=1,      # Smaller stride (was 2) to preserve spatial resolution
            padding=1, 
            bias=False
        )
        
        # Remove the initial MaxPool layer to further preserve information
        # (Replaces the pooling operation with an Identity pass-through)
        self.model.maxpool = nn.Identity()

        # 3. Modify the Output Layer (The "Head")
        # The original model outputs 1000 classes (ImageNet). We replace the final 
        # Fully Connected (Linear) layer to output our specific number of classes (43).
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.model(x)