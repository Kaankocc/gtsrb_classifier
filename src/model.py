import torch
import torch.nn as nn
from torchvision import models

class GTSRBNet(nn.Module):
    def __init__(self, num_classes=43):
        super(GTSRBNet, self).__init__()
        
        # 1. Load the pre-trained ResNet18 model
        # We access the weights enum via 'models.ResNet18_Weights'
        print("Loading pre-trained ResNet18 weights...")
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        
        # 2. Modify the input layer (The "Eye")
        # Standard ResNet shrinks images too fast for 32x32 inputs.
        # We replace the first 7x7 conv with a 3x3 conv.
        self.model.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=64, 
            kernel_size=3, # Smaller kernel
            stride=1,      # Smaller stride (keep size)
            padding=1, 
            bias=False
        )
        # Remove the first maxpool to preserve information
        self.model.maxpool = nn.Identity()

        # 3. Modify the Output Layer (The "Head")
        # Replace the 1000-class ImageNet output with our 43 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)