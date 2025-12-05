import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# Add project root to path to ensure modules can be imported
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import GTSRBDataset
from src.model import GTSRBNet

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to {seed}")

def get_data_loaders(data_dir, batch_size, val_split, seed):
    """
    Prepares the Training and Validation DataLoaders with appropriate transforms.
    """
    csv_path = os.path.join(data_dir, 'Train.csv')

    # Define Transformations
    # Training: Augmented (Rotation, Shift, Color Jitter)
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation: Clean (Resize + Normalize only)
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Instantiate Datasets
    train_dataset = GTSRBDataset(csv_file=csv_path, root_dir=data_dir, transform=train_transform, use_roi=True)
    val_dataset = GTSRBDataset(csv_file=csv_path, root_dir=data_dir, transform=val_transform, use_roi=True)

    # Split Indices
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))

    # Shuffle using the fixed seed
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Create Samplers and Loaders
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)

    return train_loader, val_loader

def train_model(args):
    """
    Main training loop.
    """
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Training on device: {device}")

    # 2. Prepare Data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size, args.val_split, args.seed)
    print(f"📊 Training Batches: {len(train_loader)} | Validation Batches: {len(val_loader)}")

    # 3. Initialize Model
    model = GTSRBNet(num_classes=43).to(device)

    # 4. Define Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 5. Training Loop
    best_acc = 0.0
    print(f"🚀 Starting training for {args.epochs} epochs...")
    start_time = time.time()

    # Ensure models directory exists
    os.makedirs('../models', exist_ok=True)

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_acc = 100 * correct / total

        # Logging
        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_acc:.2f}%")

        # Save Best Model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), '../models/gtsrb_best_model.pth')
            print(f"   🎉 New best model saved! ({best_acc:.2f}%)")
        
        # Step Scheduler
        scheduler.step()

    total_time = time.time() - start_time
    print(f"\n🏁 Training finished in {total_time/60:.2f} minutes.")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 on GTSRB")
    
    # Arguments allows running from terminal: python src/train.py --epochs 20
    parser.add_argument('--data_dir', type=str, default='../data/GTSRB', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    
    set_seed(args.seed)
    train_model(args)