import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional

def save_checkpoint(state: Dict[str, Any], filename: str = 'checkpoint.pth') -> None:
    """
    Saves the training checkpoint (model weights, optimizer state, epoch).

    Args:
        state (dict): Dictionary containing 'epoch', 'state_dict', 'optimizer', etc.
        filename (str): Path to save the checkpoint file.
    """
    # Ensure the directory exists before saving
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
        
    torch.save(state, filename)
    # print(f"✅ Checkpoint saved to {filename}")

def load_checkpoint(model: nn.Module, 
                    optimizer: Optional[optim.Optimizer] = None, 
                    filename: str = 'checkpoint.pth', 
                    device: str = 'cpu') -> Dict[str, Any]:
    """
    Loads a checkpoint and restores the model and optimizer states.

    Args:
        model (nn.Module): The model to load weights into.
        optimizer (optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint to ('cpu' or 'cuda'/'mps').

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving start_epoch).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"❌ Checkpoint file not found: {filename}")

    print(f"Loading checkpoint from '{filename}'...")
    checkpoint = torch.load(filename, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state if provided and available
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    return checkpoint

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the classification accuracy for a batch.

    Args:
        output (torch.Tensor): Raw model logits or probabilities (Batch Size x Num Classes).
        target (torch.Tensor): Ground truth labels (Batch Size).

    Returns:
        float: Accuracy as a decimal (0.0 to 1.0).
    """
    with torch.no_grad():
        # Get the index of the max log-probability
        pred = output.argmax(dim=1)
        
        # Compare prediction with target
        correct = pred.eq(target).sum().item()
        
        # Calculate ratio
        return correct / target.size(0)