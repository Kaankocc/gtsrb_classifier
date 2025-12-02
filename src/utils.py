import os
import torch

def save_checkpoint(state, filename='checkpoint.pth'):
    """Save a training checkpoint.

    Args:
        state (dict): Should contain at least 'epoch' and 'state_dict'.
        filename (str): Destination filename.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
    torch.save(state, filename)


def load_checkpoint(model, optimizer=None, filename='checkpoint.pth', device='cpu'):
    """Load checkpoint and restore model and optimizer (if provided)"""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def accuracy(output, target):
    """Compute classification accuracy. Output expected as raw logits or probabilities."""
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct / target.size(0)


