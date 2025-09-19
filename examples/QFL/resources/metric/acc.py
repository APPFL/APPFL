import torch
import numpy as np


def accuracy(y_true, y_pred):
    """Calculate accuracy metric"""
    # Handle numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Convert predictions to class indices if they are logits
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = torch.argmax(y_pred, dim=1)

    # Ensure y_true is the right type
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.long()

    correct = (y_pred == y_true).float()
    return correct.mean().item()
