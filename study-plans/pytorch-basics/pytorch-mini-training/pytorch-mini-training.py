import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    # Ensure the model is in training mode
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in dataloader:
        # 1. Reset gradients from the previous batch
        optimizer.zero_grad()
        
        # 2. Forward pass to get predictions
        predictions = model(X_batch)
        
        # 3. Compute the batch loss
        loss = criterion(predictions, y_batch)
        
        # 4. Accumulate the raw Python float value of the loss
        total_loss += loss.item()
        num_batches += 1
        
        # 5. Backward pass to calculate gradients
        loss.backward()
        
        # 6. Update model parameters
        optimizer.step()
        
    # Return the unrounded mean batch loss
    return total_loss / num_batches
