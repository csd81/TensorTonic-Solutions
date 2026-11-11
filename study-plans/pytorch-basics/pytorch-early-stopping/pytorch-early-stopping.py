import torch
import torch.nn as nn

def train_with_early_stopping(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    max_epochs: int, 
    patience: int
) -> dict:
    """
    Trains the supplied model using an early stopping criteria based on validation loss.
    Returns train_losses, val_losses as float lists, and stopped_epoch as an int.
    """
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    stopped_epoch = 0
    
    for epoch in range(1, max_epochs + 1):
        # 1. Training Pass
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Loss measured BEFORE batch update
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            loss.backward()
            optimizer.step()
            
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # 2. Validation Pass (Model after that epoch's training)
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                epoch_val_loss += loss.item()
                num_val_batches += 1
                
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        stopped_epoch = epoch
        
        # 3. Early Stopping Metric Assessment
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0  # Reset patience counter
        else:
            epochs_without_improvement += 1  # Increment on tie or worse loss
            
        if epochs_without_improvement >= patience:
            break
            
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stopped_epoch": stopped_epoch
    }
