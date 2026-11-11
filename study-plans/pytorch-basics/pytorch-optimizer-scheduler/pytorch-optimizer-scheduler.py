import torch
import torch.nn as nn

def train_with_scheduler(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.StepLR, 
    num_epochs: int
) -> dict:
    """
    Trains the supplied model for num_epochs, tracking the learning rate 
    and the unweighted mean of batch losses for each epoch.
    """
    losses = []
    lrs = []
    
    # Set the model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        # 1. Capture the learning rate used during this epoch
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(float(current_lr))
        
        batch_losses = []
        
        # 2. Iterate through batches
        for X_batch, y_batch in dataloader:
            # Clear previous gradients to avoid carrying them over
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Record the batch loss BEFORE updating parameters
            batch_losses.append(float(loss.item()))
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
        # 3. Calculate the unweighted mean loss for the epoch
        epoch_loss = sum(batch_losses) / len(batch_losses)
        losses.append(epoch_loss)
        
        # 4. Advance the scheduler once after the complete epoch
        scheduler.step()
        
    return {"losses": losses, "lrs": lrs}
