def early_stopping(val_losses: list, patience: int) -> dict:
    """
    Returns the best and stopping epoch indices in a dictionary.
    """
    # Initialize trackers using the first epoch (index 0)
    best_loss = val_losses[0]
    best_epoch = 0
    stop_epoch = len(val_losses) - 1
    wait_counter = 0
    
    # Scan remaining validation losses starting from epoch 1
    for epoch in range(1, len(val_losses)):
        current_loss = val_losses[epoch]
        
        if current_loss < best_loss:
            # Strict improvement found: update best records and reset patience counter
            best_loss = current_loss
            best_epoch = epoch
            wait_counter = 0
        else:
            # No improvement: increment counter
            wait_counter += 1
            
        # Check if patience has been exhausted
        if wait_counter == patience:
            stop_epoch = epoch
            break
            
    return {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch
    }
