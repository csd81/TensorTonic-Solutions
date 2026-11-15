import torch

def deep_supervision_train(
    x: torch.Tensor, 
    labels: torch.Tensor, 
    y_init: torch.Tensor, 
    z_init: torch.Tensor, 
    n: int, 
    T: int, 
    n_sup: int, 
    lr: float, 
    ema_decay: float, 
    params: dict[str, torch.Tensor], 
    prefix_len: int
) -> tuple[list[float], int, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    
    # Deep copy trainable parameters and initialize EMA shadow weights
    theta = {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}
    theta_ema = {k: v.clone().detach() for k, v in params.items()}
    
    # Broadcast initial carries over batch dimension (batch size is 1)
    if y_init.dim() == 1:
        y_carry = y_init.unsqueeze(0).expand(x.shape[0], -1, -1) if y_init.dim() == 2 else y_init.unsqueeze(0).unsqueeze(0) if y_init.dim() == 1 and x.ndim == 3 else y_init.clone().detach()
    else:
        y_carry = y_init.clone().detach()
        
    # Ensure y_carry and z_carry have shape (1, L, D) or matching x
    if y_carry.ndim == 1:
        y_carry = y_carry.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1).clone()
    if z_init.ndim == 1:
        z_carry = z_init.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1).clone()
    else:
        z_carry = z_init.clone().detach()

    recorded_losses = []
    steps_run = 0

    for s in range(1, n_sup + 1):
        steps_run = s
        
        # Run deep recursion step
        y_tilde, z_tilde, y_hat, q_hat = deep_recursion_step(
            x, y_carry, z_carry, n, T, theta, prefix_len
        )
        
        # Compute loss
        loss_val = trm_loss(y_hat, labels, q_hat)
        recorded_losses.append(loss_val.item())
        
        # Backward pass
        loss_val.backward()
        
        # Plain gradient descent update (no momentum or weight decay)
        with torch.no_grad():
            for k in theta:
                if theta[k].grad is not None:
                    theta[k] -= lr * theta[k].grad
                    theta[k].grad.zero_()
            
            # Update EMA weights: theta_ema = mu * theta_ema + (1 - mu) * theta
            for k in theta_ema:
                theta_ema[k] = ema_decay * theta_ema[k] + (1.0 - ema_decay) * theta[k]
                
        # Detach carry tensors for the next step (stopgrad)
        y_carry = y_tilde.detach().clone()
        z_carry = z_tilde.detach().clone()
        
        # Check adaptive halting condition (batch size is 1)
        if q_hat.item() > 0:
            break

    # Return final detached weights and carries
    final_theta = {k: v.detach().clone() for k, v in theta.items()}
    final_ema = {k: v.detach().clone() for k, v in theta_ema.items()}
    
    return recorded_losses, steps_run, final_theta, final_ema, y_carry, z_carry
