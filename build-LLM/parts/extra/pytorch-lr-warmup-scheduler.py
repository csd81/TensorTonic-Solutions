import math

def warmup_cosine_schedule(base_lr: float, warmup_steps: int, total_steps: int) -> list[float]:
    """
    Returns one Python float learning rate per step.
    """
    schedule = []
    
    for t in range(total_steps):
        if t < warmup_steps:
            # Linear warmup phase
            lr = base_lr * (t + 1) / warmup_steps
        else:
            # Cosine decay phase
            progress = (t - warmup_steps) / (total_steps - warmup_steps)
            lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))
            
        schedule.append(float(lr))
        
    return schedule
