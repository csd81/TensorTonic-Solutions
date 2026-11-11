import jax
import jax.numpy as jnp

def lr_schedule(step: jax.Array, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> jax.Array:
    """
    Returns the float32 learning rate for every entry of step, with the same shape as step.
    """
    # Ensure step is treated as float32 for smooth division and math operations
    step_f = step.astype(jnp.float32)
    
    # 1. Warmup Phase (s < W)
    warmup_lr = peak_lr * (step_f / warmup_steps)
    
    # 2. Cosine Decay Phase (s >= W)
    # Calculate progress p clamped between 0 and 1
    p = jnp.minimum((step_f - warmup_steps) / (total_steps - warmup_steps), 1.0)
    # Ensure p doesn't go negative if step < warmup_steps (handled gracefully before selection)
    p = jnp.maximum(p, 0.0) 
    
    cosine_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + jnp.cos(jnp.pi * p))
    
    # 3. Select the correct LR based on whether we are in the warmup phase
    out_lr = jnp.where(step < warmup_steps, warmup_lr, cosine_lr)
    
    return out_lr
