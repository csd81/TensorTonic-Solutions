import jax
import jax.numpy as jnp
from jax import lax

def greedy_decode(params: dict, start_tokens: jax.Array, h0: jax.Array, num_steps: int) -> dict:
    """
    Returns a dict with tokens, an int32 array of shape (B, num_steps), 
    and log_probs, a float32 array of shape (B, num_steps).
    """
    # Extract weights and biases from params dict
    embed = params["embed"]  # (V, H)
    w_hh = params["w_hh"]    # (H, H)
    b_h = params["b_h"]      # (H,)
    w_out = params["w_out"]  # (H, V)
    b_out = params["b_out"]  # (V,)

    # The carry for lax.scan contains the previous tokens and the previous hidden states
    initial_carry = (start_tokens, h0)

    def scan_fn(carry, _):
        prev_tokens, prev_h = carry
        
        # Look up embeddings for the batch: shape (B, H)
        # We index into the embedding table using advanced indexing
        e_y = embed[prev_tokens]
        
        # Compute the new hidden state: h_t = tanh(E_y + h_{t-1} @ W_hh + b_h)
        h_t = jnp.tanh(e_y + jnp.dot(prev_h, w_hh) + b_h)
        
        # Compute the vocabulary entries (logits): l_t = h_t @ W_out + b_out
        logits = jnp.dot(h_t, w_out) + b_out
        
        # Pick the highest-scoring token greedily
        y_t = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        
        # Calculate log probabilities via log_softmax
        log_probs_all = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        
        # Extract the log probability of the selected greedy token per batch element
        # Using a row index array [0, 1, ..., B-1] along with the picked tokens
        batch_indices = jnp.arange(start_tokens.shape[0])
        r_t = log_probs_all[batch_indices, y_t]
        
        # Next carry is the newly generated tokens and updated hidden state
        next_carry = (y_t, h_t)
        
        # Output elements to be stacked across all steps
        output = (y_t, r_t)
        
        return next_carry, output

    # Run the loop using jax.lax.scan over num_steps iterations
    _, (tokens, log_probs) = lax.scan(scan_fn, initial_carry, None, length=num_steps)

    # lax.scan stacks outputs with the time/step dimension first (num_steps, B)
    # We transpose back to meet the required (B, num_steps) shape
    return {
        "tokens": jnp.transpose(tokens),
        "log_probs": jnp.transpose(log_probs)
    }
