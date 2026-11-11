import jax
import jax.numpy as jnp

def ensemble_predict(params: dict, x: jax.Array) -> dict:
    """
    Returns a dict with outputs of shape (M, N, O) and mean of shape (N, O), both float32.
    """
    # Define the forward pass for a single MLP member
    def single_member_predict(single_params, x_input):
        # f(x) = relu(x * W1 + b1) * W2 + b2
        h = jnp.maximum(jnp.dot(x_input, single_params['w1']) + single_params['b1'], 0)
        return jnp.dot(h, single_params['w2']) + single_params['b2']

    # Vectorize across axis 0 of params, while keeping x shared (None)
    # The structure of in_axes mirrors the structure of the inputs to single_member_predict
    vmap_predict = jax.vmap(
        single_member_predict, 
        in_axes=({'w1': 0, 'b1': 0, 'w2': 0, 'b2': 0}, None)
    )
    
    # Compute outputs for all members simultaneously
    outputs = vmap_predict(params, x)
    
    # Calculate the average across the M members (axis 0)
    mean_output = jnp.mean(outputs, axis=0)
    
    return {
        "outputs": outputs,
        "mean": mean_output
    }
