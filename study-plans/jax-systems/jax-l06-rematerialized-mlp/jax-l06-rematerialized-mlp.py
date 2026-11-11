import jax
import jax.numpy as jnp

def remat_mlp_block(params: dict, x: jax.Array, y: jax.Array) -> dict:
    def block(p, inputs):
        h1 = jnp.matmul(inputs, p["w1"]) + p["b1"]
        hidden = jnp.tanh(h1)
        return inputs + jnp.matmul(hidden, p["w2"]) + p["b2"]

    def loss_fn(p):
        # Wrap and call the function inline
        checkpointed_block = jax.checkpoint(block, policy=jax.checkpoint_policies.dots_saveable)
        out = checkpointed_block(p, x)
        return jnp.mean((out - y) ** 2)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    return {"loss": loss_val, "grads": grads}
