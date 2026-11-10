# <span style="font-size: 20px;">Gradient Clipping</span>

## The Exploding Gradient Problem

When gradients become very large (due to ill-conditioned data, unstable architectures, or long sequences in RNNs), the parameter update can overshoot dramatically, causing training to diverge. Gradient clipping caps the gradient magnitude to prevent this.

## Two Clipping Strategies

### Clip by Value
Each gradient element is independently clipped to the range $[-c, c]$:

$$
\hat{g}_i = \max(-c, \min(g_i, c))
$$

This is simple but changes the gradient direction when components are clipped unevenly.

### Clip by Norm
If the gradient norm exceeds a threshold $c$, the entire gradient vector is scaled down to have norm $c$:

$$
\hat{g} = \begin{cases} g & \text{if } \|g\|_2 \leq c \\ c \cdot \frac{g}{\|g\|_2} & \text{if } \|g\|_2 > c \end{cases}
$$

This preserves the gradient direction while limiting its magnitude. It is the more commonly used strategy in practice.

## When to Use Each

- **Clip by value** is useful when individual gradient components are known to be problematic (e.g., one feature has much larger scale than others)
- **Clip by norm** is preferred when you want to preserve the gradient direction and only limit the step size. This is the standard in transformer and RNN training

## Typical Threshold Values

Common choices for the clipping threshold range from 1.0 to 10.0. The optimal value depends on the model and can be tuned by monitoring gradient norms during training.