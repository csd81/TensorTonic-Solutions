# <span style="font-size: 20px;">AdamW (Decoupled Weight Decay)</span>

## The Problem with L2 + Adam

In vanilla SGD, adding an L2 penalty $\frac{\lambda}{2}\|w\|^2$ to the loss is equivalent to weight decay: the gradient becomes $g + \lambda w$, and the update $w \leftarrow w - \alpha(g + \lambda w) = (1 - \alpha\lambda)w - \alpha g$ shrinks weights by a fixed fraction each step.

With adaptive optimizers like Adam, this equivalence breaks. When L2 is added to the gradient, the penalty term $\lambda w$ passes through Adam's moment estimates and gets divided by $\sqrt{\hat{v}}$. Parameters with large gradient history (large $\hat{v}$) receive almost no regularization, while parameters with small gradient history are over-regularized. The regularization becomes entangled with the adaptive learning rate.

## The AdamW Fix

Loshchilov and Hutter (2019) proposed decoupling weight decay from the gradient-based update. Instead of adding $\lambda w$ to the gradient, AdamW applies weight decay directly to the parameters after the Adam step:

**Adam + L2** (coupled):

$$
g_t' = g_t + \lambda w_t, \quad \text{then run Adam on } g_t'
$$

**AdamW** (decoupled):

$$
w_{t+1} = w_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda w_t
$$

In AdamW, the decay $\alpha \lambda w_t$ is applied uniformly to all parameters, independent of their gradient history. This restores the intended regularization behavior.

## Why It Matters

With Adam + L2:
- Parameters with large gradients have large $\hat{v}$, so the L2 term is divided by a large number: almost no regularization
- Parameters with small gradients have small $\hat{v}$, so the L2 term gets amplified: too much regularization

With AdamW:
- Every parameter is decayed by exactly $\alpha \lambda$ per step, regardless of gradient magnitude
- The adaptive learning rate only affects the gradient-based update, not the regularization

This produces more uniform regularization across parameters and typically leads to better generalization, which is why AdamW is the default optimizer for transformer training.
