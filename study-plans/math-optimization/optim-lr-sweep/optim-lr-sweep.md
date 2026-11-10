# <span style="font-size: 20px;">Learning Rate Sweep</span>

## The Learning Rate

The learning rate $\alpha$ is the single most important hyperparameter in gradient descent. It scales the gradient before each weight update:

$$
w_{t+1} = w_t - \alpha \nabla L(w_t)
$$

Choosing $\alpha$ poorly leads to one of two failure modes: convergence that is impractically slow, or divergence where the loss explodes.

## Why the Learning Rate Matters

For a quadratic loss $L(w) = \frac{1}{2} c \, w^2$ with curvature $c$, the gradient descent update becomes:

$$
w_{t+1} = (1 - \alpha c) \, w_t
$$

This converges if and only if $|1 - \alpha c| < 1$, which gives $0 < \alpha < 2/c$. The optimal rate is $\alpha^* = 1/c$, achieving convergence in a single step.

For neural networks, the "curvature" varies across parameters and changes during training, so no single $\alpha$ is optimal everywhere. A learning rate sweep empirically maps out the convergence landscape.

## The Three Regimes

A sweep over learning rates typically reveals three regimes:

- **Too small** ($\alpha \ll \alpha^*$): Each update barely changes the weights. The loss decreases but so slowly that training would require impractical numbers of epochs. The loss curve is nearly flat.

- **Optimal range** ($\alpha \approx \alpha^*$): The loss decreases steadily and reaches a low value within the training budget. This is the sweet spot for training.

- **Too large** ($\alpha > 2/L$ where $L$ is the Lipschitz constant of the gradient): Updates overshoot, causing the loss to oscillate or increase. In extreme cases, weights grow without bound and the loss diverges to infinity or NaN.

## The MLP Architecture

This problem uses a single-hidden-layer MLP with ReLU activation and softmax output:

$$
z_1 = X W_1 + b_1, \quad a_1 = \text{ReLU}(z_1)
$$

$$
z_2 = a_1 W_2 + b_2, \quad \hat{y} = \text{softmax}(z_2)
$$

The loss is cross-entropy:

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
$$

where $y_{ik}$ is the one-hot encoded label.

## Backpropagation Through the MLP

The gradient of cross-entropy with softmax simplifies to:

$$
\frac{\partial L}{\partial z_2} = \frac{\hat{y} - y}{n}
$$

Then propagating backward through the linear layer and ReLU:

$$
\begin{aligned}
\frac{\partial L}{\partial W_2} &= a_1^T \frac{\partial L}{\partial z_2} \\[6pt]
\frac{\partial L}{\partial b_2} &= \sum_i \frac{\partial L}{\partial z_{2,i}}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial a_1} &= \frac{\partial L}{\partial z_2} W_2^T \\[6pt]
\frac{\partial L}{\partial z_1} &= \frac{\partial L}{\partial a_1} \odot \mathbb{1}[z_1 > 0]
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial W_1} &= X^T \frac{\partial L}{\partial z_1} \\[6pt]
\frac{\partial L}{\partial b_1} &= \sum_i \frac{\partial L}{\partial z_{1,i}}
\end{aligned}
$$

## Weight Initialization

All learning rates must start from the same initial weights to make the comparison fair. Small random initialization (e.g., $W \sim \mathcal{N}(0, 0.01^2)$) ensures symmetry breaking while keeping initial activations in a reasonable range. Using a fixed random seed guarantees reproducibility.

## Practical Takeaways

- Always sweep learning rates on a log scale (e.g., $10^{-5}$ to $10^0$) before committing to a value
- The optimal LR depends on batch size, architecture, and data: there is no universal best value
- Loss curves that oscillate wildly indicate the LR is near the stability boundary
- Modern optimizers (Adam, AdamW) are less sensitive to LR choice but still benefit from tuning
