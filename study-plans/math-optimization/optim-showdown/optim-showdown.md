# <span style="font-size: 20px;">Optimizer Showdown</span>

## Putting It All Together

This capstone problem compares three fundamental optimizers on the same task, demonstrating how design choices affect convergence.

## SGD (Stochastic Gradient Descent)
The simplest optimizer. Updates weights directly with the gradient scaled by a learning rate:

$$
w \leftarrow w - \alpha \nabla L
$$
Simple and memory-efficient, but convergence can be slow and sensitive to learning rate.

## SGD with Momentum
Adds a velocity term that accumulates past gradients, smoothing the optimization trajectory:

$$
v \leftarrow \beta v + \nabla L, \quad w \leftarrow w - \alpha v
$$
Momentum helps the optimizer push through noisy gradients and accelerate in consistent directions.

## Adam
Combines momentum (first moment) with adaptive per-parameter learning rates (second moment), plus bias correction:

$$
m \leftarrow \beta_1 m + (1-\beta_1)\nabla L, \quad v \leftarrow \beta_2 v + (1-\beta_2)(\nabla L)^2
$$

$$
\begin{aligned}
\hat{m} &= \frac{m}{1-\beta_1^t} \\[6pt]
\hat{v} &= \frac{v}{1-\beta_2^t} \\[6pt]
w &\leftarrow w - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
\end{aligned}
$$
Adam adapts the step size per parameter, making it robust to different feature scales.

## What to Observe

Momentum typically converges fastest on well-conditioned problems. Adam is more robust across different learning rates. Plain SGD requires careful tuning. The relative performance depends on the problem structure and hyperparameters.