# <span style="font-size: 20px;">Dropout from Scratch</span>

## Overfitting in Neural Networks

Deep networks have enough capacity to memorize training data. Dropout is a regularization technique that prevents neurons from co-adapting by randomly disabling them during training.

## How Dropout Works

During each training forward pass, each hidden unit is independently set to zero with probability $p$ (the dropout rate). The surviving units are scaled by $\frac{1}{1-p}$ to maintain the expected output magnitude. This is called "inverted dropout."

$$
\tilde{a}_i = \frac{a_i \cdot m_i}{1 - p}, \quad m_i \sim \text{Bernoulli}(1 - p)
$$

At test time, no dropout is applied, and no scaling is needed (the inverted scaling during training already accounts for this).

## Why Inverted Dropout

Without the $1/(1-p)$ scaling during training, you would need to scale weights by $(1-p)$ at test time. Inverted dropout avoids modifying the network at test time, which is simpler and the standard approach in modern frameworks.

## Backpropagation Through Dropout

The dropout mask and scaling factor must be applied during backpropagation as well. If a unit was zeroed out in the forward pass, its gradient is also zeroed out:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial \tilde{a}_i} \cdot \frac{m_i}{1 - p}
$$

## Ensemble Interpretation

Dropout can be viewed as training an exponential number of sub-networks simultaneously. At test time, the full network approximates the average prediction of all these sub-networks.