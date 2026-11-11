Weight initialization determines the starting point of optimization. When all weights begin at zero, every neuron in a layer computes the same output, gradients are identical, and the network cannot break symmetry - it effectively has one neuron per layer regardless of width. Random initialization solves symmetry breaking, but the scale of the random values matters enormously.

## The Variance Problem

Consider a layer $y = Wx$ where $W$ has shape $(n_{out}, n_{in})$. If the input elements have variance $\text{Var}(x)$ and the weights are independent with variance $\text{Var}(w)$, then each output element has variance:

$$
\text{Var}(y_j) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

In a deep network with $L$ layers, the variance after $L$ layers scales as $(n \cdot \text{Var}(w))^L$. If $n \cdot \text{Var}(w) > 1$, activations explode exponentially. If $n \cdot \text{Var}(w) < 1$, they vanish exponentially.

For the normal distributions below, the second argument is the variance.

## Xavier Initialization (Glorot & Bengio, 2010)

Xavier initialization maintains the variance of activations across layers when using symmetric activations like sigmoid or tanh. The key insight: set $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$ which is a compromise between preserving variance in the forward pass (needs $1/n_{in}$) and the backward pass (needs $1/n_{out}$).

Uniform variant:

$$
W \sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\; \sqrt{\frac{6}{n_{in}+n_{out}}}\right)
$$
Normal variant:

$$
W \sim \mathcal{N}\left(0,\; \frac{2}{n_{in}+n_{out}}\right)
$$

The factor of 6 in the uniform bound comes from the variance of $U(-a, a)$ being $a^2/3$, so $a = \sqrt{3 \cdot \text{Var}} = \sqrt{\frac{6}{n_{in}+n_{out}}}$.

## He Initialization (He et al., 2015)

ReLU activations zero out roughly half of their inputs, effectively halving the variance at each layer. He initialization compensates by using a larger weight variance that only depends on $n_{in}$:

$$
\text{Var}(w) = \frac{2}{n_{in}}
$$

Uniform variant:

$$
W \sim U\left(-\sqrt{\frac{6}{n_{in}}},\; \sqrt{\frac{6}{n_{in}}}\right)
$$
Normal variant:

$$
W \sim \mathcal{N}\left(0,\; \frac{2}{n_{in}}\right)
$$

## When to Use Which

| Activation | Recommended Init |
|---|---|
| Sigmoid, Tanh | Xavier (Glorot) |
| ReLU, Leaky ReLU, ELU | He (Kaiming) |

## Common Mistakes

* Initializing all weights to zero fails to break symmetry between neurons
* Using too-small standard deviation: gradients vanish in deep networks
* Using too-large standard deviation: activations saturate, gradients explode
* Using Xavier init with ReLU: underestimates the variance needed because ReLU halves it