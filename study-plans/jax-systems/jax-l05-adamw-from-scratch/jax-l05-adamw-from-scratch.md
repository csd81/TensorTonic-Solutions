# AdamW from Scratch

AdamW is the optimizer behind most large neural network training today. It gives every parameter its own step size based on the history of that parameter's gradients, and it shrinks the weights a little at every step in a way that does not depend on those gradients. This page builds the method one piece at a time, from the two running averages of Adam, through the bias correction that fixes their slow start, to the decoupled weight decay that gives AdamW its W. It ends with how JAX code carries the optimizer's memory from one step to the next as a pytree.

## One learning rate does not fit every parameter

Plain gradient descent moves every parameter by the same learning rate times its gradient. Different parameters, though, see gradients of very different sizes. An embedding row for a rare word might get a tiny gradient once in a while, while a bias in the last layer gets a large one every step. A learning rate small enough to keep the large gradients stable makes the rare parameters crawl, and one large enough for the rare parameters makes the others diverge.

Adam (Kingma and Ba, 2015) solves this by dividing each parameter's step by a running measure of that parameter's typical gradient size. Parameters with large gradients take proportionally smaller steps, and parameters with small gradients take proportionally larger ones.

## Two running averages

Adam keeps two numbers for every parameter. The first moment $m$ is an exponential moving average of the gradient, and the second moment $v$ is an exponential moving average of the squared gradient.

$$
m = \beta_1 \, m_{\text{old}} + (1 - \beta_1) \, g, \qquad v = \beta_2 \, v_{\text{old}} + (1 - \beta_2) \, g^{2}
$$

Here $g$ is the current gradient, and $\beta_1$ and $\beta_2$ are decay rates between 0 and 1 that set how long each average remembers. Typical values are $\beta_1 = 0.9$, which remembers roughly the last 10 steps, and $\beta_2 = 0.999$, which remembers roughly the last 1,000.

The first moment is a smoothed gradient direction, much like the velocity of momentum. The square root of the second moment is the root mean square of recent gradients, a measure of their typical size regardless of sign. Adam steps by the ratio of the two. When the gradient points the same way step after step, the ratio is close to plus or minus 1. When the gradient is noisy and keeps changing sign, the first moment averages towards 0 while the second moment stays large, so the ratio and the step shrink.

## Why the averages start too small

Both moments start at zero before the first step. A moving average that starts at zero needs many steps to warm up, so early on it underestimates the quantity it tracks.

Take a gradient that is exactly 1 at every step, with $\beta_1 = 0.9$. After one, two and three steps, the first moment is 0.1, 0.19 and 0.271, far below the true average of 1. These numbers are exactly $1 - 0.9^{t}$ for steps $t = 1, 2, 3$, and this is true in general. With a constant gradient, the first moment after $t$ steps equals the gradient times $1 - \beta_1^{t}$. The same holds for the second moment, which with $\beta_2 = 0.999$ is 0.001, 0.001999 and 0.002997 after three steps.

Dividing by that factor removes the bias.

$$
\hat m = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat v = \frac{v}{1 - \beta_2^{\,t}}
$$

In the example, each corrected first moment is exactly 1. The step counter $t$ starts at 1 on the first update, because the formula describes the average after $t$ gradients have been mixed in. With $t = 0$ the denominators would be zero.

The correction matters most for the second moment, whose factor $1 - \beta_2^{t}$ stays far from 1 for hundreds of steps. Without any correction, the first step in the example would use $0.1 / \sqrt{0.001} \approx 3.162$ instead of 1, so the first updates would be about three times too large, exactly when the model is most fragile.

## What epsilon protects against

The update divides by $\sqrt{\hat v}$, which is zero for a parameter that has never had a nonzero gradient. A small constant $\epsilon$ in the denominator keeps the division finite.

$$
\text{Adam direction} = \frac{\hat m}{\sqrt{\hat v} + \epsilon}
$$

Here $\epsilon$ is added after the square root. The position matters when $\hat v$ is small. With $\hat v = 10^{-6}$ and $\epsilon = 10^{-3}$, the denominator is $0.001 + 0.001 = 0.002$ when $\epsilon$ is added after the root, and $\sqrt{0.001001} \approx 0.03164$ when it is added inside, which is about 16 times larger. Implementations that look alike can therefore take very different steps for parameters with small gradients.

## Two ways to shrink the weights

Weight decay pulls every weight a little towards zero at each step, which keeps weights small and usually helps the model generalize. For plain gradient descent there is a classic way to get it, adding $\lambda p$ to the gradient, where $\lambda$ is the decay strength. This is called L2 regularization, because $\lambda p$ is the gradient of the penalty $\frac{\lambda}{2} p^2$.

Inside Adam, that extra gradient goes through the same normalization as the real one, so it gets divided by $\sqrt{\hat v}$. Suppose $\lambda p = 0.1$ for two weights. For a weight whose gradients are large, with $\sqrt{\hat v} = 10$, the decay contributes only 0.01 to the step. For a weight with small gradients, with $\sqrt{\hat v} = 0.01$, it contributes 10. The decay ends up weakest exactly on the weights that are most active.

Loshchilov and Hutter (2019) proposed decoupling the two. AdamW computes the Adam direction from the true gradient alone and then subtracts the decay separately, scaled by the learning rate.

$$
p_{\text{new}} = p - \eta \left( \frac{\hat m}{\sqrt{\hat v} + \epsilon} + \lambda \, p \right)
$$

Here $\eta$ is the learning rate. The decay term uses the parameter from before the update and shrinks every weight by the same fraction $\eta \lambda$, however large or small its gradients are. The moments never see the decay, so they keep tracking only the gradient of the loss.

## One full step on one number

Take a single parameter $p = 2$ with gradient $g = -0.4$. The stored step counter is 2, so this update uses $t = 3$, and the moments left by earlier steps are $m_{\text{old}} = -0.2$ and $v_{\text{old}} = 0.05$. Use $\eta = 0.05$, $\beta_1 = 0.8$, $\beta_2 = 0.99$, $\epsilon = 10^{-8}$ and $\lambda = 0.1$.

The first moment becomes $0.8 \cdot (-0.2) + 0.2 \cdot (-0.4) = -0.24$, and the second moment becomes $0.99 \cdot 0.05 + 0.01 \cdot 0.16 = 0.0511$. The correction factors are $1 - 0.8^{3} = 0.488$ and $1 - 0.99^{3} \approx 0.029701$, which give $\hat m \approx -0.49180$ and $\hat v \approx 1.72048$. The second moment is scaled up far more than the first, because $\beta_2$ is closer to 1, so its average warms up more slowly.

The square root of $\hat v$ is about 1.31167, so the Adam direction is about $-0.49180 / 1.31167 \approx -0.37494$. The decay term is $0.1 \cdot 2 = 0.2$, so the parameter becomes $2 - 0.05 \cdot (-0.37494 + 0.2) \approx 2.00875$. The negative gradient pushes the parameter up, and the decay pulls it back down a little.

The new state holds the step counter 3, the first moment $-0.24$ and the second moment 0.0511, ready for the next update.

## Optimizer state is just another pytree

A JAX optimizer is a pure function. It takes the parameters, the gradients and the optimizer state, and it returns new parameters and a new state, leaving the old ones untouched. The training loop passes the returned state into the next step, so the optimizer's memory lives in an ordinary value instead of inside an object.

For AdamW, that state is a small pytree, a dict with the step counter and two trees of moments. The mu and nu trees have exactly the structure of the parameters, because every parameter needs its own first and second moment, so a tree map over the parameters and the two moment trees together reaches every leaf. Keeping the step counter as an int32 array means the whole state is made of arrays, so it can be passed through jax.jit, written to a checkpoint and split across devices in the same way as the parameters.

This layout also shows the memory cost of Adam. The two moment trees together hold twice as many numbers as the model itself, which is why large-scale training systems often split the optimizer state across devices.

## Common mistakes to avoid

- **Skipping the bias correction.** Without it the second moment is far too small for the first few hundred steps, and the early updates are several times too large.
- **Correcting with the old step count.** The correction must use $t$ = step + 1, otherwise the first update divides by zero and every later one uses the wrong factor.
- **Adding the decay to the gradient.** Mixing $\lambda p$ into the gradient turns AdamW back into Adam with L2 regularization, where the decay is divided by $\sqrt{\hat v}$ and the moments are changed.
- **Moving epsilon inside the square root.** The denominator becomes $\sqrt{\hat v + \epsilon}$, which gives very different steps whenever $\hat v$ is small.
- **Decaying the updated parameter.** The decay term uses the parameter from before the step, so applying it after the Adam step gives slightly different numbers.
- **Forgetting to advance the step.** Returning the old counter makes the next call reuse the same bias correction, and the state no longer counts the updates.