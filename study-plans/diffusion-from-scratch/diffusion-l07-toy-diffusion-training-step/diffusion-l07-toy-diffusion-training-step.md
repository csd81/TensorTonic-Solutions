# Toy Diffusion Training Step

Training a diffusion model is one small routine repeated a very large number of times, much like a student practising the same kind of exercise until it becomes easy. Each round takes some clean examples, buries them under a random amount of noise, asks the network to guess which noise was added, and nudges the weights so the next guess is a little better. A network that can name the noise at every level is exactly what a sampler later needs to remove it. This page builds that round one piece at a time, from noising a single number to a full update of a tiny network with gradient clipping, which is the training step you implement in this problem.

## Noising a batch at random timesteps

The forward process of a diffusion model has a closed form that jumps from a clean sample straight to any timestep. With $\bar\alpha_t$ the fraction of variance that still comes from the clean data at timestep $t$, a clean value $x_0$ and a standard Gaussian draw $\epsilon$ give

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon .
$$

Take $x_0 = 1.5$, $\epsilon = -0.5$ and $\bar\alpha_t = 0.64$. The signal weight is $0.8$ and the noise weight is $0.6$, so $x_t = 1.2 - 0.3 = 0.9$.

In a training step, x0 and noise have shape $(B, D)$ and timesteps holds one 0-based timestep per sample. The schedule alpha_bars holds $\bar\alpha_t$ for every timestep, so each sample reads its own entry at its own index. Each sample's two weights then scale every feature of its own row, which means they must line up with the batch axis. Mixing timesteps in one batch trains the network at many noise levels at once.

## Predicting the noise as the target

Ho et al. (2020) showed that the variational bound of a DDPM simplifies, up to weighting, to a plain regression. The network $\hat\epsilon$ looks at $x_t$ and the timestep and tries to output the $\epsilon$ that was mixed in. The loss is the mean squared error over every entry of the batch.

$$
L = \operatorname{mean}\big( (\hat\epsilon - \epsilon)^2 \big)
$$

The mean runs over all $B \cdot D$ entries. Given $x_t$ and $\bar\alpha_t$, knowing the noise is the same as knowing the clean sample, since either one can be solved from the closed form. The noise is the more convenient target because it always has variance 1, which keeps the regression on the same scale at every timestep.

## Telling the network which timestep it sees

The same value of $x_t$ could come from a nearly clean sample at a small $t$ or from almost pure noise at a large $t$, so the network must be told the timestep. A raw integer is a poor input, because its scale grows with $t$ and nearby values look almost identical to a linear layer. The sinusoidal embedding from the transformer literature maps $t$ to cosines and sines at several frequencies.

With half equal to embed_dim // 2, the frequencies are

$$
f_i = \exp\Big( -\ln(10000)\, \frac{i}{\text{half}} \Big), \qquad i = 0, \dots, \text{half} - 1 .
$$

The first frequency is always 1 and the later ones shrink geometrically towards $1 / 10000$. The embedding $e(t)$ lists $\cos(t f_i)$ for every $i$ first and then $\sin(t f_i)$ for every $i$. When embed_dim is odd, one zero is appended so the width matches.

High frequencies change quickly and separate neighbouring timesteps, and low frequencies change slowly and separate distant ones. Take embed_dim $= 5$, so half $= 2$ and the frequencies are 1 and $0.01$. At $t = 3$ the embedding is about $(-0.989992, 0.999550, 0.141120, 0.029996, 0)$, two cosines, two sines and the padding zero. The order of the halves is a convention, and the weights that read the embedding depend on it.

## A tiny network for toy data

For toy data the network can stay very small. The noisy row $x_t$ of length D and the embedding of length embed_dim are concatenated along the feature axis into one row of length $D + \text{embed\_dim}$. One hidden layer follows, then a projection back to D outputs, one predicted noise value per data feature.

$$
\hat\epsilon = \operatorname{SiLU}\big( [\, x_t,\ e(t) \,]\, W_1 + b_1 \big)\, W_2 + b_2
$$

Here $W_1$ has shape $(D + \text{embed\_dim}, \text{hidden})$, $b_1$ has shape $(\text{hidden})$, $W_2$ has shape $(\text{hidden}, D)$ and $b_2$ has shape $(D)$. The activation is $\operatorname{SiLU}(z) = z \cdot \operatorname{sigmoid}(z)$, a smooth relative of ReLU that lets small negative values through. Each sample gets the embedding of its own timestep, so the embedding rows differ across the batch.

## Measuring the whole gradient at once

Backpropagation gives one gradient tensor for each of the four weight tensors. A diffusion loss mixes very different noise levels in one batch, and an unlucky batch can produce a gradient many times larger than usual, which would throw the weights far from a good region. Gradient clipping caps the size of the step without changing its direction.

The size is measured over all parameters together. With gradients $g_1, \dots, g_K$ for the $K$ tensors, the global norm is

$$
\lVert g \rVert = \sqrt{ \textstyle\sum_k \lVert g_k \rVert_F^2 },
$$

where $\lVert g_k \rVert_F^2$ is the sum of the squares of every entry of one tensor. This is the length of the gradient if all the tensors were flattened and joined into one long vector.

## Clipping by the global norm

With $c$ equal to max_grad_norm, every gradient is multiplied by the same factor

$$
s = \min\Big( 1,\ \frac{c}{\lVert g \rVert + 10^{-6}} \Big) .
$$

When the norm is below $c$ the factor is 1 and nothing changes. When it is above, the factor shrinks the whole gradient to a length of about $c$. The small constant avoids a division by zero when every gradient vanishes. This is the rule torch.nn.utils.clip_grad_norm_ follows, and the value logged during training is the norm before clipping.

Suppose one tensor has gradient $(3, 4)$ and another has gradient $(12)$, with $c = 6.5$.

1. The tensor norms are 5 and 12, so the global norm is $\sqrt{25 + 144} = 13$.
2. The clip factor is $6.5 / (13 + 10^{-6}) \approx 0.5$.
3. The clipped gradients are about $(1.5, 2)$ and $(6)$, which point in the same direction as before at half the length.

Clipping each tensor by its own norm would leave $(3, 4)$ untouched, since 5 is below $6.5$, and shrink $(12)$ to $6.5$. That rotates the step, because one part shrinks and the other does not, and it lets the total length grow with the number of tensors.

## One plain SGD update

After clipping, plain stochastic gradient descent moves every weight against its own part of the clipped gradient, $\theta \leftarrow \theta - \eta\, s\, g_\theta$, where $\eta$ is learning_rate and $g_\theta$ is the gradient of that tensor. A weight of $0.5$ with gradient 2, a learning rate of $0.1$ and a clip factor of $0.5$ moves to $0.5 - 0.1 \cdot 0.5 \cdot 2 = 0.4$.

The loss and the gradient norm are both measured before the update. The new weights are fresh tensors with the same keys, shapes and dtype, and the caller's weights stay as they were. Real diffusion models use Adam or AdamW with an exponential moving average of the weights, and they apply the same global-norm clipping before each optimizer step.

## Where this step sits in the pipeline

This step is the inner loop of training. Repeated over many batches with fresh timesteps and noise, it teaches the network to predict the noise at every level, and the trained weights are exactly what a sampler such as DDPM needs to run the reverse process. The capstone problem that follows uses the same network and embedding to generate samples.

## Common mistakes to avoid

- **Feeding x0 to the network instead of x_t.** The network then never sees noisy inputs and cannot learn to remove noise.
- **Using one timestep for the whole batch.** Every sample must read alpha_bars and its embedding at its own timestep, otherwise all but one get the wrong noise level.
- **Swapping the sine and cosine halves of the embedding.** Every weight that reads the embedding then multiplies the wrong feature.
- **Clipping each tensor by its own norm.** Tensors are scaled unevenly, which changes the update direction and the total step size.
- **Reporting the gradient norm after clipping.** The logged value then sits at max_grad_norm whenever clipping is active and hides how large the gradients really were.
- **Updating the weights in place.** The caller's tensors change underneath it, which breaks code that keeps the old weights for comparison or averaging.