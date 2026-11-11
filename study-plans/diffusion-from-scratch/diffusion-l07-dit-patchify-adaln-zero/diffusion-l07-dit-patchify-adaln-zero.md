# DiT Patchify and adaLN-Zero Block

A Diffusion Transformer, or DiT, is a denoising network built from the same transformer blocks that power language models, applied to small square pieces of an image in place of words. Cutting a photo into tiles and laying the tiles out in reading order gives a sequence that a transformer can read like a sentence. The denoiser in a diffusion model also has to know how noisy its input is, so every block receives a short summary of the timestep and class that adjusts how it behaves. This page builds the block one piece at a time, starting from how one image becomes a few tokens and ending with the adaLN-Zero block whose gate makes a freshly built block leave its input unchanged. That pair of steps is what you implement in this problem.

## Why a transformer needs patches

Peebles and Xie (2023) introduced DiT as a replacement for the convolutional U-Net that earlier diffusion models used. A transformer consumes a sequence of vectors that all have the same width $D$. Treating every pixel as a token would make the sequence very long, and the cost of self-attention grows with the square of its length.

Cutting the image into non-overlapping $p \times p$ patches, where $p$ is patch_size, shortens the sequence by a factor of $p^2$. A $32 \times 32$ latent with $p = 2$ becomes $256$ tokens, and $p = 4$ gives $64$ tokens. Larger patches make the model cheaper and smaller patches give it finer detail, which is one of the main dials in the DiT paper.

## Reading the pixels of one patch in a fixed order

Start with one patch. The input x has shape $(B, C, \text{height}, \text{width})$, so each patch holds $C p^2$ numbers, $p^2$ pixels in each of the $C$ channels. Before a linear layer can use them, they have to be laid out as one vector, and the order matters.

This problem uses the $(C, p, p)$ order, which lists the whole first channel row by row, then the whole second channel, and so on. Entry $k p^2 + i p + j$ of the vector is the pixel at channel $k$, row $i$ and column $j$ of the patch. Take a two-channel $2 \times 2$ image with $p = 2$, whose first channel has rows $(1, 2)$ and $(3, 4)$ and whose second channel has rows $(5, 6)$ and $(7, 8)$. The single patch flattens to $(1, 2, 3, 4, 5, 6, 7, 8)$.

Many codebases move channels last and use the $(p, p, C)$ order instead, which would give $(1, 5, 2, 6, 3, 7, 4, 8)$ for the same patch. Both layouts are valid on their own, but a weight matrix written for one layout produces garbage tokens in the other, because each weight row then multiplies the wrong pixel. The $(C, p, p)$ order is also what a convolution with kernel size and stride both equal to $p$ uses, since its weight has shape $(D, C, p, p)$.

## Ordering the patches across the image

The patches form a grid of $\text{height}/p$ rows and $\text{width}/p$ columns, so there are $N = (\text{height}/p)(\text{width}/p)$ tokens. They are listed row by row, so token $n = r\, (\text{width}/p) + q$ is the patch in grid row $r$ and grid column $q$. A $4 \times 6$ image with $p = 2$ has a grid of 2 rows and 3 columns, and token 4 is the patch in grid row 1 and column 1.

Each flattened patch is embedded by the weight $W_e$, embed_weight with shape $(C p^2, D)$, and the bias $b_e$, embed_bias with shape $(D)$. A positional embedding $P$, pos_embed with shape $(N, D)$, is then added, with row $n$ belonging to token $n$. Attention on its own ignores the order of its inputs, so this is the only way the model learns where each patch came from.

$$
z = \operatorname{patches}(x)\, W_e + b_e + P
$$

The tokens $z$ have shape $(B, N, D)$. Listing the grid column by column would still give $N$ tokens, but each would be paired with the wrong row of $P$.

## LayerNorm without its own weights

Every token is normalized over its own $D$ features. Subtract the token's mean, divide by the square root of its variance plus a small constant, and the token ends up with mean 0 and variance close to 1.

$$
\operatorname{LN}(z) = \frac{z - \operatorname{mean}(z)}{\sqrt{\operatorname{var}(z) + \varepsilon}}
$$

The variance is the population variance, dividing by $D$, and $\varepsilon = 10^{-6}$. For a token $(1, 3)$ the mean is 2 and the variance is 1, so the normalized token is very close to $(-1, 1)$. A standard LayerNorm then applies a learned per-feature scale and shift, but this one has none, because the conditioning takes over that job.

The constant matters only when a token's variance is tiny, which happens right after initialization when patch embeddings are small. With a variance of $10^{-4}$, the PyTorch default of $10^{-5}$ in place of $10^{-6}$ changes the normalized values by about $4\%$. The unbiased variance changes them by a factor that depends on $D$, about $1.41$ when $D = 2$.

## Letting the conditioning steer the normalization

A denoiser must know the timestep, and a class-conditional one must know the label. DiT combines them into one conditioning vector $c$ per sample, of shape $(B, E)$ with conditioning width $E$. A small network turns $c$ into three per-feature vectors.

$$
[\, s_{\text{shift}},\ s_{\text{scale}},\ g \,] = \operatorname{SiLU}(c)\, W_a + b_a
$$

Here $\operatorname{SiLU}(c) = c \cdot \operatorname{sigmoid}(c)$, $W_a$ is ada_weight with shape $(E, 3D)$ and $b_a$ is ada_bias with shape $(3D)$. The $(B, 3D)$ result splits along its last axis into shift, scale and gate in that order, each of shape $(B, D)$. All $N$ tokens of a sample share the same three vectors.

The normalized token is modulated as $\operatorname{LN}(z) \odot (1 + s_{\text{scale}}) + s_{\text{shift}}$, where $\odot$ is elementwise multiplication. The scale is written as $1 + s_{\text{scale}}$ so that a zero output of the small network leaves the normalized token unchanged. Multiplying by $s_{\text{scale}}$ alone would wipe out every feature whenever the modulation is zero.

## A gate on the residual branch

The modulated token passes through the sublayer, here a single linear projection with $W_m$, mlp_weight of shape $(D, D)$, and $b_m$, mlp_bias of shape $(D)$. A real DiT block has self-attention and an MLP in this place, each with its own modulation. The sublayer output is multiplied by the gate before it is added back to the token.

$$
\text{out} = z + g \odot \Big( \big( \operatorname{LN}(z) \odot (1 + s_{\text{scale}}) + s_{\text{shift}} \big) W_m + b_m \Big)
$$

In adaLN-Zero, $W_a$ and $b_a$ are initialized to zero. The gate is then zero for every input, the residual branch contributes nothing, and the whole block is exactly the identity at the start of training. A deep stack of such blocks starts as a clean pass-through of the patch tokens, which keeps early gradients well behaved. Peebles and Xie found that this variant gave clearly better FID than feeding the conditioning as extra tokens or through cross-attention.

As training proceeds, the gate learns how much each block should change the tokens, separately for each timestep and class. A missing gate breaks the identity start and removes that control.

## One token through the block

Take one token $z = (1, 3)$ with $D = 2$. Suppose the conditioning network produced shift $(0.5, 0)$, scale $(1, -0.5)$ and gate $(0.2, 0)$, and the projection is the identity with zero bias.

1. The token mean is 2 and its population variance is 1, so $\operatorname{LN}(z) \approx (-1, 1)$, with a change below $10^{-6}$ from $\varepsilon$.
2. The factor $1 + s_{\text{scale}}$ is $(2, 0.5)$, so the scaled token is $(-2, 0.5)$, and adding the shift gives $(-1.5, 0.5)$.
3. The identity projection leaves $(-1.5, 0.5)$, and the gate multiplies it by $(0.2, 0)$ to give $(-0.3, 0)$.
4. Adding the residual gives $\text{out} = (1 - 0.3,\ 3 + 0) = (0.7, 3)$.

The second feature came through untouched because its gate is zero. With a zero gate on both features the output would be the token $(1, 3)$ itself, which is what every freshly initialized block does.

## Where this block sits in a full DiT

A full DiT stacks many such blocks, each with a modulated and gated self-attention sublayer and a modulated and gated MLP. A final adaLN layer and a linear map then turn each token back into a patch, and the patches are reassembled into the image shape. The network predicts the noise for latent diffusion, where the image is first compressed by an autoencoder. Scaling the width, depth and token count of this design improves FID steadily, which is the main finding of the DiT paper.

## Common mistakes to avoid

- **Flattening patches in (p, p, C) order.** The embedding weight rows then multiply the wrong pixels whenever there is more than one channel.
- **Listing the patch grid column by column.** Every token pairs with the wrong positional embedding row and the sequence no longer matches the image layout.
- **Multiplying by the scale instead of 1 + scale.** A zero modulation then zeroes the normalized tokens and destroys the identity start.
- **Forgetting to apply the gate.** The block no longer starts as the identity and loses its per-timestep control over the residual branch.
- **Using the default LayerNorm epsilon or the unbiased variance.** Both change the normalized values, most visibly when the token variance is small.
- **Swapping shift and scale when splitting the modulation.** The split order is fixed by the weights, so a swap changes every output.
- **Adding the residual to the normalized token.** The skip connection carries the original tokens $z$, and using $\operatorname{LN}(z)$ there discards their scale and mean.