## Scaled Dot-Product Attention

Attention is a mechanism that allows a model to focus on relevant parts of the input when producing each element of the output. The core operation takes three inputs: queries ($Q$), keys ($K$), and values ($V$).

The attention scores between queries and keys are computed as:

$$
S = \frac{Q K^\top}{\sqrt{d_k}}
$$

where $d_k$ is the dimensionality of the key vectors. Dividing by $\sqrt{d_k}$ prevents the dot products from becoming excessively large, which would push the softmax function into regions of extremely small gradients.

## Causal (Autoregressive) Masking

In autoregressive models (such as GPT-style language models), each position should only attend to itself and earlier positions. This is enforced by adding a mask $M$ to the scores before applying softmax:

$$
\text{Attention}(Q, K, V) = \text{softmax}(S + M) \, V
$$

The mask $M$ is defined as:

* $M_{ij} = 0$ when $j \leq i$ (attending to current or past positions is allowed)
* $M_{ij} = -\infty$ when $j > i$ (attending to future positions is blocked)

After adding $-\infty$ to future positions, the softmax operation maps those entries to zero weight, effectively preventing any information flow from the future.

## Why Masking Matters

* During training, the entire sequence is processed in parallel. Without causal masking, the model would "cheat" by looking at tokens it is supposed to predict.
* Causal masking preserves the autoregressive property: the prediction for position $i$ depends only on positions $1, 2, \ldots, i$.
* The upper-triangular structure of the mask ensures a triangular attention pattern, where earlier positions see fewer context tokens and later positions see more.

## Softmax and Weight Normalization

The softmax function converts raw scores into a probability distribution over keys for each query position:

$$
\alpha_{ij} = \frac{\exp(S_{ij} + M_{ij})}{\sum_k \exp(S_{ik} + M_{ik})}
$$

Because $\exp(-\infty) = 0$, masked positions receive exactly zero weight. The remaining weights sum to 1, ensuring each output is a proper weighted average of the allowed value vectors.