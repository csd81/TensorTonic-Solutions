# Rotary Query and Key Embeddings

Attention on its own has no sense of order, so a transformer needs some way to know where each token sits. Rotary position embeddings (RoPE), used in Llama and most modern language models, encode position by rotating pairs of coordinates in each query and key. The heart of this problem is applying the right rotation angle to each coordinate pair and seeing why it makes attention scores depend on relative position.

## Why attention needs position information

Attention scores are dot products between queries and keys. In unmasked attention, if you shuffle the tokens of a sequence, each token's query and key are unchanged, so every score is unchanged too. Without extra information the model cannot tell "dog bites man" from "man bites dog".

The original transformer added a position vector to each token embedding. RoPE instead changes the queries and keys themselves, right before the scores are computed, and leaves the values alone.

## Rotating coordinate pairs

Split a query or key vector of even width $D$ into $D/2$ adjacent pairs, $(x_0, x_1)$, $(x_2, x_3)$ and so on. Treat each pair as a point in a plane and rotate it by an angle. For pair $i$ at position $p$ the angle is

$$
\theta_i = p \, \omega_i
$$

where $i$ runs from 0 to $D/2 - 1$ and $\omega_i$, the pair's inverse frequency, is how many radians the pair turns per position. The rotated pair is

$$
x'_{2i} = x_{2i} \cos\theta_i - x_{2i+1} \sin\theta_i
$$

$$
x'_{2i+1} = x_{2i} \sin\theta_i + x_{2i+1} \cos\theta_i
$$

This is the standard 2D rotation matrix applied to each pair. Rotation never changes a vector's length, so RoPE changes direction only. Position 0 has angle 0 and leaves the vector unchanged.

## Why rotation gives relative position

Rotate a query at position $m$ by angle $m\omega$ and a key at position $n$ by angle $n\omega$. In their dot product the two rotations combine into a single rotation by $(m - n)\omega$, so for a fixed unrotated query and key the score depends on the positions only through their difference $m - n$.

This is the main reason RoPE is popular. The model learns patterns such as "look one token back" that work the same way anywhere in the sequence. The formula itself is defined at every position, although models still need extra techniques, such as rescaling the frequencies, to work well far beyond their training length.

## Each pair rotates at its own speed

Each pair gets its own inverse frequency, supplied here as a vector of length $D/2$. In practice the frequencies are usually $\omega_i = b^{-2i/D}$ with a base $b$ such as 10,000. The first pairs rotate quickly and distinguish neighbouring tokens, while the last pairs rotate slowly and keep track of long distances. Together they act like the hands of a clock running at many different speeds.

## Shifting both tokens keeps the score

Take one pair with inverse frequency $\omega = 0.3$, a query $(2, 1)$ at position 4 and a key $(0, 1)$ at position 1. The query turns by 1.2 radians and the key by 0.3 radians:

$$
q' \approx (-0.207324, \ 2.226436), \qquad k' \approx (-0.295520, \ 0.955336)
$$

Their dot product is about $2.188264$.

Now move both tokens two places later, so the query is at position 6 and the key at position 3. The rotated vectors change to about $(-1.428252, 1.720493)$ and $(-0.783327, 0.621610)$, but their dot product is still about $2.188264$. The distance between the tokens did not change, so the score did not change either.

## Positions and pairing conventions

Positions can be one list shared by every sequence in the batch, or a separate list for each sequence. Separate lists appear when several sequences are packed together or when cached decoding starts each sequence at a different offset. The same angle is used for every head.

Some libraries pair coordinate $i$ with coordinate $i + D/2$ instead of with its neighbour. Both conventions work, but weights trained with one must be permuted before they give correct results with the other. This problem uses adjacent pairs, $(2i, 2i+1)$.

Angles, sines and cosines are computed in float32, because bfloat16 cannot represent large angles precisely. The rotated tensors are converted back to the dtype of their inputs.

## Common mistakes to avoid

- **Pairing the wrong coordinates.** This problem rotates adjacent pairs, so coordinate $2i$ goes with coordinate $2i+1$.
- **Mixing up the signs.** The first output subtracts the sine term and the second adds it.
- **Using one frequency for all pairs.** Each pair uses its own entry of the inverse-frequency vector.
- **Broadcasting positions incorrectly.** A single position list is shared by every sequence, while per-sequence lists give each sequence its own angles.
- **Rotating the values.** Only queries and keys are rotated.
- **Computing angles in bfloat16.** Large positions lose precision unless angles are computed in float32.
