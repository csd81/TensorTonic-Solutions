# Multi-Head Latent Attention

The previous attention variants reduce key/value memory by sharing heads. Multi-head latent attention takes a different approach: it stores a compact latent description from which keys and values can be reconstructed.

Imagine that every token normally leaves behind two detailed records, one key and one value. MLA first writes a shorter summary. When attention needs the key or value view, it reads that same summary through two different learned transformations.

This problem implements that central compression idea in a deliberately simplified form. It does not ask you to reproduce every component of DeepSeek-V2 attention.

## One shared latent, two reconstructions

Let $X$ contain the hidden states. A down-projection compresses each token from model width to latent width:

$$
C = XW_{\mathrm{down}}
$$

The tensor $C$ is the shared latent representation. Two different up-projections turn it into keys and values:

$$
K = CW_{\mathrm{up},K}
$$

$$
V = CW_{\mathrm{up},V}
$$

Both reconstructions start from exactly the same $C$, but they are not expected to be equal. $W_{\mathrm{up},K}$ learns how the latent features should become matching information, while $W_{\mathrm{up},V}$ learns how they should become returned content.

The queries take the direct path required by this exercise:

$$
Q = XW_q
$$

Once $Q$, $K$, and $V$ have model width, they are split into heads and processed with ordinary multi-head attention.

## What “latent” means here

A latent representation is an intermediate description learned by the model. It is not directly a key, a value, or the final attention output.

Suppose the model width is 8 and the latent width is 3. Every token begins with 8 hidden features, is compressed to 3 latent features, and is then reconstructed into an 8-feature key and an 8-feature value.

The three latent numbers do not have fixed human labels. Training discovers whatever compressed features are most useful for reconstructing both attention roles.

Calling the representation shared is important. If you create one down-projection for keys and another for values, you have two latent tensors. That may still be a low-rank design, but it is not the joint K/V compression requested by this problem.

## Why the bottleneck is low rank

Consider the path from $X$ to $K$:

$$
K = XW_{\mathrm{down}}W_{\mathrm{up},K}
$$

Even though the final key has model width, all of its information must pass through the narrower latent space. The rank of the combined transformation cannot exceed the latent width. The value path has the same restriction.

This bottleneck is the price of compression. A smaller latent can be retained more compactly, but it gives the reconstruction less independent capacity. The task is not asking you to find an ideal latent width; it provides the matrices and asks you to implement their meaning correctly.

The constraints allow the latent width to equal the model width. That case offers no dimensional reduction, but it remains mathematically valid and should not be rejected.

## How this differs from MQA and GQA

MQA and GQA share explicit K/V heads. MQA creates one K/V head and lets every query head use it. GQA creates several K/V heads and shares each one within a group.

This MLA exercise does not repeat a compact K/V head. It retains one latent vector per token, reconstructs full model-width K and V from that vector, and then splits the reconstructions into the normal number of heads.

The difference can be stated plainly:

- MQA and GQA reduce how many K/V heads exist.
- MLA reduces the representation from which all K/V heads are reconstructed.

Both approaches seek a smaller persistent representation, but they impose different structures on the model.

## Attention after reconstruction

After reconstruction, divide $Q$, $K$, and $V$ into $h$ heads. One head has width

$$
d_k = \frac{d_{\mathrm{model}}}{h}
$$

Head $i$ performs

$$
H_i
=
\operatorname{softmax}\left(
\frac{Q_iK_i^{\mathsf T}}{\sqrt{d_k}} + M
\right)V_i
$$

The latent width is not used in the scaling factor. The dot product being scaled is between a query head and a reconstructed key head, both of width $d_k$.

After attention, concatenate all head outputs and apply $W_o$. This produces the final tensor with the same batch size, sequence length, and model width as $X$.

## A worked path through the function

Take a batch with 3 tokens, model width 8, latent width 4, and 4 attention heads.

First, $W_{\mathrm{down}}$ turns each 8-feature hidden state into a 4-feature latent. The returned latent tensor therefore contains one four-number summary for each token.

Next, $W_{\mathrm{up},K}$ and $W_{\mathrm{up},V}$ independently turn each four-number summary into an 8-feature key and an 8-feature value. $W_q$ directly turns the original hidden state into an 8-feature query.

Finally, each of those model-width tensors is divided into 4 heads of width 2. The four attention results are concatenated and passed through $W_o$.

There are two outputs from the function:

- the attention result, with model width 8 for every token,
- the shared latent $C$, with width 4 for every token.

Returning $K$ instead of $C$ would miss the point of the interface. $K$ is a reconstruction; $C$ is the compact state from which both reconstructions can be reproduced.

## Why return the latent tensor?

The function is designed around an inference-facing representation. Separate full-width keys and values contain $2d_{\mathrm{model}}$ numbers per token. The latent contains $d_{\mathrm{latent}}$ numbers per token.

For model width 8 and latent width 4, separate K and V contain 16 numbers per token, while the latent contains 4. This count explains the intended memory benefit of retaining the latent.

The simplified implementation in this problem still reconstructs explicit K and V before computing attention. The full MLA design includes additional algebra and positional handling, but those details are outside this function. Adding them would change the expected calculation rather than improve it.

## Causal behavior

Compression does not change which positions are allowed to communicate. In causal mode, query position $t$ may use reconstructed keys and values from positions up to $t$, but not later positions.

Apply the causal mask to the score matrix before softmax. Do not mask the latent tensor itself. Causality describes a relationship between query and key positions, while $C$ is simply the per-token representation used to build K and V.

## The minimal tensor picture

For hidden states $(B,S,d_{\mathrm{model}})$, the down-projection produces

$$
C \in \mathbb{R}^{B\times S\times d_{\mathrm{latent}}}
$$

The two up-projections return K and V to $(B,S,d_{\mathrm{model}})$. Q already has that shape. Head splitting then arranges each as $(B,h,S,d_k)$ for ordinary multi-head attention.

The latent is never split into heads. Its job is to serve as the shared bottleneck before reconstruction.

## Common mistakes to avoid

- Computing separate latent tensors for K and V breaks joint compression.
- Projecting K or V directly from $X$ bypasses the latent path.
- Projecting Q from $C$ adds a query bottleneck that this problem does not request.
- Splitting $C$ into heads confuses latent width with reconstructed head width.
- Reusing one up-projection for both paths unnecessarily forces K and V to be identical.
- Scaling by $\sqrt{d_{\mathrm{latent}}}$ is wrong; attention uses $\sqrt{d_k}$.
- Returning reconstructed K or V instead of $C$ violates the function contract.

The implementation becomes much easier to reason about when kept in three stages: compress once, reconstruct two views, then run familiar multi-head attention.
