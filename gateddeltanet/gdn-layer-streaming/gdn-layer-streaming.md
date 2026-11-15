# Gated DeltaNet Layer

The Gated DeltaNet layer is the token mixer that replaces self-attention in every Gated DeltaNet block. It reads queries, keys and values from the input path, stores associations in a recurrent state through the gated delta rule, and then normalizes, gates and projects the read-out. This problem covers that output side and the cache that lets the layer process a sequence in pieces.

## A state that never grows

A softmax attention layer at inference keeps every past key and value. After $L$ tokens its cache holds $L$ key vectors and $L$ value vectors per head, so memory and per-token work grow with the length of the conversation.

The Gated DeltaNet layer keeps two things instead.

- **The recurrent state.** Each head stores one matrix $S \in \mathbb{R}^{d_v \times d_k}$, in the paper's orientation of value width by key width, where $d_k$ is the key width and $d_v$ the value width. Writing a new association updates this matrix in place, so its size never depends on how many tokens have been read.
- **The convolution cache.** The short causal convolutions in the input path look back $K_c - 1$ tokens, where $K_c$ is the kernel width. Keeping the last $K_c - 1$ pre-convolution rows of each branch is enough to continue the convolution exactly.

Together these give a constant-size inference state. Handing them from one call to the next makes a split run identical to a single run, which is what lets the layer decode one token at a time at constant cost. In the H1 and H2 hybrids, only the sliding-window attention layers keep a key-value cache, and the window is fixed at 2K tokens, so the whole stack stays bounded.

## Why the read-out is normalized and gated

The state accumulates many decayed associations, so the scale of a read-out $o_{t,h} = S_{t,h} q'_{t,h}$ drifts with the decay, the write strengths and how well the current query matches what was stored. Following RetNet (Sun et al., 2023), the paper processes the output through normalization and gating before the output projection.

- **Per-head RMS normalization.** Each head output is divided by its own root mean square over $d_v$ channels and multiplied by a learned weight $\omega$ shared by all heads. Normalizing each head separately keeps one head with a large read-out from shrinking the others.
- **SiLU output gate.** A projection of the layer input gives one gate value per channel, and $\operatorname{SiLU}(u) = u \, \sigma(u)$ lets the input decide which read-out channels pass. Because the gate is computed from the token itself and not from the state, it can switch off a stale memory even when the state still returns a strong signal.

In the Table S.1 ablations, the full 400M model reaches an average perplexity of 27.35. Removing the output gate raises it to 29.12, while removing the output norm gives 27.55, so the paper calls the gate crucial and the norm a marginal improvement.

## Conventions this problem fixes

- The paper never scales queries, and the official kernel multiplies them by $d_k^{-1/2}$, so this problem scales the L2-normalized queries by $d_k^{-1/2}$ before the recurrence.
- The Figure 1 caption says the output gate applies a linear projection with SiLU, so the gate is SiLU and not a sigmoid.
- The output norm is RMSNorm per head over $d_v$ channels with one weight of length $d_v$ shared by heads, as in the official layer, and its epsilon defaults to $10^{-5}$.
- The cache layout is this exercise's choice. It pairs the input path's convolution cache, with the newest row last, and the recurrent state in the paper's orientation, while the official code stores the transposed state.

## Normalizing and gating one head

Take one head with $d_v = 2$ whose read-out is $o = (0.3, 0.4)$, a norm weight $\omega = (1, 0.5)$ and gate logits $(2, -2)$. The mean square is $(0.09 + 0.16)/2 = 0.125$, so the RMS factor is $1/\sqrt{0.125} \approx 2.828$ and the normalized head is $(0.849, 1.131)$. The weight turns this into $(0.849, 0.566)$. The gate values are $\operatorname{SiLU}(2) \approx 1.762$ and $\operatorname{SiLU}(-2) \approx -0.238$, so the gated head is about $(1.495, -0.135)$. The second channel is almost closed, and its small negative value shows that SiLU, unlike a sigmoid, can flip the sign of a weak channel.

## Common mistakes to avoid

- **Skipping the query scale.** Without $d_k^{-1/2}$ the read-out is larger, and although the RMS norm removes most of that scale, the epsilon term then weighs differently and the outputs no longer match the official layer.
- **Using a sigmoid gate.** A sigmoid keeps every channel between 0 and 1, which is a different function from the SiLU gate the paper describes.
- **Normalizing across all heads jointly.** One RMS factor for the concatenated heads lets a loud head dampen a quiet one, which per-head normalization is meant to prevent.
- **Gating before normalizing.** If the gate is applied first, the RMS norm partly undoes it and the gate loses control over channel magnitudes.
- **Ignoring the incoming state.** Starting every call from zeros breaks streaming, because the second call forgets everything the first call stored.
- **Returning the old convolution cache.** The next call then convolves against stale rows, so a split run no longer matches a single run.