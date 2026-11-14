# Bidirectional Mask Predictor

The mask predictor is the network at the heart of LLaDA (Nie et al., 2025). It reads a partially masked sequence and predicts every masked token at once. This problem covers the single change LLaDA makes to a standard Transformer, which is to remove the causal mask from self-attention, along with the additive bias that keeps left padding out of every softmax.

## Why an autoregressive model needs the causal mask

An autoregressive model factorizes a sequence as a product of next-token conditionals. When it is trained with teacher forcing, the whole sequence is fed in at once, so the attention scores must be masked above the diagonal. Otherwise position $i$ could read token $i + 1$, which is the very token it is being trained to predict.

LLaDA defines its model differently. A forward process replaces each token with the mask token $M$ independently, and the mask predictor $p_\theta(\cdot \mid x_t)$ is trained to recover the original token at every masked position of the noisy sequence $x_t$. A masked token in the middle of a sentence carries no information about itself, so letting it see both neighbors leaks nothing. The paper says the predictor does not use a causal mask "as its formulation allows it to see the entire input for predictions." With the mask gone, a query at position $i$ scores every key $j$, so the weight matrix $A_{b,h}$ of shape $(L, L)$ is generally full, above the diagonal as well as below it.

## Why the time input disappears

Continuous diffusion models such as DDPM add Gaussian noise and feed the noise level to the network, because the noisy input alone does not reveal how much noise was added. LLaDA's noise is an absorbing mask, and Eq. 11 of the paper shows that the clean-token prediction at a masked position depends only on the unmasked tokens, which are identical to the clean data and carry no trace of the mask ratio $t$. The mask predictor therefore takes no timestep embedding. The partially masked sequence is its entire input.

## Why bidirectionality rules out the KV cache

During autoregressive decoding, a new token never changes the keys and values of earlier positions, because those positions could not attend to it. That property is what makes a key-value cache valid. In LLaDA, revealing a token at position $j$ changes the hidden states of every position that attends to $j$, including earlier ones, so cached keys and values from the previous step become stale. The paper calls LLaDA incompatible with KV caching for this reason and uses vanilla multi-head attention, with as many key and value heads as query heads, instead of grouped query attention.

## The padding bias follows the official code

Batches in the official sampler are padded on the left. Padded keys must receive zero weight, but the score matrix still needs finite values. This exercise follows the official model code, which converts the integer mask into an additive key bias of $0$ for real tokens and $\beta_{\min}$, the most negative finite value of the dtype, for padding. Adding $\beta_{\min}$ to a padded column drives its exponential to exactly zero once the softmax subtracts the row maximum, while keeping every score finite. The bias is a function of the key position only, so it is broadcast across heads and query rows. Padded query rows are not blocked, because they still attend to the real keys and their outputs are simply ignored downstream.

## A later token moves an early output

Take one head of width $d = 1$ with three real tokens. Let the first query be $q_0 = 1$ and the keys be $k = (1, 0, -1)$. The scaled scores of the first row are $(1, 0, -1)$, and the softmax gives weights $(0.665, 0.245, 0.090)$. Now change only the last key to $k_2 = 3$. The first row's scores become $(1, 0, 3)$, and its weights become $(0.114, 0.042, 0.844)$. The output at position $0$ is now dominated by the last token. Under a causal mask, the first row would have stayed at weight $1$ on key $0$ in both cases, and changing $k_2$ could never have affected it.

## Common mistakes to avoid

- **Keeping the causal mask.** Copying an autoregressive attention block brings the upper triangular mask along, which hides exactly the right-hand context the mask predictor needs.
- **Masking padded query rows instead of padded key columns.** The bias must vary along the key axis, so it is broadcast as a row vector over every query row.
- **Dropping the padding bias entirely.** Left padding then receives attention weight, and the predictions of real tokens depend on arbitrary pad embeddings.
- **Scaling by $d$ instead of $\sqrt{d}$.** The standard scaled dot product divides by the square root of the head width, and the larger divisor flattens every distribution.
- **Taking the softmax over the query axis.** Each row of weights must be a distribution over keys, so the softmax runs along the last axis.
- **Inverting the mask convention.** An entry of one marks a real token, so the bias goes on the zero entries.
- **Adding a timestep input.** Eq. 11 shows the prediction is time-free, so the predictor takes nothing but the sequence.