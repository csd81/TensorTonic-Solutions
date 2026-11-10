# Multi-Head Attention

Scaled dot-product attention gives each token one way to decide what matters. Multi-head attention asks a natural follow-up: why force every relationship to be judged through the same representation?

Consider the sentence “The animal did not cross the road because it was tired.” To understand “it,” one useful relationship is grammatical: “it” refers to “animal.” Another is semantic: “tired” explains the animal’s state. A single attention calculation can mix these signals, but several attention heads give the model separate learned spaces in which to look for them.

A head is simply one scaled dot-product attention calculation over its own projected features. Multi-head attention runs several heads in parallel, joins their results, and learns how to mix them.

## What a head actually represents

The input token vectors do not arrive with features labeled “grammar,” “identity,” or “distance.” The model learns projection matrices that reorganize the input into useful query, key, and value features.

With several heads, different slices of those projected features are processed separately. One head may learn comparisons that are useful for nearby phrases, while another may become sensitive to longer-range references. These interpretations are examples rather than fixed rules. No head is manually assigned a linguistic job.

The important point is that each head gets its own query, key, and value feature group. It can therefore produce a different attention pattern for the same sentence.

## From hidden states to several attention problems

Let $X$ contain the hidden state of every token. Three learned projections create

$$
Q=XW_q,
\qquad
K=XW_k,
\qquad
V=XW_v
$$

Each result still has the model width $d_{\mathrm{model}}$. If there are $h$ heads, that width is divided evenly, giving each head

$$
d_k = \frac{d_{\mathrm{model}}}{h}
$$

For example, a model width of 8 with 4 heads gives each head 2 features. The model has not created four full-width copies. It has divided the projected width into four smaller views, so the total width remains 8.

This explains the divisibility requirement. If the model width cannot be divided evenly by the number of heads, there is no consistent way to form equal-sized head features.

## What happens inside each head

After splitting, head $i$ receives $Q_i$, $K_i$, and $V_i$. It performs the scaled dot-product operation from the previous problem:

$$
H_i
=
\operatorname{softmax}\left(
\frac{Q_iK_i^{\mathsf T}}{\sqrt{d_k}} + M
\right)V_i
$$

Nothing about the attention rule has changed. Each query is compared with keys, the scores are scaled and optionally masked, softmax produces weights, and those weights combine values.

The heads are independent during this calculation. Head 1 does not softmax together with head 2, and a query in one head does not use values from another head. Their information meets only after every head has produced an output.

## Why concatenation is followed by another projection

Each head returns a vector of width $d_k$. Putting the $h$ head outputs side by side restores the original model width:

$$
H = \operatorname{Concat}(H_1,\ldots,H_h)
$$

Concatenation preserves what every head found. Averaging the heads would collapse those separate feature groups and would return only $d_k$ features.

The concatenated vector is then multiplied by the output matrix:

$$
O = HW_o
$$

Before this projection, each section of the vector came from one head. The output projection can combine information across heads and place it back into the representation expected by the rest of the model. Omitting $W_o$ leaves the pieces adjacent but never learns how they should interact.

The full mechanism is therefore

$$
\operatorname{MHA}(X)
=
\operatorname{Concat}(H_1,\ldots,H_h)W_o
$$

## A concrete two-head picture

Suppose each token has a model width of 4 and there are 2 heads. After the query projection, a token might have query features

$$
[0.8,-0.2\;|\;0.1,0.9]
$$

The vertical separator is only for explanation. Head 1 receives the first two numbers, and head 2 receives the last two. Keys and values are divided in the same way.

The first head compares its two query features with the first two key features of every token. The second head makes a separate set of comparisons using the other two features. Each head returns two output features. Concatenating those two results gives four features again, and $W_o$ mixes them.

This small picture is more useful than memorizing a long list of shapes. Splitting creates independent views; concatenation restores the width; output projection mixes the views.

## Causal attention

When causal mode is enabled, a token may use itself and earlier tokens but not later ones. The causal mask is applied to the score matrix inside every head before softmax.

The same position rule is shared by all heads, but the permitted scores can still differ because each head has different projected queries and keys. Position 3 may strongly prefer position 1 in one head and position 2 in another.

When causal mode is disabled, every position may contribute to every output position. The implementation must support both paths rather than always applying a triangular mask.

## The one-head case

Setting $h=1$ is a useful way to test your understanding. The one head receives the entire model width, so there is no meaningful split. The function becomes ordinary scaled dot-product self-attention followed by the supplied output projection.

This boundary tells you that multi-head attention is not a different scoring rule. It is a structured way to run the same scoring rule across several learned subspaces.

## Following the tensors without drowning in them

Only one internal layout is essential for this implementation. After projection, arrange queries, keys, and values as

$$
(B,h,S,d_k)
$$

Here $B$ is batch size and $S$ is sequence length. For each batch item and head, the final two dimensions describe one ordinary attention problem over $S$ tokens with feature width $d_k$.

The attention scores then have one row and column per sequence position, while each head output returns to width $d_k$. After transposing the head outputs back beside each token, concatenation produces $(B,S,d_{\mathrm{model}})$, which is also the final output shape after $W_o$.

## Cost and what multiple heads do not change

Every head forms a sequence-by-sequence score matrix. Across all heads, the attention work is proportional to $BS^2d_{\mathrm{model}}$, and the stored scores are proportional to $BhS^2$.

Increasing the number of heads does not multiply the projected model width because each head becomes narrower. It also does not remove the quadratic dependence on sequence length. The purpose of the heads is representational: they let attention examine several learned feature spaces at once.

## Common mistakes to avoid

- Reshaping without moving the head axis can mix features from different tokens.
- Scaling by the full model width is wrong; each head uses $\sqrt{d_k}$.
- Softmax must normalize over key positions separately for every head and query.
- The causal mask belongs before softmax.
- Head outputs are concatenated, not averaged or summed.
- The output projection is part of the required operation.

A clear implementation follows the concept: project once, expose the head axis, run the familiar attention calculation independently, join the results, and mix them with $W_o$.
