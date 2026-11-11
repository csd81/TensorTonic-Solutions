## Multi-Head Attention

Multi-head attention is the core mechanism of the Transformer architecture. Instead of performing a single attention function, it runs multiple attention operations in parallel across different learned subspaces.

### Why Multiple Heads?

A single attention head can only focus on one type of relationship at a time. Multiple heads allow the model to:

- Attend to different positions simultaneously
- Capture different types of relationships (syntactic, semantic, positional)
- Learn complementary representations in parallel subspaces

### The Algorithm

Given input queries $Q$, keys $K$, and values $V$, each of shape $(B, S, d_{model})$:

**Step 1: Linear Projections**

Project each input through learned weight matrices:

$$
Q' = Q W_q
$$

$$
K' = K W_k
$$

$$
V' = V W_v
$$

where $W_q, W_k, W_v \in \mathbb{R}^{d_{model} \times d_{model}}$.

**Step 2: Split into Heads**

Reshape each projected tensor from $(B, S, d_{model})$ to $(B, h, S, d_k)$ where $d_k = d_{model} / h$. Each head operates on a $d_k$-dimensional slice.

**Step 3: Scaled Dot-Product Attention**

For each head $i$:

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

- $Q_i K_i^T$ computes pairwise similarity scores
- Division by $\sqrt{d_k}$ prevents large dot products that push softmax into saturated regions
- Softmax normalizes scores to attention weights that sum to 1
- Multiplication by $V_i$ produces a weighted combination of values

**Step 4: Concatenate and Project**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_o
$$

The heads are concatenated back to $d_{model}$ dimensions, then multiplied by $W_o \in \mathbb{R}^{d_{model} \times d_{model}}$ to mix information across heads.

### Scaling Factor Intuition

If the entries of $Q_i$ and $K_i$ are independent random variables with mean 0 and variance 1, then $Q_i K_i^T$ has variance $d_k$. Dividing by $\sqrt{d_k}$ restores unit variance, keeping softmax in a region where gradients are well-behaved.

### Computational Complexity

- The four projections cost $O(B \cdot S \cdot d_{model}^2)$ each
- Attention per head costs $O(B \cdot S^2 \cdot d_k)$, summed over $h$ heads gives $O(B \cdot S^2 \cdot d_{model})$
- Total complexity is $O(B \cdot S \cdot d_{model}^2 + B \cdot S^2 \cdot d_{model})$

### Implementation Notes

- The reshape from $(B, S, d_{model})$ to $(B, h, S, d_k)$ can use reshape and transpose, or equivalent per-head slicing
- After attention, call contiguous() before view() since transpose creates non-contiguous memory
- $d_{model}$ must be divisible by $h$ so that $d_k$ is an integer