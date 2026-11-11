## Attention Mechanisms in Deep Learning

### Motivation

Traditional sequence models like RNNs process tokens sequentially, creating a bottleneck when capturing long-range dependencies. The attention mechanism addresses this by allowing each position in a sequence to directly attend to every other position, regardless of distance.

Key motivations:

* Sequential processing in RNNs limits parallelization and creates information bottlenecks
* Fixed-length context vectors in encoder-decoder models lose information for long sequences
* Attention provides $O(1)$ path length between any two positions, enabling better gradient flow

### Query, Key, and Value Abstraction

The attention mechanism is built on a retrieval analogy:

* **Query** ($Q$): what information the current position is looking for
* **Key** ($K$): what information each position advertises about itself
* **Value** ($V$): the actual content each position provides when attended to

In self-attention, $Q$, $K$, and $V$ are all derived from the same input sequence via learned linear projections. In cross-attention, $Q$ comes from one sequence while $K$ and $V$ come from another.

### Scaled Dot-Product Attention

The core computation involves three steps:

* Compute raw alignment scores via the dot product $Q K^T$
* Scale by $\frac{1}{\sqrt{d_k}}$ to prevent softmax saturation
* Apply softmax to obtain normalized attention weights, then multiply by $V$

The full formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

### Why Scaling Matters

Without scaling, the dot product magnitudes grow proportionally to $d_k$. Consider two random vectors with entries drawn from a standard normal distribution:

* Their dot product has mean 0 and variance $d_k$
* For large $d_k$, some dot products become very large in absolute value
* Softmax maps large inputs to near-zero or near-one outputs, creating vanishing gradients
* Dividing by $\sqrt{d_k}$ restores the variance to approximately 1

### Softmax as a Soft Argmax

The softmax function converts raw scores into a probability distribution:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

* All output values are positive and sum to 1
* It acts as a differentiable approximation to argmax
* Higher scores receive exponentially more weight, creating a "soft" selection
* Applied along the key dimension so each query produces a valid distribution over keys

### Multi-Head Attention

In practice, a single attention function is extended to multiple heads:

* The model projects $Q$, $K$, $V$ into $h$ different subspaces
* Each head performs attention independently with dimension $d_k / h$
* Outputs are concatenated and projected back to the model dimension
* This allows the model to jointly attend to information from different representation subspaces

### Computational Complexity

For sequence length $n$ and dimension $d$:

* Computing $Q K^T$ requires $O(n^2 d)$ operations
* Storing the attention weight matrix requires $O(n^2)$ memory per head
* This quadratic cost is the primary limitation of standard attention
* Techniques like Flash Attention, sparse attention, and linear attention aim to reduce this cost

### Attention in the Transformer

The Transformer architecture uses attention in three distinct ways:

* **Encoder self-attention**: each token attends to all tokens in the input sequence
* **Decoder self-attention**: each token attends to previous tokens only (causal masking)
* **Cross-attention**: decoder tokens attend to encoder outputs

Combined with residual connections, layer normalization, and feed-forward networks, attention forms the backbone of modern language models, vision transformers, and multimodal systems.