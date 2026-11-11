## Beam Search Decoding

### Overview

Beam search is a breadth-limited search algorithm widely used in sequence generation tasks. It balances the trade-off between exhaustive search (which is computationally infeasible for large vocabularies) and greedy search (which may miss globally optimal sequences).

### Key Concepts

* **Beam**: a partial sequence of tokens paired with its cumulative log-probability score
* **Beam width** ($k$): the number of top-scoring candidates retained at each step
* **Log-probability scoring**: sequences are scored by summing the log-probabilities of individual token predictions, which is equivalent to maximizing the joint probability

$$
\text{score}(\mathbf{y}) = \sum_{t=1}^{T} \log P(y_t \mid y_1, \ldots, y_{t-1})
$$

### Algorithm Steps

* Initialize with a single beam containing the start token and score 0
* At each time step:
    * For each active beam, compute log-probabilities over the full vocabulary
    * Generate all possible one-token extensions, updating cumulative scores
    * Retain only the top $k$ candidates
    * Move any beam ending with the end token to a completed list
* Terminate when all beams are complete or max length is reached
* Return the highest-scoring sequence from both completed and active beams

### Greedy vs. Beam Search

* Greedy decoding ($k=1$) selects the single most probable token at each step
* Greedy can get trapped in locally optimal but globally suboptimal paths
* Beam search with $k > 1$ explores multiple hypotheses in parallel, often finding better overall sequences
* Increasing $k$ improves quality but increases computation linearly

### Scoring and Comparison

* Log-probabilities are used instead of raw probabilities to avoid numerical underflow from multiplying many small numbers
* Since $\log$ is monotonic, maximizing the sum of log-probabilities is equivalent to maximizing the product of probabilities

$$
\arg\max_{\mathbf{y}} \prod_{t} P(y_t \mid y_{<t}) = \arg\max_{\mathbf{y}} \sum_{t} \log P(y_t \mid y_{<t})
$$

### Handling End Tokens

* When a beam generates the end-of-sequence token, it is considered complete
* Completed beams are stored separately and not expanded further
* This allows shorter sequences to compete fairly with longer ones
* The end token itself is excluded from the final output

### Practical Considerations

* Typical beam widths in practice range from 2 to 10
* Very large beam widths show diminishing returns and may even degrade output quality in neural language models
* Length normalization can be applied to prevent bias toward shorter sequences, though the basic algorithm does not include it
* Beam search is deterministic given the same scoring function, unlike sampling-based methods

### Complexity

* Time complexity per step: $O(k \cdot V)$ where $V$ is vocabulary size
* Total time: $O(T \cdot k \cdot V)$ where $T$ is the maximum number of generated tokens, excluding the start token
* Space: $O(k \cdot T)$ to store the active beams