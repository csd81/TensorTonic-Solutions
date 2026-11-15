# <span style="font-size: 20px;">Coarse-Grained MoE Feedforward Block</span>

---

## <span style="font-size: 16px;">Introduction</span>

<span style="font-size: 14px;">A Coarse-Grained Mixture-of-Experts (MoE) feedforward block is a parameter-efficient alternative to standard MoE layers where all experts share the same weight matrices but each expert operates on a different slice of the intermediate dimension. The model partitions the intermediate dimension into contiguous chunks and a learned router decides which chunks to activate per token.</span>

<span style="font-size: 14px;">In Arcee Trinity, the coarse-grained MoE block enables scaling expert count without proportional parameter increase. Each transformer layer uses this design alongside sigmoid routing and SMEBU, creating a system where expert specialization emerges from which slices of a shared weight space get activated rather than from separate weight banks.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">In a standard (fine-grained) MoE layer, each expert is a complete independent FFN with its own gate, up, and down projections. With N experts, N full copies of these matrices are stored. This gives maximum expressivity but multiplies parameter count.</span>

<span style="font-size: 14px;">A coarse-grained MoE takes a different approach. There is one set of weight matrices shared across all experts:</span>

* <span style="font-size: 14px;">**Single W_gate:** shape (d_intermediate, d_model)</span>
* <span style="font-size: 14px;">**Single W_up:** shape (d_intermediate, d_model)</span>
* <span style="font-size: 14px;">**Single W_down:** shape (d_model, d_intermediate)</span>

<span style="font-size: 14px;">The intermediate dimension d_intermediate is divided into N equal chunks. Expert i "owns" rows s_i of W_gate and W_up, and columns s_i of W_down. When the router selects expert i, only that slice is computed and contributes to the output.</span>

<span style="font-size: 14px;">Each expert is not a separate network but a view into a shared network. Activating expert i means activating a specific band of intermediate neurons. The router learns which bands are most relevant per token, and the top-K bands are combined via weighted summation.</span>

<span style="font-size: 14px;">With N experts, the total parameter count is the same as a single dense FFN. The sparsity comes from only computing K out of N slices per token, reducing FLOPs to roughly K/N of the dense cost while retaining full parameter capacity.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The forward pass involves routing, slice computation with SwiGLU, and weighted combination.</span>

<span style="font-size: 14px;">**Router logits and sigmoid gating.** Given input x of shape (d_model,), the router computes a score for each expert:</span>

$$r_i = \sigma(W_r[i,:] \cdot x + b_r[i])$$

<span style="font-size: 14px;">where W_r is of shape (N, d_model) and sigma is the sigmoid function. Each expert gets an independent probability between 0 and 1, unlike softmax routing where scores are coupled.</span>

<span style="font-size: 14px;">**Top-K selection.** From the N scores, select the K experts with the highest values:</span>

$$\mathcal{S} = \text{TopK}(\{r_1, r_2, \ldots, r_N\}, K)$$

<span style="font-size: 14px;">**Weight renormalization.** The selected expert weights are renormalized to sum to 1:</span>

$$w_i = \frac{r_i}{\sum_{j \in \mathcal{S}} r_j}, \quad i \in \mathcal{S}$$

<span style="font-size: 14px;">**SwiGLU per expert slice.** Each selected expert i operates on its slice s_i. The slice boundaries for expert i are [i * C, (i+1) * C) where C = d_intermediate / N:</span>

$$\text{Expert}_i(x) = \text{swish}(W_{\text{gate}}[s_i,:] \cdot x) \odot (W_{\text{up}}[s_i,:] \cdot x)$$

<span style="font-size: 14px;">This produces a vector of size C for each expert.</span>

<span style="font-size: 14px;">**Down projection per expert.** Each expert's intermediate output is projected back to model dimension using its slice of W_down:</span>

$$y_i = W_{\text{down}}[:, s_i] \cdot \text{Expert}_i(x)$$

<span style="font-size: 14px;">Each y_i has shape (d_model,).</span>

<span style="font-size: 14px;">**Weighted combination.** The final output is the weighted sum of selected expert outputs:</span>

$$\text{CoarseMoE}(x) = \sum_{i \in \mathcal{S}} w_i \cdot y_i$$

<span style="font-size: 14px;">The full expression in one line:</span>

$$\text{CoarseMoE}(x) = \sum_{i \in \mathcal{S}} w_i \cdot W_{\text{down}}[:,s_i] \cdot (\text{swish}(W_{\text{gate}}[s_i,:] \cdot x) \odot W_{\text{up}}[s_i,:] \cdot x)$$

---

## <span style="font-size: 16px;">Coarse vs Fine-Grained MoE</span>

<span style="font-size: 14px;">These represent two endpoints on a spectrum of parameter sharing in sparse architectures.</span>

<span style="font-size: 14px;">**Fine-grained MoE** treats each expert as a fully independent FFN:</span>

* <span style="font-size: 14px;">**Separate weights per expert:** Expert i has its own W_gate^(i), W_up^(i), W_down^(i).</span>
* <span style="font-size: 14px;">**Total parameters:** N times a single FFN (ignoring the router).</span>
* <span style="font-size: 14px;">**Memory cost:** Scales linearly with expert count.</span>
* <span style="font-size: 14px;">**Examples:** Switch Transformer, Mixtral, GShard.</span>

<span style="font-size: 14px;">**Coarse-grained MoE** shares all weight matrices and partitions the intermediate dimension:</span>

* <span style="font-size: 14px;">**Shared weights:** One W_gate, one W_up, one W_down for all experts.</span>
* <span style="font-size: 14px;">**Total parameters:** Same as a single dense FFN. No parameter increase from adding experts.</span>
* <span style="font-size: 14px;">**Memory cost:** Constant regardless of expert count. Only the small router adds parameters.</span>
* <span style="font-size: 14px;">**Examples:** Arcee Trinity.</span>

<span style="font-size: 14px;">The tradeoff: fine-grained gives more capacity per expert but costs more parameters and memory. Coarse-grained gives computational sparsity (only K/N of the FFN computed) without parameter overhead -- a learned form of conditional computation routing tokens to the most relevant portion of a shared layer.</span>

<span style="font-size: 14px;">A practical advantage: coarse-grained MoE avoids expert parallelism challenges. Since all experts share weights, standard tensor parallelism works without sharding experts across devices.</span>

---

## <span style="font-size: 16px;">SwiGLU Within Each Expert</span>

<span style="font-size: 14px;">Each expert slice uses SwiGLU as its activation function rather than plain ReLU or GELU.</span>

<span style="font-size: 14px;">**Standard FFN** uses a single projection and pointwise activation:</span>

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

<span style="font-size: 14px;">**SwiGLU** replaces this with a gated linear unit using swish (SiLU):</span>

$$\text{SwiGLU}(x) = (\text{swish}(W_{\text{gate}} \cdot x)) \odot (W_{\text{up}} \cdot x)$$

<span style="font-size: 14px;">where swish(z) = z * sigmoid(z). SwiGLU uses two linear projections -- one gated through swish, the other left linear -- multiplied elementwise. This gating lets the network learn which intermediate dimensions to suppress or amplify.</span>

<span style="font-size: 14px;">Within a coarse-grained expert slice:</span>

$$\text{Expert}_i(x) = \text{swish}(W_{\text{gate}}[s_i,:] \cdot x) \odot (W_{\text{up}}[s_i,:] \cdot x)$$

<span style="font-size: 14px;">This gives a vector of size C = d_intermediate / N. The down projection maps it back:</span>

$$y_i = W_{\text{down}}[:,s_i] \cdot \text{Expert}_i(x)$$

<span style="font-size: 14px;">**Why SwiGLU over ReLU?**</span>

* <span style="font-size: 14px;">**Smoother gradients:** Unlike ReLU with its hard zero below threshold, swish provides gradients everywhere, reducing dead neuron problems.</span>
* <span style="font-size: 14px;">**Gating expressivity:** The elementwise product of two projections lets the model learn complex feature interactions in intermediate space.</span>
* <span style="font-size: 14px;">**Empirical gains:** SwiGLU consistently outperforms ReLU and GELU FFNs in language modeling, as shown in PaLM and LLaMA.</span>
* <span style="font-size: 14px;">**Clean slicing:** SwiGLU operates independently across the intermediate dimension, so slicing into expert chunks is mathematically clean -- each slice computes the same function on fewer dimensions.</span>

<span style="font-size: 14px;">The gate projection feeds through swish and controls information flow; the up projection provides the information being gated. Swapping them produces incorrect results with no shape error.</span>

---

## <span style="font-size: 16px;">Paper Context: Arcee Trinity</span>

<span style="font-size: 14px;">Arcee Trinity combines coarse-grained MoE with sigmoid routing and SMEBU (Sparse Mixed Expert Balancing with Uniform sampling).</span>

<span style="font-size: 14px;">**Architecture specifics:**</span>

* <span style="font-size: 14px;">**Number of experts:** The intermediate dimension is divided into multiple chunks, each acting as one expert.</span>
* <span style="font-size: 14px;">**Top-K routing:** Only the top K experts (by router score) are activated per token.</span>
* <span style="font-size: 14px;">**Sigmoid routing:** Each expert's score passes through an independent sigmoid, not softmax. Activating one expert does not suppress another. After top-K selection, chosen scores are renormalized to sum to 1.</span>
* <span style="font-size: 14px;">**SMEBU for load balancing:** Prevents expert collapse via uniform sampling during training to ensure all experts receive gradients, combined with a balancing loss penalizing uneven utilization.</span>

<span style="font-size: 14px;">**Why coarse-grained MoE fits Trinity's goals:**</span>

* <span style="font-size: 14px;">**Parameter efficiency:** Computational sparsity (faster inference) without multiplying parameter count.</span>
* <span style="font-size: 14px;">**Simpler sharding:** Shared matrices mean standard tensor parallelism works without expert-parallel strategies.</span>
* <span style="font-size: 14px;">**Complementary with sigmoid routing:** Independent scoring aligns with the idea that different slices are independently useful, with no competition imposed by softmax.</span>

<span style="font-size: 14px;">At inference, each token computes only K slices of the SwiGLU FFN, achieving speedup over the dense equivalent while maintaining full capacity.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider d_model = 4, d_intermediate = 8, N = 4 experts, K = 2 (top-2). Each expert gets a slice of size C = 8 / 4 = 2.</span>

<span style="font-size: 14px;">**Slice assignments:**</span>

* <span style="font-size: 14px;">**Expert 0:** rows [0, 1], **Expert 1:** rows [2, 3], **Expert 2:** rows [4, 5], **Expert 3:** rows [6, 7]</span>

<span style="font-size: 14px;">**Input:** x = [1.0, 0.5, -0.5, 0.2]</span>

<span style="font-size: 14px;">**Step 1: Router scores.** Raw logits W_r * x + b_r = [0.8, -0.3, 1.2, 0.1]. Applying sigmoid:</span>

* <span style="font-size: 14px;">**r_0** = sigmoid(0.8) = 0.690, **r_1** = sigmoid(-0.3) = 0.426</span>
* <span style="font-size: 14px;">**r_2** = sigmoid(1.2) = 0.769, **r_3** = sigmoid(0.1) = 0.525</span>

<span style="font-size: 14px;">**Step 2: Top-2 selection.** Highest scores: r_2 = 0.769 and r_0 = 0.690. Activate experts 0 and 2.</span>

<span style="font-size: 14px;">**Step 3: Renormalize.** Sum = 0.769 + 0.690 = 1.459. So w_0 = 0.690 / 1.459 = 0.473, w_2 = 0.769 / 1.459 = 0.527.</span>

<span style="font-size: 14px;">**Step 4: Expert 0 SwiGLU.** Uses rows [0, 1] of W_gate and W_up:</span>

* <span style="font-size: 14px;">**W_gate[0:2] * x** = [0.6, -0.4], **W_up[0:2] * x** = [0.3, 0.8]</span>
* <span style="font-size: 14px;">**swish([0.6, -0.4])** = [0.6 * 0.646, -0.4 * 0.401] = [0.388, -0.160]</span>
* <span style="font-size: 14px;">**SwiGLU output** = [0.388 * 0.3, -0.160 * 0.8] = [0.116, -0.128]</span>

<span style="font-size: 14px;">**Step 5: Expert 0 down projection.** Using columns [0, 1] of W_down:</span>

$$y_0 = W_{\text{down}}[:,0:2] \cdot [0.116, -0.128] = [0.05, -0.03, 0.08, -0.01]$$

<span style="font-size: 14px;">**Step 6: Expert 2 SwiGLU.** Uses rows [4, 5]:</span>

* <span style="font-size: 14px;">**W_gate[4:6] * x** = [1.1, 0.3], **W_up[4:6] * x** = [-0.2, 0.5]</span>
* <span style="font-size: 14px;">**swish([1.1, 0.3])** = [1.1 * 0.750, 0.3 * 0.574] = [0.825, 0.172]</span>
* <span style="font-size: 14px;">**SwiGLU output** = [0.825 * -0.2, 0.172 * 0.5] = [-0.165, 0.086]</span>

<span style="font-size: 14px;">**Step 7: Expert 2 down projection.** Using columns [4, 5] of W_down:</span>

$$y_2 = W_{\text{down}}[:,4:6] \cdot [-0.165, 0.086] = [-0.07, 0.04, -0.02, 0.06]$$

<span style="font-size: 14px;">**Step 8: Weighted combination.**</span>

$$\text{output} = 0.473 \cdot [0.05, -0.03, 0.08, -0.01] + 0.527 \cdot [-0.07, 0.04, -0.02, 0.06]$$

$$= [0.024, -0.014, 0.038, -0.005] + [-0.037, 0.021, -0.011, 0.032]$$

$$= [-0.013, 0.007, 0.027, 0.027]$$

<span style="font-size: 14px;">The output has shape (d_model = 4). Only 2 of 4 slices were computed, using 4 out of 8 intermediate neurons -- a 50% FLOP reduction versus the dense FFN.</span>

---

## <span style="font-size: 16px;">Variants and Modern Context</span>

<span style="font-size: 14px;">Coarse-grained MoE sits within a broader landscape of sparse architectures.</span>

<span style="font-size: 14px;">**Switch Transformer (2021):** Fine-grained experts with top-1 softmax routing. Each expert is a full independent FFN. Maximizes sparsity but suffers from expert imbalance.</span>

<span style="font-size: 14px;">**GShard (2020):** Scaled MoE to 600B parameters with top-2 fine-grained routing and expert capacity factors for load balancing.</span>

<span style="font-size: 14px;">**Mixtral (2023):** 8 fine-grained SwiGLU experts with top-2 routing. 46.7B total parameters, ~12.9B active per token. Matches dense models at equivalent compute but needs more memory.</span>

<span style="font-size: 14px;">**DeepSeek-MoE (2024):** Hybrid with many small fine-grained experts plus a shared expert that always activates. Philosophically closer to coarse-grained in acknowledging shared computation's value.</span>

<span style="font-size: 14px;">**Coarse-grained MoE (Arcee Trinity):** Pushes sharing to its extreme -- all experts share weights, differentiated only by slice ownership. Maximizes parameter efficiency at the cost of fully independent expert representations.</span>

<span style="font-size: 14px;">The trend is toward exploring the continuum between shared and independent experts. Coarse-grained MoE is a clean point on this spectrum prioritizing efficiency.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">Several details are easy to get wrong when implementing coarse-grained MoE.</span>

* <span style="font-size: 14px;">**Wrong slice indices:** Expert i uses rows [i*C : (i+1)*C] of W_gate and W_up, but columns [i*C : (i+1)*C] of W_down. W_down's slicing is along the second axis (columns), not the first, because it maps from intermediate back to model dimension.</span>

* <span style="font-size: 14px;">**Forgetting to renormalize router weights:** With sigmoid routing, selected weights will not sum to 1 unless explicitly renormalized. Sigmoid scores are independent, so skipping renormalization causes output magnitude to vary with raw scores, leading to instability.</span>

* <span style="font-size: 14px;">**Gate vs up confusion in SwiGLU:** The formula is swish(W_gate * x) * (W_up * x). Swapping them changes the function silently -- both have the same shape so no error is raised.</span>

* <span style="font-size: 14px;">**Dimension mismatch when recombining:** Each expert's C-dimensional output is projected to d_model via W_down, then summed. Outputs are combined in d_model space, not concatenated in intermediate space. Concatenating into a K*C vector and projecting is incorrect.</span>

* <span style="font-size: 14px;">**Not handling zero-weight experts:** Unselected experts should not be computed. If using a dense mask, ensure it zeros out both intermediate computation and weight.</span>

* <span style="font-size: 14px;">**Confusing parameter count with compute cost:** Coarse-grained MoE has the same parameters as a dense FFN but lower per-token FLOPs. It reduces compute, not parameters.</span>

* <span style="font-size: 14px;">**Uneven slice sizes:** If d_intermediate is not divisible by N, some experts get more neurons, biasing the router toward them. Always choose d_intermediate divisible by N.</span>

---