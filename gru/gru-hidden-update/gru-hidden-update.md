# <span style="font-size: 20px;">Hidden State Update</span>

<span style="font-size: 14px;">The hidden state update is the final step in a GRU cell, where the new hidden state $h_t$ is formed by linearly interpolating between the previous hidden state $h_{t-1}$ and the candidate activation $\tilde{h}_t$. Described in Cho et al. (2014), this formula is what allows a GRU to selectively remember or overwrite information at every time step.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">After the reset gate has produced a candidate activation $\tilde{h}_t$ and the update gate has produced a blending coefficient $z_t$, the GRU must combine the old hidden state with the new candidate into a single output. The hidden state update performs this combination through **linear interpolation**: a weighted average where the weights at each dimension sum to exactly one.</span>

<span style="font-size: 14px;">The result $h_t$ serves a dual role. It is the **output** of the GRU cell at time step $t$ (unlike the LSTM, which has separate cell state and hidden state), and it is also the **memory** carried forward to time step $t+1$. The interpolation formula simultaneously decides what to output and what to remember.</span>

<span style="font-size: 14px;">No additional parameters are needed for the interpolation itself. The update gate $z_t$ was already computed using learned weight matrices. The hidden state update is purely a pointwise arithmetic operation: multiply, subtract, multiply, add. Yet this simple formula is how the GRU learns when to maintain long-term memory and when to incorporate new information.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">The Interpolation Formula</span>

$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$h_t \in \mathbb{R}^H$ is the **new hidden state** (output of this time step)</span>
* <span style="font-size: 14px;">$z_t \in (0, 1)^H$ is the **update gate** output (computed earlier in the pipeline)</span>
* <span style="font-size: 14px;">$h_{t-1} \in \mathbb{R}^H$ is the **previous hidden state** (memory from the last time step)</span>
* <span style="font-size: 14px;">$\tilde{h}_t \in (-1, 1)^H$ is the **candidate hidden state** (new information via the reset gate)</span>
* <span style="font-size: 14px;">$\odot$ denotes **element-wise (Hadamard) multiplication**, not matrix multiplication</span>

### <span style="font-size: 14px;">Element-wise Form</span>

<span style="font-size: 14px;">For each hidden dimension $k$ (where $k = 1, 2, \ldots, H$):</span>

$$
h_t^{(k)} = z_t^{(k)} \cdot h_{t-1}^{(k)} + (1 - z_t^{(k)}) \cdot \tilde{h}_t^{(k)}
$$

<span style="font-size: 14px;">Each dimension operates independently. Dimension $k$ has its own gate value $z_t^{(k)}$ controlling how much old value to retain versus how much new candidate to accept.</span>

### <span style="font-size: 14px;">The Complementary Constraint</span>

$$
z_t^{(k)} + (1 - z_t^{(k)}) = 1 \quad \forall \, k
$$

<span style="font-size: 14px;">Because $z_t$ comes from a sigmoid with range $(0, 1)$, both $z_t$ and $(1 - z_t)$ are strictly positive and sum to one. The output $h_t$ is a **convex combination** of $h_{t-1}$ and $\tilde{h}_t$, always lying on the line segment between them in each dimension.</span>

---

## <span style="font-size: 16px;">Why Linear Interpolation</span>

### <span style="font-size: 14px;">Smooth Blending Between Old and New</span>

<span style="font-size: 14px;">Linear interpolation provides a continuous spectrum between fully keeping the old state ($z = 1$) and fully replacing with the new candidate ($z = 0$). Any value in between produces a smooth blend. During training, gradients can nudge $z_t$ incrementally, allowing the network to learn fine-grained memory policies rather than hard binary switches.</span>

### <span style="font-size: 14px;">Independent Control Per Dimension</span>

<span style="font-size: 14px;">Each hidden dimension has its own gate value. In a hidden state of size $H = 256$, there are 256 independent interpolation decisions at every time step. Dimension 12 might keep 95% of its old value (tracking sentence topic), while dimension 47 replaces 80% (adapting to the current word). This per-dimension independence allows a single hidden state vector to maintain multiple timescales of information.</span>

### <span style="font-size: 14px;">Gradient Flow Properties</span>

<span style="font-size: 14px;">The $z_t \odot h_{t-1}$ term creates a **direct linear path** from $h_{t-1}$ to $h_t$. When $z_t \approx 1$, the gradient $\partial h_t / \partial h_{t-1} \approx 1$ along that dimension. Dimensions that need to preserve information over many time steps learn to keep $z_t$ close to 1, creating a gradient highway analogous to skip connections in residual networks.</span>

---

## <span style="font-size: 16px;">Per-Dimension Independence</span>

<span style="font-size: 14px;">The hidden state update is not a single global decision but $H$ independent local decisions. Consider a GRU with $H = 4$:</span>

* <span style="font-size: 14px;">$z_t = [0.95, \; 0.10, \; 0.50, \; 0.02]$</span>

<span style="font-size: 14px;">This means:</span>

* <span style="font-size: 14px;">**Dimension 0** ($z = 0.95$): Keep 95% old, take 5% new. Memory mode.</span>
* <span style="font-size: 14px;">**Dimension 1** ($z = 0.10$): Keep 10% old, take 90% new. Update mode.</span>
* <span style="font-size: 14px;">**Dimension 2** ($z = 0.50$): Equal blend. Smooth transition.</span>
* <span style="font-size: 14px;">**Dimension 3** ($z = 0.02$): Near-complete replacement, effectively resetting.</span>

<span style="font-size: 14px;">All four behaviors happen simultaneously. The network learns which dimensions serve as long-term memory channels and which should respond to new inputs. If the update gate produced a single scalar for the entire hidden state, all dimensions would update at the same rate, severely limiting expressiveness.</span>

---

## <span style="font-size: 16px;">The Complete GRU Pipeline</span>

### <span style="font-size: 14px;">Step 1: Update Gate</span>

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

<span style="font-size: 14px;">Decides per dimension how much of the old state to retain. Output range: $(0, 1)^H$.</span>

### <span style="font-size: 14px;">Step 2: Reset Gate</span>

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

<span style="font-size: 14px;">Decides how much of the previous hidden state should influence the candidate. When $r_t \approx 0$, the candidate ignores previous state entirely.</span>

### <span style="font-size: 14px;">Step 3: Candidate Hidden State</span>

$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$

<span style="font-size: 14px;">The "proposed new content." The reset gate modulates how much $h_{t-1}$ participates. Tanh squashes results to $(-1, 1)$.</span>

### <span style="font-size: 14px;">Step 4: Hidden State Update (This Problem)</span>

$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
$$

<span style="font-size: 14px;">Blends old and new. The **reset gate** controls what goes into the candidate (Step 3), while the **update gate** controls how much of the candidate reaches the final state (Step 4).</span>

---

## <span style="font-size: 16px;">Paper Context</span>

### <span style="font-size: 14px;">Cho et al. (2014)</span>

<span style="font-size: 14px;">In "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," the hidden state update appears as Equation 5. The authors state: "The actual activation of the proposed unit $h_j^t$ is then a linear interpolation between the previous activation $h_j^{t-1}$ and the candidate activation $\tilde{h}_j^t$."</span>

### <span style="font-size: 14px;">Leaky Integration</span>

<span style="font-size: 14px;">The hidden state update is a form of **leaky integration** from signal processing and neuroscience. In a leaky integrator, the current state is a weighted sum of the previous state (decayed by a leak factor) and new input. The GRU's $z_t$ plays the role of the leak rate, but unlike classical leaky integrators where the rate is fixed, the GRU learns it dynamically as a function of input and state.</span>

### <span style="font-size: 14px;">Enabling Long-Term Memory</span>

<span style="font-size: 14px;">Standard RNNs compute $h_t = \tanh(W x_t + U h_{t-1})$, completely overwriting the hidden state at each step. The GRU's interpolation formula provides a selective preservation mechanism: by setting $z_t$ close to 1, the network copies information forward across many time steps with minimal degradation, analogous to skip connections in residual networks.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

### <span style="font-size: 14px;">Setup</span>

<span style="font-size: 14px;">GRU with hidden size $H = 4$, single sample ($N = 1$):</span>

$$
z_t = [0.8, \; 0.2, \; 0.6, \; 0.1]
$$

$$
h_{t-1} = [1.0, \; -0.5, \; 0.3, \; 2.0]
$$

$$
\tilde{h}_t = [-0.4, \; 0.9, \; 0.7, \; -1.0]
$$

### <span style="font-size: 14px;">Step 1: Compute $z_t \odot h_{t-1}$</span>

* <span style="font-size: 14px;">Dim 0: $0.8 \times 1.0 = 0.80$</span>
* <span style="font-size: 14px;">Dim 1: $0.2 \times (-0.5) = -0.10$</span>
* <span style="font-size: 14px;">Dim 2: $0.6 \times 0.3 = 0.18$</span>
* <span style="font-size: 14px;">Dim 3: $0.1 \times 2.0 = 0.20$</span>

$$
z_t \odot h_{t-1} = [0.80, \; -0.10, \; 0.18, \; 0.20]
$$

### <span style="font-size: 14px;">Step 2: Compute $(1 - z_t)$</span>

$$
(1 - z_t) = [0.2, \; 0.8, \; 0.4, \; 0.9]
$$

### <span style="font-size: 14px;">Step 3: Compute $(1 - z_t) \odot \tilde{h}_t$</span>

* <span style="font-size: 14px;">Dim 0: $0.2 \times (-0.4) = -0.08$</span>
* <span style="font-size: 14px;">Dim 1: $0.8 \times 0.9 = 0.72$</span>
* <span style="font-size: 14px;">Dim 2: $0.4 \times 0.7 = 0.28$</span>
* <span style="font-size: 14px;">Dim 3: $0.9 \times (-1.0) = -0.90$</span>

$$
(1 - z_t) \odot \tilde{h}_t = [-0.08, \; 0.72, \; 0.28, \; -0.90]
$$

### <span style="font-size: 14px;">Step 4: Add Both Terms</span>

* <span style="font-size: 14px;">Dim 0: $0.80 + (-0.08) = 0.72$</span>
* <span style="font-size: 14px;">Dim 1: $-0.10 + 0.72 = 0.62$</span>
* <span style="font-size: 14px;">Dim 2: $0.18 + 0.28 = 0.46$</span>
* <span style="font-size: 14px;">Dim 3: $0.20 + (-0.90) = -0.70$</span>

$$
h_t = [0.72, \; 0.62, \; 0.46, \; -0.70]
$$

### <span style="font-size: 14px;">Interpreting the Results</span>

* <span style="font-size: 14px;">**Dim 0** ($z = 0.8$): Old $= 1.0$, candidate $= -0.4$, result $= 0.72$. High $z$ kept most of the old value.</span>
* <span style="font-size: 14px;">**Dim 1** ($z = 0.2$): Old $= -0.5$, candidate $= 0.9$, result $= 0.62$. Low $z$ swung heavily toward the candidate.</span>
* <span style="font-size: 14px;">**Dim 2** ($z = 0.6$): Old $= 0.3$, candidate $= 0.7$, result $= 0.46$. Moderate $z$ blended evenly.</span>
* <span style="font-size: 14px;">**Dim 3** ($z = 0.1$): Old $= 2.0$, candidate $= -1.0$, result $= -0.70$. Very low $z$ caused near-complete replacement.</span>

---

## <span style="font-size: 16px;">Connection to LSTM</span>

### <span style="font-size: 14px;">LSTM Cell State Update</span>

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

<span style="font-size: 14px;">The LSTM uses two separate gates: $f_t$ (forget gate) and $i_t$ (input gate), computed by separate weight matrices. $f_t + i_t$ is not constrained to equal 1.</span>

### <span style="font-size: 14px;">GRU's Complementary Design</span>

$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
$$

<span style="font-size: 14px;">The GRU uses a single gate $z_t$ to replace both. $z_t$ acts as the forget gate and $(1 - z_t)$ acts as the input gate, linked by the complementary constraint.</span>

### <span style="font-size: 14px;">Practical Differences</span>

* <span style="font-size: 14px;">**LSTM**: Can have $f_t = 0.9$ and $i_t = 0.9$ simultaneously, keeping 90% old AND adding 90% new. Cell state magnitude can grow.</span>
* <span style="font-size: 14px;">**GRU**: If $z_t = 0.9$, then $(1 - z_t) = 0.1$. Keeping and replacing are a **zero-sum game**. Hidden state stays bounded.</span>
* <span style="font-size: 14px;">**LSTM**: Has an output gate $o_t$ filtering cell state: $h_t = o_t \odot \tanh(c_t)$. GRU exposes $h_t$ directly.</span>

<span style="font-size: 14px;">The GRU trades flexibility (independent gates, separate cell/hidden states) for simplicity: 3 gate matrices instead of 4. Cho et al. (2014) and Chung et al. (2014) showed this tradeoff is often favorable, especially with limited training data.</span>

---

## <span style="font-size: 16px;">Gradient Flow</span>

### <span style="font-size: 14px;">Derivative Through the Interpolation</span>

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(z_t) + \text{diag}(1 - z_t) \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}} + \text{terms from } \frac{\partial z_t}{\partial h_{t-1}}
$$

<span style="font-size: 14px;">The first term, $\text{diag}(z_t)$, provides a **direct additive path** from $h_{t-1}$ to $h_t$. When $z_t^{(k)} \approx 1$, the gradient for dimension $k$ passes through with near-unit magnitude.</span>

### <span style="font-size: 14px;">The Gradient Highway</span>

<span style="font-size: 14px;">For a dimension where $z_\tau^{(k)} \approx 1$ across time steps $t+1$ through $T$:</span>

$$
\prod_{\tau=t+1}^{T} z_\tau^{(k)} \approx 1
$$

<span style="font-size: 14px;">Unlike vanilla RNNs where gradients pass through tanh at every step (exponential decay), the GRU provides a linear shortcut. Gradients traverse many time steps without vanishing, as long as the update gate keeps those dimensions open.</span>

### <span style="font-size: 14px;">Two Gradient Paths</span>

* <span style="font-size: 14px;">**Direct path** ($z_t \odot h_{t-1}$): Gradient scaled by $z_t$. The "memory highway" preventing vanishing gradients.</span>
* <span style="font-size: 14px;">**Candidate path** ($(1 - z_t) \odot \tilde{h}_t$): Gradient flows through tanh and the reset gate. Subject to attenuation, but necessary for learning new representations.</span>

<span style="font-size: 14px;">Both paths contribute simultaneously. The direct path ensures learning signals reach earlier time steps even when the candidate path's gradient is small, analogous to residual connections in feedforward networks.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Swapping $z_t$ and $(1 - z_t)$</span>

<span style="font-size: 14px;">Writing $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ reverses the semantics: $z = 1$ would mean "take the candidate" instead of "keep old state." The correct formula has $z_t$ multiplying $h_{t-1}$. Mnemonic: $z$ is the "lazy gate" -- high $z$ means the unit does not bother updating.</span>

### <span style="font-size: 14px;">Using Matrix Multiplication Instead of Element-wise</span>

<span style="font-size: 14px;">The $\odot$ symbol means element-wise multiplication. Both $z_t$ and $h_{t-1}$ have shape $(N, H)$, and the result has shape $(N, H)$. Using `np.matmul` or `@` instead of `*` produces shape errors or silently wrong results.</span>

### <span style="font-size: 14px;">Confusing This Step with the Candidate Computation</span>

<span style="font-size: 14px;">The candidate computation uses the **reset gate** $r_t$ and applies tanh. The hidden state update uses the **update gate** $z_t$ and has **no nonlinearity**. Students sometimes apply tanh to the final output or use $r_t$ where $z_t$ belongs.</span>

### <span style="font-size: 14px;">Forgetting the Complement</span>

<span style="font-size: 14px;">Computing $h_t = z_t \odot h_{t-1} + z_t \odot \tilde{h}_t$ (using $z_t$ for both terms) breaks the convex combination. Coefficients no longer sum to one, and hidden state magnitude drifts. The $(1 - z_t)$ complement is essential.</span>

### <span style="font-size: 14px;">Assuming $z = 1$ Means "Update"</span>

<span style="font-size: 14px;">The name "update gate" is misleading. When $z_t = 1$: $h_t = 1 \cdot h_{t-1} + 0 \cdot \tilde{h}_t = h_{t-1}$. The unit does **not** update. When $z_t = 0$: $h_t = 0 \cdot h_{t-1} + 1 \cdot \tilde{h}_t = \tilde{h}_t$. Full update. The gate value represents how much to **keep**, not how much to **change**.</span>

### <span style="font-size: 14px;">Ignoring Batch Dimensions</span>

<span style="font-size: 14px;">All tensors have shape $(N, H)$ where $N$ is the batch size. The interpolation runs independently per sample. Incorrect reshaping can cause cross-sample contamination. Element-wise operations broadcast correctly across the batch dimension without special handling.</span>

---