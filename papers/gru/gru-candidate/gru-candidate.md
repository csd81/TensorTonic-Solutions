# <span style="font-size: 20px;">Candidate Hidden State</span>

<span style="font-size: 14px;">The candidate hidden state $\tilde{h}_t$ is the GRU's proposal for new content to write into memory at each time step. It is computed by applying the reset gate element-wise to the previous hidden state, concatenating the result with the current input, passing through a linear transformation, and squashing with tanh. The update gate later decides how much of this candidate replaces the old hidden state. This mechanism, introduced by Cho et al. (2014), gives the GRU fine-grained control over which parts of the past are relevant when forming new representations.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">At every time step $t$, a recurrent network must compute a new hidden state $h_t$. In a standard RNN, $h_t$ is computed directly from the full previous hidden state $h_{t-1}$ and the current input $x_t$. The GRU introduces a two-stage process: first compute a **candidate** hidden state $\tilde{h}_t$ that represents what the new state would look like if the network chose to fully update, then blend this candidate with the old state $h_{t-1}$ using the update gate $z_t$.</span>

<span style="font-size: 14px;">The candidate is where the reset gate $r_t$ exerts its influence. Before computing $\tilde{h}_t$, the network multiplies $h_{t-1}$ element-wise by $r_t$. When a dimension of $r_t$ is close to zero, the corresponding dimension of $h_{t-1}$ is effectively erased, making the candidate ignore that part of the past. When $r_t$ is close to one, the full history flows through. This selective erasure allows the GRU to model both long-range dependencies and short-range resets.</span>

<span style="font-size: 14px;">The name "candidate" is significant: $\tilde{h}_t$ is not the final hidden state. It is a proposal. The final hidden state $h_t$ is an interpolation between $h_{t-1}$ and $\tilde{h}_t$, controlled by the update gate. This separation of "what to propose" from "how much to accept" is the core design principle of the GRU.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Reset-Gated Concatenation</span>

<span style="font-size: 14px;">The first step applies the reset gate $r_t$ element-wise to the previous hidden state. The reset gate $r_t \in (0, 1)^H$ has the same dimensionality as $h_{t-1} \in \mathbb{R}^H$. The element-wise product is:</span>

$$
r_t \odot h_{t-1}
$$

<span style="font-size: 14px;">This produces a vector in $\mathbb{R}^H$ where each dimension of the previous hidden state has been scaled by the corresponding reset gate value. The result is concatenated with $x_t \in \mathbb{R}^D$:</span>

$$
[r_t \odot h_{t-1}, \; x_t] \in \mathbb{R}^{H+D}
$$

### <span style="font-size: 14px;">Linear Transformation and Tanh</span>

<span style="font-size: 14px;">The concatenated vector is multiplied by weight matrix $W_h \in \mathbb{R}^{H \times (H+D)}$, bias $b_h \in \mathbb{R}^H$ is added, and tanh is applied:</span>

$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, \; x_t] + b_h)
$$

<span style="font-size: 14px;">The weight matrix $W_h$ maps from the $(H+D)$-dimensional concatenated space back to the $H$-dimensional hidden space. It can be decomposed conceptually into two blocks: $W_{hh} \in \mathbb{R}^{H \times H}$ operating on the reset-gated hidden state, and $W_{hx} \in \mathbb{R}^{H \times D}$ operating on the input. The tanh squashes every dimension into $[-1, 1]$.</span>

---

## <span style="font-size: 16px;">How the Reset Gate Modifies the Input</span>

<span style="font-size: 14px;">The reset gate $r_t$ is a vector of values in $(0, 1)^H$, computed at the same time step from $h_{t-1}$ and $x_t$ using a sigmoid activation. Each element $r_t^{(j)}$ controls how much of the $j$-th dimension of the previous hidden state participates in computing the candidate.</span>

<span style="font-size: 14px;">When $r_t^{(j)} \approx 0$, the $j$-th dimension of $h_{t-1}$ is multiplied by nearly zero, effectively erasing that dimension. The candidate for that dimension is computed as if there were no previous hidden state, relying only on $x_t$. When $r_t^{(j)} \approx 1$, the $j$-th dimension passes through unmodified, and the candidate uses the full history for that dimension, similar to a standard RNN.</span>

<span style="font-size: 14px;">Different dimensions can be reset independently. In a sentence like "The cat, which was sitting on the mat, stood up", the reset gate might zero out dimensions encoding the relative clause while preserving dimensions encoding the subject when computing the candidate at the word "stood". This per-dimension selectivity makes the GRU more expressive than a binary "remember all or forget all" mechanism.</span>

<span style="font-size: 14px;">The element-wise multiplication $r_t \odot h_{t-1}$ happens **before** concatenation with $x_t$. This ordering is critical. If the reset gate were applied after concatenation, it would also scale the input $x_t$, defeating the purpose. The input should always flow into the candidate computation at full strength; only the past should be selectively gated.</span>

---

## <span style="font-size: 16px;">Why Tanh</span>

<span style="font-size: 14px;">**Bounded output.** Hidden states are carried across many time steps. If the candidate were unbounded (as with ReLU), repeated accumulation could cause hidden state values to grow without limit. Tanh ensures every proposed value stays within $[-1, 1]$, preventing explosive growth.</span>

<span style="font-size: 14px;">**Zero-centered.** Unlike sigmoid, which outputs values in $(0, 1)$ with a mean of $0.5$, tanh is centered at zero. The final hidden state update $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ is an interpolation. If the candidate were always positive, the hidden state would develop a positive bias that accumulates over time. Zero-centered values allow the hidden state to represent both positive and negative features symmetrically.</span>

<span style="font-size: 14px;">**Gradient properties.** The derivative of tanh is $1 - \tanh^2(x)$, which achieves its maximum of $1.0$ at $x = 0$ and decays toward zero as $|x|$ grows. While this can cause vanishing gradients in vanilla RNNs, the GRU's gating mechanism provides alternative gradient pathways through the update gate, mitigating saturation.</span>

<span style="font-size: 14px;">**Convention in gated RNNs.** Both the GRU and the LSTM use tanh for candidate/cell state computations and sigmoid for gate computations. Sigmoid gates produce values in $(0, 1)$ that act as multiplicative controllers, while tanh produces values in $[-1, 1]$ that represent actual content. Confusing these roles would break the intended semantics.</span>

---

## <span style="font-size: 16px;">The Candidate as "Proposed New State"</span>

<span style="font-size: 14px;">The candidate $\tilde{h}_t$ represents what the hidden state would become if the network decided to fully overwrite the previous state. The update gate $z_t$ controls the interpolation:</span>

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

<span style="font-size: 14px;">When $z_t^{(j)} = 1$ for some dimension $j$, the final hidden state takes the candidate value entirely: $h_t^{(j)} = \tilde{h}_t^{(j)}$. When $z_t^{(j)} = 0$, the previous state is copied unchanged: $h_t^{(j)} = h_{t-1}^{(j)}$. For intermediate values, the result is a weighted blend.</span>

<span style="font-size: 14px;">This creates a clean separation of concerns. The **reset gate** decides which dimensions of history are relevant for *computing* the candidate. The **candidate** uses the filtered history to produce a concrete proposal. The **update gate** decides how much of that proposal to accept. Each component has a distinct, well-defined role.</span>

<span style="font-size: 14px;">The candidate can afford to be aggressive. Even if $\tilde{h}_t$ proposes a radical departure from $h_{t-1}$, the update gate can soften this by setting $z_t$ close to zero. Conversely, if the candidate closely resembles $h_{t-1}$, the update gate value becomes less consequential. This interplay gives the GRU its flexibility.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The GRU was introduced in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (Cho et al., 2014). The paper's primary contribution was the encoder-decoder architecture for sequence-to-sequence learning, and the GRU was proposed as the recurrent unit within it.</span>

<span style="font-size: 14px;">The paper states: "The candidate activation is computed similarly to that of the traditional recurrent unit. However, the previous hidden state is first multiplied (element-wise) by the reset gate." This captures the entire difference between the GRU candidate and a vanilla RNN hidden state. The weight matrix, bias, tanh, and concatenation are identical; only the reset gating is new.</span>

<span style="font-size: 14px;">The authors motivate the reset gate by noting that it "allows the hidden state to drop any information that is found to be irrelevant later in the future, thus, allowing a more compact representation." When the reset gate closes, the candidate is computed as if the sequence were starting anew, which is useful in machine translation where clause boundaries and topic shifts signal that prior context is no longer relevant.</span>

<span style="font-size: 14px;">Compared to the LSTM, the GRU candidate differs in an important way. The LSTM candidate $\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$ does not gate $h_{t-1}$ before computing the candidate. The LSTM instead uses separate forget and input gates to control the cell state update. The GRU merges these roles: the reset gate handles forgetting within the candidate computation, and the update gate handles both forget and input roles. This is why the GRU has fewer parameters than the LSTM.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a GRU with hidden size $H = 3$ and input size $D = 2$. We compute $\tilde{h}_t$ step by step.</span>

### <span style="font-size: 14px;">Given Values</span>

$$
h_{t-1} = \begin{pmatrix} 0.5 \\ -0.3 \\ 0.8 \end{pmatrix}, \quad r_t = \begin{pmatrix} 0.9 \\ 0.1 \\ 0.7 \end{pmatrix}, \quad x_t = \begin{pmatrix} 1.0 \\ -0.5 \end{pmatrix}
$$

$$
W_h = \begin{pmatrix} 0.2 & -0.1 & 0.3 & 0.5 & 0.1 \\ 0.4 & 0.2 & -0.2 & 0.1 & 0.3 \\ -0.1 & 0.5 & 0.1 & -0.3 & 0.2 \end{pmatrix}, \quad b_h = \begin{pmatrix} 0.1 \\ -0.1 \\ 0.05 \end{pmatrix}
$$

### <span style="font-size: 14px;">Step 1: Apply Reset Gate</span>

<span style="font-size: 14px;">Compute $r_t \odot h_{t-1}$ element-wise:</span>

$$
r_t \odot h_{t-1} = \begin{pmatrix} 0.9 \times 0.5 \\ 0.1 \times (-0.3) \\ 0.7 \times 0.8 \end{pmatrix} = \begin{pmatrix} 0.45 \\ -0.03 \\ 0.56 \end{pmatrix}
$$

<span style="font-size: 14px;">The first dimension ($0.5$) was barely reduced (scaled by $0.9$). The second dimension ($-0.3$) was almost completely zeroed out (scaled by $0.1$, giving $-0.03$). The third dimension ($0.8$) was moderately reduced (scaled by $0.7$). The reset gate decided dimension two of past history is largely irrelevant.</span>

### <span style="font-size: 14px;">Step 2: Concatenate</span>

$$
[r_t \odot h_{t-1}, \; x_t] = \begin{pmatrix} 0.45 \\ -0.03 \\ 0.56 \\ 1.0 \\ -0.5 \end{pmatrix}
$$

<span style="font-size: 14px;">The first three entries are the reset-gated hidden state, and the last two are the raw input. The input is never gated.</span>

### <span style="font-size: 14px;">Step 3: Linear Transformation</span>

<span style="font-size: 14px;">Compute $W_h \cdot [r_t \odot h_{t-1}, \; x_t] + b_h$ row by row:</span>

<span style="font-size: 14px;">**Row 1:** $(0.2)(0.45) + (-0.1)(-0.03) + (0.3)(0.56) + (0.5)(1.0) + (0.1)(-0.5) + 0.1 = 0.09 + 0.003 + 0.168 + 0.5 - 0.05 + 0.1 = 0.811$</span>

<span style="font-size: 14px;">**Row 2:** $(0.4)(0.45) + (0.2)(-0.03) + (-0.2)(0.56) + (0.1)(1.0) + (0.3)(-0.5) - 0.1 = 0.18 - 0.006 - 0.112 + 0.1 - 0.15 - 0.1 = -0.088$</span>

<span style="font-size: 14px;">**Row 3:** $(-0.1)(0.45) + (0.5)(-0.03) + (0.1)(0.56) + (-0.3)(1.0) + (0.2)(-0.5) + 0.05 = -0.045 - 0.015 + 0.056 - 0.3 - 0.1 + 0.05 = -0.354$</span>

$$
W_h \cdot [r_t \odot h_{t-1}, \; x_t] + b_h = \begin{pmatrix} 0.811 \\ -0.088 \\ -0.354 \end{pmatrix}
$$

### <span style="font-size: 14px;">Step 4: Apply Tanh</span>

$$
\tilde{h}_t = \tanh\!\begin{pmatrix} 0.811 \\ -0.088 \\ -0.354 \end{pmatrix} = \begin{pmatrix} 0.670 \\ -0.088 \\ -0.340 \end{pmatrix}
$$

<span style="font-size: 14px;">For small $|x|$, $\tanh(x) \approx x$, which is why the second and third values barely change. The first value ($0.811$) is moderately compressed to $0.670$. All values now lie in $[-1, 1]$. This candidate $(0.670, -0.088, -0.340)^T$ is the GRU's proposal. The update gate $z_t$ will blend it with $h_{t-1}$ to produce the final $h_t$.</span>

---

## <span style="font-size: 16px;">Comparison with Standard RNN</span>

<span style="font-size: 14px;">A standard RNN computes its hidden state as:</span>

$$
h_t = \tanh(W \cdot [h_{t-1}, \; x_t] + b)
$$

<span style="font-size: 14px;">The GRU candidate computation is:</span>

$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, \; x_t] + b_h)
$$

<span style="font-size: 14px;">The only structural difference is $r_t \odot$ before $h_{t-1}$. Everything else is identical. The GRU candidate reduces to a vanilla RNN hidden state when $r_t = \mathbf{1}$ (all ones). When $r_t = \mathbf{0}$ (all zeros), the candidate becomes $\tilde{h}_t = \tanh(W_{hx} \cdot x_t + b_h)$, a simple feed-forward transformation with no recurrent connection. The reset gate thus interpolates between a fully recurrent network and a purely feed-forward one.</span>

<span style="font-size: 14px;">In a vanilla RNN, $h_t$ is the final output with no further gating, forcing the entire hidden state to be overwritten at every step. This makes it difficult to preserve information over long sequences because the gradient must flow through a chain of tanh and matrix multiplications, causing vanishing gradients. The GRU addresses this by allowing the update gate to copy $h_{t-1}$ directly to $h_t$, creating a gradient shortcut similar to a residual connection.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Applying Reset Gate After Concatenation</span>

<span style="font-size: 14px;">A common mistake is to first concatenate $[h_{t-1}, x_t]$ and then apply the reset gate to the entire vector. This is wrong because the reset gate has shape $\mathbb{R}^H$, not $\mathbb{R}^{H+D}$, and it would incorrectly gate the input $x_t$. The reset gate must be applied to $h_{t-1}$ alone, before concatenation: gate first, then concatenate.</span>

### <span style="font-size: 14px;">Forgetting Element-Wise Multiplication</span>

<span style="font-size: 14px;">The reset gate is applied via element-wise (Hadamard) product $r_t \odot h_{t-1}$, not matrix multiplication. Both $r_t$ and $h_{t-1}$ are vectors of shape $\mathbb{R}^H$, so the operation multiplies corresponding elements. Using matrix multiplication ($r_t^T h_{t-1}$) would produce a scalar, collapsing all dimensions and destroying the per-dimension selectivity.</span>

### <span style="font-size: 14px;">Wrong Concatenation Order</span>

<span style="font-size: 14px;">The concatenation order $[r_t \odot h_{t-1}, \; x_t]$ must match the weight matrix layout. If $W_h$ has its first $H$ columns for the hidden state and last $D$ for the input, reversing to $[x_t, \; r_t \odot h_{t-1}]$ pairs the wrong weights with the wrong inputs. The convention in Cho et al. (2014) places the hidden state first.</span>

### <span style="font-size: 14px;">Tanh vs. Sigmoid Confusion</span>

<span style="font-size: 14px;">The candidate uses **tanh** (range $[-1, 1]$), while the gates use **sigmoid** (range $(0, 1)$). If sigmoid were used for the candidate, $\tilde{h}_t$ would always be positive, introducing a bias that prevents representing negative features. If tanh were used for a gate, values could be negative, flipping the sign of the gated quantity rather than scaling it toward zero. This violates the semantics of a gate as a "how much to let through" controller.</span>

### <span style="font-size: 14px;">Confusing Candidate with Final Hidden State</span>

<span style="font-size: 14px;">The candidate $\tilde{h}_t$ is not $h_t$. A common error is returning $\tilde{h}_t$ as the GRU output without the update gate interpolation $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. Skipping this removes the GRU's ability to preserve information across time steps, reducing it to a vanilla RNN with an unnecessary reset gate.</span>

---