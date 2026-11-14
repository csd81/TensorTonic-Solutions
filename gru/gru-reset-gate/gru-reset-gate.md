# <span style="font-size: 20px;">Reset Gate</span>

<span style="font-size: 14px;">The reset gate is one of two gating mechanisms in the Gated Recurrent Unit (GRU), introduced by Cho et al. (2014) in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." It controls how much of the previous hidden state flows into the candidate activation computation, allowing the network to selectively forget irrelevant history when constructing a new memory proposal.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The reset gate produces a vector $\mathbf{r}_t \in (0, 1)^H$ at each time step $t$, where $H$ is the hidden state dimensionality. Each element acts as a soft switch controlling how much of the corresponding dimension of $\mathbf{h}_{t-1}$ participates in computing the candidate hidden state $\tilde{\mathbf{h}}_t$. Values close to 1 let that hidden dimension pass through; values close to 0 erase it, forcing the candidate to rely on the current input.</span>

<span style="font-size: 14px;">The reset gate sits upstream of the candidate computation. It does not directly determine the final hidden state; it shapes the raw material from which the candidate is built. A separate update gate then decides how much of this candidate replaces the old state. The reset gate governs what the model proposes as new information, while the update gate governs how much of that proposal is accepted.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^H$ and the current input $\mathbf{x}_t \in \mathbb{R}^D$ are first concatenated into a single vector $[\mathbf{h}_{t-1}, \mathbf{x}_t] \in \mathbb{R}^{H+D}$. The concatenation order matters: following Cho et al. (2014), the hidden state appears first and the input second, and this must be consistent with how $\mathbf{W}_r$ is structured.</span>

<span style="font-size: 14px;">The concatenated vector is multiplied by a learned weight matrix $\mathbf{W}_r \in \mathbb{R}^{H \times (H+D)}$ and a bias $\mathbf{b}_r \in \mathbb{R}^H$ is added, then the result is passed through the element-wise sigmoid:</span>

$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)
$$

<span style="font-size: 14px;">The sigmoid squashes each element independently into $(0, 1)$, producing $H$ gate values. The reset gate output is then applied via element-wise multiplication with $\mathbf{h}_{t-1}$ inside the candidate computation:</span>

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b})
$$

<span style="font-size: 14px;">Here $\odot$ denotes the Hadamard (element-wise) product. The term $\mathbf{r}_t \odot \mathbf{h}_{t-1}$ scales each dimension of the previous hidden state by the corresponding reset gate value. When $r_{t,j} \approx 0$, the $j$-th dimension of $\mathbf{h}_{t-1}$ is zeroed out and the candidate is computed almost entirely from $\mathbf{x}_t$.</span>

---

## <span style="font-size: 16px;">Why a Reset Gate</span>

<span style="font-size: 14px;">Standard (vanilla) RNNs compute their hidden state as a fixed function of the previous state and current input, with no mechanism to selectively suppress irrelevant history. Old information persists and pollutes subsequent computations. When the network encounters a sudden context change, it must gradually overwrite the old state across multiple time steps rather than cleanly starting fresh.</span>

<span style="font-size: 14px;">The reset gate solves this with a learned, input-dependent ability to ignore past state dimensions. Consider a machine translation encoder processing "The cat sat on the mat. Meanwhile, stock prices surged." When the encoder reaches "Meanwhile," the reset gate can activate (values near 0) to suppress hidden state features encoding "cat" and "mat," since those are irrelevant to the financial clause.</span>

<span style="font-size: 14px;">Without the reset gate, the candidate always sees the full previous hidden state. The model could try to learn to ignore past information through the candidate weight matrices alone, but this is far less flexible. The reset gate provides an explicit, dynamic mechanism to zero out past dimensions, a more direct and parameter-efficient way to achieve selective forgetting.</span>

<span style="font-size: 14px;">The reset gate is also crucial for short-range dependencies. When $\mathbf{r}_t$ is close to zero across all dimensions, the candidate behaves like the first hidden state in a sequence, responding primarily to the current input.</span>

---

## <span style="font-size: 16px;">The Sigmoid Function</span>

<span style="font-size: 14px;">The sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ is used for the reset gate because of several properties that make it ideal for gating.</span>

<span style="font-size: 14px;">**Output range [0, 1].** The sigmoid maps any real-valued input to $(0, 1)$, which is essential for a multiplicative gate: 0 means "completely block" and 1 means "completely pass through." ReLU is unbounded above, tanh outputs $(-1, 1)$ allowing sign-flipping, and softmax enforces sum-to-one across dimensions. None suit independent per-dimension gating.</span>

<span style="font-size: 14px;">**Smooth and differentiable.** Gradients flow through the gate during backpropagation through time. The derivative has a simple form:</span>

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

<span style="font-size: 14px;">This is maximized at $x = 0$ where $\sigma'(0) = 0.25$, meaning the gate learns most efficiently when pre-activations are near the transition region.</span>

<span style="font-size: 14px;">**Saturation behavior.** When $x \gg 0$, $\sigma(x) \approx 1$; when $x \ll 0$, $\sigma(x) \approx 0$. In saturated regimes the gate behaves almost like a hard binary switch. During training, gates often start near 0.5 and gradually move toward saturated values as the network learns which dimensions to keep and which to drop.</span>

<span style="font-size: 14px;">**Element-wise independence.** The sigmoid is applied independently to each of the $H$ dimensions, so the network can reset some dimensions while keeping others.</span>

---

## <span style="font-size: 16px;">How Reset Differs from Update Gate</span>

<span style="font-size: 14px;">Both gates share the same functional form (linear transform + sigmoid) but serve different roles at different stages:</span>

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}) \quad \text{(reset shapes candidate)}
$$

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(update interpolates)}
$$

<span style="font-size: 14px;">The reset gate controls the content of the proposal: what new representation is suggested. The update gate controls acceptance: how much of the proposal replaces the old state. During a topic change, the reset gate fires (values near 0) to suppress old features in the candidate, while the update gate takes high values (near 1) so the fresh candidate overwrites the old state. If the update gate were low, the fresh candidate would be ignored regardless of what the reset gate did.</span>

<span style="font-size: 14px;">Each gate has its own weight matrix and bias: $\mathbf{W}_r, \mathbf{b}_r$ for reset and $\mathbf{W}_z, \mathbf{b}_z$ for update. Both are $\mathbb{R}^{H \times (H+D)}$ but learn different patterns. The reset weights detect when history should be ignored in candidate construction; the update weights detect when the final state should change.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The reset gate was introduced in Cho et al. (2014), "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." The paper proposed the GRU as part of a novel RNN Encoder-Decoder for learning fixed-length representations of variable-length phrases for machine translation.</span>

<span style="font-size: 14px;">Cho et al. explicitly state: "The reset gate $r_j$ allows the model to forget the previously computed information." When close to zero, the hidden state ignores the previous state and resets with the current input alone. This decomposition into independent decisions about what to forget (reset) and what to keep (update) is central to the GRU design.</span>

<span style="font-size: 14px;">Their empirical analysis found that reset gates learned to be more active (closer to 0) at boundaries between semantic units, firing at punctuation marks, conjunctions, and topic transitions, precisely where forgetting old context is linguistically motivated.</span>

<span style="font-size: 14px;">The GRU was proposed as a simpler alternative to the LSTM. By directly modifying the previous hidden state before candidate computation, the reset gate eliminates the need for a separate cell state and the output gate that controls access to it, reducing three gates to two.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a GRU with hidden size $H = 3$ and input size $D = 2$.</span>

### <span style="font-size: 14px;">Given Values</span>

<span style="font-size: 14px;">Previous hidden state and current input:</span>

$$
\mathbf{h}_{t-1} = \begin{bmatrix} 0.5 \\ -0.3 \\ 0.8 \end{bmatrix}, \quad \mathbf{x}_t = \begin{bmatrix} 1.0 \\ -0.5 \end{bmatrix}
$$

<span style="font-size: 14px;">Reset gate weight matrix $\mathbf{W}_r \in \mathbb{R}^{3 \times 5}$ and bias $\mathbf{b}_r \in \mathbb{R}^3$:</span>

$$
\mathbf{W}_r = \begin{bmatrix} 0.2 & -0.1 & 0.3 & 0.5 & -0.2 \\ -0.4 & 0.6 & 0.1 & -0.3 & 0.4 \\ 0.1 & -0.5 & 0.2 & 0.7 & -0.1 \end{bmatrix}, \quad \mathbf{b}_r = \begin{bmatrix} 0.1 \\ -0.2 \\ 0.0 \end{bmatrix}
$$

### <span style="font-size: 14px;">Step 1: Concatenate</span>

$$
[\mathbf{h}_{t-1}, \mathbf{x}_t] = \begin{bmatrix} 0.5 \\ -0.3 \\ 0.8 \\ 1.0 \\ -0.5 \end{bmatrix}
$$

<span style="font-size: 14px;">The first 3 elements come from $\mathbf{h}_{t-1}$ and the last 2 from $\mathbf{x}_t$, giving dimension $H + D = 5$.</span>

### <span style="font-size: 14px;">Step 2: Linear Transform</span>

<span style="font-size: 14px;">Compute each row of $\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]$:</span>

<span style="font-size: 14px;">**Row 1:** $(0.2)(0.5) + (-0.1)(-0.3) + (0.3)(0.8) + (0.5)(1.0) + (-0.2)(-0.5) = 0.10 + 0.03 + 0.24 + 0.50 + 0.10 = 0.97$</span>

<span style="font-size: 14px;">**Row 2:** $(-0.4)(0.5) + (0.6)(-0.3) + (0.1)(0.8) + (-0.3)(1.0) + (0.4)(-0.5) = -0.20 - 0.18 + 0.08 - 0.30 - 0.20 = -0.80$</span>

<span style="font-size: 14px;">**Row 3:** $(0.1)(0.5) + (-0.5)(-0.3) + (0.2)(0.8) + (0.7)(1.0) + (-0.1)(-0.5) = 0.05 + 0.15 + 0.16 + 0.70 + 0.05 = 1.11$</span>

<span style="font-size: 14px;">Adding bias $\mathbf{b}_r$:</span>

$$
\mathbf{a}_r = \begin{bmatrix} 0.97 + 0.1 \\ -0.80 - 0.2 \\ 1.11 + 0.0 \end{bmatrix} = \begin{bmatrix} 1.07 \\ -1.00 \\ 1.11 \end{bmatrix}
$$

### <span style="font-size: 14px;">Step 3: Apply Sigmoid</span>

<span style="font-size: 14px;">Apply $\sigma(x) = \frac{1}{1 + e^{-x}}$ element-wise:</span>

<span style="font-size: 14px;">**Element 1:** $\sigma(1.07) = \frac{1}{1 + e^{-1.07}} = \frac{1}{1.3430} = 0.7447$</span>

<span style="font-size: 14px;">**Element 2:** $\sigma(-1.00) = \frac{1}{1 + e^{1.00}} = \frac{1}{3.7183} = 0.2689$</span>

<span style="font-size: 14px;">**Element 3:** $\sigma(1.11) = \frac{1}{1 + e^{-1.11}} = \frac{1}{1.3296} = 0.7521$</span>

$$
\mathbf{r}_t = \begin{bmatrix} 0.7447 \\ 0.2689 \\ 0.7521 \end{bmatrix}
$$

### <span style="font-size: 14px;">Interpreting the Result</span>

* <span style="font-size: 14px;">**Dimension 1** ($r_1 = 0.7447$): about 74% of $h_{t-1,1} = 0.5$ passes through. The candidate sees $0.7447 \times 0.5 = 0.3724$.</span>
* <span style="font-size: 14px;">**Dimension 2** ($r_2 = 0.2689$): only 27% of $h_{t-1,2} = -0.3$ passes through. The candidate sees $0.2689 \times (-0.3) = -0.0807$. This dimension is being substantially reset.</span>
* <span style="font-size: 14px;">**Dimension 3** ($r_3 = 0.7521$): about 75% of $h_{t-1,3} = 0.8$ passes through. The candidate sees $0.7521 \times 0.8 = 0.6017$.</span>

<span style="font-size: 14px;">The reset-gated previous state $\mathbf{r}_t \odot \mathbf{h}_{t-1} = [0.3724, -0.0807, 0.6017]^T$ shows that dimension 2 has been substantially attenuated while dimensions 1 and 3 are moderately preserved. The network learned that dimension 2 of the hidden state is less relevant for constructing the candidate given this particular input.</span>

---

## <span style="font-size: 16px;">Connection to LSTM</span>

<span style="font-size: 14px;">The GRU reset gate is often compared to the LSTM forget gate $\mathbf{f}_t$. Both discard past information, but they differ in mechanism. The LSTM forget gate acts on a separate cell state $\mathbf{c}_{t-1}$:</span>

$$
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

<span style="font-size: 14px;">The forget gate directly scales the old cell state before adding new content. Its influence is on the final memory, not on how the candidate is computed. The GRU reset gate acts differently: it scales $\mathbf{h}_{t-1}$ before the candidate computation, so its influence is indirect. It determines what the model proposes, and the update gate determines how much is accepted.</span>

<span style="font-size: 14px;">The LSTM uses three gates plus a cell state (four weight matrices), while the GRU uses two gates with no cell state (three weight matrices). The reset gate enables this simplification by handling responsibilities that require both a forget gate and an output gate in the LSTM. Despite the differences, empirical comparisons (Chung et al., 2014) show GRUs and LSTMs achieve similar performance, with the GRU's advantage being computational efficiency.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Wrong Concatenation Order</span>

<span style="font-size: 14px;">The concatenation must be $[\mathbf{h}_{t-1}, \mathbf{x}_t]$, not $[\mathbf{x}_t, \mathbf{h}_{t-1}]$. If reversed, the first $D$ columns of $\mathbf{W}_r$ (learned for hidden state features) multiply input features instead. The output shape is still valid and the sigmoid still gives values in $(0, 1)$, so no runtime error occurs. The result is silently incorrect.</span>

### <span style="font-size: 14px;">Forgetting the Sigmoid</span>

<span style="font-size: 14px;">Without the sigmoid, the "gate" output is an unbounded real vector. When element-wise multiplied with $\mathbf{h}_{t-1}$, values can be arbitrarily large, causing gradient explosions. The gate's entire purpose is to produce values in $[0, 1]$ for soft switching, and the sigmoid is what makes this possible.</span>

### <span style="font-size: 14px;">Incorrect Weight Matrix Dimensions</span>

<span style="font-size: 14px;">$\mathbf{W}_r$ must be $\mathbb{R}^{H \times (H+D)}$, producing $H$ outputs from an $(H+D)$-dimensional input. Common mistakes: using $\mathbb{R}^{H \times H}$ (forgetting input), $\mathbb{R}^{H \times D}$ (forgetting hidden state), or $\mathbb{R}^{(H+D) \times H}$ (transposed). The first two cause shape mismatches; the last produces an $(H+D)$-dimensional output that fails when multiplied element-wise with $\mathbf{h}_{t-1} \in \mathbb{R}^H$.</span>

### <span style="font-size: 14px;">Confusing Reset with Update Gate</span>

<span style="font-size: 14px;">Both gates have identical functional forms but different weights and roles. The reset gate output $\mathbf{r}_t$ is used inside the candidate computation ($\mathbf{r}_t \odot \mathbf{h}_{t-1}$ feeds into tanh). The update gate output $\mathbf{z}_t$ is used for the final interpolation. Swapping them produces a model that trains but learns poorly, since gradient paths through the two gates are fundamentally different.</span>

### <span style="font-size: 14px;">Applying Reset at the Wrong Stage</span>

<span style="font-size: 14px;">The reset gate must be applied to $\mathbf{h}_{t-1}$ before the candidate's linear transformation. Applying it after the tanh ($\mathbf{r}_t \odot \tanh(\ldots)$) changes the semantics: instead of controlling what past information enters the candidate, it controls what candidate information enters the update, duplicating the update gate's role.</span>

### <span style="font-size: 14px;">Sharing Weights Between Gates</span>

<span style="font-size: 14px;">Each gate must have independent weights. Sharing $\mathbf{W}$ forces $\mathbf{r}_t = \mathbf{z}_t$, eliminating the GRU's ability to independently control candidate content and state interpolation.</span>

---