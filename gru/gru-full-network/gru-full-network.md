# <span style="font-size: 20px;">Complete GRU Network</span>

<span style="font-size: 14px;">A Complete GRU Network unrolls a single Gated Recurrent Unit cell across every time step of an input sequence, accumulating information in a hidden state, and projects each hidden state to an output through a learned linear map. Introduced by Cho et al. (2014) for encoder-decoder machine translation, the GRU processes variable-length sequences with fewer parameters than the LSTM by merging the forget and input gates into a single update gate and eliminating the separate cell state.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The Complete GRU Network is the full sequence-processing system built around the GRU cell. A GRU cell defines the computation for a single time step, taking $x_t$ and $h_{t-1}$ and producing $h_t$. The complete network handles the entire sequence: it initializes $h_0$, iterates the cell $T$ times, collects all hidden states, and applies an output projection at each step to produce the final output tensor.</span>

<span style="font-size: 14px;">The architecture has three components. First, initialization of $h_0$ to zeros with shape $(N, H)$ where $N$ is batch size and $H$ is hidden dimension. Second, the recurrent loop applying GRU cell equations at each $t = 1, \ldots, T$, consuming $x_t \in \mathbb{R}^{N \times I}$ and $h_{t-1} \in \mathbb{R}^{N \times H}$ to produce $h_t \in \mathbb{R}^{N \times H}$. Third, the output projection $y_t = W_y h_t + b_y$ mapping each hidden state from dimension $H$ to output dimension $O$.</span>

<span style="font-size: 14px;">The key distinction from an LSTM network is structural simplicity. The LSTM maintains two state vectors per time step ($h_t$ and cell state $c_t$), while the GRU maintains only $h_t$. The final return value $h_{\text{last}}$ is simply $h_T$ rather than a tuple of $(h_T, c_T)$. Despite this, the GRU achieves comparable performance because the update and reset gates provide sufficient control over information flow.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The GRU cell at each time step $t$ computes the update gate $z_t$, reset gate $r_t$, and candidate hidden state $\tilde{h}_t$, then combines them into $h_t$.</span>

<span style="font-size: 14px;">**Update gate:**</span>

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

<span style="font-size: 14px;">**Reset gate:**</span>

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

<span style="font-size: 14px;">**Candidate hidden state:**</span>

$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$

<span style="font-size: 14px;">**New hidden state:**</span>

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

<span style="font-size: 14px;">Here $\sigma$ is sigmoid, $\odot$ is element-wise multiplication, $W_z, W_r, W_h \in \mathbb{R}^{I \times H}$ are input-to-hidden weights, $U_z, U_r, U_h \in \mathbb{R}^{H \times H}$ are hidden-to-hidden weights, and $b_z, b_r, b_h \in \mathbb{R}^{H}$ are biases. The update gate controls how much old state to keep versus how much candidate to accept. The reset gate controls how much previous hidden state participates in computing the candidate.</span>

<span style="font-size: 14px;">**Output projection (every time step):**</span>

$$
y_t = W_y h_t + b_y
$$

<span style="font-size: 14px;">where $W_y \in \mathbb{R}^{H \times O}$ and $b_y \in \mathbb{R}^{O}$. The network has two parameter groups: cell parameters ($W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h$) shared across all time steps, and projection parameters ($W_y, b_y$) also shared across time steps.</span>

<span style="font-size: 14px;">**Initialization:**</span>

$$
h_0 = \mathbf{0} \in \mathbb{R}^{N \times H}
$$

---

## <span style="font-size: 16px;">The Unrolling Loop</span>

<span style="font-size: 14px;">The core of the network is a loop that unrolls the cell across time. Given input $X \in \mathbb{R}^{N \times T \times I}$, the loop extracts one slice $x_t \in \mathbb{R}^{N \times I}$ at each step, feeds it with $h_{t-1}$ into the GRU cell, and stores $h_t$.</span>

* <span style="font-size: 14px;">**Step 0:** Initialize $h_0 = \mathbf{0} \in \mathbb{R}^{N \times H}$.</span>
* <span style="font-size: 14px;">**Step 1 (t=1):** Extract $x_1 = X[:, 0, :]$. Compute $h_1 = \text{GRUCell}(x_1, h_0)$. Store $h_1$.</span>
* <span style="font-size: 14px;">**Step 2 (t=2):** Extract $x_2 = X[:, 1, :]$. Compute $h_2 = \text{GRUCell}(x_2, h_1)$. Store $h_2$.</span>
* <span style="font-size: 14px;">**Step T:** Extract $x_T = X[:, T{-}1, :]$. Compute $h_T = \text{GRUCell}(x_T, h_{T-1})$. Store $h_T$.</span>

<span style="font-size: 14px;">After the loop, all hidden states are stacked into $H_{\text{all}} \in \mathbb{R}^{N \times T \times H}$. The last hidden state $h_{\text{last}} = h_T \in \mathbb{R}^{N \times H}$ is retained separately.</span>

<span style="font-size: 14px;">A critical property is **weight sharing**: the same cell parameters are used at every time step. This lets recurrent networks handle variable-length sequences. A network trained on length-10 sequences can process length-100 at inference because the same cell is applied more times. The loop must process steps in order because each depends on the previous hidden state, which enables temporal reasoning but prevents parallelization across time.</span>

---

## <span style="font-size: 16px;">Output Projection</span>

<span style="font-size: 14px;">The output projection is a linear layer applied independently at each time step. It maps $h_t \in \mathbb{R}^{N \times H}$ to $y_t \in \mathbb{R}^{N \times O}$ via $y_t = W_y h_t + b_y$. It is separate from the GRU cell and uses its own parameters.</span>

<span style="font-size: 14px;">This separation matters. The GRU cell maintains the hidden state as a compressed representation of all inputs seen so far. The output projection extracts task-specific information from that representation. For language modeling, $O$ might equal vocabulary size. For sequence tagging, $O$ equals the number of tags. The same GRU cell serves all tasks by swapping only the projection.</span>

<span style="font-size: 14px;">The projection can be applied inside the loop (computing $y_t$ right after $h_t$) or after the loop as a batched operation: $Y = H_{\text{all}} \cdot W_y + b_y$, yielding $Y \in \mathbb{R}^{N \times T \times O}$. Both produce identical results, but the batched approach is more efficient because it replaces $T$ small matrix multiplications with one large one. The output $Y$ contains one vector per time step per sample, the many-to-many pattern.</span>

---

## <span style="font-size: 16px;">Initialization</span>

<span style="font-size: 14px;">The initial hidden state $h_0$ is set to zeros with shape $(N, H)$. This is standard across PyTorch's `nn.GRU` and TensorFlow's GRU layer.</span>

<span style="font-size: 14px;">**Why zeros?** With $h_0 = \mathbf{0}$, the first step simplifies: the reset gate acts on a zero vector ($r_1 \odot h_0 = \mathbf{0}$), so the candidate becomes $\tilde{h}_1 = \tanh(W_h x_1 + b_h)$, depending only on the first input. The update gate interpolates between $\mathbf{0}$ and $\tilde{h}_1$, meaning the first hidden state is determined entirely by the first input and learned biases.</span>

<span style="font-size: 14px;">**Alternatives.** Some architectures learn $h_0$ as a trainable parameter broadcast across the batch. Others initialize $h_0$ from an encoder's output, as in Cho et al.'s encoder-decoder where the encoder's final hidden state becomes the decoder's $h_0$. Here, $h_0 = \mathbf{0}$ is used because the network is a standalone sequence processor.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Cho et al. (2014) introduced the GRU in "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." The paper proposes two contributions: the GRU as a simpler alternative to the LSTM, and the RNN Encoder-Decoder architecture for learning phrase representations useful for translation.</span>

<span style="font-size: 14px;">In the encoder-decoder framework, the encoder GRU processes the source sentence token by token, building a compressed representation in its hidden state. After the last source token, the encoder's final $h_T^{\text{enc}}$ serves as a fixed-length context vector. The decoder GRU generates the target sentence conditioned on this context and previously generated tokens.</span>

<span style="font-size: 14px;">The paper states: "The GRU is similar to the LSTM but without a separate memory cell." The design insight is that the LSTM's forget and input gates are redundant when combined. The GRU's update gate $z_t$ simultaneously controls what to forget ($1 - z_t$) and what to accept ($z_t$), reducing parameters by eliminating one gate and the separate cell state. Cho et al. evaluated on WMT'14 English-to-French, using learned phrase representations to rescore translation hypotheses, improving BLEU scores over the baseline.</span>

<span style="font-size: 14px;">The complete network in this problem corresponds to one half of the encoder-decoder. The output projection $y_t = W_y h_t + b_y$ would produce logits over the target vocabulary in translation. In the encoder, the projection might be omitted since only the final hidden state matters. This problem includes it to make the network a general-purpose sequence mapper.</span>

---

## <span style="font-size: 16px;">Sequence Processing</span>

<span style="font-size: 14px;">Information flows forward through time. At step 1, $h_1$ encodes only $x_1$. At step 2, $h_2$ encodes both $x_1$ and $x_2$ because $h_1$ feeds into the cell alongside $x_2$. By step $T$, $h_T$ has been influenced by every input $x_1, \ldots, x_T$.</span>

<span style="font-size: 14px;">The hidden state is the network's memory: a fixed-size vector of dimension $H$ regardless of sequence length. The GRU must compress all relevant information into this representation. The update gate decides how much old memory to retain versus overwrite, and the reset gate decides which parts of old memory are relevant for computing the new candidate.</span>

<span style="font-size: 14px;">In practice, the GRU is biased toward recent inputs. Information from $x_1$ must survive through $T{-}1$ gate operations to remain in $h_T$. While gating mitigates the vanishing gradient problem (gradients flow through the $(1 - z_t)$ pathway), the representational influence of early tokens still diminishes over long sequences. This motivates attention mechanisms, which allow direct access to any time step's hidden state without relying on the recurrent chain.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace a complete GRU network with $N=1$, $T=3$, $I=2$, $H=2$, $O=2$.</span>

<span style="font-size: 14px;">**GRU cell weights:**</span>

$$
W_z = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}, \; U_z = \begin{bmatrix} 0.1 & 0.0 \\ 0.0 & 0.1 \end{bmatrix}, \; b_z = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

$$
W_r = \begin{bmatrix} 0.2 & 0.1 \\ 0.4 & 0.3 \end{bmatrix}, \; U_r = \begin{bmatrix} 0.1 & 0.0 \\ 0.0 & 0.1 \end{bmatrix}, \; b_r = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

$$
W_h = \begin{bmatrix} 0.3 & 0.1 \\ 0.1 & 0.3 \end{bmatrix}, \; U_h = \begin{bmatrix} 0.2 & 0.0 \\ 0.0 & 0.2 \end{bmatrix}, \; b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

<span style="font-size: 14px;">**Output projection:**</span>

$$
W_y = \begin{bmatrix} 1.0 & 0.5 \\ 0.5 & 1.0 \end{bmatrix}, \quad b_y = \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}
$$

<span style="font-size: 14px;">**Input:** $X = [[1.0, 0.5],\; [0.5, 1.0],\; [0.0, 0.5]]$ (shape $1 \times 3 \times 2$).</span>

<span style="font-size: 14px;">**Step 0 -- Initialize.** $h_0 = [0, 0]$.</span>

<span style="font-size: 14px;">**Step 1 ($x_1 = [1.0, 0.5]$):** $z_1 = \sigma([0.20, 0.50]) = [0.5500, 0.6225]$. $r_1 = \sigma([0.25, 0.55]) = [0.5622, 0.6341]$. Since $h_0 = \mathbf{0}$, $\tilde{h}_1 = \tanh([0.35, 0.25]) = [0.3364, 0.2449]$. $h_1 = z_1 \odot \tilde{h}_1 = [0.1850, 0.1524]$. $y_1 = W_y h_1 + b_y = [0.3612, 0.3449]$.</span>

<span style="font-size: 14px;">**Step 2 ($x_2 = [0.5, 1.0]$):** $z_2 = \sigma([0.2685, 0.5652]) = [0.5667, 0.6376]$. $r_2 = \sigma([0.2185, 0.5152]) = [0.5544, 0.6260]$. $\tilde{h}_2 = \tanh([0.2705, 0.3691]) = [0.2638, 0.3539]$. $h_2 = (1{-}z_2) \odot h_1 + z_2 \odot \tilde{h}_2 = [0.2297, 0.2809]$. $y_2 = [0.4702, 0.4958]$.</span>

<span style="font-size: 14px;">**Step 3 ($x_3 = [0.0, 0.5]$):** $z_3 = \sigma([0.1230, 0.2281]) = [0.5307, 0.5568]$. $r_3 = \sigma([0.0730, 0.1781]) = [0.5182, 0.5444]$. $\tilde{h}_3 = \tanh([0.0738, 0.1806]) = [0.0737, 0.1787]$. $h_3 = (1{-}z_3) \odot h_2 + z_3 \odot \tilde{h}_3 = [0.1469, 0.2240]$. $y_3 = [0.3589, 0.3975]$.</span>

<span style="font-size: 14px;">**Final results:** $y = [[[0.3612, 0.3449],\; [0.4702, 0.4958],\; [0.3589, 0.3975]]]$ with shape $(1, 3, 2)$. $h_{\text{last}} = [[0.1469, 0.2240]]$ with shape $(1, 2)$.</span>

<span style="font-size: 14px;">The hidden state evolves: $h_1$ reflects only $x_1$, $h_2$ blends $x_1$ and $x_2$, $h_3$ carries all three. Update gate values around 0.53-0.64 show roughly half old state and half new candidate at each step.</span>

---

## <span style="font-size: 16px;">Bidirectional and Stacked GRUs</span>

<span style="font-size: 14px;">**Bidirectional GRU.** A bidirectional GRU runs two separate networks over the same input: one forward ($t = 1$ to $T$) and one backward ($t = T$ to $1$), each with its own parameters. At each step, the forward state $\overrightarrow{h}_t$ captures past context and the backward state $\overleftarrow{h}_t$ captures future context. These are concatenated: $h_t^{\text{bi}} = [\overrightarrow{h}_t ; \overleftarrow{h}_t] \in \mathbb{R}^{2H}$, so $W_y$ must be $(2H, O)$. Bidirectional GRUs are standard for tasks where the full sequence is available, such as named entity recognition.</span>

<span style="font-size: 14px;">**Stacked GRU.** Stacking uses the hidden states of one GRU layer as input for the next. Layer 1 produces $h_t^{(1)}$, which becomes $x_t^{(2)}$ for layer 2. Each layer has its own parameters. The output projection applies to the top layer: $y_t = W_y h_t^{(L)} + b_y$. Lower layers capture local patterns; higher layers capture abstract ones. Cho et al. (2014) used single-layer GRUs, but deeper recurrent networks have since proved stronger on complex tasks.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Wrong loop direction.**</span>

<span style="font-size: 14px;">The loop must iterate $t = 1, 2, \ldots, T$. Reversing the order makes $h_T$ depend only on $x_T$ and $h_1$ depend on all inputs, which is a backward GRU, not a forward one.</span>

<span style="font-size: 14px;">**2. Forgetting to save intermediate hidden states.**</span>

<span style="font-size: 14px;">Only keeping $h_T$ and discarding $h_1, \ldots, h_{T-1}$ means the output projection cannot compute $y_1, \ldots, y_{T-1}$. Append each $h_t$ to a list during the loop. Without this, the output has shape $(N, 1, O)$ instead of $(N, T, O)$.</span>

<span style="font-size: 14px;">**3. $h_{\text{last}}$ is $h_T$, not $h_0$.**</span>

<span style="font-size: 14px;">The return value $h_{\text{last}}$ is the hidden state after processing the entire sequence, not the zero initialization. It is easy to return a stale reference from before the loop.</span>

<span style="font-size: 14px;">**4. Dimension mismatch in output projection.**</span>

<span style="font-size: 14px;">$W_y$ must be $(H, O)$, not $(I, O)$ or $(O, H)$. The projection operates on hidden states, not inputs. $h_t$ is $(N, H)$ and $W_y$ must be $(H, O)$ to produce $(N, O)$.</span>

<span style="font-size: 14px;">**5. Not initializing $h_0$.**</span>

<span style="font-size: 14px;">If $h_0$ contains garbage values, every subsequent state is corrupted because GRU equations depend on $h_{t-1}$. Always set $h_0 = \mathbf{0}$ with shape $(N, H)$. Using shape $(H,)$ without the batch dimension causes broadcasting errors.</span>

<span style="font-size: 14px;">**6. Applying output projection inside the GRU cell.**</span>

<span style="font-size: 14px;">The projection $y_t = W_y h_t + b_y$ is not part of the cell. Mixing it into the cell means $y_t$ participates in the recurrence, fundamentally changing the architecture. The recurrence operates on $h_t$; the projection is a separate read-out.</span>

<span style="font-size: 14px;">**7. Confusing $y_T$ with $h_{\text{last}}$.**</span>

<span style="font-size: 14px;">$y_T \in \mathbb{R}^{N \times O}$ and $h_T \in \mathbb{R}^{N \times H}$ are different tensors with potentially different dimensions. The function returns $(Y, h_T)$, not $(Y, y_T)$.</span>

---