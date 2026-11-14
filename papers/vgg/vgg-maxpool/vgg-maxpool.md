# <span style="font-size: 20px;">Max Pooling</span>

<span style="font-size: 14px;">Max pooling is a spatial downsampling operation that reduces feature map dimensions by selecting the maximum value within each local window. In VGGNet (Simonyan and Zisserman, 2014), max pooling with a 2x2 window and stride 2 is applied after each convolutional block, systematically halving spatial dimensions while preserving channel depth and the strongest activations learned by preceding convolutional filters.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Max pooling is a fixed, parameter-free operation that slides a small window across a feature map and outputs only the maximum value found within each window position. It serves as the spatial reduction mechanism between convolutional blocks in VGGNet, compressing height and width while leaving batch size and channel count untouched.</span>

<span style="font-size: 14px;">The operation takes a 4D input tensor in NHWC format (batch, height, width, channels) and produces an output with the same batch size $N$ and channel count $C$, but with halved height and width. Every channel is pooled independently. Unlike convolutional layers, max pooling introduces no learnable parameters. Its behavior is entirely determined by the window size $k$ and the stride $s$, both fixed at 2 throughout VGGNet.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">The Max Pooling Operation</span>

<span style="font-size: 14px;">For an input feature map $X$ with shape $(N, H_{in}, W_{in}, C)$ in NHWC format, the output at batch index $n$, spatial position $(i, j)$, and channel $c$ is:</span>

$$
\text{out}[n, i, j, c] = \max_{0 \leq m < k, \; 0 \leq p < k} X[n, \; i \cdot s + m, \; j \cdot s + p, \; c]
$$

<span style="font-size: 14px;">With VGGNet's fixed $k = 2$ and $s = 2$, this simplifies to selecting the maximum of exactly 4 elements:</span>

$$
\text{out}[n, i, j, c] = \max\!\big(X[n, 2i, 2j, c], \; X[n, 2i, 2j+1, c], \; X[n, 2i+1, 2j, c], \; X[n, 2i+1, 2j+1, c]\big)
$$

<span style="font-size: 14px;">Each output element is the result of comparing exactly four input values arranged in a 2x2 spatial block.</span>

### <span style="font-size: 14px;">Output Dimensions</span>

<span style="font-size: 14px;">The general output dimension formula for pooling with kernel size $k$, stride $s$, and no padding is:</span>

$$
H_{out} = \left\lfloor \frac{H_{in} - k}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} - k}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">For VGGNet's $k = 2, s = 2$ with even input dimensions, this reduces cleanly to:</span>

$$
H_{out} = \frac{H_{in}}{2}, \quad W_{out} = \frac{W_{in}}{2}
$$

<span style="font-size: 14px;">The full output shape is $(N, H_{in}/2, W_{in}/2, C)$. Batch size and channel count pass through unchanged.</span>

---

## <span style="font-size: 16px;">Why Max Pooling</span>

<span style="font-size: 14px;">Max pooling serves three distinct purposes in convolutional architectures, all of which are essential to VGGNet's design.</span>

### <span style="font-size: 14px;">Spatial Downsampling</span>

<span style="font-size: 14px;">Each pooling layer halves both height and width, reducing the total number of spatial positions by a factor of 4. This progressive compression allows the network to build increasingly abstract representations: early layers capture fine textures at high resolution, while deeper layers capture object-level semantics at coarser resolution. Without pooling, VGGNet would need to maintain full $224 \times 224$ resolution through all layers, making the network computationally intractable.</span>

### <span style="font-size: 14px;">Translation Invariance</span>

<span style="font-size: 14px;">Max pooling provides a degree of translation invariance within each pooling window. If a feature shifts by one pixel within a 2x2 region, the maximum value often remains the same, producing an identical pooled output. Small spatial shifts in the input do not change the network's response, which is desirable for classification where the exact position of a feature matters less than its presence. This invariance compounds across VGGNet's five pooling stages.</span>

### <span style="font-size: 14px;">Reducing Computation</span>

<span style="font-size: 14px;">Convolution cost scales directly with the number of spatial positions. Halving $H$ and $W$ via pooling reduces this cost by $4\times$ for all subsequent layers. VGGNet's five pooling layers collectively reduce spatial dimensions from $224 \times 224 = 50{,}176$ positions down to $7 \times 7 = 49$ positions, a compression factor exceeding 1000x. Without this reduction, the computational cost of deeper convolutional layers would be enormous.</span>

---

## <span style="font-size: 16px;">The 2x2 Window</span>

<span style="font-size: 14px;">VGGNet's choice of a 2x2 pooling window with stride 2 is deliberate and differs from the 3x3/stride-2 overlapping pooling used in AlexNet.</span>

### <span style="font-size: 14px;">Four Elements, Pick the Max</span>

<span style="font-size: 14px;">At each output position, the pooling window covers exactly four input elements arranged in a 2x2 square. The operation selects the single largest value among these four. The other three values are discarded. This is a lossy compression: from four values, only one survives. During backpropagation, the gradient flows only to the position that held the maximum value, and the three non-max positions receive zero gradient.</span>

### <span style="font-size: 14px;">Non-Overlapping with Stride 2</span>

<span style="font-size: 14px;">Because $s = k = 2$, adjacent pooling windows do not overlap. Each input element belongs to exactly one pooling window. The feature map is partitioned into a grid of non-overlapping 2x2 blocks, with no pixel shared between windows and no pixel left uncovered (assuming even dimensions).</span>

<span style="font-size: 14px;">This contrasts with AlexNet's overlapping pooling ($k=3, s=2$), where each 3x3 window shares one row or column with its neighbors. The VGGNet authors found overlapping pooling unnecessary given their deep stacks of small 3x3 convolutions, which already provide sufficient regularization through depth.</span>

---

## <span style="font-size: 16px;">Channel Independence</span>

<span style="font-size: 14px;">Max pooling operates entirely within individual channels. The max operation at position $(i, j)$ for channel $c$ considers only the four values at that spatial location in channel $c$, never looking at values from channel $c'$ where $c' \neq c$. This has several important implications:</span>

* <span style="font-size: 14px;">**Channel count is preserved:** If the input has $C$ channels, the output has exactly $C$ channels. Pooling never creates, removes, or mixes channels.</span>
* <span style="font-size: 14px;">**Feature map semantics are preserved:** If channel 42 detects horizontal edges, the pooled output of channel 42 still represents horizontal edges, just at half the spatial resolution.</span>
* <span style="font-size: 14px;">**No cross-channel reasoning:** Pooling cannot learn that two channels should be combined. That job belongs to the convolutional layers, which mix channels via their filter weights.</span>
* <span style="font-size: 14px;">**Independent max indices:** The position of the maximum in channel $c$ is unrelated to the position of the maximum in channel $c'$. Different channels may have their max at different spatial positions within the same 2x2 window.</span>

<span style="font-size: 14px;">This per-channel independence is why the pooling formula includes the channel index $c$ on both sides of the equation without any summation or interaction term across channels.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Simonyan and Zisserman (2014) describe their pooling configuration concisely: "Max-pooling is performed over a 2x2 pixel window, with stride 2." This single sentence defines the pooling layer used throughout all VGGNet configurations (VGG-11 through VGG-19).</span>

### <span style="font-size: 14px;">Five Max Pool Layers</span>

<span style="font-size: 14px;">VGGNet organizes its convolutional layers into five blocks, with one max pooling layer at the end of each block. The number of conv layers per block varies by configuration (VGG-16 uses 2-2-3-3-3, VGG-19 uses 2-2-4-4-4), but every configuration has exactly five pooling layers.</span>

### <span style="font-size: 14px;">Spatial Halving Cascade</span>

<span style="font-size: 14px;">Starting from the standard ImageNet input of $224 \times 224$:</span>

* <span style="font-size: 14px;">**After Pool 1:** $224 \times 224 \to 112 \times 112$ (after conv block 1, 64 channels)</span>
* <span style="font-size: 14px;">**After Pool 2:** $112 \times 112 \to 56 \times 56$ (after conv block 2, 128 channels)</span>
* <span style="font-size: 14px;">**After Pool 3:** $56 \times 56 \to 28 \times 28$ (after conv block 3, 256 channels)</span>
* <span style="font-size: 14px;">**After Pool 4:** $28 \times 28 \to 14 \times 14$ (after conv block 4, 512 channels)</span>
* <span style="font-size: 14px;">**After Pool 5:** $14 \times 14 \to 7 \times 7$ (after conv block 5, 512 channels)</span>

<span style="font-size: 14px;">Each pooling layer exactly halves both spatial dimensions because $k = s = 2$ and all dimensions are even at every stage. The total spatial reduction is $224/7 = 32 = 2^5$, corresponding to the five pooling layers.</span>

<span style="font-size: 14px;">Channel count doubles at each block boundary (64, 128, 256, 512, 512), while spatial dimensions halve. This creates a roughly constant computational budget per block: halving spatial dimensions reduces FLOPs by 4x, while doubling channels increases FLOPs by roughly 4x.</span>

### <span style="font-size: 14px;">Design Philosophy</span>

<span style="font-size: 14px;">The VGGNet paper's central insight was that deep networks with small 3x3 filters outperform shallower networks with larger filters. Max pooling provides the necessary spatial reduction between blocks, allowing the network to grow deeper without exploding in computational cost. The pooling layers are not the paper's contribution, but they are essential infrastructure that makes the deep 3x3 architecture viable.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a single-channel 4x4 feature map processed by a 2x2 max pool with stride 2.</span>

### <span style="font-size: 14px;">Input Feature Map (4x4)</span>

$$
X = \begin{bmatrix} 1 & 3 & 2 & 8 \\ 5 & 6 & 1 & 4 \\ 7 & 2 & 9 & 0 \\ 3 & 4 & 6 & 5 \end{bmatrix}
$$

### <span style="font-size: 14px;">Output Dimensions</span>

$$
H_{out} = \frac{4}{2} = 2, \quad W_{out} = \frac{4}{2} = 2
$$

<span style="font-size: 14px;">The 4x4 input produces a 2x2 output. Four non-overlapping 2x2 windows tile the input exactly.</span>

### <span style="font-size: 14px;">Window-by-Window Computation</span>

<span style="font-size: 14px;">**Window (0, 0):** rows 0-1, columns 0-1</span>

$$
W_{0,0} = \begin{bmatrix} 1 & 3 \\ 5 & 6 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(1, 3, 5, 6) = 6$</span>

<span style="font-size: 14px;">**Window (0, 1):** rows 0-1, columns 2-3</span>

$$
W_{0,1} = \begin{bmatrix} 2 & 8 \\ 1 & 4 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(2, 8, 1, 4) = 8$</span>

<span style="font-size: 14px;">**Window (1, 0):** rows 2-3, columns 0-1</span>

$$
W_{1,0} = \begin{bmatrix} 7 & 2 \\ 3 & 4 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(7, 2, 3, 4) = 7$</span>

<span style="font-size: 14px;">**Window (1, 1):** rows 2-3, columns 2-3</span>

$$
W_{1,1} = \begin{bmatrix} 9 & 0 \\ 6 & 5 \end{bmatrix}
$$

<span style="font-size: 14px;">$\max(9, 0, 6, 5) = 9$</span>

### <span style="font-size: 14px;">Output Feature Map (2x2)</span>

$$
Y = \begin{bmatrix} 6 & 8 \\ 7 & 9 \end{bmatrix}
$$

<span style="font-size: 14px;">The output retains the dominant activation from each local region. The original 16 values have been compressed to 4. Because $s = k = 2$, no input element appears in more than one window: rows 0-1 feed the top output row, rows 2-3 feed the bottom, columns 0-1 feed the left output column, columns 2-3 feed the right.</span>

---

## <span style="font-size: 16px;">Max vs Average Pooling</span>

<span style="font-size: 14px;">The choice between max pooling and average pooling has meaningful consequences for what information is preserved during downsampling.</span>

### <span style="font-size: 14px;">Max Pooling Preserves the Strongest Activation</span>

<span style="font-size: 14px;">Max pooling selects the single largest value from each window. After ReLU activation (which VGGNet applies after every convolutional layer), the strongest activation indicates the most confident feature detection. If a convolution filter fires strongly at one position within a 2x2 window, max pooling preserves that response regardless of what the other three positions contain. This makes max pooling well-suited for classification where the question is "is this feature present?" rather than "how much is present on average?"</span>

### <span style="font-size: 14px;">Average Pooling Smooths</span>

<span style="font-size: 14px;">Average pooling computes the arithmetic mean of all values in the window, preserving overall magnitude but diluting strong signals. After ReLU, many activations are zero, and averaging with zeros pulls the pooled value down. A window containing $[0, 0, 0, 8]$ produces $\max = 8$ but $\text{avg} = 2$, weakening the detection signal by a factor of 4.</span>

### <span style="font-size: 14px;">VGG Chose Max Pooling</span>

<span style="font-size: 14px;">The VGGNet authors followed the convention established by AlexNet in using max pooling for spatial reduction. Max pooling was the dominant choice for classification architectures of that era because it consistently yielded better accuracy than average pooling in hidden layers. The fundamental tradeoff:</span>

* <span style="font-size: 14px;">**Max pooling:** Preserves peak response, ideal for detecting whether a feature is present somewhere in the local region.</span>
* <span style="font-size: 14px;">**Average pooling:** Preserves overall activation density, useful for estimating how broadly a feature is activated across the region.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Using the Wrong Stride</span>

<span style="font-size: 14px;">VGGNet pooling requires both $k = 2$ and $s = 2$. A common mistake is setting stride to 1 while keeping $k = 2$, which produces overlapping pooling and outputs with dimensions $H_{in} - 1$ instead of $H_{in} / 2$. For a $112 \times 112$ input, stride 1 produces $111 \times 111$ instead of the correct $56 \times 56$. This cascading error causes shape mismatches in every subsequent layer.</span>

### <span style="font-size: 14px;">Non-Divisible Dimensions</span>

<span style="font-size: 14px;">The formula $H_{out} = H_{in} / 2$ only holds when $H_{in}$ is even. If the input has odd dimensions, the floor operation applies:</span>

$$
H_{out} = \left\lfloor \frac{7 - 2}{2} \right\rfloor + 1 = 3
$$

<span style="font-size: 14px;">The rightmost column and bottom row are not covered by any window and are silently dropped. In VGGNet, this never occurs because all spatial dimensions in the pipeline are even (224, 112, 56, 28, 14), but when applying VGG-style pooling to custom inputs, odd dimensions cause unexpected size reductions.</span>

### <span style="font-size: 14px;">Pooling Across Channels</span>

<span style="font-size: 14px;">Max pooling operates per channel independently. A common implementation error is to take the max across both spatial and channel dimensions, collapsing multiple channels into one. If the input has shape $(1, 4, 4, 64)$, the correct output is $(1, 2, 2, 64)$. Pooling across channels would produce $(1, 2, 2, 1)$, destroying the per-channel feature structure. In NHWC format, the max operation must be applied only over the spatial window, never over the channel axis.</span>

### <span style="font-size: 14px;">Confusing Pooling with Strided Convolution</span>

<span style="font-size: 14px;">Both 2x2 max pooling with stride 2 and a convolution with stride 2 reduce spatial dimensions by half, but they are fundamentally different. Max pooling has no parameters and selects the maximum with fixed behavior. Strided convolution has learnable weights and computes a weighted sum. In VGGNet, all spatial reduction is performed by max pooling. Every convolutional layer uses stride 1 to preserve spatial dimensions within a block, and reduction happens exclusively at pooling layers between blocks.</span>

---