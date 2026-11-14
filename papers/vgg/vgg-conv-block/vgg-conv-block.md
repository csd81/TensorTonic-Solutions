# <span style="font-size: 20px;">Conv Block</span>

<span style="font-size: 14px;">A **conv block** is the core repeating unit of VGGNet (Simonyan & Zisserman, 2014). It stacks multiple convolution-then-ReLU layers in sequence before passing the result to a pooling layer. This deceptively simple pattern of small $3 \times 3$ filters repeated two or three times replaced the large single-layer convolutions of earlier architectures like AlexNet, achieving deeper networks with fewer parameters and stronger nonlinear capacity.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A VGG conv block takes an input tensor and applies a fixed number of convolution layers back to back, with a ReLU activation after each convolution. There is no pooling, normalization, or skip connection inside the block. Pooling happens after the block, not within it.</span>

<span style="font-size: 14px;">In the simplified (pointwise) version used in this problem, each convolution is a linear transform applied independently at every spatial position. Given an input tensor $x$ of shape $(B, H, W, C_{in})$, a weight matrix $W$ of shape $(C_{in}, C_{out})$, and a bias vector $b$ of shape $(C_{out},)$, the convolution computes:</span>

$$
\texttt{out}[b, h, w, :] = x[b, h, w, :] \cdot W + b
$$

<span style="font-size: 14px;">followed by element-wise ReLU:</span>

$$
\texttt{out} = \max(0, \texttt{out})
$$

<span style="font-size: 14px;">This pair of operations (linear transform + ReLU) repeats for each layer in the block. The output of one layer becomes the input to the next. The first layer maps from $C_{in}$ to $C_{out}$ channels; all subsequent layers map from $C_{out}$ to $C_{out}$, keeping the channel dimension constant within the block.</span>

<span style="font-size: 14px;">Spatial dimensions ($H$ and $W$) are preserved throughout the block because VGGNet uses same-padding ($p = 1$ for $3 \times 3$ kernels with stride 1). Downsampling is handled exclusively by $2 \times 2$ max pooling with stride 2 after each block.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">For a block with $L$ convolution layers, the computation is a chain of linear-then-ReLU steps. Let $a_0 = x$ be the block input.</span>

<span style="font-size: 14px;">For each layer $l = 1, 2, \ldots, L$:</span>

$$
z_l[b, h, w, :] = a_{l-1}[b, h, w, :] \cdot W_l + b_l
$$

$$
a_l = \max(0, ; z_l)
$$

<span style="font-size: 14px;">The block output is $a_L$. Each weight matrix $W_l$ and bias vector $b_l$ is a separate set of learnable parameters:</span>

* <span style="font-size: 14px;">**Layer 1**: $W_1 \in \mathbb{R}^{C_{in} \times C_{out}}$, $b_1 \in \mathbb{R}^{C_{out}}$. This layer changes the channel dimension from $C_{in}$ to $C_{out}$.</span>
* <span style="font-size: 14px;">**Layers 2 through $L$**: $W_l \in \mathbb{R}^{C_{out} \times C_{out}}$, $b_l \in \mathbb{R}^{C_{out}}$. These layers operate entirely within the $C_{out}$ channel space.</span>

<span style="font-size: 14px;">The ReLU activation $\max(0, z)$ is applied element-wise, zeroing out all negative values. It introduces nonlinearity between every pair of linear transforms. Without ReLU, stacking $L$ linear layers would collapse to a single linear layer ($W_1 W_2 \cdots W_L$ is still a matrix), defeating the purpose of depth.</span>

---

## <span style="font-size: 16px;">Why $3 \times 3$ Filters</span>

<span style="font-size: 14px;">The central insight of the VGGNet paper is that a stack of small $3 \times 3$ convolution filters can replace a single large filter while achieving the same effective receptive field. A $3 \times 3$ filter is the smallest kernel that captures directional structure: left/right, up/down, and center. The paper states explicitly: "The use of three $3 \times 3$ conv. layers instead of a single $7 \times 7$ layer leads to more non-linearities and fewer parameters."</span>

### <span style="font-size: 14px;">Receptive Field Equivalence</span>

<span style="font-size: 14px;">When convolutions use stride 1 and same-padding, stacking them grows the effective receptive field linearly:</span>

* <span style="font-size: 14px;">**One $3 \times 3$ layer**: each output position sees a $3 \times 3$ patch of the input. Receptive field = $3$.</span>
* <span style="font-size: 14px;">**Two $3 \times 3$ layers**: the second layer sees a $3 \times 3$ patch of the first layer's output, each of which already covers $3 \times 3$ of the original input. Effective receptive field = $5 \times 5$. This matches a single $5 \times 5$ kernel.</span>
* <span style="font-size: 14px;">**Three $3 \times 3$ layers**: effective receptive field = $7 \times 7$, matching a single $7 \times 7$ kernel.</span>

<span style="font-size: 14px;">The general formula for $L$ stacked $3 \times 3$ layers is:</span>

$$
\text{receptive field} = 2L + 1
$$

<span style="font-size: 14px;">So $L = 1$ gives $3$, $L = 2$ gives $5$, and $L = 3$ gives $7$.</span>

---

## <span style="font-size: 16px;">Parameter Savings</span>

<span style="font-size: 14px;">The parameter advantage of stacked $3 \times 3$ layers over a single large kernel is significant. For a single convolution layer with $C$ input channels and $C$ output channels:</span>

* <span style="font-size: 14px;">**Single $5 \times 5$ layer**: $5^2 \times C^2 = 25C^2$ parameters (ignoring bias).</span>
* <span style="font-size: 14px;">**Two $3 \times 3$ layers**: $2 \times 3^2 \times C^2 = 18C^2$ parameters. That is $\frac{18}{25} = 72\%$ of the single-layer count, a $28\%$ reduction.</span>

<span style="font-size: 14px;">For the $7 \times 7$ case:</span>

* <span style="font-size: 14px;">**Single $7 \times 7$ layer**: $7^2 \times C^2 = 49C^2$ parameters.</span>
* <span style="font-size: 14px;">**Three $3 \times 3$ layers**: $3 \times 3^2 \times C^2 = 27C^2$ parameters. That is $\frac{27}{49} \approx 55\%$ of the single-layer count, a $45\%$ reduction.</span>

<span style="font-size: 14px;">Beyond raw parameter count, each stacked layer adds a ReLU activation. Two $3 \times 3$ layers give two nonlinearities versus one for a single $5 \times 5$ layer. Three $3 \times 3$ layers give three nonlinearities versus one for a single $7 \times 7$ layer. More nonlinearities mean a more discriminative function at each stage, which the VGGNet results confirm: deeper configurations consistently outperform shallower ones on ImageNet.</span>

---

## <span style="font-size: 16px;">The Stacking Pattern</span>

<span style="font-size: 14px;">VGGNet organizes its convolutional layers into five groups (blocks), each followed by $2 \times 2$ max pooling with stride 2. The number of conv layers per block and the channel count vary across configurations. The paper evaluates six configurations (A through E), with the two most cited being VGG-16 (configuration D) and VGG-19 (configuration E).</span>

### <span style="font-size: 14px;">VGG-16 (Configuration D)</span>

* <span style="font-size: 14px;">**Block 1**: 2 conv layers, 64 channels. Input $224 \times 224 \times 3$, output $112 \times 112 \times 64$ after pooling.</span>
* <span style="font-size: 14px;">**Block 2**: 2 conv layers, 128 channels. Output $56 \times 56 \times 128$ after pooling.</span>
* <span style="font-size: 14px;">**Block 3**: 3 conv layers, 256 channels. Output $28 \times 28 \times 256$ after pooling.</span>
* <span style="font-size: 14px;">**Block 4**: 3 conv layers, 512 channels. Output $14 \times 14 \times 512$ after pooling.</span>
* <span style="font-size: 14px;">**Block 5**: 3 conv layers, 512 channels. Output $7 \times 7 \times 512$ after pooling.</span>

<span style="font-size: 14px;">Total: 13 conv layers + 3 FC layers = 16 weight layers (hence "VGG-16").</span>

### <span style="font-size: 14px;">VGG-19 (Configuration E)</span>

<span style="font-size: 14px;">Blocks 3, 4, and 5 each have 4 conv layers instead of 3. Total: 16 conv layers + 3 FC layers = 19 weight layers.</span>

<span style="font-size: 14px;">Within every block, all conv layers use the same number of output channels. The channel count doubles at each block boundary (64, 128, 256, 512, 512), while spatial dimensions halve due to pooling. This creates a consistent trade-off: as spatial resolution shrinks, feature depth grows.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Simonyan and Zisserman published "Very Deep Convolutional Networks for Large-Scale Image Recognition" in 2014 (presented at ICLR 2015). The paper's central contribution is demonstrating that network depth, when achieved through stacked small filters, is a critical factor for image classification accuracy. At the time, AlexNet (2012) used large kernels ($11 \times 11$, $5 \times 5$), and the prevailing belief was that the first layer needed a large receptive field to capture meaningful patterns from raw pixels.</span>

<span style="font-size: 14px;">VGGNet challenged this by showing that a uniform architecture of exclusively $3 \times 3$ conv filters, stacked to depth, could substantially outperform AlexNet. The paper systematically evaluates configurations from 11 weight layers (A) to 19 weight layers (E), demonstrating consistent improvement with depth. VGG-16 and VGG-19 achieved state-of-the-art results on ImageNet 2014, with a top-5 test error of 7.3% using an ensemble.</span>

<span style="font-size: 14px;">The paper also introduced a practical training strategy: start by training the shallower configuration A, then initialize deeper networks by copying the first and last layers from A and initializing new intermediate layers randomly. This staged training approach addressed the difficulty of training very deep networks before batch normalization and residual connections were widely adopted.</span>

<span style="font-size: 14px;">VGGNet's influence extends beyond classification accuracy. Its simple, uniform block structure became a template for feature extraction in object detection (Faster R-CNN), segmentation (FCN), and style transfer. The conv block pattern of "stack N convolutions of the same channel size, then pool" became a foundational design principle in CNN architectures.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a small conv block with 2 layers. Input $x$ has shape $(1, 2, 2, 3)$: one image, $2 \times 2$ spatial, 3 input channels. The block outputs 2 channels.</span>

### <span style="font-size: 14px;">Layer 1: Linear Transform + ReLU</span>

<span style="font-size: 14px;">$W_1 \in \mathbb{R}^{3 \times 2}$, $b_1 \in \mathbb{R}^{2}$:</span>

$$
W_1 = \begin{pmatrix} 1 & -1 \\ 0 & 2 \\ -1 & 1 \end{pmatrix}, \quad b_1 = \begin{pmatrix} 0 & 0 \end{pmatrix}
$$

<span style="font-size: 14px;">Pixel at $(h=0, w=0)$ with $x[0,0,0,:] = (1, 0, 2)$:</span>

$$
z_1[0,0,0,:] = (1, 0, 2) \cdot \begin{pmatrix} 1 & -1 \\ 0 & 2 \\ -1 & 1 \end{pmatrix} + (0, 0) = (1 \cdot 1 + 0 \cdot 0 + 2 \cdot (-1), ; 1 \cdot (-1) + 0 \cdot 2 + 2 \cdot 1) = (-1, ; 1)
$$

$$
a_1[0,0,0,:] = \max(0, (-1, 1)) = (0, ; 1)
$$

<span style="font-size: 14px;">Pixel at $(h=0, w=1)$ with $x[0,0,1,:] = (2, 1, 0)$:</span>

$$
z_1[0,0,1,:] = (2, 1, 0) \cdot W_1 + b_1 = (2 + 0 + 0, ; -2 + 2 + 0) = (2, ; 0)
$$

$$
a_1[0,0,1,:] = \max(0, (2, 0)) = (2, ; 0)
$$

<span style="font-size: 14px;">Pixel at $(h=1, w=0)$ with $x[0,1,0,:] = (0, 1, 1)$:</span>

$$
z_1[0,1,0,:] = (0, 1, 1) \cdot W_1 + b_1 = (0 + 0 - 1, ; 0 + 2 + 1) = (-1, ; 3)
$$

$$
a_1[0,1,0,:] = \max(0, (-1, 3)) = (0, ; 3)
$$

<span style="font-size: 14px;">Pixel at $(h=1, w=1)$ with $x[0,1,1,:] = (1, 1, 1)$:</span>

$$
z_1[0,1,1,:] = (1, 1, 1) \cdot W_1 + b_1 = (1 + 0 - 1, ; -1 + 2 + 1) = (0, ; 2)
$$

$$
a_1[0,1,1,:] = \max(0, (0, 2)) = (0, ; 2)
$$

<span style="font-size: 14px;">After Layer 1, $a_1$ has shape $(1, 2, 2, 2)$ with values $(0, 1)$, $(2, 0)$, $(0, 3)$, $(0, 2)$.</span>

### <span style="font-size: 14px;">Layer 2: Linear Transform + ReLU</span>

<span style="font-size: 14px;">$W_2 \in \mathbb{R}^{2 \times 2}$, $b_2 \in \mathbb{R}^{2}$ (note: $C_{out} \to C_{out}$, not $C_{in} \to C_{out}$):</span>

$$
W_2 = \begin{pmatrix} 1 & 0 \\ -1 & 1 \end{pmatrix}, \quad b_2 = \begin{pmatrix} 0.5 & -0.5 \end{pmatrix}
$$

<span style="font-size: 14px;">Position $(0, 0)$, input $(0, 1)$:</span>

$$
z_2[0,0,0,:] = (0, 1) \cdot \begin{pmatrix} 1 & 0 \\ -1 & 1 \end{pmatrix} + (0.5, -0.5) = (0 - 1 + 0.5, ; 0 + 1 - 0.5) = (-0.5, ; 0.5)
$$

$$
a_2[0,0,0,:] = \max(0, (-0.5, 0.5)) = (0, ; 0.5)
$$

<span style="font-size: 14px;">Position $(0, 1)$, input $(2, 0)$:</span>

$$
z_2[0,0,1,:] = (2, 0) \cdot W_2 + b_2 = (2 + 0 + 0.5, ; 0 + 0 - 0.5) = (2.5, ; -0.5)
$$

$$
a_2[0,0,1,:] = \max(0, (2.5, -0.5)) = (2.5, ; 0)
$$

<span style="font-size: 14px;">Position $(1, 0)$, input $(0, 3)$:</span>

$$
z_2[0,1,0,:] = (0, 3) \cdot W_2 + b_2 = (0 - 3 + 0.5, ; 0 + 3 - 0.5) = (-2.5, ; 2.5)
$$

$$
a_2[0,1,0,:] = \max(0, (-2.5, 2.5)) = (0, ; 2.5)
$$

<span style="font-size: 14px;">Position $(1, 1)$, input $(0, 2)$:</span>

$$
z_2[0,1,1,:] = (0, 2) \cdot W_2 + b_2 = (0 - 2 + 0.5, ; 0 + 2 - 0.5) = (-1.5, ; 1.5)
$$

$$
a_2[0,1,1,:] = \max(0, (-1.5, 1.5)) = (0, ; 1.5)
$$

<span style="font-size: 14px;">Final block output has shape $(1, 2, 2, 2)$ with values $(0, 0.5)$, $(2.5, 0)$, $(0, 2.5)$, $(0, 1.5)$. ReLU after each layer clips negatives, creating sparse activations. Without the intermediate ReLU, the two linear layers would collapse to a single matrix $W_1 W_2$ and the network would lose the representational benefit of depth.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting ReLU between conv layers.** Every conv layer in a VGG block is followed by ReLU. Omitting the activation between consecutive convolutions makes the stack equivalent to a single linear layer: $W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2$, which is just another linear transform. Missing even one intermediate ReLU collapses those two layers into one, silently reducing the block's representational capacity without any error or crash.</span>
* <span style="font-size: 14px;">**Wrong weight dimensions for the first layer vs. subsequent layers.** The first conv in the block maps from $C_{in}$ to $C_{out}$, so $W_1$ has shape $(C_{in}, C_{out})$. All subsequent layers map from $C_{out}$ to $C_{out}$, so $W_l$ has shape $(C_{out}, C_{out})$. Using $(C_{in}, C_{out})$ for every layer causes a dimension mismatch at the second layer's matrix multiply when $C_{in} \neq C_{out}$.</span>
* <span style="font-size: 14px;">**Applying ReLU after the last conv when it should not be there.** In standard VGGNet, every conv layer including the last one in each block is followed by ReLU, so this is correct for VGG. But in architectures that reuse the conv block pattern with residual connections, the final activation may need to be omitted so the raw output can be added to the skip path. Always match the activation convention to the specific architecture.</span>
* <span style="font-size: 14px;">**Confusing "conv block" with "conv layer."** A conv block contains multiple conv layers. Saying "VGG-16 has 5 conv blocks" is correct; saying "VGG-16 has 5 conv layers" is wrong (it has 13). This confusion leads to incorrect parameter counts and misunderstanding of the architecture's capacity.</span>
* <span style="font-size: 14px;">**Assuming all blocks have the same number of layers.** In VGG-16, blocks 1 and 2 have 2 conv layers each, while blocks 3, 4, and 5 have 3 each. In VGG-19, the latter three blocks have 4 layers each. Hard-coding a fixed layer count per block will produce the wrong architecture for at least one configuration.</span>
* <span style="font-size: 14px;">**Forgetting that spatial dimensions stay constant within a block.** With $3 \times 3$ kernels, stride 1, and padding 1, every conv layer preserves $H$ and $W$. Spatial downsampling in VGGNet happens only at the $2 \times 2$ max pooling between blocks. Accidentally using valid padding (no padding) inside the block shrinks spatial dimensions by 2 pixels per layer, compounding across layers and breaking all downstream shape assumptions.</span>
* <span style="font-size: 14px;">**Mixing up parameter counts per block vs. per layer.** A block with 3 conv layers of 512 channels has $3 \times (3^2 \times 512^2) = 3 \times 2{,}359{,}296 = 7{,}077{,}888$ parameters (ignoring bias). Quoting the per-layer count as the block count (or vice versa) is a frequent error in architecture analysis.</span>

---