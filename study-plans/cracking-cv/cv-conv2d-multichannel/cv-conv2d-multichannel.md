# <span style="font-size: 20px;">Multichannel 2D Convolution</span>

<span style="font-size: 14px;">Multichannel 2D convolution is the core operation of every convolutional neural network. It maps a stack of $C_{in}$ input feature maps to a stack of $C_{out}$ output feature maps by sliding a set of learned filters over the spatial dimensions, summing contributions across all input channels. It is the workhorse of AlexNet, VGG, and ResNet, and understanding its exact arithmetic, weight shape, parameter count, and output size is fundamental.</span>

---

## <span style="font-size: 16px;">What It Computes</span>

<span style="font-size: 14px;">A convolutional layer transforms an input of shape $(C_{in}, H, W)$ into an output of shape $(C_{out}, H_{out}, W_{out})$. Each of the $C_{out}$ output channels is produced by one filter that spans **all** $C_{in}$ input channels. The filter slides over the spatial plane; at every position it computes a dot product between its weights and the local patch of the input (across every input channel), then adds a per-output-channel bias.</span>

<span style="font-size: 14px;">The defining equation for output channel $c_{out}$ at spatial position $(i, j)$ is:</span>

$$
\text{out}[c_{out}, i, j] = \text{bias}[c_{out}] + \sum_{c_{in}} \sum_{p} \sum_{q} \text{weight}[c_{out}, c_{in}, p, q] \cdot \tilde{x}[c_{in},\, i s + p,\, j s + q]
$$

<span style="font-size: 14px;">where $\tilde{x}$ is the input zero-padded by $P$ on all sides, $s$ is the stride, and the sums run over all input channels $c_{in}$ and all kernel positions $p \in [0, kH)$, $q \in [0, kW)$. Two properties stand out: the sum is over **all** input channels (channels are mixed), and each output channel has an **independent** filter.</span>

---

## <span style="font-size: 16px;">The Weight Tensor Shape</span>

<span style="font-size: 14px;">The learnable weight is a 4D tensor of shape $(C_{out}, C_{in}, kH, kW)$:</span>

* <span style="font-size: 14px;">**Dimension 0 ($C_{out}$):** one filter per output channel. Selecting `weight[c_out]` gives the complete 3D filter that produces output channel $c_{out}$.</span>
* <span style="font-size: 14px;">**Dimension 1 ($C_{in}$):** each filter has one $kH \times kW$ kernel per input channel. The filter is therefore a 3D volume that matches the input's channel depth.</span>
* <span style="font-size: 14px;">**Dimensions 2 and 3 ($kH, kW$):** the spatial extent of each kernel, typically $3 \times 3$ or $1 \times 1$ in modern networks.</span>

<span style="font-size: 14px;">The bias is a 1D tensor of shape $(C_{out},)$: one scalar added to every spatial location of its output channel. The mental model is: an output channel is a single learned 3D feature detector swept over the image, firing where its pattern appears across the combined input channels. There are exactly $C_{out}$ such detectors and $C_{out}$ biases, one per produced channel.</span>

---

## <span style="font-size: 16px;">Cross-Correlation, Not True Convolution</span>

<span style="font-size: 14px;">Deep learning frameworks implement **cross-correlation**, not the mathematically defined convolution. True convolution flips the kernel along both spatial axes before the sliding dot product:</span>

$$
(f * g)[i, j] = \sum_{p, q} f[p, q] \cdot g[i - p, j - q]
$$

<span style="font-size: 14px;">while cross-correlation uses $g[i + p, j + q]$ with no flip. The difference is irrelevant for a CNN because the kernel weights are learned: the network simply learns whatever orientation it needs, so the flip would only relabel the learned weights. Frameworks skip the flip for efficiency and call the result "convolution" by convention. The problem here follows this convention: no kernel flip, matching `F.conv2d`. The only place the distinction matters in practice is when porting weights between a true-convolution library and a cross-correlation one, where the kernels must be flipped to agree.</span>

---

## <span style="font-size: 16px;">Output Size Formula</span>

<span style="font-size: 14px;">The spatial output dimensions follow the standard sliding-window formula:</span>

$$
H_{out} = \left\lfloor \frac{H + 2P - kH}{s} \right\rfloor + 1, \qquad W_{out} = \left\lfloor \frac{W + 2P - kW}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">Each term:</span>

* <span style="font-size: 14px;">$H + 2P$ is the padded height; zero padding adds $P$ rows on top and bottom.</span>
* <span style="font-size: 14px;">Subtracting $kH$ accounts for the kernel: the last valid top-left position is at row $H + 2P - kH$.</span>
* <span style="font-size: 14px;">Dividing by $s$ counts stride steps; the $+1$ counts the position at offset 0; the floor drops any incomplete final window.</span>

<span style="font-size: 14px;">A widely used special case is "same" padding for an odd kernel with stride 1: setting $P = (k - 1)/2$ makes $H_{out} = H$, preserving spatial size. A $3 \times 3$ kernel with $P = 1$ keeps the resolution, which is why VGG stacks $3 \times 3$ convolutions with padding 1 and relies on separate pooling layers, rather than the convolutions themselves, to downsample.</span>

---

## <span style="font-size: 16px;">Parameter Count and FLOPs</span>

<span style="font-size: 14px;">The number of learnable parameters is the size of the weight tensor plus the biases:</span>

$$
\text{params} = C_{out} \cdot C_{in} \cdot kH \cdot kW + C_{out}
$$

<span style="font-size: 14px;">The parameter count is **independent of the spatial size** of the input, a direct consequence of weight sharing: the same filter is reused at every position. This is the property that lets convolution scale to large images without the parameter explosion of a fully connected layer. A $3 \times 3$ convolution from 64 to 128 channels holds $128 \cdot 64 \cdot 9 + 128 = 73{,}856$ parameters regardless of whether the feature map is $56 \times 56$ or $7 \times 7$.</span>

<span style="font-size: 14px;">The compute cost, counted as multiply-accumulate operations (MACs), depends on output size because the filter is applied at every output location:</span>

$$
\text{MACs} = C_{out} \cdot H_{out} \cdot W_{out} \cdot C_{in} \cdot kH \cdot kW
$$

<span style="font-size: 14px;">Each output element costs $C_{in} \cdot kH \cdot kW$ multiply-accumulates (the dot product over the full receptive volume across every input channel), and there are $C_{out} \cdot H_{out} \cdot W_{out}$ output elements. This formula is the basis for the FLOP-reduction arguments of MobileNet and other efficient architectures covered in later problems, where the goal is to cut the $C_{in} \cdot C_{out}$ factor without losing accuracy.</span>

---

## <span style="font-size: 16px;">Receptive Field and the 1x1 Convolution</span>

<span style="font-size: 14px;">A single $kH \times kW$ convolution gives each output unit a receptive field of $kH \times kW$ input pixels per channel. Stacking convolutions grows the receptive field: two $3 \times 3$ layers give an effective $5 \times 5$ field with fewer parameters than one $5 \times 5$ layer, a key insight of VGG.</span>

<span style="font-size: 14px;">The **$1 \times 1$ convolution** is an important special case where $kH = kW = 1$. It has no spatial extent, so it does not mix neighbouring pixels; instead it mixes only across channels, acting as a per-pixel linear layer over the channel dimension. Network in Network and GoogLeNet use $1 \times 1$ convolutions to cheaply change channel depth, and ResNet bottlenecks use them to reduce then restore channels around an expensive $3 \times 3$.</span>

<span style="font-size: 14px;">VGG made the receptive-field argument explicit. Two stacked $3 \times 3$ convolutions cover the same $5 \times 5$ field as one $5 \times 5$ convolution, but use $2 \cdot 3^2 = 18$ weights per channel pair instead of $25$, and insert an extra nonlinearity between them. Three $3 \times 3$ layers match a $7 \times 7$ field with $27$ weights instead of $49$. This is why deep stacks of small kernels replaced the large kernels of AlexNet: more nonlinearity, fewer parameters, the same receptive field.</span>

---

## <span style="font-size: 16px;">Why Weight Sharing Matters</span>

<span style="font-size: 14px;">The single most important architectural property of convolution is **weight sharing**: the same filter is applied at every spatial location. A fully connected layer mapping a $(C_{in}, H, W)$ input to $(C_{out}, H, W)$ outputs would need a weight for every input-output pair, on the order of $(C_{in} H W)(C_{out} H W)$ parameters, which is astronomically large for any real image.</span>

<span style="font-size: 14px;">Convolution replaces this with a tiny shared filter reused everywhere, encoding two priors that match the structure of natural images:</span>

* <span style="font-size: 14px;">**Locality.** A feature depends only on a small neighbourhood ($kH \times kW$), reflecting that pixels far apart are weakly related for low-level features.</span>
* <span style="font-size: 14px;">**Translation equivariance.** Because the same filter slides everywhere, a pattern detected at one location is detected identically when shifted. Shifting the input shifts the output by the same amount (up to stride and boundary effects).</span>

<span style="font-size: 14px;">These priors are why CNNs generalize from limited data far better than fully connected networks on images: the inductive bias is baked into the architecture rather than learned from scratch.</span>

---

## <span style="font-size: 16px;">How Frameworks Implement It</span>

<span style="font-size: 14px;">A naive seven-nested-loop implementation (over $C_{out}, H_{out}, W_{out}, C_{in}, kH, kW$) is correct but slow. Production frameworks reshape convolution into a single large matrix multiply, which maps onto highly optimized GEMM kernels on CPU and GPU.</span>

<span style="font-size: 14px;">The classic technique is **im2col**: every $C_{in} \times kH \times kW$ receptive volume is unrolled into a column, forming a matrix of shape $(C_{in} \cdot kH \cdot kW, \; H_{out} \cdot W_{out})$. The weight tensor is reshaped to $(C_{out}, \; C_{in} \cdot kH \cdot kW)$. A single matrix product then yields all output channels at all positions at once:</span>

$$
\text{out} = W_{\text{mat}} \cdot \text{im2col}(\tilde{x}) + \text{bias}
$$

<span style="font-size: 14px;">This trades extra memory (the unrolled matrix duplicates overlapping pixels) for the speed of a dense matrix multiply. Other backends use Winograd or FFT-based convolution for specific kernel sizes, but im2col plus GEMM remains the conceptual reference and explains why convolution shape bugs usually surface as matrix-dimension mismatches.</span>

<span style="font-size: 14px;">Viewing convolution as a matrix multiply also clarifies the backward pass. The gradient with respect to the input is itself a convolution-like operation (a transposed convolution with the flipped weights), and the gradient with respect to the weights is a cross-correlation between the input patches and the upstream gradient. Both reuse the same im2col machinery, which is why a single optimized convolution primitive serves forward and backward passes alike. The transposed-convolution view of the input gradient is exactly the operation studied in the transposed-convolution problem.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $C_{in} = 2$, a $3 \times 3$ input per channel, one output channel ($C_{out} = 1$), a $2 \times 2$ kernel, stride 1, padding 0. The output is $2 \times 2$ since $\lfloor (3 - 2)/1 \rfloor + 1 = 2$.</span>

$$
x_0 = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \\ 2 & 1 & 0 \end{pmatrix}, \quad x_1 = \begin{pmatrix} 0 & 1 & 1 \\ 2 & 0 & 1 \\ 1 & 1 & 2 \end{pmatrix}
$$

$$
w_0 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad w_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \text{bias} = 0
$$

<span style="font-size: 14px;">1. **Output $(0,0)$:** channel-0 patch top-left $\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$ dotted with $w_0$ gives $1 \cdot 1 + 1 \cdot 1 = 2$. Channel-1 patch $\begin{pmatrix} 0 & 1 \\ 2 & 0 \end{pmatrix}$ dotted with $w_1$ gives $1 \cdot 1 + 2 \cdot 1 = 3$. Sum across channels: $2 + 3 = 5$.</span>

<span style="font-size: 14px;">2. **Output $(0,1)$:** channel-0 patch $\begin{pmatrix} 2 & 0 \\ 1 & 3 \end{pmatrix} \cdot w_0 = 2 + 3 = 5$; channel-1 patch $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} \cdot w_1 = 1 + 0 = 1$; sum $= 6$.</span>

<span style="font-size: 14px;">3. **Output $(1,0)$:** channel-0 $\begin{pmatrix} 0 & 1 \\ 2 & 1 \end{pmatrix} \cdot w_0 = 0 + 1 = 1$; channel-1 $\begin{pmatrix} 2 & 0 \\ 1 & 1 \end{pmatrix} \cdot w_1 = 0 + 1 = 1$; sum $= 2$.</span>

<span style="font-size: 14px;">4. **Output $(1,1)$:** channel-0 $\begin{pmatrix} 1 & 3 \\ 1 & 0 \end{pmatrix} \cdot w_0 = 1 + 0 = 1$; channel-1 $\begin{pmatrix} 0 & 1 \\ 1 & 2 \end{pmatrix} \cdot w_1 = 1 + 1 = 2$; sum $= 3$.</span>

$$
\text{out} = \begin{pmatrix} 5 & 6 \\ 2 & 3 \end{pmatrix}
$$

<span style="font-size: 14px;">The example makes the channel mixing concrete: each output value is the sum of two single-channel cross-correlations, one per input channel, plus the bias.</span>

<span style="font-size: 14px;">Counting parameters for this layer confirms the formula: $C_{out} \cdot C_{in} \cdot kH \cdot kW + C_{out} = 1 \cdot 2 \cdot 2 \cdot 2 + 1 = 9$ learnable values (eight weights and one bias). Counting MACs: $C_{out} \cdot H_{out} \cdot W_{out} \cdot C_{in} \cdot kH \cdot kW = 1 \cdot 2 \cdot 2 \cdot 2 \cdot 2 \cdot 2 = 32$ multiply-accumulates for the whole layer, matching the four output cells each costing eight MACs.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Flipping the kernel.** Implementing true convolution with a kernel flip will not match `F.conv2d`, which performs cross-correlation. Since the problem and PyTorch both skip the flip, adding one silently produces wrong outputs on asymmetric kernels (symmetric kernels mask the bug).</span>
* <span style="font-size: 14px;">**Forgetting to sum over input channels.** Each output channel is the sum of contributions from every input channel. Computing only the matching channel, or treating channels independently, drops the cross-channel mixing that is the whole point of multichannel convolution and gives the wrong shape and values.</span>
* <span style="font-size: 14px;">**Off-by-one in the output size.** The floor in $\lfloor (H + 2P - kH)/s \rfloor + 1$ silently drops a partial window. Omitting padding from the formula, or forgetting the $+1$, yields a shape mismatch that crashes deep in the network.</span>
* <span style="font-size: 14px;">**Mis-indexing the padded input.** The patch for output $(i, j)$ starts at $(i s, j s)$ in the **padded** tensor, not the original. Indexing into the unpadded input, or applying the stride before padding, shifts every window and corrupts the result.</span>

---