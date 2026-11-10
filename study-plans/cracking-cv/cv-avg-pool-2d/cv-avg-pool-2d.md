# <span style="font-size: 20px;">Average Pool 2D</span>

<span style="font-size: 14px;">Average pooling is a parameter-free downsampling operation that replaces each window of a feature map with the **mean** of its values. Used since the earliest convolutional networks (LeCun et al., 1998) and prominent as the global pooling head of GoogLeNet (Szegedy et al., 2015) and ResNet (He et al., 2016), it shrinks spatial resolution while smoothing local responses, preserving overall intensity rather than selecting a single peak.</span>

---

## <span style="font-size: 16px;">What It Does</span>

<span style="font-size: 14px;">Average pooling partitions the spatial dimensions into windows and summarizes each window by its arithmetic mean. Where max pooling asks "what is the strongest response here," average pooling asks "what is the typical response here." The result is a low-pass, smoothing reduction.</span>

<span style="font-size: 14px;">The operation serves the same structural roles as any pooling layer, with a smoothing flavour:</span>

* <span style="font-size: 14px;">**Downsampling:** it reduces $H$ and $W$, lowering activation memory and the compute of every later layer. A $2 \times 2$ stride-2 pool quarters the spatial area.</span>
* <span style="font-size: 14px;">**Noise suppression:** averaging is a linear low-pass filter, so isolated spikes and pixel noise are attenuated rather than propagated.</span>
* <span style="font-size: 14px;">**Context aggregation:** by mixing every value in the window, it preserves the distribution of activity instead of discarding all but the maximum, which is useful when the spatial extent of a feature matters as much as its peak.</span>

<span style="font-size: 14px;">Like max pooling, average pooling has **no learnable parameters** and acts **independently per channel**, so a $(C, H, W)$ tensor becomes $(C, H_{out}, W_{out})$ with the channel count unchanged.</span>

---

## <span style="font-size: 16px;">Output Size Formula</span>

<span style="font-size: 14px;">For input height $H$ and width $W$, window size $k$, stride $s$, and padding $p$, the output dimensions are:</span>

$$
H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1, \quad W_{out} = \left\lfloor \frac{W + 2p - k}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">With no padding ($p = 0$) this matches the problem statement. The floor reflects that a partial window at the right or bottom edge is dropped by default. The same arithmetic governs every sliding-window operation, pooling and convolution alike: the numerator is the number of valid starting offsets, dividing by the stride counts the steps, and the $+1$ counts the window at position 0. The channel dimension never appears in this formula because pooling leaves channels untouched.</span>

---

## <span style="font-size: 16px;">The Pooling Operation</span>

<span style="font-size: 14px;">For each output position $(i, j)$, the window's top-left corner in the input is $(i \cdot s, j \cdot s)$, and the output is the mean over the $k \times k$ block:</span>

$$
\text{out}[i, j] = \frac{1}{k \cdot k} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} \text{image}[i \cdot s + u,\; j \cdot s + v]
$$

<span style="font-size: 14px;">The algorithm in steps:</span>

<span style="font-size: 14px;">1. **Compute output shape** using the formula above and allocate an $H_{out} \times W_{out}$ result.</span>

<span style="font-size: 14px;">2. **Slide the window** over output indices $i$ and $j$; multiply by stride $s$ to find the input offset.</span>

<span style="font-size: 14px;">3. **Reduce** by summing the $k \times k$ block and dividing by $k^2$, the fixed number of elements per window in the no-padding case.</span>

<span style="font-size: 14px;">Average pooling is a **linear** operation: the output is a fixed-weight linear combination of inputs, with every input in a window weighted equally by $1/k^2$. This is precisely a convolution with a constant kernel of value $1/k^2$ and the given stride. The linearity has a practical consequence: an average pool can be fused into an adjacent linear layer, something impossible for the nonlinear max.</span>

---

## <span style="font-size: 16px;">Backward Pass</span>

<span style="font-size: 14px;">Because the mean is differentiable, the backward pass is simple and dense. Each input in a window contributed $1/k^2$ of the output, so it receives $1/k^2$ of the upstream gradient:</span>

$$
\frac{\partial \, \text{out}[i,j]}{\partial \, \text{image}[a,b]} = \frac{1}{k^2} \quad \text{for every } (a,b) \text{ in the window}
$$

<span style="font-size: 14px;">This contrasts sharply with max pooling, where the gradient is routed to a single argmax position and every other element receives zero. Average pooling spreads the gradient evenly, so **every input receives a learning signal**. When windows overlap, an input belonging to several windows accumulates a contribution from each. The dense, uniform gradient makes average pooling smooth to optimize but also means it never sharpens a single dominant feature the way max pooling does.</span>

<span style="font-size: 14px;">Because the gradient is a known constant rather than a data-dependent route, the backward pass needs no information cached from the forward pass beyond the output shape. This makes average pooling slightly cheaper to differentiate and free of the index-bookkeeping that max pooling requires, a minor but real engineering advantage in memory-constrained training.</span>

---

## <span style="font-size: 16px;">Max Pooling Versus Average Pooling</span>

<span style="font-size: 14px;">The two operations share their arithmetic for output shape and differ only in the reduction, yet they behave quite differently:</span>

* <span style="font-size: 14px;">**Selection versus smoothing.** Max keeps the single strongest activation, acting as a feature detector robust to clutter. Average blends all values, acting as a low-pass filter that preserves background and overall intensity.</span>
* <span style="font-size: 14px;">**Salient features versus context.** If a window contains one strong edge response surrounded by near-zero values, max preserves the edge at full strength while average divides it by $k^2$, diluting it. When the goal is to detect that a sharp feature exists anywhere in the region, max wins; when the goal is to summarize the region's general activity, average wins.</span>
* <span style="font-size: 14px;">**Gradient density.** Max gives a sparse gradient (one location per window); average gives a dense, uniform gradient (every location). This makes average pooling gentler on optimization.</span>
* <span style="font-size: 14px;">**Linearity.** Average pooling is linear and fusible; max pooling is nonlinear and cannot be folded into a neighbouring linear layer.</span>

<span style="font-size: 14px;">Historically max pooling dominated the feature-extraction stages of AlexNet and VGG, where sharp, sparse responses needed to survive downsampling. Average pooling found its decisive role at the classifier head.</span>

<span style="font-size: 14px;">A useful way to remember the trade-off: max pooling is **invariant to where** the strong feature sits in the window but **discards how many** strong responses there were; average pooling is **sensitive to how many and how strong** the responses are but **blurs where** they sit. Tasks that care about presence and salience lean toward max; tasks that care about overall magnitude and distribution lean toward average. Some architectures even concatenate both pooled views to keep the strengths of each.</span>

---

## <span style="font-size: 16px;">Global Average Pooling and the Network Head</span>

<span style="font-size: 14px;">The most influential modern use of average pooling is **global average pooling** (GAP), introduced in Network in Network (Lin et al., 2014) and adopted by GoogLeNet and ResNet. GAP is average pooling with a window equal to the entire spatial extent, collapsing a $(C, H, W)$ tensor to $(C, 1, 1)$, one number per channel.</span>

<span style="font-size: 14px;">The motivation is concrete. Classic networks ended with large fully connected layers (AlexNet's final FC layers held the bulk of its parameters and were prone to overfitting). GAP replaces them with a parameter-free reduction that produces one feature per channel, which feeds directly into a small classifier. The benefits the authors cite are fewer parameters, stronger regularization, and a more interpretable mapping where each channel corresponds to a category-confidence map. ResNet's head is exactly a global average pool followed by a single linear layer.</span>

<span style="font-size: 14px;">Average pooling is the natural choice here, not max, because the goal is to summarize the **whole** spatial response of each channel into a stable descriptor, and the mean is far less sensitive to a single outlier location than the max would be.</span>

<span style="font-size: 14px;">GAP also decouples the network from a fixed input resolution. Because the head reduces whatever spatial size remains to $1 \times 1$, the same trained classifier accepts images of varying size, a property fully connected heads lack since they hard-code the flattened feature length. This flexibility is a direct precursor to the adaptive pooling generalization covered in the next problem, where the target spatial size is an arbitrary $(\text{out}_h, \text{out}_w)$ rather than $1 \times 1$.</span>

---

## <span style="font-size: 16px;">Parameters, FLOPs, and Receptive Field</span>

<span style="font-size: 14px;">Average pooling is as cheap as max pooling:</span>

* <span style="font-size: 14px;">**Parameters:** zero. The $1/k^2$ weights are constant, not learned.</span>
* <span style="font-size: 14px;">**FLOPs:** each output element costs $k^2 - 1$ additions plus one multiply by $1/k^2$, giving roughly $C \cdot H_{out} \cdot W_{out} \cdot k^2$ operations, negligible beside a channel-mixing convolution.</span>
* <span style="font-size: 14px;">**Memory:** unlike max pooling, no argmax indices need caching; the backward pass only needs the scalar $1/k^2$ and the output shape.</span>

<span style="font-size: 14px;">The receptive-field growth is identical to max pooling: a window $k$ stride $s$ pool multiplies the network jump by $s$ and expands the receptive field by $(k-1)$ times the prior jump. Stacked pooling stages are how a deep unit comes to summarize a large image region from purely local operations.</span>

<span style="font-size: 14px;">Because average pooling equals convolution with a constant $1/k^2$ kernel, frameworks can and do implement it through the same optimized convolution kernels, sharing the data-movement and tiling machinery. This equivalence is more than a curiosity: it explains why average pooling integrates seamlessly into the computational graph and why its gradient, the uniform $1/k^2$ spread, is exactly the gradient of that constant-weight convolution.</span>

---

## <span style="font-size: 16px;">Smoothing Interpretation</span>

<span style="font-size: 14px;">Viewing average pooling as a box filter clarifies its signal-processing behaviour. A $k \times k$ average is a normalized box kernel, which is a low-pass filter: it attenuates high spatial frequencies (fine detail, sharp edges, pixel noise) and passes low frequencies (smooth gradients, large regions). The strided application then resamples the smoothed signal at a coarser grid.</span>

<span style="font-size: 14px;">This has two practical implications:</span>

* <span style="font-size: 14px;">**Anti-aliasing tendency.** Because it smooths before subsampling, average pooling is gentler on aliasing artifacts than a bare strided subsample. Work on shift-invariant CNNs (Zhang, 2019) revived blurred-downsampling for exactly this reason, noting that naive max pooling and strided convs can alias and break translation equivariance.</span>
* <span style="font-size: 14px;">**Loss of sharpness.** The same low-pass property that removes noise also removes legitimate fine structure, so average pooling is rarely used where preserving crisp boundaries matters, such as the early layers of a detector.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take a $4 \times 4$ input, $k = 2$, $s = 2$, $p = 0$:</span>

$$
\text{image} = \begin{pmatrix} 1 & 3 & 2 & 4 \\ 5 & 6 & 1 & 2 \\ 7 & 2 & 9 & 0 \\ 1 & 8 & 3 & 4 \end{pmatrix}
$$

<span style="font-size: 14px;">Output size: $H_{out} = \lfloor (4-2)/2 \rfloor + 1 = 2$ and likewise $W_{out} = 2$, so the result is $2 \times 2$. Each cell is the mean of four values.</span>

<span style="font-size: 14px;">1. **Top-left** (rows 0-1, cols 0-1): $(1 + 3 + 5 + 6)/4 = 15/4 = 3.75$.</span>

<span style="font-size: 14px;">2. **Top-right** (rows 0-1, cols 2-3): $(2 + 4 + 1 + 2)/4 = 9/4 = 2.25$.</span>

<span style="font-size: 14px;">3. **Bottom-left** (rows 2-3, cols 0-1): $(7 + 2 + 1 + 8)/4 = 18/4 = 4.5$.</span>

<span style="font-size: 14px;">4. **Bottom-right** (rows 2-3, cols 2-3): $(9 + 0 + 3 + 4)/4 = 16/4 = 4.0$.</span>

$$
\text{out} = \begin{pmatrix} 3.75 & 2.25 \\ 4.5 & 4.0 \end{pmatrix}
$$

<span style="font-size: 14px;">Compare with max pooling on the same input, which gives $\begin{pmatrix} 6 & 4 \\ 8 & 9 \end{pmatrix}$. The averaged output is uniformly smaller and smoother, with the strong $9$ in the bottom-right window pulled down to $4.0$ by its weaker neighbours. This single example captures the whole distinction: average pooling dampens peaks, max pooling preserves them.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the floor and dropping a partial window.** With $H = 5$, $k = 2$, $s = 2$, the formula gives $\lfloor 3/2 \rfloor + 1 = 2$ outputs, so the last row is discarded by default. Expecting three outputs causes a shape mismatch downstream.</span>
* <span style="font-size: 14px;">**The count_include_pad subtlety.** When padding is used, the divisor can be either $k^2$ (counting padded zeros) or the number of real elements in the window. The PyTorch default counts the padding, so padded edge windows are divided by the full $k^2$ even though some entries are zero, biasing edge outputs toward smaller magnitudes. A from-scratch version that divides by the real-element count will not match.</span>
* <span style="font-size: 14px;">**Diluting sparse features.** Average pooling divides a lone strong activation by $k^2$. In a region that is mostly inactive, a real but sparse feature can be smoothed into near-nothing, which is why average pooling is a poor choice for the early feature-detection layers that max pooling was designed for.</span>
* <span style="font-size: 14px;">**Pooling across the wrong axis.** Pooling is strictly spatial and per-channel; the output channel count must equal the input channel count. Reducing over the channel axis collapses feature identity and destroys the representation.</span>

---