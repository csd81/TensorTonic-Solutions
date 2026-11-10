# <span style="font-size: 20px;">Dilated 2D Convolution</span>

<span style="font-size: 14px;">Dilated convolution, also called **atrous convolution**, inserts gaps between kernel taps so a filter covers a larger input region without adding parameters or downsampling. Popularized by the DeepLab semantic-segmentation work (Chen et al., 2016) and by WaveNet (van den Oord et al., 2016), it expands the receptive field exponentially with depth while keeping spatial resolution intact, which is essential for dense per-pixel prediction such as semantic segmentation, where every pixel needs a label at full resolution.</span>

---

## <span style="font-size: 16px;">What It Does</span>

<span style="font-size: 14px;">A standard convolution samples a contiguous $kH \times kW$ patch. A dilated convolution with dilation factor $d$ samples the same number of taps but spaces them $d$ pixels apart in both spatial directions. Tap $(p, q)$ of the kernel reads the input at offset $(p \cdot d, \; q \cdot d)$ instead of $(p, q)$.</span>

<span style="font-size: 14px;">With $d = 1$ the operation is identical to ordinary convolution: taps are adjacent. With $d = 2$ a $3 \times 3$ kernel reads pixels at offsets $0, 2, 4$ along each axis, skipping every other pixel, so it "sees" a $5 \times 5$ region while still using only nine weights. The name atrous (French for "with holes") refers to these gaps, and the term dilation rate is used interchangeably with dilation factor for $d$.</span>

<span style="font-size: 14px;">The output equation, a dilated cross-correlation with no kernel flip, is:</span>

$$
\text{out}[c_o, i, j] = \text{bias}[c_o] + \sum_{c_i} \sum_{p} \sum_{q} \text{weight}[c_o, c_i, p, q] \cdot \text{image}[c_i,\; i + p d,\; j + q d]
$$

<span style="font-size: 14px;">The only change from standard convolution is the factor $d$ multiplying the kernel index. Channels are still summed over and each output channel still has an independent filter, and the weight tensor keeps the same $(C_{out}, C_{in}, kH, kW)$ shape.</span>

<span style="font-size: 14px;">An equivalent way to view dilation is as inserting $d - 1$ zeros between adjacent kernel taps to build a sparse dense kernel, then performing ordinary convolution. A $3 \times 3$ kernel at $d = 2$ becomes a $5 \times 5$ kernel that is mostly zeros, with the nine real weights at the corners, edge midpoints, and centre. This "kernel with holes" picture is where the name atrous comes from and explains why the effective kernel is $5 \times 5$ while the parameter count stays nine. Frameworks do not actually materialize the zero-padded kernel; they index the input with the stride $d$ directly, which is both faster and the convention this problem follows.</span>

---

## <span style="font-size: 16px;">Effective Kernel Size and Output Formula</span>

<span style="font-size: 14px;">Dilation stretches the kernel's spatial footprint. A $k$-tap kernel with dilation $d$ spans an **effective kernel size**:</span>

$$
k_{\text{eff}} = (k - 1) \cdot d + 1
$$

<span style="font-size: 14px;">The intuition: there are $k - 1$ gaps between $k$ taps, each gap widened to $d$ pixels, plus the final tap. For $k = 3, d = 2$ this gives $k_{\text{eff}} = 5$; for $d = 4$ it gives $9$. The number of **weights** stays $k$, but the **reach** grows linearly with $d$. Note that $d = 1$ recovers $k_{\text{eff}} = k$, confirming that standard convolution is the special case of dilated convolution with unit dilation.</span>

<span style="font-size: 14px;">Plugging $k_{\text{eff}}$ into the standard output-size formula with the problem's fixed stride 1 and padding 0:</span>

$$
H_{out} = H - (k_H - 1) d, \qquad W_{out} = W - (k_W - 1) d
$$

<span style="font-size: 14px;">This is the general $\lfloor (H + 2P - k_{\text{eff}})/s \rfloor + 1$ formula specialized to $P = 0$, $s = 1$. A larger dilation shrinks the output more, because the wider effective kernel cannot start as far to the right or bottom of the input. In practice dilated convolutions are usually paired with padding $P = d \cdot (k-1)/2$ to preserve resolution, but this problem fixes $P = 0$ so the shrinkage is visible.</span>

---

## <span style="font-size: 16px;">Receptive Field Growth</span>

<span style="font-size: 14px;">The receptive field is where dilation earns its place. For a single layer, the receptive field equals the effective kernel size $k_{\text{eff}} = (k-1)d + 1$. The power emerges when layers are stacked. Suppose layer $\ell$ uses dilation $d_\ell$ with a $k \times k$ kernel and stride 1. The receptive field $r_\ell$ grows as:</span>

$$
r_\ell = r_{\ell - 1} + (k - 1) \cdot d_\ell
$$

<span style="font-size: 14px;">If the dilation doubles each layer ($d_\ell = 2^{\ell-1}$) with $k = 3$, the receptive field after $L$ layers is $r_L = 1 + 2(2^L - 1)$, growing **exponentially** in depth. WaveNet exploited exactly this: stacks of dilated causal convolutions with rates $1, 2, 4, \ldots, 512$ give a single output a receptive field of thousands of audio samples using only a handful of layers, each with a tiny kernel.</span>

<span style="font-size: 14px;">Contrast this with a non-dilated stack, where the receptive field grows only **linearly** as $1 + L(k-1)$. To reach the same context, an undilated network would need far more layers, far more parameters, and far more compute. Dilation buys reach for free in both parameters and per-layer FLOPs.</span>

---

## <span style="font-size: 16px;">Why Dilation Instead of Pooling</span>

<span style="font-size: 14px;">The DeepLab authors faced a tension specific to semantic segmentation. Classification backbones aggressively downsample (pooling and strided convs) to grow the receptive field and reduce compute, ending with a tiny feature map. But segmentation needs a per-pixel label at full resolution, so the lost spatial detail must somehow be recovered. The standard fix, upsampling a coarse map, blurs object boundaries.</span>

<span style="font-size: 14px;">Dilated convolution resolves this by **decoupling receptive-field growth from resolution loss**. The reasoning the paper gives:</span>

* <span style="font-size: 14px;">**No downsampling.** With stride 1 and dilation, a layer enlarges its receptive field without reducing the feature-map size, so fine spatial detail is preserved.</span>
* <span style="font-size: 14px;">**No extra parameters.** The kernel keeps $k$ weights; only the sampling spacing changes. This is unlike using a larger dense kernel, which would multiply the parameter count.</span>
* <span style="font-size: 14px;">**Exponential reach with stacked dilation.** Stacking layers with dilation $1, 2, 4, 8, \ldots$ grows the receptive field exponentially with depth while every layer stays cheap, a structure WaveNet used for long-range audio context.</span>

<span style="font-size: 14px;">DeepLab takes a classification network, removes the last downsampling stages, and converts the following convolutions to dilated ones, recovering the receptive field those stages would have provided but at full resolution.</span>

<span style="font-size: 14px;">Concretely, suppose a ResNet stage normally applies a stride-2 downsample followed by $3 \times 3$ convolutions. Removing the stride keeps the resolution but halves the receptive field each affected layer would have had. Setting dilation $d = 2$ on those convolutions restores the original receptive field exactly, because the effective kernel doubles its reach. The feature map stays at the higher resolution while every neuron sees the same context it would have in the original network. This surgical conversion, downsampling removed, dilation added, is the heart of the DeepLab recipe and lets a pretrained classification backbone be repurposed for dense prediction with minimal change.</span>

---

## <span style="font-size: 16px;">Stacking Dilations and Atrous Spatial Pyramid Pooling</span>

<span style="font-size: 14px;">A single dilation captures one scale. To capture multiple object scales at once, DeepLab v2 introduced **Atrous Spatial Pyramid Pooling (ASPP)**: several parallel dilated convolutions with different rates (for example $d = 6, 12, 18$) applied to the same feature map, their outputs fused. Each branch probes the image at a different scale, and the combination handles objects of varying size without resizing the input. A small dilation captures fine local structure while a large dilation captures broad context, and fusing them gives the classifier evidence at every scale simultaneously, which is exactly what segmenting a scene of mixed object sizes demands.</span>

<span style="font-size: 14px;">Stacking dilations in series is equally important. If consecutive layers use the same dilation, some input positions are never sampled, the **gridding** artifact. Choosing dilations that share no common factor (for example $1, 2, 3$ rather than $2, 4, 8$ alone) ensures the stacked receptive field covers every position densely, a refinement studied in later dilated-network work.</span>

<span style="font-size: 14px;">The gridding problem is worth understanding precisely. With dilation $d$ on every layer of a stack, each output depends only on input positions whose coordinates are congruent modulo $d$ to the output position. The map effectively splits into $d^2$ independent sub-lattices that never exchange information, so neighbouring pixels in the same sub-lattice are well connected while pixels in different sub-lattices are blind to each other. The visual result is a checkerboard of inconsistent predictions. Varying dilation rates, or interleaving a $d = 1$ layer, stitches the sub-lattices back together by introducing taps that cross the lattice boundaries.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take one input channel, a $5 \times 5$ image, a $3 \times 3$ kernel, dilation $d = 2$, stride 1, padding 0. The effective kernel is $(3-1) \cdot 2 + 1 = 5$, so $H_{out} = 5 - (3-1) \cdot 2 = 1$ and the output is $1 \times 1$.</span>

$$
\text{image} = \begin{pmatrix} 1 & 0 & 2 & 0 & 3 \\ 0 & 0 & 0 & 0 & 0 \\ 4 & 0 & 5 & 0 & 6 \\ 0 & 0 & 0 & 0 & 0 \\ 7 & 0 & 8 & 0 & 9 \end{pmatrix}, \quad w = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}, \quad \text{bias} = 0
$$

<span style="font-size: 14px;">With dilation 2, the nine taps sample input positions at row and column offsets $\{0, 2, 4\}$ from the single output location $(0,0)$. Those positions hold exactly the corner and midpoint values $1, 2, 3, 4, 5, 6, 7, 8, 9$:</span>

<span style="font-size: 14px;">1. **Row offset 0** (input row 0): columns $0, 2, 4$ give $1, 2, 3$.</span>

<span style="font-size: 14px;">2. **Row offset 2** (input row 2): columns $0, 2, 4$ give $4, 5, 6$.</span>

<span style="font-size: 14px;">3. **Row offset 4** (input row 4): columns $0, 2, 4$ give $7, 8, 9$.</span>

<span style="font-size: 14px;">Summing all nine weighted by 1: $1 + 2 + \ldots + 9 = 45$, so $\text{out}[0,0] = 45$. The zeros between the sampled positions are skipped entirely, demonstrating the "holes" in the kernel.</span>

<span style="font-size: 14px;">It is instructive to compare against the same kernel with $d = 1$. A dense $3 \times 3$ at the top-left would read input rows and columns $0, 1, 2$, the block $\begin{pmatrix} 1 & 0 & 2 \\ 0 & 0 & 0 \\ 4 & 0 & 5 \end{pmatrix}$, summing to $12$, and would produce a $3 \times 3$ output. The dilated version instead reaches the far corners of the image in a single application, summing to $45$, but collapses the output to $1 \times 1$ because the wide effective kernel barely fits. This is the trade in miniature: more reach per tap, less spatial room to slide.</span>

---

## <span style="font-size: 16px;">Parameters, FLOPs, and Modern Context</span>

<span style="font-size: 14px;">Dilation changes only the sampling pattern, not the layer's size:</span>

* <span style="font-size: 14px;">**Parameters:** identical to a standard convolution, $C_{out} \cdot C_{in} \cdot kH \cdot kW + C_{out}$. Dilation adds zero parameters, which is its defining advantage over simply using a larger dense kernel.</span>
* <span style="font-size: 14px;">**FLOPs:** also unchanged per output element, $C_{in} \cdot kH \cdot kW$ MACs, since the tap count is fixed. The total scales only with the output size, which dilation slightly reduces under valid padding, so a dilated layer is never more expensive in arithmetic than its dense counterpart.</span>
* <span style="font-size: 14px;">**Memory access:** the strided sampling is less cache-friendly than contiguous reads, so a dilated convolution can run a touch slower than a dense one of equal weight count despite identical FLOPs.</span>
* <span style="font-size: 14px;">**Receptive field:** grows with $d$ at no parameter cost, the entire reason the operation exists. This is the lever DeepLab and WaveNet pull to obtain wide context cheaply.</span>

<span style="font-size: 14px;">Dilated convolutions remain standard in segmentation (DeepLab v3, v3+), in dense prediction generally, and in some sequence models. Even as transformer-based segmentation emerges, atrous convolution stays a compact, parameter-free tool for enlarging context.</span>

<span style="font-size: 14px;">Beyond segmentation, the same idea recurs wherever long-range context is needed cheaply. Temporal convolutional networks use dilated causal convolutions as a parallelizable alternative to recurrent models for sequence tasks. Audio generation (WaveNet) and some time-series forecasters rely on dilated stacks for the same exponential-context property. The unifying theme is that dilation lets a model aggregate information from a wide window without the quadratic cost of attention or the sequential bottleneck of recurrence.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using the raw kernel size in the output formula.** The output size must use the effective kernel $(k-1)d + 1$, not $k$. Plugging in $k$ overestimates the output dimensions and causes a shape mismatch against `F.conv2d` whenever $d > 1$.</span>
* <span style="font-size: 14px;">**Indexing without the dilation factor.** Tap $(p, q)$ samples the input at $(i + p d, \; j + q d)$, not $(i + p, j + q)$. Forgetting to multiply the kernel index by $d$ silently computes an ordinary convolution, matching the reference only when $d = 1$.</span>
* <span style="font-size: 14px;">**Gridding artifacts from repeated dilation.** Stacking layers that all use the same dilation leaves periodic input positions unsampled, producing checkerboard-like coverage gaps and degraded segmentation. Vary the dilation rates across layers so the combined receptive field is dense.</span>
* <span style="font-size: 14px;">**Confusing dilation with stride.** Stride moves the kernel's start position between outputs and shrinks the output by subsampling. Dilation spaces the taps within a single application and shrinks the output by enlarging the effective kernel. They occupy different places in the output formula and are not interchangeable.</span>

---