# <span style="font-size: 20px;">Depthwise Separable Convolution</span>

<span style="font-size: 14px;">Depthwise separable convolution factorizes a standard convolution into two cheaper operations: a **depthwise** convolution that filters each channel independently, followed by a **pointwise** $1 \times 1$ convolution that mixes channels. Introduced for vision by Xception (Chollet, 2017) and made central to efficient mobile networks by MobileNet (Howard et al., 2017), it cuts the parameter and FLOP cost of a convolution by roughly an order of magnitude with minimal accuracy loss, making competitive vision models practical on phones and embedded hardware.</span>

---

## <span style="font-size: 16px;">The Core Idea: Factorizing Convolution</span>

<span style="font-size: 14px;">A standard convolution does two jobs at once in a single dot product: it filters **spatially** (combining neighbouring pixels within a window) and it combines **channels** (mixing across the input depth). Depthwise separable convolution observes that these two jobs can be separated into successive steps, and that the separated form is far cheaper.</span>

<span style="font-size: 14px;">The factorization has two stages:</span>

* <span style="font-size: 14px;">**Depthwise convolution:** apply one $k \times k$ spatial filter per input channel, with no cross-channel mixing. Channel $c$ is convolved only with its own kernel, producing one output channel per input channel. This handles the spatial filtering.</span>
* <span style="font-size: 14px;">**Pointwise convolution:** apply a $1 \times 1$ convolution over the depthwise output. With no spatial extent it only mixes channels, projecting $C_{in}$ channels to $C_{out}$. This handles the channel combination.</span>

<span style="font-size: 14px;">Together they approximate what a standard convolution computes, but because each stage is much smaller than the joint operation, the total cost drops dramatically.</span>

<span style="font-size: 14px;">The key realization is that a standard convolution's weight tensor $(C_{out}, C_{in}, k, k)$ couples all four axes, so its size scales with the product $C_{out} \cdot C_{in} \cdot k^2$. The factorization replaces this single dense tensor with two much smaller ones: a depthwise tensor that scales as $C_{in} \cdot k^2$ and a pointwise tensor that scales as $C_{out} \cdot C_{in}$. Neither smaller tensor carries the full product, which is precisely why the combined cost is far below the dense cost. The trade is a modest loss of expressiveness, the separable form cannot represent every filter the dense form can, in exchange for a large efficiency gain that empirically costs little accuracy.</span>

---

## <span style="font-size: 16px;">Step 1: Depthwise Convolution</span>

<span style="font-size: 14px;">Each input channel is filtered by its own kernel, independently of all others. For channel $c$:</span>

$$
\text{dw}[c, i, j] = \sum_{p} \sum_{q} \text{depthwise\_weight}[c, 0, p, q] \cdot \tilde{x}[c,\; i s + p,\; j s + q]
$$

<span style="font-size: 14px;">where $\tilde{x}$ is the input padded by $p$ and $s$ is the stride. There is **no sum over channels**: each output channel depends only on the matching input channel. The output has $C_{in}$ channels and spatial size:</span>

$$
H_{dw} = \left\lfloor \frac{H + 2p - k_H}{s} \right\rfloor + 1, \qquad W_{dw} = \left\lfloor \frac{W + 2p - k_W}{s} \right\rfloor + 1
$$

<span style="font-size: 14px;">In framework terms this is exactly a grouped convolution with the number of groups equal to $C_{in}$, written `F.conv2d(..., groups=C_in)`. The depthwise weight tensor has shape $(C_{in}, 1, k_H, k_W)$: one single-channel kernel per input channel. The singleton second dimension reflects that each group has exactly one input channel, in contrast to the $(C_{out}, C_{in}, k, k)$ tensor of a dense convolution.</span>

---

## <span style="font-size: 16px;">Step 2: Pointwise Convolution</span>

<span style="font-size: 14px;">The pointwise step is an ordinary $1 \times 1$ convolution with stride 1 and padding 0. It takes the $C_{in}$-channel depthwise output and produces $C_{out}$ channels by mixing across channels at each spatial location:</span>

$$
\text{out}[c_{out}, i, j] = \text{bias}[c_{out}] + \sum_{c_{in}=0}^{C_{in}-1} \text{pointwise\_weight}[c_{out}, c_{in}, 0, 0] \cdot \text{dw}[c_{in}, i, j]
$$

<span style="font-size: 14px;">Because the kernel is $1 \times 1$, it does not touch spatial neighbours; it is a per-pixel linear projection over the channel axis. The pointwise weight tensor has shape $(C_{out}, C_{in}, 1, 1)$, and the bias is added once here, at the very end of the two-stage block, one scalar per output channel. The spatial size is unchanged from the depthwise output, so the final output is $(C_{out}, H_{dw}, W_{dw})$.</span>

---

## <span style="font-size: 16px;">The Cost Reduction, Quantified</span>

<span style="font-size: 14px;">This is the heart of why the block exists. Compare a standard $k \times k$ convolution from $C_{in}$ to $C_{out}$ channels against the separable form, counting multiply-accumulates per output spatial location.</span>

<span style="font-size: 14px;">A **standard convolution** costs, per output pixel:</span>

$$
\text{MACs}_{\text{std}} = C_{out} \cdot C_{in} \cdot k^2
$$

<span style="font-size: 14px;">The **separable** form costs the depthwise stage ($C_{in} \cdot k^2$, one $k \times k$ filter per channel) plus the pointwise stage ($C_{out} \cdot C_{in}$, a $1 \times 1$ mix):</span>

$$
\text{MACs}_{\text{sep}} = C_{in} \cdot k^2 + C_{out} \cdot C_{in}
$$

<span style="font-size: 14px;">The ratio of separable to standard cost is:</span>

$$
\frac{\text{MACs}_{\text{sep}}}{\text{MACs}_{\text{std}}} = \frac{1}{C_{out}} + \frac{1}{k^2}
$$

<span style="font-size: 14px;">For a $3 \times 3$ kernel ($k^2 = 9$) with a typical large $C_{out}$, the first term is negligible and the ratio is about $1/9$. The MobileNet paper reports roughly an 8 to 9 times reduction in both computation and parameters for $3 \times 3$ kernels, with only a small drop in ImageNet accuracy. The same ratio applies to parameter counts, since both are dominated by the same $C_{out} \cdot C_{in} \cdot k^2$ term in the standard case. The saving grows with the kernel size and is independent of the spatial resolution, since resolution multiplies both forms equally.</span>

---

## <span style="font-size: 16px;">A Concrete Cost Comparison</span>

<span style="font-size: 14px;">Numbers make the savings vivid. Consider a layer mapping $C_{in} = 128$ to $C_{out} = 256$ channels with a $3 \times 3$ kernel on a $28 \times 28$ feature map.</span>

<span style="font-size: 14px;">**Standard convolution** parameters: $256 \cdot 128 \cdot 9 = 294{,}912$ weights. MACs: that count times $28 \cdot 28 = 784$ output pixels, about $231$ million.</span>

<span style="font-size: 14px;">**Separable** parameters: depthwise $128 \cdot 9 = 1{,}152$ plus pointwise $256 \cdot 128 = 32{,}768$, totalling $33{,}920$. MACs: depthwise $1{,}152 \cdot 784 \approx 0.9$ million plus pointwise $32{,}768 \cdot 784 \approx 25.7$ million, about $26.6$ million.</span>

<span style="font-size: 14px;">The separable form uses roughly $8.7$ times fewer parameters and $8.7$ times fewer MACs, matching the predicted $1/C_{out} + 1/k^2 \approx 1/256 + 1/9 \approx 0.115$ ratio. Notice that the pointwise step dominates both budgets: once the spatial cost is factored out, the remaining expense is the channel mixing, which is why later designs such as MobileNetV2 work hard to keep the pointwise projections lean through inverted residuals and linear bottlenecks.</span>

---

## <span style="font-size: 16px;">Paper Context: MobileNet and Xception</span>

<span style="font-size: 14px;">**MobileNet** was designed for on-device inference where compute and memory are tight. Its entire backbone is a stack of depthwise separable blocks, each a depthwise convolution and a pointwise convolution, with batch norm and ReLU after each. The factorization is what lets a competitive ImageNet network run on a phone within a tight latency and memory budget. MobileNet adds two global hyperparameters, a width multiplier that scales the channel counts and a resolution multiplier that scales the input size, to trade accuracy for latency along a smooth curve.</span>

<span style="font-size: 14px;">**Xception** ("Extreme Inception") arrived at the same primitive from a different direction. The Inception module had already been factorizing convolutions into branches that partly decouple spatial and channel processing. Xception pushed this to the limit: completely separate the cross-channel mixing (a $1 \times 1$ convolution) from the spatial filtering (a per-channel spatial convolution). The paper frames depthwise separable convolution as the extreme point of the Inception hypothesis, that spatial and channel correlations can be mapped independently, and showed it outperformed a comparably sized Inception V3 on large-scale benchmarks.</span>

<span style="font-size: 14px;">A subtle ordering difference: MobileNet applies depthwise then pointwise with a nonlinearity between, while Xception found that omitting the intermediate nonlinearity worked slightly better in its setting. The two-step factorization is shared; the activation placement is a design knob.</span>

---

## <span style="font-size: 16px;">Grouped Convolution and the Spectrum</span>

<span style="font-size: 14px;">Depthwise convolution is the extreme case of **grouped convolution**, a generalization that sits on a spectrum between standard and depthwise. A grouped convolution with $g$ groups splits the $C_{in}$ input channels into $g$ disjoint groups and convolves each group only with its own filters, so cross-group mixing is forbidden.</span>

* <span style="font-size: 14px;">**$g = 1$:** the standard convolution, every output channel sees every input channel.</span>
* <span style="font-size: 14px;">**$g = C_{in}$:** the depthwise convolution, every channel is its own group with no mixing at all.</span>
* <span style="font-size: 14px;">**Intermediate $g$:** ResNeXt and the original AlexNet (which split across two GPUs) use moderate group counts to cut cost while retaining some cross-channel mixing.</span>

<span style="font-size: 14px;">Grouped convolution reduces the parameter and FLOP cost by a factor of $g$, because each filter spans only $C_{in}/g$ channels instead of all $C_{in}$. The depthwise case takes this to its limit, eliminating the channel sum entirely, and then restores cross-channel mixing cheaply through the separate $1 \times 1$ pointwise step. Viewing depthwise separable convolution this way clarifies that it is not a different operation but the endpoint of a well-understood family.</span>

---

## <span style="font-size: 16px;">Modern Evolution</span>

<span style="font-size: 14px;">The separable block became the standard efficiency primitive and continued to evolve:</span>

* <span style="font-size: 14px;">**MobileNetV2** (Sandler et al., 2018) wraps the depthwise step in an inverted residual: a pointwise expansion to a higher channel count, the depthwise spatial filter, then a pointwise projection back down, with a residual connection and a linear (no ReLU) final projection to avoid destroying information in the low-dimensional bottleneck.</span>
* <span style="font-size: 14px;">**EfficientNet** (Tan and Le, 2019) builds on these blocks and adds compound scaling of depth, width, and resolution.</span>
* <span style="font-size: 14px;">**Squeeze-and-excitation** channel attention is often inserted into separable blocks to recover some of the cross-channel modelling power lost by the factorization.</span>

<span style="font-size: 14px;">Across all of these, the depthwise-then-pointwise core remains, which is why mastering the two-stage computation and its cost arithmetic is foundational to understanding the entire efficient-network lineage.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take $C_{in} = 2$, a $3 \times 3$ input per channel, $C_{out} = 1$, a $2 \times 2$ depthwise kernel, stride 1, padding 0, bias 0. The depthwise output is $2 \times 2$ per channel.</span>

$$
x_0 = \begin{pmatrix} 1 & 2 & 0 \\ 0 & 1 & 3 \\ 2 & 1 & 0 \end{pmatrix}, \quad x_1 = \begin{pmatrix} 0 & 1 & 1 \\ 2 & 0 & 1 \\ 1 & 1 & 2 \end{pmatrix}
$$

$$
\text{dw}_0 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad \text{dw}_1 = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}, \quad \text{pw} = [2, 3]
$$

<span style="font-size: 14px;">1. **Depthwise channel 0** with kernel $\text{dw}_0$ (identity-like): top-left patch $\begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$ gives $1 + 1 = 2$. The four positions give $\begin{pmatrix} 2 & 5 \\ 1 & 1 \end{pmatrix}$.</span>

<span style="font-size: 14px;">2. **Depthwise channel 1** with all-ones kernel sums each $2 \times 2$ patch: positions give $\begin{pmatrix} 3 & 3 \\ 4 & 4 \end{pmatrix}$.</span>

<span style="font-size: 14px;">3. **Pointwise mix** with weights $[2, 3]$: $\text{out} = 2 \cdot \text{dw}_0 + 3 \cdot \text{dw}_1$. At $(0,0)$: $2 \cdot 2 + 3 \cdot 3 = 13$. The full output is $\begin{pmatrix} 13 & 19 \\ 14 & 14 \end{pmatrix}$.</span>

<span style="font-size: 14px;">The example shows the separation clearly: depthwise filters within each channel, pointwise blends the per-channel results into the final feature. The two depthwise outputs never interact until the pointwise step, which is the structural signature of the factorization.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Summing over channels in the depthwise step.** The depthwise convolution must keep channels independent; each output channel uses only the matching input channel and its own kernel. Accidentally summing across channels turns it into a standard convolution and destroys the cost savings and the intended factorization.</span>
* <span style="font-size: 14px;">**Adding the bias twice or in the wrong stage.** The bias belongs only to the pointwise output and is added once, at the end. Adding a bias after the depthwise step, or twice, shifts every value and diverges from the reference. The depthwise stage in this formulation has no bias.</span>
* <span style="font-size: 14px;">**Applying stride or padding to the pointwise step.** The spatial downsampling lives in the depthwise convolution. The pointwise $1 \times 1$ uses stride 1 and padding 0, so it preserves the depthwise spatial size. Putting the stride on the pointwise step gives the wrong output shape.</span>
* <span style="font-size: 14px;">**Confusing the depthwise output channel count.** The depthwise step outputs exactly $C_{in}$ channels (one per input channel), not $C_{out}$. The channel-count change to $C_{out}$ happens only in the pointwise step. Mis-sizing the intermediate tensor breaks the pointwise matmul and produces a dimension-mismatch error.</span>

---