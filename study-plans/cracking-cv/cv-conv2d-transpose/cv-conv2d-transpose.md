# <span style="font-size: 20px;">2D Transposed Convolution</span>

<span style="font-size: 14px;">Transposed convolution, also called **fractionally-strided convolution** or (loosely) deconvolution, is a learnable upsampling operation. It increases the spatial resolution of a feature map using trainable weights, making it the standard tool for the decoder of segmentation networks (FCN, Long et al., 2015) and the generator of image-synthesis models (DCGAN, Radford et al., 2016). Mathematically it is the transpose of the forward convolution's linear map, which is exactly the operation a convolution's backward pass performs.</span>

---

## <span style="font-size: 16px;">What It Does</span>

<span style="font-size: 14px;">A standard convolution with stride $> 1$ shrinks spatial resolution: many input pixels map to one output. Transposed convolution does the inverse routing: each input pixel is multiplied by the kernel and the result is **scattered** onto a larger output buffer, with overlapping contributions summed. The effect is upsampling with learned filters rather than a fixed rule like nearest-neighbour or bilinear interpolation, so the network discovers how best to expand resolution for the task at hand.</span>

<span style="font-size: 14px;">The name "transposed" comes from linear algebra. Any convolution can be written as a multiplication by a sparse matrix $C$ that maps a flattened input to a flattened output, $y = Cx$. Transposed convolution multiplies by $C^T$, mapping the smaller space back to the larger one. The term "deconvolution" is common but misleading: this operation does not invert a convolution and cannot recover the original input from a convolved output, it only transposes the linear map and shares its sparsity pattern.</span>

---

## <span style="font-size: 16px;">The Scatter-Add Formulation</span>

<span style="font-size: 14px;">The clearest way to compute it is scatter-add. First build a zero buffer of shape $(C_{out}, (H-1)s + k_H, (W-1)s + k_W)$. Then for every input position $(c, h, w)$ and every kernel tap $(p, q)$, add the weighted input value into the buffer:</span>

$$
\text{buf}[c_{out},\, h s + p,\, w s + q] \mathrel{+}= \text{weight}[c, c_{out}, p, q] \cdot \text{image}[c, h, w]
$$

<span style="font-size: 14px;">Each input pixel "stamps" a scaled copy of the kernel onto the output, placed at the location determined by the stride. Where stamps from neighbouring input pixels overlap, their contributions **sum**. After the buffer is filled, crop $p$ (padding) rows and columns from every side and add the per-channel bias.</span>

<span style="font-size: 14px;">This scatter view is the conceptual opposite of the standard convolution's **gather**: a forward convolution gathers many input pixels into one output, while a transposed convolution scatters one input pixel into many outputs.</span>

<span style="font-size: 14px;">The buffer dimensions deserve a closer look. The unpadded buffer height $(H-1)s + k_H$ accounts for the last input row's stamp landing at offset $(H-1)s$ and extending $k_H$ rows further. Before any cropping, this is the raw extent the scatter produces. The final crop of $p$ rows from each side then yields $H_{out} = (H-1)s - 2p + k_H$. Implementing the buffer at the wrong size, or cropping before the scatter completes, is a frequent source of off-by-one errors.</span>

---

## <span style="font-size: 16px;">The Weight Layout and Output Size</span>

<span style="font-size: 14px;">The weight tensor for transposed convolution has the layout $(C_{in}, C_{out}, k_H, k_W)$, with **input channels first**, the reverse of a standard convolution's $(C_{out}, C_{in}, k_H, k_W)$. This is a direct consequence of the transpose: the forward map's $(C_{out}, C_{in})$ indexing becomes $(C_{in}, C_{out})$ when transposed, because the roles of input and output channels swap.</span>

<span style="font-size: 14px;">The output spatial size grows rather than shrinks:</span>

$$
H_{out} = (H - 1) s - 2p + k_H, \qquad W_{out} = (W - 1) s - 2p + k_W
$$

<span style="font-size: 14px;">This is precisely the algebraic inverse of the forward convolution's size formula. If a forward convolution with stride $s$, padding $p$, kernel $k$ maps size $H_{out}$ to $H$, then the transposed convolution with the same parameters maps $H$ back to $H_{out}$. The stride now **expands**: a stride of 2 roughly doubles the spatial size, which is why transposed convolution is the natural upsampling counterpart to strided downsampling.</span>

---

## <span style="font-size: 16px;">Relationship to the Convolution Gradient</span>

<span style="font-size: 14px;">Transposed convolution is not a new idea bolted on for upsampling; it is the same operation a convolution already performs during backpropagation. In the forward pass, a convolution computes $y = Cx$. The gradient of the loss with respect to the input is:</span>

$$
\frac{\partial L}{\partial x} = C^T \frac{\partial L}{\partial y}
$$

<span style="font-size: 14px;">Multiplying the upstream gradient by $C^T$ is exactly a transposed convolution. So the backward pass of a downsampling convolution and the forward pass of a transposed convolution are the same linear map. This is why frameworks implement both with shared code, and why the operation appears under the name "the gradient of convolution" in the FCN paper. It also means the transposed convolution's own backward pass is an ordinary convolution, the two are mutual transposes.</span>

<span style="font-size: 14px;">This symmetry is more than an implementation convenience. It guarantees that gradients flow cleanly through a transposed-convolution layer: the upstream gradient is gathered by a standard convolution exactly as in any conv layer, so the decoder of an FCN or the generator of a GAN trains with the same well-understood dynamics as the encoder, with no special-case gradient handling required anywhere in the network.</span>

---

## <span style="font-size: 16px;">The Fractional-Stride View</span>

<span style="font-size: 14px;">The alternative name "fractionally-strided convolution" comes from an equivalent construction. A transposed convolution with stride $s$ can be computed as an **ordinary** convolution applied to the input after inserting $s - 1$ zeros between every pair of input pixels (and adding appropriate border padding). The zero insertion spreads the input out to the target resolution, and a stride-1 convolution then fills in the values.</span>

<span style="font-size: 14px;">This is where "fractional stride" comes from: moving the kernel one step in the dilated input corresponds to moving a fraction $1/s$ of a step in the original input. The scatter-add formulation and the zero-insertion formulation compute the same result; scatter-add is usually clearer to implement from scratch, while zero-insertion makes the relationship to ordinary convolution explicit and is how some frameworks realize the operation internally.</span>

<span style="font-size: 14px;">Both views make the same key point: the stride in transposed convolution controls **expansion** of the output, not subsampling. A stride of 2 inserts one zero between pixels and roughly doubles each spatial dimension, the mirror image of a stride-2 forward convolution that roughly halves each spatial dimension.</span>

---

## <span style="font-size: 16px;">Paper Context: FCN and DCGAN</span>

<span style="font-size: 14px;">**FCN** (Fully Convolutional Networks, Long et al., 2015) needed to turn a coarse, heavily downsampled classification feature map back into a full-resolution per-pixel segmentation. It used transposed convolution as a learnable upsampling layer, initialized to bilinear interpolation and then fine-tuned, so the network could learn a better upsampling than a fixed interpolation rule. This let a classification backbone be repurposed for dense prediction end to end, and FCN combined coarse, deep upsampled features with finer, shallower ones through skip connections to sharpen boundaries.</span>

<span style="font-size: 14px;">**DCGAN** (Radford et al., 2016) built its generator entirely from transposed convolutions, mapping a low-dimensional noise vector through successively larger feature maps up to a full image. Each transposed-convolution layer roughly doubled the resolution while learning the upsampling filters, replacing the hand-designed upsampling of earlier generative models. The paper's architectural guidelines, fractionally-strided convolutions in the generator, strided convolutions in the discriminator, batch norm throughout, became a template for many later generative models built on convolutional generators.</span>

---

## <span style="font-size: 16px;">Checkerboard Artifacts</span>

<span style="font-size: 14px;">The scatter-add overlap creates a well-known failure mode: **checkerboard artifacts**. When the kernel size is not divisible by the stride, the overlapping stamps deposit uneven amounts of contribution across output positions. Some output cells receive contributions from more input stamps than their neighbours, producing a periodic high-low pattern that looks like a checkerboard, especially visible in the smooth regions of generated images where the regular intensity ripple stands out against an otherwise flat background.</span>

<span style="font-size: 14px;">Odena et al. (2016) analyzed this and recommended two fixes:</span>

* <span style="font-size: 14px;">**Choose kernel size divisible by stride** (for example $k = 4, s = 2$) so every output position receives an equal number of overlapping contributions, evening out the coverage.</span>
* <span style="font-size: 14px;">**Resize-convolution:** replace transposed convolution with a fixed upsampling (nearest-neighbour or bilinear) followed by an ordinary convolution. This decouples upsampling from learning and avoids the uneven overlap entirely, and is now common in image-generation decoders.</span>

---

## <span style="font-size: 16px;">Parameters, FLOPs, and Alternatives</span>

<span style="font-size: 14px;">Transposed convolution shares its parameter and compute structure with ordinary convolution, since it is the same linear map transposed:</span>

* <span style="font-size: 14px;">**Parameters:** $C_{in} \cdot C_{out} \cdot k_H \cdot k_W + C_{out}$, identical in count to a standard convolution but with the input and output channel roles swapped in the tensor layout. The weights are learnable, which is the entire advantage over a fixed interpolation kernel.</span>
* <span style="font-size: 14px;">**FLOPs:** each input pixel contributes $C_{out} \cdot k_H \cdot k_W$ multiply-accumulates per input channel, scattered into the output. The total scales with the input size times the kernel volume, comparable to a convolution producing the larger map at the upsampled resolution.</span>

<span style="font-size: 14px;">Because the learned upsampling can misbehave (checkerboard artifacts, and a tendency to need careful initialization), practitioners often weigh it against parameter-free alternatives:</span>

* <span style="font-size: 14px;">**Bilinear or nearest-neighbour upsampling** followed by a regular convolution, which separates the resolution increase from the learned filtering and sidesteps uneven overlap entirely, now the default in many segmentation decoders.</span>
* <span style="font-size: 14px;">**Pixel shuffle** (sub-pixel convolution), which produces extra channels with an ordinary convolution and then rearranges them into spatial positions, popular in super-resolution for its efficiency and resistance to checkerboard artifacts.</span>

<span style="font-size: 14px;">Transposed convolution remains widely used where a learnable, end-to-end-trainable upsampler is wanted, particularly in segmentation decoders and GAN generators that follow the DCGAN lineage.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Take one input channel, a $2 \times 2$ input, $C_{out} = 1$, a $2 \times 2$ kernel, stride 1, padding 0. The buffer size is $(2-1) \cdot 1 + 2 = 3$, so the output is $3 \times 3$.</span>

$$
\text{image} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad w = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad \text{bias} = 0
$$

<span style="font-size: 14px;">Each input value stamps a scaled copy of $w$ at its strided location:</span>

<span style="font-size: 14px;">1. **Input $(0,0) = 1$** stamps $1 \cdot w$ at buffer offset $(0,0)$: adds $1$ at $(0,0)$ and $1$ at $(1,1)$.</span>

<span style="font-size: 14px;">2. **Input $(0,1) = 2$** stamps at offset $(0,1)$: adds $2$ at $(0,1)$ and $2$ at $(1,2)$.</span>

<span style="font-size: 14px;">3. **Input $(1,0) = 3$** stamps at offset $(1,0)$: adds $3$ at $(1,0)$ and $3$ at $(2,1)$.</span>

<span style="font-size: 14px;">4. **Input $(1,1) = 4$** stamps at offset $(1,1)$: adds $4$ at $(1,1)$ and $4$ at $(2,2)$.</span>

<span style="font-size: 14px;">Summing the overlapping contributions, the center cell $(1,1)$ collects $1 + 4 = 5$:</span>

$$
\text{out} = \begin{pmatrix} 1 & 2 & 0 \\ 3 & 5 & 2 \\ 0 & 3 & 4 \end{pmatrix}
$$

<span style="font-size: 14px;">The overlap at the center is the scatter-add at work, the defining mechanic of the operation. Note how the four input values each appear once on the diagonal, while the off-diagonal stamps from the identity kernel populate the corners, and only the center accumulates two contributions. With a stride of 2 the stamps would be spread further apart and the output would be larger, illustrating how stride drives expansion.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using the standard weight layout.** The transposed-convolution weight is $(C_{in}, C_{out}, k_H, k_W)$, input channels first. Indexing it as if output channels came first (the standard convolution layout) transposes the channel mapping and produces wrong outputs whenever $C_{in} \ne C_{out}$.</span>
* <span style="font-size: 14px;">**Overwriting instead of accumulating.** The buffer must be filled with `+=`, summing overlapping stamps. Assigning instead of adding discards every contribution but the last, silently corrupting all overlap regions, which are exactly where the upsampling detail lives.</span>
* <span style="font-size: 14px;">**Mishandling padding as addition rather than cropping.** In transposed convolution, padding **removes** rows and columns from the output buffer, the opposite of standard convolution where padding adds them. Adding zeros instead of cropping yields the wrong output size and shifts every value.</span>
* <span style="font-size: 14px;">**Checkerboard artifacts from a stride that does not divide the kernel.** When $k$ is not a multiple of $s$, uneven overlap creates periodic intensity patterns in the output. Choosing $k$ divisible by $s$, or switching to resize-then-convolve, avoids the artifact, a fix the network cannot fully learn away on its own.</span>

---