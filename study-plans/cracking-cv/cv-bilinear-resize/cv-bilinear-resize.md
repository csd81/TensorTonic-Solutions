# <span style="font-size: 20px;">Bilinear Resize</span>

<span style="font-size: 14px;">Bilinear resizing changes an image's resolution by estimating each output pixel as a **weighted average of its four nearest source pixels**. It is the default resampling method in nearly every deep learning pipeline (`F.interpolate(mode='bilinear')`, `transforms.Resize`) because it is smooth, cheap, and differentiable, which makes it usable inside a network as a sampling layer as well as in offline preprocessing.</span>

---

## <span style="font-size: 16px;">The Core Idea</span>

<span style="font-size: 14px;">Resizing maps a grid of output pixels onto the continuous coordinate space of the input image. An output pixel almost never lands exactly on an input pixel; it falls somewhere between four of them. **Nearest-neighbour** resizing just snaps to the closest one, producing blocky artifacts. **Bilinear** interpolation instead blends all four neighbours in proportion to how close the sample point is to each, giving a smooth result.</span>

<span style="font-size: 14px;">"Bilinear" means linear interpolation applied twice: once along the $x$ axis and once along the $y$ axis. The two passes commute, so the four-corner formula below computes both at once. The result is continuous everywhere, which is what removes the blocky stair-stepping that nearest-neighbour leaves behind.</span>

---

## <span style="font-size: 16px;">Source-Coordinate Mapping (Corner-Aligned)</span>

<span style="font-size: 14px;">The first step is deciding, for output pixel $(i, j)$, which continuous source coordinate it samples. This problem uses **corner-aligned** sampling (`align_corners=True`), where the corner pixels of the input and output are pinned to the same locations. For output row $i$ out of $\text{new\_h}$ rows:</span>

$$
y_{\text{src}} = i \cdot \frac{H - 1}{\text{new\_h} - 1}
$$

<span style="font-size: 14px;">(and $y_{\text{src}} = 0$ when $\text{new\_h} = 1$, to avoid dividing by zero). The same formula gives $x_{\text{src}}$ from column $j$ using $W$ and $\text{new\_w}$. This maps output index $0$ to source $0$ and output index $\text{new\_h} - 1$ to source $H - 1$ exactly, so the extreme corner pixels are preserved without interpolation.</span>

---

## <span style="font-size: 16px;">Locating the Four Neighbours</span>

<span style="font-size: 14px;">Given the fractional source coordinate $y_{\text{src}}$, the two bracketing integer rows and the vertical blend weight are:</span>

* <span style="font-size: 14px;">$y_0 = \lfloor y_{\text{src}} \rfloor$ is the upper row</span>
* <span style="font-size: 14px;">$y_1 = \min(y_0 + 1, H - 1)$ is the lower row, clamped so it never runs off the bottom edge</span>
* <span style="font-size: 14px;">$w_y = y_{\text{src}} - y_0$ is the fractional distance from $y_0$ toward $y_1$, lying in $[0, 1)$</span>

<span style="font-size: 14px;">The horizontal direction is analogous: $x_0 = \lfloor x_{\text{src}} \rfloor$, $x_1 = \min(x_0 + 1, W - 1)$, $w_x = x_{\text{src}} - x_0$. The four neighbours are then the pixels at $(y_0, x_0)$, $(y_0, x_1)$, $(y_1, x_0)$, $(y_1, x_1)$.</span>

---

## <span style="font-size: 16px;">From Two 1D Interpolations to One 2D Formula</span>

<span style="font-size: 14px;">The four-corner formula is easiest to understand as two stacked 1D linear interpolations. First interpolate along $x$ on the top row and on the bottom row separately:</span>

$$
\text{top} = (1 - w_x) I[y_0, x_0] + w_x I[y_0, x_1], \qquad \text{bot} = (1 - w_x) I[y_1, x_0] + w_x I[y_1, x_1]
$$

<span style="font-size: 14px;">This produces two intermediate values, one for each bracketing row, each itself a 1D linear blend. Then interpolate between those two along $y$:</span>

$$
\text{out} = (1 - w_y)\, \text{top} + w_y\, \text{bot}
$$

<span style="font-size: 14px;">Expanding this nested form reproduces the four-term formula exactly. The order does not matter: interpolating along $y$ first and then $x$ gives the identical result, which is why the operation is called bi-linear rather than directional. This decomposition is also how efficient implementations work, computing the two row blends with one set of weights and the final column blend with another.</span>

---

## <span style="font-size: 16px;">The Interpolation Formula</span>

<span style="font-size: 14px;">The output value is the four corner pixels weighted by the product of their distance weights:</span>

$$
\text{out}[i, j] = (1 - w_y)(1 - w_x)\, I[y_0, x_0] + (1 - w_y)\, w_x\, I[y_0, x_1] + w_y (1 - w_x)\, I[y_1, x_0] + w_y w_x\, I[y_1, x_1]
$$

<span style="font-size: 14px;">Each coefficient is a rectangle of "opposite-corner area". The weight on $I[y_0, x_0]$ (the top-left neighbour) is $(1 - w_y)(1 - w_x)$, largest when the sample point is near the top-left. The four weights always sum to $1$:</span>

$$
(1 - w_y)(1 - w_x) + (1 - w_y)w_x + w_y(1 - w_x) + w_y w_x = (1 - w_y) + w_y = 1
$$

<span style="font-size: 14px;">so the output is a true convex combination and always stays within the range of the four input values. There is no brightening, darkening, or overshoot.</span>

---

## <span style="font-size: 16px;">Worked Example (upsample a 2x2 to 3x3)</span>

<span style="font-size: 14px;">Resize $I = \begin{pmatrix} 0 & 10 \\ 20 & 30 \end{pmatrix}$ ($H = W = 2$) to $3 \times 3$ with corner alignment.</span>

<span style="font-size: 14px;">The scale factor is $(H - 1)/(\text{new\_h} - 1) = 1/2 = 0.5$, so source coordinates for output indices $0, 1, 2$ are $0.0, 0.5, 1.0$.</span>

<span style="font-size: 14px;">1. **Center pixel** $(i, j) = (1, 1)$: $y_{\text{src}} = x_{\text{src}} = 0.5$, so $y_0 = x_0 = 0$, $y_1 = x_1 = 1$, $w_y = w_x = 0.5$. All four weights equal $0.25$. Output $= 0.25(0 + 10 + 20 + 30) = 0.25 \times 60 = 15$.</span>

<span style="font-size: 14px;">2. **Top-edge midpoint** $(0, 1)$: $y_{\text{src}} = 0$, $x_{\text{src}} = 0.5$, so $w_y = 0$, $w_x = 0.5$. Only the top row contributes: $\text{out} = 0.5 \cdot I[0,0] + 0.5 \cdot I[0,1] = 0.5(0) + 0.5(10) = 5$.</span>

<span style="font-size: 14px;">3. **Corner** $(0, 0)$: $y_{\text{src}} = x_{\text{src}} = 0$, all fractional weights zero, so $\text{out} = I[0, 0] = 0$ exactly, confirming corners are preserved.</span>

<span style="font-size: 14px;">The full output is $\begin{pmatrix} 0 & 5 & 10 \\ 10 & 15 & 20 \\ 20 & 25 & 30 \end{pmatrix}$, a smoothly interpolated grid whose four corners match the original input exactly, with the new pixels filling in the linear ramps between them.</span>

---

## <span style="font-size: 16px;">A Second Example (a single off-grid point)</span>

<span style="font-size: 14px;">To exercise unequal weights, interpolate the point $y_{\text{src}} = 0.25$, $x_{\text{src}} = 0.75$ in the same $2 \times 2$ image $\begin{pmatrix} 0 & 10 \\ 20 & 30 \end{pmatrix}$. Then $y_0 = 0$, $y_1 = 1$, $w_y = 0.25$; $x_0 = 0$, $x_1 = 1$, $w_x = 0.75$. The four weights are:</span>

* <span style="font-size: 14px;">**Top-left** $(1 - 0.25)(1 - 0.75) = 0.75 \times 0.25 = 0.1875$ on $I[0,0] = 0$</span>
* <span style="font-size: 14px;">**Top-right** $(0.75)(0.75) = 0.5625$ on $I[0,1] = 10$</span>
* <span style="font-size: 14px;">**Bottom-left** $(0.25)(0.25) = 0.0625$ on $I[1,0] = 20$</span>
* <span style="font-size: 14px;">**Bottom-right** $(0.25)(0.75) = 0.1875$ on $I[1,1] = 30$</span>

<span style="font-size: 14px;">The weights sum to $0.1875 + 0.5625 + 0.0625 + 0.1875 = 1.0$. The output is $0 + 5.625 + 1.25 + 5.625 = 12.5$. The sample sits closer to the top and right, so the top-right neighbour $(10)$ dominates the blend, exactly as the largest weight $0.5625$ indicates.</span>

---

## <span style="font-size: 16px;">align_corners: the Half-Pixel Nuance</span>

<span style="font-size: 14px;">The single subtlest part of resizing is the coordinate convention, and it is the source of countless silent bugs. There are two competing schemes:</span>

* <span style="font-size: 14px;">**Corner-aligned (`align_corners=True`)**, used here: $y_{\text{src}} = i \cdot (H - 1)/(\text{new\_h} - 1)$. The extreme pixel **centers** of input and output coincide. Endpoints are exact, but the mapping treats pixels as points at integer coordinates.</span>
* <span style="font-size: 14px;">**Half-pixel centers (`align_corners=False`)**, the modern default in PyTorch and OpenCV: $y_{\text{src}} = (i + 0.5) \cdot H/\text{new\_h} - 0.5$. Each pixel is treated as a little square whose center sits at half-integer coordinates. This makes the sampling geometry consistent regardless of scale and avoids a systematic shift, but the corners are no longer pinned.</span>

<span style="font-size: 14px;">The two give different results, especially for small images and large scale changes. Mixing them up - training a model with one convention and exporting with the other - shifts every feature map by a fraction of a pixel and degrades accuracy in a way that is hard to diagnose. This problem fixes the corner-aligned convention; the half-pixel formula must not be used here.</span>

---

## <span style="font-size: 16px;">Why Clamping y1 and x1 Matters</span>

<span style="font-size: 14px;">At the bottom and right edges, $y_{\text{src}}$ can equal $H - 1$ exactly, making $y_0 = H - 1$ and $y_0 + 1 = H$, which is out of bounds. The clamp $y_1 = \min(y_0 + 1, H - 1)$ prevents an index error by reusing the last valid row. In that case $w_y = 0$ anyway (the sample sits exactly on $y_0$), so the clamped neighbour receives zero weight and the answer is unaffected. The clamp is a safety net for the boundary, not a change to the interpolation.</span>

---

## <span style="font-size: 16px;">Downsampling and Its Limits</span>

<span style="font-size: 14px;">For upsampling, bilinear interpolation reads four neighbours and the smoothing is appropriate. For **downsampling**, the same four-tap read becomes an undersample: it skips over many source pixels and only ever looks at four of them per output, so it can alias high-frequency detail (moire patterns, jagged thin lines). Proper downsampling should first low-pass filter (anti-alias) before sampling, which is why `transforms.Resize` has an `antialias` flag. Plain bilinear downsampling without that filter is fast but theoretically lossy on fine texture.</span>

---

## <span style="font-size: 16px;">Comparison with Other Resamplers</span>

<span style="font-size: 14px;">Bilinear sits in the middle of a quality-cost spectrum:</span>

* <span style="font-size: 14px;">**Nearest-neighbour** copies the single closest source pixel. Fastest and exact for integer scale factors, but produces blocky, aliased output. It is the right choice for label masks and segmentation maps, where blending class indices would be meaningless.</span>
* <span style="font-size: 14px;">**Bilinear** blends four neighbours. Smooth, cheap, differentiable; the default for photographic content and feature maps.</span>
* <span style="font-size: 14px;">**Bicubic** fits a cubic surface over sixteen neighbours, preserving sharper edges and reducing the slight blur of bilinear, at four times the read cost. Common for high-quality image enlargement.</span>
* <span style="font-size: 14px;">**Lanczos / area** use larger windowed kernels and are preferred for high-quality downscaling because they incorporate anti-aliasing.</span>

<span style="font-size: 14px;">The continuity properties differ too: nearest-neighbour is discontinuous, bilinear is continuous but has discontinuous first derivatives (visible as faint creases along pixel boundaries), and bicubic is smooth in its first derivative. For most network inputs the bilinear blur is harmless and its speed wins.</span>

---

## <span style="font-size: 16px;">Use Inside Neural Networks</span>

<span style="font-size: 14px;">Because bilinear interpolation is a fixed, differentiable linear map, it is used as a parameter-free upsampling layer throughout modern architectures. Semantic segmentation decoders (FCN, U-Net, DeepLab) upsample coarse feature maps back to input resolution with bilinear interpolation, often as the final step before the per-pixel classifier. Feature pyramid networks and many GAN generators do the same. Spatial transformer networks and `grid_sample` use the identical four-corner blend to sample feature maps at learned, continuous coordinates, with gradients flowing back to the predicted coordinates because each weight is a smooth function of $w_y$ and $w_x$.</span>

<span style="font-size: 14px;">The differentiability is the key property. Nearest-neighbour sampling has zero gradient almost everywhere with respect to the sample location, so it cannot be learned through; bilinear's weights vary continuously, making it the workhorse for any layer that must sample at non-integer positions.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Each output pixel reads exactly four source pixels and does a constant amount of arithmetic, so the total cost is $O(\text{new\_h} \cdot \text{new\_w})$, independent of the input size. Memory is the output buffer. The four reads per pixel are why bilinear is far cheaper than bicubic (which reads sixteen) yet much smoother than nearest-neighbour (which reads one). Because the operation is a fixed linear combination of inputs, gradients flow through it, so it can sit inside a network as a learnable-free upsampling layer.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using the wrong coordinate convention.** Corner-aligned and half-pixel mappings differ by up to half a pixel. Applying $i \cdot H/\text{new\_h}$ or the half-pixel formula when corner alignment is required shifts every output and fails exact-match tests. Use $i \cdot (H-1)/(\text{new\_h}-1)$ exactly.</span>
* <span style="font-size: 14px;">**Dividing by zero when an output dimension is 1.** If $\text{new\_h} = 1$, the denominator $\text{new\_h} - 1$ is zero. Special-case it to $y_{\text{src}} = 0$, sampling the top row, rather than producing NaN.</span>
* <span style="font-size: 14px;">**Forgetting to clamp the second neighbour index.** At the bottom or right edge $y_0 + 1$ can equal $H$, an out-of-bounds index. Clamp to $H - 1$; since the weight is zero there, the result is unchanged but the crash is avoided.</span>
* <span style="font-size: 14px;">**Aliasing on downsampling.** Bilinear with only four taps undersamples when shrinking by a large factor, producing moire and jagged edges. For heavy downscaling, low-pass filter first or use an area/antialiased resampler.</span>
* <span style="font-size: 14px;">**Computing the floor on a negative coordinate.** With the half-pixel convention (not used here, but a frequent cross-contamination), $y_{\text{src}}$ can go slightly negative near the top edge, and $\lfloor -0.3 \rfloor = -1$ indexes the last row in Python. Under the corner-aligned convention used here this cannot happen, but mixing conventions reintroduces it.</span>

---