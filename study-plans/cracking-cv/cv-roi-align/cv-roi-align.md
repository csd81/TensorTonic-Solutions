# <span style="font-size: 20px;">ROI Align</span>

<span style="font-size: 14px;">**ROI Align** (He et al., 2017, Mask R-CNN) extracts a fixed-size feature map from an arbitrary region of interest using **bilinear interpolation at exact sub-pixel locations**, with no quantization. It replaced ROI Pool, whose two rounding steps misaligned features with the input by up to a pixel, a tolerable error for box classification but a severe one for the per-pixel masks Mask R-CNN predicts. Eliminating that misalignment was the single change that made accurate instance segmentation truly possible.</span>

---

## <span style="font-size: 16px;">The Problem with ROI Pool</span>

<span style="font-size: 14px;">ROI Pool converts a floating-point region of interest into a fixed grid of features through two quantizations. First it rounds the RoI's continuous coordinates to integer feature-map cells. Second, when dividing the rounded region into the output bins, it rounds each bin boundary to integers again. Each rounding shifts the sampled region by up to half a cell, and the two compound.</span>

<span style="font-size: 14px;">For whole-object classification this misalignment is harmless, because a class label does not depend on exact pixel positions. But Mask R-CNN predicts a spatial mask aligned to the image, and a half-pixel shift in the extracted features translates directly into a half-pixel error in the predicted mask boundary. He et al. found that fixing this alignment improved mask accuracy substantially, by 10 to 50 percent relative on strict-IoU metrics, more than any other single architectural choice in the paper.</span>

---

## <span style="font-size: 16px;">The Aligned Coordinate Convention</span>

<span style="font-size: 14px;">ROI Align with `aligned=True` (matching `torchvision.ops.roi_align`) first shifts every RoI coordinate by $-0.5$:</span>

$$
x_1' = x_1 - 0.5, \quad y_1' = y_1 - 0.5, \quad x_2' = x_2 - 0.5, \quad y_2' = y_2 - 0.5
$$

<span style="font-size: 14px;">This half-pixel shift accounts for the convention that a feature-map value at integer index $(i, j)$ represents the **center** of that cell, not its corner. Without the shift, sampling coordinates would be offset by half a cell from the true pixel centers, reintroducing exactly the kind of misalignment ROI Align exists to remove. The $-0.5$ correction is precisely the difference between torchvision's `aligned=True` and the original `aligned=False` behaviour, and it matters for pixel-accurate masks.</span>

---

## <span style="font-size: 16px;">Binning and Sampling Points</span>

<span style="font-size: 14px;">The shifted RoI is divided into an $\texttt{output\_h} \times \texttt{output\_w}$ grid of equal bins, with bin sizes:</span>

$$
\texttt{bin\_h} = \frac{y_2' - y_1'}{\texttt{output\_h}}, \qquad \texttt{bin\_w} = \frac{x_2' - x_1'}{\texttt{output\_w}}
$$

<span style="font-size: 14px;">Bin boundaries are kept as floating-point values; this is the key departure from ROI Pool, which would round them. Within each bin, $S \times S$ regularly spaced sample points are taken, where $S = \texttt{sampling\_ratio}$. For bin $(i, j)$ and sub-sample $(s_y, s_x)$ with $s_y, s_x \in \{0, \ldots, S-1\}$, the sample coordinate is:</span>

$$
y = y_1' + \texttt{bin\_h}\cdot\!\left(i + \frac{s_y + 0.5}{S}\right), \quad x = x_1' + \texttt{bin\_w}\cdot\!\left(j + \frac{s_x + 0.5}{S}\right)
$$

<span style="font-size: 14px;">The term $(s_y + 0.5)/S$ places the $S$ samples at the centers of $S$ equal sub-intervals of the bin, so they are symmetric and evenly spaced rather than clustered at an edge. The final feature for output cell $(c, i, j)$ is the average of the $S^2$ interpolated sample values, which approximates an integral of the underlying continuous feature over the bin.</span>

---

## <span style="font-size: 16px;">Bilinear Interpolation</span>

<span style="font-size: 14px;">Each sample coordinate $(y, x)$ is generally non-integer, so its value is computed by **bilinear interpolation** of the four surrounding integer grid points. With $y_0 = \lfloor y \rfloor$, $x_0 = \lfloor x \rfloor$, $y_1 = y_0 + 1$, $x_1 = x_0 + 1$, and fractional parts $\delta_y = y - y_0$, $\delta_x = x - x_0$, the interpolated value is:</span>

$$
V = (1{-}\delta_y)(1{-}\delta_x)\,f_{00} + (1{-}\delta_y)\,\delta_x\,f_{01} + \delta_y(1{-}\delta_x)\,f_{10} + \delta_y\,\delta_x\,f_{11}
$$

<span style="font-size: 14px;">where $f_{00}, f_{01}, f_{10}, f_{11}$ are the feature values at the four corners. Bilinear interpolation is what makes the operation differentiable with respect to both the features and, in principle, the box coordinates, and it is what removes the quantization: the sampled value varies smoothly as the RoI moves, instead of jumping when a coordinate crosses an integer boundary.</span>

---

## <span style="font-size: 16px;">Boundary Handling</span>

<span style="font-size: 14px;">Samples can fall outside the feature map. The convention matching torchvision is: a sample with $y < -1$, $y > H$, $x < -1$, or $x > W$ contributes exactly $0$ to its bin's average. Samples within that slightly-enlarged margin but outside the valid index range are handled by **clamping** each coordinate to $[0, H-1] \times [0, W-1]$ before the bilinear blend, so the interpolation reads only in-bounds feature values.</span>

<span style="font-size: 14px;">The asymmetry matters: the $[-1, H]$ acceptance window is wider than the valid $[0, H-1]$ index range, so a sample just past the border still contributes a clamped (edge) value rather than zero, while a sample far outside contributes zero. Getting this window wrong shifts which samples count and breaks parity with the reference on RoIs near the image edge.</span>

---

## Worked Example (one bin, $S = 1$)

<span style="font-size: 14px;">Take a single-channel $2 \times 2$ feature map $f = [[1, 2], [3, 4]]$, RoI $[0, 0, 2, 2]$, $\texttt{output\_h} = \texttt{output\_w} = 1$, $S = 1$.</span>

<span style="font-size: 14px;">**Shift coords**: $x_1' = -0.5$, $y_1' = -0.5$, $x_2' = 1.5$, $y_2' = 1.5$.</span>

<span style="font-size: 14px;">**Bin size**: $\texttt{bin\_h} = (1.5 - (-0.5))/1 = 2$, $\texttt{bin\_w} = 2$. One bin, $(i, j) = (0, 0)$.</span>

<span style="font-size: 14px;">**Sample** with $S = 1$, $s_y = s_x = 0$: $y = -0.5 + 2\cdot(0 + 0.5/1) = -0.5 + 1 = 0.5$, $x = 0.5$.</span>

<span style="font-size: 14px;">**Interpolate** at $(0.5, 0.5)$: $y_0 = x_0 = 0$, $\delta_y = \delta_x = 0.5$. $V = 0.25\cdot f_{00} + 0.25\cdot f_{01} + 0.25\cdot f_{10} + 0.25\cdot f_{11} = 0.25(1 + 2 + 3 + 4) = 2.5$.</span>

<span style="font-size: 14px;">**Average** the single sample: $\texttt{out}[0,0,0] = 2.5$. The result is the exact center-of-feature-map value, demonstrating that the $-0.5$ shift plus the $(s+0.5)/S$ sampling lands precisely at the geometric center. ROI Pool, by contrast, would have rounded the RoI and bin to integers and returned a single corner value such as $1$ or $4$, off by up to $1.5$ from the aligned answer.</span>

---

## <span style="font-size: 16px;">Averaging as Integration</span>

<span style="font-size: 14px;">Averaging the $S^2$ samples within a bin is a discrete approximation of integrating the continuous feature field over the bin's area, then dividing by the area. This is the natural generalization of average pooling to sub-pixel geometry: where average pooling sums the integer cells inside a region, ROI Align averages bilinearly-interpolated samples at sub-pixel positions, so it works for bins whose boundaries do not align with the integer grid.</span>

<span style="font-size: 14px;">Because each sample is itself a smooth bilinear blend and the sample positions are fixed fractions of the bin, the whole bin value is a smooth, piecewise-linear function of the RoI coordinates. This smoothness is what gives ROI Align well-defined gradients with respect to the input features (used during training) and avoids the discontinuous jumps that plagued ROI Pool, where moving the box by a sub-pixel amount could suddenly change which integer cell a bin pooled from.</span>

---

## <span style="font-size: 16px;">Why Sub-Pixel Sampling for Masks</span>

<span style="font-size: 14px;">A segmentation mask is a dense per-pixel prediction, so the alignment between the extracted RoI features and the original image pixels must be exact. ROI Pool's quantization breaks this: the pooled feature for a given output cell may actually correspond to a region shifted by up to a pixel in the input. When the mask head then upsamples and thresholds these features into a binary mask, the boundary lands in the wrong place.</span>

<span style="font-size: 14px;">ROI Align removes both quantizations, so the feature at output cell $(i, j)$ corresponds to the precise continuous location in the input that the binning geometry dictates. The mask head therefore learns and predicts on pixel-aligned features, and the predicted boundary matches the true object boundary. He et al. emphasized that this alignment, not the mask head architecture, was the decisive factor in Mask R-CNN's accuracy.</span>

---

## <span style="font-size: 16px;">The Sampling Ratio</span>

<span style="font-size: 14px;">The sampling ratio $S$ controls how finely each bin is sampled. With $S = 2$, four points per bin are interpolated and averaged; with $S = 1$, a single center point is used. Averaging multiple samples approximates the average of the continuous feature over the bin, which is more faithful than a single point, especially for large bins where one sample undersamples the region.</span>

<span style="font-size: 14px;">He et al. found the result robust to the exact value of $S$, reporting that even $S = 1$ already captures most of the benefit over ROI Pool, because the gain comes mainly from removing quantization rather than from dense sampling. Torchvision's default of $S = 0$ means adaptive (choose $S$ to roughly match the bin size), but a fixed positive $S$ as here is deterministic and easy to verify.</span>

---

## <span style="font-size: 16px;">Bilinear Interpolation in Detail</span>

<span style="font-size: 14px;">Bilinear interpolation can be read as two stages of linear interpolation. First interpolate along $x$ on the top row and the bottom row separately, producing two intermediate values; then interpolate between those two along $y$. The weights $(1-\delta_y)(1-\delta_x)$, $(1-\delta_y)\delta_x$, $\delta_y(1-\delta_x)$, $\delta_y\delta_x$ are the products of the per-axis linear weights, and they always sum to one, so the interpolated value is a convex combination of the four corners and stays within their range.</span>

<span style="font-size: 14px;">A useful sanity check: when the sample lands exactly on an integer grid point, three of the four weights are zero and the interpolation returns that grid value exactly, with no blending. When it lands at the center of the four cells, all weights are $0.25$ and the value is their plain mean, as the worked example showed. These limiting cases make bilinear interpolation easy to verify by hand on small inputs.</span>

---

## <span style="font-size: 16px;">Relationship to ROI Pool and ROI Warp</span>

<span style="font-size: 14px;">The progression of region-feature extractors clarifies what each fixes. ROI Pool (Fast R-CNN) quantizes twice and pools the max over integer cells. ROI Warp (an intermediate proposed in instance-segmentation work) interpolates but still quantizes the RoI to the grid once. ROI Align removes all quantization, interpolates at exact sub-pixel sample points, and averages. He et al. ablated these and showed ROI Align's gain over ROI Pool was almost entirely due to removing the quantization, not the switch from max to average, confirming alignment as the operative factor.</span>

<span style="font-size: 14px;">This lineage is why ROI Align is now the default region extractor in nearly all two-stage detectors and instance segmenters, including Mask R-CNN, Cascade R-CNN, and their descendants. The same operation also feeds keypoint heads, where sub-pixel alignment is just as critical as it is for masks.</span>

---

## <span style="font-size: 16px;">Output Shape and Cost</span>

<span style="font-size: 14px;">For $N$ RoIs, $C$ channels, and output grid $\texttt{output\_h} \times \texttt{output\_w}$, the result is a 4D tensor of shape $(N, C, \texttt{output\_h}, \texttt{output\_w})$, each value rounded to 4 decimals. The work per RoI is $C \cdot \texttt{output\_h} \cdot \texttt{output\_w} \cdot S^2$ bilinear interpolations, each reading four feature values. This is cheap relative to the backbone and fully parallelizable, since every output cell, channel, and RoI is independent.</span>

<span style="font-size: 14px;">Boxes here are already in feature-map coordinates, so no `spatial_scale` factor is applied. In a full detector the RoI comes from image space and is divided by the feature stride first; that scaling is folded out of this problem so the geometry is exact and the focus stays on the alignment and interpolation mechanics.</span>

<span style="font-size: 14px;">When combined with FPN, ROI Align is applied on whichever pyramid level a RoI is assigned to by its size, and each level carries its own spatial scale. The alignment logic is identical at every level; only the stride used to map the box into that level's coordinates changes. This composability is why ROI Align slots cleanly into the FPN-based detectors that dominate practice, extracting a uniform fixed-size feature from objects of any scale.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the $-0.5$ coordinate shift.** Omitting the shift implements the older `aligned=False` behaviour, which reintroduces a half-pixel misalignment. Every sampled coordinate is then offset, masks shift, and the output no longer matches `roi_align(aligned=True)`. The shift applies to all four RoI coordinates before binning.</span>
* <span style="font-size: 14px;">**Quantizing bin boundaries or RoI coordinates.** Rounding the floating-point bin edges or the RoI corners turns ROI Align back into ROI Pool. The entire point is that boundaries and sample points stay continuous; any rounding before interpolation defeats the operation.</span>
* <span style="font-size: 14px;">**Mishandling the out-of-bounds rule.** Samples with $y < -1$, $y > H$, $x < -1$, or $x > W$ contribute zero, while samples just inside that window are clamped to the valid index range before blending. Treating all out-of-range samples as zero, or clamping everything without the zero region, produces wrong values on edge RoIs.</span>
* <span style="font-size: 14px;">**Wrong sample placement within a bin.** The $S$ samples sit at $(s + 0.5)/S$ fractions of the bin, i.e. at sub-interval centers, not at the bin edges or at $s/S$. Placing them incorrectly skews every bin's average and silently degrades accuracy without any crash.</span>

---