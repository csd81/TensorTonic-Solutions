# <span style="font-size: 20px;">Gaussian Blur 2D</span>

<span style="font-size: 14px;">Gaussian blur smooths an image by replacing each pixel with a **weighted average of its neighbourhood**, where the weights follow a 2D Gaussian (bell curve) centered on that pixel. It is the single most important low-pass filter in computer vision: it removes high-frequency noise and fine detail while preserving large-scale structure, and it is the smoothing step that precedes edge detection, downsampling, and scale-space construction in detectors like SIFT.</span>

---

## <span style="font-size: 16px;">Why Gaussian Weights</span>

<span style="font-size: 14px;">A naive blur is the box filter, which averages all pixels in a window with equal weight. It works but produces visible artifacts: hard ringing at edges and a square, directional bias. The Gaussian fixes both. Its weights decay smoothly with distance from the center, so nearby pixels contribute more than distant ones, and it is **rotationally symmetric**, so it blurs equally in all directions.</span>

<span style="font-size: 14px;">The Gaussian is also the only filter that is smooth, strictly positive, and separable while having no overshoot - it never introduces values outside the local range. These properties make it the canonical smoothing kernel and the basis of scale-space theory, where blurring an image by successively larger Gaussians constructs a continuous family of progressively coarser representations.</span>

---

## <span style="font-size: 16px;">The 1D Gaussian Kernel</span>

<span style="font-size: 14px;">For an odd kernel size $k$, the 1D kernel samples a Gaussian centered at the middle index $c = (k - 1)/2$:</span>

$$
g_{1d}[i] = \exp\left(-\frac{(i - c)^2}{2\sigma^2}\right), \quad i \in \{0, 1, \ldots, k-1\}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$c = (k-1)/2$ is the center index, so the peak weight $1$ sits on the pixel being filtered</span>
* <span style="font-size: 14px;">$\sigma$ controls the spread: larger $\sigma$ gives a wider, flatter bell and stronger blur; smaller $\sigma$ concentrates weight on the center and blurs less</span>
* <span style="font-size: 14px;">the kernel must be **normalized** so its entries sum to $1$; otherwise the average is biased and the overall image brightness would change</span>

<span style="font-size: 14px;">The kernel size $k$ and the spread $\sigma$ are related but independent: $k$ sets how many taps are computed, and $\sigma$ sets how fast they decay. A common rule of thumb is $k \approx 6\sigma + 1$ so the kernel captures the bulk of the bell, but here both are given.</span>

---

## <span style="font-size: 16px;">From 1D to 2D: The Outer Product</span>

<span style="font-size: 14px;">The 2D Gaussian kernel is built as the **outer product** of the 1D kernel with itself:</span>

$$
G_{2d}[i, j] = g_{1d}[i] \cdot g_{1d}[j]
$$

<span style="font-size: 14px;">then normalized so $\sum_{i,j} G_{2d}[i, j] = 1$. This works because the 2D Gaussian factors exactly: $\exp(-(x^2 + y^2)/2\sigma^2) = \exp(-x^2/2\sigma^2)\cdot\exp(-y^2/2\sigma^2)$. The exponent of a sum becomes a product of exponentials, so the 2D bell is the product of two 1D bells. This factorization is precisely what makes the Gaussian **separable**, the key to its efficiency.</span>

---

## <span style="font-size: 16px;">Separability and the Cost Reduction</span>

<span style="font-size: 14px;">Because $G_{2d} = g_{1d} \otimes g_{1d}$, blurring with the full 2D kernel gives the identical result as two cheaper 1D passes: first convolve every row with $g_{1d}$, then convolve every column of that result with $g_{1d}$ (or columns then rows). The order does not matter.</span>

<span style="font-size: 14px;">The cost difference is dramatic. A direct 2D convolution touches $k^2$ kernel entries per output pixel, costing $O(k^2 \cdot H \cdot W)$. Two 1D passes touch $k$ entries each, costing $O(2k \cdot H \cdot W) = O(k \cdot H \cdot W)$. The speedup is $k^2 / 2k = k/2$. For a $9 \times 9$ kernel that is a $4.5\times$ reduction; for a $25 \times 25$ kernel it is over $12\times$; the advantage grows without bound as the kernel widens. This is why production blur implementations always use the separable form even though the problem defines the 2D kernel explicitly. Mathematically the two routes give the same numbers up to floating-point rounding; only the operation count differs. The separable form also has better cache behaviour, since each 1D pass streams contiguously along one axis rather than gathering a 2D block per pixel.</span>

---

## <span style="font-size: 16px;">Worked Example (build a 3x3 kernel)</span>

<span style="font-size: 14px;">Take $k = 3$, $\sigma = 1$, so $c = (3-1)/2 = 1$. The 1D kernel before normalization:</span>

<span style="font-size: 14px;">1. $g_{1d}[0] = \exp(-(0-1)^2/2) = \exp(-0.5) = 0.6065$</span>

<span style="font-size: 14px;">2. $g_{1d}[1] = \exp(-(1-1)^2/2) = \exp(0) = 1.0000$</span>

<span style="font-size: 14px;">3. $g_{1d}[2] = \exp(-(2-1)^2/2) = \exp(-0.5) = 0.6065$</span>

<span style="font-size: 14px;">The 2D kernel is the outer product. The center entry is $1.0 \times 1.0 = 1.0$, the edge-midpoints are $1.0 \times 0.6065 = 0.6065$, and the corners are $0.6065 \times 0.6065 = 0.3679$:</span>

$$
\begin{pmatrix} 0.3679 & 0.6065 & 0.3679 \\ 0.6065 & 1.0000 & 0.6065 \\ 0.3679 & 0.6065 & 0.3679 \end{pmatrix}
$$

<span style="font-size: 14px;">The sum of all nine entries is $4(0.3679) + 4(0.6065) + 1.0 = 1.4716 + 2.4260 + 1.0 = 4.8976$. Dividing every entry by $4.8976$ normalizes the kernel so it sums to $1$. The normalized center weight becomes $1.0/4.8976 = 0.2042$, the edges $0.1238$, and the corners $0.0751$. These nine weights now sum to $1$, so applying them to a neighbourhood produces a genuine weighted average that preserves overall brightness.</span>

---

## <span style="font-size: 16px;">A Blur Applied to a Patch</span>

<span style="font-size: 14px;">Apply the normalized $3 \times 3$ kernel (center $0.2042$, edges $0.1238$, corners $0.0751$) to a small step edge. Take the neighbourhood $\begin{pmatrix} 0 & 0 & 100 \\ 0 & 0 & 100 \\ 0 & 0 & 100 \end{pmatrix}$ and compute the output at the center pixel.</span>

<span style="font-size: 14px;">Only the right column is nonzero, with values $100$. Its three weights are the top-right corner $0.0751$, the right edge $0.1238$, and the bottom-right corner $0.0751$. The weighted sum is $100 \times (0.0751 + 0.1238 + 0.0751) = 100 \times 0.2740 = 27.40$.</span>

<span style="font-size: 14px;">The center pixel, originally $0$, becomes $27.40$: the blur has pulled some of the bright right side leftward, softening the hard edge into a gradual ramp. This smoothing of sharp transitions is exactly the visible effect of a Gaussian, and it is why blurring before edge detection both suppresses noise and slightly widens true edges.</span>

---

## <span style="font-size: 16px;">Applying the Kernel</span>

<span style="font-size: 14px;">With the normalized $G_{2d}$, each output pixel is the cross-correlation (no flip, and the Gaussian is symmetric so it would not matter) of the kernel with the $k \times k$ neighbourhood of the input. Zero padding of width $(k-1)/2$ on every side keeps the output shape equal to the input.</span>

<span style="font-size: 14px;">For the $3 \times 3$ kernel above centered on a pixel whose neighbourhood is all $100$, the output is $100 \times \sum(\text{weights}) = 100 \times 1 = 100$: a flat region passes through unchanged, because the normalized weights sum to one. Only where intensity varies does the blur average values together and soften the transition, which is the entire purpose of a low-pass filter: leave smooth regions alone and attenuate sharp changes.</span>

---

## <span style="font-size: 16px;">The Continuous Gaussian Behind the Kernel</span>

<span style="font-size: 14px;">The discrete kernel is a sampling of the continuous 2D Gaussian function:</span>

$$
G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
$$

<span style="font-size: 14px;">The leading $1/(2\pi\sigma^2)$ is the normalization that makes the continuous function integrate to $1$. For a discrete kernel we drop this constant and instead normalize by the sum of the sampled weights, because sampling and truncating to a finite window changes the total. That is why this problem computes the raw exponentials first and divides by their sum afterward: it is the discrete analogue of the continuous normalization, corrected for the finite kernel.</span>

<span style="font-size: 14px;">The distance term $x^2 + y^2$ is the squared radial distance from the center, which is what gives the kernel its circular symmetry. Two pixels equidistant from the center receive identical weight regardless of direction, so the blur has no preferred orientation, unlike a box filter whose square shape biases the horizontal and vertical axes.</span>

---

## <span style="font-size: 16px;">Effect of Sigma</span>

<span style="font-size: 14px;">The single parameter $\sigma$ governs the strength of the blur. As $\sigma \to 0$, the bell collapses onto the center pixel: the kernel approaches the identity and there is no blur. As $\sigma$ grows, weight spreads outward, the center weight drops, and the image is smoothed more aggressively. In the frequency domain, the Fourier transform of a Gaussian is another Gaussian, so a wide spatial Gaussian (large $\sigma$) is a narrow frequency-domain Gaussian, which is exactly a stronger low-pass filter that removes more high-frequency detail.</span>

<span style="font-size: 14px;">A practical consequence: if $\sigma$ is large relative to $k$, the kernel is **truncated** too aggressively and the tails are clipped, so normalization compensates but the shape is no longer a faithful Gaussian. Conversely if $k$ is much larger than $6\sigma$, the outer taps are essentially zero and only waste computation without changing the result.</span>

---

## <span style="font-size: 16px;">Comparison with the Box Filter</span>

<span style="font-size: 14px;">It is worth contrasting the Gaussian with the equal-weight box filter, the other common smoother:</span>

* <span style="font-size: 14px;">**Box filter** weights every pixel in the window equally at $1/k^2$. It is the cheapest blur and is separable too, but its frequency response has large side lobes, producing ringing and a blocky look. Repeated box filters approximate a Gaussian (by the central limit theorem), which some fast implementations exploit.</span>
* <span style="font-size: 14px;">**Gaussian filter** weights by distance, with a smooth frequency response and no ringing. It is the gold standard when quality matters and is the only smoother whose repeated application stays in the same family (a Gaussian blurred by a Gaussian is a wider Gaussian).</span>

<span style="font-size: 14px;">A third option, the **bilateral filter**, weights by both spatial distance and intensity difference so it smooths flat regions while preserving edges. It is edge-aware but non-separable and far more expensive. The plain Gaussian smooths edges along with noise, which is acceptable and often desirable before gradient computation.</span>

---

## <span style="font-size: 16px;">Composition Property</span>

<span style="font-size: 14px;">Gaussians compose in a uniquely clean way: applying a Gaussian blur with spread $\sigma_1$ followed by one with $\sigma_2$ is exactly equivalent to a single Gaussian blur with $\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}$. The variances add. This semigroup property is the mathematical foundation of scale-space: an image can be progressively coarsened by repeatedly applying small Gaussians, and the cumulative blur is itself a well-defined Gaussian. No other common smoothing kernel has this property, which is another reason the Gaussian is singled out in computer vision theory.</span>

---

## <span style="font-size: 16px;">Where Gaussian Blur Is Used</span>

<span style="font-size: 14px;">Gaussian smoothing precedes nearly every gradient-based operation. Canny edge detection blurs first so that noise does not create spurious edges. The Laplacian-of-Gaussian and Difference-of-Gaussian detectors, and the SIFT scale space, are built entirely from Gaussians at increasing $\sigma$. Anti-aliasing before downsampling uses a Gaussian to remove the high frequencies that would otherwise alias into the smaller image. In deep learning, Gaussian blur is a standard data-augmentation transform and is the smoothing used in techniques like feature-map blurring for anti-aliased strided convolutions.</span>

<span style="font-size: 14px;">A particularly instructive use is the **Difference of Gaussians**: subtracting an image blurred at $\sigma_1$ from the same image blurred at a larger $\sigma_2$ isolates the spatial frequencies between the two scales, approximating a band-pass filter and the Laplacian-of-Gaussian. SIFT detects keypoints as extrema of a Difference-of-Gaussians pyramid, so the normalized Gaussian kernel built here is the literal first ingredient of one of the most influential feature detectors in vision.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to normalize.** An unnormalized Gaussian kernel sums to more than $1$ (about $4.9$ in the example), so the blurred image is uniformly brightened by that factor and likely saturates. Always divide by the kernel sum so the weights total exactly $1$.</span>
* <span style="font-size: 14px;">**Normalizing the 1D kernels separately vs the 2D kernel.** Normalizing each 1D kernel to sum to $1$ and then taking the outer product yields a 2D kernel that already sums to $1$, which is equivalent. But normalizing the 2D kernel after the outer product (as the spec says) is the safe, unambiguous route; mixing partial normalizations can leave a residual scale factor.</span>
* <span style="font-size: 14px;">**Wrong center index.** The center must be $c = (k-1)/2$. Using $k/2$ (integer division) for an odd $k$ happens to match, but off-by-one errors here shift the bell off-center and make the blur asymmetric, introducing a directional smear.</span>
* <span style="font-size: 14px;">**Insufficient padding.** The padding width must be exactly $(k-1)/2$ on every side to keep the output at $H \times W$. Too little padding shrinks the output and shifts indices; treating out-of-bounds neighbours as nonzero changes the border values.</span>
* <span style="font-size: 14px;">**Darkened borders from zero padding.** Because the kernel near a border overlaps zero-padded pixels that are included in the weighted sum but contribute $0$, border outputs are pulled toward zero and appear darkened. Some implementations renormalize the kernel by only the weights that fall inside the image to avoid this; with plain zero padding the darkening is expected and consistent with the spec.</span>

---