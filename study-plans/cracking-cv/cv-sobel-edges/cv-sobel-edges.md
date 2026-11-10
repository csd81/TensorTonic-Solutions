# <span style="font-size: 20px;">Sobel Edge Detection</span>

<span style="font-size: 14px;">The Sobel operator detects edges by estimating the **spatial gradient** of image intensity. Edges are exactly the places where brightness changes sharply, so a large gradient marks an edge. Sobel uses two small $3 \times 3$ kernels - one for the horizontal gradient and one for the vertical - and combines them into an edge-strength map. It is one of the oldest and most widely used building blocks in classical computer vision, and it is the gradient stage inside the Canny edge detector.</span>

---

## <span style="font-size: 16px;">Edges as Image Gradients</span>

<span style="font-size: 14px;">Treat the image as a function $I(x, y)$ giving brightness at each location. Its gradient $\nabla I = (\partial I/\partial x,\ \partial I/\partial y)$ points in the direction of steepest brightness increase, and its magnitude measures how steep that increase is. A flat region has near-zero gradient; a sharp boundary between dark and light has a large gradient perpendicular to the boundary.</span>

<span style="font-size: 14px;">Because a digital image is discrete, the derivatives are approximated by finite differences. The crudest estimate of $\partial I/\partial x$ is $I[i, j+1] - I[i, j-1]$, a central difference. Sobel improves on this by also averaging over the three neighbouring rows, which suppresses the high-frequency noise that a single-line difference would otherwise amplify.</span>

---

## <span style="font-size: 16px;">The Two Sobel Kernels</span>

<span style="font-size: 14px;">The horizontal-gradient kernel $G_x$ responds to vertical edges (left-right intensity change), and the vertical-gradient kernel $G_y$ responds to horizontal edges (top-bottom change):</span>

$$
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}
$$

<span style="font-size: 14px;">Reading $G_x$: it subtracts the left column from the right column, so it measures how much brighter the right side is than the left. The center row gets weight $\pm 2$ and the outer rows $\pm 1$, giving the pixel's own row more influence than its neighbours. $G_y$ is simply $G_x$ transposed, measuring bottom-minus-top with the center column weighted most heavily.</span>

---

## <span style="font-size: 16px;">Separability and the Smoothing Built In</span>

<span style="font-size: 14px;">Each Sobel kernel factors into an outer product of a 1D **derivative** filter and a 1D **smoothing** filter:</span>

$$
G_x = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}
$$

<span style="font-size: 14px;">The row vector $[-1, 0, 1]$ is the central-difference derivative along $x$; the column vector $[1, 2, 1]$ is a small triangular smoothing (a binomial blur) along $y$. So Sobel simultaneously differentiates in one direction and blurs in the perpendicular direction. The blur is what distinguishes Sobel from a bare difference: it averages out single-pixel noise that would otherwise produce spurious edges. The $[1, 2, 1]$ weights are the second row of Pascal's triangle, a discrete approximation to a Gaussian and the reason Sobel is more noise-robust than a plain central difference.</span>

---

## <span style="font-size: 16px;">Cross-Correlation, Not Convolution</span>

<span style="font-size: 14px;">The operator is applied by sliding the kernel over the zero-padded image and computing a weighted sum at each position. This problem specifies **cross-correlation** (no kernel flip), which is what deep learning frameworks call "convolution" in their conv layers. True mathematical convolution flips the kernel both horizontally and vertically before the slide.</span>

<span style="font-size: 14px;">For the symmetric-magnitude Sobel kernels the flip only changes the sign of the gradient, not its magnitude, but the spec is explicit: correlate directly with $G_x$ and $G_y$ as written. With zero padding of width $1$ on every side, each output pixel has a full $3 \times 3$ neighbourhood, so the output stays $H \times W$.</span>

---

## <span style="font-size: 16px;">Gradient Magnitude</span>

<span style="font-size: 14px;">The two directional gradients are combined into a single edge-strength value at each pixel via the Euclidean norm:</span>

$$
M[i, j] = \sqrt{g_x[i, j]^2 + g_y[i, j]^2}
$$

<span style="font-size: 14px;">This is the length of the gradient vector $(g_x, g_y)$, invariant to edge orientation: a strong edge produces a large $M$ whether it runs horizontally, vertically, or diagonally. The gradient **direction** $\theta = \text{atan2}(g_y, g_x)$ gives the edge's orientation, measured perpendicular to the edge itself, and is used by Canny for non-maximum suppression, though this problem returns only the magnitude alongside the two signed component images.</span>

<span style="font-size: 14px;">A cheaper approximation $|g_x| + |g_y|$ (the L1 norm) is sometimes used in hardware, but it overestimates diagonal edges by up to $\sqrt{2}$. The L2 norm used here is orientation-isotropic and is the standard choice.</span>

---

## <span style="font-size: 16px;">Worked Example (a 3x3 patch)</span>

<span style="font-size: 14px;">Consider a vertical edge: the left half dark, the right half bright. Take the $3 \times 3$ neighbourhood $\begin{pmatrix} 0 & 0 & 100 \\ 0 & 0 & 100 \\ 0 & 0 & 100 \end{pmatrix}$ centered on the middle pixel.</span>

<span style="font-size: 14px;">1. **Apply $G_x$**: multiply element-wise and sum. The left column $(0, 0, 0)$ times $(-1, -2, -1)$ gives $0$; the middle column times $0$ gives $0$; the right column $(100, 100, 100)$ times $(1, 2, 1)$ gives $100 + 200 + 100 = 400$. So $g_x = 400$.</span>

<span style="font-size: 14px;">2. **Apply $G_y$**: the top row $(0, 0, 100)$ times $(-1, -2, -1)$ gives $-100$; the middle row times $0$ gives $0$; the bottom row $(0, 0, 100)$ times $(1, 2, 1)$ gives $+100$. So $g_y = -100 + 100 = 0$.</span>

<span style="font-size: 14px;">3. **Magnitude**: $M = \sqrt{400^2 + 0^2} = 400$.</span>

<span style="font-size: 14px;">The result is exactly what intuition demands: a purely vertical edge has a strong horizontal gradient $(g_x = 400)$ and zero vertical gradient $(g_y = 0)$. If the edge had been horizontal instead (dark top, bright bottom), the roles would swap: $g_x = 0$, $g_y = 400$.</span>

---

## <span style="font-size: 16px;">A Diagonal Edge</span>

<span style="font-size: 14px;">To see both components fire, take a diagonal step where the top-left is dark and the bottom-right is bright: $\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 100 \\ 0 & 100 & 100 \end{pmatrix}$.</span>

<span style="font-size: 14px;">**$G_x$ response**: right column $(0, 100, 100) \cdot (1, 2, 1) = 0 + 200 + 100 = 300$; left column is all zero; so $g_x = 300$.</span>

<span style="font-size: 14px;">**$G_y$ response**: bottom row $(0, 100, 100) \cdot (1, 2, 1) = 0 + 200 + 100 = 300$; top row is all zero; so $g_y = 300$.</span>

<span style="font-size: 14px;">**Magnitude**: $M = \sqrt{300^2 + 300^2} = 300\sqrt{2} \approx 424.26$. Both components are equal and positive, and the gradient direction $\theta = \text{atan2}(300, 300) = 45°$ points exactly perpendicular to the diagonal edge, as expected.</span>

---

## <span style="font-size: 16px;">A Flat Patch Gives Zero</span>

<span style="font-size: 14px;">For a constant patch $\begin{pmatrix} 50 & 50 & 50 \\ 50 & 50 & 50 \\ 50 & 50 & 50 \end{pmatrix}$, the $G_x$ response is $50(-1 + 1) + 50(-2 + 2) + 50(-1 + 1) = 0$, and likewise $g_y = 0$, so $M = 0$. This is the crucial property that makes Sobel an edge detector: any region of uniform brightness, no matter how bright, yields zero gradient. Only changes in intensity survive, because the kernel weights sum to zero.</span>

---

## <span style="font-size: 16px;">Why the Kernel Weights Sum to Zero</span>

<span style="font-size: 14px;">Both kernels sum to $0$: $(-1 - 2 - 1) + (1 + 2 + 1) = 0$ for $G_x$. This is required of any derivative filter. A constant added to the whole image must not change the gradient (the derivative of a constant is zero), and a zero-sum kernel guarantees exactly that: adding $c$ to every pixel adds $c \cdot \sum(\text{weights}) = 0$ to the output. The operator therefore responds to relative brightness differences, not absolute brightness, which makes it robust to uniform lighting changes.</span>

---

## <span style="font-size: 16px;">Sign of the Gradient</span>

<span style="font-size: 14px;">The component gradients are signed, and the sign carries information that the magnitude discards. A positive $g_x$ means brightness increases from left to right (a dark-to-light edge); a negative $g_x$ means the opposite (light-to-dark). In the vertical-edge example $g_x = +400$ because the patch went dark-to-light rightward; had the patch been bright on the left, $g_x = -400$.</span>

<span style="font-size: 14px;">This is why the problem returns `gx` and `gy` separately in addition to the magnitude: the signed components preserve edge polarity and direction, which are needed for orientation analysis and for downstream steps like non-maximum suppression. Taking the magnitude collapses the two signed numbers into one non-negative strength and throws away both the sign and the direction.</span>

---

## <span style="font-size: 16px;">Padding and Border Behaviour</span>

<span style="font-size: 14px;">With zero padding of width $1$, pixels on the image border have their missing neighbours treated as $0$. This creates an artificial intensity step at the image boundary, so Sobel typically reports spurious strong edges along the outermost rows and columns. Alternatives like reflect or replicate padding reduce this artifact, but this problem fixes zero padding to keep the output size at $H \times W$ and the computation deterministic. The border response is a known artifact of zero padding, not a bug, and downstream stages usually mask out the one-pixel frame.</span>

---

## <span style="font-size: 16px;">Relation to Other Edge Operators</span>

<span style="font-size: 14px;">Sobel is one of a family of first-derivative edge operators that differ mainly in their smoothing weights:</span>

* <span style="font-size: 14px;">**Prewitt** uses uniform smoothing $[1, 1, 1]$ instead of $[1, 2, 1]$, so its kernels have $\pm 1$ everywhere. It is slightly noisier than Sobel because it weights all three rows equally.</span>
* <span style="font-size: 14px;">**Scharr** uses $[3, 10, 3]$ smoothing, which is tuned for better rotational symmetry, giving more accurate gradient directions than Sobel at the same kernel size.</span>
* <span style="font-size: 14px;">**Roberts cross** uses $2 \times 2$ diagonal differences, the smallest and fastest operator but the most noise-sensitive.</span>

<span style="font-size: 14px;">All of these are first-derivative operators that peak at an edge. A different approach, the **Laplacian**, is a second-derivative operator that crosses zero at an edge. Sobel's middle weight of $2$ is the sweet spot between Prewitt's under-smoothing and heavier kernels' over-smoothing, which is why it remains the most popular default.</span>

---

## <span style="font-size: 16px;">Sobel Inside Canny and Modern Networks</span>

<span style="font-size: 14px;">The Sobel gradient is the first stage of the **Canny edge detector** (Canny, 1986). Canny computes $g_x$, $g_y$, and the magnitude exactly as here, then adds non-maximum suppression (thin the ridges to single-pixel edges using the gradient direction) and hysteresis thresholding (link strong and weak edges). Understanding Sobel is therefore a prerequisite for Canny.</span>

<span style="font-size: 14px;">In deep learning, the Sobel kernels are a classic example of what the first convolutional layer of a CNN learns on its own. Early filters in trained networks frequently resemble oriented edge detectors, and Sobel-initialized or Sobel-shaped filters appear in edge-aware loss functions and in lightweight models that hard-code gradient features. The operator also underlies image-gradient regularizers used in super-resolution and depth estimation, where penalizing differences in $g_x$ and $g_y$ encourages sharp, correctly oriented edges in the output.</span>

---

## <span style="font-size: 16px;">Complexity and Optimization</span>

<span style="font-size: 14px;">Naively, each output pixel costs nine multiply-adds per kernel, so the full operation is $O(9 \cdot H \cdot W)$ per kernel. Exploiting separability cuts this: applying the 1D $[1, 2, 1]$ and $[-1, 0, 1]$ passes in sequence costs $3 + 3 = 6$ operations per pixel instead of $9$, and the saving grows for larger kernels. Separability is the single most important optimization for any image filter that factors this way, and it generalizes to the Gaussian blur.</span>

<span style="font-size: 14px;">In practice the gradient computation is memory-bound rather than compute-bound: each output reads a small neighbourhood and does little arithmetic, so the cost is dominated by reading the image. On modern hardware the two Sobel passes are usually fused into a single kernel that loads each pixel once and accumulates both $g_x$ and $g_y$, then writes all three outputs. This is also why edge detection runs comfortably in real time even on high-resolution video.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Confusing cross-correlation with convolution.** Frameworks correlate (no flip); textbooks convolve (flip). For Sobel the flip negates $g_x$ and $g_y$, which leaves the magnitude unchanged but flips the sign of the components. Match the spec: correlate directly with the kernels as written.</span>
* <span style="font-size: 14px;">**Integer overflow and premature clipping.** Gradients are signed and can exceed the input range ($g_x = 400$ above, well past $255$). Computing on `uint8` overflows, and clipping to $[0, 255]$ before taking the magnitude destroys strong edges. Use a signed float type throughout.</span>
* <span style="font-size: 14px;">**Swapping $G_x$ and $G_y$.** $G_x$ detects vertical edges via a horizontal gradient; $G_y$ detects horizontal edges. Swapping them transposes the gradient field, which is silent for the magnitude but wrong for the direction and for the individual `gx`/`gy` outputs this problem requires.</span>
* <span style="font-size: 14px;">**Forgetting the border padding.** Without zero padding the output shrinks to $(H-2) \times (W-2)$, breaking the required $H \times W$ shape and shifting every index. Pad by exactly $1$ on all sides.</span>
* <span style="font-size: 14px;">**Using the L1 magnitude $|g_x| + |g_y|$.** The cheaper sum-of-absolutes overestimates diagonal edges by up to a factor of $\sqrt{2}$ and is not rotation-isotropic. This problem requires the Euclidean $\sqrt{g_x^2 + g_y^2}$, so the two differ on any non-axis-aligned edge.</span>

---