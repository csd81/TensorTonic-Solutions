## What Is Interpolation?

Interpolation estimates values at positions between known data points. In image processing, it is used when:
- Resizing images (scaling up or down)
- Rotating images
- Correcting lens distortion
- Any geometric transformation

The input is a grid of known pixel values. The output needs values at non-integer positions.

---

## The Bilinear Idea

Bilinear interpolation uses the four nearest pixels to estimate a value at any point. It performs:
1. Linear interpolation in the x-direction (horizontal)
2. Linear interpolation in the y-direction (vertical)

The result is a smooth blend of the four neighbors.

---

## The Formula

For a point $(x, y)$ where $(x_0, y_0)$ is the floor (top-left neighbor):

Let $dx = x - x_0$ and $dy = y - y_0$ be the fractional parts.

The four neighbors are:
- Top-left: $I[y_0][x_0]$
- Top-right: $I[y_0][x_1]$ where $x_1 = x_0 + 1$
- Bottom-left: $I[y_1][x_0]$ where $y_1 = y_0 + 1$
- Bottom-right: $I[y_1][x_1]$

The interpolated value:

$$
v = I[y_0][x_0](1-dx)(1-dy) + I[y_0][x_1] \cdot dx(1-dy)
$$

$$
+ I[y_1][x_0](1-dx) \cdot dy + I[y_1][x_1] \cdot dx \cdot dy
$$

---

## Understanding the Weights

Each neighbor contributes based on distance:

**Top-left weight:** $(1-dx)(1-dy)$
- Large when point is near top-left

**Top-right weight:** $dx(1-dy)$
- Large when point is near top-right

**Bottom-left weight:** $(1-dx) \cdot dy$
- Large when point is near bottom-left

**Bottom-right weight:** $dx \cdot dy$
- Large when point is near bottom-right

The weights always sum to 1.

---

## Numerical Example

**Known pixels (2x2 grid):**

10  30
20  40

**Query point:** (0.25, 0.75)
- x = 0.25, so dx = 0.25
- y = 0.75, so dy = 0.75
- x0 = 0, y0 = 0

**Neighbors:**
- Top-left (0,0): 10
- Top-right (0,1): 30
- Bottom-left (1,0): 20
- Bottom-right (1,1): 40

**Weights:**
- Top-left: (1-0.25)(1-0.75) = 0.75 * 0.25 = 0.1875
- Top-right: 0.25 * 0.25 = 0.0625
- Bottom-left: 0.75 * 0.75 = 0.5625
- Bottom-right: 0.25 * 0.75 = 0.1875

**Result:**
10 * 0.1875 + 30 * 0.0625 + 20 * 0.5625 + 40 * 0.1875
= 1.875 + 1.875 + 11.25 + 7.5
= 22.5

The point is closer to the bottom-left (20), and the result (22.5) reflects that.

---

## Image Resizing with Bilinear

To resize from (H, W) to (H_new, W_new):

For each output pixel (i, j):
1. Map to source coordinates:

$$
\text{src}_y = i \cdot \frac{H - 1}{H_{new} - 1}
$$

$$
\text{src}_x = j \cdot \frac{W - 1}{W_{new} - 1}
$$

2. Apply bilinear interpolation at (src_y, src_x)
3. Store result in output[i][j]

---

## Coordinate Mapping Details

**Why (H-1)/(H_new-1)?**

This maps corners to corners:
- Output (0, 0) maps to input (0, 0)
- Output (H_new-1, W_new-1) maps to input (H-1, W-1)

**Alternative: (H)/(H_new)**
- Maps center of first pixel to center of first pixel
- Slightly different alignment
- Both are valid; the formula above is more common

**Edge case: H_new = 1**
- Division by zero
- Handle specially: src = 0

---

## Boundary Handling

When the interpolation point is at the image edge:

**Problem:** x_1 = x_0 + 1 or y_1 = y_0 + 1 may be out of bounds

**Solution:** Clamp to valid range
- x_1 = min(x_0 + 1, W - 1)
- y_1 = min(y_0 + 1, H - 1)

This effectively replicates the edge pixel.

---

## Bilinear vs. Nearest Neighbor

**Nearest neighbor:**
- Fast (no multiplication)
- Blocky results
- Preserves sharp edges (but with staircase artifacts)

**Bilinear:**
- Smooth results
- Slight blurring of sharp edges
- Standard for most applications

**When to use nearest neighbor:**
- Pixel art (intentionally blocky)
- Binary images
- When speed is critical

---

## Bilinear vs. Bicubic

**Bilinear:**
- Uses 4 neighbors
- Fast
- Slight blurring

**Bicubic:**
- Uses 16 neighbors (4x4 grid)
- Smoother, sharper results
- Slower (16 weights to compute)
- Better for high-quality resizing

For most applications, bilinear is good enough. Bicubic is used when quality matters (photo editing, print).

---

## Separability

Bilinear interpolation is separable:
1. Interpolate horizontally to get two values
2. Interpolate vertically between those two values

This is mathematically equivalent to the 4-point formula and can be slightly faster.

**Horizontal first:**
- top = I[y0][x0] * (1-dx) + I[y0][x1] * dx
- bottom = I[y1][x0] * (1-dx) + I[y1][x1] * dx
- result = top * (1-dy) + bottom * dy

---

## Applications

- **Image scaling**: enlarging or shrinking images
- **Texture mapping**: mapping 2D textures onto 3D surfaces
- **Image registration**: aligning images from different sources
- **Video processing**: frame interpolation, stabilization
- **ROI pooling**: extracting features at arbitrary positions