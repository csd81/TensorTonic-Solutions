## What Are Morphological Operations?

Morphological operations are nonlinear filters for binary (black and white) images. They modify shapes based on their local structure using a small template called a structuring element.

The two fundamental operations are:
- **Erosion**: shrinks foreground regions
- **Dilation**: expands foreground regions

---

## Binary Image Convention

In binary images:
- **1 (foreground)**: the objects of interest
- **0 (background)**: everything else

Morphological operations modify the boundary between foreground and background.

---

## Structuring Element

The structuring element (kernel) defines the neighborhood shape:

**Square 3x3 (most common):**

1 1 1
1 1 1
1 1 1

**Cross 3x3:**

0 1 0
1 1 1
0 1 0

**Disk (circular, various sizes):**
- Approximates a circle
- Used for size-independent operations

The structuring element is centered on each pixel, and the operation examines all positions where the element has a 1.

---

## Erosion

Erosion shrinks foreground regions. A foreground pixel remains foreground only if ALL positions in the structuring element overlap with foreground.

$$
\text{eroded}[i][j] = \begin{cases} 1 & \text{if all kernel positions match foreground} \\ 0 & \text{otherwise} \end{cases}
$$

**Effects of erosion:**
- Removes small objects (smaller than the element)
- Removes thin protrusions
- Separates touching objects
- Shrinks all objects

---

## Dilation

Dilation expands foreground regions. A pixel becomes foreground if ANY position in the structuring element overlaps with foreground.

$$
\text{dilated}[i][j] = \begin{cases} 1 & \text{if any kernel position matches foreground} \\ 0 & \text{otherwise} \end{cases}
$$

**Effects of dilation:**
- Fills small holes
- Connects nearby objects
- Expands all objects
- Smooths convex boundaries

---

## Step-by-Step Erosion Example

**Input image:**

0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0

**3x3 square structuring element**

**Center pixel (2,2):**
- All 9 neighbors are 1
- Output: 1

**Pixel (1,1):**
- Top-left neighbor is 0
- Output: 0

**Result:**

0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0

The 3x3 square shrunk to a single pixel.

---

## Step-by-Step Dilation Example

**Input image:**

0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0

**3x3 square structuring element**

**Pixel (1,1):**
- One neighbor (2,2) is 1
- Output: 1

**Result:**

0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0

The single pixel expanded to a 3x3 square.

---

## Opening and Closing

**Opening (erosion followed by dilation):**
- Removes small bright spots
- Separates thin connections
- Smooths contours without changing size much

**Closing (dilation followed by erosion):**
- Fills small dark holes
- Connects nearby objects
- Smooths contours without changing size much

These are more useful than erosion or dilation alone because they remove noise without significantly changing object size.

---

## Boundary Detection

The boundary can be extracted using morphology:

$$
\text{boundary} = \text{image} - \text{eroded(image)}
$$

This gives the outline of all foreground objects.

---

## Applications

**Noise removal:**
- Opening removes salt noise (small white spots)
- Closing removes pepper noise (small black spots)

**Object separation:**
- Erosion can separate touching objects
- Useful before counting or measuring

**Hole filling:**
- Dilation followed by logical AND with original
- Or use specialized fill algorithm

**Skeletonization:**
- Repeated thinning operations
- Reduces shapes to single-pixel-wide skeletons

**Text processing:**
- Connect broken characters (dilation)
- Separate touching characters (erosion)

---

## Padding

Morphological operations need to handle boundaries:

**Zero padding:**
- Treat outside pixels as 0 (background)
- Objects touching borders will erode
- Standard for most applications

**Replication:**
- Copy border pixels
- Preserves objects at boundaries

This problem uses zero padding.

---

## Multi-Channel Images

For grayscale images, morphology generalizes:
- **Grayscale erosion**: local minimum
- **Grayscale dilation**: local maximum

This is useful for noise removal and contrast enhancement on non-binary images.