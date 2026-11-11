## The Problem ROI Pooling Solves

In object detection, a region proposal network suggests many regions of interest (ROIs) of different sizes. But the classifier that follows needs fixed-size inputs.

**The challenge:**
- ROI 1: 100x150 pixels
- ROI 2: 50x50 pixels
- ROI 3: 200x80 pixels
- Classifier expects: 7x7 pixels

ROI pooling converts arbitrary-sized regions into fixed-size feature maps.

---

## How ROI Pooling Works

Given:
- A feature map (the CNN output)
- A list of ROIs (bounding boxes in feature map coordinates)
- Target output size (e.g., 7x7)

For each ROI:
1. Divide the ROI into a grid of output_size x output_size bins
2. Apply max pooling within each bin
3. Result: one output_size x output_size feature map per ROI

---

## Bin Calculation

For an ROI with coordinates $(x_1, y_1, x_2, y_2)$:

ROI dimensions:
- roi_h = y2 - y1
- roi_w = x2 - x1

For bin (i, j) in the output grid:

$$
h_{start} = y_1 + \left\lfloor \frac{i \cdot \text{roi}_h}{\text{output size}} \right\rfloor
$$

$$
h_{end} = y_1 + \left\lfloor \frac{(i+1) \cdot \text{roi}_h}{\text{output size}} \right\rfloor
$$

Similarly for width. Then take the max value in that bin region.

---

## Numerical Example

**Feature map (6x8):**

```
 1  2  3  4  5  6  7  8
 2  3  4  5  6  7  8  9
 3  4  5  6  7  8  9 10
 4  5  6  7  8  9 10 11
 5  6  7  8  9 10 11 12
 6  7  8  9 10 11 12 13
```

**ROI:** [1, 1, 5, 5] (x1=1, y1=1, x2=5, y2=5)
**Output size:** 2x2

ROI dimensions: roi_h = 4, roi_w = 4

**Bin (0, 0):**
- h_start = 1 + floor(0 * 4 / 2) = 1
- h_end = 1 + floor(1 * 4 / 2) = 3
- w_start = 1 + floor(0 * 4 / 2) = 1
- w_end = 1 + floor(1 * 4 / 2) = 3
- Region: rows 1-2, cols 1-2 = [[3,4], [4,5]]
- Max: 5

**Bin (0, 1):**
- w_start = 1 + floor(1 * 4 / 2) = 3
- w_end = 1 + floor(2 * 4 / 2) = 5
- Region: rows 1-2, cols 3-4 = [[5,6], [6,7]]
- Max: 7

**Bin (1, 0):**
- Region: rows 3-4, cols 1-2 = [[5,6], [6,7]]
- Max: 7

**Bin (1, 1):**
- Region: rows 3-4, cols 3-4 = [[7,8], [8,9]]
- Max: 9

**Output:**

```
5  7
7  9
```

---

## Quantization Issues

ROI pooling has quantization errors:

**Problem 1: ROI coordinates are quantized**
- Original ROI might be [1.3, 2.7, 5.8, 6.2]
- Quantized to [1, 2, 5, 6]
- Loses sub-pixel precision

**Problem 2: Bin boundaries are quantized**
- Bins may have different numbers of pixels
- Some bins might be empty (if ROI is tiny)

These errors hurt accuracy, especially for small objects.

---

## Handling Edge Cases

**Empty bins:**
- If h_end == h_start, set h_end = h_start + 1
- Ensures every bin has at least one pixel

**ROI outside feature map:**
- Clip coordinates to valid range
- Or pad feature map with zeros

**Very small ROIs:**
- May have fewer pixels than output bins
- Some bins share pixels

---

## ROI Align: The Improvement

ROI Align (from Mask R-CNN) fixes quantization issues:

**Key differences:**
- No rounding of ROI coordinates
- Uses bilinear interpolation instead of max pooling
- Samples at fixed sub-pixel locations within each bin

**Result:**
- More accurate localization
- Essential for pixel-level tasks (segmentation)
- Standard in modern detectors

---

## Multi-Channel ROI Pooling

For a feature map with C channels:
- Apply ROI pooling to each channel independently
- Output: output_size x output_size x C per ROI

The pooling operation is the same; just repeated for each channel.

---

## Where ROI Pooling Is Used

**Faster R-CNN:**
- RPN proposes ~300 ROIs
- ROI pooling extracts 7x7 features from each
- Followed by FC layers for classification and refinement

**Fast R-CNN:**
- ROIs come from external proposals (Selective Search)
- Same ROI pooling mechanism

**Feature Pyramid Networks:**
- ROIs assigned to different FPN levels based on size
- Larger ROIs use coarser feature maps

---

## ROI Pooling vs. Spatial Pyramid Pooling

**ROI Pooling:**
- Multiple ROIs from one image
- Each ROI produces one fixed-size output
- ROIs can overlap

**Spatial Pyramid Pooling (SPP):**
- Multiple pooling sizes for one region
- Concatenates outputs from different grid sizes
- Creates multi-scale representation

Both solve the "variable size to fixed size" problem but in different ways.

---

## Implementation Strategy

For each ROI:
1. Extract ROI bounds
2. Compute bin boundaries (with floor division)
3. For each bin:
   a. Handle empty bin case
   b. Extract the region from feature map
   c. Compute max value
   d. Store in output
4. Return output tensor

When multiple ROIs are processed, the result is a list (or batch) of fixed-size feature maps.