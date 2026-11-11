## What is Non-Maximum Suppression?

Non-Maximum Suppression (NMS) is a technique used to eliminate redundant overlapping detections, keeping only the best one. When object detectors produce multiple bounding boxes for the same object, NMS filters them down to a single detection by suppressing boxes that are not local maxima in terms of confidence scores.

---

## Why NMS is Necessary

**Detector behavior**: Object detectors like YOLO, Faster R-CNN, and SSD generate thousands of candidate bounding boxes with confidence scores.

**Overlapping predictions**: Multiple boxes often cover the same object with varying degrees of accuracy.

**Sliding window legacy**: Even modern detectors can propose multiple boxes for one object due to anchor boxes or grid-based predictions.

**Clean output requirement**: Downstream tasks need exactly one detection per object for counting, tracking, or analysis.

---

## The Core Idea

**Goal**: For each group of overlapping detections of the same object, keep only the one with the highest confidence score.

**Method**: Iteratively select the highest-scoring box, remove all boxes that overlap significantly with it, and repeat.

**Key parameter**: IoU threshold - defines what counts as "significant overlap"

---

## Intersection over Union (IoU)

IoU measures the overlap between two bounding boxes:

$$
\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}
$$

**Properties**:
- Range: [0, 1]
- IoU = 0: No overlap
- IoU = 1: Perfect overlap (identical boxes)
- IoU = 0.5: Boxes overlap about half their combined area

**Calculation**:

$$
\text{Area}(A \cap B) = \max(0, x_2^{int} - x_1^{int}) \times \max(0, y_2^{int} - y_1^{int})
$$

Where intersection coordinates:
- $x_1^{int} = \max(x_1^A, x_1^B)$
- $y_1^{int} = \max(y_1^A, y_1^B)$
- $x_2^{int} = \min(x_2^A, x_2^B)$
- $y_2^{int} = \min(y_2^A, y_2^B)$

$$
\text{Area}(A \cup B) = \text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)
$$

---

## The NMS Algorithm

**Input**: 
- List of bounding boxes with coordinates
- Confidence scores for each box
- IoU threshold (typically 0.5)

**Process**:
1. Sort all boxes by confidence score (descending)
2. Select the box with highest score, add to output
3. Remove all boxes with IoU > threshold with the selected box
4. Repeat steps 2-3 until no boxes remain

**Output**: Filtered list of non-overlapping boxes

---

## Worked Example

**Detections** (box coordinates and scores):
- Box A: score=0.9
- Box B: score=0.75, IoU with A = 0.8
- Box C: score=0.6, IoU with A = 0.2, IoU with B = 0.3
- Box D: score=0.5, IoU with A = 0.7, IoU with B = 0.9, IoU with C = 0.1

**NMS with threshold = 0.5**:

**Iteration 1**:
- Select Box A (highest score 0.9)
- Check overlaps: B has IoU 0.8 > 0.5 (suppress), D has IoU 0.7 > 0.5 (suppress)
- Remove B and D
- Remaining: C

**Iteration 2**:
- Select Box C (score 0.6)
- No remaining boxes to compare
- Done

**Output**: [Box A, Box C]

Box A and C represent different objects (low IoU). Boxes B and D were duplicates of A.

---

## IoU Threshold Selection

**High threshold (e.g., 0.7)**:
- More lenient - keeps more boxes
- Risk: May keep duplicate detections
- Use when: Objects are densely packed

**Low threshold (e.g., 0.3)**:
- More aggressive - removes more boxes
- Risk: May suppress valid nearby detections
- Use when: Objects are well-separated

**Common default**: 0.5 balances false positives and missed detections

---

## Class-Aware NMS

When detecting multiple object classes:

**Option 1 - Per-class NMS**:
- Apply NMS separately within each class
- A car box does not suppress a person box regardless of overlap
- Most common approach

**Option 2 - Class-agnostic NMS**:
- Apply NMS across all classes
- High-confidence box suppresses any overlapping low-confidence box
- Used when classes are mutually exclusive at a location

---

## Soft-NMS

Standard NMS completely removes overlapping boxes. Soft-NMS reduces their scores instead:

**Linear decay**:

$$
s_i = \begin{cases} s_i & \text{if IoU}(M, b_i) < \text{threshold} \\ s_i (1 - \text{IoU}(M, b_i)) & \text{otherwise} \end{cases}
$$

**Gaussian decay**:

$$
s_i = s_i \cdot e^{-\frac{\text{IoU}(M, b_i)^2}{\sigma}}
$$

Where $M$ is the selected box and $b_i$ is another box.

**Benefit**: Better handles cases where objects are genuinely close together

---

## Computational Complexity

**Naive implementation**: O(N²) where N is number of boxes
- Compare every pair of boxes

**Optimized implementations**:
- Sort by score: O(N log N)
- Spatial indexing can reduce comparisons
- GPU-accelerated versions for real-time applications

**Practical consideration**: N is typically small after confidence thresholding (hundreds, not thousands)

---

## Bounding Box Representation

Common formats:

**Corner format**: (x1, y1, x2, y2) - top-left and bottom-right corners

**Center format**: (cx, cy, w, h) - center coordinates and dimensions

**YOLO format**: (cx, cy, w, h) normalized to [0, 1] relative to image size

**Conversion required**: IoU calculation typically uses corner format

---

## Edge Cases

**No boxes**: Return empty list

**Single box**: Return that box (no suppression needed)

**All boxes suppressed**: Possible if one high-confidence box overlaps all others

**Ties in confidence**: Order may affect results; typically handled by stable sort

---

## Limitations of Standard NMS

**Greedy selection**: May not find optimal global solution

**Hard threshold**: Binary decision to keep or remove

**Occlusion handling**: Struggles when objects genuinely overlap

**Speed**: Can be bottleneck for real-time systems with many detections

---

## NMS Variants

**Weighted NMS**: Averages coordinates of merged boxes weighted by confidence

**Cluster-NMS**: Groups boxes into clusters before suppression

**Matrix NMS**: Parallelizable version using matrix operations

**DIoU-NMS**: Uses Distance-IoU instead of standard IoU, considers center distance

---

## Integration in Detection Pipelines

**Typical flow**:
1. Detector produces raw predictions (thousands of boxes)
2. Confidence thresholding removes low-score boxes
3. NMS removes duplicate detections
4. Final output: clean set of detections

**When applied**: Post-processing step after model inference

---

## Where Non-Maximum Suppression Shows Up

- **Object Detection**: YOLO, SSD, Faster R-CNN all require NMS post-processing

- **Face Detection**: Eliminating multiple detections of the same face

- **Pedestrian Detection**: Autonomous vehicles detecting people

- **Edge Detection**: Canny edge detector uses NMS to thin edges

- **Feature Detection**: Harris corners, SIFT keypoints use local NMS

- **Text Detection**: OCR systems detecting text regions

- **Medical Imaging**: Detecting lesions, tumors, or anatomical structures

- **Pose Estimation**: Selecting best keypoint candidates

- **Tracking**: Initializing tracks from multiple detections

- **Instance Segmentation**: Selecting best mask proposals
