# <span style="font-size: 20px;">Pairwise IoU</span>

<span style="font-size: 14px;">**Intersection-over-Union (IoU)**, also called the **Jaccard index**, measures how well two axis-aligned bounding boxes overlap, returning a value in $[0, 1]$. It is the single most important geometric primitive in object detection: it defines the matching rule between predictions and ground truth, drives Non-Maximum Suppression, and is the basis of the mean Average Precision (mAP) metric used to rank every detector on COCO and PASCAL VOC. Almost every downstream primitive in a detector, from anchor assignment to evaluation, is built on top of a pairwise IoU computation.</span>

---

## <span style="font-size: 16px;">What IoU Measures</span>

<span style="font-size: 14px;">Given two boxes, IoU is the ratio of the area they share to the total area they jointly cover. A value of $0$ means the boxes are disjoint, $1$ means they coincide exactly, and intermediate values quantify partial overlap. Because it is a ratio of areas, IoU is **scale-invariant**: a 10-pixel offset matters far more for a tiny box than for a large one, which is exactly the behaviour a detector needs when objects appear at many sizes.</span>

<span style="font-size: 14px;">The Jaccard index of two sets $A$ and $B$ is $|A \cap B| / |A \cup B|$. Treating each box as the set of points it encloses, IoU is precisely the Jaccard index restricted to axis-aligned rectangles, which is what makes the intersection and union computable in closed form rather than by counting pixels.</span>

<span style="font-size: 14px;">A **pairwise IoU matrix** generalizes this to two sets of boxes. Given $N$ boxes in set $A$ and $M$ boxes in set $B$, the result is an $N \times M$ matrix where entry $(i, j)$ is the IoU between box $i$ of $A$ and box $j$ of $B$. This is the workhorse used to assign anchors to ground-truth objects, to compute recall during evaluation, and as the suppression test inside NMS. The reference implementation matches `torchvision.ops.box_iou`.</span>

---

## <span style="font-size: 16px;">The Geometry of Intersection</span>

<span style="font-size: 14px;">Boxes are stored in `xyxy` format as $[x_1, y_1, x_2, y_2]$ with $x_1 \leq x_2$ and $y_1 \leq y_2$, where $(x_1, y_1)$ is the top-left corner and $(x_2, y_2)$ the bottom-right. The intersection of two axis-aligned boxes is itself an axis-aligned box whose corners are found by taking the inner extremes:</span>

$$
x_1^{\cap} = \max(b_1.x_1, b_2.x_1), \quad y_1^{\cap} = \max(b_1.y_1, b_2.y_1)
$$

$$
x_2^{\cap} = \min(b_1.x_2, b_2.x_2), \quad y_2^{\cap} = \min(b_1.y_2, b_2.y_2)
$$

<span style="font-size: 14px;">The intersection corners take the maximum of the two left/top edges and the minimum of the two right/bottom edges. Intuitively, the overlapping region cannot start before the rightmost left edge and cannot end after the leftmost right edge. The same logic applies independently to the $x$ and $y$ axes because the boxes are axis-aligned, which is what lets the 2D problem factor into two 1D interval-overlap problems.</span>

<span style="font-size: 14px;">If the boxes do not overlap on an axis, the inner extremes cross over and produce a negative width or height, which must be clamped to zero:</span>

$$
\text{inter} = \max(0,\, x_2^{\cap} - x_1^{\cap}) \cdot \max(0,\, y_2^{\cap} - y_1^{\cap})
$$

<span style="font-size: 14px;">The $\max(0, \cdot)$ clamps are not optional decoration: without them, two disjoint boxes would yield a spurious **positive** intersection because a negative width times a negative height is positive. This is the most common bug in hand-rolled IoU and it produces overlaps for boxes that share no pixels.</span>

---

## <span style="font-size: 16px;">Union and the Final Ratio</span>

<span style="font-size: 14px;">By the inclusion-exclusion principle, the area of the union equals the sum of the two box areas minus their intersection, since the overlapping region would otherwise be counted twice:</span>

$$
\text{area}_k = (x_2^k - x_1^k)\,(y_2^k - y_1^k)
$$

$$
\text{union} = \text{area}_1 + \text{area}_2 - \text{inter}
$$

$$
\text{IoU} = \frac{\text{inter}}{\text{area}_1 + \text{area}_2 - \text{inter}}
$$

<span style="font-size: 14px;">Computing the union as $\text{area}_1 + \text{area}_2 - \text{inter}$ rather than enumerating union pixels is exact and runs in $O(1)$ per pair. Each individual area is just width times height; with the convention $x_1 \leq x_2$ and $y_1 \leq y_2$ these are guaranteed non-negative, so no clamp is needed on the areas themselves, only on the intersection. Note that $0 \leq \text{inter} \leq \min(\text{area}_1, \text{area}_2) \leq \text{union}$, which guarantees the ratio always lands in $[0, 1]$.</span>

---

## <span style="font-size: 16px;">Decomposition into 1D Overlaps</span>

<span style="font-size: 14px;">Because the boxes are axis-aligned, the 2D intersection separates cleanly into two independent 1D interval-overlap problems, one per axis. On the $x$-axis the two boxes occupy intervals $[x_1^1, x_2^1]$ and $[x_1^2, x_2^2]$; their overlap length is $\max(0, \min(x_2^1, x_2^2) - \max(x_1^1, x_1^2))$. The same expression on the $y$-axis gives the overlap height. The intersection area is the product of the two overlap lengths.</span>

<span style="font-size: 14px;">This factorization is why the corner formulas use max for the lower bounds and min for the upper bounds: each axis is solved by the 1D rule independently. It also explains why the clamp must be applied per axis before multiplying. If the boxes overlap on $x$ but not on $y$, the $y$ overlap is zero and the whole intersection is zero, which is correct, but only because each axis is clamped on its own. Clamping the product instead of the two factors would lose this behaviour and admit the negative-times-negative bug.</span>

---

## <span style="font-size: 16px;">Properties of the IoU Function</span>

<span style="font-size: 14px;">Beyond symmetry, IoU has several properties worth knowing when reasoning about detector behaviour:</span>

* <span style="font-size: 14px;">**Reflexivity.** $\text{IoU}(a, a) = 1$ for any non-degenerate box, since intersection and union are both the box itself.</span>
* <span style="font-size: 14px;">**Containment cap.** If box $a$ is fully inside box $b$, then $\text{inter} = \text{area}_a$ and $\text{IoU} = \text{area}_a / \text{area}_b$. A small box centered inside a large one can have a low IoU even with perfect containment, which is why containment alone does not imply a match.</span>
* <span style="font-size: 14px;">**Not a metric.** IoU is a similarity, not a distance. The complement $1 - \text{IoU}$ is the Jaccard distance and does satisfy the triangle inequality, which is occasionally used in clustering boxes.</span>
* <span style="font-size: 14px;">**Monotonicity under shrinking overlap.** Sliding two equal boxes apart strictly decreases IoU until they become disjoint, where it pins at $0$ and stops providing gradient.</span>

---

## <span style="font-size: 16px;">Why IoU and Not Other Distances</span>

<span style="font-size: 14px;">A detector could in principle match boxes by corner distance or center distance, but those metrics are not scale-invariant and carry arbitrary units (pixels). IoU is dimensionless and bounded, which gives it three properties that make it the standard:</span>

* <span style="font-size: 14px;">**Bounded and interpretable.** The range $[0, 1]$ lets a single threshold (commonly $0.5$ for VOC, or a sweep $0.5{:}0.05{:}0.95$ for COCO mAP) define what counts as a correct detection across all object scales.</span>
* <span style="font-size: 14px;">**Scale-invariant.** A small absolute shift is penalized heavily for small objects and lightly for large ones, matching human notions of localization quality.</span>
* <span style="font-size: 14px;">**Symmetric.** $\text{IoU}(a, b) = \text{IoU}(b, a)$, so the matrix is well-defined regardless of which set is rows and which is columns.</span>

<span style="font-size: 14px;">These properties are also why IoU appears in the matching step of nearly every detector: Faster R-CNN assigns an anchor as positive if its max IoU with any ground-truth box exceeds $0.7$ and negative if below $0.3$; SSD uses a $0.5$ threshold; and COCO evaluation thresholds detections at ten IoU levels to reward precise localization.</span>

---

## <span style="font-size: 16px;">Role in mAP Evaluation</span>

<span style="font-size: 14px;">During evaluation, predicted boxes are ranked by confidence and greedily matched to ground-truth boxes. A prediction is a true positive only if its IoU with an unmatched ground-truth box of the same class meets the threshold; otherwise it is a false positive. Sweeping the confidence threshold traces a precision-recall curve, and the area under it is the Average Precision for that class and IoU level.</span>

<span style="font-size: 14px;">PASCAL VOC reports AP at a single IoU of $0.5$. COCO reports the primary metric as the mean of AP over IoU thresholds from $0.5$ to $0.95$ in steps of $0.05$, written as AP$@[.5:.95]$. The higher thresholds reward tight localization, which is why a well-built pairwise IoU routine that matches `box_iou` to the last decimal is foundational: an off-by-one in the area convention shifts the entire metric.</span>

---

## Worked Example ($2 \times 2$ matrix)

<span style="font-size: 14px;">Let set $A = \{[0, 0, 2, 2],\ [1, 1, 3, 3]\}$ and set $B = \{[1, 1, 3, 3],\ [4, 4, 6, 6]\}$. Every box here is $2 \times 2$, so each area equals $4$.</span>

<span style="font-size: 14px;">**Entry (0, 0)**: $A_0 = [0,0,2,2]$ vs $B_0 = [1,1,3,3]$. Intersection corners $x_1^{\cap} = \max(0,1) = 1$, $y_1^{\cap} = 1$, $x_2^{\cap} = \min(2,3) = 2$, $y_2^{\cap} = 2$. So $\text{inter} = (2-1)(2-1) = 1$, $\text{union} = 4 + 4 - 1 = 7$, $\text{IoU} = 1/7 \approx 0.1429$.</span>

<span style="font-size: 14px;">**Entry (0, 1)**: $A_0 = [0,0,2,2]$ vs $B_1 = [4,4,6,6]$. Here $x_1^{\cap} = \max(0,4) = 4$ and $x_2^{\cap} = \min(2,6) = 2$, so $x_2^{\cap} - x_1^{\cap} = -2 \to \max(0, -2) = 0$. The boxes are disjoint, $\text{inter} = 0$, $\text{IoU} = 0$.</span>

<span style="font-size: 14px;">**Entry (1, 0)**: $A_1 = [1,1,3,3]$ vs $B_0 = [1,1,3,3]$. Identical boxes, $\text{inter} = 4$, $\text{union} = 4 + 4 - 4 = 4$, $\text{IoU} = 1.0$.</span>

<span style="font-size: 14px;">**Entry (1, 1)**: $A_1 = [1,1,3,3]$ vs $B_1 = [4,4,6,6]$. Disjoint, $\text{IoU} = 0$. The final matrix rounded to 4 decimals is $[[0.1429,\ 0.0],\ [1.0,\ 0.0]]$.</span>

---

## <span style="font-size: 16px;">Vectorized Computation</span>

<span style="font-size: 14px;">For the full matrix, broadcasting computes all $N \times M$ pairs at once without a Python loop. Areas are precomputed once per set as length-$N$ and length-$M$ vectors. The intersection top-left corner is an elementwise max over broadcast box arrays of shape $(N, 1, 2)$ and $(1, M, 2)$, giving $(N, M, 2)$; the bottom-right uses min the same way.</span>

<span style="font-size: 14px;">Width and height are clamped with a single $\max(0, \cdot)$ before multiplying to get an $(N, M)$ intersection. The union uses $\text{area}_A[:, \text{None}] + \text{area}_B[\text{None}, :] - \text{inter}$ via broadcasting. This is exactly the structure `torchvision.ops.box_iou` uses, and it runs in $O(NM)$ time and $O(NM)$ memory. For very large box sets the $O(NM)$ memory of the full matrix can dominate, which is why production NMS kernels fuse the IoU test into the suppression loop rather than materializing the entire matrix.</span>

---

## <span style="font-size: 16px;">IoU Loss Variants</span>

<span style="font-size: 14px;">Plain IoU is a poor regression loss because it is flat (zero gradient) whenever two boxes do not overlap, so it gives no signal to pull far-apart predictions together. A family of differentiable extensions fixes this:</span>

* <span style="font-size: 14px;">**GIoU** (Rezatofighi et al., 2019) subtracts a term proportional to the area of the smallest enclosing box not covered by the union, giving a non-zero gradient even when boxes are disjoint.</span>
* <span style="font-size: 14px;">**DIoU** adds a penalty on the normalized distance between box centers, which converges faster than GIoU.</span>
* <span style="font-size: 14px;">**CIoU** further adds an aspect-ratio consistency term and is the default regression loss in YOLOv4 and later.</span>

<span style="font-size: 14px;">All of these reduce to standard IoU when boxes overlap perfectly, so the pairwise IoU primitive remains the core building block they extend.</span>

---

## <span style="font-size: 16px;">Where the Matrix Is Consumed</span>

<span style="font-size: 14px;">The pairwise IoU matrix is rarely the final answer; it is an intermediate that other primitives reduce. Three reduction patterns dominate object detection:</span>

* <span style="font-size: 14px;">**Anchor assignment.** With rows as anchors and columns as ground-truth boxes, a row-wise max picks each anchor's best-matching object, and thresholds on that max value label the anchor positive, negative, or ignored. A column-wise argmax additionally guarantees every ground-truth box is matched to at least one anchor, which is the standard Faster R-CNN rule.</span>
* <span style="font-size: 14px;">**NMS.** A single-set self-IoU matrix (rows and columns both the detection set) feeds the suppression test: after picking the top-scoring box, every other box with IoU above the threshold against it is removed. The matrix can be recomputed lazily rather than materialized in full.</span>
* <span style="font-size: 14px;">**Recall and mAP.** With rows as predictions and columns as ground truth, greedy matching by descending confidence reads off true and false positives, from which the precision-recall curve and Average Precision are computed.</span>

<span style="font-size: 14px;">In every case the correctness of the downstream metric depends on the IoU matrix being computed exactly, which is why matching `torchvision.ops.box_iou` to four decimals is the bar for this primitive.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the zero-clamp on width and height.** Disjoint boxes produce a negative intersection width and height; multiplied together they give a positive area and a fictitious overlap. Always apply $\max(0, x_2^{\cap} - x_1^{\cap})$ and $\max(0, y_2^{\cap} - y_1^{\cap})$ independently before multiplying.</span>
* <span style="font-size: 14px;">**Off-by-one from pixel-vs-coordinate conventions.** Some legacy codebases (the original VOC devkit) treat boxes as inclusive pixel grids and use width $= x_2 - x_1 + 1$. The continuous-coordinate convention here and in torchvision uses $x_2 - x_1$. Mixing the two shifts every IoU and silently changes which detections pass threshold.</span>
* <span style="font-size: 14px;">**Division by zero on degenerate boxes.** A box with $x_1 = x_2$ or $y_1 = y_2$ has zero area; if both boxes are degenerate and disjoint, the union is $0$ and IoU becomes $0/0$. Guard with a small epsilon in the denominator or define IoU as $0$ when the union is $0$.</span>
* <span style="font-size: 14px;">**Using IoU as a loss directly without care.** Because plain IoU has zero gradient for non-overlapping boxes, it cannot drive far-apart predictions toward the target. Regression heads use smooth-L1 on encoded deltas, or the GIoU/DIoU/CIoU variants that add a penalty term for the non-overlapping case.</span>

---