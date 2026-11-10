# <span style="font-size: 20px;">Non-Maximum Suppression</span>

<span style="font-size: 14px;">**Non-Maximum Suppression (NMS)** is the greedy post-processing step that collapses a dense set of overlapping detections into a clean set of one box per object. Every anchor-based detector emits many overlapping boxes for the same object, and NMS is the deduplication rule that keeps the single most confident box and discards its near-duplicates. It is a fixed, non-learned algorithm present at the end of R-CNN, Faster R-CNN, SSD, YOLO, and RetinaNet pipelines.</span>

---

## <span style="font-size: 16px;">Why Suppression Is Needed</span>

<span style="font-size: 14px;">A detector scores tens of thousands of anchors, and many neighbouring anchors fire on the same object because they all overlap it well enough to pass the objectness threshold. After decoding, these produce a cluster of highly overlapping boxes around each true object, all with high confidence. Returning all of them would inflate the false-positive count and ruin precision. NMS reduces each cluster to its single best representative.</span>

<span style="font-size: 14px;">The core assumption is that boxes overlapping heavily with a higher-scoring box are redundant detections of the same object, not separate objects. The IoU threshold encodes how much overlap counts as redundant. This assumption is usually correct but breaks in crowded scenes where two distinct objects genuinely overlap, which is the motivation for Soft-NMS and learned alternatives.</span>

<span style="font-size: 14px;">NMS is also the reason detectors can afford to predict densely. Because suppression cleans up duplicates afterward, the training objective can encourage many anchors near an object to fire, which improves recall, without paying a precision penalty at inference. The dense-prediction-then-suppress pattern is a deliberate division of labour: the network maximizes recall, and NMS recovers precision.</span>

---

## <span style="font-size: 16px;">The Greedy Algorithm</span>

<span style="font-size: 14px;">NMS processes boxes in descending order of score and greedily commits the top box, then suppresses everything too similar to it:</span>

<span style="font-size: 14px;">1. **Sort** all $N$ box indices by score in descending order. Ties are broken by lower original index first, so the result is deterministic.</span>

<span style="font-size: 14px;">2. **Pop** the head $i$ of the sorted list (the highest remaining score) and append $i$ to the `keep` list.</span>

<span style="font-size: 14px;">3. **Suppress** every remaining index $j$ for which $\text{IoU}(b_i, b_j) > \texttt{iou\_threshold}$. The comparison is strictly greater, so a box exactly at the threshold survives.</span>

<span style="font-size: 14px;">4. **Repeat** from step 2 with the surviving indices until the list is empty.</span>

<span style="font-size: 14px;">The output is `keep`, the list of integer indices in the order they were picked, which is score-descending. The reference behaviour matches `torchvision.ops.nms`.</span>

---

## <span style="font-size: 16px;">The IoU Test</span>

<span style="font-size: 14px;">The suppression decision uses standard axis-aligned IoU between the kept box $b_i$ and each candidate $b_j$. The intersection corners are the inner extremes, the width and height are zero-clamped, and the union is computed by inclusion-exclusion:</span>

$$
\text{inter} = \max(0, \min(x_2^i, x_2^j) - \max(x_1^i, x_1^j)) \cdot \max(0, \min(y_2^i, y_2^j) - \max(y_1^i, y_1^j))
$$

$$
\text{IoU} = \frac{\text{inter}}{\text{area}_i + \text{area}_j - \text{inter}}
$$

<span style="font-size: 14px;">The strict inequality $\text{IoU}(b_i, b_j) > \texttt{iou\_threshold}$ is a deliberate convention: a box whose overlap exactly equals the threshold is kept, matching torchvision. This edge case rarely fires on real floats but matters for reproducibility on synthetic tests with clean integer boxes.</span>

---

## <span style="font-size: 16px;">Lazy versus Precomputed IoU</span>

<span style="font-size: 14px;">There are two ways to supply the IoU values the algorithm needs. The lazy approach computes IoU on demand, only between the just-kept box and the still-surviving candidates, which avoids work for pairs that will never be compared because one box was already suppressed. The precomputed approach builds the full $N \times N$ self-IoU matrix once, then reads entries during the loop, which trades memory for the ability to run the comparisons in parallel.</span>

<span style="font-size: 14px;">On CPU with modest box counts the lazy approach is usually faster and lighter. On GPU the precomputed matrix wins because the $O(N^2)$ comparisons map cleanly onto parallel hardware and the sequential loop becomes a cheap bitmask reduction over the matrix. Either way the result is identical; only the time and memory profile differ.</span>

---

## <span style="font-size: 16px;">Why Greedy and Score-First</span>

<span style="font-size: 14px;">Processing highest-score-first encodes a simple principle: the most confident detection in a cluster is the best estimate of the object, so it should be the survivor and the reference against which others are judged. Once a box is kept, it can never be removed, which makes the algorithm a single forward pass with no backtracking.</span>

<span style="font-size: 14px;">Greedy NMS is not globally optimal; it does not solve any explicit optimization. But it is fast, deterministic, and works well empirically, which is why it has remained the default for nearly a decade. A box suppressed by a kept box is gone even if it would have kept other boxes had it survived, so suppression is one-directional and the kept set is exactly the boxes that are local score maxima within their IoU neighbourhood.</span>

---

## Worked Example ($\texttt{iou\_threshold} = 0.5$)

<span style="font-size: 14px;">Consider four boxes with scores. Box 0 $= [0,0,10,10]$ score $0.9$; box 1 $= [1,1,11,11]$ score $0.8$; box 2 $= [20,20,30,30]$ score $0.7$; box 3 $= [21,21,31,31]$ score $0.6$.</span>

<span style="font-size: 14px;">**Sort by score**: order is $[0, 1, 2, 3]$.</span>

<span style="font-size: 14px;">**Pick 0.** Compute IoU(0, 1): intersection $[1,1,10,10]$ area $9 \times 9 = 81$, union $100 + 100 - 81 = 119$, IoU $\approx 0.68 > 0.5$, so suppress 1. IoU(0, 2) and IoU(0, 3) are $0$ (disjoint), so 2 and 3 survive. `keep` $= [0]$.</span>

<span style="font-size: 14px;">**Pick 2.** IoU(2, 3): intersection $[21,21,30,30]$ area $9 \times 9 = 81$, union $119$, IoU $\approx 0.68 > 0.5$, suppress 3. `keep` $= [0, 2]$.</span>

<span style="font-size: 14px;">**List empty.** Final result `keep` $= [0, 2]$: one box per object cluster, in score-descending pick order.</span>

<span style="font-size: 14px;">The example shows the two regimes NMS handles. Within a cluster (boxes 0 and 1, or 2 and 3) the high IoU triggers suppression and only the top box survives. Across clusters (box 0 versus box 2) the zero IoU means no interaction, so both clusters keep their representative. Note that box 1 is never used as a reference: once suppressed by box 0 it is removed entirely, so it cannot suppress or save any other box, which is the one-directional nature of greedy suppression in action.</span>

---

## <span style="font-size: 16px;">The Suppression Graph View</span>

<span style="font-size: 14px;">It helps to picture the boxes as nodes of a graph with an edge between any two boxes whose IoU exceeds the threshold. Greedy NMS then visits nodes in descending score order, and when it commits a node it deletes that node and all of its still-present neighbours. The kept set is the set of nodes that were never deleted by a higher-scoring neighbour.</span>

<span style="font-size: 14px;">This view clarifies a subtle point: the kept set depends on processing order, not just on the graph structure. A box can be saved purely because the higher-scoring box that would have suppressed it was itself suppressed first by an even higher-scoring box. Greedy NMS therefore does not compute a maximum independent set or any clean graph property; it computes a specific order-dependent traversal. That order-dependence is why deterministic tie-breaking matters: two valid sort orders of equal-scoring boxes can yield different kept sets.</span>

---

## <span style="font-size: 16px;">Choosing the IoU Threshold</span>

<span style="font-size: 14px;">The IoU threshold trades precision against recall. A high threshold (near $0.7$) suppresses only very heavily overlapping boxes, so it keeps more detections and is forgiving in crowded scenes but admits more duplicates. A low threshold (near $0.3$) suppresses aggressively, removing duplicates cleanly but risking deletion of genuinely separate nearby objects.</span>

<span style="font-size: 14px;">Common defaults sit at $0.5$ to $0.7$. The right value depends on the dataset's object density: pedestrian and face detection in crowds favour higher thresholds or Soft-NMS, while sparse scenes tolerate lower ones. Because the threshold is applied at inference only, it can be tuned on a validation set without retraining, making it one of the cheapest knobs for trading precision and recall in a deployed detector.</span>

---

## <span style="font-size: 16px;">Class-Aware NMS</span>

<span style="font-size: 14px;">In multi-class detection, NMS must not suppress a dog box just because it overlaps a more confident cat box; the two are different objects. The standard solution is **per-class NMS**: run the algorithm independently within each class so suppression only happens between boxes of the same label. A common vectorized trick (used by torchvision's `batched_nms`) offsets each box's coordinates by a large per-class constant before a single NMS call, which makes boxes of different classes effectively non-overlapping and so never suppress each other.</span>

<span style="font-size: 14px;">This detail is easy to miss and produces a subtle bug: a single global NMS over all classes will silently delete correct detections of one class that happen to sit near a higher-scoring detection of another class, lowering recall in exactly the multi-object scenes that matter most.</span>

---

## <span style="font-size: 16px;">Complexity and Practical Cost</span>

<span style="font-size: 14px;">The naive implementation is $O(N^2)$ in the worst case: each kept box is compared against all remaining boxes, and with $K$ kept boxes the total work is $O(NK)$, bounded by $O(N^2)$. Sorting adds $O(N \log N)$. For the tens of thousands of boxes a detector emits, NMS is non-trivial and is often the latency bottleneck after the backbone, especially on CPU where the sequential loop cannot be parallelized.</span>

<span style="font-size: 14px;">Two standard mitigations precede NMS. First, a **score threshold** drops low-confidence boxes before sorting, often cutting the count by an order of magnitude. Second, a **top-k pre-filter** keeps only the highest-scoring few thousand boxes per image. GPU NMS kernels then compute the pairwise IoU matrix in parallel and resolve suppression with a bitmask, turning the sequential greedy loop into a near-constant-time reduction in practice.</span>

---

## <span style="font-size: 16px;">Variants of the Reduction Rule</span>

<span style="font-size: 14px;">Beyond the score-versus-recall threshold tuning, several variants change what NMS reduces over rather than how it reduces:</span>

* <span style="font-size: 14px;">**Class-agnostic NMS** ignores labels entirely and keeps one box per spatial region, used when only localization matters or when a downstream stage assigns labels.</span>
* <span style="font-size: 14px;">**Weighted NMS** averages the coordinates of all boxes in a suppressed cluster, weighted by score, rather than taking the single top box, which can sharpen localization.</span>
* <span style="font-size: 14px;">**Matrix NMS** (used in SOLOv2) approximates the sequential decay with a single parallel matrix operation, removing the iterative loop for mask-based detectors.</span>

<span style="font-size: 14px;">All of these share the same IoU primitive and the same score ordering; they differ only in the rule applied once an overlap is detected. Plain greedy NMS, which simply deletes the lower-scoring overlapper, remains the baseline against which they are measured.</span>

---

## <span style="font-size: 16px;">Limitations and Successors</span>

<span style="font-size: 14px;">Hard NMS has a structural weakness in crowded scenes: when two real objects overlap more than the IoU threshold, the lower-scoring object's box is deleted even though it is a correct detection, capping recall. Lowering the threshold to keep such boxes instead admits more duplicates, so no single threshold is ideal. This precision-recall tension is exactly what later methods address:</span>

* <span style="font-size: 14px;">**Soft-NMS** (Bodla et al., 2017) decays neighbour scores by a function of IoU instead of deleting them, letting a strongly overlapping true detection survive with a reduced score.</span>
* <span style="font-size: 14px;">**Learned NMS** and relation networks predict suppression directly rather than thresholding.</span>
* <span style="font-size: 14px;">**DETR** (Carion et al., 2020) removes NMS entirely by training with bipartite matching so the model emits one box per object by construction.</span>

<span style="font-size: 14px;">Despite these successors, hard NMS remains the default in production because it is parameter-free to train, trivial to implement, fast on GPU, and robust across datasets. Most deployed detectors still ship with it, which is why a correct, deterministic, torchvision-parity implementation is a core skill.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using greater-than-or-equal instead of strict greater-than.** The torchvision convention suppresses only when IoU strictly exceeds the threshold, so a box exactly at the threshold is kept. Flipping to $\geq$ changes which boxes survive on boundary cases and breaks parity with the reference.</span>
* <span style="font-size: 14px;">**Non-deterministic tie-breaking.** When two boxes share a score, the order they are processed changes the output. Always break ties by lower original index so the result is reproducible; relying on an unstable sort gives different `keep` lists across runs.</span>
* <span style="font-size: 14px;">**Running a single global NMS across all classes.** Suppressing across class labels deletes correct detections of one class that overlap a higher-scoring detection of another. Run NMS per class, or use the coordinate-offset trick to isolate classes.</span>
* <span style="font-size: 14px;">**Forgetting the zero-clamp inside the IoU.** The suppression IoU has the same disjoint-box trap as any IoU: without clamping the intersection width and height, non-overlapping boxes can report a fictitious positive overlap and get wrongly suppressed.</span>

---