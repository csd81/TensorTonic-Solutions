# <span style="font-size: 20px;">Non-Maximum Suppression (NMS)</span>

<span style="font-size: 14px;">Non-Maximum Suppression is the essential post-processing step in object detection that removes redundant detections. A detector like YOLO or Faster R-CNN produces many overlapping bounding boxes for each object. NMS selects the best detection per object by iteratively keeping the highest-scoring box and suppressing overlapping boxes, converting thousands of raw detections into a clean set of final predictions.</span>

---

## <span style="font-size: 16px;">Intersection over Union (IoU)</span>

<span style="font-size: 14px;">IoU measures the overlap between two bounding boxes on a scale from 0 (no overlap) to 1 (perfect overlap):</span>

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

<span style="font-size: 14px;">For axis-aligned boxes with coordinates $[x_1, y_1, x_2, y_2]$ (top-left and bottom-right corners):</span>

<span style="font-size: 14px;">1. **Intersection**: compute the overlap rectangle. Its coordinates are $[\max(x_1^A, x_1^B), \max(y_1^A, y_1^B), \min(x_2^A, x_2^B), \min(y_2^A, y_2^B)]$. If the intersection width or height is negative, there is no overlap.</span>

<span style="font-size: 14px;">2. **Union**: $\text{Area}(A) + \text{Area}(B) - \text{Intersection}$ (inclusion-exclusion principle).</span>

<span style="font-size: 14px;">IoU is the standard metric for both NMS thresholding and evaluating detection accuracy (mAP computation). An IoU threshold of 0.5 (PASCAL VOC) or 0.5:0.95 (COCO) determines whether a detection counts as correct.</span>

---

## <span style="font-size: 16px;">Standard NMS Algorithm</span>

<span style="font-size: 14px;">The greedy NMS algorithm:</span>

<span style="font-size: 14px;">1. **Sort** all detections by confidence score in descending order</span>
<span style="font-size: 14px;">2. **Select** the highest-scoring box and add it to the kept list</span>
<span style="font-size: 14px;">3. **Suppress** all remaining boxes whose IoU with the selected box exceeds the threshold</span>
<span style="font-size: 14px;">4. **Repeat** from step 2 with the remaining (non-suppressed) boxes</span>

<span style="font-size: 14px;">The IoU threshold controls the trade-off: lower threshold = more aggressive suppression (fewer detections, may miss nearby objects), higher threshold = less suppression (more detections, may have duplicates). Typical values: 0.5 for PASCAL VOC, 0.5-0.7 for COCO.</span>

<span style="font-size: 14px;">NMS is applied per-class: detections of different classes do not suppress each other. A "person" box should not suppress a nearby "bicycle" box even if they overlap.</span>

---

## <span style="font-size: 16px;">Soft-NMS</span>

<span style="font-size: 14px;">Standard NMS has a critical flaw: it uses a hard 0/1 decision. Two genuine objects that are close together (e.g., two people standing side by side) may have IoU above the threshold, causing one to be incorrectly suppressed. Soft-NMS (Bodla et al., 2017) addresses this by decaying scores instead of hard removal:</span>

<span style="font-size: 14px;">**Linear decay**: $s_j \leftarrow s_j (1 - \text{IoU})$ when $\text{IoU} > \theta$. Simple and effective. Boxes with moderate overlap retain moderate scores.</span>

<span style="font-size: 14px;">**Gaussian decay**: $s_j \leftarrow s_j \exp(-\text{IoU}^2 / \sigma)$. Smooth decay applied to all boxes regardless of threshold. The $\sigma$ parameter controls decay rate. No hard cutoff.</span>

<span style="font-size: 14px;">After processing, boxes with scores below a minimum threshold (e.g., 0.001) are removed. Soft-NMS typically improves mAP by 1-2% on COCO, with no additional training required - it is a drop-in replacement for standard NMS.</span>

---

## <span style="font-size: 16px;">Variants and Modern Approaches</span>

<span style="font-size: 14px;">**Batched NMS**: applies NMS independently per class by adding class-specific offsets to coordinates, then running NMS once on all offset boxes.</span>

<span style="font-size: 14px;">**DIoU-NMS**: replaces IoU with Distance-IoU, which considers center point distance. Two boxes with the same IoU but different center distances get different suppression - preferring to suppress boxes whose centers are closer to the selected box's center.</span>

<span style="font-size: 14px;">**Weighted NMS**: instead of simply keeping the highest-scoring box, computes a weighted average of all suppressed boxes' coordinates, weighted by their scores. This can produce more precise localization.</span>

<span style="font-size: 14px;">**NMS-free detectors**: DETR (Detection Transformer) uses set-based predictions with Hungarian matching loss, eliminating the need for NMS entirely. Each query learns to predict a unique object. This is a paradigm shift from anchor-based detection.</span>

---

## <span style="font-size: 16px;">Implementation Details</span>

<span style="font-size: 14px;">**Coordinate format**: most implementations use $[x_1, y_1, x_2, y_2]$ (corner format). Some use $[x_c, y_c, w, h]$ (center format). Always verify which format your detector outputs.</span>

<span style="font-size: 14px;">**Vectorized IoU**: for efficiency, compute IoU between one box and all remaining boxes at once using numpy broadcasting, rather than looping over pairs. This reduces NMS from $O(N^2)$ element-wise operations to $O(N)$ vectorized operations per selected box.</span>

<span style="font-size: 14px;">**GPU NMS**: frameworks like torchvision provide CUDA-accelerated NMS (torchvision.ops.nms). For inference at scale (thousands of boxes per image), GPU NMS is essential for real-time performance.</span>

<span style="font-size: 14px;">**Score threshold pre-filtering**: before NMS, filter out low-confidence detections (score < 0.05). This dramatically reduces the number of boxes NMS must process, improving speed without affecting quality.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


**Q: Why is NMS applied per-class rather than globally?**

A: <span style="font-size: 14px;">Different object classes can legitimately overlap in an image - a person riding a bicycle, a cup on a table, a tie on a person. If NMS were applied globally, the higher-scoring detection (e.g., person) would suppress the lower-scoring overlapping detection (e.g., bicycle), causing a miss. Per-class NMS ensures that suppression only happens between detections of the same class. The implementation typically adds a large class-specific offset to box coordinates (e.g., class_id * max_coordinate), runs NMS once on all offset boxes, then removes the offsets.</span>

**Q: What are the failure modes of NMS and how can they be mitigated?**

A: <span style="font-size: 14px;">Main failure modes: (1) Suppressing true positives when objects are close together (e.g., crowd scenes) - mitigate with Soft-NMS, higher IoU threshold, or NMS-free detectors like DETR. (2) Missing detections when the IoU threshold is too low - use higher thresholds or adaptive thresholds. (3) Keeping false positives when the IoU threshold is too high - use lower thresholds or better score calibration. (4) Sensitivity to the threshold value - no single threshold works for all scenarios. Soft-NMS is more robust because it decays scores gradually rather than using a hard cutoff.</span>

**Q: Compare standard NMS, Soft-NMS, and NMS-free approaches - when would you use each?**

A: <span style="font-size: 14px;">Standard NMS: simple, fast, works well for sparse scenes where objects are well-separated. Good default choice. Soft-NMS: better for crowded scenes (pedestrian detection, cell counting) where genuine objects may overlap. Drop-in replacement with ~1-2% mAP improvement, no retraining needed. NMS-free (DETR): eliminates NMS entirely through set-based prediction. Best for scenes with variable density and when you want end-to-end trainability. Downside: DETR requires longer training (300+ epochs vs 12-36 for Faster R-CNN) and currently struggles with small objects. For production systems, standard NMS with careful threshold tuning is still the most common choice.</span>

**Q: How does the choice of IoU threshold affect precision and recall?**

A: <span style="font-size: 14px;">Lower IoU threshold (e.g., 0.3): more aggressive suppression, fewer detections. Higher precision (fewer false positives from duplicate detections) but lower recall (may suppress genuine nearby objects). Higher IoU threshold (e.g., 0.7): less suppression, more detections. Higher recall (keeps nearby objects) but lower precision (may keep duplicate detections). The optimal threshold depends on the application: for safety-critical applications (autonomous driving), higher recall is preferred; for applications where false positives are costly (medical imaging alerts), higher precision may be preferred.</span>

**Q: How would you implement NMS efficiently for real-time detection?**

A: <span style="font-size: 14px;">Key optimizations: (1) Pre-filter by score threshold (remove boxes with score < 0.05 before NMS), reducing the candidate set from thousands to hundreds. (2) Vectorize IoU computation: compute IoU between the selected box and all remaining boxes using numpy/tensor broadcasting rather than looping. (3) Use GPU-accelerated NMS (torchvision.ops.nms) for inference. (4) Early termination: stop NMS when remaining boxes have scores below the threshold. (5) For multi-class: use batched NMS with coordinate offsets rather than running NMS separately per class. These optimizations together can make NMS negligible in the inference pipeline - typically < 1ms even with 1000+ boxes.</span>

---