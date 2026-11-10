# <span style="font-size: 20px;">Anchor Box Generation</span>

<span style="font-size: 14px;">**Anchor boxes** are a fixed set of reference rectangles of predetermined scales and aspect ratios, tiled densely across a convolutional feature map. They convert the open-ended problem of localizing objects of arbitrary size and shape into a tractable problem of classifying and refining a discrete grid of priors. Anchors were introduced by the Region Proposal Network in Faster R-CNN (Ren et al., 2015) and became the backbone of SSD (Liu et al., 2016) and RetinaNet (Lin et al., 2017).</span>

---

## <span style="font-size: 16px;">Why Anchors Exist</span>

<span style="font-size: 14px;">A detector must answer two questions at every image location: is there an object here, and what are its exact box coordinates. Predicting absolute coordinates directly from scratch is hard, because a single feature-map cell would need to regress wildly different boxes for a distant pedestrian versus a nearby bus. Anchors solve this by giving the network a menu of **prior shapes** at each location. The network only has to (1) score each anchor for objectness or class, and (2) regress a small offset that nudges the chosen anchor onto the true object.</span>

<span style="font-size: 14px;">This reframing has two benefits. First, regressing a small correction relative to a sensible prior is far easier than regressing absolute size from nothing, so training converges faster. Second, by placing multiple anchors of different aspect ratios at the same location, the network can detect a tall person and a wide car centered on the same pixel without ambiguity, because each is matched to a differently shaped anchor.</span>

---

## <span style="font-size: 16px;">The Feature-Map-to-Image Mapping</span>

<span style="font-size: 14px;">Anchors are generated on the coordinate grid of a feature map but expressed in image pixel coordinates. The link between the two is the **stride** $s$, the downsampling factor of the backbone at that level. A feature map produced after total downsampling of $16\times$ has stride $s = 16$, meaning each feature cell corresponds to a $16 \times 16$ pixel region in the input image.</span>

<span style="font-size: 14px;">For a feature cell at integer location $(f_y, f_x)$, its center in image coordinates is placed at the middle of the corresponding pixel block:</span>

$$
c_x = (f_x + 0.5)\,s, \qquad c_y = (f_y + 0.5)\,s
$$

<span style="font-size: 14px;">The $+0.5$ offset centers the anchor inside its receptive-field cell rather than at the top-left corner, so anchors are spaced uniformly by $s$ pixels and symmetric about the cell. Without it, every anchor would be shifted half a stride toward the origin, biasing all matches and regressions.</span>

---

## <span style="font-size: 16px;">Scale and Aspect Ratio Parametrization</span>

<span style="font-size: 14px;">At each center, anchors of multiple sizes and shapes are emitted. A **scale** multiplier stretches a base size, and an **aspect ratio** $r = w/h$ controls the width-to-height shape. Given a base size $b$, a scale factor, and ratio $r$, the anchor dimensions are:</span>

$$
w = b \cdot \text{scale} \cdot \sqrt{r}, \qquad h = b \cdot \text{scale} / \sqrt{r}
$$

<span style="font-size: 14px;">The $\sqrt{r}$ split is deliberate: it keeps the anchor **area** constant across aspect ratios for a fixed scale, since $w \cdot h = (b \cdot \text{scale})^2$ regardless of $r$. This means changing the aspect ratio reshapes the box without growing it, so the scale parameter alone controls area and the ratio parameter alone controls shape. A ratio $r = 1$ gives a square; $r = 2$ gives a box twice as wide as tall; $r = 0.5$ gives a tall box.</span>

<span style="font-size: 14px;">Once width and height are known, the anchor is converted from center-size form to the `xyxy` corner format used everywhere downstream:</span>

$$
\text{anchor} = [\,c_x - w/2,\ c_y - h/2,\ c_x + w/2,\ c_y + h/2\,]
$$

---

## <span style="font-size: 16px;">Base Size, Scale, and Their Roles</span>

<span style="font-size: 14px;">Three quantities jointly set an anchor's size, and they play distinct roles. The **base size** $b$ is a fixed reference length tied to the feature level, often chosen proportional to the stride so that anchors and receptive fields stay commensurate. The **scale** is a per-anchor multiplier that produces several sizes around the base. The **aspect ratio** reshapes without resizing. Separating these lets a designer reason about coverage independently: pick base sizes to span the dataset's object scales, add a few intra-level scales for finer granularity, and add ratios to cover non-square shapes.</span>

<span style="font-size: 14px;">Concretely, the effective area of an anchor is $(b \cdot \text{scale})^2$, independent of ratio. This is why the area sweep of an anchor set is fully determined by $b$ and the scale list, and why two detectors with the same base sizes but different ratio lists still cover the same range of object areas. The ratios only change which shapes within each area band are representable.</span>

---

## <span style="font-size: 16px;">Iteration Order and Output Layout</span>

<span style="font-size: 14px;">The full anchor set is produced by four nested loops, ordered outermost to innermost as $f_y \to f_x \to \text{scale} \to r$. For a feature map of shape $(H, W)$ with $S$ scales and $R$ aspect ratios, this yields $H \cdot W \cdot S \cdot R$ anchors, returned as a 2D list of shape $(H \cdot W \cdot S \cdot R,\ 4)$.</span>

<span style="font-size: 14px;">This row-major ordering matters because the detection head produces a parallel tensor of scores and regression deltas, and the two must be aligned anchor-for-anchor. If the anchor list is generated in one order and the prediction tensor is flattened in another, every prediction is silently attached to the wrong anchor, and the model trains to garbage without any crash. The convention here, location-major then scale then ratio, matches how the head's $(S \cdot R)$ output channels are typically reshaped per spatial location.</span>

---

## Worked Example ($H = W = 1$, $s = 16$, one scale, two ratios)

<span style="font-size: 14px;">Take a $1 \times 1$ feature map, stride $s = 16$, base size $b = 4$, scale $= 1$, and aspect ratios $r \in \{1, 2\}$. There is a single cell at $(f_y, f_x) = (0, 0)$.</span>

<span style="font-size: 14px;">**Center**: $c_x = (0 + 0.5)\cdot 16 = 8$, $c_y = 8$.</span>

<span style="font-size: 14px;">**Ratio $r = 1$**: $\sqrt{1} = 1$, so $w = 4 \cdot 1 \cdot 1 = 4$, $h = 4 \cdot 1 / 1 = 4$. Anchor $= [8 - 2,\ 8 - 2,\ 8 + 2,\ 8 + 2] = [6, 6, 10, 10]$.</span>

<span style="font-size: 14px;">**Ratio $r = 2$**: $\sqrt{2} \approx 1.4142$, so $w = 4 \cdot 1.4142 = 5.6569$, $h = 4 / 1.4142 = 2.8284$. Center stays $(8, 8)$, so $w/2 = 2.8284$, $h/2 = 1.4142$. Anchor $= [8 - 2.8284,\ 8 - 1.4142,\ 8 + 2.8284,\ 8 + 1.4142] = [5.1716, 6.5858, 10.8284, 9.4142]$.</span>

<span style="font-size: 14px;">Both anchors have area $4 \times 4 = 16$ (the $r = 2$ box is $5.6569 \times 2.8284 \approx 16$), confirming the $\sqrt{r}$ split preserves area. Rounded to 4 decimals the output is $[[6.0, 6.0, 10.0, 10.0],\ [5.1716, 6.5858, 10.8284, 9.4142]]$.</span>

---

## <span style="font-size: 16px;">Anchors and the Receptive Field</span>

<span style="font-size: 14px;">Each feature cell sees only a limited region of the input, its receptive field, which grows with network depth and downsampling. Anchors are sized to align with this receptive field: an anchor much larger than the cell's receptive field asks the network to predict an object it cannot fully see, while an anchor much smaller wastes resolution. Tying the base size to the stride keeps anchors and receptive fields commensurate, which is why coarser levels with larger strides carry larger anchors.</span>

<span style="font-size: 14px;">This alignment also explains the multi-scale pyramid design philosophically. Rather than forcing one feature level to host anchors spanning a huge size range, the network distributes anchors so that each is matched to a level whose receptive field can actually capture objects of that size. The result is that the objectness and regression predictions at every cell stay within the information the cell has access to.</span>

---

## <span style="font-size: 16px;">Anchors Across the Pyramid</span>

<span style="font-size: 14px;">A single feature level cannot cover the full range of object sizes. Faster R-CNN tiles all scales and ratios on one feature map, but SSD and RetinaNet instead assign each scale to a different pyramid level: coarse, large-stride maps carry large anchors for big objects, and fine, small-stride maps carry small anchors for small objects. Because a larger stride spaces anchors farther apart in image space, large objects do not need dense coverage, while small objects, packed onto the fine level, get the dense sampling they require.</span>

<span style="font-size: 14px;">RetinaNet, built on a Feature Pyramid Network, uses three aspect ratios $\{0.5, 1, 2\}$ and three intra-level scale multipliers $\{2^0, 2^{1/3}, 2^{2/3}\}$ per level, giving nine anchors per location. The base size grows by the stride across levels (areas from $32^2$ to $512^2$). This design lets a single shared head operate identically at every level, since the anchors themselves encode the scale.</span>

<span style="font-size: 14px;">SSD takes a related but distinct approach, generating anchors directly on feature maps drawn from several stages of a single backbone rather than from a fused pyramid. It linearly interpolates a minimum and maximum scale across the chosen layers and adds an extra scale of $\sqrt{s_k \cdot s_{k+1}}$ at ratio $1$ for finer coverage between bands. Despite the architectural differences, all three detectors share the same per-location generation rule: a center from the stride, dimensions from scale and $\sqrt{r}$, and a conversion to `xyxy`.</span>

---

## <span style="font-size: 16px;">Count, Cost, and Precomputation</span>

<span style="font-size: 14px;">The number of anchors grows as $H \cdot W \cdot S \cdot R$, which is large: a $50 \times 50$ feature map with $9$ anchors per location already yields $22{,}500$ anchors at a single level, and a full pyramid easily exceeds $100{,}000$. This dominates the input to NMS and the regression head, so the layout and ordering must be efficient and consistent.</span>

<span style="font-size: 14px;">Crucially, anchors depend only on the feature-map geometry, the stride, and the fixed scale and ratio lists, never on the image content. They are therefore identical for every image of a given input resolution and can be generated once and cached, rather than recomputed per forward pass. A common implementation precomputes a base set of $S \cdot R$ anchors at the origin and adds the per-cell shift grid $(c_x, c_y)$ by broadcasting, turning the four nested loops into a single broadcasted addition.</span>

---

## <span style="font-size: 16px;">From Anchors to Detections</span>

<span style="font-size: 14px;">Anchors are only priors; the network output transforms them into final boxes in two stages. During training, each anchor is labeled by its IoU with ground-truth boxes: above a high threshold it is a positive whose regression target is the encoded offset to the matched object, below a low threshold it is a background negative, and in between it is ignored. At inference the classification head scores each anchor and the regression head emits deltas that are decoded onto the anchor to produce the predicted box, after which Non-Maximum Suppression removes duplicates.</span>

<span style="font-size: 14px;">The quality of the anchor set directly bounds achievable recall: if no anchor has sufficient IoU with a given object, that object can never be matched as a positive and is effectively invisible to the detector. This is why scales and ratios are chosen to match the dataset's object-size distribution, and why anchor-free detectors (FCOS, CenterNet) were later proposed to remove this hand-tuned prior entirely.</span>

---

## <span style="font-size: 16px;">Anchor-Based versus Anchor-Free</span>

<span style="font-size: 14px;">Anchors introduce hyperparameters that must be tuned to the data: the number of scales, the ratio set, the base sizes, and the positive/negative IoU thresholds. Poor choices cap recall and waste computation on anchors that never match anything. This motivated a wave of **anchor-free** detectors that drop the prior boxes entirely.</span>

* <span style="font-size: 14px;">**FCOS** (Tian et al., 2019) treats every feature location as a sample and directly regresses the four distances from that point to the object's box edges, removing scales and ratios.</span>
* <span style="font-size: 14px;">**CenterNet** (Zhou et al., 2019) predicts object centers as a heatmap and regresses width and height at the peak, framing detection as keypoint estimation.</span>
* <span style="font-size: 14px;">**DETR** (Carion et al., 2020) replaces both anchors and NMS with a set of learned object queries and bipartite matching.</span>

<span style="font-size: 14px;">Despite these alternatives, anchor-based detectors remain widely deployed because the anchor generation step is cheap, deterministic, and easy to reason about, and because a well-tuned anchor set still delivers state-of-the-art accuracy. Understanding anchor generation is therefore foundational even when working with anchor-free models, since the matching and encoding machinery is shared.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Omitting the $+0.5$ center offset.** Placing anchors at $f_x \cdot s$ instead of $(f_x + 0.5)\cdot s$ shifts every anchor half a stride toward the top-left. The error is small per anchor but systematic, biasing all IoU matches and regression targets and degrading localization, especially at large strides.</span>
* <span style="font-size: 14px;">**Splitting aspect ratio without the square root.** Using $w = b \cdot \text{scale} \cdot r$ and $h = b \cdot \text{scale}$ makes area scale with $r$, so non-square anchors no longer share area with square ones. The $\sqrt{r}$ form is what keeps area constant; getting it wrong distorts the scale-versus-shape separation the design relies on.</span>
* <span style="font-size: 14px;">**Mismatched iteration order versus the prediction tensor.** Generating anchors in $f_y \to f_x \to \text{scale} \to r$ order but flattening the head's output differently misaligns every score and delta with its anchor. There is no crash; the model simply trains to noise. The anchor layout and the head reshape must use the identical nesting.</span>
* <span style="font-size: 14px;">**Letting anchors run off the image without handling.** Anchors near the border extend past the image bounds. Faster R-CNN drops cross-boundary anchors during training to avoid large erroneous gradients but keeps them (clipped) at inference. Forgetting to clip or drop them produces invalid boxes and unstable losses.</span>

---