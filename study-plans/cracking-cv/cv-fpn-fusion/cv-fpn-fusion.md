# <span style="font-size: 20px;">FPN Top-Down Fusion</span>

<span style="font-size: 14px;">The **Feature Pyramid Network (FPN)** (Lin et al., 2017) builds a multi-scale feature representation in which every level carries strong semantics, by fusing a deep backbone's feature maps top-down. Its top-down pathway upsamples coarse, semantically rich features and adds them to finer, spatially precise features through lateral connections. This produces a pyramid where small objects are detected on high-resolution levels and large objects on low-resolution levels, all with comparable feature quality. FPN became the standard neck for Faster R-CNN, Mask R-CNN, and RetinaNet, and remains widely used today.</span>

---

## <span style="font-size: 16px;">The Multi-Scale Problem</span>

<span style="font-size: 14px;">Objects appear at vastly different sizes, so a detector needs features at multiple resolutions. The naive options each fail in a specific way. Detecting on a single high-level feature map gives strong semantics but poor resolution, missing small objects. Detecting on multiple backbone levels directly (the SSD approach) gives resolution but the shallow, high-resolution levels lack semantic depth, so small objects are localized but poorly classified. Building an image pyramid and running the network at every scale is accurate but multiplies compute and memory.</span>

<span style="font-size: 14px;">FPN's insight is that a deep convolutional network already computes a feature hierarchy in its forward pass, with a built-in pyramid of resolutions from the successive downsampling stages. The problem is purely that the high-resolution early layers are semantically weak. FPN fixes this by propagating the semantic strength of deep layers back down to the high-resolution layers, for almost no extra cost, rather than recomputing features at multiple scales.</span>

---

## <span style="font-size: 16px;">Lateral Connections and the Top-Down Path</span>

<span style="font-size: 14px;">FPN has two pathways. The **bottom-up** pathway is the backbone's normal forward pass, producing feature maps $C_2, C_3, \ldots$ at successively coarser resolutions, each roughly half the spatial size of the previous and semantically richer. The **top-down** pathway runs in the opposite direction, starting from the coarsest map and progressively building higher-resolution pyramid features.</span>

<span style="font-size: 14px;">A **lateral connection** links each bottom-up level to the corresponding top-down level. It is a $1 \times 1$ convolution applied to the bottom-up feature, serving two purposes: it projects every level to the same channel count $C$ (commonly $256$) so they can be added, and it lets the network learn how much of each bottom-up feature to inject. In this problem the lateral convolutions are assumed already applied, so the inputs $F[l]$ are the post-lateral feature maps, all with $C$ channels.</span>

---

## <span style="font-size: 16px;">The Fusion Equations</span>

<span style="font-size: 14px;">Given $L$ feature maps $F[0], \ldots, F[L-1]$ ordered highest to lowest resolution, where consecutive levels follow a dyadic $2\times$ ratio ($H_{l+1} = H_l/2$, $W_{l+1} = W_l/2$), the pyramid outputs $P[l]$ are built top-down:</span>

$$
P[L-1] = F[L-1], \qquad P[l] = F[l] + \mathrm{Up}_2(P[l+1])
$$

<span style="font-size: 14px;">The coarsest level $P[L-1]$ is just its own feature map, since there is nothing coarser to fuse. Every finer level adds its own (post-lateral) feature to a $2\times$ upsampled copy of the already-fused next-coarser pyramid level. Because the recursion starts at the top, the semantic content of the deepest layer flows all the way down: $P[0]$ contains contributions from every coarser level, each upsampled the appropriate number of times.</span>

<span style="font-size: 14px;">The addition is elementwise and requires the two operands to match in spatial size, which is exactly why $P[l+1]$ must be upsampled by $2\times$ before being added to $F[l]$. The dyadic shape assumption guarantees that a single $2\times$ upsample aligns the resolutions at every step.</span>

---

## <span style="font-size: 16px;">Nearest-Neighbour Upsampling</span>

<span style="font-size: 14px;">The upsampling operator $\mathrm{Up}_2$ is nearest-neighbour $2\times$: each input pixel becomes a $2 \times 2$ block of identical values in the output, doubling both spatial dimensions. This matches `F.interpolate(scale_factor=2, mode='nearest')`. For an input of shape $(C, H, W)$ the output is $(C, 2H, 2W)$, where output pixel $(2i, 2j)$, $(2i, 2j{+}1)$, $(2i{+}1, 2j)$, and $(2i{+}1, 2j{+}1)$ all copy input pixel $(i, j)$.</span>

<span style="font-size: 14px;">Nearest-neighbour is chosen over bilinear deliberately. The FPN paper notes that the top-down features carry semantic, not precise spatial, information, so the coarse upsampling does not need sub-pixel accuracy; the high-resolution detail comes from the lateral $F[l]$ term added afterward. Nearest-neighbour is also cheap and parameter-free, keeping the neck nearly free relative to the backbone. After the addition, FPN applies a $3 \times 3$ convolution to each merged level to reduce the aliasing artefacts of nearest-neighbour upsampling, though that smoothing conv is outside the fusion step computed here.</span>

---

## Worked Example ($L = 2$, $C = 1$)

<span style="font-size: 14px;">Let $F[1]$ (coarsest) be the $1 \times 1$ map $[[4]]$ and $F[0]$ (finest) be the $2 \times 2$ map $[[1, 2], [3, 4]]$, single channel.</span>

<span style="font-size: 14px;">**Top level**: $P[1] = F[1] = [[4]]$, unchanged.</span>

<span style="font-size: 14px;">**Upsample $P[1]$ by $2\times$**: the single value $4$ becomes a $2 \times 2$ block $[[4, 4], [4, 4]]$.</span>

<span style="font-size: 14px;">**Fuse the fine level**: $P[0] = F[0] + \mathrm{Up}_2(P[1]) = [[1, 2], [3, 4]] + [[4, 4], [4, 4]] = [[5, 6], [7, 8]]$.</span>

<span style="font-size: 14px;">The output, in input order, is $[P[0], P[1]] = [\,[[5, 6], [7, 8]],\ [[4]]\,]$, each value rounded to 4 decimals. Every fine-level pixel has absorbed the coarse level's semantic signal (the constant $4$) while retaining its own spatial detail.</span>

<span style="font-size: 14px;">The nearest-neighbour upsample is visible here: the single coarse value $4$ is broadcast identically to all four positions of the $2 \times 2$ block, so every fine-level cell receives the same coarse contribution and differs only by its own lateral value. A bilinear upsample of a $1 \times 1$ map would also yield all-$4$ here, but on any map larger than a single pixel the two modes diverge, which is why matching `mode='nearest'` exactly is essential for parity.</span>

---

## <span style="font-size: 16px;">The Recursion Unrolled</span>

<span style="font-size: 14px;">Expanding the recursion shows how each level accumulates contributions. With $L = 4$ levels, $P[3] = F[3]$, then $P[2] = F[2] + \mathrm{Up}_2(F[3])$, then $P[1] = F[1] + \mathrm{Up}_2(F[2] + \mathrm{Up}_2(F[3]))$, and $P[0] = F[0] + \mathrm{Up}_2(P[1])$. The finest level $P[0]$ therefore contains $F[0]$ plus a once-upsampled $F[1]$, plus a twice-upsampled $F[2]$, plus a three-times-upsampled $F[3]$, each blurred by the repeated nearest-neighbour expansion.</span>

<span style="font-size: 14px;">This nesting is why the order of computation matters: $P[l]$ depends on the fully fused $P[l+1]$, not on the raw $F[l+1]$. Computing $P[l] = F[l] + \mathrm{Up}_2(F[l+1])$ with the raw feature instead of the fused pyramid level would only inject one level of coarse semantics rather than propagating all coarser levels down. The recursion on $P$, not $F$, is what makes the semantic signal travel the full height of the pyramid.</span>

---

## <span style="font-size: 16px;">Why Addition and Not Concatenation</span>

<span style="font-size: 14px;">FPN merges the lateral and top-down features by elementwise addition rather than channel concatenation. Addition keeps the channel count fixed at $C$ throughout the pyramid, so a single detection head with one set of weights can be shared across all levels, which is central to FPN's efficiency and to RetinaNet's design. Concatenation would double channels at each merge and force level-specific heads, increasing parameters and breaking weight sharing.</span>

<span style="font-size: 14px;">Addition also has a clean interpretation: the lateral $1 \times 1$ conv has already projected the bottom-up feature into the same $C$-dimensional space as the top-down feature, so the two are summed as compatible representations. The network learns, through the lateral conv weights, how strongly to blend the high-resolution and coarse-semantic contributions. This is analogous to a residual connection, adding a learned refinement onto a propagated base.</span>

---

## <span style="font-size: 16px;">Assigning Objects to Levels</span>

<span style="font-size: 14px;">Once the pyramid is built, RoIs or anchors are assigned to levels by size. FPN assigns a region of interest of width $w$ and height $h$ to level $k = \lfloor k_0 + \log_2(\sqrt{wh}/224) \rfloor$, so larger objects land on coarser levels and smaller objects on finer levels. This matches each object to the level whose resolution and receptive field suit its scale, which is the entire point of building the pyramid.</span>

<span style="font-size: 14px;">Because every pyramid level now carries strong semantics thanks to the top-down fusion, this size-based routing does not sacrifice classification quality on the fine levels, which was the weakness of the plain multi-level SSD approach. Small objects detected on $P[0]$ get both the high resolution they need for localization and the deep semantics propagated down from the coarsest level.</span>

---

## <span style="font-size: 16px;">Channel and Shape Bookkeeping</span>

<span style="font-size: 14px;">Every level in the pyramid carries the same channel count $C$ because the lateral $1 \times 1$ convolutions project them all to $C$ before fusion. This uniformity is structural: the elementwise addition requires matching channels, the shared head requires matching channels, and nearest-neighbour upsampling preserves channels exactly (it only touches the spatial dimensions). So the only quantity that changes between levels is spatial resolution, which the dyadic assumption keeps as clean powers of two.</span>

<span style="font-size: 14px;">The dyadic $2\times$ ratio between consecutive levels is what allows a single fixed upsampling factor to align resolutions everywhere. If the backbone produced non-dyadic ratios, the upsample factor would vary per level and the clean recursion would break. Standard backbones (ResNet stages) downsample by exactly $2\times$ per stage, so this assumption holds in practice, and the implementation can safely hard-code the factor of two everywhere.</span>

---

## <span style="font-size: 16px;">Cost and Impact</span>

<span style="font-size: 14px;">The top-down pathway adds only the lateral $1 \times 1$ convolutions, the parameter-free nearest-neighbour upsamples, the elementwise additions, and the per-level $3 \times 3$ smoothing convs. This is a small fraction of the backbone's compute, so FPN delivers multi-scale features at near-zero marginal cost compared to an image pyramid. The FPN paper reported large gains, particularly on small objects, when added to Faster R-CNN, and the design has been near-universal in detectors and segmenters since.</span>

<span style="font-size: 14px;">Variants extended the idea: PANet (Liu et al., 2018) adds a second bottom-up path on top of the top-down one to shorten the information path for low-level features, and BiFPN (Tan et al., 2020) makes the fusion weighted and repeatable. NAS-FPN searched for the fusion topology automatically. All of them keep FPN's core operation, the upsample-and-add top-down merge, as the foundation, which is why understanding this single fusion step generalizes across the whole family of detection necks.</span>

<span style="font-size: 14px;">FPN also generalizes beyond detection. The same top-down fusion underpins panoptic and semantic segmentation heads, where the high-resolution fused levels provide the dense per-pixel features a segmentation decoder needs. In every case the mechanism is identical: propagate coarse semantics down through nearest-neighbour upsampling and elementwise addition onto laterally-projected high-resolution features.</span>

---

## <span style="font-size: 16px;">Semantic versus Spatial Trade-off</span>

<span style="font-size: 14px;">Each fused level resolves a tension between two kinds of information. The lateral term $F[l]$ comes from a shallow, high-resolution stage of the backbone: it knows precisely where edges and textures are but little about what object they belong to. The top-down term $\mathrm{Up}_2(P[l+1])$ comes from deep, low-resolution stages: it knows what the object is but has lost fine spatial detail through downsampling. Adding them gives a level that is both spatially sharp and semantically informed.</span>

<span style="font-size: 14px;">This is the precise reason FPN improves small-object detection so much. Small objects live on the finest levels, which in a plain backbone are semantically weak. The top-down injection of deep semantics is what makes those fine levels good enough to classify small objects, not just locate them. The paper's ablations show that removing either the lateral connections or the top-down path collapses much of the gain, confirming both halves are necessary.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Upsampling the wrong operand or using the wrong factor.** The next-coarser pyramid level $P[l+1]$ is the one upsampled, not the current $F[l]$, and the factor is exactly $2\times$ to match the dyadic ratio. Upsampling $F[l]$ or using a non-2 factor produces a shape mismatch in the elementwise add.</span>
* <span style="font-size: 14px;">**Using bilinear instead of nearest-neighbour upsampling.** The reference uses `mode='nearest'`, where each pixel becomes an identical $2 \times 2$ block. Bilinear interpolation blends neighbours and produces different values, so the fused outputs will not match. Always match the exact interpolation mode used by the reference.</span>
* <span style="font-size: 14px;">**Fusing bottom-up instead of top-down.** The recursion must start at the coarsest level and propagate down; building $P$ from the finest level up reverses the semantic flow and defeats the purpose. The coarsest level seeds the pyramid as $P[L-1] = F[L-1]$ with no fusion.</span>
* <span style="font-size: 14px;">**Returning the pyramid in the wrong order.** The output must be $[P[0], \ldots, P[L-1]]$ in input (finest-to-coarsest) order, even though it is computed coarsest-first. Emitting it in computation order reverses the levels and silently misaligns every downstream head.</span>

---