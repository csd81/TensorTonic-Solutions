# <span style="font-size: 20px;">Adaptive Average Pool 2D</span>

<span style="font-size: 14px;">Adaptive average pooling produces a **fixed output spatial size regardless of the input size**, by partitioning the input into a grid of $\text{out}_h \times \text{out}_w$ regions and averaging each region. It is the operation behind the global-average-pool head of ResNet and GoogLeNet and the mechanism that lets a single trained classifier accept images of varying resolution.</span>

---

## <span style="font-size: 16px;">What It Does</span>

<span style="font-size: 14px;">Ordinary pooling fixes the **window** ($k$) and **stride** ($s$); the output size then depends on the input size. Adaptive pooling inverts this contract: the **output size** is fixed and the window boundaries are derived from the input size. The user asks for an $\text{out}_h \times \text{out}_w$ result and the layer computes whatever region boundaries are needed to produce exactly that shape.</span>

<span style="font-size: 14px;">The defining use case is the network head. A convolutional backbone reduces an image to some feature map whose spatial size depends on the input resolution. To feed a fixed-width classifier, the spatial dimensions must be collapsed to a known size. `nn.AdaptiveAvgPool2d((1, 1))` collapses any feature map to $1 \times 1$ per channel, the global average pool. More general targets such as $(7, 7)$ are used to feed fixed-size detector or pooling heads, where a known spatial grid is required downstream.</span>

<span style="font-size: 14px;">Like ordinary pooling, it has **no learnable parameters** and operates **independently per channel**, so a $(C, H, W)$ tensor becomes $(C, \text{out}_h, \text{out}_w)$.</span>

<span style="font-size: 14px;">The mental model is a deterministic resampling. Adaptive average pooling is conceptually an area-based resize of the spatial map down (or to) a target grid, where each output cell takes the mean of the input area it covers. Unlike a learned downsampling, the mapping is fixed and depends only on the two integer shapes involved, which is what makes it cheap, reproducible, and resolution-agnostic.</span>

---

## <span style="font-size: 16px;">How Bin Boundaries Are Computed</span>

<span style="font-size: 14px;">The key to matching `F.adaptive_avg_pool2d` exactly is the boundary formula. For output index $i$ along the height axis, the input region spans:</span>

$$
\text{start}_h = \left\lfloor \frac{i \cdot H}{\text{out}_h} \right\rfloor, \qquad \text{end}_h = \left\lceil \frac{(i+1) \cdot H}{\text{out}_h} \right\rceil
$$

<span style="font-size: 14px;">and analogously for the width axis with $j$, $W$, and $\text{out}_w$. The output cell is the mean over the resulting rectangle:</span>

$$
\text{out}[i, j] = \text{mean}\big(\text{image}[\text{start}_h : \text{end}_h,\; \text{start}_w : \text{end}_w]\big)
$$

<span style="font-size: 14px;">The crucial detail is the **asymmetric rounding**: the start uses floor and the end uses ceiling. This guarantees full coverage of the input (every input row and column belongs to at least one region) and means regions can **overlap** or **vary in size** when the input does not divide evenly by the output.</span>

<span style="font-size: 14px;">Two invariants follow directly from the formula and are worth internalizing. First, **region 0 always starts at 0** because $\lfloor 0 \rfloor = 0$, and the **last region always ends at $H$** because $\lceil H \rceil = H$. So the union of regions always covers the whole axis from edge to edge, with no input pixel left out. Second, consecutive starts and ends are monotonically non-decreasing, so the regions march left to right and never run backward. These two facts are what make the tiling well-defined even when the arithmetic is messy.</span>

* <span style="font-size: 14px;">**When $H$ is a multiple of $\text{out}_h$,** every region has identical size $H / \text{out}_h$ and the operation reduces to ordinary average pooling with $k = s = H / \text{out}_h$.</span>
* <span style="font-size: 14px;">**When $H$ is not a multiple,** the floor and ceil produce regions of two adjacent sizes, and adjacent regions may share a boundary row. The mean is taken over the actual (possibly overlapping, possibly unequal) region, so the divisor varies per cell.</span>

---

## <span style="font-size: 16px;">Why Not a Fixed Window</span>

<span style="font-size: 14px;">A natural question is why not simply apply ordinary average pooling with a window and stride chosen to hit the target size. The problem is that a single integer $(k, s)$ pair generally **cannot** map an arbitrary $H$ to an arbitrary $\text{out}_h$. For example, mapping $H = 10$ to $\text{out}_h = 3$ has no integer window-stride pair that tiles cleanly: $10 / 3$ is not an integer. Adaptive pooling sidesteps this by allowing the region size to vary across the output, using the floor/ceil rule to absorb the remainder.</span>

<span style="font-size: 14px;">This is exactly the flexibility that decouples the network from input resolution. Without it, a fully connected classifier head would require a fixed feature-map size, forcing every input image to one resolution. With adaptive pooling, the same ResNet weights process $224 \times 224$, $256 \times 256$, or any other resolution, because the head always emits the same shape.</span>

<span style="font-size: 14px;">A second reason the fixed-window approach fails is that even when a clean division exists, the desired target may not correspond to any single window the convolution stack naturally produces. Object-detection and pooling heads frequently want a specific spatial resolution (for instance a $7 \times 7$ region descriptor) that is independent of how the backbone happened to subsample. Adaptive pooling provides that target directly, decoupling the head's expected geometry from the backbone's stride schedule. This same idea underlies region-of-interest pooling, where each region of arbitrary size must be reduced to a fixed grid before the classifier.</span>

---

## <span style="font-size: 16px;">Paper Context: The Global Average Pool Head</span>

<span style="font-size: 14px;">Global average pooling was introduced in Network in Network (Lin et al., 2014) as a replacement for the large fully connected classifier layers of earlier CNNs. The authors argued that GAP is more native to convolution: each channel's averaged value can be read as the confidence of a category, with no extra parameters to overfit and a built-in regularizing effect. The huge parameter cost and overfitting risk of AlexNet's final dense layers, which held the majority of the network's weights, are avoided entirely.</span>

<span style="font-size: 14px;">GoogLeNet (Szegedy et al., 2015) and ResNet (He et al., 2016) both adopted this design, ending the backbone with a global average pool followed by a single linear layer. ResNet's head is precisely `AdaptiveAvgPool2d((1, 1))` followed by a flatten and a fully connected classifier. Average is preferred over max here because the goal is a stable per-channel descriptor that summarizes the entire feature map, and the mean is far less sensitive to a single outlier location than the max would be.</span>

<span style="font-size: 14px;">Adaptive average pooling generalizes the GAP head from the $1 \times 1$ special case to any target grid, which is why frameworks expose it as a single primitive: it covers global pooling, fixed-grid spatial pooling for detection heads, and resolution-agnostic feature extraction with one operation.</span>

<span style="font-size: 14px;">The interpretability argument from Network in Network is worth restating because it explains the design choice. With GAP, the final convolutional layer can be made to output one channel per class, and each channel's spatial map is read as a heat map of where that category's evidence appears. Averaging that map yields a single confidence per class, with no dense layer in between mixing channels. The mapping from feature map to prediction is then transparent, and there are no extra weights to overfit, which the authors credited as a structural regularizer that improved generalization over the heavy fully connected heads of the time.</span>

---

## <span style="font-size: 16px;">Numerical Example: Even Division</span>

<span style="font-size: 14px;">Take a $4 \times 4$ input with target $(2, 2)$. Here $H / \text{out}_h = 4/2 = 2$ divides evenly, so every region is $2 \times 2$ and the operation matches ordinary average pooling with $k = s = 2$.</span>

$$
\text{image} = \begin{pmatrix} 1 & 3 & 2 & 4 \\ 5 & 6 & 1 & 2 \\ 7 & 2 & 9 & 0 \\ 1 & 8 & 3 & 4 \end{pmatrix}
$$

<span style="font-size: 14px;">For $i = 0$: $\text{start}_h = \lfloor 0 \cdot 4 / 2 \rfloor = 0$, $\text{end}_h = \lceil 1 \cdot 4 / 2 \rceil = 2$. For $i = 1$: $\text{start}_h = \lfloor 4/2 \rfloor = 2$, $\text{end}_h = \lceil 8/2 \rceil = 4$. Because the division is exact, the ceil and floor coincide and the two regions $[0{:}2]$ and $[2{:}4]$ tile the axis with no overlap, identical to a stride-2 window-2 average pool. The four $2 \times 2$ means are $15/4 = 3.75$, $9/4 = 2.25$, $18/4 = 4.5$, $16/4 = 4.0$:</span>

$$
\text{out} = \begin{pmatrix} 3.75 & 2.25 \\ 4.5 & 4.0 \end{pmatrix}
$$

---

## <span style="font-size: 16px;">Numerical Example: Uneven Division</span>

<span style="font-size: 14px;">The interesting case is when sizes do not divide. Take a length-5 row $[1, 2, 3, 4, 5]$ pooled to $\text{out} = 3$ along that axis. The region boundaries are:</span>

<span style="font-size: 14px;">1. **$i = 0$:** $\text{start} = \lfloor 0 \cdot 5 / 3 \rfloor = 0$, $\text{end} = \lceil 1 \cdot 5 / 3 \rceil = \lceil 1.67 \rceil = 2$. Region $[1, 2]$, mean $1.5$.</span>

<span style="font-size: 14px;">2. **$i = 1$:** $\text{start} = \lfloor 5/3 \rfloor = 1$, $\text{end} = \lceil 10/3 \rceil = \lceil 3.33 \rceil = 4$. Region $[2, 3, 4]$, mean $3.0$.</span>

<span style="font-size: 14px;">3. **$i = 2$:** $\text{start} = \lfloor 10/3 \rfloor = 3$, $\text{end} = \lceil 15/3 \rceil = 5$. Region $[4, 5]$, mean $4.5$.</span>

<span style="font-size: 14px;">Result $[1.5, 3.0, 4.5]$. Note the regions are sizes $2, 3, 2$ (not equal) and they **overlap**: index $1$ appears in both region 0 and region 1, and index $3$ appears in both region 1 and region 2. The floor/ceil rule produces exactly this overlapping, variable-size tiling, which is why a naive equal-split implementation diverges from PyTorch.</span>

<span style="font-size: 14px;">It is instructive to contrast this with a wrong equal-split guess. Splitting $[1, 2, 3, 4, 5]$ into three "equal" chunks of sizes $2, 2, 1$ would give means $1.5, 3.5, 5.0$, which differs from the correct $1.5, 3.0, 4.5$ in two of three cells. The discrepancy is not a rounding artifact, it is a structural consequence of the overlap and the centred coverage that the floor/ceil rule enforces. Any implementation that does not reproduce the exact boundary arithmetic will fail to match the reference on every non-dividing case.</span>

---

## <span style="font-size: 16px;">Parameters, FLOPs, and Backward Pass</span>

* <span style="font-size: 14px;">**Parameters:** zero, like all pooling. The boundaries are derived arithmetically from the two shapes, so nothing is stored or trained.</span>
* <span style="font-size: 14px;">**FLOPs:** each output cell costs (region size minus one) additions plus one division. Since regions tile the input with bounded overlap, the total is on the order of $C \cdot H \cdot W$ additions, independent of the output grid size.</span>
* <span style="font-size: 14px;">**Backward pass:** within each region the gradient is spread uniformly, $1/(\text{region size})$ to each member, exactly as in average pooling. Where regions overlap, an input that belongs to two regions accumulates a contribution from each, so its total gradient can exceed that of a non-shared neighbour. This is the analogue of overlapping-window gradient accumulation.</span>

<span style="font-size: 14px;">In the $1 \times 1$ global-pool case the backward pass simplifies further: the single output per channel sends $1/(H \cdot W)$ of its gradient back to every spatial location uniformly. This even spreading is one reason GAP is gentle to train and does not create the sharp, sparse gradient pathways that a global max pool would.</span>

---

## <span style="font-size: 16px;">Two-Dimensional Separability</span>

<span style="font-size: 14px;">The height and width region computations are fully independent: the formula for $\text{start}_h, \text{end}_h$ uses only $i, H, \text{out}_h$, and the width formula uses only $j, W, \text{out}_w$. The two axes are therefore **separable**, and the 2D region for output cell $(i, j)$ is simply the Cartesian product of the 1D height region and the 1D width region.</span>

<span style="font-size: 14px;">This separability is useful both conceptually and computationally. A correct implementation precomputes the $\text{out}_h$ height intervals and the $\text{out}_w$ width intervals once, then for each output cell takes the mean over the rectangle formed by pairing one height interval with one width interval. There is no need to recompute boundaries inside the inner loop, and verifying correctness reduces to verifying the 1D boundary logic on each axis in isolation.</span>

<span style="font-size: 14px;">It also means the uneven-division behaviour can occur on one axis while the other divides cleanly. A $10 \times 8$ input pooled to $(3, 4)$ has uneven, overlapping regions along height but clean size-2 regions along width, and the two behaviours coexist without interacting.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Assuming equal-size regions.** The most common error is splitting the input into $\text{out}_h$ equal chunks. This only matches PyTorch when $H$ divides $\text{out}_h$ evenly. For uneven sizes the floor/ceil rule yields regions of differing size that overlap, and an equal-split implementation produces wrong values at the boundaries.</span>
* <span style="font-size: 14px;">**Swapping floor and ceil.** The start must use floor and the end must use ceil. Reversing them (ceil on start, floor on end) can leave input rows uncovered or produce empty regions, breaking coverage and causing division by zero on a zero-width region.</span>
* <span style="font-size: 14px;">**Dividing by a fixed window area.** Because region sizes vary, the divisor must be the actual element count of each region, not a constant $k^2$. Hard-coding the divisor biases every uneven cell.</span>
* <span style="font-size: 14px;">**Forgetting it is per-channel.** Adaptive pooling reduces only the spatial axes; the channel count is preserved. Reducing over channels would collapse feature identity and destroy the representation that the classifier head depends on, and would produce a tensor of the wrong shape for the following linear layer.</span>

---