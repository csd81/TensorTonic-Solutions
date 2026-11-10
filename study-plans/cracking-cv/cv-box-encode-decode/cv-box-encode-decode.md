# <span style="font-size: 20px;">Box Encode and Decode</span>

<span style="font-size: 14px;">**Bounding-box encoding** is the parametrization that turns a localization problem into a well-conditioned regression problem. Instead of predicting absolute box coordinates, a detector predicts four offsets $(t_x, t_y, t_w, t_h)$ that describe how to transform a reference box (a proposal or anchor) into the ground-truth box. This parametrization was introduced by R-CNN and refined in Faster R-CNN (Ren et al., 2015), and it is the form used by essentially every modern anchor-based detector.</span>

---

## <span style="font-size: 16px;">Why Encode Offsets Instead of Coordinates</span>

<span style="font-size: 14px;">A detection head outputs the same four numbers for every anchor regardless of where that anchor sits or how large it is. If those four numbers were absolute coordinates, the network would have to memorize a different mapping for each of the hundred thousand anchors, since an object at the top-left and an identical object at the bottom-right would require completely different outputs. Encoding offsets **relative to the anchor** makes the regression target translation-invariant and scale-normalized, so one shared set of weights works for every anchor.</span>

<span style="font-size: 14px;">The key property the encoding provides is that the targets are dimensionless and roughly zero-centered with unit variance. A perfectly placed anchor has all four deltas equal to zero. Small corrections produce small deltas. This is exactly the regime in which a smooth-L1 regression loss behaves well, and it is why detectors train stably even though raw box coordinates span hundreds of pixels.</span>

---

## <span style="font-size: 16px;">Center-Size Form</span>

<span style="font-size: 14px;">Boxes arrive in `xyxy` corner format but the encoding operates in **center-size** form $(c_x, c_y, w, h)$. For a box $[x_1, y_1, x_2, y_2]$:</span>

$$
w = x_2 - x_1, \quad h = y_2 - y_1, \quad c_x = x_1 + w/2, \quad c_y = y_1 + h/2
$$

<span style="font-size: 14px;">Center-size form separates the two things a correction must adjust: where the box is (its center) and how big it is (its width and height). The position deltas $t_x, t_y$ act on the center, and the size deltas $t_w, t_h$ act on the dimensions. Doing this in corner space would entangle position and size, since moving $x_1$ alone changes both the center and the width.</span>

---

## <span style="font-size: 16px;">The Encoding Equations</span>

<span style="font-size: 14px;">Given a proposal $p$ and a ground-truth box $g$ in center-size form, the four encoding deltas are:</span>

$$
t_x = \frac{c_x^g - c_x^p}{w^p}, \quad t_y = \frac{c_y^g - c_y^p}{h^p}, \quad t_w = \log\!\left(\frac{w^g}{w^p}\right), \quad t_h = \log\!\left(\frac{h^g}{h^p}\right)
$$

<span style="font-size: 14px;">Two design choices are doing the work here. First, the **center offsets are normalized by the proposal's size**: $t_x$ is the horizontal shift measured in units of proposal widths, not pixels. This makes the same delta mean a proportionally larger move for a large box than a small one, matching the scale-invariant nature of IoU and ensuring a fixed-magnitude prediction error has a consistent effect on overlap across object sizes.</span>

<span style="font-size: 14px;">Second, the **size deltas are in log space**. The ratio $w^g / w^p$ is always positive, and its logarithm maps the multiplicative interval $(0, \infty)$ onto the additive line $(-\infty, \infty)$. A delta of $0$ means no size change, $+\log 2$ means double the width, $-\log 2$ means halve it. Log space makes growing and shrinking symmetric (a factor of $2$ up and down are equal in magnitude), keeps the target unbounded so any size change is representable, and guarantees the decoded width stays positive because $e^{t_w} > 0$ always.</span>

---

## <span style="font-size: 16px;">The Decoding Equations</span>

<span style="font-size: 14px;">Decoding is the exact inverse: it applies the predicted deltas to the proposal to recover a box. In center-size form:</span>

$$
c_x^d = c_x^p + t_x \cdot w^p, \quad c_y^d = c_y^p + t_y \cdot h^p
$$

$$
w^d = w^p \cdot e^{t_w}, \quad h^d = h^p \cdot e^{t_h}
$$

<span style="font-size: 14px;">Each operation undoes its counterpart. The center decode multiplies the normalized offset back by the proposal size and adds it to the proposal center; the size decode exponentiates the log-ratio and multiplies by the proposal dimension. The result is then converted back to `xyxy`:</span>

$$
[\,c_x^d - w^d/2,\ c_y^d - h^d/2,\ c_x^d + w^d/2,\ c_y^d + h^d/2\,]
$$

<span style="font-size: 14px;">Because encode and decode are mathematical inverses, decoding the deltas that were encoded from a ground-truth box must return that same ground-truth box up to floating-point precision. This round-trip identity is a strong correctness check on any implementation.</span>

---

## Worked Example (one box pair)

<span style="font-size: 14px;">Let the proposal be $p = [10, 10, 30, 30]$ and the ground truth $g = [12, 14, 36, 38]$.</span>

<span style="font-size: 14px;">**Proposal center-size**: $w^p = 20$, $h^p = 20$, $c_x^p = 20$, $c_y^p = 20$.</span>

<span style="font-size: 14px;">**Ground-truth center-size**: $w^g = 24$, $h^g = 24$, $c_x^g = 24$, $c_y^g = 26$.</span>

<span style="font-size: 14px;">**Encode**: $t_x = (24 - 20)/20 = 0.2$, $t_y = (26 - 20)/20 = 0.3$, $t_w = \log(24/20) = \log 1.2 \approx 0.1823$, $t_h = \log(24/20) \approx 0.1823$.</span>

<span style="font-size: 14px;">**Decode** back onto $p$: $c_x^d = 20 + 0.2 \cdot 20 = 24$, $c_y^d = 20 + 0.3 \cdot 20 = 26$, $w^d = 20 \cdot e^{0.1823} = 20 \cdot 1.2 = 24$, $h^d = 24$.</span>

<span style="font-size: 14px;">**Convert to `xyxy`**: $[24 - 12,\ 26 - 12,\ 24 + 12,\ 26 + 12] = [12, 14, 36, 38]$, which is exactly $g$. The round trip recovers the ground-truth box, confirming the inverse relationship.</span>

<span style="font-size: 14px;">Notice the deltas are small and dimensionless: a 4-pixel center shift on a 20-pixel proposal is $0.2$, and a $20\%$ size increase is $\log 1.2 \approx 0.18$. Both land comfortably in the quadratic region of the smooth-L1 loss. Had the proposal been twice as large, the same 4-pixel shift would encode to $0.1$, illustrating how the size normalization makes the target depend on relative rather than absolute displacement.</span>

---

## <span style="font-size: 16px;">Anchors versus Proposals as the Reference</span>

<span style="font-size: 14px;">The reference box that the deltas transform can be either an anchor or a proposal, depending on the stage. In a one-stage detector like RetinaNet or SSD, the reference is the fixed anchor, and the single regression step decodes deltas onto anchors to get final boxes. In a two-stage detector the reference changes between stages: the Region Proposal Network encodes relative to anchors, and the second-stage R-CNN head encodes relative to the proposals the RPN produced.</span>

<span style="font-size: 14px;">The encoding math is identical in both cases; only the reference box differs. This is what makes the parametrization composable. The RPN refines anchors into proposals, and the head then treats those proposals as fresh references and refines them again. Each refinement uses the same center-normalized, log-size encoding, so a single well-understood primitive serves the whole cascade.</span>

---

## <span style="font-size: 16px;">Smooth-L1 and the Shape of the Loss</span>

<span style="font-size: 14px;">The regression loss applied to the deltas is smooth-L1, defined per component as a quadratic for small residuals and a linear function beyond a threshold $\beta$:</span>

$$
\text{smooth}_{L_1}(x) = \begin{cases} 0.5\,x^2/\beta & |x| < \beta \\ |x| - 0.5\,\beta & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">The quadratic region gives a smoothly vanishing gradient as the prediction approaches the target, while the linear region caps the gradient magnitude for large errors, preventing a few badly matched boxes from dominating the update. This pairs naturally with the encoding: because the targets are zero-centered and scale-normalized, most residuals fall in the quadratic region during normal training, and only hard or mislabeled examples reach the linear tail.</span>

---

## <span style="font-size: 16px;">Variance Normalization in Practice</span>

<span style="font-size: 14px;">Real detectors add a fixed per-component scaling, often called box variances or weights, dividing each delta by a constant (commonly $(0.1, 0.1, 0.2, 0.2)$ in SSD or $(10, 10, 5, 5)$ as multipliers in Faster R-CNN code). This rescales the four targets so they have comparable magnitude and roughly unit variance, which balances the regression loss across the four outputs. The plain form above omits these constants, but conceptually they are just a fixed linear reweighting applied after encoding and undone before decoding; they do not change the structure of the parametrization.</span>

<span style="font-size: 14px;">Without this balancing, the log-space size deltas (typically small) and the normalized center deltas (which can be larger) contribute unequally to the loss, and the network over-optimizes one at the expense of the other. The variances are a cheap way to equalize the gradient contributions of position and size, and they are a fixed constant rather than a learned parameter, so they add no model capacity.</span>

---

## <span style="font-size: 16px;">Where Encoding Sits in the Pipeline</span>

<span style="font-size: 14px;">During training, every positive anchor is matched to a ground-truth box by IoU, and the encoding produces the regression target the head is trained to predict with a smooth-L1 (Huber) loss. The smooth-L1 loss is quadratic for small errors and linear for large ones, which suppresses the influence of outlier boxes whose deltas are large, a property that pairs naturally with the zero-centered encoded targets.</span>

<span style="font-size: 14px;">At inference, the head emits raw deltas for every anchor, decoding turns them into candidate boxes, and Non-Maximum Suppression deduplicates the result. The same encode/decode pair is reused at both stages of two-stage detectors: once in the Region Proposal Network to refine anchors into proposals, and again in the second-stage head to refine proposals into final detections. Each stage typically learns its own variance constants.</span>

---

## <span style="font-size: 16px;">Why Log Space Specifically</span>

<span style="font-size: 14px;">An alternative size parametrization is the raw ratio $w^g / w^p$ or the difference $w^g - w^p$. Both are inferior. The raw difference is not scale-normalized, so a 10-pixel size error means something different for a small box than a large one. The raw ratio is bounded below by zero, so the network could predict a negative or near-zero ratio and produce a degenerate box; it is also asymmetric, since doubling ($2.0$) and halving ($0.5$) sit at different distances from the no-change value of $1.0$.</span>

<span style="font-size: 14px;">The logarithm fixes all three issues at once: it is scale-normalized (a multiplicative change), symmetric about zero, and unbounded so the exponential decode always yields a strictly positive dimension. This is the precise reason Faster R-CNN chose $\log(w^g/w^p)$ rather than any simpler size encoding.</span>

---

## <span style="font-size: 16px;">Numerical Properties of the Round Trip</span>

<span style="font-size: 14px;">Encode followed by decode is an exact algebraic identity, but in floating point the two directions accumulate tiny rounding differences. The center path involves one division then one multiplication, which nearly cancel; the size path involves a logarithm then an exponential, which are inverse transcendental functions whose composition is identity only up to machine precision. In practice the recovered box matches the ground truth to within roughly $10^{-6}$ relative error in float32, which is why correctness is checked up to floating-point precision rather than exact equality.</span>

<span style="font-size: 14px;">The division by $w^p$ and $h^p$ also means proposals must have strictly positive width and height. A degenerate proposal with zero width makes $t_x$ undefined and $\log(w^g/0)$ infinite. Matching by IoU normally prevents degenerate references from being assigned targets, but data-loading bugs that emit zero-area boxes will surface here as NaN deltas that then poison the loss.</span>

---

## <span style="font-size: 16px;">Connection to the Broader Detector</span>

<span style="font-size: 14px;">The encode/decode pair is one of three primitives, together with IoU and NMS, that every anchor-based detector composes. IoU decides which anchor matches which object and therefore what target each anchor is encoded against; encoding turns that match into a trainable regression target; and NMS cleans up the decoded predictions at inference. A subtle error in any one of them degrades the others: a wrong IoU match assigns the wrong target to encode, and a wrong decode produces boxes that NMS then suppresses incorrectly.</span>

<span style="font-size: 14px;">This composability is why Faster R-CNN's contribution was as much about the parametrization as the architecture. The paper showed that learning offsets in this normalized space, shared across a dense anchor grid, made end-to-end training of a region proposal network feasible for the first time, replacing the slow external proposal generators used by earlier R-CNN variants.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Encoding in corner space instead of center-size.** Computing deltas directly on $x_1, y_1, x_2, y_2$ entangles position and size and is not the Faster R-CNN parametrization. Always convert to $(c_x, c_y, w, h)$ first; mixing conventions silently produces wrong targets that still train but localize poorly.</span>
* <span style="font-size: 14px;">**Forgetting to normalize the center offset by the proposal size.** Using $t_x = c_x^g - c_x^p$ in raw pixels instead of dividing by $w^p$ removes scale-invariance, so the same predicted error distorts small and large boxes unequally and recall on small objects collapses.</span>
* <span style="font-size: 14px;">**Applying $\exp$ to an unclipped size delta.** A large predicted $t_w$ exponentiates to an enormous width, producing absurd boxes and occasionally overflow. Production decoders clamp $t_w, t_h$ (for example to $\log(1000/16)$) before the exponential to bound the maximum decoded size.</span>
* <span style="font-size: 14px;">**Mismatched variance constants between encode and decode.** If training encodes with one set of variances and inference decodes with another (or none), every predicted box is rescaled incorrectly. The encode and decode steps must use identical constants, and they must match whatever the trained weights expect.</span>

---