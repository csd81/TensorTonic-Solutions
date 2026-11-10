# <span style="font-size: 20px;">Sigmoid Focal Loss</span>

<span style="font-size: 14px;">Focal loss (Lin et al., 2017, "Focal Loss for Dense Object Detection", the RetinaNet paper) is a reshaping of binary cross-entropy that down-weights well-classified examples so training focuses on a sparse set of hard ones. It was introduced to let single-stage dense detectors match the accuracy of two-stage detectors like Faster R-CNN.</span>

---

## <span style="font-size: 16px;">The Problem It Solves</span>

<span style="font-size: 14px;">Dense detectors (RetinaNet, SSD) evaluate a fixed grid of $10^4$ to $10^5$ candidate anchor boxes per image. The vast majority are easy background (negatives), and only a handful contain objects (positives). This creates an extreme **foreground-background class imbalance**, typically on the order of 1000:1.</span>

<span style="font-size: 14px;">Two failure modes follow from naive cross-entropy training:</span>

* <span style="font-size: 14px;">**Training is inefficient.** Most anchors are easy negatives that contribute no useful learning signal individually, yet there are so many of them that they dominate.</span>
* <span style="font-size: 14px;">**The loss is overwhelmed.** Summed over thousands of easy negatives, their small individual losses collectively swamp the gradient from the few informative positives, steering the model toward degenerate solutions.</span>

<span style="font-size: 14px;">Two-stage detectors sidestep this with a region proposal stage that filters most background before classification, plus heuristics like fixed foreground-background sampling ratios. The RetinaNet paper argues these heuristics are a crutch and proposes fixing the loss itself instead, so a single dense pass can be trained end to end.</span>

---

## <span style="font-size: 16px;">From Cross-Entropy to Focal Loss</span>

<span style="font-size: 14px;">Start from binary cross-entropy. For a single example with label $y \in \{0, 1\}$ and predicted probability $p = \sigma(z)$ for the positive class, define the probability of the **true** class:</span>

$$
p_t = p \cdot y + (1 - p)(1 - y)
$$

<span style="font-size: 14px;">so $p_t = p$ when $y = 1$ and $p_t = 1 - p$ when $y = 0$. Cross-entropy is then simply:</span>

$$
\mathrm{CE}(p_t) = -\log(p_t)
$$

<span style="font-size: 14px;">The key observation is that even when $p_t$ is large (say $0.9$, an example already classified correctly), $\mathrm{CE} = -\log(0.9) \approx 0.105$ is still non-trivial. Multiplied by thousands of easy negatives, these "small" losses dominate. Focal loss multiplies cross-entropy by a **modulating factor** $(1 - p_t)^\gamma$:</span>

$$
\mathrm{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)
$$

<span style="font-size: 14px;">The tunable **focusing parameter** $\gamma \ge 0$ controls how aggressively easy examples are suppressed. With $\gamma = 0$, focal loss reduces exactly to cross-entropy. The paper found $\gamma = 2$ works best in practice.</span>

---

## <span style="font-size: 16px;">How the Modulating Factor Works</span>

<span style="font-size: 14px;">The factor $(1 - p_t)^\gamma$ behaves as a soft, per-example weight that depends on how well the example is currently classified:</span>

* <span style="font-size: 14px;">**Easy examples** have $p_t \to 1$, so $(1 - p_t)^\gamma \to 0$ and their loss is driven near zero. With $\gamma = 2$ and $p_t = 0.9$, the factor is $(0.1)^2 = 0.01$, a 100x down-weighting.</span>
* <span style="font-size: 14px;">**Hard examples** have $p_t$ small, so $(1 - p_t)^\gamma \to 1$ and their loss is essentially unchanged. With $p_t = 0.1$, the factor is $(0.9)^2 = 0.81$, almost full weight.</span>

<span style="font-size: 14px;">The effect is relative, not absolute: hard examples retain nearly their full loss while easy ones are suppressed, so as the model improves on an example its contribution automatically fades. This continuously reallocates the gradient budget toward the shrinking set of still-misclassified examples.</span>

<span style="font-size: 14px;">The choice of $\gamma$ trades off how sharply this happens. Larger $\gamma$ extends the range over which loss is reduced and increases the suppression of easy examples: at $\gamma = 5$ even moderately confident examples are heavily down-weighted, while at $\gamma = 0$ nothing changes. The paper sweeps $\gamma \in [0, 5]$ and finds accuracy peaks around $\gamma = 2$, with the benefit relatively robust in the range $[0.5, 5]$.</span>

---

## <span style="font-size: 16px;">The Alpha Balancing Term</span>

<span style="font-size: 14px;">Modulation handles the easy-vs-hard axis but not the raw positive-vs-negative count imbalance. RetinaNet adds a class-balancing weight $\alpha_t$, defined analogously to $p_t$:</span>

$$
\alpha_t = \alpha \cdot y + (1 - \alpha)(1 - y)
$$

<span style="font-size: 14px;">giving the full $\alpha$-balanced focal loss:</span>

$$
\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

<span style="font-size: 14px;">Here $\alpha \in [0, 1]$ weights positives and $1 - \alpha$ weights negatives. Counterintuitively the paper uses $\alpha = 0.25$, which puts **less** weight on the rare positives. The reason is that the $\gamma$ term already shifts emphasis onto positives (which tend to be the hard examples), so $\alpha$ is tuned jointly with $\gamma$ and a value below $0.5$ best balances the two effects. The paper reports $\alpha = 0.25$, $\gamma = 2$ as the best joint setting.</span>

---

## <span style="font-size: 16px;">Why Sigmoid and Not Softmax</span>

<span style="font-size: 14px;">RetinaNet performs classification with $K$ **independent sigmoids**, one per class, rather than a single $(K+1)$-way softmax that includes an explicit background class. Each anchor produces $K$ binary focal-loss terms. There are two reasons the paper prefers this:</span>

* <span style="font-size: 14px;">**No explicit background class.** Background is represented by all $K$ sigmoid outputs being near zero, so there is no competition that forces probability mass onto a background logit. This meshes naturally with the prior-probability initialization that sets every output near $0.01$.</span>
* <span style="font-size: 14px;">**Empirically more stable** with focal loss, and the paper notes results are insensitive to the exact form (sigmoid vs softmax focal loss) as long as the modulation and balancing are applied. The binary form also generalizes cleanly to multi-label settings where a region can belong to more than one class.</span>

<span style="font-size: 14px;">This is why the canonical API is named **sigmoid** focal loss and operates element-wise on $(N, K)$ logits with $\{0, 1\}$ targets, summing or averaging over all elements.</span>

---

## <span style="font-size: 16px;">Why It Works: The Gradient</span>

<span style="font-size: 14px;">The intuition is clearest in the gradient with respect to the logit $z$. For plain cross-entropy with $y = 1$, the gradient is the clean residual $\partial \mathrm{CE} / \partial z = p - 1$, whose magnitude $1 - p$ stays appreciable even for confident-correct predictions. Focal loss multiplies this signal by the modulating factor (plus a smaller correction term from differentiating $(1 - p_t)^\gamma$), so the gradient contributed by an easy example is suppressed by roughly $(1 - p_t)^\gamma$.</span>

<span style="font-size: 14px;">Summed over a batch, the total gradient is therefore dominated by hard examples rather than by the sheer count of easy negatives. The network spends its capacity learning the decision boundary where it is actually wrong, which is precisely what fixed sampling heuristics try to achieve manually.</span>

---

## <span style="font-size: 16px;">Stable Implementation From Logits</span>

<span style="font-size: 14px;">Production implementations such as `torchvision.ops.sigmoid_focal_loss` operate on **raw logits** $z$, never on probabilities, to preserve numerical stability. Computing $\log(\sigma(z))$ naively underflows for large negative $z$; the fused `binary_cross_entropy_with_logits` form uses the log-sum-exp trick internally. The standard recipe is:</span>

<span style="font-size: 14px;">1. **Probability**: $p = \sigma(z) = 1 / (1 + e^{-z})$.</span>

<span style="font-size: 14px;">2. **Cross-entropy**: compute $\mathrm{ce} = \text{BCE\_with\_logits}(z, y)$ in a numerically stable fused op.</span>

<span style="font-size: 14px;">3. **True-class probability**: $p_t = p y + (1 - p)(1 - y)$.</span>

<span style="font-size: 14px;">4. **Loss**: $\ell = \mathrm{ce} \cdot (1 - p_t)^\gamma$, then multiply by $\alpha_t$ if $\alpha \ge 0$.</span>

<span style="font-size: 14px;">5. **Reduce**: with `reduction="mean"`, return the mean of $\ell$ over all $N$ elements.</span>

---

## <span style="font-size: 16px;">Comparison With Alternatives</span>

<span style="font-size: 14px;">Before focal loss, the standard ways to handle imbalance in detection were sampling heuristics, and the RetinaNet paper benchmarks against them:</span>

* <span style="font-size: 14px;">**Hard Negative Mining (OHEM, Shrivastava et al. 2016).** Explicitly selects the highest-loss examples and discards the rest. This is a hard, discontinuous version of focal loss: it throws easy examples away entirely rather than smoothly down-weighting them, and it fully discards the (still small) gradient from easy negatives that focal loss preserves.</span>
* <span style="font-size: 14px;">**Fixed foreground-background ratios** (e.g. 1:3 in SSD/Faster R-CNN). A coarse, dataset-independent heuristic that ignores per-example difficulty.</span>
* <span style="font-size: 14px;">**Plain $\alpha$-balanced cross-entropy.** Corrects only the positive-negative count imbalance, not the easy-hard imbalance. The paper shows it alone is insufficient.</span>

<span style="font-size: 14px;">RetinaNet with focal loss reached state-of-the-art COCO AP while remaining a simple single-stage network, demonstrating that the imbalance problem was a loss-design problem, not an architecture problem.</span>

---

## <span style="font-size: 16px;">The Prior Probability Initialization</span>

<span style="font-size: 14px;">A subtle but important detail from the paper: at the very start of training, an untrained dense detector predicts $p \approx 0.5$ for every one of the $\sim10^5$ anchors. Because almost all are negatives, the summed loss is enormous and can destabilize the first iterations. RetinaNet initializes the bias of the final classification layer so that the predicted foreground probability starts at a small prior $\pi$ (e.g. $\pi = 0.01$), via $b = -\log((1 - \pi)/\pi)$. This makes the network start out "expecting background", drastically reducing the initial loss and stabilizing early training. This trick is what makes focal-loss training converge at all.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 2$ with $\alpha = 0.25$, $\gamma = 2$. Element 0: logit $z_0 = 2.0$, label $y_0 = 1$ (a confident correct positive). Element 1: logit $z_1 = -1.0$, label $y_1 = 0$ (a confident correct negative).</span>

<span style="font-size: 14px;">**Element 0**: $p_0 = \sigma(2.0) = 0.8808$. Since $y_0 = 1$: $\mathrm{ce}_0 = -\log(0.8808) = 0.1269$, $p_{t,0} = 0.8808$, $\alpha_{t,0} = 0.25$. Modulating factor $(1 - 0.8808)^2 = (0.1192)^2 = 0.01421$. So $\ell_0 = 0.25 \cdot 0.01421 \cdot 0.1269 = 0.000451$.</span>

<span style="font-size: 14px;">**Element 1**: $p_1 = \sigma(-1.0) = 0.2689$. Since $y_1 = 0$: $\mathrm{ce}_1 = -\log(1 - 0.2689) = -\log(0.7311) = 0.3133$, $p_{t,1} = 1 - 0.2689 = 0.7311$, $\alpha_{t,1} = 0.75$. Factor $(1 - 0.7311)^2 = (0.2689)^2 = 0.07231$. So $\ell_1 = 0.75 \cdot 0.07231 \cdot 0.3133 = 0.01699$.</span>

<span style="font-size: 14px;">**Mean**: $(0.000451 + 0.01699) / 2 = 0.00872$. Both are easy examples, so the modulating factor drives the total loss far below what cross-entropy alone would give.</span>

---

## <span style="font-size: 16px;">Descendants and Modern Context</span>

<span style="font-size: 14px;">Focal loss became the default classification loss for dense detectors and spawned a family of refinements:</span>

* <span style="font-size: 14px;">**Generalized Focal Loss (Li et al. 2020)** extends the discrete $\{0,1\}$ target to continuous quality scores (the IoU-aware classification score), unifying classification and localization-quality estimation under one focal-style loss.</span>
* <span style="font-size: 14px;">**Varifocal Loss (VarifocalNet, 2021)** makes the modulation asymmetric, only down-weighting negatives while leaving positives at full weight, to better rank detections by IoU.</span>
* <span style="font-size: 14px;">**Quality / IoU-aware variants** in detectors like FCOS and ATSS keep focal loss for classification while adding separate centerness or IoU branches.</span>

<span style="font-size: 14px;">Outside detection, focal loss is widely reused anywhere a sigmoid/BCE head faces heavy class imbalance: medical segmentation, long-tailed multi-label classification, and keypoint heatmap regression. In every case the core idea is identical, namely reshape BCE so the loss landscape is dominated by hard examples rather than by example count.</span>

---

## <span style="font-size: 16px;">Properties to Keep in Mind</span>

* <span style="font-size: 14px;">**Reduces to BCE.** At $\gamma = 0$ and $\alpha = 0.5$ (or $\alpha$ disabled), focal loss is exactly binary cross-entropy, so it is a strict generalization.</span>
* <span style="font-size: 14px;">**Non-negative and bounded below by 0.** Each per-element term is a product of non-negative factors times $-\log(p_t) \ge 0$.</span>
* <span style="font-size: 14px;">**Always less than or equal to $\alpha_t \cdot \mathrm{CE}$** since $(1 - p_t)^\gamma \le 1$, so it can never increase the loss of any single example relative to balanced cross-entropy; it only redistributes emphasis.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Passing probabilities instead of logits.** The torchvision API and most reference implementations expect raw pre-sigmoid logits and apply $\sigma$ internally. Feeding already-sigmoided values double-applies the nonlinearity and silently produces wrong losses with no crash.</span>
* <span style="font-size: 14px;">**Detaching or stop-gradient on the modulating factor.** $(1 - p_t)^\gamma$ depends on $p_t$, hence on the logits. Some assume it is a constant weight and block its gradient. While the dominant signal comes through $\mathrm{ce}$, the official formulation keeps the factor differentiable; treating it as a frozen weight changes the optimization.</span>
* <span style="font-size: 14px;">**Wrong $\alpha$ direction.** It is tempting to set $\alpha > 0.5$ to "up-weight" rare positives, but RetinaNet uses $\alpha = 0.25$ because $\gamma$ already emphasizes positives. The two must be tuned jointly; flipping $\alpha$ degrades results.</span>
* <span style="font-size: 14px;">**Normalizing the mean by the wrong count.** RetinaNet normalizes the total loss by the number of **assigned anchors (positives)**, not by all anchors, since easy-negative losses are near zero and would otherwise shrink the mean. The torchvision `"mean"` reduction divides by all $N$ elements, so match whichever convention the task specifies.</span>

---