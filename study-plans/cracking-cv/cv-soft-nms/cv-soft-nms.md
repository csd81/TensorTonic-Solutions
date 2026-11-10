# <span style="font-size: 20px;">Soft Non-Maximum Suppression</span>

<span style="font-size: 14px;">**Soft-NMS** (Bodla et al., 2017) is a drop-in replacement for greedy NMS that decays the scores of overlapping detections instead of deleting them outright. Where hard NMS makes a binary keep-or-kill decision at an IoU threshold, Soft-NMS reduces a neighbour's confidence by a continuous function of its overlap with the just-selected box. This single change recovers detections in crowded scenes that hard NMS would erase, improving recall and mAP with no model retraining at all.</span>

---

## <span style="font-size: 16px;">The Problem Soft-NMS Solves</span>

<span style="font-size: 14px;">Greedy NMS deletes any box whose IoU with a kept box exceeds a threshold. This is fatal in crowded scenes: when two distinct objects genuinely overlap more than the threshold, the lower-scoring object's box is removed even though it is a correct detection. Lowering the threshold to spare such boxes instead lets duplicates through, so a single hard threshold cannot simultaneously suppress duplicates and preserve overlapping true positives.</span>

<span style="font-size: 14px;">Bodla et al. observed that the hard cutoff is the culprit: a box with IoU $0.51$ is deleted while one at $0.49$ survives, despite being nearly identical situations. Their fix replaces the step function with a smooth penalty. A box that overlaps a lot has its score reduced a lot; a box that barely overlaps is almost untouched. No box is ever deleted by overlap alone; instead, decayed scores eventually fall below a final score threshold and drop out naturally.</span>

---

## <span style="font-size: 16px;">The Gaussian Decay Rule</span>

<span style="font-size: 14px;">The Gaussian variant of Soft-NMS, the form implemented here, decays each remaining candidate's score by a Gaussian function of its IoU with the just-selected box $b_i$:</span>

$$
s_j \leftarrow s_j \cdot \exp\!\left(-\dfrac{\text{IoU}(b_i, b_j)^2}{\sigma}\right)
$$

<span style="font-size: 14px;">When $\text{IoU} = 0$ the multiplier is $e^0 = 1$ and the score is unchanged; as IoU grows toward $1$ the multiplier shrinks smoothly toward $e^{-1/\sigma}$. The parameter $\sigma$ controls how aggressive the decay is: a small $\sigma$ produces a sharp penalty that approaches hard NMS, while a large $\sigma$ produces a gentle decay that barely suppresses. The Gaussian is continuous and differentiable in IoU, which is what removes the brittle threshold behaviour of hard suppression.</span>

<span style="font-size: 14px;">The original paper also proposed a **linear** variant, $s_j \leftarrow s_j (1 - \text{IoU})$ applied only above a threshold. The Gaussian form is generally preferred because it has no discontinuity at any IoU value, so its behaviour does not jump at a boundary, whereas the linear variant still has a kink at its rescore threshold.</span>

---

## <span style="font-size: 16px;">The Iterative Procedure</span>

<span style="font-size: 14px;">Soft-NMS reuses the greedy outer loop of hard NMS but changes the inner update from deletion to decay:</span>

<span style="font-size: 14px;">1. **Select** the candidate $i$ with the largest current score; break ties by lowest index. Remove $i$ from the candidate pool.</span>

<span style="font-size: 14px;">2. **Emit** the pair $(i, \text{round}(s_i, 4))$ to the output. Rounding happens only here, at pick time, when the entry enters the output.</span>

<span style="font-size: 14px;">3. **Decay** every remaining candidate $j$ by the Gaussian of $\text{IoU}(b_i, b_j)$, updating $s_j$ in place with the unrounded value.</span>

<span style="font-size: 14px;">4. **Repeat** until no candidate has a score strictly greater than $\texttt{score\_threshold}$.</span>

<span style="font-size: 14px;">The output is a list of $[\texttt{int\_index}, \texttt{float\_decayed\_score}]$ pairs in pick order. A crucial detail: because scores change as the algorithm runs, the candidate selected at each step depends on all prior decays, so the pick order is not simply the original score order.</span>

---

## <span style="font-size: 16px;">Rounding and Internal Precision</span>

<span style="font-size: 14px;">The specification draws a sharp line between internal state and emitted output. Internal score updates use full unrounded precision: each decay multiplies the exact current $s_j$ by the exact Gaussian factor, and these compounded products must not be rounded between steps. Only when an index is selected and enters the output is its score rounded to 4 decimals.</span>

<span style="font-size: 14px;">This matters because a box can be decayed multiple times before it is ever selected, once for each higher-scoring box that overlaps it. Rounding after each decay would accumulate error across those multiplications and shift later selection decisions, since the choice of which candidate has the largest current score is sensitive to small differences. Keeping internal values exact and rounding only at emission keeps the selection sequence deterministic and matches the reference.</span>

---

## Worked Example ($\sigma = 0.5$, three boxes)

<span style="font-size: 14px;">Take box 0 $= [0,0,10,10]$ score $0.9$; box 1 $= [1,1,11,11]$ score $0.8$; box 2 $= [50,50,60,60]$ score $0.7$. Let $\texttt{score\_threshold} = 0.1$.</span>

<span style="font-size: 14px;">**Step 1.** Largest score is box 0 at $0.9$. Emit $(0, 0.9)$. Decay others. IoU(0, 1): intersection $9 \times 9 = 81$, union $200 - 81 = 119$, IoU $\approx 0.6807$. Multiplier $= \exp(-0.6807^2 / 0.5) = \exp(-0.9267) \approx 0.3959$. New $s_1 = 0.8 \cdot 0.3959 \approx 0.3167$. IoU(0, 2) $= 0$, so $s_2$ stays $0.7$.</span>

<span style="font-size: 14px;">**Step 2.** Current scores: $s_1 \approx 0.3167$, $s_2 = 0.7$. Largest is box 2 at $0.7$. Emit $(2, 0.7)$. IoU(2, 1) $= 0$, so $s_1$ unchanged at $0.3167$.</span>

<span style="font-size: 14px;">**Step 3.** Only box 1 remains at $\approx 0.3167 > 0.1$. Emit $(1, 0.3167)$.</span>

<span style="font-size: 14px;">**Done.** Output $= [[0, 0.9],\ [2, 0.7],\ [1, 0.3167]]$. Box 1 survives, unlike hard NMS which at threshold $0.5$ would have deleted it outright (IoU $0.68 > 0.5$). Its score is decayed to reflect the overlap, so a downstream confidence filter still ranks it below the clean detections.</span>

<span style="font-size: 14px;">The example also illustrates that pick order can differ from original score order through decay. Box 1 started as the second-highest at $0.8$, but after its decay to $0.3167$ it fell behind box 2 at $0.7$ and was selected last. Had box 1 not overlapped box 0, it would have been picked second. This reordering is the visible signature of Soft-NMS and the reason the algorithm must re-select the current maximum at every step rather than walk a fixed sorted list.</span>

---

## <span style="font-size: 16px;">Why Decay Beats Deletion</span>

<span style="font-size: 14px;">The insight is that overlap is evidence of redundancy, not proof of it. A high IoU usually means a duplicate, but sometimes means a second object. Hard NMS bets entirely on the duplicate interpretation and discards the box; Soft-NMS hedges by reducing the box's score in proportion to how likely it is to be a duplicate. A genuine second object that happens to overlap survives the decay with enough residual score to be reported, while a true duplicate is decayed so far that it falls below the final threshold and disappears anyway.</span>

<span style="font-size: 14px;">Bodla et al. reported consistent mAP gains (around $1{-}1.5$ points on COCO) from this change alone, applied to already-trained Faster R-CNN and R-FCN detectors with no retraining. The gain is largest on the crowded subset of images, exactly where the hard threshold fails. Because it is a pure post-processing swap, it costs nothing at training time and almost nothing at inference, and the same trained weights are reused unchanged.</span>

---

## <span style="font-size: 16px;">Effect of the Sigma Parameter</span>

<span style="font-size: 14px;">The bandwidth $\sigma$ is the only tuning knob in Gaussian Soft-NMS and it sets the trade-off between suppression strength and recall preservation. Because the decay factor is $\exp(-\text{IoU}^2/\sigma)$, the penalty at a given IoU is governed entirely by $\sigma$. At IoU $= 0.5$, for example, the multiplier is $\exp(-0.25/\sigma)$: with $\sigma = 0.5$ that is $\approx 0.61$, with $\sigma = 0.1$ it is $\approx 0.08$, and with $\sigma = 1.0$ it is $\approx 0.78$.</span>

<span style="font-size: 14px;">A small $\sigma$ thus makes Soft-NMS behave almost like hard NMS, crushing overlapping boxes to near-zero, while a large $\sigma$ barely touches them and risks leaving duplicates. The paper found $\sigma = 0.5$ a robust default on COCO. Crucially $\sigma$ is an inference-time hyperparameter, so it can be swept on a validation set without retraining, much like the IoU threshold in hard NMS.</span>

---

## <span style="font-size: 16px;">Compounding Decay Across Steps</span>

<span style="font-size: 14px;">A single box can be decayed several times before it is ever selected, once per higher-scoring box that overlaps it. The decays multiply: if a box overlaps two earlier selections with factors $f_1$ and $f_2$, its score becomes $s \cdot f_1 \cdot f_2$. This compounding is the mechanism by which boxes in a dense cluster of duplicates are driven below threshold: each successive kept box in the cluster shaves the survivor's score again.</span>

<span style="font-size: 14px;">It also explains why a box that overlaps only one true object survives while a box surrounded by many duplicates does not. A genuine second object overlaps just one neighbour and is decayed once, retaining most of its score; a redundant duplicate sits inside a tight cluster and is decayed repeatedly until it underflows the threshold. The multiplicative form makes this distinction emerge automatically from the overlap structure, without any explicit duplicate-counting logic.</span>

---

## <span style="font-size: 16px;">The Score Threshold and Termination</span>

<span style="font-size: 14px;">Hard NMS terminates when the candidate list empties. Soft-NMS instead terminates when no candidate exceeds $\texttt{score\_threshold}$, because no box is ever removed by overlap, only by its decayed score falling too low. This threshold therefore does double duty: it prunes weak original detections and it provides the stopping condition for the decay loop.</span>

<span style="font-size: 14px;">Setting it too low keeps many heavily-decayed near-zero boxes in the output, inflating false positives; setting it too high discards real detections that were lightly decayed. A common value is around $0.001$ to $0.05$, low enough to keep lightly-overlapped true objects but high enough to flush boxes that were decayed to near-zero by repeated strong overlaps. The strict greater-than comparison means a box exactly at the threshold is not emitted, which keeps the boundary behaviour deterministic on synthetic test cases.</span>

---

## <span style="font-size: 16px;">Relationship to Hard NMS</span>

<span style="font-size: 14px;">Soft-NMS generalizes hard NMS rather than replacing it. The linear variant with a hard rescore reduces exactly to greedy NMS when the decay is made a step function: multiply by $0$ above the IoU threshold and $1$ below. The Gaussian variant approaches this step as $\sigma \to 0$, since the decay factor then collapses to near-zero for any non-trivial overlap. Viewed this way, hard NMS is the limiting case of an infinitely sharp decay, and Soft-NMS simply softens that cliff into a slope.</span>

<span style="font-size: 14px;">Both algorithms share the same building blocks: an IoU computation, a score-ordered greedy selection, and a per-step update against the selected box. Only the update rule differs. This is why Soft-NMS is a true drop-in: any pipeline already running hard NMS can switch by changing a handful of lines in the inner loop.</span>

<span style="font-size: 14px;">One practical consequence of the soft rescoring is that Soft-NMS changes the output's score distribution, not just its membership. Hard NMS returns boxes with their original scores; Soft-NMS returns decayed scores for boxes that overlapped earlier picks. Downstream consumers that rank or threshold by score therefore see a smoother confidence profile, which is itself part of the mAP improvement: the average-precision metric rewards correctly ranking a true overlapping detection below the cleaner one rather than discarding it.</span>

---

## <span style="font-size: 16px;">Complexity and Cost</span>

<span style="font-size: 14px;">Soft-NMS has the same asymptotic complexity as hard NMS, $O(N^2)$ in the worst case, but with a higher constant. Hard NMS deletes boxes, so the candidate pool shrinks quickly and many comparisons are skipped. Soft-NMS only removes the single selected box per step and decays the rest, so the pool shrinks by one each iteration and nearly all pairwise IoUs are eventually computed. For $N$ candidates the loop runs $N$ times, each step decaying up to $N$ survivors, giving the quadratic bound with little of the early-exit savings hard NMS enjoys.</span>

<span style="font-size: 14px;">In practice the extra cost is modest because a score pre-filter still trims the candidate set before the loop, and the per-step work is a cheap vectorized Gaussian over the survivors. The accuracy gain almost always justifies the small overhead, which is why Soft-NMS ships as an option in most detection frameworks.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Rounding internal scores between steps.** The decayed scores must stay full-precision until a box is selected and emitted. Rounding after every decay accumulates error across the repeated multiplications and can change which candidate is picked next, diverging from the reference output.</span>
* <span style="font-size: 14px;">**Re-sorting once at the start instead of re-selecting each step.** Because decay changes scores during the run, the next pick is the current max, not the next entry in the original sort order. Sorting once and iterating in that fixed order is the most common Soft-NMS bug and produces wrong pick sequences.</span>
* <span style="font-size: 14px;">**Recomputing IoU against original boxes incorrectly, or decaying the selected box.** Decay is applied only to the remaining candidates against the just-selected box; the selected box is removed from the pool first. Decaying it or leaving it in the pool corrupts the loop.</span>
* <span style="font-size: 14px;">**Wrong termination comparison.** Stopping must use strictly greater-than against $\texttt{score\_threshold}$. Using $\geq$ emits an extra boundary box, and forgetting the threshold entirely loops until every score underflows, emitting a long tail of junk detections.</span>

---