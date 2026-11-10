# <span style="font-size: 20px;">Hungarian Matching</span>

<span style="font-size: 14px;">Hungarian matching solves the **linear sum assignment problem**: given a cost matrix, find a one-to-one pairing of rows to columns that minimizes total cost. In modern object detection it is the heart of DETR (Carion et al., 2020, "End-to-End Object Detection with Transformers"), where it matches a fixed set of predictions to the ground-truth objects, enabling detection without anchors or non-maximum suppression.</span>

---

## <span style="font-size: 16px;">The Problem It Solves</span>

<span style="font-size: 14px;">Classical detectors emit thousands of overlapping candidate boxes and rely on hand-designed post-processing, anchor matching, and **non-maximum suppression (NMS)** to collapse duplicates. These steps are heuristic, non-differentiable, and require tuning. DETR reframes detection as **direct set prediction**: the network outputs a fixed-size set of $N$ predictions (e.g. 100), and the task is to compare this predicted set against the ground-truth set as sets, with no notion of order.</span>

<span style="font-size: 14px;">The difficulty is that there is no canonical ordering. To compute a loss we must first decide which prediction is responsible for which ground-truth object. A bad matching strategy (e.g. nearest-neighbor) lets multiple predictions claim the same object, reintroducing duplicates and the need for NMS. DETR instead requires a **one-to-one** assignment: each ground-truth object is matched to exactly one prediction. This bijective constraint is precisely what removes duplicates by construction, and finding the optimal one is the assignment problem.</span>

---

## <span style="font-size: 16px;">The Assignment Problem</span>

<span style="font-size: 14px;">Given a cost matrix $C \in \mathbb{R}^{N \times M}$ where $C_{ij}$ is the cost of pairing row $i$ with column $j$, find an assignment that minimizes total cost:</span>

$$
\min_{\sigma} \sum_{i} C[\,r_i,\, c_i\,]
$$

<span style="font-size: 14px;">subject to each row and each column appearing in at most one matched pair. When $N = M$ this is a permutation $\sigma$ over the indices. When the matrix is rectangular, the number of matched pairs is $\min(N, M)$, and the longer axis leaves $|N - M|$ entries unmatched. This is exactly what `scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)` computes.</span>

<span style="font-size: 14px;">A brute-force search over all assignments is $O(N!)$, hopeless for $N = 100$. The Hungarian algorithm (Kuhn, 1955, building on Konig and Egervary) solves it exactly in polynomial time, classically $O(n^3)$.</span>

---

## <span style="font-size: 16px;">How the Hungarian Algorithm Works</span>

<span style="font-size: 14px;">The algorithm rests on a key invariant: subtracting a constant from any entire row or any entire column of the cost matrix does not change which assignment is optimal, because every complete assignment uses exactly one entry from that row or column, so all total costs shift by the same constant. The algorithm exploits this to create zeros and then find an assignment using only zero-cost entries. For a square matrix the steps are:</span>

<span style="font-size: 14px;">1. **Row reduction**: subtract the minimum of each row from that row, creating at least one zero per row.</span>

<span style="font-size: 14px;">2. **Column reduction**: subtract the minimum of each column from that column, creating at least one zero per column.</span>

<span style="font-size: 14px;">3. **Cover all zeros**: find the minimum number of horizontal and vertical lines that cover every zero in the matrix.</span>

<span style="font-size: 14px;">4. **Test for optimality**: if the number of covering lines equals $n$, an assignment using only zeros exists and is optimal; extract it. Otherwise continue.</span>

<span style="font-size: 14px;">5. **Adjust**: let $\delta$ be the smallest uncovered entry. Subtract $\delta$ from all uncovered entries and add $\delta$ to entries covered twice (at line intersections). This creates new zeros while preserving the invariant, then return to step 3.</span>

<span style="font-size: 14px;">Each adjustment strictly increases the number of covering lines or the size of the matchable zero set, guaranteeing termination. Modern solvers (including scipy's) use the equivalent **Jonker-Volgenant** shortest-augmenting-path formulation, which is faster and numerically cleaner but computes the same optimum.</span>

<span style="font-size: 14px;">The duality view is worth keeping in mind: the row and column subtractions maintain a feasible solution to the dual of the assignment LP, and the algorithm terminates exactly when the dual and primal objectives meet (complementary slackness), which is the formal certificate that the found assignment is globally optimal, not merely locally good. This is why the Hungarian algorithm always returns the true minimum, unlike greedy heuristics that can get stuck.</span>

---

## <span style="font-size: 16px;">The Bipartite Graph View</span>

<span style="font-size: 14px;">The assignment problem is equivalent to **minimum-weight perfect matching in a bipartite graph**. Put the $N$ rows (predictions) on one side and the $M$ columns (ground truths) on the other, and draw an edge between every row and column with weight $C_{ij}$. A valid assignment is a matching, a set of edges with no shared endpoints, and we want the perfect matching (covering the smaller side fully) of minimum total weight.</span>

<span style="font-size: 14px;">This framing explains why the problem is tractable. General combinatorial optimization is often NP-hard, but bipartite matching has special structure (the constraint matrix is totally unimodular), so the linear-programming relaxation has integer optimal vertices. That means we can solve the continuous relaxation and automatically get a valid one-to-one integer assignment, which is the deep reason a polynomial algorithm exists at all.</span>

<span style="font-size: 14px;">The modern shortest-augmenting-path solvers make this concrete: they repeatedly find an augmenting path in the bipartite graph that improves the matching at minimum marginal cost, growing the matching one edge at a time until it is perfect, in $O(n^3)$ total.</span>

---

## <span style="font-size: 16px;">The DETR Matching Cost</span>

<span style="font-size: 14px;">In DETR the cost matrix entry $C_{ij}$ measures how well prediction $i$ matches ground-truth object $j$, combining a classification term and a box term:</span>

$$
C_{ij} = -\,\hat{p}_i(c_j) + \lambda_{\mathrm{L1}} \| b_i - b_j \|_1 + \lambda_{\mathrm{giou}}\, \big(1 - \mathrm{GIoU}(b_i, b_j)\big)
$$

<span style="font-size: 14px;">where $\hat{p}_i(c_j)$ is prediction $i$'s probability for ground-truth class $c_j$ (entered with a minus sign so high probability means low cost), $\| b_i - b_j \|_1$ is the $\ell_1$ box-coordinate distance, and the GIoU term penalizes poor box overlap. The $\lambda$ coefficients balance the terms. Crucially, the matching cost is computed but **not differentiated through**: scipy returns a fixed assignment, and gradients flow only through the subsequent loss evaluated on the matched pairs. The Hungarian step is treated as a non-differentiable oracle that decides responsibility.</span>

<span style="font-size: 14px;">Predictions left unmatched (since $N$ predictions typically exceeds the number of objects) are assigned the special "no object" class. After matching, the actual training loss (the Hungarian loss) is the negative log-likelihood of the matched class plus the box loss on matched pairs.</span>

<span style="font-size: 14px;">A subtle but important detail from the paper: the cost used inside the matcher and the loss used for backpropagation are deliberately **different functions**. The matcher uses the raw class **probability** $\hat{p}_i(c_j)$ (not its log) so that the classification term is commensurable with the bounded box terms, keeping the assignment well-conditioned. The training loss, by contrast, uses the standard negative log-likelihood, which has the right gradient behavior for optimization. Conflating the two (using log-probabilities in the matcher) makes the cost magnitudes blow up and destabilizes matching.</span>

---

## <span style="font-size: 16px;">Why One-to-One Matching Removes NMS</span>

<span style="font-size: 14px;">This is the conceptual payoff. Because the assignment is a bijection, each ground-truth object is the target of exactly one prediction during training. The loss therefore **rewards a single prediction** for each object and pushes all other predictions toward "no object". The network learns to produce, at most, one high-confidence box per object on its own. There are no near-duplicate high-confidence boxes to suppress, so NMS becomes unnecessary, and detection becomes a clean, end-to-end differentiable pipeline (apart from the matching oracle itself).</span>

<span style="font-size: 14px;">Contrast this with anchor-based detectors, where many anchors can be assigned to the same object (a one-to-many scheme), guaranteeing duplicate predictions that must be filtered afterward. The set-based, one-to-one formulation is what makes DETR genuinely end-to-end.</span>

---

## <span style="font-size: 16px;">Complexity and Practical Notes</span>

<span style="font-size: 14px;">The classical Hungarian algorithm runs in $O(n^3)$ for an $n \times n$ matrix, an enormous improvement over the $O(n!)$ brute force. For DETR with $N = 100$ predictions and a handful of ground-truth objects per image, this is negligible compared to the forward and backward passes through the transformer, so the matching is effectively free.</span>

<span style="font-size: 14px;">Two practical points matter in the detection setting:</span>

* <span style="font-size: 14px;">**Per-image matching.** The assignment is solved independently for each image in the batch, since each image has its own predictions and ground truths. scipy's solver is called once per image on the CPU.</span>
* <span style="font-size: 14px;">**Numerical robustness.** Costs combine probabilities, $\ell_1$ distances, and GIoU values on different scales, weighted by $\lambda$ coefficients. The solver only needs relative ordering of total costs, so absolute scale does not matter, but the cost matrix must be finite (no NaN or inf), which can otherwise arise from degenerate boxes feeding the GIoU term.</span>

<span style="font-size: 14px;">Although DETR popularized Hungarian matching in vision, the same linear sum assignment underpins multi-object tracking (matching detections to tracks across frames), entity resolution, and any task needing an optimal one-to-one correspondence between two sets.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a $3 \times 3$ cost matrix (3 predictions, 3 ground truths):</span>

$$
C = \begin{pmatrix} 4 & 1 & 3 \\ 2 & 0 & 5 \\ 3 & 2 & 2 \end{pmatrix}
$$

<span style="font-size: 14px;">**Row reduction** (subtract row minima 1, 0, 2):</span>

$$
\begin{pmatrix} 3 & 0 & 2 \\ 2 & 0 & 5 \\ 1 & 0 & 0 \end{pmatrix}
$$

<span style="font-size: 14px;">**Column reduction** (subtract column minima 1, 0, 0):</span>

$$
\begin{pmatrix} 2 & 0 & 2 \\ 1 & 0 & 5 \\ 0 & 0 & 0 \end{pmatrix}
$$

<span style="font-size: 14px;">Now seek an assignment of zeros, one per row and column. Row 0 has a zero only at column 1; row 1 also only at column 1, a conflict, so they cannot both take column 1. Row 2 has zeros at columns 0, 1, 2. Assign row 0 to column 1 (its only zero), row 2 to column 0, and row 1 needs a zero in a remaining column but only has column 1, which is taken. So the current zeros do not yield a full assignment; one more adjustment step would be needed in the full algorithm.</span>

The minimum uses row-column pairs (0, 1), (1, 0), and (2, 2). Their original costs are 1, 2, and 2, so the total is 5.0.

---

## <span style="font-size: 16px;">Rectangular Matrices and Conventions</span>

Rectangular matrices require exactly as many pairs as the smaller dimension. The remaining entries on the longer axis stay unmatched. Return the pairs in increasing row order.

- **Objective.** This problem minimizes the sum of the selected entries from the original matrix.

<span style="font-size: 14px;">The total cost is the sum of $C$ at the matched $(\text{row}, \text{col})$ pairs, computed on the original (un-reduced) matrix.</span>

Several assignments can share the same minimum cost. Any such assignment is accepted; matching a library solver's particular choice is not required. Implement the matching logic without calling a prebuilt assignment solver.

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Reporting the reduced cost instead of the original.** Row and column reductions change the matrix entries, but the reported `total_cost` must sum the **original** $C$ values at the matched positions, not the reduced ones. Summing reduced entries gives a smaller, wrong total.</span>
* <span style="font-size: 14px;">**Greedy nearest-neighbor matching.** Assigning each row to its individually cheapest column is not optimal: a locally cheap choice can force expensive choices elsewhere. Only a global solver (Hungarian / Jonker-Volgenant) minimizes the total, and greedy reintroduces the duplicate-assignment problem DETR aims to remove.</span>
* <span style="font-size: 14px;">**Ignoring the rectangular case.** With $N \ne M$ only $\min(N, M)$ pairs exist; assuming a full permutation crashes or mismatches indices. The longer axis legitimately leaves entries unmatched (the unmatched DETR predictions become "no object").</span>
* <span style="font-size: 14px;">**Backpropagating through the matching.** The assignment is a discrete, non-differentiable operation and is treated as a fixed oracle. Attempting to differentiate the argmin of the matching, rather than only the loss on the chosen pairs, breaks the training graph.</span>

---