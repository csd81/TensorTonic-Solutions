## The Ranking Evaluation Problem

In information retrieval and recommendation, systems return ranked lists of items. We need metrics that evaluate both:
1. **What** items are retrieved (are they relevant?)
2. **Where** they appear in the ranking (are relevant items near the top?)

Mean Average Precision (MAP) addresses both by measuring how well relevant items are ranked.

---

## Precision and Recall Recap

**Precision at k (P@k):** Of the top k results, what fraction is relevant?

$$
P@k = \frac{\text{relevant items in top } k}{k}
$$

**Recall at k (R@k):** Of all relevant items, what fraction appears in top k?

$$
R@k = \frac{\text{relevant items in top } k}{\text{total relevant items}}
$$

---

## Average Precision (AP) for One Query

Average Precision computes precision at each position where a relevant item appears, then averages these values.

$$
\text{AP} = \frac{1}{R} \sum_{k=1}^{n} P@k \times rel(k)
$$

where:
- $R$ is the total number of relevant items
- $n$ is the total number of items in the ranking
- $P@k$ is precision at position $k$
- $rel(k)$ is 1 if item at position $k$ is relevant, 0 otherwise

The multiplication by $rel(k)$ means we only sum precision at positions with relevant items.

---

## Worked Example: Single Query

**Ranking:** [R, N, R, N, R, N, N, R] where R = relevant, N = not relevant

Total relevant items: $R = 4$

**Precision at each relevant position:**

Position 1 (R): $P@1 = 1/1 = 1.0$

Position 3 (R): $P@3 = 2/3 = 0.667$

Position 5 (R): $P@5 = 3/5 = 0.6$

Position 8 (R): $P@8 = 4/8 = 0.5$

**Average Precision:**

$$
\text{AP} = \frac{1}{4}(1.0 + 0.667 + 0.6 + 0.5) = \frac{2.767}{4} = 0.692
$$

---

## Why AP Rewards Early Relevant Items

Compare two rankings with the same items:

**Ranking A:** [R, R, R, R, N, N, N, N] (all relevant first)

Position 1: P@1 = 1/1 = 1.0

Position 2: P@2 = 2/2 = 1.0

Position 3: P@3 = 3/3 = 1.0

Position 4: P@4 = 4/4 = 1.0

AP = (1.0 + 1.0 + 1.0 + 1.0) / 4 = 1.0

**Ranking B:** [N, N, N, N, R, R, R, R] (all relevant last)

Position 5: P@5 = 1/5 = 0.2

Position 6: P@6 = 2/6 = 0.333

Position 7: P@7 = 3/7 = 0.429

Position 8: P@8 = 4/8 = 0.5

AP = (0.2 + 0.333 + 0.429 + 0.5) / 4 = 0.366

Ranking A has perfect AP (1.0) while Ranking B has low AP (0.366), even though both contain all relevant items.

---

## Mean Average Precision (MAP)

MAP is the mean of AP across multiple queries:

$$
\text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}_q
$$

where $Q$ is the number of queries.

**Example:**

Query 1: AP = 0.8

Query 2: AP = 0.6

Query 3: AP = 0.9

Query 4: AP = 0.5

MAP = (0.8 + 0.6 + 0.9 + 0.5) / 4 = 0.7

---

## Handling Missing Relevant Items

If some relevant items are not in the ranking at all, they contribute 0 to the AP sum but still count in the denominator $R$.

**Example:** 5 relevant items exist, but only 3 appear in the ranking

Ranking: [R, N, R, N, R] (positions 1, 3, 5 are the 3 retrieved relevant items)

P@1 = 1.0, P@3 = 0.667, P@5 = 0.6

AP = (1.0 + 0.667 + 0.6 + 0 + 0) / 5 = 0.453

The two missing relevant items contribute 0 to the numerator but increase the denominator, penalizing incomplete retrieval.

---

## AP vs. Precision vs. Recall

**Precision@k:** Only looks at top k, ignores ranking within top k

**Recall@k:** Measures coverage, ignores ranking

**AP:** Considers the entire ranking, rewards both coverage and good positioning

AP combines aspects of both precision and recall in a single metric that is sensitive to ranking quality.

---

## Interpolated Precision

Some definitions use **interpolated precision** at each recall level:

$$
P_{\text{interp}}(r) = \max_{r' \geq r} P(r')
$$

This smooths the precision-recall curve by taking the maximum precision at each recall level or higher. The 11-point interpolated AP samples at recall levels 0, 0.1, 0.2, ..., 1.0.

---

## MAP in Object Detection

In computer vision, MAP evaluates object detection models:

1. For each class, rank all detections by confidence score
2. A detection is "relevant" (true positive) if it matches a ground truth box with IoU above threshold
3. Compute AP for each class
4. MAP = mean AP across all classes

Common IoU thresholds: 0.5 (MAP@0.5), 0.75 (MAP@0.75), or averaged from 0.5 to 0.95 (MAP@[0.5:0.95]).

---

## MAP vs. NDCG

**MAP:**
- Binary relevance (relevant or not)
- Emphasizes recall (finding all relevant items)
- Traditional information retrieval metric

**NDCG:**
- Graded relevance (multiple levels)
- Position discount is logarithmic
- Common in recommendation and web search

Use MAP when relevance is binary. Use NDCG when you have graded relevance scores.

---

## Interpreting MAP Values

**MAP = 1.0:** Perfect ranking. All relevant items appear before all non-relevant items for every query.

**MAP = 0.5:** Moderate performance. Relevant items are mixed with non-relevant.

**MAP close to 0:** Poor performance. Relevant items appear late in rankings.

**Context matters:** A MAP of 0.3 might be excellent for a hard task or poor for an easy one. Always compare to baselines.

---

## Computational Notes

**Time complexity:** O(n) per query where n is the ranking length

**Memory:** Store the ranking and relevance labels

**Efficiency:** AP can be computed in a single pass through the ranking:

1. Initialize sum = 0, relevant_count = 0
2. For each position k:
   - If item is relevant: relevant_count += 1, sum += relevant_count / k
3. AP = sum / total_relevant

---

## Common Pitfalls

**Forgetting unjudged items:** Items without relevance labels should be handled consistently (ignore or treat as non-relevant).

**Truncating rankings:** MAP@k only considers the top k items. This changes the metric semantics.

**Averaging over queries with no relevant items:** Queries with zero relevant items have undefined AP. Exclude them or assign AP = 0.