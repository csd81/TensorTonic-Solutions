# Supervision Loss

The supervision loss is the objective that the Tiny Recursive Model (TRM) applies after every supervision step. It has two parts. The first is a token cross-entropy that pushes the decoded answer toward the label sequence, and the second is a binary cross-entropy that trains a halting head to recognize when the answer is already right. This problem covers both parts under both normalizations, with the averaging and weighting of the released code.

## Two ways to turn logits into probabilities

Write $u \in \mathbb{R}^{V}$ for the logits at one position and $c$ for its label. Softmax exponentiates every logit, so a logit of 30 contributes $e^{30}$ to the denominator. On small puzzle datasets a few logits can grow very large, and training becomes unstable.

Stablemax, which the paper's hyperparameter section adopts "for improved stability" and attributes to Prieto et al., 2025, replaces the exponential with a map that grows only linearly. Positive logits become $u + 1$ and negative logits become $1/(1-u)$, so the map is positive everywhere, continuous at zero, and increasing. The probabilities are these mapped values divided by their sum. A logit of 30 now contributes 31, so one runaway class cannot swamp the others.

The two normalizations agree on which class is largest, because both maps are increasing. They disagree on how sharp the distribution is, which is why the exercise takes the normalization as an input. The paper's pseudocode in Figure 3 writes softmax cross-entropy, while the released configuration file selects stablemax, so both appear here. The code computes stablemax in float64 and adds $10^{-30}$ inside the negative branch, and this exercise keeps the logits dtype, since that constant is far below any tolerance that matters.

## Averaging the answer loss

Let $\mathcal{V}_b$ be the positions of sequence $b$ whose label is not the ignore value $-100$. The answer loss averages the token losses inside each sequence first and then averages those sequence losses over the batch. This exercise fixes that order because the code divides each token loss by its own sequence's valid count before summing over the batch. A flat mean over every valid token in the batch would give long sequences more weight than short ones, which the code avoids.

## A halting target built from exact correctness

The halting head outputs one logit $q_b$ per sequence. Its target $a_b$ is a single boolean that is true only when the arg max at every valid position equals the label, so one wrong cell in a Sudoku makes the target false. It is computed without gradients, so it acts as a fixed label.

This is TRM's main simplification of adaptive computation. The Hierarchical Reasoning Model of Wang et al., 2025 trains two halting quantities with Q-learning. One is a halt logit and the other is a continue logit whose target comes from a second forward pass of the whole model. The TRM paper proposes to "only learn a halting probability through a Binary-Cross-Entropy loss of having reached the correct solution", which removes the continue loss and therefore the extra forward pass. Table 1 reports roughly unchanged generalization after this change.

The total loss weights the halting term by one half. Figure 3 of the paper adds the binary cross-entropy with weight one, but the released loss head returns the answer loss plus one half of the halting loss, and this exercise follows the code. In the full training loop, the Deep Supervision problem calls this objective with softmax, and the Adaptive Halting problem uses the halting logit to decide when a row stops.

## One position, both normalizations

Take one sequence with a single valid position, logits $u = (-1.0, 1.0)$ and label $c = 1$. Under stablemax, the mapped values are $1/(1+1) = 0.5$ and $1 + 1 = 2$, so the label's probability is $2/2.5 = 0.8$ and the token loss is $-\log 0.8 = 0.2231$. Under softmax, the label's probability is $1/(1 + e^{-2}) = 0.8808$ and the token loss is $0.1269$, which is smaller because the exponential sharpens the gap.

The arg max is class 1, so the halting target is true. With a halting logit of $-1.0$, the binary cross-entropy is $\log(1 + e^{1}) = 1.3133$, and half of it is $0.6566$. The stablemax total is $0.2231 + 0.6566 = 0.8798$, and the halting term supplies most of it because the head doubted a correct answer.

## Common mistakes to avoid

- **Taking a flat token mean.** Averaging every valid token in the batch at once weights long sequences more heavily than the per-sequence average does.
- **Using weight one on the halting term.** The released code halves the halting loss, and the paper's pseudocode differs on this point.
- **Using a per-token accuracy as the target.** The target is one boolean per sequence that is true only when every valid position is right.
- **Letting ignored positions leak in.** Ignored labels must be excluded from both the token loss and the exact-correctness check, whatever their logits say.
- **Applying $u+1$ to negative logits.** That value can be zero or negative, so the stablemax probabilities break, and the negative branch must use $1/(1-u)$.
- **Inverting the halting target.** Flipping the label trains the head to stop on wrong answers.
- **Returning floating halting targets.** The targets are booleans, and they carry no gradient.