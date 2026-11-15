# Deep Recursion

Deep recursion is the forward pass that the Tiny Recursive Model (TRM) of Jolicoeur-Martineau, 2025 runs once per supervision step. It repeats the latent recursion process $T$ times on the carried answer $y$ and latent $z$, keeps gradients for the last repetition only, and reads an answer and a halting decision from the refined answer. This problem covers that gradient schedule, the detached carry, and the two heads.

## HRM needed a fixed point to stay cheap

The Hierarchical Reasoning Model (HRM) of Wang et al., 2025 runs a low-level network $f_L$ and a high-level network $f_H$ in a nested loop. With its settings $n = 2$ and $T = 2$, one forward pass makes six network evaluations. HRM backpropagates through only the last two, one $f_L$ and one $f_H$. It justifies this with the implicit function theorem and the 1-step gradient approximation, which take the gradient in a single step once a recurrence has converged to a fixed point.

TRM argues that the assumption does not hold. HRM never iterates to a fixed point, it simply runs four evaluations and treats the latent as converged, and its own residual plots show the latent is still far from stable. The approximation discards most of the gradient for a reason that is not met.

## Backpropagate the whole last process instead

TRM drops the fixed-point story. It defines a full recursion process as $n$ evaluations of the latent update followed by one answer update, and backpropagates through that entire process. Deep supervision trains the network to improve any carry it is given, so extra processes run without gradients should still move $(y, z)$ toward the solution. TRM therefore runs $T - 1$ processes under no-gradient mode to refine the carry cheaply, then one process with full backpropagation. In Table 1 of the paper, replacing the 1-step gradient with this schedule raises Sudoku-Extreme accuracy from 56.5% to 87.4%.

Three schedules are worth keeping apart.

- **Full backpropagation through time.** Every one of the $T(n + 1)$ evaluations is differentiated, so activation memory grows with the whole recursion depth.
- **HRM's 1-step gradient.** Only the final latent update and the final answer update carry gradients, and the rest of the last process is treated as a constant.
- **TRM's schedule.** All $n + 1$ evaluations of the last process carry gradients, and the $T - 1$ earlier processes carry none, so memory scales with one process while the forward pass still sees all of them.

## Effective depth and the carry

Each evaluation passes through $n_{\text{layers}}$ layers, so one supervision step has an effective depth of $T(n + 1) n_{\text{layers}}$. The paper's setting $T = 3$, $n = 6$ and two layers gives $3 \times 7 \times 2 = 42$ layers per step, while gradients flow through only the last 14. Deep supervision then chains up to 16 such steps.

The returned carry is detached, as in Figure 3 and the released code, so the next supervision step starts a fresh graph. The logits and halting logit stay attached because the current step's loss is built from them. Following the released code, the answer head drops the first $P$ prefix positions, which hold puzzle embeddings in the real model, and the halting head reads position zero of the refined answer. The released halting head has two outputs, and TRM mode uses only the first. The network receives the element-wise sum of its inputs, as in Section 4.3 and the code. Section 4.3 labels the frequent module $f_H$, which is a typo for $f_L$. The supplied stand-in network omits token mixing so the schedule stays cheap to run, and the Tiny Mixer Network problem covers the full block.

## Counting evaluations for three short cycles

Take $T = 3$ and $n = 2$. Each recursion process makes $n + 1 = 3$ evaluations, so the step makes 9. Evaluations 1 to 6 belong to the first two processes and run without gradients, and evaluations 7 to 9 run with gradients. The effective depth with two layers is $3 \times 3 \times 2 = 18$, and gradients pass through 6 of those layers.

The first process starts from the incoming carry without gradients, so nothing connects the logits to that carry and its gradient is zero. The question embedding and every weight receive gradients through evaluations 7 to 9. With $T = 1$ the same step has no gradient-free process, so the single process starts from the incoming carry and the carry receives a gradient as well.

## Common mistakes to avoid

- **Backpropagating through every cycle.** Removing the no-gradient context gives full backpropagation through time, which stores all $T(n + 1)$ evaluations.
- **Keeping only the last two evaluations differentiable.** That is HRM's 1-step gradient, which the paper replaces because no fixed point is reached.
- **Running every cycle without gradients.** The logits then have no gradient function, and the step cannot train anything.
- **Returning an attached carry.** The next supervision step would backpropagate into this step's graph, which deep supervision is designed to avoid.
- **Detaching the logits along with the carry.** This step's loss needs them attached.
- **Reading the heads from the wrong place.** Answer logits skip the prefix positions, the halting logit uses position zero of the answer, and neither head reads the latent or the carry from before the last process.