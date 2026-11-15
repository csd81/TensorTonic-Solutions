# Deep Supervision

Deep supervision is the outer training loop of the Tiny Recursive Model (TRM, Jolicoeur-Martineau, 2025). It improves one answer over up to $N_{\text{sup}}$ supervision steps, and every step is a complete forward pass, loss and optimizer update that starts from the carry the previous step left behind. This problem covers what happens between those steps: cutting the carry from the graph, taking the gradient step, keeping an exponential moving average of the weights, and stopping early.

## Depth without a deep graph

The obvious way to give a model more reasoning depth is to stack more layers and backpropagate through all of them. Memory grows with every layer, and tiny datasets make large networks overfit. Deep supervision reaches a similar depth in a different way. The paper describes the carried features as reused "after detaching them from the computational graph so that their gradients do not propagate", which turns one very deep computation into a chain of shallow ones.

Each supervision step runs deep recursion, which applies $T(n+1)$ network evaluations, and backpropagates only through its last recursion process. The next step starts from the detached carry $(y^{(s)}, z^{(s)})$, so the model effectively reasons across $N_{\text{sup}} \, T(n+1)$ evaluations while the backward pass of any single update stays short. The carry acts like a residual connection between steps.

## Why the carry has to be cut

The deep recursion helper here returns its carry still attached to the graph. That is a deliberate convention of this exercise, so that detaching between steps is part of your work, as it is in Figure 2 of the paper and in the released carry handling. If the carry stays attached, the next loss can reach back into the previous step's graph. Gradients would then flow into a computation built with weights the optimizer has already changed, and memory would grow with every step.

Detaching only the carry, after the update, keeps the current logits trainable and hands the next step plain tensors with no gradient history.

## Stopping when the answer is right

The halting logit $\hat{q}^{(s)}$ is trained to predict whether the current answer is already fully correct. Figure 3 of the paper stops with a check of the form $\hat{q} > 0$ after the optimizer step, so the step that triggers halting still trains. The released code halts per row and refills halted rows with new data, which the Adaptive Halting problem covers. This exercise uses a batch of one, so the paper's single comparison is well defined.

Three more conventions keep the loop small and deterministic. The update is plain gradient descent instead of AdamW, because the optimizer is not the mechanism being taught. The loss helper uses softmax token loss and a halting weight of one half, following the released loss head, while Figure 3 writes the halting term with weight one and the released configuration uses stablemax. The stand-in network inside the helper omits token mixing so the loop runs cheaply, and the Tiny Mixer Network problem covers the full block.

## Raw weights versus averaged weights

On small datasets the raw weights can overfit quickly and then collapse. TRM keeps a shadow copy $\bar{\theta}$ updated after every optimizer step:

$$
\bar{\theta} \leftarrow \mu \, \bar{\theta} + (1 - \mu) \, \theta
$$

The paper uses $\mu = 0.999$ and reports that the average lifts Sudoku-Extreme accuracy from 79.9% to 87.4% in Table 1. The released code switches to the shadow weights for evaluation, while training continues on the raw weights. The shadow starts at the initial weights and closes only a fraction $1 - \mu$ of the gap each step, so it smooths noisy updates.

## A two-step average by hand

Take one scalar weight $\theta_0 = 1.0$, learning rate $\eta = 0.5$ and decay $\mu = 0.9$. Suppose the first step has gradient $0.4$ and halting logit $-0.4$, and the second has gradient $-0.2$ and halting logit $0.3$.

- **Step one.** The weight becomes $1.0 - 0.5 \times 0.4 = 0.8$, the average becomes $0.9 \times 1.0 + 0.1 \times 0.8 = 0.98$, and the negative halting logit lets the loop continue.
- **Step two.** The weight becomes $0.8 + 0.5 \times 0.2 = 0.9$, the average becomes $0.9 \times 0.98 + 0.1 \times 0.9 = 0.972$, and the positive halting logit stops the loop after this update.

The step count is 2, and the average ends at 0.972, still close to the starting weight.

## Common mistakes to avoid

- **Passing the attached carry forward.** The next step's gradients leak into a graph built with old weights, so the updates and the final weights change.
- **Updating the average before the gradient step.** The shadow then lags one extra step behind and never includes the latest update.
- **Swapping the two decay factors.** Writing $(1 - \mu)\bar{\theta} + \mu \theta$ makes the average follow the raw weights almost exactly when $\mu$ is near one.
- **Forgetting to clear gradients.** Each step's update then adds every earlier gradient again.
- **Checking the halting logit before the update.** The step that decides to stop must still train on its own loss.
- **Resetting the carry to the initial vectors every step.** Each step then starts from scratch, and the chain of improvements that gives deep supervision its depth disappears.
- **Training the given weight tensors in place.** The caller's weights change, and the starting point of the average is lost.