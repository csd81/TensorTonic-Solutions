# <span style="font-size: 20px;">Mini-Batch Training Loop</span>

<span style="font-size: 14px;">The training loop is where theory meets practice. Every neural network, from a two-layer MLP to GPT-4, is trained by the same three-step recipe: forward pass (compute predictions), backward pass (compute gradients), parameter update (adjust weights). Understanding this loop end-to-end is what separates candidates who can build DL systems from those who only understand the math.</span>

---

## <span style="font-size: 16px;">The Three Training Modes</span>

<span style="font-size: 14px;">**Stochastic Gradient Descent (SGD)**: update after every single sample (</span>$B = 1$<span style="font-size: 14px;">). Very noisy gradients, but each update is fast. The noise can help escape local minima.</span>

<span style="font-size: 14px;">**Full Batch Gradient Descent**: update after processing the entire dataset (</span>$B = N$<span style="font-size: 14px;">). Exact gradient, stable convergence, but slow per-update and may get stuck in sharp minima.</span>

<span style="font-size: 14px;">**Mini-Batch Gradient Descent**: update after</span> $B$ <span style="font-size: 14px;">samples (typically 32-256). The standard in practice. Balances noise and stability, and enables GPU parallelism since all samples in a batch can be processed simultaneously.</span>

---

## <span style="font-size: 16px;">The SGD Update Rule</span>

<span style="font-size: 14px;">For a mini-batch</span> $\{(x_i, y_i)\}_{i=1}^{B}$<span style="font-size: 14px;">:</span>

$$
\overline{g}_W = \frac{1}{B} \sum_{i=1}^{B} \frac{\partial L_i}{\partial W}
$$

$$
W \leftarrow W - \eta \cdot \overline{g}_W
$$

<span style="font-size: 14px;">where</span> $\eta$ <span style="font-size: 14px;">is the learning rate. The gradient for each sample is computed independently (via backprop), then averaged. The average reduces variance by a factor of</span> $B$ <span style="font-size: 14px;">compared to single-sample SGD.</span>

---

## <span style="font-size: 16px;">Data Shuffling</span>

<span style="font-size: 14px;">Shuffling the dataset before each epoch ensures that mini-batches contain different sample combinations across epochs. Without shuffling:</span>

* <span style="font-size: 14px;">The optimizer sees the same sequence of gradients every epoch, creating a biased trajectory</span>
* <span style="font-size: 14px;">Correlated samples within a batch increase gradient variance in unhelpful directions</span>
* <span style="font-size: 14px;">The model may learn ordering artifacts rather than the underlying pattern</span>

<span style="font-size: 14px;">In practice, shuffling provides a form of implicit regularization and typically improves both convergence speed and final accuracy.</span>

---

## <span style="font-size: 16px;">Overfitting as a Diagnostic</span>

<span style="font-size: 14px;">A crucial debugging technique: before worrying about generalization, first verify your model can **perfectly memorize** a small dataset. If a network with sufficient capacity cannot overfit 4 XOR samples, the bug is in the training loop, not the data. This is the first thing experienced engineers check when a model fails to train.</span>

<span style="font-size: 14px;">Common causes of failure to overfit:</span>

* <span style="font-size: 14px;">Gradient sign error (weights moving away from the optimum)</span>
* <span style="font-size: 14px;">Forgotten bias update</span>
* <span style="font-size: 14px;">Wrong learning rate (too high causes divergence, too low causes imperceptible change)</span>
* <span style="font-size: 14px;">All-dead ReLU neurons from bad initialization</span>
* <span style="font-size: 14px;">Gradient averaging bug (dividing by wrong batch size)</span>

---

## <span style="font-size: 16px;">Learning Rate Selection</span>

<span style="font-size: 14px;">The learning rate</span> $\eta$ <span style="font-size: 14px;">is the most important hyperparameter. Too large: loss oscillates or diverges. Too small: training is prohibitively slow. In practice:</span>

* <span style="font-size: 14px;">Start with</span> $\eta = 0.01$ <span style="font-size: 14px;">or</span> $\eta = 0.001$ <span style="font-size: 14px;">as a baseline</span>
* <span style="font-size: 14px;">Use learning rate warmup for transformers (start very small, ramp up)</span>
* <span style="font-size: 14px;">Decay the learning rate over training (cosine schedule, step decay)</span>
* <span style="font-size: 14px;">The optimal lr scales with batch size: doubling</span> $B$ <span style="font-size: 14px;">often allows doubling</span> $\eta$ <span style="font-size: 14px;">(linear scaling rule)</span>

---

## <span style="font-size: 16px;">From SGD to Modern Optimizers</span>

<span style="font-size: 14px;">Vanilla SGD has known limitations: it oscillates in steep directions and moves slowly in flat directions. Modern optimizers address this:</span>

* <span style="font-size: 14px;">**SGD with Momentum**: adds a velocity term that accumulates past gradients, dampening oscillations</span>
* <span style="font-size: 14px;">**RMSProp**: divides the learning rate by a running average of gradient magnitudes, normalizing the step size per parameter</span>
* <span style="font-size: 14px;">**Adam**: combines momentum and RMSProp. The default choice for most DL tasks. Uses bias-corrected estimates of first and second moments of the gradient</span>
* <span style="font-size: 14px;">**AdamW**: Adam with decoupled weight decay. The standard optimizer for transformer training in 2026</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


* <span style="font-size: 14px;">**Why shuffle before each epoch instead of once?** Single shuffle creates fixed mini-batch compositions. The same samples always appear together, limiting the diversity of gradient signals. Per-epoch shuffling ensures that the model sees different combinations, which acts as a form of regularization and improves convergence</span>
* <span style="font-size: 14px;">**What happens if the last batch is smaller?** The gradient is still averaged over the actual batch size (not the target batch_size). Some implementations drop the last incomplete batch, but averaging over the smaller batch is more common and wastes no data</span>
* <span style="font-size: 14px;">**How does batch size affect training?** Larger batches produce lower-variance gradient estimates, allowing higher learning rates and faster wall-clock convergence on GPUs. However, very large batches can converge to sharp minima that generalize poorly. The "generalization gap" of large-batch training is an active research area</span>
* <span style="font-size: 14px;">**When would you use full-batch gradient descent?** When the dataset is small enough to fit in memory and you want stable, reproducible convergence (e.g., fine-tuning with a few hundred samples). Also in second-order methods (L-BFGS) that require consistent gradients across iterations</span>
* <span style="font-size: 14px;">**What is gradient accumulation?** A technique to simulate large batch sizes on limited GPU memory: accumulate gradients over multiple forward-backward passes before performing a single parameter update. Mathematically identical to using the larger batch size, but uses less memory</span>

---