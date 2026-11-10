## <span style="font-size: 16px;">The Overfitting Problem</span>

<span style="font-size: 14px;">During training, the training loss $\mathcal{L}_{\text{train}}$ keeps decreasing, but validation loss $\mathcal{L}_{\text{val}}$ eventually starts increasing. The point where $\mathcal{L}_{\text{val}}$ reaches its minimum is the sweet spot: the model has learned the underlying pattern without memorizing training noise.</span>

---

## <span style="font-size: 16px;">Patience-Based Early Stopping</span>

<span style="font-size: 14px;">The algorithm tracks the best validation metric seen so far and counts how many epochs pass without improvement:</span>

<span style="font-size: 14px;">1. Initialize $\text{best\_loss} = \infty$, $\text{patience\_counter} = 0$</span>
<span style="font-size: 14px;">2. After each epoch, compute $\mathcal{L}_{\text{val}}$</span>
<span style="font-size: 14px;">3. If $\mathcal{L}_{\text{val}} < \text{best\_loss} - \delta$ (where $\delta$ is the minimum improvement threshold):</span>
<span style="font-size: 14px;">   - Update $\text{best\_loss} = \mathcal{L}_{\text{val}}$</span>
<span style="font-size: 14px;">   - Reset $\text{patience\_counter} = 0$</span>
<span style="font-size: 14px;">   - Save the current model weights</span>
<span style="font-size: 14px;">4. Otherwise: increment $\text{patience\_counter}$</span>
<span style="font-size: 14px;">5. If $\text{patience\_counter} \geq P$ (patience limit), stop training</span>

---

## <span style="font-size: 16px;">Why It Works</span>

<span style="font-size: 14px;">Early stopping limits the effective complexity of the model. With gradient descent, longer training lets the model move further from initialization, increasing its expressiveness. Stopping early keeps parameters close to their initial values, which acts as an implicit regularizer.</span>

---

## <span style="font-size: 16px;">Relationship to $L_2$ Regularization</span>

<span style="font-size: 14px;">For linear models and quadratic losses, early stopping with learning rate $\eta$ and $T$ training steps is approximately equivalent to $L_2$ regularization with $\lambda \approx \frac{1}{\eta T}$. As $T \to \infty$, the implicit regularization vanishes ($\lambda \to 0$), recovering the unregularized solution.</span>

---

## <span style="font-size: 16px;">Practical Notes</span>

<span style="font-size: 14px;">Typical patience values range from $5$ to $20$ epochs depending on the learning rate and dataset size. A minimum improvement threshold $\delta > 0$ prevents reacting to noise. Always restore the best model weights after stopping, not the final ones.</span>