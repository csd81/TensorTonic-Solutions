# <span style="font-size: 20px;">Apply One Full SGD Training Step</span>

<span style="font-size: 14px;">A complete training step connects prediction, loss, backpropagation, and parameter update. This problem performs all four stages manually for one tanh neuron over a batch, then measures the loss again with the updated parameters.</span>

---

## <span style="font-size: 16px;">Batch predictions</span>

<span style="font-size: 14px;">Each input row produces one scalar preactivation and prediction:</span>

$$
a_i=x_i^{\mathsf T}w+b
$$

$$
p_i=\tanh(a_i)
$$

<span style="font-size: 14px;">The same weight vector and bias are shared across all examples. The batch prediction tensor contains one value per row.</span>

---

## <span style="font-size: 16px;">The current summed loss</span>

<span style="font-size: 14px;">The old loss is the sum of squared prediction errors:</span>

$$
L=\sum_{i=1}^{B}(p_i-y_i)^2
$$

<span style="font-size: 14px;">This value is measured before the update. It describes the current parameters and must be returned even if the learning rate later leaves those parameters unchanged.</span>

---

## <span style="font-size: 16px;">From loss to preactivation gradients</span>

<span style="font-size: 14px;">For one example, squared error contributes:</span>

$$
\frac{\partial L}{\partial p_i}=2(p_i-y_i)
$$

<span style="font-size: 14px;">Tanh contributes:</span>

$$
\frac{\partial p_i}{\partial a_i}=1-p_i^2
$$

<span style="font-size: 14px;">Combining them gives the gradient with respect to each preactivation:</span>

$$
\delta_i=2(p_i-y_i)(1-p_i^2)
$$

<span style="font-size: 14px;">The vector $\delta$ contains the complete local backward signal for every batch example.</span>

---

## <span style="font-size: 16px;">Weight and bias gradients across the batch</span>

<span style="font-size: 14px;">Each weight influences every example through its aligned input coordinate. Summing those contributions gives:</span>

$$
\frac{\partial L}{\partial w}=X^{\mathsf T}\delta
$$

<span style="font-size: 14px;">The bias enters every preactivation with local derivative one, so its batch gradient is:</span>

$$
\frac{\partial L}{\partial b}=\sum_{i=1}^{B}\delta_i
$$

<span style="font-size: 14px;">These are current-step gradients derived from the supplied batch and current parameters. Any gradient values left on tensors by earlier computations must not influence them.</span>

---

## <span style="font-size: 16px;">The SGD update</span>

<span style="font-size: 14px;">Stochastic gradient descent subtracts the learning-rate-scaled gradients:</span>

$$
w'=w-\eta\frac{\partial L}{\partial w}
$$

$$
b'=b-\eta\frac{\partial L}{\partial b}
$$

<span style="font-size: 14px;">Both updates use gradients computed from the same pre-update parameter state. Updating one parameter before deriving the other would mix two different model states.</span>

---

## <span style="font-size: 16px;">A one-example descent step</span>

<span style="font-size: 14px;">Consider one input value of $1$, target $1$, weight $0$, bias $0$, and learning rate $0.1$. The initial prediction is zero, so the old loss is:</span>

$$
L=(0-1)^2=1
$$

<span style="font-size: 14px;">The tanh derivative at zero is one, giving:</span>

$$
\delta=2(0-1)(1-0^2)=-2
$$

<span style="font-size: 14px;">Both the weight and bias gradients are $-2$. Subtracting the scaled gradients moves both parameters to $0.2$.</span>

<span style="font-size: 14px;">The updated preactivation is $0.4$, so the new prediction is $\tanh(0.4)$, approximately $0.379949$. The new loss is approximately $0.384463$, which is lower than the old loss.</span>

---

## <span style="font-size: 16px;">Why the new loss needs another forward pass</span>

<span style="font-size: 14px;">The old predictions belong to $w$ and $b$. The new loss belongs to $w'$ and $b'$, so it must be computed from new preactivations and new tanh predictions.</span>

<span style="font-size: 14px;">A zero learning rate leaves both parameters unchanged, making old and new loss equal. A zero current gradient also leaves the parameters unchanged even when the learning rate is positive.</span>

---

## <span style="font-size: 16px;">Manual gradients and tensor contract</span>

<span style="font-size: 14px;">All derivatives are formed with tensor arithmetic rather than automatic differentiation state. This guarantees that stale parameter gradients cannot leak into the result and directly exercises the chain rule.</span>

<span style="font-size: 14px;">Inputs, targets, weights, and bias share one floating-point dtype and device. Returned tensors preserve both, and every supplied tensor remains unchanged.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using mean loss.** The required batch loss and gradients use a sum over examples.</span>
* <span style="font-size: 14px;">**Forgetting the tanh derivative.** Prediction error must be scaled by $1-p_i^2$.</span>
* <span style="font-size: 14px;">**Reading stale gradients.** Only current tensor arithmetic defines this step's gradients.</span>
* <span style="font-size: 14px;">**Updating parameters at different times.** Both updates must use gradients from the same original parameters.</span>
* <span style="font-size: 14px;">**Reporting old loss as new loss.** Updated parameters require a second forward evaluation.</span>
* <span style="font-size: 14px;">**Mutating supplied parameters.** The function returns updated tensors while preserving its inputs.</span>

<span style="font-size: 14px;">One full SGD step is a closed loop: evaluate the current model, measure its error, derive fresh gradients, update all parameters together, and evaluate the updated model.</span>

---