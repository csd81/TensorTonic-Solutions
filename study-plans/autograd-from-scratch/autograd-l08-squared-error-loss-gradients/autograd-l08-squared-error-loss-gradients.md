# <span style="font-size: 20px;">Calculate Squared-Error Loss Gradients</span>

<span style="font-size: 14px;">Squared-error loss measures the gap between predictions and targets. It turns each difference into a nonnegative penalty, sums those penalties, and provides a gradient that points each prediction toward its aligned target.</span>

---

## <span style="font-size: 16px;">Prediction errors</span>

<span style="font-size: 14px;">For prediction $p_i$ and target $y_i$, define the error:</span>

$$
e_i=p_i-y_i
$$

<span style="font-size: 14px;">The sign of $e_i$ records direction. A positive error means the prediction is above the target, while a negative error means it is below the target.</span>

---

## <span style="font-size: 16px;">The summed squared loss</span>

<span style="font-size: 14px;">The task squares every error and adds the results:</span>

$$
L=\sum_{i=1}^{n}(p_i-y_i)^2
$$

<span style="font-size: 14px;">Squaring removes the error sign and penalizes larger misses more strongly. The loss is never negative and equals zero exactly when every prediction matches its target.</span>

<span style="font-size: 14px;">This is a summed loss, not a mean. Doubling the number of identical errors doubles the loss rather than leaving it unchanged.</span>

---

## <span style="font-size: 16px;">How loss scale changes with the data</span>

<span style="font-size: 14px;">Every element contributes independently to the total. Adding another incorrect prediction increases the loss by that prediction's squared error and adds one aligned gradient entry.</span>

<span style="font-size: 14px;">A larger error grows quadratically in the loss but linearly in the gradient magnitude. Doubling an error multiplies its squared penalty by four while multiplying its prediction gradient by two.</span>

---

## <span style="font-size: 16px;">Derivative with respect to each prediction</span>

<span style="font-size: 14px;">Each prediction appears in one aligned squared term. Differentiating that term gives:</span>

$$
\frac{\partial L}{\partial p_i}=2(p_i-y_i)=2e_i
$$

<span style="font-size: 14px;">The gradient preserves the sign of the error. A prediction above its target receives a positive gradient, while a prediction below its target receives a negative gradient.</span>

<span style="font-size: 14px;">Gradient descent subtracts this value, so both cases move in the correct direction: high predictions move downward and low predictions move upward.</span>

---

## <span style="font-size: 16px;">A two-element example</span>

<span style="font-size: 14px;">Consider predictions $1$ and $-1$ with aligned targets $1$ and $1$. The errors are:</span>

$$
e_1=1-1=0
$$

$$
e_2=-1-1=-2
$$

<span style="font-size: 14px;">The summed loss is:</span>

$$
L=0^2+(-2)^2=4
$$

<span style="font-size: 14px;">The prediction gradients are:</span>

$$
\frac{\partial L}{\partial p_1}=0
$$

$$
\frac{\partial L}{\partial p_2}=-4
$$

<span style="font-size: 14px;">The first prediction already matches its target and receives no correction. The second lies below its target and receives a negative gradient, so a descent update would increase it.</span>

---

## <span style="font-size: 16px;">Shape and element alignment</span>

<span style="font-size: 14px;">Predictions and targets have identical shapes. Subtraction is elementwise, so every prediction is compared only with the target at the same position.</span>

<span style="font-size: 14px;">Summation reduces all squared errors to one scalar tensor. The gradient does not reduce: it retains the complete prediction shape because every prediction needs its own derivative.</span>

---

## <span style="font-size: 16px;">The empty case</span>

<span style="font-size: 14px;">Empty prediction and target tensors contain no errors. The sum over an empty set is zero, so the scalar loss is zero and the gradient tensor is empty with the same shape as the predictions.</span>

<span style="font-size: 14px;">This behavior follows directly from elementwise subtraction and summation. It does not require a fabricated prediction or special penalty.</span>

---

## <span style="font-size: 16px;">Shared dtype and device</span>

<span style="font-size: 14px;">Predictions and targets share one floating-point dtype and device. The errors, loss, and gradients preserve both.</span>

<span style="font-size: 14px;">The calculation remains in PyTorch tensors on the shared device. Neither input is modified, and the scalar loss is not converted into a Python number.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Computing mean squared error.** Dividing by the element count changes both the required loss and every gradient.</span>
* <span style="font-size: 14px;">**Dropping the factor of two.** Differentiating a square produces $2(p_i-y_i)$.</span>
* <span style="font-size: 14px;">**Reversing the error.** Using target minus prediction flips the requested gradient sign.</span>
* <span style="font-size: 14px;">**Reducing the gradient to a scalar.** The gradient tensor must match the prediction shape.</span>
* <span style="font-size: 14px;">**Rejecting empty inputs.** Empty aligned tensors produce a valid zero loss and empty gradient.</span>
* <span style="font-size: 14px;">**Changing dtype.** Both returned tensors must preserve the shared input dtype.</span>

<span style="font-size: 14px;">Squared-error loss combines a simple nonnegative measurement with a direct corrective signal: the loss records total mismatch, while each gradient records the signed local error for one prediction.</span>

---