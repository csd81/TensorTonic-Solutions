# <span style="font-size: 20px;">Gradient-Check a Scalar Neuron</span>

<span style="font-size: 14px;">A gradient check compares derivatives obtained from calculus with derivatives estimated from small numerical perturbations. Agreement between the two provides strong evidence that the analytic formulas and their implementation are correct.</span>

---

## <span style="font-size: 16px;">The neuron being checked</span>

<span style="font-size: 14px;">The scalar neuron computes a weighted sum, adds a bias, and applies tanh:</span>

$$
a = \sum_{i=1}^{n} w_i x_i + b
$$

$$
y = \tanh(a)
$$

<span style="font-size: 14px;">The inputs remain fixed during this problem. The parameters being checked are every weight and the scalar bias.</span>

---

## <span style="font-size: 16px;">Analytic parameter gradients</span>

<span style="font-size: 14px;">The tanh derivative supplies the local factor:</span>

$$
\frac{dy}{da}=1-y^2
$$

<span style="font-size: 14px;">Since weight $w_i$ contributes $w_ix_i$ to the preactivation, its derivative is:</span>

$$
\frac{\partial y}{\partial w_i}=(1-y^2)x_i
$$

<span style="font-size: 14px;">The bias enters the preactivation directly, so its derivative is:</span>

$$
\frac{\partial y}{\partial b}=1-y^2
$$

<span style="font-size: 14px;">These are the analytic reference values. The numerical calculation should recover nearly the same sensitivities without using these formulas.</span>

---

## <span style="font-size: 16px;">One parameter at a time</span>

<span style="font-size: 14px;">A forward difference measures how the output changes after increasing one parameter by a small positive step $h$. For weight $w_i$:</span>

$$
g_i^{\mathrm{num}}
=
\frac{y(w_i+h)-y(w_i)}{h}
$$

<span style="font-size: 14px;">Every other weight and the bias must remain at their original values. The bias estimate follows the same idea:</span>

$$
g_b^{\mathrm{num}}
=
\frac{y(b+h)-y(b)}{h}
$$

<span style="font-size: 14px;">Each perturbation is an independent experiment that starts from the same baseline neuron output. Carrying one changed parameter into the next experiment would mix several partial derivatives.</span>

---

## <span style="font-size: 16px;">A numerical example</span>

<span style="font-size: 14px;">Consider two inputs with values $1$ and $-2$. Their aligned weights are $0.5$ and $-0.25$, and the bias is $0.1$.</span>

$$
a = (0.5)(1)+(-0.25)(-2)+0.1=1.1
$$

$$
y=\tanh(1.1)\approx 0.800499
$$

<span style="font-size: 14px;">The local tanh factor is approximately:</span>

$$
1-y^2 \approx 0.359201
$$

<span style="font-size: 14px;">The analytic weight gradients are therefore:</span>

$$
\frac{\partial y}{\partial w_1}\approx 0.359201
$$

$$
\frac{\partial y}{\partial w_2}\approx -0.718403
$$

<span style="font-size: 14px;">The analytic bias gradient is approximately $0.359201$. Forward differences with a step of one millionth produce nearby values, with a small disagreement caused by finite-step approximation and floating-point arithmetic.</span>

---

## <span style="font-size: 16px;">Interpreting the maximum error</span>

<span style="font-size: 14px;">The check compares every analytic weight gradient with its numerical counterpart and also compares the two bias gradients. The largest absolute difference summarizes the worst disagreement:</span>

$$
\text{max error}
=
\max\!\left(
\max_i |g_i^{\mathrm{analytic}}-g_i^{\mathrm{num}}|,
|g_b^{\mathrm{analytic}}-g_b^{\mathrm{num}}|
\right)
$$

<span style="font-size: 14px;">A small value supports the analytic implementation. It should not be expected to equal zero because forward differences are approximations and subtraction can expose floating-point rounding.</span>

---

## <span style="font-size: 16px;">Step size, precision, and saturation</span>

<span style="font-size: 14px;">A large step measures slope across an interval that may be too wide to represent the local derivative well. An extremely small step can make two floating-point outputs nearly indistinguishable. Performing the check in 64-bit floating point reduces this numerical tension.</span>

<span style="font-size: 14px;">A saturated neuron has an output very close to $1$ or $-1$, so its analytic gradients can be tiny. The numerical output may not change at all after a small perturbation, producing a numerical gradient of zero. That difference can still be acceptably small in absolute terms.</span>

<span style="font-size: 14px;">With no weights, both weight-gradient vectors are empty and only the bias is checked. The maximum error then comes from the bias comparison alone.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Perturbing several parameters together.** The resulting slope combines sensitivities and cannot verify one analytic partial.</span>
* <span style="font-size: 14px;">**Reusing a perturbed parameter vector.** Every numerical estimate must start from the original weights.</span>
* <span style="font-size: 14px;">**Checking in low precision.** Small output differences can disappear in 16-bit or 32-bit arithmetic.</span>
* <span style="font-size: 14px;">**Expecting exact equality.** A correct gradient check normally leaves a small finite-difference error.</span>
* <span style="font-size: 14px;">**Ignoring the empty case.** With no weights, the bias error is still meaningful and must determine the maximum.</span>

<span style="font-size: 14px;">The strength of a gradient check comes from independence: calculus provides one answer, controlled perturbations provide another, and their agreement tests the derivative logic from two different directions.</span>

---