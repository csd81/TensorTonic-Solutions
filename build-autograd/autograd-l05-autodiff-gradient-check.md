# <span style="font-size: 20px;">Gradient-Check an Autodiff Graph</span>

<span style="font-size: 14px;">A graph-wide gradient check tests whether reverse-mode autodiff produces the correct derivative for every reachable leaf. It compares the gradients obtained by backpropagation with independent numerical estimates formed by perturbing one leaf at a time.</span>

---

## <span style="font-size: 16px;">Two independent views of the same derivative</span>

<span style="font-size: 14px;">The analytic calculation follows the graph's local derivative rules. Addition copies an incoming gradient, multiplication scales it by the opposite parent value, and tanh contributes the factor $1-y^2$ using its saved output.</span>

<span style="font-size: 14px;">The numerical calculation does not use those backward rules. It observes how the selected scalar output changes after one leaf moves by a small positive step. Agreement matters because the two methods can fail for different reasons.</span>

---

## <span style="font-size: 16px;">Analytic gradients from reverse mode</span>

<span style="font-size: 14px;">Only the selected output and its ancestors participate. Their values are evaluated in topological order so every operation sees its parent values first.</span>

<span style="font-size: 14px;">The output gradient is seeded with one:</span>

$$
\frac{\partial f}{\partial f}=1
$$

<span style="font-size: 14px;">Traversing the reachable order backward applies each operation's local vector-Jacobian rule. At the end of this reverse traversal, each reachable leaf holds the analytic derivative of the selected output with respect to that leaf.</span>

---

## <span style="font-size: 16px;">Numerical gradients from forward differences</span>

<span style="font-size: 14px;">For one reachable leaf $x$, keep every other leaf fixed, increase $x$ by $h$, and evaluate the selected output again:</span>

$$
g_{\mathrm{num}}(x)
=
\frac{f(x+h)-f(x)}{h}
$$

<span style="font-size: 14px;">The baseline output $f(x)$ is shared, but every perturbed evaluation begins from the original leaf values. After checking one leaf, its perturbation must not remain active while checking another.</span>

<span style="font-size: 14px;">Disconnected leaves are excluded. Changing one cannot affect the selected output, and the required dictionaries contain only leaves in the output's reachable tree.</span>

---

## <span style="font-size: 16px;">A product followed by tanh</span>

<span style="font-size: 14px;">Consider two reachable leaves with values $0.7$ and $-0.4$. Their product becomes the input to tanh:</span>

$$
p=xy=(0.7)(-0.4)=-0.28
$$

$$
f=\tanh(p)\approx -0.272905
$$

<span style="font-size: 14px;">The local tanh derivative is:</span>

$$
1-f^2 \approx 0.925523
$$

<span style="font-size: 14px;">Multiplication then routes this value through the opposite leaf:</span>

$$
\frac{\partial f}{\partial x}
=
(1-f^2)y
\approx
-0.370209
$$

$$
\frac{\partial f}{\partial y}
=
(1-f^2)x
\approx
0.647866
$$

<span style="font-size: 14px;">Forward differences with a step of $0.0001$ produce nearby estimates. The small gap reflects the finite step and tanh curvature, not necessarily an error in reverse mode.</span>

---

## <span style="font-size: 16px;">Matching leaves and reporting error</span>

<span style="font-size: 14px;">The analytic and numerical dictionaries use the same reachable leaf identifiers and preserve original leaf input order. Matching by identifier prevents a correct value from being compared with the wrong leaf.</span>

<span style="font-size: 14px;">The largest absolute disagreement summarizes the worst individual check:</span>

$$
\text{max error}
=
\max_{x\ \mathrm{reachable}}
\left|
g_{\mathrm{analytic}}(x)-g_{\mathrm{num}}(x)
\right|
$$

<span style="font-size: 14px;">The full dictionaries remain important because the maximum alone does not reveal which leaf produced the discrepancy or whether several leaves disagree in different ways.</span>

---

## <span style="font-size: 16px;">Step size and floating-point behavior</span>

<span style="font-size: 14px;">A forward difference is an approximation. If $h$ is too large, it measures change across an interval where the graph may curve. If $h$ is extremely small, subtracting nearly equal outputs can lose useful precision. The task uses 64-bit arithmetic to reduce this numerical tension.</span>

<span style="font-size: 14px;">Simple addition and multiplication graphs may agree almost exactly because they are linear in one leaf when the others are fixed. A tanh path generally leaves a larger, though still small, forward-difference error because its slope changes across the perturbation interval.</span>

---

## <span style="font-size: 16px;">Why the tree restriction matters</span>

<span style="font-size: 14px;">Within the reachable subgraph, a node appears in at most one operation parent position. This tree-shaped contract avoids shared-path accumulation in this problem. The gradient check still validates a complete reverse traversal across multiple operation types.</span>

<span style="font-size: 14px;">The selected output may also be a leaf. In that case, its analytic gradient is one, and perturbing the leaf changes the output by exactly the perturbation step, so the numerical gradient is also one.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Perturbing several leaves together.** The measured slope combines directions and cannot verify one partial derivative.</span>
* <span style="font-size: 14px;">**Carrying a perturbation forward.** Every leaf check must start from the same original graph.</span>
* <span style="font-size: 14px;">**Including disconnected leaves.** Both dictionaries contain only leaves that can influence the selected output.</span>
* <span style="font-size: 14px;">**Expecting exact equality.** Nonlinear graphs normally produce a small finite-difference disagreement.</span>
* <span style="font-size: 14px;">**Comparing by position without identifiers.** Analytic and numerical values must be matched to the same reachable leaf.</span>

<span style="font-size: 14px;">A graph gradient check is valuable because it tests the whole derivative path at once: reachability, forward values, reverse ordering, local rules, leaf selection, and numerical agreement.</span>

---