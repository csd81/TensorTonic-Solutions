# <span style="font-size: 20px;">Accumulate Shared-Path Gradients</span>

<span style="font-size: 14px;">A computation graph is not always a tree. One value may be used by several later operations, or it may occupy both parent positions of the same operation. When several paths connect a node to the output, its gradient is the sum of the contributions carried by all those paths.</span>

---

## <span style="font-size: 16px;">Why one node can receive several contributions</span>

<span style="font-size: 14px;">A derivative measures every way a small change can influence the selected output. If node $a$ reaches output $y$ through several directed paths, each path contributes its own chain-rule product:</span>

$$
\frac{\partial y}{\partial a}
=
\sum_{p\in P(a,y)} c_p
$$

<span style="font-size: 14px;">Here $P(a,y)$ is the set of paths from $a$ to $y$, and $c_p$ is the derivative contribution along one path. Replacing an earlier gradient would discard paths that were already processed.</span>

---

## <span style="font-size: 16px;">A repeated input in addition</span>

<span style="font-size: 14px;">Consider an addition that uses the same leaf twice:</span>

$$
y=a+a
$$

<span style="font-size: 14px;">Each parent position has local derivative one. With output gradient one, the first position contributes one and the second position contributes another one:</span>

$$
\frac{\partial y}{\partial a}=1+1=2
$$

<span style="font-size: 14px;">The parent identifier is repeated, but both operand positions remain meaningful. Deduplicating the parent list or assigning the gradient twice would produce the wrong result.</span>

---

## <span style="font-size: 16px;">A repeated input in multiplication</span>

<span style="font-size: 14px;">The same idea appears in a square represented as multiplication:</span>

$$
y=aa=a^2
$$

<span style="font-size: 14px;">The first parent position contributes the value from the second position, and the second contributes the value from the first:</span>

$$
\frac{\partial y}{\partial a}=a+a=2a
$$

<span style="font-size: 14px;">At $a=3$, the two contributions are $3$ and $3$, so the accumulated gradient is $6$. Keeping only the final contribution would incorrectly return $3$.</span>

---

## <span style="font-size: 16px;">A diamond-shaped graph</span>

<span style="font-size: 14px;">Shared paths also arise when one leaf feeds separate branches that later meet. Suppose:</span>

$$
u=xc
$$

$$
v=x+c
$$

$$
y=u+v
$$

<span style="font-size: 14px;">The leaf $x$ reaches $y$ through multiplication and addition. Its two contributions are:</span>

$$
\frac{\partial u}{\partial x}=c
$$

$$
\frac{\partial v}{\partial x}=1
$$

<span style="font-size: 14px;">Since the final addition sends gradient one into both branches:</span>

$$
\frac{\partial y}{\partial x}=c+1
$$

<span style="font-size: 14px;">At $c=3$, the accumulated gradient for $x$ is $4$. The gradient for $c$ also receives two paths: $x$ through the product and one through the sum, giving $x+1$.</span>

---

## <span style="font-size: 16px;">Why reverse topological order is essential</span>

<span style="font-size: 14px;">A node should propagate to its parents only after every reachable child has contributed to that node's gradient. Reverse topological order provides this guarantee because all children appear before their parents during the backward traversal.</span>

<span style="font-size: 14px;">Each node gradient begins at zero, except for the selected output, which is seeded with one. Every local contribution is added to the parent's current gradient:</span>

$$
\bar{p}\leftarrow\bar{p}+c
$$

<span style="font-size: 14px;">The addition in this update is the central difference between correct accumulation and accidental overwriting.</span>

---

## <span style="font-size: 16px;">Saved values and local rules</span>

<span style="font-size: 14px;">Shared-path accumulation does not change the local derivatives. Addition still copies the upstream gradient, multiplication still uses the opposite parent value, and tanh still uses $1-y^2$ from its saved output.</span>

<span style="font-size: 14px;">What changes is how parent gradients are updated. The same local contribution may arrive several times from repeated parent positions or from different children, and every valid arrival must remain in the total.</span>

---

## <span style="font-size: 16px;">Reachable leaves and output format</span>

<span style="font-size: 14px;">Only ancestors of the selected output participate. A disconnected leaf has no path contribution and is excluded rather than returned with an artificial zero.</span>

<span style="font-size: 14px;">The final dictionary contains one accumulated gradient for each reachable leaf, preserving original leaf input order. Intermediate node gradients are necessary during backpropagation but are not part of the requested result.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Replacing instead of adding.** Assignment discards gradient contributions that arrived through earlier paths.</span>
* <span style="font-size: 14px;">**Deduplicating repeated parents.** Two operand positions represent two derivative contributions, even when their identifiers match.</span>
* <span style="font-size: 14px;">**Propagating a parent too early.** A node must collect contributions from all reachable children before sending its total backward.</span>
* <span style="font-size: 14px;">**Changing local derivative rules.** Sharing affects accumulation, not the derivative of addition, multiplication, or tanh.</span>
* <span style="font-size: 14px;">**Returning disconnected leaves.** Only leaves with a path to the selected output belong in the result.</span>

<span style="font-size: 14px;">Gradient accumulation is the feature that lets reverse-mode autodiff handle real directed acyclic graphs. One stored gradient represents the sum of every downstream route through which a node influences the output.</span>

---