# <span style="font-size: 20px;">Run Reverse-Mode Autodiff</span>

<span style="font-size: 14px;">Reverse-mode automatic differentiation computes how one scalar output changes with respect to many earlier leaves. It first evaluates the selected expression forward, then sends a gradient backward through the same graph using local derivative rules.</span>

---

## <span style="font-size: 16px;">The reachable scalar tree</span>

<span style="font-size: 14px;">Leaves provide scalar values. Operation nodes apply addition, multiplication, or tanh to their parents. Only the selected output and its ancestors participate; disconnected leaves and operations are irrelevant to both the output value and its gradients.</span>

<span style="font-size: 14px;">Within the reachable subgraph, each node is used in at most one parent position. This tree-shaped restriction means a reachable node has only one downstream route to the selected output, although the backward calculation can still use ordinary gradient accumulation.</span>

---

## <span style="font-size: 16px;">Forward values require topological order</span>

<span style="font-size: 14px;">Each operation needs its parent values before it can be evaluated. A deterministic topological order places all reachable parents before their children and uses original record order to resolve valid ties.</span>

<span style="font-size: 14px;">Scanning that order forward produces one saved value for every reachable node:</span>

$$
z=a+b
$$

$$
z=ab
$$

$$
z=\tanh(a)
$$

<span style="font-size: 14px;">These saved values are not only forward results. Multiplication needs both parent values during backward propagation, and tanh uses its saved output to compute $1-z^2$.</span>

---

## <span style="font-size: 16px;">Why the output gradient starts at one</span>

<span style="font-size: 14px;">The selected output is differentiated with respect to itself:</span>

$$
\frac{\partial y}{\partial y}=1
$$

<span style="font-size: 14px;">Reverse mode therefore seeds that node with gradient one:</span>

$$
\bar{y}=1
$$

<span style="font-size: 14px;">Every earlier gradient is measured relative to this seed. If a different upstream value were supplied, all resulting contributions would be scaled by that value.</span>

---

## <span style="font-size: 16px;">Moving backward through local rules</span>

<span style="font-size: 14px;">Traversing the topological order backward ensures that a child propagates before its parents are processed. Let $g$ be the gradient stored at the current operation.</span>

<span style="font-size: 14px;">Addition sends $g$ unchanged to both parents:</span>

$$
\left(g,\ g\right)
$$

<span style="font-size: 14px;">Multiplication scales each contribution by the opposite parent value:</span>

$$
\left(gb,\ ga\right)
$$

<span style="font-size: 14px;">Tanh uses its saved output $z$:</span>

$$
g(1-z^2)
$$

<span style="font-size: 14px;">Each contribution is added to the corresponding parent gradient. Addition is important as a general rule because shared graphs may send several contributions into one node, even though this problem limits the reachable graph to a tree.</span>

---

## <span style="font-size: 16px;">A mixed-operation example</span>

<span style="font-size: 14px;">Consider leaves $x$, $y$, and $b$ with values $2$, $-3$, and $1$. The graph first multiplies $x$ and $y$, then adds the bias leaf:</span>

$$
p=xy=2(-3)=-6
$$

$$
o=p+b=-6+1=-5
$$

<span style="font-size: 14px;">Seed the output gradient with one. Addition sends one to both $p$ and $b$, so the bias gradient is $1$. Multiplication then sends the opposite factor to each leaf:</span>

$$
\frac{\partial o}{\partial x}=y=-3
$$

$$
\frac{\partial o}{\partial y}=x=2
$$

<span style="font-size: 14px;">The selected output value is $-5$, and the reachable leaf gradients are $-3$ for $x$, $2$ for $y$, and $1$ for $b$.</span>

---

## <span style="font-size: 16px;">A nonlinear example</span>

<span style="font-size: 14px;">Suppose a leaf value and a bias are first added, then passed through tanh:</span>

$$
s=x+b
$$

$$
o=\tanh(s)
$$

<span style="font-size: 14px;">The tanh node sends $1-o^2$ to $s$. The addition node copies that same gradient to both leaves. This shows how reverse mode composes local rules: the nonlinear derivative is computed once, then the earlier addition routes it to both of its parents.</span>

---

## <span style="font-size: 16px;">Output and gradient ordering</span>

<span style="font-size: 14px;">The result contains the selected scalar output value and one gradient for every reachable leaf. Leaf gradients preserve the original leaf input order. Disconnected leaves are excluded because they have no path to the selected output.</span>

<span style="font-size: 14px;">If the selected output is itself a leaf, its value is returned and its gradient is one. No operation rule is needed because the seed already represents the derivative of that leaf with respect to itself.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Evaluating disconnected nodes.** Only ancestors of the selected output belong to the forward or backward calculation.</span>
* <span style="font-size: 14px;">**Using the wrong traversal direction.** Values move through topological order, while gradients move through its reverse.</span>
* <span style="font-size: 14px;">**Forgetting the output seed.** Without a gradient of one at the selected output, no backward signal enters the graph.</span>
* <span style="font-size: 14px;">**Overwriting parent gradients.** Contributions should be added so the rule remains correct for any node receiving more than one path.</span>
* <span style="font-size: 14px;">**Recomputing the wrong tanh derivative.** The local rule uses the saved operation output in $1-z^2$.</span>
* <span style="font-size: 14px;">**Returning disconnected leaves.** The gradient dictionary contains only reachable leaf identifiers in original leaf order.</span>

<span style="font-size: 14px;">Reverse-mode autodiff works because a large derivative problem can be decomposed into saved forward values, a valid graph order, one output seed, and a sequence of small local backward rules.</span>

---