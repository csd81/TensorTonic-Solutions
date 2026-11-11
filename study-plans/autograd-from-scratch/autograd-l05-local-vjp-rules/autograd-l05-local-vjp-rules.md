# <span style="font-size: 20px;">Apply Local Vector-Jacobian Rules</span>

<span style="font-size: 14px;">Reverse-mode automatic differentiation moves gradients backward one operation at a time. Each operation receives a scalar gradient from later computation and converts it into one contribution for each input. This local backward action is a vector-Jacobian product.</span>

---

## <span style="font-size: 16px;">Local derivatives and upstream influence</span>

<span style="font-size: 14px;">Suppose an operation produces scalar output $z$ from inputs $x_1$ through $x_m$. Let $g$ be the gradient arriving at $z$:</span>

$$
g = \frac{\partial L}{\partial z}
$$

<span style="font-size: 14px;">Each input receives the upstream gradient multiplied by its local derivative:</span>

$$
\frac{\partial L}{\partial x_i}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial x_i}
=
g\frac{\partial z}{\partial x_i}
$$

<span style="font-size: 14px;">The supported operations are addition, multiplication, and tanh. Their local rules are small enough to derive directly.</span>

---

## <span style="font-size: 16px;">Addition sends the gradient to both inputs</span>

<span style="font-size: 14px;">For scalar addition:</span>

$$
z=a+b
$$

<span style="font-size: 14px;">Both local derivatives equal one:</span>

$$
\frac{\partial z}{\partial a}=1
$$

$$
\frac{\partial z}{\partial b}=1
$$

<span style="font-size: 14px;">The two backward contributions are therefore:</span>

$$
\left(g,\ g\right)
$$

<span style="font-size: 14px;">An upstream gradient of $4$ gives a contribution of $4$ to each input. Addition copies the incoming influence because changing either input changes the output by the same amount.</span>

---

## <span style="font-size: 16px;">Multiplication uses the opposite input</span>

<span style="font-size: 14px;">For scalar multiplication:</span>

$$
z=ab
$$

<span style="font-size: 14px;">The local derivative with respect to one input is the other input:</span>

$$
\frac{\partial z}{\partial a}=b
$$

$$
\frac{\partial z}{\partial b}=a
$$

<span style="font-size: 14px;">After scaling by the upstream gradient, the contributions are:</span>

$$
\left(gb,\ ga\right)
$$

<span style="font-size: 14px;">If the inputs are $-2$ and $3$, and the upstream gradient is $0.5$, the first input receives $1.5$ while the second receives $-1$. The order matters: the first contribution uses the second input, and the second contribution uses the first.</span>

---

## <span style="font-size: 16px;">Tanh uses its saved output</span>

<span style="font-size: 14px;">For a tanh operation:</span>

$$
t=\tanh(a)
$$

<span style="font-size: 14px;">Its local derivative can be expressed using the saved forward output $t$:</span>

$$
\frac{dt}{da}=1-t^2
$$

<span style="font-size: 14px;">The single input contribution is:</span>

$$
g(1-t^2)
$$

<span style="font-size: 14px;">For an input whose saved tanh output is approximately $0.761594$ and an upstream gradient of $2$, the contribution is approximately $0.839949$. Using the saved output avoids evaluating tanh again and matches the information normally retained by an autograd node.</span>

---

## <span style="font-size: 16px;">Why this is called a vector-Jacobian product</span>

<span style="font-size: 14px;">A scalar operation with several inputs has a row of local partial derivatives, one for each input. Multiplying that row by the upstream scalar produces an ordered collection of input contributions.</span>

$$
g
\left(
\frac{\partial z}{\partial x_1},
\ldots,
\frac{\partial z}{\partial x_m}
\right)
$$

<span style="font-size: 14px;">Reverse mode never needs to construct a large global Jacobian for this task. It repeatedly applies small local rules and passes their results to parent nodes.</span>

---

## <span style="font-size: 16px;">How the upstream gradient changes the result</span>

<span style="font-size: 14px;">The upstream gradient scales every contribution produced by the operation. If $g$ is zero, every input receives zero because the final loss is locally insensitive to this operation's output. If $g$ is negative, it reverses the sign of each local contribution.</span>

<span style="font-size: 14px;">This scaling is what connects a local derivative to the rest of the graph. The same addition, multiplication, or tanh node can produce different backward contributions depending on how strongly later computation depends on its output.</span>

---

## <span style="font-size: 16px;">Arity and input order</span>

<span style="font-size: 14px;">Addition and multiplication each require two inputs, while tanh requires one. The number of returned contributions must match that arity exactly.</span>

<span style="font-size: 14px;">Contributions must also follow input order. For multiplication, swapping the two returned values changes which parent receives which derivative, even though the forward product itself is commutative. The saved output is essential for tanh but does not alter the addition or multiplication rules.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Returning only local derivatives.** Every local derivative must be scaled by the upstream gradient.</span>
* <span style="font-size: 14px;">**Using the same multiplier for both product inputs.** Each contribution uses the opposite input value.</span>
* <span style="font-size: 14px;">**Recomputing tanh from the wrong quantity.** The supplied output is already the saved forward result needed by $1-t^2$.</span>
* <span style="font-size: 14px;">**Changing contribution order.** Returned positions must align with the original input positions.</span>
* <span style="font-size: 14px;">**Returning the wrong number of contributions.** Binary operations return two values, while tanh returns one.</span>

<span style="font-size: 14px;">Local vector-Jacobian rules are the reusable core of reverse-mode autodiff. Each rule combines the gradient from later computation with the derivative of one saved forward operation.</span>

---