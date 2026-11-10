# <span style="font-size: 20px;">Forward-Mode Autodiff with Dual Numbers</span>

## The Problem of Computing Derivatives

Computing derivatives is the central computational task in machine learning. Every training step of every neural network requires computing the gradient of the loss with respect to millions of parameters. There are four ways to compute derivatives, each with different tradeoffs:

**Symbolic differentiation** applies calculus rules to produce a formula for the derivative. It gives exact results but can produce exponentially large expressions ("expression swell") and requires the function to be given as a formula.

**Numerical differentiation** (finite differences) approximates the derivative using function evaluations: $f'(x) \approx (f(x+h) - f(x))/h$. It is simple but suffers from floating-point errors: too large $h$ gives truncation error, too small $h$ gives cancellation error.

**Reverse-mode automatic differentiation** (backpropagation) traces the computation forward, then propagates derivatives backward through the computation graph. It computes the gradient of one scalar output with respect to all inputs in one backward pass. This is what PyTorch and TensorFlow use.

**Forward-mode automatic differentiation** propagates derivatives forward alongside the computation, using dual numbers. It computes the derivative of all outputs with respect to one input in one forward pass. This is what JAX's `jvp` uses internally.

This problem focuses on forward-mode AD using dual numbers.

## Dual Numbers

### Definition

A dual number is a pair $(a, a')$ written as $a + a'\epsilon$, where $\epsilon$ is an abstract quantity satisfying $\epsilon^2 = 0$ but $\epsilon \neq 0$. The first component $a$ is the **value** (or primal), and the second component $a'$ is the **derivative** (or tangent).

This is analogous to complex numbers $a + bi$ where $i^2 = -1$, except here $\epsilon^2 = 0$. This single property $\epsilon^2 = 0$ is what makes dual numbers compute derivatives automatically.

### Why Dual Numbers Compute Derivatives

Consider a smooth function $f$ and a dual number input $x + \epsilon$:

$$f(x + \epsilon) = f(x) + f'(x)\epsilon + \frac{f''(x)}{2}\epsilon^2 + \cdots = f(x) + f'(x)\epsilon$$

The Taylor series truncates at the first-order term because $\epsilon^2 = 0$. The result is a dual number whose primal part is $f(x)$ (the function value) and whose tangent part is $f'(x)$ (the derivative). By setting the input to $x + 1\cdot\epsilon$ (tangent = 1), we get both $f(x)$ and $f'(x)$ in a single evaluation.

This is not an approximation: it gives the exact derivative, using only the four arithmetic operations and elementary function evaluations.

### Arithmetic Rules

From the definition $\epsilon^2 = 0$, the arithmetic of dual numbers follows:

**Addition:**

$$(a + a'\epsilon) + (b + b'\epsilon) = (a + b) + (a' + b')\epsilon$$

The values add and the derivatives add. This corresponds to the sum rule of calculus: $(f + g)' = f' + g'$.

**Multiplication:**

$$(a + a'\epsilon)(b + b'\epsilon) = ab + (ab' + a'b)\epsilon + a'b'\epsilon^2 = ab + (ab' + a'b)\epsilon$$

The $\epsilon^2$ term vanishes. The derivative component $ab' + a'b$ is exactly the product rule: $(fg)' = fg' + f'g$.

**Subtraction:**

$$(a + a'\epsilon) - (b + b'\epsilon) = (a - b) + (a' - b')\epsilon$$

**Division:**

$$\frac{a + a'\epsilon}{b + b'\epsilon} = \frac{a}{b} + \frac{a'b - ab'}{b^2}\epsilon$$

This is the quotient rule: $(f/g)' = (f'g - fg')/g^2$.

**Power (integer):**

$$(a + a'\epsilon)^n = a^n + na^{n-1}a'\epsilon$$

This is the power rule: $(x^n)' = nx^{n-1}$.

### Scalar-Dual Arithmetic

When a plain number (scalar) interacts with a dual number, the scalar is promoted to a dual number with zero derivative:

$$c + (a + a'\epsilon) = (c + a) + a'\epsilon$$

$$c \cdot (a + a'\epsilon) = ca + ca'\epsilon$$

This corresponds to the fact that the derivative of a constant is zero.

## Elementary Functions on Dual Numbers

### Sine

$$\sin(a + a'\epsilon) = \sin(a) + a'\cos(a)\epsilon$$

The value is $\sin(a)$ and the derivative is $a'\cos(a)$. This implements the chain rule: $\frac{d}{dx}\sin(f(x)) = f'(x)\cos(f(x))$.

### Cosine

$$\cos(a + a'\epsilon) = \cos(a) - a'\sin(a)\epsilon$$

### Exponential

$$\exp(a + a'\epsilon) = \exp(a) + a'\exp(a)\epsilon = \exp(a)(1 + a'\epsilon)$$

The derivative of $\exp$ is $\exp$ itself, so the tangent is $a'\exp(a)$.

### Logarithm

$$\log(a + a'\epsilon) = \log(a) + \frac{a'}{a}\epsilon$$

### Square Root

$$\sqrt{a + a'\epsilon} = \sqrt{a} + \frac{a'}{2\sqrt{a}}\epsilon$$

### General Rule

For any differentiable function $g$:

$$g(a + a'\epsilon) = g(a) + a'g'(a)\epsilon$$

The tangent is always $a' \cdot g'(a)$: the incoming tangent times the local derivative. This is the chain rule in action.

## The Chain Rule Emerges Automatically

The magic of dual numbers is that the chain rule is not explicitly programmed - it emerges from the arithmetic. Consider $h(x) = \sin(x^2)$:

1. Start with $x = (x_0, 1)$ (seed the tangent with 1)
2. Compute $x^2 = (x_0^2, 2x_0)$ using the multiplication rule
3. Compute $\sin(x^2) = (\sin(x_0^2), 2x_0 \cos(x_0^2))$ using the sine rule

The final tangent $2x_0\cos(x_0^2)$ is exactly $h'(x_0) = 2x\cos(x^2)$, obtained by the chain rule. But we never explicitly applied the chain rule - it was implicit in the dual number arithmetic.

This works for arbitrarily complex compositions. Each operation applies the local derivative rule, and the chain rule accumulates automatically through the tangent component.

## Implementation as a Class

A dual number can be implemented as a class with two attributes:

- `val`: the primal value (a regular float)
- `der`: the tangent/derivative value (a regular float)

The class needs to support:

- `__add__`, `__radd__`: addition with other duals or scalars
- `__mul__`, `__rmul__`: multiplication with other duals or scalars
- `__sub__`, `__rsub__`: subtraction
- `__neg__`: negation
- `__pow__`: power (for squaring and other integer powers)
- `__truediv__`, `__rtruediv__`: division

And standalone functions (or methods) for:
- `dual_sin(d)`: sine of a dual number
- `dual_exp(d)`: exponential of a dual number

The `__radd__` and `__rmul__` methods handle cases like `3 + dual` or `2 * dual`, where the scalar is on the left side and Python would normally call the scalar's `__add__` or `__mul__`, which don't know about dual numbers.

## Computing Derivatives with Dual Numbers

To compute $f'(x_0)$:

1. Create a dual number $x = \text{Dual}(x_0, 1.0)$ (seed tangent = 1)
2. Evaluate $f(x)$ using dual-number arithmetic
3. Read off the result: $f(x).\text{val} = f(x_0)$ and $f(x).\text{der} = f'(x_0)$

The seed tangent of 1 represents $\frac{dx}{dx} = 1$. If we wanted $\frac{\partial f}{\partial x_i}$ in a multivariate setting, we would seed $x_i$ with tangent 1 and all other inputs with tangent 0.

## The Test Function

The function $f(x) = x^2 \cdot \sin(x) + \exp(x)$ exercises multiple operations:

- $x^2$: multiplication ($x \cdot x$) or power
- $x^2 \cdot \sin(x)$: multiplication of two expressions
- $\exp(x)$: elementary function
- The sum $+$: addition

The analytical derivative is:

$$f'(x) = 2x\sin(x) + x^2\cos(x) + \exp(x)$$

This comes from the product rule on $x^2\sin(x)$ and the derivative of $\exp(x)$.

At $x = 2$:
- $f(2) = 4\sin(2) + e^2 \approx 4(0.9093) + 7.3891 \approx 11.026$
- $f'(2) = 4\sin(2) + 4\cos(2) + e^2 \approx 3.637 + (-1.665) + 7.389 \approx 9.362$

## Forward Mode vs Reverse Mode

### Forward Mode (Dual Numbers / JVP)

Forward-mode AD computes one column of the Jacobian per pass. For a function $f: \mathbb{R}^n \to \mathbb{R}^m$:

- One forward pass with seed tangent $\mathbf{e}_i$ gives $\frac{\partial f_j}{\partial x_i}$ for all $j$
- Computing the full Jacobian requires $n$ passes (one per input variable)
- Cost per pass: same as one function evaluation (constant factor overhead)

### Reverse Mode (Backpropagation / VJP)

Reverse-mode AD computes one row of the Jacobian per pass:

- One forward pass + backward pass with seed cotangent $\mathbf{e}_j$ gives $\frac{\partial f_j}{\partial x_i}$ for all $i$
- Computing the full Jacobian requires $m$ passes (one per output)

### When to Use Which

- **Forward mode is better when** $n \ll m$ (few inputs, many outputs). Example: computing the derivative of a function from $\mathbb{R}$ to $\mathbb{R}^{1000}$.
- **Reverse mode is better when** $m \ll n$ (many inputs, few outputs). Example: computing the gradient of a scalar loss with respect to millions of parameters.

Since neural network training computes the gradient of one scalar loss with respect to many parameters, reverse mode (backpropagation) is the standard choice. But forward mode has important uses:

- Computing Jacobian-vector products (JVPs) directly
- Hessian-vector products via forward-over-reverse AD
- Functions with scalar input (univariate derivatives)
- Research and prototyping

## Connection to JAX

JAX implements both forward and reverse mode AD:

- `jax.jvp(f, (x,), (v,))` computes the JVP: $f(x)$ and $J_f(x) \cdot v$ using forward mode
- `jax.vjp(f, x)` returns $f(x)$ and a function that computes $v^\top \cdot J_f(x)$ using reverse mode

Under the hood, `jax.jvp` uses a dual-number-like mechanism (called "tracers") to propagate tangents through the computation. The DualNumber class in this problem is a simplified version of what JAX does internally.

## Higher-Order Derivatives

Dual numbers can be nested to compute higher-order derivatives. To compute $f''(x)$, use dual numbers whose components are themselves dual numbers:

$$x = (x_0 + 1\cdot\epsilon_1) + 1\cdot\epsilon_2$$

where $\epsilon_1^2 = \epsilon_2^2 = 0$ but $\epsilon_1\epsilon_2 \neq 0$. Evaluating $f$ on this "hyper-dual" number gives $f(x_0)$, $f'(x_0)$, and $f''(x_0)$ simultaneously.

In JAX, this is done with nested `jvp` calls:

```
f''(x) = jvp(lambda x: jvp(f, (x,), (1.0,))[1], (x,), (1.0,))[1]
```

This "forward-over-forward" approach computes second derivatives exactly, without numerical approximation.

## Practical Considerations

### Operator Overloading

The implementation relies on Python's operator overloading: `__add__`, `__mul__`, etc. The same function code that works on regular floats also works on dual numbers, because the arithmetic operations are overloaded to carry derivatives. This is called "operator overloading" AD and is the simplest form of automatic differentiation.

### Handling Scalars

When a dual number interacts with a plain float (e.g., `2.0 * dual`), Python calls the float's `__mul__` first, which doesn't know about dual numbers. It returns `NotImplemented`, and Python then tries the dual number's `__rmul__`. This is why both `__mul__` and `__rmul__` must be implemented.

### Numerical Accuracy

Unlike finite differences, dual number AD computes exact derivatives (up to floating-point arithmetic). There is no step size $h$ to choose and no tradeoff between truncation and cancellation errors. The only source of error is the inherent floating-point imprecision of the elementary operations, which is typically on the order of machine epsilon ($\sim 10^{-16}$).

## Worked Example: Step by Step

Let us trace the computation of $f(x) = x^2 \cdot \sin(x) + \exp(x)$ at $x = 2$ using dual numbers.

### Initialization

$$x = (2, 1)$$

The value is 2 and the tangent is 1 (we are computing $df/dx$, and $dx/dx = 1$).

### Computing $x^2$

Using dual multiplication $(a, a') \cdot (b, b') = (ab, ab' + a'b)$:

$$x \cdot x = (2, 1) \cdot (2, 1) = (4, 2 \cdot 1 + 1 \cdot 2) = (4, 4)$$

The value is $2^2 = 4$ and the tangent is $2 \cdot 2 = 4$, which is the derivative of $x^2$ evaluated at $x = 2$.

### Computing $\sin(x)$

Using the dual sine rule $\sin(a, a') = (\sin(a), a'\cos(a))$:

$$\sin(x) = \sin(2, 1) = (\sin(2), 1 \cdot \cos(2)) = (0.9093, -0.4161)$$

### Computing $x^2 \cdot \sin(x)$

$$(4, 4) \cdot (0.9093, -0.4161) = (4 \cdot 0.9093, \; 4 \cdot (-0.4161) + 4 \cdot 0.9093)$$

$$= (3.6372, \; -1.6645 + 3.6372) = (3.6372, 1.9727)$$

The tangent $1.9727$ equals $2x\sin(x) + x^2\cos(x)$ at $x = 2$, which is the product rule derivative.

### Computing $\exp(x)$

Using the dual exp rule $\exp(a, a') = (\exp(a), a'\exp(a))$:

$$\exp(x) = \exp(2, 1) = (e^2, 1 \cdot e^2) = (7.3891, 7.3891)$$

### Final Addition

$$(3.6372, 1.9727) + (7.3891, 7.3891) = (11.0263, 9.3618)$$

So $f(2) = 11.0263$ and $f'(2) = 9.3618$.

### Verification

The analytical derivative $f'(x) = 2x\sin(x) + x^2\cos(x) + e^x$ at $x = 2$:

$$f'(2) = 4\sin(2) + 4\cos(2) + e^2 = 3.6372 - 1.6645 + 7.3891 = 9.3618$$

The dual number result matches exactly.

## The Algebra of Dual Numbers

### Formal Definition

The dual numbers form a commutative ring $\mathbb{R}[\epsilon]/(\epsilon^2)$. This means they are real polynomials in $\epsilon$ modulo the relation $\epsilon^2 = 0$. Every dual number can be written uniquely as $a + b\epsilon$ with $a, b \in \mathbb{R}$.

### Key Properties

The dual numbers share many properties with the real numbers:

- **Commutativity:** $(a + a'\epsilon)(b + b'\epsilon) = (b + b'\epsilon)(a + a'\epsilon)$
- **Associativity:** both addition and multiplication are associative
- **Distributivity:** multiplication distributes over addition
- **Additive identity:** $0 + 0\epsilon$
- **Multiplicative identity:** $1 + 0\epsilon$

However, dual numbers differ from real numbers in one important way: they have zero divisors. The element $0 + 1\epsilon$ is nonzero, but $(0 + 1\epsilon)^2 = 0 + 0\epsilon$. This means not every nonzero dual number has a multiplicative inverse. Specifically, $a + a'\epsilon$ has an inverse if and only if $a \neq 0$:

$$\frac{1}{a + a'\epsilon} = \frac{1}{a} - \frac{a'}{a^2}\epsilon$$

### Comparison with Complex Numbers

| Property | Complex: $a + bi$ | Dual: $a + b\epsilon$ |
|----------|-------------------|----------------------|
| Key relation | $i^2 = -1$ | $\epsilon^2 = 0$ |
| Geometric meaning | Rotation | Tangent/derivative |
| Zero divisors? | No | Yes |
| Field? | Yes | No (ring only) |
| Used for | Signal processing, quantum mechanics | Automatic differentiation |

## Applications Beyond Scalar Derivatives

### Gradient Computation

For $f: \mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f$ can be computed by running $n$ forward passes, each with a different seed tangent (the $i$-th standard basis vector). This gives $\frac{\partial f}{\partial x_i}$ for each $i$. While this requires $n$ passes (expensive for large $n$), it is simple and useful for functions with few inputs.

### Jacobian-Vector Products

The JVP $J_f(x) \cdot v$ can be computed in a single forward pass by seeding the tangent with $v$ instead of a basis vector. This is useful for:

- Hessian-vector products: $H v = \frac{\partial}{\partial t}\Big|_{t=0} \nabla f(x + tv)$, computed by applying forward mode to the gradient function
- Conjugate gradient methods that only need matrix-vector products, not the full matrix
- Neural tangent kernel computation

### Sensitivity Analysis

In scientific computing, forward-mode AD is used to compute how the output of a simulation depends on input parameters. If a simulation has few parameters but many outputs (e.g., a fluid simulation with a few boundary conditions but millions of grid points), forward mode is more efficient than reverse mode.
