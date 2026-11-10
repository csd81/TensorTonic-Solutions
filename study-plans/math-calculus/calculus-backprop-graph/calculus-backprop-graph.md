# <span style="font-size: 20px;">Backpropagation Through a Computation Graph</span>

## What Is a Computation Graph?

Every numerical computation can be represented as a directed acyclic graph (DAG). The nodes represent operations or values, and the edges represent data flow. Leaf nodes hold input values (weights, data). Internal nodes perform operations (multiply, add, apply activation). The final node produces the output (the loss).

For the computation `output = relu(w1 * x + w2 * x^2)`, the graph has:

- **Leaf nodes:** `w1`, `w2`, `x` (the inputs)
- **Operation nodes:**
  - `x_sq = x * x` (squaring)
  - `prod1 = w1 * x` (first product)
  - `prod2 = w2 * x_sq` (second product)
  - `sum_val = prod1 + prod2` (addition)
  - `output = relu(sum_val)` (activation)

The graph is "directed" because data flows from inputs to output, and "acyclic" because there are no loops.

## The Forward Pass

The forward pass evaluates the graph from leaves to root. Starting with known values for `w1`, `w2`, and `x`, each node computes its value from its inputs:

1. Compute `x_sq = x * x`
2. Compute `prod1 = w1 * x`
3. Compute `prod2 = w2 * x_sq`
4. Compute `sum_val = prod1 + prod2`
5. Compute `output = relu(sum_val)`

This is just evaluating the function. The key insight of backpropagation is that we save all intermediate values during the forward pass, because they are needed for the backward pass.

## The Backward Pass: Reverse-Mode Differentiation

### The Goal

Given the forward computation, we want to compute the gradient of the output with respect to every leaf node: $\frac{\partial \text{output}}{\partial w_1}$, $\frac{\partial \text{output}}{\partial w_2}$, and $\frac{\partial \text{output}}{\partial x}$.

### The Key Idea

The backward pass propagates gradients from the output back to the inputs, using the chain rule at each node. For each node $v$ in the graph:

$$
\begin{aligned}
\frac{\partial \text{output}}{\partial v} &= \sum_{u \in \text{children}(v)} \frac{\partial \text{output}}{\partial u} \cdot \frac{\partial u}{\partial v}
\end{aligned}
$$

where the sum is over all nodes $u$ that directly depend on $v$, and $\frac{\partial u}{\partial v}$ is the local derivative of the operation at $u$ with respect to its input $v$.

This is the multivariate chain rule in action: when a node feeds into multiple downstream nodes, the gradients from all paths are summed.

### Seeding the Backward Pass

The backward pass starts at the output node with gradient 1:

$$\frac{\partial \text{output}}{\partial \text{output}} = 1$$

This seed is then propagated backward through the graph.

## Local Derivatives of Each Operation

### Multiplication: $z = a \cdot b$

The partial derivatives are:

$$\frac{\partial z}{\partial a} = b, \quad \frac{\partial z}{\partial b} = a$$

If the upstream gradient is $\bar{z} = \frac{\partial \text{output}}{\partial z}$, then:

$$\bar{a} = \bar{z} \cdot b, \quad \bar{b} = \bar{z} \cdot a$$

The gradient with respect to each input is the upstream gradient times the other input. This is the chain-rule version of the product rule.

### Addition: $z = a + b$

The partial derivatives are:

$$\frac{\partial z}{\partial a} = 1, \quad \frac{\partial z}{\partial b} = 1$$

Both inputs receive the same upstream gradient:

$$\bar{a} = \bar{z}, \quad \bar{b} = \bar{z}$$

Addition distributes gradient equally. This makes intuitive sense: increasing either input by $\delta$ increases the output by $\delta$.

### ReLU: $z = \text{relu}(a) = \max(0, a)$

The derivative is:

$$\frac{\partial z}{\partial a} = \begin{cases} 1 & \text{if } a > 0 \\ 0 & \text{if } a < 0 \end{cases}
$$

The gradient passes through unchanged when the input is positive, and is killed (set to zero) when the input is negative. At $a = 0$ exactly, the subgradient convention uses 0.

$$\bar{a} = \bar{z} \cdot \mathbb{1}[a > 0]$$

This is why ReLU networks can suffer from "dead neurons": if a neuron's pre-activation is always negative, its gradient is always zero, and it can never learn.

### Squaring: $z = a^2 = a \cdot a$

This is a special case of multiplication where both inputs are the same:

$$\frac{\partial z}{\partial a} = 2a$$

But from the perspective of the computation graph, if we implement squaring as `x * x`, the gradient accumulates from both "uses" of `x`:

$$\bar{a} = \bar{z} \cdot a + \bar{z} \cdot a = 2\bar{z} \cdot a$$

This is gradient accumulation in action: the variable `x` participates in the multiplication twice, so its gradient is the sum of contributions from both roles.

## Gradient Accumulation

### The Fan-Out Problem

When a single node feeds into multiple downstream operations, its gradient is the sum of all downstream contributions. In our computation:

$$x \text{ feeds into } \begin{cases} \text{prod1} = w_1 \cdot x \\ x\_sq = x \cdot x \end{cases}
$$

So the gradient of the output with respect to $x$ has two components:

$$
\begin{aligned}
\frac{\partial \text{output}}{\partial x} &= \underbrace{\frac{\partial \text{output}}{\partial \text{prod1}} \cdot \frac{\partial \text{prod1}}{\partial x}}_{\text{path through prod1}} \\
&\quad + \underbrace{\frac{\partial \text{output}}{\partial x\_sq} \cdot \frac{\partial x\_sq}{\partial x}}_{\text{path through x\_sq}}
\end{aligned}
$$

This summing of gradients from multiple paths is called **gradient accumulation** and is one of the most important aspects of backpropagation. In PyTorch, this is why `loss.backward()` accumulates gradients (they are added, not overwritten) and why you need to call `optimizer.zero_grad()` before each backward pass.

### Why Accumulation Is Correct

The mathematical justification comes from the total derivative. If $f(x) = g(x, h(x))$ where $x$ appears in both the first and second arguments, then:

$$\frac{df}{dx} = \frac{\partial g}{\partial x}\bigg|_{\text{direct}} + \frac{\partial g}{\partial h} \cdot \frac{dh}{dx}$$

Each path from $x$ to the output contributes independently to the total gradient, and these contributions add up.

## Tracing Through the Backward Pass

Let us work through the backward pass for `output = relu(w1 * x + w2 * x^2)`:

### Step 1: Output Node

$$\bar{\text{output}} = 1$$

### Step 2: Through ReLU

$$\bar{\text{sum\_val}} = \bar{\text{output}} \cdot \mathbb{1}[\text{sum\_val} > 0]$$

If `sum_val > 0`, then $\bar{\text{sum\_val}} = 1$. If `sum_val <= 0`, then $\bar{\text{sum\_val}} = 0$ and all downstream gradients are zero.

### Step 3: Through Addition

$$\bar{\text{prod1}} = \bar{\text{sum\_val}}, \quad \bar{\text{prod2}} = \bar{\text{sum\_val}}$$

### Step 4: Through First Multiplication ($\text{prod1} = w_1 \cdot x$)

$$\bar{w_1} = \bar{\text{prod1}} \cdot x, \quad \bar{x} \mathrel{+}= \bar{\text{prod1}} \cdot w_1$$

Note the $\mathrel{+}=$: this is gradient accumulation. We add to $\bar{x}$ rather than overwriting, because $x$ is used elsewhere in the graph.

### Step 5: Through Second Multiplication ($\text{prod2} = w_2 \cdot x\_sq$)

$$\bar{w_2} = \bar{\text{prod2}} \cdot x\_sq, \quad \bar{x\_sq} = \bar{\text{prod2}} \cdot w_2$$

### Step 6: Through Squaring ($x\_sq = x \cdot x$)

$$\bar{x} \mathrel{+}= \bar{x\_sq} \cdot 2x$$

Again, we accumulate into $\bar{x}$.

### Final Gradients

Combining:

$$\frac{\partial \text{output}}{\partial w_1} = \bar{\text{sum\_val}} \cdot x$$

$$\frac{\partial \text{output}}{\partial w_2} = \bar{\text{sum\_val}} \cdot x^2$$

$$\frac{\partial \text{output}}{\partial x} = \bar{\text{sum\_val}} \cdot (w_1 + 2 w_2 x)$$

where $\bar{\text{sum\_val}} = \mathbb{1}[w_1 x + w_2 x^2 > 0]$.

## Connection to PyTorch Autograd

### The Tape

PyTorch's autograd records a "tape" of operations during the forward pass. Each operation (tensor multiplication, addition, activation) creates a node in the computation graph, storing references to its inputs and the function needed for the backward pass. When you call `loss.backward()`, PyTorch traverses this graph in reverse topological order, applying exactly the procedure described above.

### The `backward` Method

Each autograd function in PyTorch has a `backward` method that computes local gradients. For example:

- `MulBackward` computes the multiplication backward rule (swap the inputs and multiply by upstream gradient)
- `AddBackward` passes the upstream gradient through unchanged
- `ReluBackward` masks the gradient using the saved pre-activation sign

### The `.grad` Attribute

After `loss.backward()`, each leaf tensor's `.grad` attribute contains the accumulated gradient. This is the sum over all paths from the output to that leaf, computed by reverse-mode AD.

### Why Reverse Mode Is Efficient

For a function $f: \mathbb{R}^n \to \mathbb{R}$ (like a neural network with $n$ parameters and scalar loss), reverse-mode computes all $n$ partial derivatives in a single backward pass. Forward mode would require $n$ separate forward passes. Since neural networks typically have millions of parameters, reverse mode is vastly more efficient.

The cost of one backward pass is approximately 2-3x the cost of one forward pass, regardless of the number of parameters. This is why training a neural network is only a constant factor more expensive than inference.

## Topological Ordering

### Why Order Matters

The backward pass must process nodes in reverse topological order: a node's gradient must be fully accumulated before it is used to compute gradients for its inputs. If we process nodes out of order, we might use an incomplete gradient.

In our example, we must compute $\bar{\text{sum\_val}}$ before processing the multiplication nodes, and we must compute $\bar{x\_sq}$ before accumulating its contribution to $\bar{x}$.

### Computing the Order

For a DAG, a topological sort can be computed using depth-first search (DFS) or Kahn's algorithm. The reverse of this order gives the backward processing order. PyTorch computes this ordering dynamically as part of `backward()`.

## Generalizing to Larger Graphs

### Multi-Layer Networks

A multi-layer neural network is a deeper computation graph:

$$L = \text{loss}(\text{relu}(W_3 \cdot \text{relu}(W_2 \cdot \text{relu}(W_1 \cdot x))))$$

The backward pass applies the same rules layer by layer, propagating gradients from the loss back through each activation and weight multiplication.

### Shared Parameters

When parameters are shared (used in multiple places), gradient accumulation ensures correct gradients. For example, in recurrent neural networks (RNNs), the same weight matrix is used at every time step. The gradient of the loss with respect to the shared weight is the sum of contributions from all time steps.

### Control Flow

Modern frameworks like PyTorch handle control flow (if-statements, loops) by recording the actual operations that were performed during the forward pass. The backward pass only differentiates through the operations that actually executed. This is called "define-by-run" and is what makes PyTorch's autograd flexible.

## Numerical Verification

### Finite Difference Check

The computed gradients can be verified using finite differences:

$$\frac{\partial f}{\partial w_1} \approx \frac{f(w_1 + h, w_2, x) - f(w_1 - h, w_2, x)}{2h}$$

for a small $h$ (typically $10^{-5}$). If the backpropagation gradients match the finite differences within a relative tolerance of $10^{-5}$, the implementation is likely correct. This is what `torch.autograd.gradcheck` does.

### Common Bugs

The most common backpropagation bugs are:

- **Forgetting gradient accumulation:** Using assignment instead of accumulation when a variable has multiple consumers
- **Wrong local derivative:** Swapping the operands in multiplication backward, or using the wrong derivative for an activation
- **Missing the relu gate:** Not masking the gradient when the pre-activation is negative
- **Off-by-one in topological order:** Processing a node before its gradient is fully accumulated

## The Mathematics of Backpropagation

### Matrix Formulation

For a single dense layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ followed by activation $\mathbf{a} = \sigma(\mathbf{z})$, the backward pass computes:

$$\frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{a}} \odot \sigma'(\mathbf{z})$$

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{z}} \cdot \mathbf{x}^\top$$

$$\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$$

$$\frac{\partial L}{\partial \mathbf{x}} = W^\top \cdot \frac{\partial L}{\partial \mathbf{z}}$$

where $\odot$ denotes elementwise multiplication. The gradient $\frac{\partial L}{\partial \mathbf{x}}$ is passed to the previous layer as its "upstream gradient."

### The Jacobian-Vector Product View

Each backward step computes a vector-Jacobian product (VJP). For a function $\mathbf{y} = f(\mathbf{x})$ with Jacobian $J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$, the backward pass computes:

$$\bar{\mathbf{x}} = J^\top \bar{\mathbf{y}}$$

where $\bar{\mathbf{y}}$ is the upstream gradient. This VJP can be computed without forming the full Jacobian, which is why backpropagation is efficient.

### Computational Cost

The backward pass through a computation graph visits each edge exactly once, computing one multiplication and one addition per edge. For a graph with $E$ edges:

- Forward pass: $O(E)$ operations (one per edge)
- Backward pass: $O(E)$ operations (one per edge)

The backward pass also requires $O(V)$ storage for the saved forward values, where $V$ is the number of nodes. This memory cost is the main bottleneck in training large models, and techniques like gradient checkpointing trade additional computation for reduced memory.

## Beyond Simple Graphs

### Gradient Checkpointing

For very deep networks, storing all intermediate activations requires memory proportional to the depth. Gradient checkpointing saves memory by only storing activations at selected "checkpoints" and recomputing intermediate values during the backward pass. This trades $O(\sqrt{n})$ memory for $O(n)$ recomputation, where $n$ is the depth.

### Higher-Order Gradients

Backpropagation can be applied to itself to compute second-order derivatives. This is used in:

- **Hessian-vector products:** Compute $H\mathbf{v}$ by differentiating the gradient-vector product $\nabla f(\mathbf{x})^\top \mathbf{v}$
- **Meta-learning (MAML):** Differentiate through the optimization process, requiring gradients of gradients
- **Adversarial training:** The inner maximization requires differentiating through gradient steps

PyTorch supports this via `create_graph=True` in `backward()`, which tells the autograd engine to build a computation graph of the backward pass itself.

### Implicit Differentiation

Some computations involve implicit equations, such as the output of an optimization problem or a fixed-point iteration. The implicit function theorem provides a way to compute gradients through these operations without unrolling the iteration. This is the basis of deep equilibrium models (DEQ) and differentiable optimization layers.

## Historical Context

### The Discovery of Backpropagation

The algorithm we now call backpropagation was discovered independently multiple times:

- Seppo Linnainmaa (1970) described the reverse mode of automatic differentiation
- Paul Werbos (1974) applied it to neural networks in his PhD thesis
- Rumelhart, Hinton, and Williams (1986) popularized it with their Nature paper

The key insight that made deep learning practical was not the algorithm itself, but the realization that gradient computation through many layers could be done efficiently in $O(n)$ time rather than the $O(n^2)$ that naive approaches would require.

### Modern Autograd Frameworks

Modern frameworks generalize backpropagation beyond simple feedforward networks:

- **PyTorch** uses dynamic computation graphs ("define-by-run"), building the graph during each forward pass
- **JAX** uses program transformation, converting Python functions into their gradient functions via `jax.grad`
- **TensorFlow 2.x** uses eager mode with GradientTape, similar to PyTorch's approach

All of these implement the same mathematical algorithm: reverse-mode automatic differentiation through a computation graph, with gradient accumulation at fan-out nodes
