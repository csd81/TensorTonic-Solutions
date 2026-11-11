# SGD with Momentum over a Pytree

In JAX, the parameters of a model usually live in ordinary Python dicts and lists with arrays inside them, and an optimizer has to update every one of those arrays without knowing in advance how they are nested. This page starts with momentum on a single number and explains why it helps. It then shows how JAX treats a nested container of arrays as one object called a pytree, and how a single function can be applied to every array inside it. The problem combines the two ideas into a momentum update that works on any parameter tree.

## Plain gradient descent on one number

Take one parameter $p$ that starts at 0, and suppose its gradient $g$ is 1 at every step. Plain stochastic gradient descent (SGD) moves the parameter a small distance against the gradient.

$$
p \leftarrow p - \eta \, g
$$

The learning rate $\eta$ sets the size of that distance. With $\eta = 0.1$, three steps take the parameter to $-0.1$, $-0.2$ and $-0.3$. Every step has the same length because each one only looks at the current gradient.

This has two weaknesses. When the gradient is small but points the same way for many steps, progress is slow. When the loss surface is a narrow valley, the gradient flips sign from one side of the valley to the other, and the parameter zigzags across it instead of moving along it.

## Momentum remembers the recent direction

Momentum keeps a second number for each parameter, the velocity $v$, which collects recent gradients. Each step first updates the velocity and then moves the parameter by the new velocity.

$$
v_{\text{new}} = \mu \, v + g, \qquad p_{\text{new}} = p - \eta \, v_{\text{new}}
$$

The momentum coefficient $\mu$ is a number between 0 and 1 that says how much of the old velocity survives each step. With $\mu = 0$ the velocity is just the current gradient and the rule becomes plain SGD again.

Repeat the earlier experiment with $\mu = 0.9$, the same constant gradient of 1 and the velocity starting at 0. The velocity after three steps is 1, 1.9 and 2.71, and the parameter reaches $-0.1$, $-0.29$ and $-0.561$, almost twice as far as plain SGD after three steps. If the gradient stays constant forever, the velocity approaches $g / (1 - \mu)$, which is 10 here. In a steady direction, momentum acts like a learning rate ten times larger.

Now let the gradient alternate between $+1$ and $-1$, as it does when a parameter bounces between the walls of a valley. The velocity becomes 1, then $-0.1$, then 0.91, then $-0.181$. Each new gradient mostly cancels the remembered one, so the movement along the bouncing direction stays small while consistent directions keep building speed. This picture of a heavy ball rolling downhill is why the method is often called heavy-ball momentum (Polyak, 1964).

Some libraries fold the learning rate into the velocity and subtract the velocity directly. The two forms trace the same path as long as the learning rate never changes, but the stored velocity differs by a factor of $\eta$, so a velocity saved by one form cannot be fed into the other. This problem uses the form above, in which the velocity collects raw gradients.

## Why JAX keeps parameters in pytrees

JAX functions are pure, which means they read their inputs and return outputs without hidden state. A JAX model has no module object that owns its weights. The parameters are passed into the loss function as an argument, the gradients come back as a value, and the optimizer returns new parameters as another value.

A real model has many arrays, such as a weight matrix and a bias vector for every layer, so they are grouped into nested containers. JAX calls any such nesting a pytree. Dicts, lists, tuples and None are the inner nodes of the tree, and everything else, usually an array or a number, is a leaf. A single array on its own also counts as a pytree with one leaf.

## Leaves and structure

Consider the tree {"w": [1.0, 2.0], "b": 0.5, "layers": [{"k": a 2 by 2 matrix}]}, where the value under w is one array. It has three leaves, the array under w, the scalar under b and the matrix under k. JAX describes its shape as a tree definition (often written treedef), which prints as PyTreeDef({'b': \*, 'layers': [{'k': \*}], 'w': \*}). Each star marks a place where a leaf sits, and the dict keys appear in sorted order because JAX always visits dict entries by sorted key.

Flattening a tree splits it into two parts, the list of leaves in that fixed order and the tree definition. Unflattening puts a list of leaves back into the same structure. Two trees with equal tree definitions line up leaf for leaf, which is exactly what an optimizer needs. The gradients returned by jax.grad have the same structure as the parameters, and a velocity tree created from the parameters has it too, so the three trees match position by position.

## Mapping a function over matching leaves

The tool that ties this together is a tree map. It takes a function and one or more trees, calls the function on each group of matching leaves, and rebuilds a tree of the same structure from the results. With one tree it behaves like Python's map applied to every leaf. With several trees, the first tree decides the structure, and the others must have the same structure, otherwise JAX raises an error that names the mismatch.

The function inside a tree map only ever sees arrays, never containers. That is why one optimizer update can serve a two-leaf toy model and a transformer with hundreds of arrays, because the nesting is handled by the tree map and the arithmetic is written once for a single leaf.

## A two-leaf example

Take params = {"w": [1.0, -1.0], "b": 0.5}, grads = {"w": [0.2, 0.4], "b": -1.0} and velocity = {"w": [1.0, 0.0], "b": 0.5}, with $\mu = 0.5$ and $\eta = 0.1$.

For the leaf w, the new velocity is $0.5 \cdot [1.0, 0.0] + [0.2, 0.4] = [0.7, 0.4]$, and the new parameter is $[1.0, -1.0] - 0.1 \cdot [0.7, 0.4] = [0.93, -1.04]$. For the leaf b, the new velocity is $0.5 \cdot 0.5 - 1.0 = -0.75$, and the new parameter is $0.5 + 0.075 = 0.575$.

The results are new_params = {"b": 0.575, "w": [0.93, -1.04]} and new_velocity = {"b": -0.75, "w": [0.7, 0.4]}. Each leaf was updated with the same rule, and the two results kept the structure of params.

## Returning new trees instead of changing old ones

JAX arrays are immutable, so no operation can change the numbers inside an existing array. The containers around them are ordinary Python dicts and lists, though, and those can be changed in place. Writing new arrays into the params dict that was passed in also changes the caller's copy, which the caller may still need, for example to compare the loss before and after a step.

There is a second reason to avoid it. When a function runs under jax.jit, JAX hands it fresh containers built from the traced inputs, so changes to those containers never reach the caller. Code that relies on in-place changes therefore behaves differently with and without jit. Returning new trees gives the same result in both cases.

## Keeping each leaf's dtype

Large models often store some parameters in bfloat16, a 16-bit floating-point format with the exponent range of float32 and fewer digits of precision. An optimizer should not silently turn those leaves into float32, because that doubles their memory and changes the dtype the rest of the program expects.

JAX helps here through weak typing. A plain Python float such as 0.9 has no fixed precision, so multiplying it by a bfloat16 array gives a bfloat16 result. A NumPy or JAX scalar with an explicit float32 dtype, such as one made with numpy.float32(0.9), is strongly typed, and multiplying it by a bfloat16 array promotes the result to float32. Passing the hyperparameters as Python floats keeps every leaf in its own dtype.

## Common mistakes to avoid

- **Looping over the top-level dict.** A loop over params.items() only handles one flat dict and breaks on nested dicts, lists of layers or a bare array.
- **Stepping with the old velocity.** The parameter must move by the velocity that already includes the current gradient, otherwise the newest gradient has no effect until the next step.
- **Putting the learning rate inside the velocity.** Scaling the gradient by $\eta$ before adding it to the velocity follows a different convention and gives different numbers whenever the incoming velocity is not zero.
- **Changing the input containers.** Assigning into params, grads or velocity changes the caller's trees, and the change disappears under jax.jit.
- **Promoting low-precision leaves.** A float32 scalar or an explicit cast turns bfloat16 leaves into float32 and changes the dtype of the model.
- **Converting leaves to NumPy.** NumPy functions return NumPy arrays instead of JAX arrays and fail on traced values inside jax.jit.