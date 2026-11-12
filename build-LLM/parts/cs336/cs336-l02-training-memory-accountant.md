# Mixed-Precision Training Memory Accountant

Before training a model you need to know whether it fits on your GPUs. Most of the memory goes to four things: the parameters, their gradients, the activations saved for the backward pass, and the optimizer's extra state. The heart of this problem is counting elements exactly from tensor shapes and charging each category its own bytes per element.

## Elements from shapes

A tensor with shape $(d_1, d_2, \ldots, d_n)$ holds

$$
N = \prod_{i=1}^{n} d_i
$$

elements, and storing them at $w$ bytes per element costs $M = N w$ bytes.

Two edge cases follow from the product. An empty shape is a scalar, and the empty product is 1, so a scalar holds one element. Any dimension of size 0 makes the whole product 0, so that tensor costs nothing.

## The four categories

**Parameters** are the model weights. Their cost is the total parameter element count times the parameter byte width.

**Gradients** have exactly the same shapes as the parameters, because there is one gradient value per weight. They use the same element count with their own byte width, which can differ from the parameter width in mixed-precision training.

**Activations** are the intermediate values from the forward pass that the backward pass needs. Their shapes depend on batch size and sequence length as well as layer widths, so they cannot be worked out from the parameter shapes alone.

**Optimizer state** depends on the optimizer. Each state tensor has the same shape as the parameters, so its cost is a multiple of the parameter element count:

- SGD keeps no extra state, so it adds nothing.
- AdaGrad keeps one running sum of squared gradients per parameter.
- Adam keeps two values per parameter, the running mean of the gradient and the running mean of its square.

The total is the sum of the four categories.

## Why every category has its own byte width

Mixed-precision training stores different tensors at different precisions. A common setup keeps parameters and activations in 16-bit formats (2 bytes) to save memory and speed up matrix multiplication, while optimizer state stays in 32-bit floats (4 bytes) because the running averages need the extra precision. Gradients may be 2 or 4 bytes depending on the setup.

Because each category really is stored at its own precision, one shared width would misstate the total. Many mixed-precision setups also keep a full 32-bit master copy of the weights for the optimizer to update, which adds 4 more bytes per parameter. This estimate leaves that copy out, along with memory lost to the allocator.

## Why optimizer state dominates

Consider a model with $P$ parameters trained with Adam, using 2-byte parameters and gradients and 4-byte optimizer state. Per parameter, the weights cost 2 bytes, the gradient 2 bytes, and the two Adam states 8 bytes. The optimizer state alone is two thirds of the 12 bytes per parameter, before any activations.

For a 7 billion parameter model that is about 84 GB, nearly the entire memory of an 80 GB GPU before a single activation is stored. This kind of estimate is why techniques such as ZeRO (Rajbhandari et al., 2020) split optimizer state across GPUs.

## A worked example

Take parameter shapes $[3, 2]$ and $[2]$, activation shapes $[4, 2]$ and $[4, 2]$, Adam, 2-byte parameters, 4-byte gradients, 2-byte activations and 4-byte optimizer state.

The parameters hold $3 \times 2 + 2 = 8$ elements and the activations hold $4 \times 2 + 4 \times 2 = 16$ elements. The four categories are

$$
\text{parameters} = 8 \times 2 = 16, \qquad \text{gradients} = 8 \times 4 = 32
$$

$$
\text{activations} = 16 \times 2 = 32, \qquad \text{optimizer state} = 2 \times 8 \times 4 = 64
$$

The total is $16 + 32 + 32 + 64 = 144$ bytes. Switching to SGD removes the 64 bytes of optimizer state and leaves 80 bytes, and switching to AdaGrad halves the optimizer state to 32 bytes, for a total of 112.

## Byte counts outgrow fixed-width integers

Real training runs have billions of parameters and activations, and byte counts quickly pass the range of 32-bit integers. Python integers grow as needed, so plain integer multiplication gives exact answers. Floating-point arithmetic would round large counts.

## Common mistakes to avoid

- **Treating an empty shape as zero elements.** A scalar has one element, because the empty product is 1.
- **Using the parameter width for gradients.** Each category has its own bytes per element.
- **Basing optimizer state on activations.** Optimizer state follows the parameter element count only.
- **Getting the state count wrong.** SGD has zero state tensors, AdaGrad one and Adam two.
- **Adding a master-weight copy.** This estimate excludes extra copies and allocator overhead.
- **Using floating-point arithmetic.** Large counts must stay exact Python integers.
- **Counting optimizer state per tensor instead of per element.** Adam's two states are full-size copies of every parameter, so the multiplier applies to the element count.
