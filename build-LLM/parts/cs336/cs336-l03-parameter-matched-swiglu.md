# Parameter-Matched SwiGLU Block

SwiGLU is the MLP block used in Llama, PaLM and most recent language models. It replaces the classic two-matrix feed-forward layer with a gated version that uses three matrices. The heart of this problem is choosing the hidden width so the gated block uses the same parameter budget as the block it replaces, then computing the gate and value paths correctly.

## The classic feed-forward block

The original transformer MLP has two weight matrices. The first projects from model width $d$ up to a hidden width, usually $4d$. An activation such as ReLU or GELU is applied, and the second matrix projects back down to $d$. With no biases, the block holds

$$
2 \times d \times 4d = 8d^2
$$

parameters. This budget is the reference that other MLP designs are compared against.

## Gating with SiLU

A gated linear unit computes two projections of the same input. One passes through an activation and acts as a gate, and the other carries the values. Multiplying them elementwise lets the network decide, per hidden unit and per token, how much of each value to let through.

SwiGLU uses the SiLU activation, also called Swish, for the gate:

$$
\operatorname{SiLU}(z) = \frac{z}{1 + e^{-z}}
$$

SiLU is close to zero for large negative $z$, close to $z$ for large positive $z$, and smooth in between. For an input row $x$ of width $d$ and a hidden width $h$, the block uses gate weights $W_g \in \mathbb{R}^{d \times h}$, value weights $W_v \in \mathbb{R}^{d \times h}$ and output weights $W_o \in \mathbb{R}^{h \times d}$, and computes

$$
y = \left( \operatorname{SiLU}(x W_g) \odot (x W_v) \right) W_o
$$

where $\odot$ is elementwise multiplication. For example, a gate pre-activation of 2 and a value of 3 give $\operatorname{SiLU}(2) \times 3 = 1.761594 \times 3 = 5.284782$, while a gate pre-activation of $-5$ would shrink the same value to about $-0.1$.

## Matching the parameter budget

The block therefore holds $3dh$ parameters. Keeping a parameter budget $P$ fixed means choosing

$$
h \approx \frac{P}{3d}
$$

For the classic budget $P = 8d^2$ this gives $h = 8d/3$, which is why the original Llama models used a hidden width of about $2.7d$ rather than $4d$. Giving a SwiGLU block the same width as a classic block would give it 50 percent more parameters and make the comparison unfair.

Because $P / 3d$ is rarely a whole number, the width has to be rounded to a whole count of hidden units. This exercise rounds to the nearest integer and breaks exact ties upward. The width is then kept between 1 and $H$. The lower bound guarantees at least one hidden unit, and the upper bound is the largest width the supplied weights can provide.

## Using a slice of the weight pool

The exercise supplies weights at a maximum width $H$ so that one set of tensors can serve any budget. A block of width $h$ uses the first $h$ columns of the gate and value matrices and the first $h$ rows of the output matrix. Those slices line up, since hidden unit $j$ is column $j$ of $W_g$ and $W_v$ and row $j$ of $W_o$.

Only the sliced matrices count toward the budget, so the parameter count is $3dh$ for the selected width $h$.

## A budget that falls between widths

Take $d = 4$, $P = 100$ and a pool width $H = 16$. Then

$$
\frac{100}{3 \times 4} = 8.333\ldots
$$

which rounds to $h = 8$. The block uses $3 \times 4 \times 8 = 96$ parameters, the closest whole-unit match to 100, since a width of 9 would use 108.

For an exact tie, take $d = 2$ and $P = 27$. The ratio is exactly 4.5, and ties round up, so the width is 5. If the pool only had $H = 3$, the width would be capped at 3 and the parameter count would be 18.

## Why the gate runs in float32

The SiLU exponential and the elementwise product of two projections lose accuracy quickly in bfloat16. In this problem the projections and the gating are computed in float32, and the output is converted back to the input's format.

## Common mistakes to avoid

- **Applying SiLU to the value path.** Only the gate projection goes through the activation.
- **Rounding exact ties down.** A ratio that ends in exactly one half rounds up to the next width.
- **Forgetting the clamps.** The width must be at least 1 and at most the pool width $H$.
- **Slicing the wrong axis.** The gate and value matrices are sliced by columns and the output matrix by rows.
- **Counting the whole pool.** Only the selected $3dh$ parameters belong to the block.
- **Matching width instead of parameters.** Using $h = 4d$ gives the gated block 50 percent more parameters than the classic block.
