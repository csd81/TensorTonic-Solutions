# Tensor-Parallel MLP

Data parallelism gives every device a full copy of the weights, which stops working once the weights themselves are too large for one device. Tensor parallelism splits each weight matrix across devices instead, so every device stores and multiplies only a slice. This page shows why a matrix product can be split by columns for free, why splitting by rows needs a sum, and how pairing the two splits lets a whole two-layer MLP run with a single collective.

## When the weights are the problem

The feed-forward block of a transformer is a two-layer MLP. With an input width of 1024 and a hidden width of 4096, its two weight matrices hold $1024 \cdot 4096 + 4096 \cdot 1024 = 8{,}388{,}608$ numbers. Large models use widths many times bigger and stack dozens of these blocks, so the weights alone can exceed the memory of one accelerator.

Splitting the batch does not help here, because every device would still need every weight. The fix is to split the weights so that each of the 8 devices stores one eighth of them, which is 1,048,576 numbers for the block above. The question is how to split them so that the devices still compute exactly the same output while sending each other as little data as possible.

## Splitting a matrix product by columns

Write a product as $y = x W$, where $x$ has shape (B, D) and $W$ has shape (D, H). Column $j$ of $y$ depends only on column $j$ of $W$, since $y_{b,j} = \sum_i x_{b,i} W_{i,j}$. So if device 0 holds the first half of the columns of $W$ and device 1 holds the second half, each device computes its own half of the columns of $y$ from the full $x$.

No device needs anything from any other device. The output simply ends up split by columns, the same way the weight was. This is called a column-parallel layer, and its cost is that every device needs the full input $x$.

## Splitting by rows needs a sum

Now split $W$ by rows instead, so device 0 holds rows 0 to $D/2 - 1$ and device 1 holds the rest, and split $x$ by columns to match. Each device can multiply the pieces it holds, but the sum over $i$ in the formula above runs across both halves. Device 0 gets $\sum_{i < D/2} x_{b,i} W_{i,j}$ and device 1 gets the rest of the sum.

Each device now holds a partial result with the full output shape, and the true output is the sum of the partial results. Adding them across devices is exactly what jax.lax.psum does. This is called a row-parallel layer, and its cost is one all-reduce of the output.

## Pairing the two splits

An MLP chains two products with a nonlinearity in between.

$$
h = \operatorname{relu}(x W_1 + b_1), \qquad \text{out} = h W_2 + b_2
$$

Here $W_1$ has shape (D, H), $b_1$ has shape (H,), $W_2$ has shape (H, O) and $b_2$ has shape (O,). Split $W_1$ by columns, so device $d$ holds a block of hidden units. The first product then leaves $h$ split by columns, and relu acts on each entry on its own, so applying it to each device's block gives exactly that block of the true $h$.

The second product needs $h$ split along its columns as input, which is the same as $W_2$ split along its rows, and that is exactly the layout the first layer produced. So split $W_2$ by rows with the same blocks of hidden units. Each device multiplies its block of $h$ by its rows of $W_2$ and gets a partial output, and a single psum turns the partial outputs into the true output. The whole MLP needs one collective, and the hidden activations never leave the device that computed them. This pairing is the layout of the Megatron-LM MLP (Shoeybi et al., 2019).

## A two-device example

Take $x = [1, -1]$, a hidden width of 4 and two devices, with

$$
W_1 = \begin{bmatrix} 1 & 0 & 2 & -1 \\ 0 & 1 & 1 & 1 \end{bmatrix}, \quad b_1 = [0, 0.5, -0.5, 0], \quad W_2 = \begin{bmatrix} 1 \\ 2 \\ -1 \\ 3 \end{bmatrix}, \quad b_2 = [0.5].
$$

The full computation gives the pre-activations $x W_1 + b_1 = [1, -0.5, 0.5, -2]$, the hidden values $h = [1, 0, 0.5, 0]$ and the output $h W_2 + b_2 = 1$.

Now split it. Device 0 holds hidden units 0 and 1, so it computes the hidden values [1, 0] and multiplies them by its rows of $W_2$, [1, 2], giving the partial output 1. Device 1 holds hidden units 2 and 3, computes the hidden values [0.5, 0] and multiplies them by its rows [-1, 3], giving -0.5. The psum gives $1 + (-0.5) = 0.5$, and adding $b_2$ gives 1, the same answer as before.

## Why the nonlinearity decides the split order

The order of the two splits is forced by relu. Suppose you split $W_1$ by rows instead, and leave out $b_1$ to keep the numbers short. Device 0 would hold row 0 of $W_1$ scaled by $x_0 = 1$, the partial pre-activation [1, 0, 2, -1], and device 1 would hold row 1 scaled by $x_1 = -1$, which is [0, -1, -1, -1]. Their sum is [1, -1, 1, -2], and relu of that sum is [1, 0, 1, 0].

If each device applied relu to its own partial result and the results were summed afterwards, the answer would be [1, 0, 2, 0], which is wrong in the third entry. Relu of a sum is not the sum of relus, so a row split before a nonlinearity forces a psum before the nonlinearity and another one after the second layer. The column-then-row order avoids the first collective entirely.

## Where the biases go

$b_1$ belongs to the hidden units, so it is split exactly like the columns of $W_1$, and each device adds its own slice. $b_2$ belongs to the output, which only exists in full after the psum. If each device adds $b_2$ to its partial output before the psum, the sum contains $b_2$ once per device. In the two-device example that would give $0.5 + 2 \cdot 0.5 = 1.5$ instead of 1, and with 8 devices the bias would be counted 8 times.

Adding $b_2$ after the psum, either inside the shard_map body or outside it, counts it once.

## What the devices store and send

With the split above, each device stores $D \cdot H/8 + H/8 \cdot O$ weight entries, one eighth of the total. The only communication is one all-reduce of the partial output, which has shape (B, O), the same size as the layer's output. That is small for a single MLP, and it is why tensor parallelism is usually kept inside groups of devices with very fast links, since it needs that all-reduce in every layer of every forward and backward pass.

Inside jax.shard_map the layout is spelled out in the specs. Column blocks of $W_1$ come from an in_specs entry that names the mesh axis on the second dimension, row blocks of $W_2$ name it on the first dimension, and the input is replicated. The hidden activations can leave through an out_specs entry that keeps them split, which matches how a real model would pass them straight to the backward pass without gathering them.

## Common mistakes to avoid

- **Adding the output bias before the psum.** Every device adds it to its partial output, so the combined output contains it once per device.
- **Splitting the second matrix by columns.** Each device would then need the full hidden vector, which only exists after gathering every device's block.
- **Forgetting the psum.** Each device's partial output covers only its own eighth of the hidden units, so without the sum every device reports a different and incomplete output.
- **Adding a relu to the output.** The output layer of this MLP is linear, so relu belongs only on each device's hidden block before the second product.
- **Mismatching the hidden blocks.** The columns of $W_1$ and the rows of $W_2$ on a device must cover the same hidden units, which happens automatically when both specs name the same mesh axis.
- **Gathering the hidden activations.** Replicating $h$ on every device costs memory and communication that the column split was designed to avoid.