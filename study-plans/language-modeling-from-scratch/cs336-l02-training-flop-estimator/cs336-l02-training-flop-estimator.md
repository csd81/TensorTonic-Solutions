# Transformer Training FLOP Estimator

Planning a training run starts with an estimate of how much computation it needs. Almost all of a transformer's work is matrix multiplication, and each multiplication has a cost that follows directly from its shape. The heart of this problem is counting each multiply and each add in a matrix multiplication, and understanding why the backward pass costs about twice the forward pass.

## Counting one matrix multiplication

Multiplying an input of shape $(B, D)$ by a weight matrix of shape $(D, K)$ produces an output of shape $(B, K)$. Each of the $BK$ output entries is a dot product of length $D$, which takes $D$ multiplications and roughly $D$ additions. Counting each multiplication and each addition as one floating-point operation (FLOP) gives

$$
F_{\text{matmul}} = 2BDK
$$

Strictly a dot product of length $D$ needs only $D - 1$ additions, but the factor of 2 is the standard convention. It matches hardware that performs $D$ fused multiply-add operations per dot product, each counted as two FLOPs.

In a transformer, $B$ is usually the number of tokens in the batch, $D$ is the input width of a layer and $K$ is its output width. A zero in any dimension means no work.

## The forward pass

A forward pass runs many matrix multiplications: the query, key, value and output projections in attention, and the up and down projections in the MLP. Their costs add up.

Attention also multiplies activations by activations, first queries by keys and then attention weights by values. These products involve no parameters and grow with the square of the sequence length, so they do not fit the weight-matrix pattern above. This problem passes their cost in as a separate term:

$$
F_{\text{forward}} = \sum_m 2 B_m D_m K_m + F_{\text{attention}}
$$

Here $m$ indexes the listed multiplications and $B_m$, $D_m$ and $K_m$ are the dimensions of multiplication $m$.

## Why backward costs twice the forward

For each weight multiplication $Y = XW$, let $\nabla_Y$, of shape $(B, K)$, be the gradient of the loss with respect to the output $Y$. The backward pass needs two gradients from it. The gradient with respect to the input, which is passed on to earlier layers, is $\nabla_X = \nabla_Y W^{\mathsf T}$. The gradient with respect to the weights, which the optimizer uses, is $\nabla_W = X^{\mathsf T} \nabla_Y$.

Each of these is a matrix multiplication with the same three dimensions $B$, $D$ and $K$, just arranged differently, so each costs $2BDK$. Two of them cost $4BDK$, twice the forward cost. Applying the same ratio to the whole forward pass gives the standard estimate:

$$
F_{\text{backward}} = 2 F_{\text{forward}}, \qquad F_{\text{total}} = F_{\text{forward}} + F_{\text{backward}} = 3 F_{\text{forward}}
$$

The ratio is an estimate. The very first layer, for example, needs no gradient with respect to its input, but such differences are small for a deep model.

## Where the 6NT rule comes from

A weight matrix of shape $(D, K)$ holds $DK$ parameters, so its forward cost $2BDK$ is two FLOPs per parameter per token. If a model has $N$ parameters and nearly all of them sit in weight matrices, each token costs about $2N$ FLOPs forward and $6N$ FLOPs for forward plus backward. Training on $T$ tokens therefore costs roughly $6NT$ FLOPs.

This rule is how compute budgets are usually quoted. A 1 billion parameter model trained on 20 billion tokens needs about $6 \times 10^9 \times 2 \times 10^{10} = 1.2 \times 10^{20}$ FLOPs. The estimator in this problem is the same idea computed layer by layer.

## A worked example

Take two multiplications, $[4, 8, 2]$ and $[4, 2, 8]$, with an attention cost of 64 FLOPs. Both multiplications cost the same:

$$
2 \times 4 \times 8 \times 2 = 128, \qquad 2 \times 4 \times 2 \times 8 = 128
$$

The forward pass costs $128 + 128 + 64 = 320$ FLOPs. The backward pass costs $2 \times 320 = 640$, and the full training step costs $320 + 640 = 960$.

The attention cost is doubled along with everything else, because it is added before the backward estimate is taken.

## FLOP counts outgrow fixed-width integers

Training runs reach $10^{20}$ FLOPs and beyond, past the limit of about $9.2 \times 10^{18}$ for signed 64-bit integers. Python integers have no fixed limit, so multiplying them directly gives exact counts. Converting to floating point would round the large values.

## Common mistakes to avoid

- **Forgetting the factor of 2.** A multiplication costs $2BDK$ because each multiply comes with an add.
- **Adding attention after the backward estimate.** Attention is part of the forward cost, so it is doubled too.
- **Counting only the weight gradient in backward.** That makes backward equal to forward and gives a total of twice forward instead of three times.
- **Treating a zero dimension as an error.** A zero-sized multiplication costs 0 FLOPs and should add nothing to the total.
- **Using floats for large counts.** Exact Python integers avoid rounding errors.
