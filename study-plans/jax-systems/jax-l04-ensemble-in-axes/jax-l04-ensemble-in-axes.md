# Model Ensemble with in_axes

An ensemble is a group of models trained separately on the same task whose predictions are combined, usually by averaging. An ensemble is usually more accurate than its members, and the spread of the members' predictions is a practical measure of how uncertain the group is about an input (Lakshminarayanan et al., 2017). Running the members one after another is slow when they share an architecture. This page shows how to store all members as one set of stacked arrays and evaluate them in one call with jax.vmap, using its in_axes argument to say which inputs belong to each member and which inputs every member shares.

## Stacking the members into one set of weights

Every member here is the same two-layer MLP, a multilayer perceptron that applies a linear map, the ReLU function $\operatorname{relu}(z) = \max(z, 0)$ and a second linear map. For member $m$ and an input row $x$ of width D,

$$
f_m(x) = \operatorname{relu}\left(x W_1^{(m)} + b_1^{(m)}\right) W_2^{(m)} + b_2^{(m)} ,
$$

where $W_1^{(m)}$ has shape (D, K), $b_1^{(m)}$ has shape (K), $W_2^{(m)}$ has shape (K, O) and $b_2^{(m)}$ has shape (O). Here K is the hidden width and O is the output width.

Because every member has arrays of the same shapes, the M members can be stored together by stacking each kind of array along a new leading axis. The first-layer weights of all members then form one array of shape (M, D, K), the first-layer biases one array of shape (M, K), and so on. These four arrays live in a dict. JAX calls any nested structure of dicts, lists and tuples with arrays at the leaves a pytree, and its transformations accept pytrees wherever they accept arrays.

## The single-member function

The cleanest way to write the ensemble is to write one member first. That function receives a dict whose arrays have no member axis, together with the input batch x of shape (N, D), and returns the member's predictions of shape (N, O). Nothing in it mentions M, so it reads like ordinary model code and could run any single trained model.

What remains is calling it for every member at once. The stacked dict has a member axis in every leaf, while x has no member axis at all, because all members see the same inputs. The two arguments therefore need different treatment.

## What in_axes says

The in_axes argument of jax.vmap gives one entry per positional argument of the function. An integer entry names the axis of that argument to map over, and the entry None says that the argument is passed whole to every call. When an argument is a pytree, an integer entry applies to every leaf, and a pytree of entries with the same structure can set the axis leaf by leaf.

A tiny function shows the effect. Let it multiply a scalar weight w by a vector x, and map it with in_axes equal to (0, None) over the weights (1, 2, 3) and the vector (10, 20). The weights are sliced into three scalars and the vector is shared, so the three calls return (10, 20), (20, 40) and (30, 60), stacked into an array of shape (3, 2).

Mapping both arguments with in_axes equal to (0, 0) means something else entirely. It pairs the first weight with the first entry of x, the second weight with the second entry, and so on. For these inputs the sizes do not agree, so JAX raises an error saying that vmap got inconsistent sizes for the array axes to be mapped, one of size 3 and one of size 2. If the sizes had matched, the call would have succeeded and silently computed a pairing, which is the wrong result for an ensemble.

At least one entry must be mapped. With in_axes equal to (None, None) there is no axis to map over, and JAX reports that vmap must have at least one non-None value in in_axes.

## Why one batched call beats a loop

A Python loop over the members also produces the right numbers, but under tracing it unrolls. JAX records every operation of every iteration, so the traced program grows with M. For an ensemble of 8 members, the loop version of this MLP records 121 operations, while the vmapped version records 8, the same count it records for any M.

The vmapped program is also faster to run. Each layer becomes a single batched matrix product whose batch dimension is the member axis, so the hardware processes all members together instead of launching many tiny products. In the traced program, the first layer is one product of x with the stacked weights, and x is never copied once per member.

## A two-member example

Take two members with D = K = O = 1 and the shared input x with rows 1 and 2. Member 0 has $W_1 = 1$, $b_1 = 0$, $W_2 = 2$ and $b_2 = 0$. Member 1 has $W_1 = -1$, $b_1 = 3$, $W_2 = 1$ and $b_2 = 1$.

For member 0 the hidden values are relu(1) = 1 and relu(2) = 2, so the outputs are 2 and 4. For member 1 the hidden values are relu(-1 + 3) = 2 and relu(-2 + 3) = 1, so the outputs are 3 and 2. Stacked along the member axis, the outputs have shape (2, 2, 1). The ensemble mean over the member axis is 2.5 for the first input and 3 for the second.

The members disagree more about the second input, with predictions 4 and 2, than about the first, with predictions 2 and 3. The variance across members is 0.25 for the first input and 1 for the second, so the ensemble would report more uncertainty for the second input.

## Reducing over the right axis

After the vmapped call, axis 0 of the output is the member axis, because vmap places the mapped axis at position 0 of its output by default. The ensemble mean is therefore a mean over axis 0, which turns (M, N, O) into (N, O), one averaged prediction per input row. Averaging over axis 1 would average each member over its inputs and return an array of shape (M, O), which answers a different question.

The out_axes argument of jax.vmap can move the mapped axis elsewhere in the output. With out_axes equal to 1, the scalar example from earlier returns shape (2, 3) in place of (3, 2). Leaving it at the default keeps the member axis first, which is the usual layout for stacked models.

## Where this pattern appears

The same pattern runs many copies of a model with different parameters on shared data. Hyperparameter sweeps evaluate many initializations or learning rates at once, population-based training evaluates a whole population of networks, and evaluation code compares several checkpoints on one batch. In every case the parameters carry a leading copy axis mapped with 0, and the data is shared with None.

## Common mistakes to avoid

- **Mapping the inputs along with the weights.** With in_axes equal to (0, 0), member m sees only row m of x, which fails when N differs from M and computes the wrong pairing when they match.
- **Hiding the shared input in a closure.** Reading x from the enclosing scope runs, but None in in_axes states the sharing in the call and keeps the single-member function reusable with any data.
- **Averaging over the wrong axis.** The member axis is axis 0 of the vmapped output, and a mean over axis 1 averages each member's predictions across inputs.
- **Slicing members by hand.** A Python loop over the member axis unrolls into a traced program that grows with M and runs many small products.
- **Dropping the ReLU or a bias.** The ensemble repeats whatever one member computes, so a single-member function that skips part of the formula is wrong for every member.
- **Changing the params dict.** Popping or replacing entries in the dict that was passed in changes the caller's model, so new values belong in new variables.