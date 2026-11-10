# Functional Array Update

In NumPy you change an array by assigning into it, for example by setting one entry or by adding to entries inside a loop. JAX arrays can never be changed after they are created, so every update produces a new array instead. This page explains why JAX makes that choice, how its indexed update syntax gives you the same power without mutation, and how it handles the case at the heart of this problem, which is many updates landing on the same position.

## What goes wrong with in-place assignment

In NumPy, the statement x[1] = 10 overwrites the second entry of x, and every other piece of code holding x sees the new value. In JAX the same statement raises a TypeError that says JAX arrays are immutable. An immutable array is one whose contents are fixed from the moment it is created.

The reason comes from how JAX runs code. Its transformations, such as jax.jit, which compiles a function, and jax.grad, which differentiates one, work by tracing. Tracing means that JAX calls your Python function once with placeholder arrays called tracers, records every operation applied to them, and turns that record into a program. This only works cleanly for pure functions, which are functions whose output depends only on their inputs and which change nothing outside themselves. If an array could be overwritten after it was read, the recorded operations would no longer say which version of the data each step used, and both compilation and differentiation would have to guess.

## Asking for a changed copy

JAX replaces assignment with the indexed update syntax. The expression x.at[1].set(10.0) describes the array you would get by setting position 1 of x to 10, and it returns that array as a new value. With x equal to [1, 2, 3], the result is [1, 10, 3], and x is still [1, 2, 3] afterwards.

Indices follow the NumPy rules. A negative index counts from the end, so x.at[-1].set(7.0) gives [1, 2, 7]. Slices and integer arrays work as indices too. Besides set, the .at property offers add, mul, min and max, which combine the new value with the old one, and get, which reads values.

## Why the copy is cheap

Making a new array for every update sounds wasteful. Outside jit, each update does allocate a fresh buffer. Inside jit, the program is handed to XLA, the compiler that JAX uses to turn traced programs into machine code. XLA sees the whole program at once, and when the old array is never read again after the update, it writes the new values into the old memory. You write pure code, and the compiled code still updates in place wherever that is safe.

## Counting items into bins

A histogram has K bins, and each bin counts how many items fell into it. The data arrives as a list of bin numbers, one per item. Start from three bins holding zeros and the bin numbers [0, 2, 2, 2]. Bin 0 receives one item and bin 2 receives three, so the histogram is [1, 0, 3].

When the index is an integer array, one indexed update applies every item at once. Adding 1.0 at the indices [0, 2, 2, 2] to three zeros returns [1, 0, 3] directly. Printing the traced program with jax.make_jaxpr shows one scatter-add operation, next to a few small steps that handle negative indices. A scatter takes a list of positions and a list of values and sends each value to its position, and the add variant sums the values that arrive at the same place.

## Why set loses repeated updates

The same update with set gives a different answer. Setting 1.0 at the indices [0, 2, 2, 2] in three zeros returns [1, 0, 1]. The method set writes a value into a position, and later writes to the same position replace earlier ones, so bin 2 ends up holding 1 instead of 3. When the colliding values differ, JAX does not even promise which one survives.

A tempting repair is to read the current values, add the new ones, and write the sums back with set. That reads every bin once before any write lands, so two updates aimed at the same bin both start from the same old value and one of them is lost. The combining methods add, mul, min and max avoid this, because they define how colliding updates merge. Taking the max of the values [0.5, 1.0, 2.0, 0.25] at the indices [0, 2, 2, 2] in three zeros, for example, returns [0.5, 0, 2].

## Weighted updates on top of existing counts

A weighted histogram lets each item add its own amount instead of 1. Start from three bins that already hold [1, 1, 1], and apply the weights [0.5, 1.0, 2.0, 0.25] at the indices [0, 2, 2, 2]. Bin 0 becomes $1 + 0.5 = 1.5$, bin 1 stays at 1, and bin 2 becomes $1 + 1.0 + 2.0 + 0.25 = 4.25$. The result is [1.5, 1, 4.25].

Starting from existing counts matters whenever statistics are gathered across many batches. A training loop might keep a running count of how often each token appears, or how often a mixture of experts router sends tokens to each expert. Each batch adds its contribution to the totals carried over from the earlier batches, so throwing the starting values away would reset the statistics every step.

The function jnp.bincount computes the same kind of weighted count from zeros. Called on the indices [0, 2, 2, 2] with the weights above and a length of 3, it returns [0.5, 0, 3.25]. Under jit its length argument has to be given explicitly, because the output shape must be known before any index values exist.

## Indices outside the array

NumPy raises an IndexError for an index past the end of an array. Compiled JAX code cannot raise an error that depends on data values, so it follows fixed rules instead. By default an update at an index past the end is dropped and a read is clamped to the nearest valid position. With x equal to [1, 2, 3], x.at[5].set(9.0) returns [1, 2, 3] unchanged, and x.at[5].get() returns 3. The mode argument of the update methods selects other behaviours, such as clipping every index into range.

Negative indices follow a separate rule. Before the update is applied, JAX counts a negative index from the end of the array, exactly as NumPy does, so -1 names the last position and x.at[-1].add(5.0) returns [1, 2, 8]. Those small steps in the printed program add the length to every negative index, and they run whatever mode is chosen, so dropping out-of-range updates does not stop -1 from reaching the last bin. Only an index below minus the length stays out of range after the wrap, which is why x.at[-4].add(5.0) leaves x unchanged.

Data pipelines often mark a discarded item with the index -1. When such an item is passed straight into an indexed update, it is quietly counted in the last bin, and a histogram built with one bin too few loses its last bin's updates without any error message. Code that must ignore such items has to make sure they cannot land on any bin before the scatter runs.

## The order of floating-point additions

Floating-point addition is not associative, which means the grouping of a sum can change its result. In float32, computing $(10^8 + 1) - 10^8$ gives 0, because $10^8 + 1$ rounds back to $10^8$, while $(10^8 - 10^8) + 1$ gives 1. A scatter-add running on parallel hardware may combine colliding updates in any order, so float results can differ in their last digits between devices or runs. Integer counts are always exact.

## A loop or one scatter

A Python loop that applies one update per item also gives the right answer under jit, because the loop runs while the function is being traced. It records N separate updates, so the compiled program, and the time spent compiling it, grows with the number of items. A single indexed update with an index array records one scatter-add whatever N is.

## Common mistakes to avoid

- **Counting with set.** Repeated indices overwrite each other, so a bin hit three times ends up holding a single update.
- **Reading old values before writing sums back.** Every update to a shared bin starts from the same old value, so all but one of them is lost.
- **Starting from zeros.** Ignoring counts drops the totals carried over from earlier batches.
- **Assigning into the array.** Item assignment raises a TypeError because JAX arrays are immutable.
- **Converting to NumPy for the update.** A tracer has a shape and a dtype but no values while the function compiles, so converting it to a NumPy array fails under jit.
- **Passing negative indices straight to the update.** A negative index counts from the end of the array, so an item marked with -1 lands in the last bin instead of being ignored.
- **Trusting out-of-range indices to fail loudly.** Updates past the end of the array are dropped silently, so an off-by-one in bin numbering hides instead of crashing.