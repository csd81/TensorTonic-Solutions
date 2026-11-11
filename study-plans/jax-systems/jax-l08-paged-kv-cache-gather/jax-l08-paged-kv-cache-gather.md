# Paged KV Cache Gather

When a transformer generates text, every new token attends to the keys and values of all the tokens before it, so a serving system keeps those keys and values in memory as a KV cache. Storing each request's cache as one long contiguous buffer wastes most of the memory, and paged attention (Kwon et al., 2023) fixes this by cutting the cache into small fixed-size pages, much like an operating system manages memory. This page explains why pages help, how a block table finds a token's page, and how to rebuild every sequence's cache from the shared pool with fixed-shape JAX operations, which is what you implement here.

## What the KV cache holds

In attention, each token produces a query, a key and a value vector for every head. To generate token $t$, the model compares the new query with the keys of tokens $0$ to $t - 1$ and averages their values. Those old keys and values never change, so recomputing them at every step would waste work, and the server stores them instead.

The cache is large. For a model with 32 layers, 32 heads and a head dimension of 128, stored in bfloat16 (2 bytes per number), one token needs $2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524{,}288$ bytes, half a mebibyte, where the leading 2 counts the key and the value. A single 4,096-token conversation then needs 2 GiB of cache, so how this memory is laid out decides how many requests fit on one accelerator.

## Why one buffer per request wastes memory

The simplest layout reserves one contiguous buffer per request, sized for the longest output it might produce. The server cannot know in advance when a request will stop, so it has to reserve the maximum. If every request reserves 2,048 token slots but the average request only uses 300, about 85 percent of the reserved memory, $(2048 - 300) / 2048 \approx 0.854$, sits empty.

That empty memory cannot be lent to another request, because each buffer must stay contiguous. The server ends up running far fewer requests at once than the hardware could hold, which lowers throughput.

## Pages and block tables

Paged attention splits every sequence's cache into blocks of $S$ consecutive tokens, where $S$ is the page size. All requests share one big pool of physical pages, and each page holds exactly one block. A request only takes a new page when its last page fills up, so the waste per sequence is at most $S - 1$ slots. With pages of 16 tokens, a sequence wastes 7.5 slots on average if its length is spread evenly over the possible remainders, which is about 2.5 percent of a 300-token sequence.

Because the pages of one sequence can sit anywhere in the pool, each sequence needs a lookup table that says which page holds its block 0, which holds its block 1, and so on. This is the block table, and it plays the same role as a page table in an operating system. Row $b$ of the table belongs to sequence $b$, and entry $i$ of that row is the page index of block $i$.

Token position $j$ of a sequence lives in block $\lfloor j / S \rfloor$, at slot $j \bmod S$ inside that block's page. The floor brackets mean rounding down, and $j \bmod S$ is the remainder after dividing by $S$. Positions 0 to $S - 1$ are block 0, positions $S$ to $2S - 1$ are block 1, and the remainder tells you where inside the block the token sits.

## A small pool

Take a pool of 4 pages with page size $S = 3$, and give the slots simple scalar keys so you can read them off directly. Page $p$ at slot $s$ holds the key $10p + s$, so page 1 holds 10, 11 and 12, and page 3 holds 30, 31 and 32.

Sequence A has 4 tokens and the block table row $[1, 3]$. Sequence B has 2 tokens and the row $[0, -1]$, where -1 marks a block the sequence does not use yet. The table has 2 columns, so the full padded length is $L = 2 \cdot 3 = 6$ positions.

For positions 0 to 5 the blocks are $[0, 0, 0, 1, 1, 1]$ and the slots are $[0, 1, 2, 0, 1, 2]$. Looking the blocks up in row A gives pages $[1, 1, 1, 3, 3, 3]$, so the raw gathered keys are $[10, 11, 12, 30, 31, 32]$. Sequence A has only 4 tokens, so positions 4 and 5 must become zero, which leaves $[10, 11, 12, 30, 0, 0]$. The values 31 and 32 are leftovers in page 3 that this sequence has not written yet.

For sequence B the pages are $[0, 0, 0, -1, -1, -1]$. Its 2 tokens give $[0, 1, 0, 0, 0, 0]$ after masking. Positions 0 and 1 copy slots 0 and 1 of page 0, and everything from position 2 on is zero.

## Fixed shapes under jit

jax.jit compiles a function for specific input shapes, and every intermediate and output shape must follow from those shapes alone. The lengths in seq_lens are data, so their values are unknown while JAX traces the function. Slicing a sequence down to its true length, as in taking the first seq_lens[b] positions, would give a shape that depends on a value, and JAX refuses it with an IndexError saying slice entries must be static integers.

The standard fix is to pad every sequence to the same length $L$, the width of the block table times the page size, and to carry a boolean mask that marks which positions are real. The output shape then depends only on the shapes of the inputs, and the lengths enter only through the mask. Attention later uses the same mask to give padded positions zero weight.

## Gathering every position at once

A Python loop over positions would trace one gather per position, and a loop over the lengths would fail outright because the lengths are traced. Integer array indexing does the whole gather in one operation. When you index an array with several integer arrays, JAX broadcasts the index arrays against each other, and each element of the broadcast result picks one entry of the indexed axes. The remaining trailing axes, here the heads and the head dimension, come along unchanged.

Take a 3 by 4 array A whose entry at row $r$ and column $c$ is $10r + c$. Indexing A with the row array [[0], [2]], of shape (2, 1), and the column array [1, 3], of shape (2,), broadcasts both to shape (2, 2) and returns [[1, 3], [21, 23]]. Each output entry pairs one row index with one column index from the broadcast arrays. If A had a third axis of size 2, the same indexing would return shape (2, 2, 2), because each picked entry is then a whole vector along the axis that was not indexed.

## Stale slots and padding entries

Two kinds of positions must not reach the output. The first is a slot past the end of a sequence inside its last page. The page belongs to the sequence, but the slot still holds data from whatever request used the page before, like the 31 and 32 in the example.

The second is a padding entry of -1 in the block table. JAX indexing never raises an out-of-bounds error inside a compiled program. A negative index wraps around from the end as in NumPy, so -1 picks the last page of the pool, and an index past the end is clamped to the last valid entry. In the example, sequence B's padding entries would copy 30, 31 and 32 from page 3. jnp.take with its default mode instead fills out-of-range positions with NaN for floats.

Either way the gathered array can hold garbage in those positions, so the result has to set them to zero. Multiplying by a 0 or 1 mask gives the right zeros when the stale data is finite, which is the case in this problem's tests. It is unsafe in a real server, because a stale page can hold NaN or infinity, and both NaN times 0 and infinity times 0 give NaN. Selecting with jnp.where takes the zero branch outright, so whatever was gathered there disappears.

## Keeping the dtype

Serving systems often store the cache in bfloat16 to halve its size. JAX follows type promotion rules when it mixes dtypes, and a bfloat16 array multiplied by a float32 mask comes out as float32, which silently doubles the size of the result. A Python scalar zero is weakly typed and adopts the dtype of the other operand, so jnp.where of a bfloat16 array and the scalar 0.0 stays bfloat16. Creating the zero with the pool's own dtype gives the same guarantee.

## Where the gather sits in serving

Production paged attention kernels never build the padded per-sequence cache in memory. They read the pages directly inside the attention kernel, using the block table to find each page as they go. Writing the gather explicitly, as here, produces exactly the keys and values those kernels see, and it is how reference implementations and tests of paged attention check that the block table bookkeeping is right.

## Common mistakes to avoid

- **Skipping the mask.** Slots past seq_lens inside the last page hold old data, so copying whole pages leaks another request's keys into the result.
- **Trusting the padding entries.** A -1 in the block table silently indexes the last page of the pool, so its positions must be zeroed like every other position past the end.
- **Relying on a mask multiply for safety.** A multiply zeroes finite stale data, but NaN or infinite values in a stale page survive it, and a float32 mask also promotes bfloat16 data to float32.
- **Swapping block and slot.** The block is the position divided by the page size and the slot is the remainder, and reversing them reads the wrong page.
- **Slicing to each sequence's length.** Shapes that depend on seq_lens cannot be traced under jit, so every sequence must keep the full padded length.
- **Getting the mask boundary wrong.** Position $j$ is real only when $j$ is strictly less than the sequence length, so a less-than-or-equal test marks one position too many.