# Pad and Mask to Bucket Length

Token sequences come in many lengths, while a function compiled with jax.jit works on arrays of fixed shape. Padding fills every sequence up to a shared length, and a mask records which slots hold real tokens. This page shows how a ragged batch is stored as one flat array, why the padded length should come from a short list of buckets, and how to build the padded batch inside jit with one cumulative sum and one gather.

## How a ragged batch is stored

A batch of B sequences with different lengths cannot be one rectangular array. Tokenizers and data loaders therefore keep the token ids of the whole batch back to back in a single flat array, together with a second array that holds the length of each sequence. This is often called a ragged layout. For three sequences [11, 12, 13], an empty sequence and [21, 22], the flat array is [11, 12, 13, 21, 22] and the lengths are [3, 0, 2].

The goal is a rectangular batch of shape (B, L), where L is the padded length. Row b holds sequence b starting at column 0, every later column holds a padding id called pad_id, and a boolean mask of the same shape is True exactly at the real tokens.

## A compiled function remembers shapes

When a jitted function is called, JAX traces it, which means it runs the Python body once with placeholder values that carry only a shape and a dtype, and records the operations as a small program called a jaxpr. That program is compiled and stored in a cache keyed by the shapes and dtypes of the traced arguments and by the values of any static arguments. A later call that matches a cache entry skips Python entirely.

A Python counter placed inside the body counts traces, because the body only runs while tracing. Four batches with lengths [3, 2], [1, 4], [5, 0] and [2, 3] were sent to the padding function used in this problem. Padding each batch to its own longest length gave the padded lengths 3, 4, 5 and 3, and the counter recorded three traces. Padding to the smallest fitting bucket from [4, 8] gave 4, 4, 8 and 4, and the counter recorded two traces, one per bucket.

The flat array's length is part of the shape as well. Two batches with totals of 5 and 6 tokens, both padded to length 4, still produced two traces. Every batch in the experiment above had 5 tokens, which is why the bucket alone decided the count. Loaders that want exactly one program per bucket also keep the flat buffer at a fixed size, for example by giving every batch the same token budget.

## Buckets trade padding for compilations

Padding every batch to the largest supported length, say 512, gives a single shape and wastes a lot of work. For three sequences with lengths 3, 5 and 11, the batch fills 1,536 slots, of which 1,517, about 98.8 percent, are padding. Padding to the batch's own longest length uses 33 slots with 14 padding slots, about 42.4 percent, but nearly every batch then has a new shape.

Buckets sit between the two. With buckets 8, 16 and 32 the same batch goes to bucket 16 and fills 48 slots, of which 29, about 60.4 percent, are padding. In a simulated stream of 1,000 batches whose longest length was drawn at random from 1 to 512, there were 434 distinct longest lengths, so padding to the longest would have compiled 434 programs. With buckets 64, 128, 256 and 512 the same stream needed 4.

## Choosing the bucket outside jit

The bucket length decides the output shape, so it has to be a plain Python int before tracing starts. Inside a jitted function the lengths are placeholders, and calling int(jnp.max(lengths)) there raises a ConcretizationTypeError whose message begins "Abstract tracer value encountered where concrete value is expected". The same call outside jit works, because lengths is then an ordinary JAX array with real values. It waits for the device to finish computing the maximum and copies that single number to the host, which is cheap compared with copying the whole token array.

With the longest length in hand, the bucket is the smallest value in buckets that is at least that length. A sequence whose length equals a bucket fits it exactly. Bucket lists in configuration files are not always sorted, so the choice has to look at every bucket instead of stopping at the first one that fits.

The chosen length is then passed to the jitted function as a static argument, named in static_argnames or numbered in static_argnums. A static argument reaches the Python body as its real value, so jnp.arange(L) produces an array of known size, and the value becomes part of the cache key. Each bucket therefore compiles one program, and every later batch in that bucket with the same array shapes reuses it.

## Start offsets from a cumulative sum

Sequence b begins in the flat array right after all earlier sequences end. Writing $\ell_c$ for the length of sequence c, its start offset is

$$
s_b = \sum_{c=0}^{b-1} \ell_c ,
$$

which is 0 for the first row. jnp.cumsum returns the inclusive running totals $\ell_0, \ell_0 + \ell_1, \dots$, and subtracting each row's own length turns them into the exclusive totals above. For the lengths [3, 0, 2] the running totals are [3, 3, 5] and the start offsets are [0, 3, 3]. The empty second row shares its start with the third row, which is harmless because none of its slots will be read as real tokens.

## Gathering every slot at once

Slot j of row b, for j from 0 to L - 1, reads the flat index $s_b + j$. Broadcasting the column of start offsets against the row of positions from jnp.arange(L) builds all of these indices as one (B, L) integer array. With L equal to 4 for the example batch, the index grid is [[0, 1, 2, 3], [3, 4, 5, 6], [3, 4, 5, 6]].

Indices 5 and 6 point past the end of the five-token array. JAX does not raise an error for an out-of-range gather index, and in a quick check indexing the five-token array at position 9 returned the value at position 4, the last one. Clipping the grid to the range 0 to 4 states that behaviour explicitly and gives [[0, 1, 2, 3], [3, 4, 4, 4], [3, 4, 4, 4]]. Indexing the flat array with this grid is a single gather operation, and it returns [[11, 12, 13, 21], [21, 22, 22, 22], [21, 22, 22, 22]]. The padded slots now hold tokens borrowed from other rows, and the next step replaces them.

## Writing pad_id and building the mask

Position j of row b is real exactly when $j < \ell_b$. Comparing the row of positions with the column of lengths gives the mask [[True, True, True, False], [False, False, False, False], [True, True, False, False]]. jnp.where then keeps the gathered value where the mask is True and writes pad_id everywhere else, so with pad_id equal to 0 the padded tokens are [[11, 12, 13, 0], [0, 0, 0, 0], [21, 22, 0, 0]].

The mask has to come from the lengths. Recovering it by comparing the padded tokens with pad_id fails whenever a real token shares that id. Padding with 0 collides with any vocabulary in which token 0 is real, and GPT-2 has no dedicated padding token, so a common practice pads with its end-of-text id 50256, which also appears inside real text.

## What the compiled program contains

The jaxpr of the padding function for two rows and L equal to 16 contains one iota of length 16 from jnp.arange and exactly one gather, whose output has shape int32[2, 16]. The size of that program does not depend on the batch size. Measured on batches of 8 and of 64 rows, it had 24 equations and a single gather both times.

A version that loops over the rows in Python inside jit produces the same numbers through a different program. The loop runs during tracing, so every row adds its own copy of the operations. For 8 rows its jaxpr had 165 equations with 8 gathers, and for 64 rows it had 1,285 equations with 64 gathers, so tracing and compilation slow down as the batch grows. A loop outside jit is worse again, since every small operation is dispatched to the device on its own, and copying the tokens to NumPy moves the work off the device entirely.

The rest of the model uses the mask in the same fixed shape. Attention layers give masked positions a very large negative score before the softmax, and losses average only over real positions with a masked mean, so neither needs the original lengths.

## Common mistakes to avoid

- Padding each batch to its own longest length makes every new longest length a new shape and a fresh compilation.
- Passing the longest length or a tuple of the lengths as a static argument compiles again for every new batch, even when the bucket stays the same.
- Leaving the bucket length traced makes jnp.arange fail with a ConcretizationTypeError, because a shape needs a Python int.
- Defining the jitted function inside the outer function creates a new function object on every call, so JAX traces and compiles it again each time.
- Using the inclusive running totals from jnp.cumsum as the start offsets shifts every row forward by one whole sequence.
- Taking the first bucket that fits in list order can pick a much larger bucket than needed when the list is unsorted.
- Building the mask by comparing tokens with pad_id marks real tokens that share that id as padding.
- Looping over rows in Python or copying tokens to NumPy replaces the single compiled gather with work that grows with the batch size.