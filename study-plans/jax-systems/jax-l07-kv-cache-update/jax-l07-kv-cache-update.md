# KV Cache Update at a Position

A language model generates text one token at a time, and every new token attends to the keys and values of all the tokens before it. Recomputing those old keys and values at every step would waste most of the work, so inference code stores them in a key-value cache, usually called the KV cache. This page starts from why the cache exists, explains why JAX code keeps it as a fixed-size buffer, and then shows how lax.dynamic_update_slice writes new entries into that buffer at a position that is only known while the program runs. That write is what you implement in this problem.

## Why decoding keeps a cache

In attention, each token produces a query, a key and a value by multiplying its hidden state with three weight matrices. A new token's query is compared with the keys of every earlier token, and the resulting weights mix their values. The keys and values of earlier tokens never change once they are computed, because in a decoder a token's hidden state depends only on itself and the tokens before it.

Without a cache, generating the fourth token means recomputing the keys and values for tokens 1, 2, 3 and 4, and over four steps that is $1 + 2 + 3 + 4 = 10$ key computations for each layer. With a cache, each step computes the key and value for its own token only and appends them, so four steps cost 4 key computations. The gap grows quadratically with the sequence length, which is why every serious inference engine caches keys and values.

## Why the cache has a fixed size in JAX

The natural way to append in NumPy would be to concatenate a new row onto the end of the cache, so the cache grows by one slot per token. That works badly in JAX. A function compiled with jax.jit is compiled for one exact set of input shapes. Call it again with a cache that is one slot longer and JAX traces and compiles a brand new program. Generating 1000 tokens would then mean 1000 compilations, each far slower than the step itself.

The fix is to allocate the cache once at its maximum size and never change its shape. A cache for B sequences, S slots, H heads and head size D has shape (B, S, H, D) for the whole of generation. Only the first few slots hold real tokens at any moment, and a separate counter, the length, says how many. Every decoding step then has identical shapes, so the step is compiled once and reused for every token.

## Why a Python slice fails on a traced position

When jax.jit traces a function, each array argument becomes a tracer, a stand-in that records the operations applied to it and knows only its shape and dtype. The write position is an int32 scalar array, so inside the traced function it is a tracer with no concrete value.

A single traced index works. Reading cache[:, position] becomes a dynamic_slice operation in the traced program, and cache.at[:, position].set(x) becomes a scatter, the general operation that writes values at computed indices. A slice whose bounds are traced is different. The slice from position to position + L needs a size that is known at trace time, and JAX rejects any slice whose start or stop is a tracer, even though L itself is fixed. Calling int on the position fails for a similar reason, since there is no number to convert yet. Making the position static (a Python int baked into the program) would avoid the error, but then every new position would trigger a recompile, which is exactly the problem the fixed-size cache was meant to solve.

## Writing a block with dynamic_update_slice

lax.dynamic_update_slice is the operation designed for this. It takes three things.

- **The operand.** This is the array to write into, here a cache of shape (B, S, H, D).
- **The update.** This is a smaller block with the same number of axes, here the new vectors of shape (B, L, H, D) for L new tokens.
- **The start indices.** There is one start index per axis, and any of them may be a traced scalar.

It returns a new array with the update copied in, starting at those indices, and every other element unchanged. Only the start indices may be traced. The size of the block comes from the update's shape, which is fixed at trace time, so the compiled program knows exactly how much data moves. A block write at a traced offset is exactly what dynamic_update_slice expresses, as one contiguous copy. A scatter with a computed list of indices can produce the same numbers, but it is a more general and heavier operation built for arbitrary index lists.

JAX arrays are immutable, which means no operation changes an array in place. The call therefore returns a new cache, and the old one is still valid afterwards. That sounds expensive for a large cache, but XLA can reuse the input buffer for the output when the caller donates it to the compiled function, so in a real decoding loop the write usually costs only the size of the new block.

## A small write, step by step

Take a one-dimensional cache with five slots holding $(0.1, 0.2, 0.3, 0, 0)$, where three tokens have been stored and the length is 3. Two new tokens arrive with values 0.4 and 0.5, and the write starts at slot 3. The result is $(0.1, 0.2, 0.3, 0.4, 0.5)$, and the length becomes $3 + 2 = 5$.

For a real cache the same thing happens along one axis while the others are written in full. The start index is 0 for the batch, head and feature axes, and the position for the sequence axis, so every batch row and every head receives its new vectors in the same slots.

## What happens when the block does not fit

dynamic_update_slice never writes outside the operand. If the start index would push the block past the end, it quietly moves the start back until the block fits. Writing the block $(8, 9)$ into $(1, 2, 3, 4)$ with a requested start of 3 gives $(1, 2, 8, 9)$, because a block of two cannot start at slot 3 of a four-slot array, so the start is clamped to 2.

This clamping means an overflowing cache does not crash. It silently overwrites the last slots instead. Real inference engines prevent this by stopping generation, or by evicting old entries, before the length reaches S. In this problem the position is always small enough for the block to fit.

## The length counter

The cache alone does not say how many slots are real, because unused slots look like any other numbers. The length counter carries that information, and the attention step uses it to mask out the slots that are not filled yet. After writing L tokens starting at position, the new length is position + L. Computing it from the traced position keeps it a traced int32 scalar, so the next decoding step can feed it straight back in without leaving the compiled program.

## Cache dtype and memory

The KV cache is often the largest piece of memory during inference. A model with 32 layers, 8 key-value heads and head size 128 stores, for every token, a key and a value in every layer, so in bfloat16 (a 16-bit floating-point format with 2 bytes per number) one token costs $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131072$ bytes, which is 128 KiB. A single sequence of 4096 tokens then needs 512 MiB. Storing the cache in bfloat16 instead of float32 halves this, so many systems do it.

The new keys and values usually come out of a float32 computation, while the cache is bfloat16. dynamic_update_slice requires the update and the operand to have the same dtype, and it raises an error when they differ. The update therefore has to be converted to the cache dtype before the write. The conversion rounds each number to the nearest bfloat16 value, so 0.1234 is stored as 0.12353515625, and that small rounding is the accepted price of the memory saving.

## Common mistakes to avoid

- **Slicing with the traced position.** Python slice bounds must be concrete integers under jax.jit, so a slice that starts at the position fails during tracing.
- **Growing the cache by concatenation.** Each new shape forces a fresh compilation, and the returned cache must keep its shape anyway.
- **Writing along the wrong axis.** The position belongs on the sequence axis, which is axis 1 of a (B, S, H, D) cache, and every other start index is 0.
- **Forgetting the dtype conversion.** A float32 block cannot be written into a bfloat16 cache, so the new vectors must take on the cache dtype first.
- **Counting the length wrong.** The new length is position plus the number of new tokens L, which is more than position + 1 whenever several tokens are written at once.
- **Shifting the start by one.** The first new token goes into slot position itself, so starting at position + 1 leaves a gap and loses the last slot.
