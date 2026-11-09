# MoE Token Dispatch — CUDA C++ and Numba Results

Challenge: given `x [T, D]` and `expert_idx [T]`, write each expert's tokens
into `dispatched_x [E, capacity, D]` (`capacity = T`) preserving the original
token order within an expert, and fill `token_counts [E]`.  Benchmark:
**T = 16,384, D = 512, E = 8** — 32 MB read + 32 MB written, so the copy
floor is ~0.8 ms at this GPU's ~80 GB/s.

Files:

- `moe_token_dispatch.cu` — CUDA C++ implementation of
  `solve(x, expert_idx, dispatched_x, token_counts, T, D, E, capacity)`
  (renamed from `starter.cu`).
- `moe_token_dispatch_naive.cu` — the naive baseline: the starter's two
  kernels verbatim (serial single-thread slot assignment, one block per token
  copying scalars).
- `moe_token_dispatch.py` — Numba CUDA port with the same signature.

## Results at T = 16,384, D = 512, E = 8

Same-process interleaved medians (20 rounds):

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernels** | 8.39 ms | 8.0 |
| **shipped** | **0.880 ms** | **76.4** |
| Numba CUDA port | 1.07 ms | 62.7 |

Speedup: **9.5×**.  Phase split of the shipped code, timed separately:
slot assignment 0.036 ms, copy ≈ 0.84 ms — the copy is at the memory floor.

Correctness: the challenge example (`T=4, D=3, E=2, idx=[0,1,0,1]` →
counts `[2,2]`) and random cases `1x1 e2`, `1000x7 e8`, `1000x512 e8`,
`5000x64 e3`, `65536x4 e64` (maximal expert pool), plus adversarial
routings (all tokens to expert 0, perfect round-robin) and the benchmark
shape.  The harness checks counts and every dispatched element for equality
against the token's original row (exact float copies).

## Why the starter is slow

Its slot kernel uses **one thread** to walk all T tokens, with a dependent
global load per token (`expert_idx[t]`) and a read-modify-write of the
per-expert counter — about 7.5 ms of the 8.39 ms total.  The copy kernel
(one block per token, scalar loads) then runs at roughly the expected
bandwidth.

## Implementation — CUDA C++

Two kernels:

1. **Stable slot assignment — one warp per expert** (`slot_kernel`, grid `E`,
   block 32).  Each warp walks the tokens in ascending order and uses the
   classic ordered-compaction primitive:

   ```
   mask    = __ballot_sync(0xffffffff, exp == e)   // 32-token window
   rank    = __popc(mask & lanes_before_me)        // position within the window
   if (exp == e) token_slots[i] = running + rank;
   running += __popc(mask);
   ```

   The ballot mask is uniform across the warp, so `running` stays uniform in
   every lane — no shuffles, no atomics, no shared memory, no
   `__syncthreads`.  Order is preserved because windows are processed in
   ascending token order and lanes take ascending positions inside a window.
   The last window is handled with a uniform trip count and `exp = -1` for
   tail lanes: `__ballot_sync` with a full mask while some lanes have exited
   is undefined, and the naive `for (i = lane; i < T; i += 32)` form
   produced wrong slots whenever T was not a multiple of 32 (caught by the
   `T = 1000/500` harness cases before shipping).
2. **Copy — one warp per token** (`dispatch_kernel`).  Each lane moves
   `float4`s (`D = 512` → four 16-byte loads/stores per lane); a token row is
   four coalesced 512-byte transactions.  A scalar path handles `D` not
   divisible by 4 (row starts would lose 16-byte alignment).  Destination
   offsets are 64-bit (`E·capacity·D` can exceed 2^31).

The `token_slots` workspace is allocated once and cached across `solve` calls:
a `cudaMalloc`/`cudaFree` pair per call (as the starter does) would cost more
than the slot kernel itself.

Measured variants: launching the slot kernel with more warps per expert
(chunked prefix) was not needed — the single-warp version costs 36 µs, 4% of
the runtime.

## Numba port

Numba exposes `cuda.ballot_sync` and `cuda.popc`, so the slot kernel is the
same algorithm, including the uniform-trip-count window loop.  The copy
kernel cannot use `float4` (Numba has no vector data type), so each warp
moves scalars at a 32-lane stride — still one coalesced 128-byte transaction
per instruction — and destination offsets are built in `int64`.

Measured 1.07 ms (62.7 GB/s), 22% slower than the C++ kernel.
