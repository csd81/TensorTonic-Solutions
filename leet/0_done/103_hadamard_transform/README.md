# Batched Hadamard Transform — CUDA C++ and Numba Results

Challenge: apply the orthonormal Walsh–Hadamard transform
`y_row = (1/√N) H_N x_row` to every row of a `B × N` matrix (`N` a power of
two).  Benchmark: **B = 8192, N = 4096** — 268 MB of traffic (134 MB read +
134 MB written), so the problem is memory-bound: a pure `float4` read+write
of the same volume measures **3.29 ms (81.5 GB/s)** on this GPU.

Files:

- `hadamard_transform.cu` — CUDA C++ implementation of `solve(input, output,
  B, N)` (renamed from `starter.cu`).
- `hadamard_transform_naive.cu` — the naive baseline: the starter's kernel
  verbatim (one block per row, a division and modulo per butterfly, scalar
  global and shared accesses).
- `hadamard_transform.py` — Numba CUDA port with the same signature.

## Results at B = 8192, N = 4096

Same-process interleaved medians (20 rounds):

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 9.71 ms (10.06 standalone) | 27.6 |
| **shipped** | **3.41 ms** | **78.8** |
| copy floor (float4 read+write only) | 3.29 ms | 81.5 |
| Numba CUDA port | 6.65–6.98 ms | 38–40 |

Speedup of the shipped kernel over the starter: **2.85×**; it runs within 3%
of the pure-copy floor, so the transform work is almost entirely hidden
behind the required memory traffic.

Correctness: the challenge example (`B=1, N=4, [1,2,3,4] → [5,-1,-2,0]`) and
random cases `B,N = 1x1, 1x2, 3x8, 17x64, 64x256, 129x512, 1000x1024,
8192x4096` all match a float64 reference within the challenge tolerance
(`abs ≤ 1e-4 + 1e-4·|ref|`; max absolute error 4e-7).

## Why the starter is slow

At 10.06 ms it moves 268 MB at only 26.7 GB/s.  For every butterfly it
recovers the pair indices with an integer division and modulo by the runtime
stride `h` (`i/h`, `i%h`) — 12 stages × 2048 butterflies per row, all paying
that cost — and the small-`h` stages have bank-conflicted shared accesses.
The global loads and stores are also scalar.

## Implementation — CUDA C++

One block (512 threads) per row, so the row never leaves the SM:

1. **Global traffic as `float4`.** Rows start 16-byte aligned for `N ≥ 8`
   (N is a multiple of 4), so each thread loads/stores 16-byte vectors.
2. **Stages `h = 1, 2, 4` in registers.** Each thread loads 8 consecutive
   elements (two `float4`s) and runs the first three butterfly stages on
   them before anything reaches shared memory — all three stages pair
   elements inside the thread's own 8 registers.  This is the decisive step:
   adding it took the kernel from 3.97 ms to 3.33 ms, i.e. from 17% above the
   copy floor to within 1%.  (The extra `float4` loads of a warp are 32-byte
   strided, but the paired second load covers the complementary half of every
   128-byte line, so DRAM traffic is unchanged.)
3. **Remaining stages `h = 8 … N/2` in shared memory**, four butterflies at a
   time: a thread loads two `float4`s a distance `h` apart, does the add/sub
   elementwise and stores both back.  Consecutive lanes touch consecutive
   16-byte chunks — conflict-free for 128-bit shared accesses.
4. **Shift/mask index math.** With `h = 1<<e` and `v` indexing the
   float4-pairs of a stage: `section = v >> (e-2)`,
   `offset = (v & (h/4-1)) << 2`.  No divisions.
5. **The last stage (`h = N/2`) writes its results directly to global
   memory**, scaled — it covers every element of the row, so the final shared
   round trip disappears.  (Measured separately, this fusion is neutral on
   its own; the win is all in step 2.)

Small `N` is handled by a one-thread-per-row fallback kernel for `N = 1, 2, 4`
and by the same row kernel with fewer threads for `8 ≤ N ≤ 4096`.

Variants that measured worse: 256 or 128 threads per block (4.53 / 5.14 ms —
fewer warps per row cannot hide the shared-memory latency); a version without
the register stages (3.97 ms); skipping the last-stage fusion (3.99 ms).

## Numba port

Numba has neither `float4` accesses nor a practical way to hold several
elements per lane without spilling to local memory, so the port is the
classic shape: one block per row, coalesced scalar loads into a *dynamic*
shared array (`cuda.shared.array(0, dtype)` with the byte size passed as the
fourth launch-configuration item), all `log2(N)` stages in shared memory, and
coalesced scaled stores.  The starter's per-butterfly division/modulo is
replaced by shifts and masks (`section = v >> e`, `offset = v & (h-1)`).

Measured 6.65–6.98 ms (≈ 2× slower than the C++ kernel) with 256 threads per
block, the best of 128/256/512/1024 (8.20/6.98/7.75/9.38 ms).
