# Gaussian Blur — CUDA C++ and Numba Results

Challenge: 2-D convolution of an image with a normalized Gaussian kernel,
zero padding at the borders.  Benchmark: **512 × 512 image, 7 × 7 kernel** —
12.8 M multiply-adds over 1 MB in and 1 MB out.  This is a tiny kernel: a
pure `float4` copy of the same 2 MB measures **31.8 µs including launch**,
and that is the floor.

Files:

- `gaussian_blur.cu` — CUDA C++ implementation of
  `solve(input, kernel, output, input_rows, input_cols, kernel_rows,
  kernel_cols)` (renamed from `starter.cu`).
- `gaussian_blur_naive.cu` — the naive baseline: the starter's kernel
  verbatim (fixed 36×36 tile sized for the max radius, one output per
  thread, one scalar shared load per multiply-add).
- `gaussian_blur.py` — Numba CUDA port with the same signature.

## Results at 512 × 512, 7 × 7

Same-process interleaved medians (100 rounds):

| Variant | Time | Effective GB/s |
|---|---:|---:|
| **naive = starter kernel** | 211.5 µs | 9.9 |
| **shipped** | **57.7–59.0 µs** | **36** |
| copy floor (float4 read+write only) | 31.8 µs | — |
| Numba CUDA port | 233.8 µs | 9.0 |

Speedup of the shipped kernel over the starter: **3.6×**.  It runs at ~1.8×
the pure-copy floor; the rest is the convolution arithmetic and shared-memory
traffic, which at this size never fully overlap with the 2 MB of I/O.

Correctness: both challenge examples (checked exactly) and random cases
`1×1 k3`, `1×7 k7×3`, `100×37 k5×3`, `64×64 k21×21` (maximum kernel),
`33×65 k9×13`, `512×512 k21×7`, `2048×2048 k5×5`, `4096×4096 k3×3` (maximum
image) all match a float64 reference within the challenge tolerance
(`abs ≤ 1e-4 + 1e-4·|ref|`; max absolute error 1.4e-7).

## Why the starter is slow

- Fixed shared tile `36×36` (sized for the maximum radius 10 whatever the
  kernel is) with a flattening division per element.
- One output per thread with one scalar shared load per multiply-add, so the
  shared-memory pipe is the bottleneck: 12.8 M loads at ~32 floats/cycle/SM.
- Kernel-size loops with runtime bounds cannot unroll, so every
  multiply-add also pays index arithmetic.

## Implementation — CUDA C++

Block = 16×16 threads computing a 64×16 output tile; each thread produces
**4 adjacent output columns**.

1. **Tile sized to the actual halo**: dynamic shared memory of
   `(BH + 2·ph) × (BW + 2·pw)` floats, loaded with nested row/column stride
   loops (no flat-index division by the runtime pitch) and zero padding.
2. **Register window via compile-time kernel width**: the kernel is
   instantiated for each odd width 3…21 (the challenge constraint; a runtime
   fallback exists for other widths).  With `KC` a compile-time constant the
   inner loop fully unrolls, and since the overlapping window reads of the
   unrolled iterations have identical addresses and nothing writes to shared
   memory in between, the compiler keeps the `4 + KC − 1` window values in
   registers: ~10 shared loads per 28 multiply-adds instead of 28.  This is
   the decisive step: the same kernel with a runtime kernel width measures
   84 µs, with the template 55 µs.
3. **Vectorized stores**: `float4` when the row stride and column are
   4-aligned (the benchmark case), guarded scalars otherwise.

Variants measured in the same harness (512×512, 7×7): one output per thread
with a 32×8 block and dynamic tile, 221 µs (same as the starter: the LSU
bound is unchanged); 4 columns per thread with runtime kernel width, 84–89 µs;
templated 4 columns per thread, 55–59 µs (shipped); 8 columns per thread,
66–69 µs; 4×4 outputs per thread with runtime width, 145 µs; 32-row block,
59 µs.

## Numba port

Same block shape and 4-columns-per-thread structure, with the shared tile in
dynamic shared memory (`cuda.shared.array(0, dtype)` with the byte size at
launch).  Numba cannot specialize a kernel on a runtime argument, so the
inner loop keeps a runtime kernel width; to avoid one shared load per
multiply-add anyway, the port uses a **rolling four-register window**
(`w0..w3`): each `kc` step issues one shared load and shifts the registers.
The tile load uses nested row/column loops because a flat index would lower
to a 64-bit division per element.

Measured 234 µs — slower than even the C++ naive kernel.  The gap is Numba's
codegen for this kind of scalar shared-memory stencil loop, not launch
overhead (an empty Numba kernel launch + sync measures 34 µs here).
