# Multi-Adapter LoRA Batched Linear — CUDA C++ and Numba Results

Challenge: batched multi-LoRA projection.  For token `t` with adapter
`a = adapter_ids[t]`:

```
y_t = x_t W + scale * (x_t A_a) B_a     (a >= 0)
y_t = x_t W                              (a = -1)
```

`x [T,K]`, `W [K,N]`, `A [L,K,R]` (`A_a[k,r]` row-major), `B [L,R,N]`
(`B_a[r,n]` row-major), `y [T,N]`, all float32.  Constraints: `T ≤ 4,096`,
`K,N ≤ 8,192`, `R ≤ 64`, `L ≤ 64`.  Performance is measured at **T = 1024,
K = 4096, N = 4096, R = 16, L = 8** — 34.4 GFLOP, 96% of it in the base
product `x W`.

Files:

- `multi_lora_batched_linear.cu` — CUDA C++ implementation of
  `solve(x, W, A, B, adapter_ids, y, T, K, N, R, L, scale)` (renamed from
  `starter.cu`).
- `multi_lora_batched_linear_naive.cu` — the naive baseline: the starter's
  two kernels verbatim (one thread per output for the base product; a LoRA
  kernel that recomputes the low-rank vector per output element).
- `multi_lora_batched_linear.py` — Numba CUDA port with the same signature.

## Results at the benchmark shape

| Variant | Time | TFLOP/s |
|---|---:|---:|
| **naive = starter kernels** | 3604–3736 ms | 0.009–0.010 |
| **shipped** | **30.5–31.5 ms** | **1.10–1.13** |
| Numba CUDA port | 110.7 ms | 0.313 |

Speedup of the shipped kernel over the starter: **~120×**.

Where the time goes at the benchmark shape, measured by launching each phase
alone:

| Phase | starter / naive | shipped |
|---|---:|---:|
| base `y = x W` (34.4 GFLOP) | 298.9 ms (0.116 TFLOP/s) | 28.7 ms (1.206 TFLOP/s) |
| LoRA path | 3400.2 ms | 2.32 ms |
| total | 3736 ms | 31.4 ms |

Correctness: the challenge example (`T=3, K=2, N=2, R=1, L=2, ids=[0,1,-1]`,
`scale=0.5`) plus random cases `1x1x1`, `5x3x7 r2`, `17x33x19 r5`,
`64x128x64 r16`, `100x257x129 r64`, `256x512x512 r16`, `129x130x131 r3 l64`
(max adapter pool), an all-`-1` id case, and the benchmark shape all pass
against a float64 CPU reference: small cases element-for-element, the
benchmark shape over 4096 random output elements (a full check would be
1.7e10 CPU multiply-adds).  Tolerance: `1e-2 + 1e-3·|ref|` (float32
accumulation of 4096 terms).

## Why the starter is slow

Two independent problems:

1. **Base GEMM**: one thread per output element with a K-loop reading `W`
   column-strided; no tiling or reuse — 0.116 TFLOP/s (299 ms).
2. **LoRA path**: for *every output element* `(t, n)` the kernel recomputes
   `x_t · A_a` from scratch — `R·K = 65,536` multiply-adds per element, so
   `O(T·N·R·K) = 2.7e11` operations.  It is 91% of the starter's 3.74 s.

The LoRA update is low-rank, so the fix is to not recompute it per element:
compute `temp_t = x_t A_a` once per token (an `R`-vector) and apply it to the
token's `N` outputs.  That turns the cost into `O(T·K·R + T·R·N)` = 2.7e8
operations — 1000× less arithmetic.

## Implementation — CUDA C++

**Base product**: the register-blocked GEMM shape that won the GEMM challenge
on this GPU — 128×128 block tile, 16×16 threads, 8×8 per thread, BK=16,
interleaved per-thread columns (consecutive lanes read consecutive shared
elements, conflict-free), scalar coalesced tile loads with bounds-checked
zero fill, FP32 FMA accumulation.  This measures 1.206 TFLOP/s at the
benchmark shape, with the same compile options (`-O3 -arch=sm_61`).  Note
that sm_61 has no tensor cores, so FP32 register blocking is the ceiling;
the practical FP32 peak here is ~2.27 TFLOP/s and ~1.2 TFLOP/s is the
realistic GEMM ceiling.

**LoRA path**: one block (128 threads) per token:

- `temp[r] = Σ_k x_t[k] · A_a[k,r]` — the 128 lanes split over both `r` and
  `LORA_THREADS/R` chunks of K (8 chunks at R = 16), each partial accumulated
  in a register and reduced through shared memory.  The K-loop runs once per
  token instead of once per output element.
- `y[t,n] += scale · Σ_r temp[r] · B_a[r,n]` for the token's N columns, 128
  lanes striding over n; B loads are coalesced, `temp` is read from shared
  memory.

LoRA block size was swept (64/128/256/512): 128 measured best and most stable
(30.72–30.83 ms full solve versus 30.52–31.44 for 256, 30.71 for 64).

## Numba port

The port keeps the same two-phase structure:

- **Base GEMM**: Numba cannot hold an 8×8 accumulator tile in registers from
  a value array, so the port uses a 64×64 block tile with 16 *explicit
  scalar* accumulators per thread (4 rows × 4 interleaved columns, 16×16
  threads, BK=32).  Shared reads per FMA are 0.5 versus 0.25 for the C++
  128×128/8×8 tile.  The PTX confirms no local-memory spills and 16 FP32
  FMAs per k-step.
- **LoRA path**: the same one-block-per-token kernel; Numba supports the
  shared `temp`/`partial` arrays directly.

Two Numba-specific requirements were hit here: shared-array shapes must be
lifted to module-level literal constants (`(BM, BKP)` instead of
`(BM, BK + 1)`, which types as `int64` and is rejected), and the output must
be passed as a flat 1-D device array (indexing a 2-D device array returns a
row, so `y[t*N+n] += …` fails to type).

Measured 110.7 ms at the benchmark shape (0.313 TFLOP/s), 3.5× slower than
the C++ kernel.
