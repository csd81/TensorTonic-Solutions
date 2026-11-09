# Rainbow Table (R × FNV-1a) — CUDA C++ and Numba Results

Challenge: apply `R` rounds of FNV-1a to each 32-bit integer, feeding each
round's hash into the next.  `1 ≤ N ≤ 10,000,000`, `1 ≤ R ≤ 100`; performance
is measured at `N = 5,000,000`.

Files:

- `rainbow_table.cu` — CUDA C++ implementation of `solve(input, output, N, R)`
  (renamed from `starter.cu`).
- `rainbow_table_naive.cu` — the naive baseline: one thread per element,
  scalar load, the R-round chain, scalar store.
- `rainbow_table.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 5,000,000

The traffic is 20 MB in + 20 MB out = 40 MB, so a pure copy costs **0.49 ms
(81 GB/s)**.  What actually decides the runtime is `R`, because each hash is
a chain of four *dependent* 32-bit multiplies (`h = (h ^ byte) * prime`):

| R | optimized kernel | bound by |
|---:|---:|---|
| 1 | 0.495 ms | memory (at the copy floor) |
| 10 | 0.903 ms | multiply issue / latency |
| 100 | 9.27 ms | instruction throughput (~2e9 dependent multiplies) |

The benchmark's `R` is not fixed by the challenge, so all three regimes are
reported.

## Implementation — CUDA C++

Each thread loads one `int4` (four elements) and runs **four independent hash
chains interleaved** in the R loop.  The chains are independent, so the
scheduler can overlap their multiplies instead of waiting out each
dependency chain — this is worth ~15% at R = 10.  At R = 100 the kernel is
issue-throughput-bound rather than latency-bound, so ILP only buys ~3%.

The `N % 4` tail is handled by a **separate tiny kernel** launched only when
needed.  Putting the tail loop in the hot kernel cost ~15% at R = 10 —
apparently register pressure from the second hash loop — even though only one
thread would ever execute it.  If either pointer is not 16-byte aligned, a
scalar one-element-per-thread kernel runs instead.

### What was swept (medians, N = 5,000,000)

| Variant | R=1 | R=10 | R=100 |
|---|---:|---:|---:|
| scalar, 256 thr (starter shape) | 0.490 | 1.055 | 9.52 |
| int4, 128 thr | 0.493 | 0.919 | 9.38 |
| int4, 256 thr | 0.490 | 0.904 | 9.27 |
| **int4, 512 thr** | **0.491** | **0.897** | **9.24** |
| int4, 8 elem/thread (2× int4) | 0.491 | 0.900 | 9.30 |
| int4, hash written as a 4-iteration loop | 0.492 | 0.905 | 9.28 |

Widening beyond four elements per thread adds register pressure without more
useful ILP.  The unrolled and loop forms of the hash compile to the same
thing.

### Naive baseline

`rainbow_table_naive.cu` is the plainest correct form: one thread per
element, one element at a time, the R rounds as a plain loop.  It measures
**0.495 / 1.068 / 9.55 ms** for R = 1 / 10 / 100.  There is no memory
inefficiency to exploit — one element per thread is already fully coalesced —
so the gap only appears where the dependent multiply chain matters: 1.18× at
R = 10 and 1.03× at R = 100.

(The starter's scalar kernel is slower still at R = 10 — 1.21 ms vs 1.07 ms
for this naive form — so how nvcc compiles the two hash loops does matter
even before ILP.)

## Implementation — Numba

Scalar one-element-per-thread kernel (`int4` is not available in Numba).
Measured: **0.723 / 3.66 / 29.4 ms** for R = 1 / 10 / 100.

The large-R gap (3.2×) has a specific cause: Numba's type inference promotes
32-bit integer multiplication to 64-bit (`uint32 * uint32 → uint64`, verified
with `inspect_types`), and 64-bit multiplies run at roughly a quarter of the
32-bit rate on this GPU.  That also forced a correctness fix in the port: the
top-byte extraction in the fourth FNV step must be masked (`(v >> 24) & 0xFF`)
because the intermediate `v` carries bits above 31, which an unmasked shift
would fold into the next round.  With the mask, the low 32 bits — all the
store keeps — are correct.  Kernels use `cache=True`.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians after a warm-up, including `cudaDeviceSynchronize()`; the
C++ variants were compiled against the same harness and run sequentially on
an idle GPU.  The compute-bound R = 10/100 timings vary by up to ~10% with
the GPU clock state (the first measurement after an idle period reads slow);
the quoted values are steady-state, from back-to-back runs.

## Correctness — CUDA C++

The hash is exact 32-bit integer arithmetic, so every case is checked for
bit-exact equality against a CPU reference.  The reference itself is checked
against the challenge: `fnv1a(fnv1a(123)) = 1636807824`, which matches
example 1, and all three values of example 2 match as well.

| Case | Result |
|---|---|
| challenge example 1 (`[123,456,789]`, R=2) | exact |
| challenge example 2 (`[0,1,2147483647]`, R=3) | exact |
| single 0, R=1 / single 2147483647, R=100 | exact |
| all zero, R=100 | exact |
| sizes 2, 3, 5, 1023, 1025, 4099, 100,000 with R=7 | exact |
| N=10,007 with R=1 / R=10 / R=100 | exact |
| benchmark shape, R = 1 / 10 / 100 | exact |

## Correctness — Numba CUDA

All cases pass bit-exact against a vectorized NumPy FNV-1a reference,
including both challenge examples, R = 1 / 7 / 100, and the benchmark shape.

## Performance — N = 5,000,000

| Implementation | R=1 | R=10 | R=100 |
|---|---:|---:|---:|
| `rainbow_table_naive.cu` (scalar, 1 elem/thread) | 0.495 ms | 1.068 ms | 9.55 ms |
| starter (scalar, loop hash) | 0.493 ms | 1.211 ms | 9.53 ms |
| **`rainbow_table.cu`** (int4, 4 chains, 512 thr) | **0.495 ms** | **0.903 ms** | **9.27 ms** |
| Numba CUDA | 0.723 ms | 3.66 ms | 29.4 ms |
| *copy floor (40 MB, uint4)* | *0.49 ms* | — | — |

At R = 1 everything is at the memory floor; the implementation earns its
keep at R = 10 (1.18× over the naive form, 1.34× over the starter) and is
limited by raw multiply throughput at R = 100.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test rainbow_table.cu test_harness.cu
```

`THREADS` can be overridden with `-D` (the default is 512).

Numba:

```bash
python3 rainbow_table.py
```

The script runs the correctness cases and the performance cases.  For best
performance, pass Numba device arrays; host arrays incur copies to and from
the device.
