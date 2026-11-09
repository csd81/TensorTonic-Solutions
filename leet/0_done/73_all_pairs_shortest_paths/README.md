# All-Pairs Shortest Paths (Floyd-Warshall) — CUDA C++ and Numba Results

Challenge: Floyd-Warshall on a dense `N × N` float distance matrix
(`dist[i*N+j]` is the edge weight, `+inf` = no edge, diagonal 0, no negative
cycles).  `N ≤ 4096`; performance is measured at `N = 2048`.

Files:

- `all_pairs_shortest_paths.cu` — CUDA C++ implementation of
  `solve(dist, output, N)` (renamed from `starter.cu`).
- `all_pairs_shortest_paths_naive.cu` — the naive baseline: one kernel per
  intermediate vertex `k`.
- `all_pairs_shortest_paths.py` — Numba CUDA port with the same signature.

## Where the time goes at N = 2048

The matrix is 16 MB and a single `k` step touches every element, so the
starter's one-kernel-per-`k` form launches **2048 kernels**, each streaming
~32 MB: ~65 GB of traffic and 2048 synchronisations.  It measures
**777 ms**.

The shipped kernel is the standard **blocked Floyd-Warshall**: 32×32 tiles,
the `k` loop processed one 32-vertex block at a time:

| Variant | Time | vs optimized |
|---|---:|---:|
| starter (one kernel per k) | 776.9 ms | 8.2× slower |
| `all_pairs_shortest_paths_naive.cu` (same form) | 687.7 ms | 7.3× slower |
| **`all_pairs_shortest_paths.cu`** (blocked FW, 32×32 tiles) | **94.0–94.3 ms** | 1.0× |
| Numba CUDA (one kernel per k, Python loop) | 805.7 ms | 8.5× slower |

## Implementation — CUDA C++

Four kernels per 32-vertex `k`-block (64 blocks → 256 launches instead of
2048):

1. **phase 1** — the diagonal tile `(kb,kb)` runs the plain sequential FW
   over its 32 `k` values, publishing each element and syncing after every
   `k` so the in-tile updates feed the next step.
2. **phase 2** — every tile in the k-block's column `(I,kb)` is updated
   against the diagonal tile, walking `k` sequentially inside the tile (the
   tile's own `k`-column is updated in place and reused).
3. **phase 3** — every tile in the k-block's row `(kb,J)` likewise.
4. **phase 4** — every remaining tile `(I,J)` does the min-plus
   `D[I][J] = min(D[I][J], D[I][kb] + D[kb][J])` with the updated column and
   row blocks staged in shared memory (two 32×32 tiles, 8 KB).

Traffic drops from ~65 GB to ~4 GB, and the inner `relax` has no infinity
branch: `inf + finite = inf` and `inf + inf = inf`, so `min` handles the
no-edge case naturally.  Removing that branch alone took 113 → 94 ms.

Out-of-range threads in phases 1–3 stay alive for the `__syncthreads()`
inside the `k` loop (with a per-thread `active` flag) — returning early
would deadlock when `N` is not a multiple of the tile size.

### Naive baseline

`all_pairs_shortest_paths_naive.cu` is the plainest correct form: one kernel
per `k`, one thread per `(i,j)`, a sync between steps.  It measures
**687.7 ms** — 7.3× slower.

## Implementation — Numba

The per-`k` kernel driven from a Python loop (the blocked four-phase
structure is not practical in Numba).  It measures **805.7 ms**: the GPU work
matches the C++ naive form plus one Python-side dispatch per launch (2048 of
them).  Kernels use `cache=True`.

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
C++ variants were compiled against the same harness and run sequentially.

## Correctness — CUDA C++

Checked against a float64 CPU Floyd-Warshall with a relative tolerance of
1e-3 (and exact infinity handling: `inf` in the reference must be `inf` in
the output):

| Case | Result |
|---|---|
| challenge example (4×4) | exact |
| 1×1, 2×2 | exact |
| random sparse graphs N = 7, 33, 100 | exact |
| benchmark graph N = 2048 (~20% edge density) | exact |

All 7 harness cases pass.

## Correctness — Numba CUDA

All 6 cases pass with the same tolerance.

## Performance — N = 2048

| Implementation | Time | vs optimized |
|---|---:|---:|
| starter (one kernel per k) | 776.9 ms | 8.2× slower |
| `all_pairs_shortest_paths_naive.cu` (same form) | 687.7 ms | 7.3× slower |
| **`all_pairs_shortest_paths.cu`** (blocked FW) | **94.3 ms** | 1.0× |
| Numba CUDA (one kernel per k) | 805.7 ms | 8.5× slower |

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test all_pairs_shortest_paths.cu test_harness.cu
```

Numba:

```bash
python3 all_pairs_shortest_paths.py
```

The script runs the correctness cases and the performance case.
