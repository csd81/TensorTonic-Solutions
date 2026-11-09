# Multi-Agent Simulation (Boids) — CUDA C++ and Numba Results

Challenge: one flocking step.  For every agent `i`, find the agents within
radius `r = 5`, average their velocities (`v_i` itself when the
neighbourhood is empty), then

```
v_new = v + 0.05 * (v_avg - v)
p_new = p + v_new
```

Agents are stored as 4 floats each (`x, y, vx, vy`); the result goes to
`agents_next`.  `1 ≤ N ≤ 100,000`; performance is measured at `N = 10,000`.

Files:

- `multi_agent_sim.cu` — CUDA C++ implementation of
  `solve(agents, agents_next, N)` (renamed from `starter.cu`).
- `multi_agent_sim.py` — Numba CUDA port with the same signature.

## The starter

An O(N²) sweep: one thread per agent, every agent compared against every other
through 256-agent shared-memory tiles.  At N = 10,000 that is 10⁸ distance
tests, and it measures **1.35–1.51 ms** — essentially the same for any spatial
distribution, because the work is fixed by N alone.

## Optimization 1 — cheaper comparisons (always on)

- The tile is split into a `float2` position array and a `float2` velocity
  array, so the inner loop moves 8 bytes per candidate instead of a 16-byte
  `float4`, and only touches the velocity when the radius test passes.
- Every agent counts itself (its distance is 0, which always passes) and
  subtracts once at the end.  That removes the `i != j` test from the inner
  loop entirely.

Measured at N = 10,000: 1.34–1.51 ms → **0.87 ms** (1.55–1.75×).

## Optimization 2 — spatial hash (the big win)

With a fixed radius the O(N²) sweep is mostly wasted: an agent can only be
affected by agents in its own cell and the eight neighbours, because a cell is
exactly `r` wide — any point within `r` must lie in that 3×3 block.

The binning is a counting sort over a **hashed** cell index, so no bounding
box is needed and any coordinate range works:

| Kernel | Work |
|---|---|
| `count_kernel` | one atomic per agent into `counts[hash(cell)]` |
| `scan_blocks_kernel` + `scan_sums_kernel` | two-level exclusive scan of the bucket counts |
| `scatter_kernel` | counting-sort scatter of positions and velocities |
| `query_kernel` | for each agent, scan the nine hashed buckets, de-duplicated |

Two details matter:

- **Hash collisions must not double count.**  Two different cells can hash to
  the same bucket, so the query kernel collects the nine bucket indices and
  skips duplicates — every bucket is scanned at most once and every
  neighbouring cell is still covered.
- **The scan has to be parallel.**  A single block scanning all 65,536
  buckets cost 0.22 ms by itself; a two-level scan (chunk prefixes in parallel
  blocks, chunk totals in one block, with the chunk prefix folded into the
  scatter and the query) removed that.

The same self-count trick applies: the agent is in its own cell, so it is
always found once and subtracted at the end.

Small inputs (`N < 4096`) keep the direct sweep, where the four extra launches
would not pay for themselves.

## Measurements — N = 10,000

Three spatial densities, all with the same N (the starter's time barely moves;
the spatial hash's does not):

| Spread of positions | starter | **CUDA C++ final** | Numba CUDA |
|---|---:|---:|---:|
| ±50 (dense, ~78 neighbours) | 1,512 µs | **312 µs** (4.8×) | 1,226 µs |
| ±200 | 1,317 µs | **105 µs** (12.5×) | 446 µs |
| ±2,000 (sparse) | 1,350 µs | **95 µs** (14.2×) | 377 µs |

The dense case is the expensive one because the query genuinely has to test
~225 candidates per agent; the sparse cases collapse to a handful of tiny
kernels.  (A pure read of the agents costs ~5 µs, so the sparse case is
launch-bound.)

## Implementation — Numba

The same five-kernel pipeline with a single-block scan of the bucket counts
(Numba is dominated by its ~60 µs per-launch overhead: the five launches are
~300 µs of the total).  It still beats the starter by 1.2–3.6×.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, 6 SMs) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-09 |

Times are medians of 30–50 runs after a warm-up, including
`cudaDeviceSynchronize()`.

## Correctness — CUDA C++

Compared against a float64 CPU implementation of the same rule; every case
matches (maximum relative error 7.6e-08):

| Case | N |
|---|---:|
| challenge example (`(0,0,v=1,0)` and `(3,4,v=0,-1)`, distance exactly 5 → not neighbours) | 2 |
| single agent | 1 |
| three mutually close agents | 3 |
| all isolated (no neighbours) | 7 |
| clustered grid | 64 |
| random, dense | 100, 4,000 |
| random, medium | 1,000 |
| random, sparse | 2,000 |
| benchmark size (sampled check, three densities) | 10,000 |

## Correctness — Numba CUDA

All seven cases match as well.

## Build / run

CUDA C++ (the file exports `solve`; compile it with a harness that provides
`main`):

```bash
nvcc -O3 -arch=sm_61 -o test multi_agent_sim.cu test_harness.cu
```

Numba:

```bash
python3 multi_agent_sim.py
```

The script runs the correctness cases and the three performance cases.  For
best performance, pass Numba device arrays; host arrays incur copies to and
from the device.

## Naive baseline

`multi_agent_sim_naive.cu` is the same computation written as plainly as possible:
one thread per agent scanning every other agent in global memory (a full O(N^2) sweep) — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `multi_agent_sim_naive.cu` | 2100.4 / 1948.5 / 1938.5 us | 1.00x |
| `multi_agent_sim.cu` (optimized) | 277.1 / 106.0 / 88.3 us | 7.6 / 18.4 / 22.0x |

Times are for position spreads of +/-50, +/-200 and +/-2000; the optimized kernel bins agents into a spatial hash so each one tests only its 3x3 cell neighbourhood.
