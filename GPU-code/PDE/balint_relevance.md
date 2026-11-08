# How relevant is Bálint Tóth's Poisson-solver talk to this code?

Summary of the talk (`balint.txt`) and its applicability to the solvers in this
folder (`2D-Poisson-Jacobi.{cu,py}`, `2D-Poisson-RedBlack-SOR.{cu,py}`).

## What the talk reports

1. 2-D Poisson for plasma simulations is **memory-bound**; Jacobi/Gaussian
   elimination are too slow, so the team built a **geometric multigrid
   V-cycle**.
2. Launching one kernel per grid level left the GPU idle; **fusing all levels
   below 32×32 into a single block with device-side synchronisation** gave
   ~10× over the multi-kernel version and ~80% utilisation.
3. They then replaced it with a **spectral solver**: the 2-D sine transform
   implemented as an FFT (cuFFT) using the mirror-and-flip trick, with the
   whole pipeline in a **CUDA Graph**. It was 17–20× faster than their
   optimised multigrid and ~100× faster than the original baseline on an A100.
4. Because the Poisson solve was ~half of the simulation's runtime, fixing it
   moved the bottleneck elsewhere.

## What this code does

| File | Method | Measured (C++) | Per iteration |
|---|---|---|---|
| `2D-Poisson-Jacobi.cu` | naive 5-point Jacobi, float32, ping-pong buffers, 500 fixed iterations, **1 launch/iteration** | N=1024: 81.6 ms kernel | 0.163 ms |
| `2D-Poisson-RedBlack-SOR.cu` | red and black kernels (in place), float64, ω = 2/(1+sin(π/N)), residual checked every 20 iterations | N=1024: 3100 iterations, 1968 ms | 0.635 ms |

## Finding-by-finding relevance

### "GPUs are memory-bound" — confirmed, and it caps the headroom

Jacobi at N=1024 moves ~12 MB per iteration (read `u_old` and `f`, write
`u_new`) in 0.163 ms ≈ **74 GB/s**; Red-Black SOR (double) moves ~40 MB in
0.635 ms ≈ **63 GB/s**. Both are at 75–88% of this P2000's practical ~84 GB/s
ceiling, so stencil micro-optimisation (shared memory, `float4` loads) can win
at most ~15% here. This matches the conclusion already in `README.md`.

### Kernel fusion — same idea, different location

His fusion was *across grid levels* of a multigrid; this code has no
multigrid, so there is nothing analogous to fuse. The underlying principle —
replace kernel boundaries with device-side synchronisation — does apply to
the Red-Black pair, which costs **2 launches per iteration (6,200 launches at
N=1024)**. A single cooperative kernel doing red → `cg::this_grid().sync()` →
black (the pattern used in `leet/50_rms_normalization`) halves that.

A second instance of the same phenomenon: the convergence check calls
`cudaDeviceSynchronize()` and copies a 4-byte residual to the host every 20
iterations — **155 full GPU stalls** at N=1024. That is his "GPU sits idle"
problem, caused by the stopping criterion rather than by launches. The fix is
to evaluate the residual on-device and only synchronise when it may have
converged (or at the end).

### CUDA Graphs — directly applicable, valuable only at small N

The iteration loops are host-driven. At N=256 a Jacobi iteration takes 5.4 µs
total, most of it launch overhead; SOR at N=256 takes ~35 µs per iteration for
two launches. Capturing blocks of iterations in a graph (keeping the
convergence check outside) removes most of that. At N=1024 the iteration is
0.163 ms / 0.635 ms, so launch overhead is already <2% and graphs would change
nothing.

### Multigrid — absent here, and the biggest algorithmic gap

The README's own measurements show why he called Jacobi too slow: at N=256 the
residual is **still 0.957 after 5,000 iterations**, and SOR needs 3,100
iterations at N=1024 to reach 1e-4. Multigrid converges with O(N) work and a
much smaller constant. His 10× fusion win and the 17–20× multigrid→spectral gap
both sit *on top of* multigrid's advantage over Jacobi/SOR.

### Spectral (DST via FFT) — most relevant conceptually, biggest change

The PDE here is exactly the one his solver targets: 2-D Poisson, zero Dirichlet
boundary conditions, uniform rectangular grid. The sine transform diagonalises
the 5-point Laplacian, so 3,100 SOR iterations collapse to one forward
transform, one pointwise multiply and one inverse transform — machine
precision in O(N log N).

Caveats for this repository:

- cuFFT is an external library. The `leet/` challenges forbid external
  libraries outright; the GPU-code brief allows them for parts but not as a
  substitute for the solution. A hand-written radix-2 FFT already exists in
  `GPU-code/Fourier` and could be extended into a DST to stay library-free —
  feasible, but a substantial piece of work.
- His mirror-and-flip DST-via-FFT (and the whole spectral method) only works
  for zero Dirichlet on a uniform rectangle. Jacobi/SOR generalise trivially
  to other boundary conditions and coefficients.

### His specific numbers — not transferable

The 100× and 17–20× figures are against an unspecified baseline on an A100.
Our baseline is hand-written and already near DRAM peak on a Quadro P2000. What
transfers is the ordering of impact, below.

## If the talk's ideas were applied here, in order of payoff

1. **DST spectral solver** — 3,100 iterations become one transform pair; the
   only route to a ~100× class improvement. Requires an FFT (custom, to stay
   library-free).
2. **Multigrid V-cycle** — if the solver must stay general; then adopt his
   coarse-level fusion (all levels below 32×32 in one block) and CUDA graphs,
   because that is exactly where they matter.
3. **Cheap wins in the current code** — fuse red/black into one cooperative
   kernel, make the residual check device-side (removing the 155 stalls), and
   graph the fixed-iteration blocks. Realistic gain: 10–30% at N=256, under
   10% at N=1024 — and no change to the convergence rate, which is the actual
   bottleneck.
