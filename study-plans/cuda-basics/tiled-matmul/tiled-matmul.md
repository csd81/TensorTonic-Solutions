# <span style="font-size: 20px;">Tiled Matrix Multiplication</span>

<span style="font-size: 14px;">Tiled matmul computes the same GEMM, $C[i,j] = \sum_k A[i,k]\, B[k,j]$, but it is the **optimized** counterpart of the naive kernel: instead of every thread streaming whole rows and columns from global memory, a block cooperatively stages `TILE x TILE` sub-blocks of $A$ and $B$ into `__shared__` memory and reuses each loaded element `TILE` times. That single structural change - **tiling** - raises arithmetic intensity by roughly a factor of `TILE` and carries GEMM across the roofline ridge from memory-bound to **compute-bound**. This is the central performance lesson of the whole CUDA track.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For output indices $i$ in $[0, M)$ and $j$ in $[0, N)$:</span>

$$
C[i,j] = \sum_{k=0}^{K-1} A[i,k]\, B[k,j]
$$

<span style="font-size: 14px;">$A$ is $M \times K$ row-major, $B$ is $K \times N$ row-major, $C$ is $M \times N$. The math is identical to the naive kernel; only the data movement changes. The trick is to split the shared dimension $K$ into chunks of width `TILE` and accumulate the output one tile of $K$ at a time.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The decomposition is still **one thread per output element**, on a 2-D grid of `TILE x TILE` (commonly `16x16`) blocks tiling the $M \times N$ output. Each thread owns one `C[i,j]` and accumulates into a register across the $K$ dimension:</span>

$$
j = \text{blockIdx.x} \times \text{TILE} + \text{threadIdx.x}, \quad i = \text{blockIdx.y} \times \text{TILE} + \text{threadIdx.y}
$$

<span style="font-size: 14px;">What changes is that a block now acts as a **cooperative unit**. The $K$ loop is broken into $\lceil K / \text{TILE} \rceil$ phases; in each phase the block's 256 threads collectively load one `TILE x TILE` tile of $A$ and one of $B$ into shared memory, then every thread multiplies through those tiles. The block, not the thread, is the natural object - that is the difference from the embarrassingly parallel naive version.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Two `__shared__` arrays, `As[TILE][TILE]` and `Bs[TILE][TILE]`, hold the current tiles. Each thread loads exactly one element of $A$ and one of $B$ into shared memory per phase, and the load is arranged so consecutive lanes hit consecutive global addresses - a **coalesced** fill. Once a tile is resident, each of its elements is read `TILE` times: every row of `As` is consumed by all `TILE` threads sharing that output row, and every column of `Bs` by all `TILE` threads sharing that output column.</span>

<span style="font-size: 14px;">That reuse is the entire point. In the naive kernel each $A$ and $B$ element crosses the DRAM bus $O(N)$ or $O(M)$ times; with tiling each element is fetched from global memory once per tile and then served `TILE` times from on-chip shared memory, which is roughly an order of magnitude lower latency than global. The redundant global traffic is cut by a factor of `TILE`.</span>

<span style="font-size: 14px;">Shared memory is split into 32 **banks**. A square `Bs[TILE][TILE]` of 32-wide rows can map a column onto a single bank, so the column reads during accumulation become a bank conflict that serializes the warp. Padding the tile to `[TILE][TILE+1]` shifts each row by one bank so column accesses span 32 distinct banks and run **conflict-free**, at the cost of one wasted float per row.</span>

---

## <span style="font-size: 16px;">Memory-Bound to Compute-Bound</span>

<span style="font-size: 14px;">With tiling, each global element load now feeds `TILE` multiply-adds instead of one. The effective **arithmetic intensity** rises from the naive $0.25$ to roughly:</span>

$$
\text{intensity} \approx \frac{2 \cdot \text{TILE}}{8} = \frac{\text{TILE}}{4} \text{ FLOP/byte}
$$

<span style="font-size: 14px;">For `TILE = 16` that is about $4$ FLOP/byte, a 16x jump, and larger tiles push it further. Crossing the **roofline** ridge point of tens of FLOPs per byte turns GEMM from **memory-bound** into **compute-bound**: the multiply-add units, idle in the naive kernel, now become the limiting resource. This is the whole roofline lesson in one kernel - the operation always had the FLOPs ($2MNK$), and tiling supplies the reuse that lets the hardware actually deliver them. Past this point, faster arithmetic (wider tiles, register blocking, tensor cores) finally matters, which it never did before.</span>

---

## <span style="font-size: 16px;">Execution Model: Synchronization and Occupancy</span>

<span style="font-size: 14px;">Each tile phase needs **two `__syncthreads()`**. The first comes after the cooperative load, ensuring every lane has finished writing `As` and `Bs` before any lane reads them - without it, threads consume stale or half-written tiles, a shared-memory race. The second comes after the accumulation loop, before the next phase overwrites the tiles, so no thread is still reading the old tile while another loads the new one. Both barriers must sit outside any divergent branch, or the block hangs.</span>

<span style="font-size: 14px;">The shared-memory budget is an **occupancy** knob. Two padded `16x16` float tiles cost about `2 * 16 * 17 * 4` bytes, roughly 2 KB per block. Since each SM has a fixed shared-memory capacity in the tens of KB, a larger `TILE` raises reuse but also raises per-block shared usage, which can cap the number of resident blocks and reduce the warps available for latency hiding. The tile size is therefore a balance between arithmetic intensity and occupancy, not a free dial.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">The naive kernel rereads each operand $O(M)$ or $O(N)$ times from global memory at intensity $0.25$, pinned to the bandwidth ceiling. The tiled kernel restructures the same arithmetic into cooperative phases:</span>

<span style="font-size: 14px;">1. **Cooperative load**: each thread fills one element of `As` and one of `Bs` for the current $K$-tile, with coalesced global reads.</span>

<span style="font-size: 14px;">2. **`__syncthreads()`**: wait until the whole tile is resident in shared memory.</span>

<span style="font-size: 14px;">3. **Accumulate**: each thread runs a length-`TILE` inner loop over the shared tiles, adding `As[ty][k] * Bs[k][tx]` into its register, reusing each loaded element `TILE` times.</span>

<span style="font-size: 14px;">4. **`__syncthreads()`**: wait before the next phase overwrites the tiles, then advance the $K$-tile.</span>

<span style="font-size: 14px;">The quantified payoff: global input traffic drops by a factor of `TILE` (16x for a `16x16` tile), intensity rises from $0.25$ to about `TILE/4` FLOP/byte, and the kernel moves from memory-bound to compute-bound - the difference between a small fraction of peak FLOP/s and most of it.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take a `2x2` tile (`TILE = 2`) and a single tile phase over $K = 2$. The block has 4 threads computing a `2x2` output.</span>

* <span style="font-size: 14px;">**Load phase**: the 4 threads cooperatively bring `As = [[A00,A01],[A10,A11]]` and `Bs = [[B00,B01],[B10,B11]]` into shared memory, one element each, then `__syncthreads()`.</span>
* <span style="font-size: 14px;">**Reuse count**: `A00` is read by both output threads in row 0 (`C[0,0]` and `C[0,1]`), so the single global load of `A00` serves 2 multiply-adds - the `TILE = 2` reuse factor in miniature.</span>

<span style="font-size: 14px;">With $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$ and $B = \begin{bmatrix}5&6\\7&8\end{bmatrix}$, every input element is fetched from global memory once into the tile and then reused twice from shared memory, producing $C = \begin{bmatrix}19&22\\43&50\end{bmatrix}$. The naive kernel would have fetched each $A$ row and $B$ column from DRAM once per output cell that uses it.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Missing or misplaced `__syncthreads()`.** Both barriers per phase are mandatory; dropping the post-load sync reads stale tiles, dropping the post-compute sync corrupts the next tile, and placing either inside a divergent branch hangs the block.</span>
* <span style="font-size: 14px;">**Unpadded shared tiles.** A square `[TILE][TILE]` array makes column accesses a 32-way bank conflict that serializes the warp; pad to `[TILE][TILE+1]`.</span>
* <span style="font-size: 14px;">**Oversized tiles starving occupancy.** A large `TILE` raises reuse but inflates per-block shared memory, capping resident blocks and the warps needed to hide latency; balance intensity against occupancy.</span>
* <span style="font-size: 14px;">**Mishandling the $K$-tile remainder.** When $K$ is not a multiple of `TILE`, the last phase must zero-pad or bound-check the shared loads, or out-of-range global reads poison the accumulation.</span>

---