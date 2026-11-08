# <span style="font-size: 20px;">Cosine Similarity</span>

<span style="font-size: 14px;">Cosine similarity measures the angle between two vectors as their dot product divided by the product of their magnitudes. The systems-interesting structure is that it needs three separate sums - the dot product and the two squared norms - and the optimized kernel fuses all three into a single pass. This is a **multi-accumulator reduction**: each thread carries three running totals at once, and one combine step collapses all three together, so the inputs are read exactly once instead of three times. The arithmetic is the same either way; what changes is how many times the two input arrays cross the memory bus.</span>

---

## <span style="font-size: 16px;">The Operation</span>

<span style="font-size: 14px;">For two inputs $A$ and $B$ of length $N$, the kernel produces one float:</span>

$$
\text{result} = \frac{\sum_i A[i] B[i]}{\sqrt{\sum_i A[i]^2}\,\sqrt{\sum_i B[i]^2}}
$$

<span style="font-size: 14px;">Both inputs are contiguous row-major buffers of $N$ 32-bit floats in device global memory; the output is a single float. Three reductions hide inside that one expression: $\sum A_i B_i$, $\sum A_i^2$, and $\sum B_i^2$. The key decision is whether to compute them in three passes or one.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">Each thread owns one index. It loads `A[idx]` and `B[idx]` once with `idx = blockIdx.x * blockDim.x + threadIdx.x`, then from those two register values computes all three contributions: the product `A[idx]*B[idx]`, the square `A[idx]*A[idx]`, and the square `B[idx]*B[idx]`. Each feeds a separate **tree reduction** running in parallel through the same $\log_2 N$ steps. This is the multi-accumulator pattern: three independent sums share one traversal of the data because they all need the same two loaded values.</span>

<span style="font-size: 14px;">A block size of 256 is conventional - a multiple of the 32-lane **warp**, enough warps per **SM (Streaming Multiprocessor)** for **occupancy**. With 256 threads per block the grid needs $\lceil N / 256 \rceil$ blocks. Tail threads with `idx >= N` contribute zero to all three accumulators, the additive identity, so the bounds check resolves to a neutral element rather than reading out of bounds.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">The fused kernel reads each of the two inputs exactly once - 8 bytes per element - and the three squared and crossed terms are derived from the two values already sitting in registers. Both streams are read at consecutive `idx` across a warp, so each is fully **coalesced**. The reuse story is the heart of the kernel: once `A[idx]` and `B[idx]` are in registers, three products come for free with no extra DRAM traffic.</span>

<span style="font-size: 14px;">The three partial sums live in `__shared__` memory during the in-block tree, requiring three `__shared__` arrays (or one array of triples), with `__syncthreads()` between levels. That triples the per-block shared-memory footprint relative to a single reduction, which slightly tightens the occupancy ceiling, but it is the price of fusing three sums into one traversal. The tree pairs `s` with `s + stride` and halves the stride to keep surviving lanes contiguous and avoid **bank conflicts**.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per element the fused kernel moves 8 bytes (two loads) and performs three multiplies and three adds, six FLOPs. Its **arithmetic intensity** is:</span>

$$
\frac{6 \text{ FLOP}}{8 \text{ bytes}} = 0.75 \text{ FLOP/byte}
$$

<span style="font-size: 14px;">Higher than a plain dot product, but still far below the **roofline** ridge point of tens of FLOPs per byte, so cosine similarity remains **memory-bound**. The three multiply-adds are nearly free; runtime is set by how fast the two input streams arrive from DRAM. This is exactly why fusion wins: it does not lower the FLOP count, it lowers the byte count, and bytes are the bottleneck.</span>

---

## <span style="font-size: 16px;">One Pass vs Three Passes</span>

<span style="font-size: 14px;">The naive structure runs three separate reduction kernels - one for the dot product, one for each squared norm - each reading its inputs from DRAM. The dot kernel reads both arrays, and each norm kernel reads one array, so the inputs are streamed a total of four times across the three launches. The fused kernel reads each array once, cutting input traffic by roughly $4\times$ and replacing three kernel launches with one.</span>

<span style="font-size: 14px;">Because the kernel is memory-bound, that traffic reduction translates almost directly into a speedup; the extra arithmetic from carrying three accumulators is hidden under the memory latency the SMs were already waiting on. The three-pass version also pays three kernel-launch overheads and three full reduction trees, where the fused version pays one of each. The lesson generalizes: whenever several reductions share the same input loads, fuse them into one multi-accumulator pass, because the cost is registers and shared memory (cheap and plentiful relative to the work) while the saving is DRAM traffic (the actual bottleneck).</span>

---

## <span style="font-size: 16px;">Cross-Block Combine and the Final Scalar</span>

<span style="font-size: 14px;">A kernel launch cannot synchronize across blocks, so each block's three partials must be combined separately. A **two-kernel combine** writes a triple per block to scratch and a second launch sums the triples; alternatively three `atomicAdd` calls per block accumulate into three global scalars, keeping contention to `gridDim.x` writers each instead of $N$. Only after all three global sums exist does a final step compute `dot * rsqrtf(normA) * rsqrtf(normB)` - using `rsqrtf` to turn the two square-root divisions into cheap multiplies on the single final scalar.</span>

---

## <span style="font-size: 16px;">Hardware Utilization and Latency Hiding</span>

<span style="font-size: 14px;">The load phase behaves like a streaming map with two input streams; the GPU hides hundreds of cycles of DRAM latency through **massive multithreading**, switching to another resident warp whenever one stalls. The two loads per thread are independent, so the second is in flight behind the first. There is no **warp divergence** in the body - every active lane runs the same three multiply-adds - and the only branch is the boundary bounds check. The reduction has the usual short tail below one warp, handled by warp-shuffle.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $A = [1, 0]$, $B = [1, 1]$, $N = 2$, in one block. Each lane loads its pair once and produces three contributions.</span>

* <span style="font-size: 14px;">**Lane 0:** product $1\cdot1 = 1$, $A^2 = 1$, $B^2 = 1$.</span>
* <span style="font-size: 14px;">**Lane 1:** product $0\cdot1 = 0$, $A^2 = 0$, $B^2 = 1$.</span>
* <span style="font-size: 14px;">**Combine (stride 1):** the three trees add in lockstep: dot $= 1$, $\sum A^2 = 1$, $\sum B^2 = 2$.</span>

<span style="font-size: 14px;">The final scalar is $1 / (\sqrt{1}\cdot\sqrt{2}) = 1/\sqrt{2} \approx 0.707$. All three sums came from one traversal: lane 0 read its pair exactly once and produced all three of its contributions from registers.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Running three separate reduction passes.** That re-reads the inputs from DRAM up to four times on a memory-bound kernel; carry three accumulators and fuse into one pass.</span>
* <span style="font-size: 14px;">**Re-loading from global for the squares.** Computing `A[idx]*A[idx]` with a second global load instead of reusing the register doubles traffic; load once, derive all three terms.</span>
* <span style="font-size: 14px;">**Atomic contention across three accumulators.** Per-thread atomics on the three global scalars serialize; reduce within the block first, then one atomic per block per accumulator.</span>
* <span style="font-size: 14px;">**Division by a zero norm.** A zero-magnitude input makes the denominator zero; guard with `rsqrtf` handling or accept the defined edge-case behavior rather than emitting NaNs.</span>

---