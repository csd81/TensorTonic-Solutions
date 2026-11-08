# <span style="font-size: 20px;">Cross-Entropy Loss</span>

<span style="font-size: 14px;">Cross-entropy turns a batch of class logits into a single scalar loss: for each row it computes a stable log-softmax, picks out the target class's negative log-probability, and averages over the batch. It is the canonical **row-parallel reduction**: one block owns one row, collapses that row's $C$ logits with two passes in `__shared__` memory, and contributes its scalar loss across rows. Unlike a map, no single thread sees a whole row, so the work must collapse in stages, and the cross-row combine forces an **atomic** accumulation.</span>

---

## The Operation, $L = -\frac{1}{B}\sum_b \log \text{softmax}(x_b)[t_b]$

<span style="font-size: 14px;">Given logits $X$ of shape $(B, C)$ and integer targets $t$ of shape $(B,)$, the per-row loss is the negative log-probability of the target class:</span>

$$
L = -\frac{1}{B} \sum_{b=0}^{B-1} \left( x_{b, t_b} - \max_c x_{b,c} - \log \sum_{c} e^{\,x_{b,c} - \max_c x_{b,c}} \right)
$$

<span style="font-size: 14px;">$X$ is a contiguous row-major buffer, so row $b$ occupies the $C$ consecutive elements starting at offset $b \cdot C$. The max-subtract inside the exponent is the **log-sum-exp** trick: it is mathematically identity but keeps every `expf` argument $\le 0$, so no term overflows. The output is a single scalar.</span>

---

## <span style="font-size: 16px;">Parallelization Strategy</span>

<span style="font-size: 14px;">The decomposition is **one block per row**. The grid is sized to $B$ blocks; the threads of a block cooperate to reduce that block's $C$ logits. A logit is addressed as `X[blockIdx.x * C + threadIdx.x]`, and when $C$ exceeds `blockDim.x` each thread strides over several logits in a grid-stride pattern within the row. A block size of 256 keeps full warps and high occupancy while comfortably covering typical vocabulary or class counts.</span>

<span style="font-size: 14px;">A row's loss requires two reductions that cannot be fused into one pass: first the **maximum** over $C$ for stability, then the **sum of exponentials** that depends on that maximum. Each reduction is a $\log_2 C$-depth tree in `__shared__` memory, separated by `__syncthreads()` so every lane sees the finished max before it exponentiates. After the two reductions the block holds the row's log-sum-exp; the single thread that owns the target index reads `X[b*C + t_b]` and forms that row's negative log-probability.</span>

<span style="font-size: 14px;">This two-level structure - cooperate within a block, then combine across blocks - is the universal shape of a parallel reduction. The within-row work is the expensive part and happens entirely in fast on-chip memory; the across-row combine is a single scalar per block. Mapping one row to one block, rather than splitting a row across blocks, keeps every reduction inside one block's shared memory so no inter-block synchronization is needed mid-row. Inter-block synchronization on the GPU effectively means a separate kernel launch, so confining a row to a block is what makes a single launch sufficient.</span>

---

## <span style="font-size: 16px;">Memory Hierarchy and Access Pattern</span>

<span style="font-size: 14px;">Each logit is read from global memory, and consecutive threads in a warp read consecutive logits within the row, so the loads are **coalesced**. The two passes both read the same $C$ logits; a kernel that keeps the row resident in `__shared__` memory between the max pass and the exp pass reads global once instead of twice, halving DRAM traffic for the row. Partial maxima and partial sums live in shared memory during the tree reductions - roughly an order of magnitude lower latency than global - and the final per-row scalar lives in a register before the atomic.</span>

<span style="font-size: 14px;">The targets buffer is tiny: one integer per row, read once by the lane that needs it. There is no reuse beyond the row itself, which is exactly why shared memory caches the row but nothing larger. The shared-memory footprint - one row of $C$ values plus a small scratch array for the tree - sets the **occupancy** ceiling: a very large $C$ can shrink the number of blocks resident per SM, which is the tradeoff behind caching the row versus rereading it from global memory.</span>

---

## <span style="font-size: 16px;">Atomic Accumulation Across Rows</span>

<span style="font-size: 14px;">Each block produces one scalar, and the loss is the sum of those scalars over all $B$ rows. Blocks run concurrently and in unspecified order, so they cannot write to a shared accumulator without serializing the read-modify-write. The standard pattern is one **`atomicAdd`** per block into a single global accumulator: the block does all $C$-element work cooperatively, reduces to one number, and only then issues a single atomic. With one atomic per block rather than per element, contention is $B$ writers on one address - cheap relative to the per-row reduction - and the final divide by $B$ is applied once, either in a trivial trailing kernel or on the host.</span>

<span style="font-size: 14px;">Pushing the atomic down to per-thread granularity would be a classic mistake: $B \cdot C$ atomics on one address serialize the entire kernel. The block-local reduction then single atomic is what keeps the cross-row combine off the critical path.</span>

---

## <span style="font-size: 16px;">Memory-Bound or Compute-Bound?</span>

<span style="font-size: 14px;">Per logit the kernel moves 4 bytes and performs a small handful of operations: a comparison in the max pass and an `expf` plus an add in the sum pass. Counting the transcendental generously, that is a few FLOPs per 4 bytes if the row is read twice, or per 4 bytes once if it is cached in shared memory:</span>

$$
\text{intensity} \approx \frac{\text{a few FLOP}}{4 \text{ bytes}}
$$

<span style="font-size: 14px;">That sits well below the **roofline** ridge point, so the kernel is **memory-bound**, with the `expf` units idle most of the time waiting on DRAM. The transcendental nudges intensity slightly toward compute versus a plain sum, but at batch scale bandwidth still dominates. The optimization that matters most is reading each row once - keeping it in `__shared__` memory across the two passes - not making the arithmetic faster.</span>

<span style="font-size: 14px;">The classification also explains why fusing the two reduction passes matters. Each extra global pass over $X$ adds $4BC$ bytes of traffic, and at $0.5$ to $1$ FLOP per byte the runtime tracks bytes moved almost exactly. Cutting from three row reads to one is therefore close to a three-times speedup, far larger than anything a faster `expf` could deliver. On a memory-bound kernel, traffic reduction is the only optimization that moves the needle.</span>

---

## <span style="font-size: 16px;">Naive vs Optimized</span>

<span style="font-size: 14px;">A naive kernel reads the row from global memory three separate times - once for the max, once for the sum, once to fetch the target logit - and reduces with a shared-memory tree that idles half its lanes at every step. A faster version makes three changes:</span>

<span style="font-size: 14px;">1. **Single global read**: stage the row into `__shared__` memory once, then run both reductions and the target lookup against shared memory, cutting global traffic by up to three times.</span>

<span style="font-size: 14px;">2. **Warp-shuffle reduction**: replace the shared-memory tree's last stages with `__shfl_down_sync`, collapsing the final 32 values register-to-register with no shared-memory round trips and no bank conflicts.</span>

<span style="font-size: 14px;">3. **One atomic per block**: reduce fully in-block, then issue a single `atomicAdd`, keeping contention at $B$ writers instead of $B \cdot C$.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $B = 2$, $C = 4$, targets $t = [2, 0]$, and consider row 0 with logits $[1, 3, 2, 0]$. The block reduces $C = 4$ values in two `__syncthreads()`-separated steps.</span>

* <span style="font-size: 14px;">**Max pass**: pairs reduce $(1,3)\to 3$ and $(2,0)\to 2$, then $(3,2)\to 3$. The row max is $3$.</span>
* <span style="font-size: 14px;">**Shifted exponentials**: $e^{1-3}, e^{3-3}, e^{2-3}, e^{0-3}$, i.e. about $0.135, 1, 0.368, 0.050$, summing to $\approx 1.553$, so $\log \sum \approx 0.440$.</span>
* <span style="font-size: 14px;">**Target term**: the target is class $2$, so the loss for row 0 is $-(x_{0,2} - 3 - 0.440) = -(2 - 3 - 0.440) = 1.440$.</span>

<span style="font-size: 14px;">Block 0 issues one `atomicAdd(acc, 1.440)`; block 1 does the same for its row. After both blocks finish, a final step divides `acc` by $B = 2$ to produce the mean loss. The arithmetic is verifiable by hand; the systems lesson is that two synchronized reductions plus one atomic per block produce the scalar with each logit read once.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Skipping the max-subtract.** Exponentiating raw logits overflows `expf` for large values and produces `inf`/`nan`; the log-sum-exp shift is a correctness requirement, not an optimization.</span>
* <span style="font-size: 14px;">**Per-element atomics.** Issuing `atomicAdd` per logit serializes $B \cdot C$ writers on one address; reduce in-block first and emit one atomic per block.</span>
* <span style="font-size: 14px;">**Missing `__syncthreads()` between passes.** Reading the shared-memory max before all lanes have written it is a race that yields nondeterministic wrong losses.</span>
* <span style="font-size: 14px;">**Float accumulation order.** The atomic sum over rows combines in nondeterministic order, so the result is not bit-identical to a serial reference; this is why the test uses a tolerance.</span>

---