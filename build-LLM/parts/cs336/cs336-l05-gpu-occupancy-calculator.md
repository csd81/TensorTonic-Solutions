# GPU Occupancy Calculator

A GPU runs a kernel as many thread blocks, and each streaming multiprocessor (SM) can hold several blocks at once. How many fit depends on which on-chip resource runs out first. The heart of this problem is checking every per-SM limit, finding the tightest one, and turning the result into occupancy, the fraction of the SM's warp slots that are in use.

## Warps, blocks and SMs

Threads on an NVIDIA GPU execute in groups called warps, normally 32 threads that issue the same instruction together. A thread block is a group of warps that runs on a single SM and can share fast on-chip memory. A GPU has dozens to more than a hundred SMs.

A block is resident when its threads, registers and shared memory are allocated on an SM, so its warps can be scheduled. Registers are the per-thread variables stored in the SM's register file, a fixed pool shared by every resident thread. Each SM keeps several blocks resident at the same time and switches between their warps every cycle.

With warp size $W$, the number of warps a block of $T$ threads needs is

$$
w = \left\lceil \frac{T}{W} \right\rceil
$$

A block of 70 threads with 32-thread warps uses 3 warps, and the last warp has only 6 active threads. The hardware still reserves a full warp, so the block consumes $wW = 96$ thread slots and the registers for 96 threads.

## The scarcest resource sets the block count

Every resident block consumes resources from a fixed per-SM budget, and all five limits must hold at once.

- **Threads.** Each block uses $wW$ thread slots out of the SM's thread limit.
- **Warps.** Each block uses $w$ warp slots out of the SM's warp limit $W_{\max}$.
- **Blocks.** The SM can track only a limited number of blocks, however small they are.
- **Registers.** A block uses the registers per thread times $wW$.
- **Shared memory.** Each block uses its declared bytes of shared memory.

For each resource, the number of whole blocks that fit is that resource's budget divided by one block's use, rounded down. The SM holds the smallest of the five counts, so a single scarce resource can cap the whole SM even when the others have plenty of room.

Real GPUs also round register and shared-memory allocations up to fixed granularities. This problem leaves that out and uses the simple model above.

## From blocks to occupancy

If $b$ blocks fit, the SM holds $bw$ resident warps. Occupancy compares that to the maximum number of warps the SM supports:

$$
\text{occupancy} = \frac{bw}{W_{\max}}
$$

An occupancy of 1 means every warp slot is filled.

## Why occupancy matters

A load from global memory takes hundreds of cycles. While one warp waits, the SM's scheduler switches to another warp that is ready to run, and switching costs nothing because every resident warp keeps its registers on chip. More resident warps give the scheduler more chances to find ready work, which hides memory latency.

Higher occupancy is not automatically better, though. A kernel that uses many registers per thread can keep more data close to the arithmetic units and may run faster at lower occupancy. Working out occupancy shows which resource limits the kernel, so you can decide whether to trade registers or shared memory for more resident warps. Attention and matrix multiplication kernels make exactly this trade when they choose how large a tile of data each block keeps in shared memory.

## A register-limited example

Take blocks of 200 threads, 40 registers per thread and 12,288 bytes of shared memory per block. The SM allows 2,048 threads, 64 warps, 32 blocks, 65,536 registers and 102,400 bytes of shared memory.

Each block needs $\lceil 200 / 32 \rceil = 7$ warps, so it counts as 224 threads and uses $40 \times 224 = 8{,}960$ registers. The five limits give

$$
\left\lfloor \tfrac{2048}{224} \right\rfloor = 9, \quad \left\lfloor \tfrac{64}{7} \right\rfloor = 9, \quad 32, \quad \left\lfloor \tfrac{65536}{8960} \right\rfloor = 7, \quad \left\lfloor \tfrac{102400}{12288} \right\rfloor = 8
$$

Registers are the tightest limit, so 7 blocks fit. That is $7 \times 7 = 49$ resident warps and an occupancy of $49 / 64 = 0.765625$. Cutting register use to 32 per thread would raise the register limit to 9 blocks, and shared memory would then become the bottleneck at 8.

## Common mistakes to avoid

- **Treating a partial warp as a fraction of a warp.** A block of 70 threads still occupies 3 full warps and 96 thread slots.
- **Using the raw thread count for registers.** Register use is per allocated thread, including the unused lanes of the last warp.
- **Skipping a limit.** The block count is the minimum over threads, warps, blocks, registers and shared memory.
- **Dividing by zero use.** Zero registers or zero shared memory places no limit on the block count.
- **Rounding block counts up.** Only whole blocks can be resident, so every division rounds down.
- **Dividing occupancy by the thread limit.** Occupancy is resident warps over the SM's maximum warps.
