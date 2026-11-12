# Global-Memory Coalescing Counter

A GPU reads global memory in whole aligned chunks called cache lines, and when the threads of a warp read nearby addresses, their requests are combined into a few line transfers. This problem centres on working out which lines a group of accesses touches and how much of the transferred data was actually requested.

## Warps, lanes and transactions

GPU threads run in groups of 32 called warps, and each thread in a warp is called a lane. The lanes of a warp issue their memory requests together, and the memory system serves them with transactions, where one transaction moves one cache line, commonly 128 bytes.

Global memory is the GPU's large main memory, and reading it is slow compared with arithmetic. Many kernels in a language model, where a kernel is a function that runs on the GPU, spend much of their time waiting for memory rather than computing. Elementwise operations, normalization and attention during token-by-token decoding are typical examples. For these kernels, the number of bytes moved sets the speed.

If the lanes read consecutive addresses that fit in one aligned line, a single transaction serves them all. This is called a coalesced access. If the lanes read addresses far apart, each lane may need its own line, and in the simple line-based model used here the warp moves many full lines to deliver a few useful bytes. Real GPUs split each line into smaller sectors, and this problem uses whole lines to keep the counting simple.

## Which lines an access touches

Each lane requests $W$ consecutive bytes starting at address $a_i$. Integer division by the line size $L$ tells you which $L$-byte bucket an address falls in, so the first and last bytes fall in lines

$$
s_i = \left\lfloor \frac{a_i}{L} \right\rfloor, \qquad e_i = \left\lfloor \frac{a_i + W - 1}{L} \right\rfloor
$$

The last byte is at $a_i + W - 1$, because the $W$ bytes start at $a_i$ itself and end one address before $a_i + W$.

The lane touches every line from $s_i$ to $e_i$. Often $s_i = e_i$, but an access that crosses a line boundary touches every line in between. That is two lines for a small access that starts near the end of a line, and can be more when $W$ is larger than $L$. Lines shared by several lanes are fetched once, so the transaction count is the number of distinct lines touched by any lane.

## Measuring wasted bandwidth

Transactions alone do not show efficiency, because a line can be fetched to serve only a few useful bytes. The useful byte fraction compares the distinct requested bytes $U$ with the bytes actually moved, $C$ lines of $L$ bytes each:

$$
\text{useful byte fraction} = \frac{U}{CL}
$$

A fraction of 1 means every transferred byte was requested. If two lanes request overlapping bytes, the shared bytes count once in $U$, since they are transferred once.

## A strided example

Take 32 lanes that each read 4 bytes, with a stride of 8 bytes between lanes, starting at address 256. The addresses are $256, 264, 272, \ldots, 504$. This pattern appears when a kernel reads one field of an array of two-float structures.

The requested bytes span 256 to 507, so they touch lines 2 and 3 of 128 bytes each. The warp needs 2 transactions and moves 256 bytes, but only $32 \times 4 = 128$ distinct bytes were requested. The useful byte fraction is

$$
\frac{128}{2 \times 128} = 0.5
$$

Half of the bandwidth is wasted on the second float of each structure. Storing the field in its own contiguous array would bring this back to one transaction and a fraction of 1.

Alignment matters too. If 32 lanes read consecutive floats starting at address 448, their bytes run from 448 to 575 and straddle lines 3 and 4. That also takes 2 transactions with a fraction of 0.5, even though the access is perfectly contiguous.

## How this connects to kernel design

Coalescing is why tensor layout matters so much on GPUs. Kernels are written so that consecutive threads handle consecutive elements along the contiguous dimension. A transpose of a large matrix is often done through shared memory, a small fast on-chip buffer that the threads of a block share, so that both the read from and the write to global memory stay coalesced. Counting transactions for an access pattern, as in this problem, is a quick way to predict whether a kernel will run near memory bandwidth or far below it.

## Common mistakes to avoid

- **Using the address after the last byte.** The last touched byte is $a_i + W - 1$, so an access ending exactly on a line boundary stays in one line.
- **Assuming a crossing access touches exactly two lines.** A wide access can span several boundaries and touches every line from $s_i$ to $e_i$.
- **Counting a shared line twice.** Each distinct line costs one transaction, however many lanes use it.
- **Counting overlapping bytes twice.** Distinct requested bytes are counted once in the useful fraction.
- **Dividing by the requested bytes.** The denominator is the bytes actually moved, $C$ lines of $L$ bytes, which is never smaller than the bytes requested.
