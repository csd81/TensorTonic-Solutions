# Shared-Memory Bank Conflict Analyzer

Shared memory is the fast on-chip scratchpad that threads in a block use to exchange data, and it is split into banks that can each serve one request per cycle. When several threads in a warp ask the same bank for different addresses, the requests are serialized. The heart of this problem is mapping each address to its bank and counting how many distinct addresses each bank has to serve.

## Warps, lanes and banks

A warp is a group of 32 threads that issue a memory instruction together, and a lane is one thread's position, 0 to 31, within its warp. The active lanes are the ones taking part in a given access.

To feed all those lanes in a single cycle, shared memory is divided into equal-sized banks that work in parallel, usually 32 banks that are each 4 bytes wide. Consecutive 4-byte words live in consecutive banks, and the pattern wraps around after the last bank. If the lanes of a warp touch different banks, every request is served at once.

If two or more lanes need different addresses in the same bank, that bank serves them one after another. A conflict where $n$ distinct addresses hit one bank makes the access take about $n$ times as long. Kernels such as matrix transposes and tiled matrix multiplication can lose most of their shared-memory speed to conflicts.

## Mapping an address to a bank

For lane $i$ with byte address $a_i$, bank width $w$ bytes and $B$ banks, the bank is

$$
b_i = \left\lfloor \frac{a_i}{w} \right\rfloor \bmod B
$$

Dividing by the bank width gives the word index, and taking it modulo the number of banks wraps it onto the banks. With 32 banks of 4 bytes, addresses 8, 136 and 264 all land in bank 2, because 32 banks of 4 bytes cover 128 bytes before the mapping wraps back to bank 0.

## Identical addresses are fetched once

A bank's cost for one access is the number of distinct addresses it has to fetch, and every lane mapped to that bank waits that long. If several lanes read the very same address, the hardware fetches it once and broadcasts it to all of them, so identical addresses add nothing to the cost.

A degree of 1 means no conflict. A degree of 4 means that bank needs four separate accesses, and the whole warp waits for the slowest bank.

Real hardware broadcasts at word granularity, so two different byte addresses inside the same 4-byte word are also served together. This problem uses the simpler rule of counting distinct addresses.

## A small mixed example

Take five lanes with byte addresses $[12, 140, 140, 16, 268]$, 32 banks and 4-byte banks. The word indices are 3, 35, 35, 4 and 67, so the banks are

$$
[3, \ 3, \ 3, \ 4, \ 3]
$$

Bank 3 is asked for addresses 12, 140 and 268. Lanes 1 and 2 share address 140, which is fetched once, so bank 3 has 3 distinct addresses. Bank 4 serves only address 16. The conflict degrees in lane order are $[3, 3, 3, 1, 3]$, and this access would take about three times as long as a conflict-free one.

## The padding trick

A classic case is reading a column of a $32 \times 32$ tile of floats stored row by row. Each row is 128 bytes long, so every element of a column sits in the same bank, and a warp reading one column has a 32-way conflict.

Adding one unused float to the end of every row makes each row 132 bytes long. Element $i$ of column $c$ is then at word $33i + c$, and $(33i + c) \bmod 32 = (i + c) \bmod 32$, so the 32 lanes hit 32 different banks at the cost of one unused word per row.

## Where conflicts show up in LLM kernels

Attention and matrix multiplication kernels load tiles of their inputs into shared memory and then read them along both rows and columns, so at least one direction is at risk of conflicts. Besides padding, many high-performance kernels use swizzling, which permutes the column position of each row with a cheap bit operation so that column reads spread across banks without wasting any memory. Checking bank IDs and conflict degrees for an access pattern, as in this problem, is how you confirm that a layout avoids serialization.

## Common mistakes to avoid

- **Using the byte address directly.** Divide by the bank width first, then take the result modulo the number of banks.
- **Counting lanes instead of distinct addresses.** Lanes reading the same address share one broadcast and add one to their bank's degree.
- **Forgetting the bank width.** With 8-byte banks, addresses 0 and 4 share a word and a bank.
- **Reporting only the worst bank.** Each lane gets the degree of its own bank, in the original lane order.
- **Forgetting the wraparound.** Addresses one full row of banks apart, 128 bytes with the default sizes, land in the same bank.
