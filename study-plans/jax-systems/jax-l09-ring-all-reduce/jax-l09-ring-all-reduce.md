# Ring All-Reduce with ppermute

An all-reduce adds one array from every device and gives every device the total. It is the collective behind the data-parallel gradient average, and jax.lax.psum performs it in one call. This page opens up that call. It shows why the simple ways of summing across devices waste bandwidth, how passing small chunks around a ring keeps every link busy, and how the ring splits into a reduce-scatter phase and an all-gather phase.

## The simple ways and their cost

Suppose each of $n$ devices holds a vector of $C$ numbers. The most direct plan sends every vector to device 0, which adds them and sends the total back. Device 0 then receives $(n - 1) C$ numbers and sends another $(n - 1) C$, while the other devices mostly wait. The time grows with $n$, so adding devices makes the sum slower.

Another plan has every device send its whole vector to every other device, and each device adds what it receives. Now every device sends $(n - 1) C$ numbers. That is just as much traffic per device, spread across more links. Both plans move far more data than necessary, because they ship whole vectors when each device only needs to help with a part of the sum.

## Passing data along a ring

Arrange the devices in a ring, where device $d$ only ever sends to device $(d + 1) \bmod n$ and receives from device $(d - 1) \bmod n$. In one step, every device sends one message to its right neighbour at the same time, so every link in the ring carries data and none sits idle.

JAX expresses one such step with jax.lax.ppermute. Inside shard_map, ppermute takes a value, an axis name and a list of (source, destination) pairs, and every device receives the value sent by the device whose pair points at it. The pairs $(0, 1), (1, 2), \dots, (7, 0)$ describe a rotation by one position on an 8-device ring. ppermute moves data and never adds it, so the arithmetic is up to you.

## Reduce-scatter on three devices

Start with $n = 3$ devices holding the vectors [1, 2, 3], [10, 20, 30] and [100, 200, 300], whose sum is [111, 222, 333]. Cut each vector into $n = 3$ chunks, here one number each, and number the chunks 0, 1 and 2.

In step 0 every device sends the chunk whose number equals its own position. Device 1 receives chunk 0 from device 0, which is 1, and adds its own chunk 0, which gives 11. Device 2 receives chunk 1 from device 1, 20, and adds its own 200 to get 220. Device 0 receives chunk 2 from device 2, 300, and adds its own 3 to get 303.

In step 1 every device sends the running sum it just computed. Device 2 receives 11, adds its chunk 0 of 100 and gets 111. Device 0 receives 220, adds its chunk 1 of 2 and gets 222. Device 1 receives 303, adds its chunk 2 of 30 and gets 333.

After $n - 1 = 2$ steps, each device owns one chunk of the final answer. Device 0 holds chunk 1 (222), device 1 holds chunk 2 (333) and device 2 holds chunk 0 (111). Every chunk has visited every device exactly once and collected its contribution. This first phase is called reduce-scatter, because the reduced result ends up scattered across the devices, one chunk each.

## All-gather finishes the job

In the second phase nothing is added. Each device sends the finished chunk it owns to its right neighbour, and in the following steps it forwards whatever finished chunk it just received. In the three-device example, after one step device 0 holds the finished chunks 0 and 1, device 1 holds chunks 1 and 2, and device 2 holds chunks 2 and 0. After the second step every device holds [111, 222, 333].

This phase is called all-gather, because every device gathers every finished chunk. It also takes $n - 1$ steps, so the full all-reduce takes $2(n - 1)$ steps. For the 8 devices in this problem that is 7 steps of reduce-scatter and 7 steps of all-gather.

## Keeping track of which chunk moves

The bookkeeping follows a pattern you can read off the example. In reduce-scatter, the chunk a device sends at each step is the one it has just finished adding to, and the chunk it receives is the one its left neighbour was working on. Each step therefore shifts the chunk number by one relative to the device position, and after the last step device $d$ owns chunk $(d + 1) \bmod n$, as device 0 owned chunk 1 above.

In all-gather the shift continues, so the chunk arriving at each step is always one position further back around the ring. Inside shard_map, a device learns its own position with jax.lax.axis_index, which returns a traced integer. Chunk numbers computed from it are traced too, so selecting and updating chunks uses array indexing with traced indices and functional updates, never Python integer conversions.

## Why the ring uses the least bandwidth

In each of the $2(n - 1)$ steps, every device sends exactly one chunk of $C/n$ numbers. The total each device sends is

$$
2 (n - 1) \cdot \frac{C}{n} = 2 \, \frac{n - 1}{n} \, C.
$$

For a vector of one million numbers, that is 1,000,000 numbers per device with 2 devices, 1,500,000 with 4, 1,750,000 with 8 and 1,968,750 with 64. The amount stays below $2C$ however many devices join, and the work is spread evenly, so every link carries the same load. This matches the lower bound on the data each device must send in an all-reduce (Patarasuk and Yuan, 2009), which is why ring schedules and their close relatives sit underneath the collective libraries that training frameworks call.

The chunking is what makes this work. If devices passed whole vectors around the ring instead, each of the $n - 1$ steps would move $C$ numbers, for $(n - 1) C$ in total, which is 7 million numbers per device for the example with 8 devices.

## Exact sums and floating point

Floating point addition is not associative. In float32, $(10^8 + 1) - 10^8$ evaluates to 0, because $10^8 + 1$ rounds back to $10^8$, while $(10^8 - 10^8) + 1$ evaluates to 1. The ring adds the 8 contributions to each chunk in a different order for every chunk, and a library psum may use yet another order, so on general data the results can differ in the last bits.

When every input is a multiple of 0.5 with small magnitude, every partial sum is exactly representable in float32, and every order gives the same bits. That is why a ring built correctly matches psum exactly on such data. Real training tolerates the last-bit differences, but they explain why two all-reduce implementations rarely agree to every bit on random gradients.

## Common mistakes to avoid

- **Sending whole rows around the ring.** Each step then moves C numbers per device, which is 8 times more traffic than one chunk of C/8.
- **Running the wrong number of steps.** With 8 devices each phase needs exactly 7 steps, and one step fewer leaves every chunk missing one device's contribution.
- **Adding during the all-gather.** Chunks are already complete after reduce-scatter, so adding again counts some contributions twice.
- **Reading the chunk number with a Python int.** The device index from axis_index is traced, so converting it to a Python number fails under jit.
- **Mixing up sent and received chunk numbers.** The chunk a device adds to at a step is the one its left neighbour sent, which is one position behind the chunk the device itself sent.
- **Using a permutation that is not a ring step.** Pairs that skip devices or send two messages to the same device break the rotation that every chunk relies on.