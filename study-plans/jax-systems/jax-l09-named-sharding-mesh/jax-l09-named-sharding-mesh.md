# Device Mesh and Named Sharding

A large model does not fit on one accelerator, and even when it fits, one accelerator is too slow to train it. JAX spreads work over many devices by cutting arrays into blocks and giving each device some of the blocks. This page starts from eight numbers on eight devices and builds up to the two objects that describe any layout in JAX, the device mesh and the named sharding. Every later problem in this section relies on them.

## One array, many devices

Start with the simplest possible case. You have a float32 vector with 8 values and a machine with 8 devices. You could keep the whole vector on device 0, or you could give each device one value. In the second layout, device $d$ holds element $d$, and the vector costs each device 4 bytes instead of 32.

That saving is the first reason to split arrays. The second reason is speed. If each device holds one eighth of a batch, each device can run the forward pass on its own eighth at the same time, and the whole batch finishes in roughly one eighth of the time.

JAX calls each piece a shard. A single global array is still one JAX array with one shape and one set of values. Its shards are the blocks that physically sit in the memory of each device, and JAX keeps a record of which device holds which block.

## Arranging devices in a grid

Real systems split arrays in more than one way at once. A training job might split the batch across some devices and split each weight matrix across others. To describe that, JAX arranges the devices into a grid called a mesh and gives every axis of the grid a name.

With 8 devices you can build a mesh of shape (8,), (2, 4), (4, 2) or (2, 2, 2), since the sizes must multiply to 8. A (2, 4) mesh with axis names data and model has 2 rows and 4 columns. Filling it in row-major order puts devices 0, 1, 2 and 3 in row 0 and devices 4, 5, 6 and 7 in row 1. Device 6 sits at row 1, column 2, because its row-major flat index is $1 \cdot 4 + 2 = 6$.

The names matter more than the shape. Every later instruction refers to the mesh by axis name, such as "split this dimension along model", so the same code works when the mesh grows from 8 to 512 devices. On real accelerators the order of devices in the grid follows the physical network, so that neighbours in the grid are also neighbours on the wires. On CPU devices there is no network, and the grid simply follows the device ids.

## Describing a layout with a partition spec

A partition spec says, for each dimension of an array, which mesh axes cut it. Write $s_k$ for the entry of dimension $k$. If $s_k$ is None, dimension $k$ stays whole. If $s_k$ names an axis of size $m$, dimension $k$ is cut into $m$ equal blocks, and the device at position $c$ along that axis receives block $c$.

Take an array of shape (2, 8) holding the numbers 0 to 15, row by row, on the (2, 4) mesh above, with the spec (None, model). The first dimension stays whole, and the second dimension of size 8 is cut into 4 blocks of 2 columns, because the model axis has 4 devices. Device 1 sits in column 1 of the mesh, so it receives columns 2 and 3, which hold [[2, 3], [10, 11]]. Its block has shape (2, 2) and starts at index (0, 2).

In general, if dimension $k$ has size $n_k$ and is cut into $m_k$ blocks, every block has size $n_k / m_k$ along that dimension, and block number $c$ starts at index

$$
\text{start}_k = c \cdot \frac{n_k}{m_k}.
$$

The sizes must divide evenly, which is why the batch size and the hidden width in sharded models are usually multiples of the mesh axis sizes.

## Axes that are not named hold copies

The spec (None, model) never mentions the data axis. Devices that sit in the same mesh column but in different rows, such as device 1 and device 5, therefore receive exactly the same block. Running this layout in JAX confirms it, since device 5 also holds [[2, 3], [10, 11]] starting at index (0, 2).

This is called replication along the data axis. It costs memory, because the global array of 64 bytes is stored as 8 blocks of 16 bytes, which is 128 bytes in total, two full copies. Replication is often exactly what you want. Model weights in data parallelism are replicated on every device so that each device can run the full model on its slice of the batch.

A spec with no entries at all replicates the array on every device, and each block is the full array starting at index 0 in every dimension. A spec with fewer entries than the array has dimensions leaves the remaining dimensions whole, so (data) on a 2-D array means the same as (data, None).

## Splitting one dimension over two axes

Sometimes one dimension should be cut across several mesh axes at once. A vector of 8 values on the (4, 2) mesh with names data and model can be cut into 8 single-element blocks if its one dimension is split over both axes. The spec entry is then a tuple of names, and the number of blocks is the product of their sizes, $4 \cdot 2 = 8$.

The order inside the tuple decides which device gets which block. Block numbers count over the named axes in row-major order, with the first name changing slowest, just like digits in a number. Write $c_{\text{data}}$ and $c_{\text{model}}$ for a device's coordinates in the mesh.

- **The tuple (data, model).** The block number is $2 \, c_{\text{data}} + c_{\text{model}}$. In this mesh that equals the device id, so device $d$ holds element $d$.
- **The tuple (model, data).** The block number is $4 \, c_{\text{model}} + c_{\text{data}}$. Device 1 sits at data 0 and model 1, so it holds block 4, and device 2 sits at data 1 and model 0, so it holds block 1.

Running the second layout gives the devices 0 to 7 the elements 0, 4, 1, 5, 2, 6, 3 and 7. Both layouts give every device one value, and they differ only in which value goes where. That difference matters as soon as devices exchange data, because the devices that hold neighbouring blocks are the ones that talk to each other.

## Reading the layout back

You rarely compute layouts by hand. A placed JAX array carries a sharding object, and the sharding knows exactly where every block lives. Each array also exposes its addressable shards, one per device, and every shard holds three things, the device, its index into the global array and the block of data itself.

The index is a tuple with one Python slice per dimension. For the (4, 6) array split by (data) on the (2, 4) mesh, device 5 holds slice(2, 4) along the first dimension and slice(None, None) along the second. A slice of None to None covers the whole dimension, so its start is 0 and its stop is the dimension size, which is what the slice method indices returns when you pass it the size.

The sharding offers the same information for any shape through devices_indices_map, which maps every device to its index tuple without needing the data. Both views agree, and both are the ground truth that compiled code and collectives rely on.

## Why the layout matters for later computation

When a function is compiled with jax.jit, the compiler looks at the sharding of every input and plans the computation around it. An elementwise operation on a sharded array runs on every device's block with no communication at all. A sum over a sharded dimension needs the devices to exchange partial results, and the compiler inserts that communication for you.

The rest of this section takes that planning into your own hands. With jax.shard_map you write the code that runs on a single block, and you call the collectives yourself, such as psum to add partial results or ppermute to pass blocks between neighbours. The mesh and the partition spec are the language those functions use, so the exact meaning of every spec entry carries through all of them.

## Common mistakes to avoid

- **Mixing up mesh axes and array dimensions.** A spec entry belongs to an array dimension and names a mesh axis, so (None, model) leaves dimension 0 whole and cuts dimension 1 along the model axis.
- **Building the mesh from a reordered device list.** The tables are indexed by device id, so a mesh that does not hold the devices in id order puts every block on the wrong row.
- **Reporting the global shape as the shard shape.** Every device holds a block whose size along a split dimension is the dimension size divided by the number of blocks.
- **Reporting block numbers instead of start indices.** Block 2 of a dimension cut into blocks of size 3 starts at index 6, which is the number the layout table needs.
- **Forgetting that unnamed axes replicate.** Two devices that differ only along an axis missing from the spec hold identical blocks and the same start indices.
- **Swapping the order inside a tuple entry.** The first name in a tuple changes slowest, so (model, data) and (data, model) send the same blocks to different devices.
- **Leaving the array where it was.** Computing the tables without calling jax.device_put gives the right numbers while the array itself still sits on one device.