# Color Inversion — CUDA C++ and Numba Results

Challenge: invert an image in place, given as a 1-D array of RGBA bytes
(`width * height * 4` elements).  `out.rgb = 255 - in.rgb`; alpha must stay
unchanged.  Performance is measured with `width = 4096`, `height = 5120`
(an 80 MiB image).

Files:

- `color_inversion.cu` — CUDA C++ implementation of
  `solve(image, width, height)` (renamed from `starter.cu`).
- `color_inversion.py` — Numba CUDA port with the same `solve` signature.

## Implementation

Since `255 - c == c ^ 0xFF` for a byte, a whole pixel is inverted by XORing
its packed 32-bit value with `0x00FFFFFF` — alpha is the high byte and is
left untouched by the mask.  This makes the operation a pure stream:

- one thread loads a `uint4` (four pixels, 16 bytes), flips the four packed
  pixels, and stores it back — 128-bit load/store instead of the starter's
  six byte accesses per pixel;
- 256 threads per block, one `uint4` per thread (20,480 blocks at the
  benchmark size);
- the 16-byte path needs a 16-byte aligned base pointer, so 4-byte
  (one pixel per thread) and byte-per-pixel fallbacks cover unaligned
  inputs;
- when the pixel count is not a multiple of four, thread 0 of block 0
  handles the at-most-three leftover pixels in the same launch.

The starter was already correct — this is a bandwidth-bound problem, so the
win from vectorising is small in wall time (2.114 ms → 2.034 ms).  Both
versions are limited by DRAM: the image is 80 MiB, read once and written
once per call.

## Measurement environment

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy | 2.5.4 |
| Measurement date | 2026-11-08 |

Kernel times are medians of 30 runs (C++) / 20 runs (Numba) after a warm-up,
with the image already on the device, and include
`cudaDeviceSynchronize()`.

## Correctness

The operation is exact integer arithmetic, so every case below matches
byte-for-byte with zero mismatching bytes:

| Case | width | height | Pixels |
|---|---:|---:|---:|
| challenge example 1 | 1 | 2 | 2 |
| challenge example 2 | 2 | 1 | 2 |
| single pixel | 1 | 1 | 1 |
| odd pixels | 3 | 5 | 15 |
| 21 pixels | 7 | 3 | 21 |
| wide | 100 | 37 | 3,700 |
| row | 1024 | 1 | 1,024 |
| column | 1 | 1024 | 1,024 |
| prime-ish | 12345 | 1 | 12,345 |
| large | 257 | 131 | 33,667 |
| square | 512 | 512 | 262,144 |

The challenge examples produce the published outputs:

- example 1: `[255,0,128,255, 0,255,0,255]` → `[0,255,127,255, 255,0,255,255]`
- example 2: `[10,20,30,255, 100,150,200,255]` → `[245,235,225,255, 155,105,55,255]`

The CUDA C++ version was additionally tested with a base pointer offset by
one byte (exercising the byte fallback); it also matched exactly.  The Numba
port was tested with both host NumPy arrays (in-place) and Numba device
arrays.

## Performance — 4096 × 5120 (80 MiB image)

Traffic per call: the full image is read (80 MiB) and written (80 MiB),
i.e. 160 MiB of hardware traffic.  Usable RGB traffic is 140 MiB
(4 channels read, 3 written).

| Implementation | Median time | Hardware bandwidth | RGB bandwidth |
|---|---:|---:|---:|
| starter (byte accesses) | 2.114 ms | 79.4 GB/s | 69.4 GB/s |
| **CUDA C++ final** | **2.034 ms** | **82.5 GB/s** | **72.2 GB/s** |
| Numba CUDA | 2.102 ms | 79.8 GB/s | 69.8 GB/s |

For reference, the memory-bound kernels in the other challenges reach
~84 GB/s on this GPU (e.g. the vector-add and reduction results), so the
final kernel runs at ~98% of the observed practical DRAM ceiling.  At this
point the kernel is entirely memory-bound: the improvement over the starter
is limited to ~4%.

### Variants benchmarked (4096 × 5120)

| Variant | Time |
|---|---:|
| starter (3 byte loads + 3 byte stores per pixel) | 2.114 ms |
| `uint4` ×1, 128 threads | 2.028 ms |
| **`uint4` ×1, 256 threads (final)** | **2.034 ms** |
| `uint4` ×1, 512 threads | 2.033 ms |
| `uint4` ×2 per thread | 2.061 ms |
| `uint4` ×4 per thread | 2.135 ms |
| grid-stride loop (512 or 1024 blocks) | 2.05 ms |
| streaming loads/stores (`__ldcs`/`__stcs`) | 2.05 ms |

All sensible configurations land within a few percent of each other because
the DRAM bus, not instruction throughput, is the limit.  Streaming stores and
larger per-thread tiles do not help since there is no cache reuse to protect.

## Build / run

CUDA C++ (the file exports `solve`, so compile it with a harness that
provides `main`):

```bash
nvcc -O3 -arch=sm_61 -o test color_inversion.cu test_harness.cu
```

Numba:

```bash
python3 color_inversion.py
```

The Python script runs the correctness cases and the performance case.  For
best performance, pass a Numba device array; host arrays incur a copy to and
from the device.

Note: the challenge's stated constraint `width * height ≤ 8,388,608` is
smaller than the stated performance size `4096 × 5120 = 20,971,520` pixels.
The implementation handles both (and anything up to the `int` range used for
indexing).

## Naive baseline

`color_inversion_naive.cu` is the same computation written as plainly as possible:
one thread per pixel doing three byte loads and three byte stores — no shared-memory tiling, no vector types, no warp intrinsics and no
fast-math.  It passes the same correctness cases as the optimized kernel
(checked against a float64 CPU reference in the shared harness).

| Implementation | Time | vs naive |
|---|---:|---:|
| `color_inversion_naive.cu` | 2.070 ms | 1.00x |
| `color_inversion.cu` (optimized) | 2.048 ms | 1.01x |

Pure streaming: both forms move the same 160 MB, so the naive one is already at the wall.
