"""Numba CUDA port of matrix_transpose.cu.

out[col, row] = in[row, col], both row-major float32.

The kernel uses the classic 32x32 shared-memory tile with 32x8 threads:
each thread loads/stores four elements, global accesses are coalesced along
32 consecutive threads, and the shared tile is padded to 32x33 to avoid
bank conflicts.

solve(input, output, rows, cols) keeps the challenge signature. It accepts
2-D or flat arrays and host NumPy arrays or Numba device arrays.
"""

import numpy as np
from numba import cuda, float32

TILE = 32
TILE_PAD = TILE + 1


@cuda.jit
def matrix_transpose_kernel(input, output, rows, cols):
    tile = cuda.shared.array((TILE, TILE_PAD), float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    in_col = cuda.blockIdx.x * TILE + tx
    if in_col < cols:
        for jj in range(4):
            j = jj * cuda.blockDim.y
            in_row = cuda.blockIdx.y * TILE + ty + j
            if in_row < rows:
                tile[ty + j, tx] = input[in_row, in_col]

    cuda.syncthreads()

    out_col = cuda.blockIdx.y * TILE + tx
    if out_col < rows:
        for jj in range(4):
            j = jj * cuda.blockDim.y
            out_row = cuda.blockIdx.x * TILE + ty + j
            if out_row < cols:
                output[out_row, out_col] = tile[tx, ty + j]


def _as_2d(array, rows, cols):
    if getattr(array, "ndim", 2) == 2:
        return array
    return array.reshape(rows, cols)


def solve(input, output, rows, cols):
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")

    in2 = _as_2d(input, rows, cols)
    out2 = _as_2d(output, cols, rows)

    threads = (32, 8)
    blocks = ((cols + TILE - 1) // TILE, (rows + TILE - 1) // TILE)
    matrix_transpose_kernel[blocks, threads](in2, out2, rows, cols)
    cuda.synchronize()


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    cases = ((2, 3), (3, 1), (17, 33), (513, 129), (129, 513))
    for rows, cols in cases:
        A = (np.arange(rows * cols, dtype=np.float32) * 0.25 - 7.0).reshape(
            rows, cols)
        out = np.zeros((cols, rows), dtype=np.float32)

        dA = cuda.to_device(A)
        dOut = cuda.to_device(out)
        solve(dA, dOut, rows, cols)
        got = dOut.copy_to_host()

        ref = A.T
        max_err = float(np.max(np.abs(got - ref)))
        print(f"rows={rows} cols={cols}: max_err={max_err:.3e} "
              f"{'OK' if max_err == 0.0 else 'FAIL'}")

    # Performance: 7000 x 6000, all ones, median of 20 runs.
    rows, cols = 7000, 6000
    n = rows * cols
    dA = cuda.to_device(np.ones((rows, cols), dtype=np.float32))
    dOut = cuda.device_array((cols, rows), dtype=np.float32)
    solve(dA, dOut, rows, cols)  # warm-up

    import statistics
    import time

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(dA, dOut, rows, cols)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)

    out = dOut.copy_to_host()
    ok = bool(np.all(out[::997] == 1.0))
    gbps = 2.0 * n * 4 / (ms * 1e6)
    print(f"Perf rows={rows} cols={cols}: {ms:.3f} ms, "
          f"~{gbps:.1f} GB/s, spot_check={'OK' if ok else 'FAIL'}")
