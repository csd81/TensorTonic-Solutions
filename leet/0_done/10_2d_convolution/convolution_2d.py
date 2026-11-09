"""Numba CUDA port of convolution_2d.cu.

2D "valid" convolution:
    out[i][j] = sum_{m<KR, n<KC} in[i+m][j+n] * k[m][n].

Same mapping as the CUDA C++ version: a block of 16x16 threads produces a
64x32 tile of outputs, each thread owning 2 rows x 4 columns.  The input tile
(with halo) and the kernel (rows padded to a multiple of four floats, zero
filled) are staged in shared memory, and each thread keeps its 2x4 input
window in registers while walking the taps four at a time.

Numba has no float4 shared loads, so the window is kept in scalar registers;
the structure is otherwise identical.  Dynamic shared memory is used so the
tile size matches the actual kernel size.

solve(input, kernel, output, input_rows, input_cols, kernel_rows,
      kernel_cols) keeps the challenge signature.  Host arrays are copied to
the device and back by Numba.
"""

import numpy as np
from numba import cuda, float32

TX = 16
TY = 16
RM = 2
RN = 4
BW = TX * RN                     # 64 output columns per block
BH = TY * RM                     # 32 output rows per block
MAX_KERNEL_ELEMS = 992           # 31 rows of 32 floats (padded)


@cuda.jit
def conv2d_kernel(input, kernel, output, input_rows, input_cols, kernel_rows,
                  kernel_cols, output_rows, output_cols):
    s_kernel = cuda.shared.array(shape=MAX_KERNEL_ELEMS, dtype=float32)
    s_input = cuda.shared.array(shape=0, dtype=float32)  # dynamic

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * TX + tx
    block_out_x = cuda.blockIdx.x * BW
    block_out_y = cuda.blockIdx.y * BH

    tile_w = BW + kernel_cols - 1
    stride = ((tile_w + 3) // 4) * 4
    tile_h = BH + kernel_rows - 1
    ks = ((kernel_cols + 3) // 4) * 4

    # Stage the kernel, zero-padding each row to a multiple of four floats.
    i = tid
    while i < kernel_rows * ks:
        r = i // ks
        c = i - r * ks
        if c < kernel_cols:
            s_kernel[i] = kernel[r * kernel_cols + c]
        else:
            s_kernel[i] = 0.0
        i += TX * TY

    # Stage the input tile with halo; out-of-range elements are zeroed.
    ly = ty
    while ly < tile_h:
        gy = block_out_y + ly
        base = ly * stride
        row_ok = gy < input_rows
        lx = tx
        while lx < tile_w:
            gx = block_out_x + lx
            if row_ok and gx < input_cols:
                s_input[base + lx] = input[gy * input_cols + gx]
            else:
                s_input[base + lx] = 0.0
            lx += TX
        ly += TY
    cuda.syncthreads()

    oy = block_out_y + ty * RM
    ox = block_out_x + tx * RN
    ly0 = ty * RM
    lx0 = tx * RN

    a00 = float32(0.0)
    a01 = float32(0.0)
    a02 = float32(0.0)
    a03 = float32(0.0)
    a10 = float32(0.0)
    a11 = float32(0.0)
    a12 = float32(0.0)
    a13 = float32(0.0)

    m = 0
    while m < kernel_rows:
        row0 = (ly0 + m) * stride + lx0
        row1 = (ly0 + 1 + m) * stride + lx0
        krow = m * ks
        x00 = s_input[row0]
        x01 = s_input[row0 + 1]
        x02 = s_input[row0 + 2]
        x03 = s_input[row0 + 3]
        x10 = s_input[row1]
        x11 = s_input[row1 + 1]
        x12 = s_input[row1 + 2]
        x13 = s_input[row1 + 3]

        n = 0
        while n < ks:
            y00 = s_input[row0 + n + 4]
            y01 = s_input[row0 + n + 5]
            y02 = s_input[row0 + n + 6]
            y03 = s_input[row0 + n + 7]
            y10 = s_input[row1 + n + 4]
            y11 = s_input[row1 + n + 5]
            y12 = s_input[row1 + n + 6]
            y13 = s_input[row1 + n + 7]
            k0 = s_kernel[krow + n]
            k1 = s_kernel[krow + n + 1]
            k2 = s_kernel[krow + n + 2]
            k3 = s_kernel[krow + n + 3]

            a00 += x00 * k0
            a01 += x01 * k0
            a02 += x02 * k0
            a03 += x03 * k0
            a10 += x10 * k0
            a11 += x11 * k0
            a12 += x12 * k0
            a13 += x13 * k0

            a00 += x01 * k1
            a01 += x02 * k1
            a02 += x03 * k1
            a03 += y00 * k1
            a10 += x11 * k1
            a11 += x12 * k1
            a12 += x13 * k1
            a13 += y10 * k1

            a00 += x02 * k2
            a01 += x03 * k2
            a02 += y00 * k2
            a03 += y01 * k2
            a10 += x12 * k2
            a11 += x13 * k2
            a12 += y10 * k2
            a13 += y11 * k2

            a00 += x03 * k3
            a01 += y00 * k3
            a02 += y01 * k3
            a03 += y02 * k3
            a10 += x13 * k3
            a11 += y10 * k3
            a12 += y11 * k3
            a13 += y12 * k3

            x00 = y00
            x01 = y01
            x02 = y02
            x03 = y03
            x10 = y10
            x11 = y11
            x12 = y12
            x13 = y13
            n += 4
        m += 1

    if oy < output_rows:
        base = oy * output_cols + ox
        if ox + 0 < output_cols:
            output[base + 0] = a00
        if ox + 1 < output_cols:
            output[base + 1] = a01
        if ox + 2 < output_cols:
            output[base + 2] = a02
        if ox + 3 < output_cols:
            output[base + 3] = a03
    if oy + 1 < output_rows:
        base = (oy + 1) * output_cols + ox
        if ox + 0 < output_cols:
            output[base + 0] = a10
        if ox + 1 < output_cols:
            output[base + 1] = a11
        if ox + 2 < output_cols:
            output[base + 2] = a12
        if ox + 3 < output_cols:
            output[base + 3] = a13


def solve(input, kernel, output, input_rows, input_cols, kernel_rows,
          kernel_cols):
    if input is None or kernel is None or output is None:
        return
    if min(input_rows, input_cols, kernel_rows, kernel_cols) <= 0:
        raise ValueError("dimensions must be positive")
    if kernel_rows > 31 or kernel_cols > 31:
        raise ValueError("kernel dimensions must be <= 31")

    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    if output_rows <= 0 or output_cols <= 0:
        raise ValueError("kernel must not exceed the input")

    tile_w = BW + kernel_cols - 1
    stride = ((tile_w + 3) // 4) * 4
    tile_h = BH + kernel_rows - 1
    shared_bytes = (stride * tile_h + 16) * 4

    grid = ((output_cols + BW - 1) // BW, (output_rows + BH - 1) // BH)
    conv2d_kernel[grid, (TX, TY), 0, shared_bytes](
        input, kernel, output, input_rows, input_cols, kernel_rows,
        kernel_cols, output_rows, output_cols)
    cuda.synchronize()


def _reference(inp, ker, rows, cols, kr, kc):
    out_r = rows - kr + 1
    out_c = cols - kc + 1
    inp64 = inp.astype(np.float64).reshape(rows, cols)
    ker64 = ker.astype(np.float64).reshape(kr, kc)
    ref = np.zeros((out_r, out_c))
    for m in range(kr):
        for n in range(kc):
            ref += ker64[m, n] * inp64[m:m + out_r, n:n + out_c]
    return ref


def _run(name, inp, ker, rows, cols, kr, kc):
    out_r = rows - kr + 1
    out_c = cols - kc + 1
    d_in = cuda.to_device(inp)
    d_k = cuda.to_device(ker)
    d_out = cuda.device_array(out_r * out_c, dtype=np.float32)
    solve(d_in, d_k, d_out, rows, cols, kr, kc)
    got = d_out.copy_to_host().reshape(out_r, out_c)

    ref = _reference(inp, ker, rows, cols, kr, kc)
    max_err = float(np.abs(got - ref).max()) if got.size else 0.0
    max_ref = float(np.abs(ref).max()) if ref.size else 0.0
    rel = max_err / max(1e-30, max_ref)
    ok = rel < 1e-4
    print(f"{name:>18} {rows}x{cols} k={kr}x{kc} out={out_r}x{out_c}: "
          f"max_err={max_err:.3e} rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example 1", np.array([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    dtype=np.float32),
               np.array([0, 1, 1, 0], dtype=np.float32), 3, 3, 2, 2)
    ok &= _run("example 2",
               np.array([1, 1, 1, 1, 1, 2, 3, 1, 1, 4, 5, 1, 1, 1, 1, 1],
                        dtype=np.float32),
               np.array([1, 0, 1], dtype=np.float32), 4, 4, 1, 3)

    for rows, cols, kr, kc in [(1, 1, 1, 1), (5, 5, 1, 1), (100, 100, 1, 3),
                               (100, 100, 3, 1), (100, 100, 7, 5),
                               (257, 129, 15, 15), (64, 64, 31, 31),
                               (33, 31, 1, 31), (17, 19, 17, 19),
                               (300, 200, 2, 2)]:
        inp = rng.normal(0.0, 1.0, rows * cols).astype(np.float32)
        ker = rng.normal(0.0, 1.0, kr * kc).astype(np.float32)
        ok &= _run("random", inp, ker, rows, cols, kr, kc)

    # Performance: 3072 x 3072 input, 15 x 15 kernel, median of 20 runs.
    import statistics
    import time

    rows, cols, kr, kc = 3072, 3072, 15, 15
    out_r = rows - kr + 1
    out_c = cols - kc + 1
    inp = rng.normal(0.0, 1.0, rows * cols).astype(np.float32)
    ker = rng.normal(0.0, 1.0, kr * kc).astype(np.float32)
    d_in = cuda.to_device(inp)
    d_k = cuda.to_device(ker)
    d_out = cuda.device_array(out_r * out_c, dtype=np.float32)
    solve(d_in, d_k, d_out, rows, cols, kr, kc)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(d_in, d_k, d_out, rows, cols, kr, kc)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    gflops = 2.0 * out_r * out_c * kr * kc / (ms * 1e6)

    got = d_out.copy_to_host()
    inp64 = inp.astype(np.float64).reshape(rows, cols)
    ker64 = ker.astype(np.float64).reshape(kr, kc)
    max_err = 0.0
    max_ref = 0.0
    for i in range(0, out_r, 173):
        for j in range(0, out_c, 197):
            s = float(np.sum(inp64[i:i + kr, j:j + kc] * ker64))
            max_err = max(max_err, abs(float(got[i * out_c + j]) - s))
            max_ref = max(max_ref, abs(s))
    rel = max_err / max(1e-30, max_ref)
    print(f"Perf {rows}x{cols} k={kr}x{kc}: {ms:.3f} ms, {gflops:.1f} GFLOP/s, "
          f"rel={rel:.3e}, {'OK' if rel < 1e-4 else 'FAIL'}")
