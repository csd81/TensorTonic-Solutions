"""Numba CUDA port of convolution_3d.cu.

3D "valid" convolution:
    out[i][j][k] = sum_{d<KD, r<KR, c<KC} in[i+d][j+r][k+c] * k[d][r][c].

Same mapping as the CUDA C++ version: a block of 16x16 threads computes
a 64x16x4 tile of outputs, each thread owning 4 consecutive columns
for each of 4 output depths.  The input tile plus halo is staged in
shared memory (dynamic size, padded column stride) and the kernel rows are
padded to a multiple of four floats and zero filled.

Numba has no float4 shared loads, so the window is kept in scalar
registers; the structure is otherwise identical to the CUDA C++ kernel.

solve(input, kernel, output, input_depth, input_rows, input_cols,
      kernel_depth, kernel_rows, kernel_cols) keeps the challenge
signature.  Host arrays are copied to the device and back by Numba.
"""

import numpy as np
from numba import cuda, float32

TX = 16
TY = 16
RN = 4
RD = 4
BW = TX * RN   # 64 output columns per block
BH = TY        # 16 output rows per block
MAX_KERNEL_PADDED = 200  # 5*5 rows of 8 floats


@cuda.jit
def conv3d_kernel(input, kernel, output, in_d, in_r, in_c, kd, kr, kc,
                  out_d, out_r, out_c):
    s_kernel = cuda.shared.array(shape=MAX_KERNEL_PADDED, dtype=float32)
    s_tile = cuda.shared.array(shape=0, dtype=float32)  # dynamic

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tid = ty * TX + tx
    bx = cuda.blockIdx.x * BW
    by = cuda.blockIdx.y * BH
    bz = cuda.blockIdx.z * RD

    ks = ((kc + 3) // 4) * 4
    planes = RD + kd - 1
    rows = BH + kr - 1
    cols = BW + kc - 1
    stride = ((cols + 3) // 4) * 4

    # Stage the kernel, zero-padding each (d, r) row to a multiple of four.
    i = tid
    while i < kd * kr * ks:
        r = i // ks
        c = i - r * ks
        if c < kc:
            s_kernel[i] = kernel[r * kc + c]
        else:
            s_kernel[i] = 0.0
        i += TX * TY

    # Zero only the row padding; the staging loop writes every valid column.
    i = tid
    while i < planes * rows:
        base = i * stride
        c = cols
        while c < stride:
            s_tile[base + c] = 0.0
            c += 1
        i += TX * TY

    # Stage the input tile with halo; out-of-range elements are zeroed.
    d2 = 0
    while d2 < planes:
        gz = bz + d2
        z_ok = gz < in_d
        gplane = gz * in_r * in_c
        r2 = ty
        while r2 < rows:
            gy = by + r2
            y_ok = z_ok and gy < in_r
            base = (d2 * rows + r2) * stride
            grow = gplane + gy * in_c
            c2 = tx
            while c2 < cols:
                gx = bx + c2
                if y_ok and gx < in_c:
                    s_tile[base + c2] = input[grow + gx]
                else:
                    s_tile[base + c2] = 0.0
                c2 += TX
            r2 += TY
        d2 += 1
    cuda.syncthreads()

    ox = bx + tx * RN
    oy = by + ty
    cbase = tx * RN

    a00 = float32(0.0)
    a01 = float32(0.0)
    a02 = float32(0.0)
    a03 = float32(0.0)
    a10 = float32(0.0)
    a11 = float32(0.0)
    a12 = float32(0.0)
    a13 = float32(0.0)
    a20 = float32(0.0)
    a21 = float32(0.0)
    a22 = float32(0.0)
    a23 = float32(0.0)
    a30 = float32(0.0)
    a31 = float32(0.0)
    a32 = float32(0.0)
    a33 = float32(0.0)

    dk = 0
    while dk < kd:
        rk = 0
        while rk < kr:
            krow = (dk * kr + rk) * ks
            trow = ty + rk
            w0 = ((0 + dk) * rows + trow) * stride + cbase
            x00 = s_tile[w0]
            x01 = s_tile[w0 + 1]
            x02 = s_tile[w0 + 2]
            x03 = s_tile[w0 + 3]
            w1 = ((1 + dk) * rows + trow) * stride + cbase
            x10 = s_tile[w1]
            x11 = s_tile[w1 + 1]
            x12 = s_tile[w1 + 2]
            x13 = s_tile[w1 + 3]
            w2 = ((2 + dk) * rows + trow) * stride + cbase
            x20 = s_tile[w2]
            x21 = s_tile[w2 + 1]
            x22 = s_tile[w2 + 2]
            x23 = s_tile[w2 + 3]
            w3 = ((3 + dk) * rows + trow) * stride + cbase
            x30 = s_tile[w3]
            x31 = s_tile[w3 + 1]
            x32 = s_tile[w3 + 2]
            x33 = s_tile[w3 + 3]

            n = 0
            while n + 4 <= kc:
                y00 = s_tile[w0 + n + 4 + 0]
                y01 = s_tile[w0 + n + 4 + 1]
                y02 = s_tile[w0 + n + 4 + 2]
                y03 = s_tile[w0 + n + 4 + 3]
                y10 = s_tile[w1 + n + 4 + 0]
                y11 = s_tile[w1 + n + 4 + 1]
                y12 = s_tile[w1 + n + 4 + 2]
                y13 = s_tile[w1 + n + 4 + 3]
                y20 = s_tile[w2 + n + 4 + 0]
                y21 = s_tile[w2 + n + 4 + 1]
                y22 = s_tile[w2 + n + 4 + 2]
                y23 = s_tile[w2 + n + 4 + 3]
                y30 = s_tile[w3 + n + 4 + 0]
                y31 = s_tile[w3 + n + 4 + 1]
                y32 = s_tile[w3 + n + 4 + 2]
                y33 = s_tile[w3 + n + 4 + 3]
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
                a20 += x20 * k0
                a21 += x21 * k0
                a22 += x22 * k0
                a23 += x23 * k0
                a30 += x30 * k0
                a31 += x31 * k0
                a32 += x32 * k0
                a33 += x33 * k0
                a00 += x01 * k1
                a01 += x02 * k1
                a02 += x03 * k1
                a03 += y00 * k1
                a10 += x11 * k1
                a11 += x12 * k1
                a12 += x13 * k1
                a13 += y10 * k1
                a20 += x21 * k1
                a21 += x22 * k1
                a22 += x23 * k1
                a23 += y20 * k1
                a30 += x31 * k1
                a31 += x32 * k1
                a32 += x33 * k1
                a33 += y30 * k1
                a00 += x02 * k2
                a01 += x03 * k2
                a02 += y00 * k2
                a03 += y01 * k2
                a10 += x12 * k2
                a11 += x13 * k2
                a12 += y10 * k2
                a13 += y11 * k2
                a20 += x22 * k2
                a21 += x23 * k2
                a22 += y20 * k2
                a23 += y21 * k2
                a30 += x32 * k2
                a31 += x33 * k2
                a32 += y30 * k2
                a33 += y31 * k2
                a00 += x03 * k3
                a01 += y00 * k3
                a02 += y01 * k3
                a03 += y02 * k3
                a10 += x13 * k3
                a11 += y10 * k3
                a12 += y11 * k3
                a13 += y12 * k3
                a20 += x23 * k3
                a21 += y20 * k3
                a22 += y21 * k3
                a23 += y22 * k3
                a30 += x33 * k3
                a31 += y30 * k3
                a32 += y31 * k3
                a33 += y32 * k3
                x00 = y00
                x01 = y01
                x02 = y02
                x03 = y03
                x10 = y10
                x11 = y11
                x12 = y12
                x13 = y13
                x20 = y20
                x21 = y21
                x22 = y22
                x23 = y23
                x30 = y30
                x31 = y31
                x32 = y32
                x33 = y33
                n += 4

            while n < kc:
                kk = s_kernel[krow + n]
                a00 += x00 * kk
                a01 += x01 * kk
                a02 += x02 * kk
                a03 += x03 * kk
                x00 = x01
                x01 = x02
                x02 = x03
                x03 = s_tile[w0 + n + 4]
                a10 += x10 * kk
                a11 += x11 * kk
                a12 += x12 * kk
                a13 += x13 * kk
                x10 = x11
                x11 = x12
                x12 = x13
                x13 = s_tile[w1 + n + 4]
                a20 += x20 * kk
                a21 += x21 * kk
                a22 += x22 * kk
                a23 += x23 * kk
                x20 = x21
                x21 = x22
                x22 = x23
                x23 = s_tile[w2 + n + 4]
                a30 += x30 * kk
                a31 += x31 * kk
                a32 += x32 * kk
                a33 += x33 * kk
                x30 = x31
                x31 = x32
                x32 = x33
                x33 = s_tile[w3 + n + 4]
                n += 1
            rk += 1
        dk += 1

    # Masked stores.
    if bz + 0 < out_d and oy < out_r:
        obase = ((bz + 0) * out_r + oy) * out_c + ox
        if ox + 0 < out_c:
            output[obase + 0] = a00
        if ox + 1 < out_c:
            output[obase + 1] = a01
        if ox + 2 < out_c:
            output[obase + 2] = a02
        if ox + 3 < out_c:
            output[obase + 3] = a03
    if bz + 1 < out_d and oy < out_r:
        obase = ((bz + 1) * out_r + oy) * out_c + ox
        if ox + 0 < out_c:
            output[obase + 0] = a10
        if ox + 1 < out_c:
            output[obase + 1] = a11
        if ox + 2 < out_c:
            output[obase + 2] = a12
        if ox + 3 < out_c:
            output[obase + 3] = a13
    if bz + 2 < out_d and oy < out_r:
        obase = ((bz + 2) * out_r + oy) * out_c + ox
        if ox + 0 < out_c:
            output[obase + 0] = a20
        if ox + 1 < out_c:
            output[obase + 1] = a21
        if ox + 2 < out_c:
            output[obase + 2] = a22
        if ox + 3 < out_c:
            output[obase + 3] = a23
    if bz + 3 < out_d and oy < out_r:
        obase = ((bz + 3) * out_r + oy) * out_c + ox
        if ox + 0 < out_c:
            output[obase + 0] = a30
        if ox + 1 < out_c:
            output[obase + 1] = a31
        if ox + 2 < out_c:
            output[obase + 2] = a32
        if ox + 3 < out_c:
            output[obase + 3] = a33


def solve(input, kernel, output, input_depth, input_rows, input_cols,
          kernel_depth, kernel_rows, kernel_cols):
    if input is None or kernel is None or output is None:
        return
    dims = (input_depth, input_rows, input_cols, kernel_depth, kernel_rows,
            kernel_cols)
    if min(dims) <= 0:
        raise ValueError("dimensions must be positive")
    if kernel_depth > 5 or kernel_rows > 5 or kernel_cols > 5:
        raise ValueError("kernel dimensions must be <= 5")

    out_d = input_depth - kernel_depth + 1
    out_r = input_rows - kernel_rows + 1
    out_c = input_cols - kernel_cols + 1
    if out_d <= 0 or out_r <= 0 or out_c <= 0:
        raise ValueError("kernel must not exceed the input")

    ks = ((kernel_cols + 3) // 4) * 4
    planes = RD + kernel_depth - 1
    rows = BH + kernel_rows - 1
    cols = BW + kernel_cols - 1
    stride = ((cols + 3) // 4) * 4
    shared_bytes = (planes * rows * stride + 8 +
                    kernel_depth * kernel_rows * ks) * 4

    grid = ((out_c + BW - 1) // BW, (out_r + BH - 1) // BH,
            (out_d + RD - 1) // RD)
    conv3d_kernel[grid, (TX, TY), 0, shared_bytes](
        input, kernel, output, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, out_d, out_r, out_c)
    cuda.synchronize()


def _reference(inp, ker, idim, kdim):
    d, r, c = idim
    kd, kr, kc = kdim
    od, orr, oc = d - kd + 1, r - kr + 1, c - kc + 1
    v = inp.astype(np.float64).reshape(d, r, c)
    w = ker.astype(np.float64).reshape(kd, kr, kc)
    ref = np.zeros((od, orr, oc))
    for dd in range(kd):
        for rr in range(kr):
            for cc in range(kc):
                ref += w[dd, rr, cc] * v[dd:dd + od, rr:rr + orr,
                                          cc:cc + oc]
    return ref


def _run(name, inp, ker, idim, kdim):
    od = idim[0] - kdim[0] + 1
    orr = idim[1] - kdim[1] + 1
    oc = idim[2] - kdim[2] + 1
    d_in = cuda.to_device(inp)
    d_k = cuda.to_device(ker)
    d_out = cuda.device_array(od * orr * oc, dtype=np.float32)
    solve(d_in, d_k, d_out, *idim, *kdim)
    got = d_out.copy_to_host().reshape(od, orr, oc)

    ref = _reference(inp, ker, idim, kdim)
    max_err = float(np.abs(got - ref).max())
    max_ref = float(np.abs(ref).max())
    rel = max_err / max(1e-30, max_ref)
    ok = rel < 1e-4
    print(f"{name:>16} in={idim} k={kdim}: max_err={max_err:.3e} "
          f"rel={rel:.3e} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not cuda.is_available():
        raise SystemExit("No CUDA GPU available")

    rng = np.random.default_rng(42)
    ok = True

    ok &= _run("example 1",
               np.arange(1, 28, dtype=np.float32),
               np.array([1, 0, 0, 1, 1, 1, 0, 0, 0,
                         1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=np.float32),
               (3, 3, 3), (2, 3, 3))
    ok &= _run("example 2",
               np.arange(1, 9, dtype=np.float32),
               np.ones(8, dtype=np.float32), (2, 2, 2), (2, 2, 2))

    for idim, kdim in [((1, 1, 1), (1, 1, 1)), ((5, 5, 5), (1, 1, 1)),
                       ((9, 9, 9), (5, 5, 5)), ((8, 8, 8), (3, 5, 2)),
                       ((17, 19, 23), (5, 5, 5)), ((5, 5, 5), (5, 5, 5)),
                       ((10, 10, 10), (1, 5, 5)), ((10, 10, 10), (5, 1, 1)),
                       ((33, 31, 29), (4, 4, 4)), ((64, 64, 64), (5, 5, 5))]:
        inp = rng.normal(0.0, 1.0, idim[0] * idim[1] * idim[2])
        inp = inp.astype(np.float32)
        ker = rng.normal(0.0, 1.0, kdim[0] * kdim[1] * kdim[2])
        ker = ker.astype(np.float32)
        ok &= _run("random", inp, ker, idim, kdim)

    # Performance: 128^3 input, 5^3 kernel, median of 20 runs.
    import statistics
    import time

    idim, kdim = (128, 128, 128), (5, 5, 5)
    od, orr, oc = 124, 124, 124
    inp = rng.normal(0.0, 1.0, 128 * 128 * 128).astype(np.float32)
    ker = rng.normal(0.0, 1.0, 125).astype(np.float32)
    d_in = cuda.to_device(inp)
    d_k = cuda.to_device(ker)
    d_out = cuda.device_array(od * orr * oc, dtype=np.float32)
    solve(d_in, d_k, d_out, *idim, *kdim)  # warm-up
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        solve(d_in, d_k, d_out, *idim, *kdim)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = statistics.median(times)
    gflops = 2.0 * od * orr * oc * 125 / (ms * 1e6)

    got = d_out.copy_to_host().reshape(od, orr, oc)
    v = inp.astype(np.float64).reshape(128, 128, 128)
    w = ker.astype(np.float64).reshape(5, 5, 5)
    max_err = 0.0
    max_ref = 0.0
    for i in range(0, od, 7):
        for j in range(0, orr, 11):
            for kk in range(0, oc, 13):
                s = float(np.sum(v[i:i + 5, j:j + 5, kk:kk + 5] * w))
                max_err = max(max_err, abs(float(got[i, j, kk]) - s))
                max_ref = max(max_ref, abs(s))
    rel = max_err / max(1e-30, max_ref)
    print(f"Perf 128^3 k=5^3: {ms:.3f} ms, {gflops:.1f} GFLOP/s, "
          f"rel={rel:.3e}, {'OK' if rel < 1e-4 else 'FAIL'}")
