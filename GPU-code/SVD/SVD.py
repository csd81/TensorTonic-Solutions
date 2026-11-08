"""One-sided Jacobi SVD with Numba CUDA.

A is stored column-major as ``At`` (n_work x m, row j is column j of A) so
that every kernel reads/writes contiguous rows.  For odd n an extra zero
column is appended (n_work = n + 1); pairs involving it are no-ops.

Per sweep, the host runs the round-robin ("circle method") schedule of
n_work - 1 sub-sweeps, each with n_work / 2 disjoint column pairs:

  1. ``jacobi_dots_kernel``   computes alpha, beta, gamma for every pair
                              with a shared-memory block reduction.
  2. ``jacobi_rotate_kernel`` computes cos/sin once per pair and applies the
                              same rotation to At's rows and Vt's rows.

A sweep converges when no pair needed a rotation.  Finally the column norms
are the singular values and the normalized columns are U (host side).

Requires m >= n.  The demo compares against a NumPy Jacobi reference and
``numpy.linalg.svd``.
"""

import math

import numpy as np
from numba import cuda, float64, int32

THREADS = 256
DEFAULT_TOL = 1e-12
DEFAULT_MAX_SWEEPS = 30


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@cuda.jit
def jacobi_dots_kernel(At, pair_p, pair_q, alpha, beta, gamma,
                       round_idx, n_pairs, m):
    """alpha = ap.ap, beta = aq.aq, gamma = ap.aq for each column pair."""
    pair = cuda.blockIdx.x
    if pair >= n_pairs:
        return

    p = pair_p[round_idx, pair]
    q = pair_q[round_idx, pair]
    tid = cuda.threadIdx.x

    local_a = 0.0
    local_b = 0.0
    local_g = 0.0
    i = tid
    while i < m:
        ap = At[p, i]
        aq = At[q, i]
        local_a += ap * ap
        local_b += aq * aq
        local_g += ap * aq
        i += cuda.blockDim.x

    s_a = cuda.shared.array(THREADS, float64)
    s_b = cuda.shared.array(THREADS, float64)
    s_g = cuda.shared.array(THREADS, float64)
    s_a[tid] = local_a
    s_b[tid] = local_b
    s_g[tid] = local_g
    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            s_a[tid] += s_a[tid + stride]
            s_b[tid] += s_b[tid + stride]
            s_g[tid] += s_g[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        alpha[pair] = s_a[0]
        beta[pair] = s_b[0]
        gamma[pair] = s_g[0]


@cuda.jit
def jacobi_rotate_kernel(At, Vt, pair_p, pair_q, alpha, beta, gamma,
                         rot_count, round_idx, n_pairs, m, n_work, tol):
    """Apply one Givens rotation to a pair of A columns and V columns."""
    pair = cuda.blockIdx.x
    if pair >= n_pairs:
        return

    p = pair_p[round_idx, pair]
    q = pair_q[round_idx, pair]
    tid = cuda.threadIdx.x

    s_c = cuda.shared.array(1, float64)
    s_s = cuda.shared.array(1, float64)
    s_do = cuda.shared.array(1, int32)

    if tid == 0:
        a = alpha[pair]
        b = beta[pair]
        g = gamma[pair]
        scale = math.sqrt(a) * math.sqrt(b)
        if scale == 0.0 or abs(g) <= tol * scale:
            s_c[0] = 1.0
            s_s[0] = 0.0
            s_do[0] = 0
        else:
            zeta = (b - a) / (2.0 * g)
            t = (1.0 if zeta >= 0.0 else -1.0) / (
                abs(zeta) + math.sqrt(1.0 + zeta * zeta)
            )
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c
            s_c[0] = c
            s_s[0] = s
            s_do[0] = 1
            cuda.atomic.add(rot_count, 0, 1)
    cuda.syncthreads()

    if s_do[0] != 0:
        c = s_c[0]
        s = s_s[0]

        i = tid
        while i < m:
            ap = At[p, i]
            aq = At[q, i]
            At[p, i] = c * ap - s * aq
            At[q, i] = s * ap + c * aq
            i += cuda.blockDim.x

        i = tid
        while i < n_work:
            vp = Vt[p, i]
            vq = Vt[q, i]
            Vt[p, i] = c * vp - s * vq
            Vt[q, i] = s * vp + c * vq
            i += cuda.blockDim.x


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------

def _circle_pairs(n_work):
    """Round-robin pairing for even n_work: (n_work-1, n_work/2) index arrays."""
    if n_work % 2:
        raise ValueError("n_work must be even")
    n_half = n_work // 2
    n_rounds = n_work - 1
    pp = np.empty((n_rounds, n_half), dtype=np.int32)
    qq = np.empty((n_rounds, n_half), dtype=np.int32)
    for r in range(n_rounds):
        pp[r, 0] = 0
        qq[r, 0] = r + 1
        for k in range(1, n_half):
            pp[r, k] = (r + k) % (n_rounds) + 1
            qq[r, k] = (r - k) % (n_rounds) + 1
    return pp, qq


def _rotation_params(a, b, g, tol):
    """Return (c, s) that orthogonalizes columns with moments a, b, g."""
    scale = math.sqrt(a) * math.sqrt(b)
    if scale == 0.0 or abs(g) <= tol * scale:
        return 1.0, 0.0
    zeta = (b - a) / (2.0 * g)
    t = (1.0 if zeta >= 0.0 else -1.0) / (
        abs(zeta) + math.sqrt(1.0 + zeta * zeta)
    )
    c = 1.0 / math.sqrt(1.0 + t * t)
    return c, t * c


def _finalize(At, Vt, m, n):
    """Column norms -> singular values; normalized columns -> U; Vh = V^T."""
    cols = At[:n]
    norms = np.sqrt(np.einsum("ij,ij->i", cols, cols))
    order = np.argsort(-norms)

    s = norms[order]
    sorted_cols = cols[order]

    U = np.zeros((m, n), dtype=np.float64)
    nonzero = s > 0.0
    if np.any(nonzero):
        U[:, nonzero] = (sorted_cols[nonzero] / s[nonzero, None]).T

    Vh = Vt[:n, :n][order, :]
    return U, s, Vh


# ---------------------------------------------------------------------------
# Sequential reference (same schedule, same rotation formula)
# ---------------------------------------------------------------------------

def jacobi_svd_cpu(A, tol=DEFAULT_TOL, max_sweeps=DEFAULT_MAX_SWEEPS):
    """NumPy one-sided Jacobi SVD reference; returns (U, s, Vh)."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if m < n:
        raise ValueError("m >= n is required")

    n_work = n + (n & 1)
    At = np.zeros((n_work, m), dtype=np.float64)
    At[:n] = A.T
    Vt = np.eye(n_work, dtype=np.float64)
    pp, qq = _circle_pairs(n_work)

    sweeps = 0
    for _ in range(max_sweeps):
        rotations = 0
        for r in range(pp.shape[0]):
            for k in range(pp.shape[1]):
                p = int(pp[r, k])
                q = int(qq[r, k])
                ap = At[p]
                aq = At[q]
                a = float(ap @ ap)
                b = float(aq @ aq)
                g = float(ap @ aq)
                c, s = _rotation_params(a, b, g, tol)
                if s == 0.0:
                    continue

                new_p = c * ap - s * aq
                new_q = s * ap + c * aq
                At[p] = new_p
                At[q] = new_q

                vp = Vt[p]
                vq = Vt[q]
                new_vp = c * vp - s * vq
                new_vq = s * vp + c * vq
                Vt[p] = new_vp
                Vt[q] = new_vq
                rotations += 1
        sweeps += 1
        if rotations == 0:
            break

    U, s, Vh = _finalize(At, Vt, m, n)
    return U, s, Vh


# ---------------------------------------------------------------------------
# GPU driver
# ---------------------------------------------------------------------------

def jacobi_svd_gpu(A, tol=DEFAULT_TOL, max_sweeps=DEFAULT_MAX_SWEEPS,
                   return_info=False):
    """One-sided Jacobi SVD on the GPU; returns (U, s, Vh)."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if m < n:
        raise ValueError("m >= n is required")
    if n < 1:
        raise ValueError("A must have at least one column")

    n_work = n + (n & 1)
    At = np.zeros((n_work, m), dtype=np.float64)
    At[:n] = A.T
    Vt = np.eye(n_work, dtype=np.float64)
    pp, qq = _circle_pairs(n_work)

    d_At = cuda.to_device(At)
    d_Vt = cuda.to_device(Vt)
    d_pp = cuda.to_device(pp)
    d_qq = cuda.to_device(qq)
    d_alpha = cuda.device_array(pp.shape[1], dtype=np.float64)
    d_beta = cuda.device_array(pp.shape[1], dtype=np.float64)
    d_gamma = cuda.device_array(pp.shape[1], dtype=np.float64)
    d_rot_count = cuda.device_array(1, dtype=np.int32)

    n_pairs = pp.shape[1]
    n_rounds = pp.shape[0]
    sweeps = 0
    converged = False
    for _ in range(max_sweeps):
        d_rot_count[0] = 0
        for r in range(n_rounds):
            jacobi_dots_kernel[n_pairs, THREADS](
                d_At, d_pp, d_qq, d_alpha, d_beta, d_gamma,
                r, n_pairs, m,
            )
            jacobi_rotate_kernel[n_pairs, THREADS](
                d_At, d_Vt, d_pp, d_qq, d_alpha, d_beta, d_gamma,
                d_rot_count, r, n_pairs, m, n_work, tol,
            )
        sweeps += 1
        if int(d_rot_count.copy_to_host()[0]) == 0:
            converged = True
            break

    At_h = d_At.copy_to_host()
    Vt_h = d_Vt.copy_to_host()
    U, s, Vh = _finalize(At_h, Vt_h, m, n)

    if return_info:
        info = {
            "sweeps": sweeps,
            "converged": converged,
            "n_work": n_work,
            "n_pairs_per_round": n_pairs,
            "n_rounds": n_rounds,
        }
        return U, s, Vh, info
    return U, s, Vh


# ---------------------------------------------------------------------------
# Demo / validation
# ---------------------------------------------------------------------------

def _errors(A, U, s, Vh):
    recon = (U * s) @ Vh
    rec = np.linalg.norm(A - recon) / np.linalg.norm(A)
    orth_u = np.linalg.norm(U.T @ U - np.eye(U.shape[1]), ord=np.inf)
    orth_v = np.linalg.norm(Vh @ Vh.T - np.eye(Vh.shape[0]), ord=np.inf)
    return rec, orth_u, orth_v


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    m, n = 128, 64
    A = rng.standard_normal((m, n))

    t0 = time.perf_counter()
    U_c, s_c, Vh_c = jacobi_svd_cpu(A)
    cpu_s = time.perf_counter() - t0

    # Warm-up (JIT)
    jacobi_svd_gpu(A)
    t0 = time.perf_counter()
    U_g, s_g, Vh_g, info = jacobi_svd_gpu(A, return_info=True)
    gpu_s = time.perf_counter() - t0

    _, s_ref, _ = np.linalg.svd(A, full_matrices=False)

    print(f"matrix {m}x{n}, sweeps={info['sweeps']}, converged={info['converged']}")
    print(f"CPU Jacobi: {cpu_s * 1000:.2f} ms | GPU Jacobi: {gpu_s * 1000:.2f} ms")
    for name, U, s, Vh in (("CPU", U_c, s_c, Vh_c), ("GPU", U_g, s_g, Vh_g)):
        rec, ou, ov = _errors(A, U, s, Vh)
        s_err = np.max(np.abs(s - s_ref)) / s_ref[0]
        print(f"{name}: rec={rec:.3e} | U^TU-I={ou:.3e} | VV^T-I={ov:.3e} "
              f"| sigma rel={s_err:.3e}")
