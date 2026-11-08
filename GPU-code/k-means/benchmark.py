"""CPU vs GPU benchmark for kmeans.py.

Run from this directory:
    python benchmark.py                      # 1e4, 1e5, 1e6 points
    python benchmark.py 10000 50000 200000   # custom input sizes
"""

import pathlib
import statistics
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except Exception:
    pass

import kmeans  # noqa: E402


def _cpu_assign(X, centroids, labels, chunk_rows):
    """Vectorized nearest-centroid assignment, chunked to bound memory."""
    n_points = X.shape[0]
    for start in range(0, n_points, chunk_rows):
        end = min(start + chunk_rows, n_points)
        diff = X[start:end, None, :] - centroids[None, :, :]
        labels[start:end] = (diff * diff).sum(axis=-1).argmin(axis=1)


def cpu_kmeans(X, k, max_iters=20, seed=42, tol=1e-4):
    """Sequential NumPy Lloyd k-means (float64 reference)."""
    n_points = X.shape[0]
    rng = np.random.RandomState(seed)
    centroids = X[rng.choice(n_points, k, replace=False)].copy()
    labels = np.empty(n_points, dtype=np.int32)
    chunk_rows = max(1, 2_000_000 // k)

    converged = False
    iterations = 0
    for it in range(max_iters):
        iterations = it + 1
        _cpu_assign(X, centroids, labels, chunk_rows)

        new_centroids = centroids.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)

        if np.allclose(centroids, new_centroids, atol=tol):
            centroids = new_centroids
            converged = True
            break
        centroids = new_centroids

    # Final labels consistent with the returned centroids.
    _cpu_assign(X, centroids, labels, chunk_rows)
    return centroids, labels, iterations, converged


def inertia(X64, centroids, labels):
    diff = X64 - centroids[labels].astype(np.float64)
    return float((diff * diff).sum())


def benchmark_case(n_points, dim, k, max_iters, repeats, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_points, dim).astype(np.float32)
    X64 = X.astype(np.float64)

    # Warm-up: JIT compilation and allocator warm-up, excluded from timings.
    kmeans.run_gpu_kmeans(X, k, max_iters=1, seed=seed)

    gpu_wall, gpu_kernel, gpu_assign, gpu_update = [], [], [], []
    gpu_iters = []
    gpu_result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        c_gpu, l_gpu, stats = kmeans.run_gpu_kmeans(
            X, k, max_iters=max_iters, seed=seed
        )
        gpu_wall.append(time.perf_counter() - t0)
        gpu_kernel.append(stats["assign_ms"] + stats["zero_ms"] + stats["update_ms"])
        gpu_assign.append(stats["assign_ms"])
        gpu_update.append(stats["zero_ms"] + stats["update_ms"])
        gpu_iters.append(stats["iterations"])
        gpu_result = (c_gpu, l_gpu)

    cpu_times = []
    cpu_result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        cpu_result = cpu_kmeans(X64, k, max_iters=max_iters, seed=seed)
        cpu_times.append(time.perf_counter() - t0)

    c_gpu, l_gpu = gpu_result
    c_cpu, l_cpu, cpu_iters, cpu_converged = cpu_result
    inertia_gpu = inertia(X64, c_gpu, l_gpu)
    inertia_cpu = inertia(X64, c_cpu, l_cpu)
    rel_err = abs(inertia_gpu - inertia_cpu) / max(abs(inertia_cpu), 1e-30)

    return {
        "n": n_points,
        "dim": dim,
        "k": k,
        "gpu_wall_s": statistics.median(gpu_wall),
        "gpu_kernel_ms": statistics.median(gpu_kernel),
        "gpu_assign_ms": statistics.median(gpu_assign),
        "gpu_update_ms": statistics.median(gpu_update),
        "cpu_s": statistics.median(cpu_times),
        "inertia_gpu": inertia_gpu,
        "inertia_cpu": inertia_cpu,
        "inertia_rel_err": rel_err,
        "gpu_iters": gpu_iters[0],
        "cpu_iters": cpu_iters,
        "cpu_converged": cpu_converged,
    }


def main():
    sizes = [int(arg) for arg in sys.argv[1:]] or [10_000, 100_000, 1_000_000]
    dim, k, max_iters, repeats = 8, 8, 15, 3

    print(
        f"CPU vs GPU k-means benchmark | D={dim}, K={k}, "
        f"max_iters={max_iters}, repeats={repeats}, median reported"
    )
    print(f"GPU: {kmeans.cuda.get_current_device().name.decode()}")
    print("CPU reference uses float64 NumPy; GPU kernels use float32.")
    print()

    header = (
        f"{'N':>10} | {'GPU kernel':>11} | {'GPU wall':>10} | {'CPU':>10} | "
        f"{'speedup':>8} | {'inertia rel.err':>15} | {'iters G/C':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for n in sizes:
        r = benchmark_case(n, dim, k, max_iters, repeats, seed=42)
        results.append(r)
        speedup = r["cpu_s"] / r["gpu_wall_s"]
        print(
            f"{r['n']:>10,} | {r['gpu_kernel_ms']:>9.2f} ms | "
            f"{r['gpu_wall_s'] * 1000:>7.2f} ms | "
            f"{r['cpu_s'] * 1000:>7.2f} ms | "
            f"{speedup:>7.2f}x | {r['inertia_rel_err']:>15.2e} | "
            f"{r['gpu_iters']}/{r['cpu_iters']}"
        )

    print()
    print("GPU phases (median, includes the final assignment pass):")
    for r in results:
        print(
            f"  N={r['n']:>10,}: assign={r['gpu_assign_ms']:.2f} ms, "
            f"zero+update={r['gpu_update_ms']:.2f} ms, "
            f"kernel total={r['gpu_kernel_ms']:.2f} ms, "
            f"end-to-end={r['gpu_wall_s'] * 1000:.2f} ms"
        )


if __name__ == "__main__":
    main()
