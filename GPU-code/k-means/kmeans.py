import time

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256

# --- 1. NUMBA CUDA KERNELS ---
@cuda.jit
def kmeans_assign_kernel(X, centroids, labels):
    """Assign every point to the nearest centroid (squared Euclidean distance)."""
    start_idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    n_points = X.shape[0]
    k = centroids.shape[0]
    dim = X.shape[1]

    for idx in range(start_idx, n_points, stride):
        min_dist = 1e30
        best_cluster = 0
        for j in range(k):
            dist_sq = 0.0
            for d in range(dim):
                diff = X[idx, d] - centroids[j, d]
                dist_sq += diff * diff
            if dist_sq < min_dist:
                min_dist = dist_sq
                best_cluster = j
        labels[idx] = best_cluster


@cuda.jit
def kmeans_update_kernel(X, labels, new_centroids, cluster_counts):
    """Atomically accumulate per-cluster coordinate sums and point counts."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    n_points = X.shape[0]
    dim = X.shape[1]

    for i in range(idx, n_points, stride):
        c_idx = labels[i]
        for d in range(dim):
            cuda.atomic.add(new_centroids, (c_idx, d), X[i, d])
        cuda.atomic.add(cluster_counts, c_idx, 1)


@cuda.jit
def zero_buffers_kernel(new_centroids, cluster_counts):
    """Zero the per-iteration accumulators on the device."""
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    k = new_centroids.shape[0]
    dim = new_centroids.shape[1]

    for i in range(idx, k * dim, stride):
        new_centroids[i // dim, i % dim] = 0.0
    for i in range(idx, k, stride):
        cluster_counts[i] = 0


# --- 2. INPUT VALIDATION ---
def _validate_inputs(X, k, max_iters):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_points, dim), got shape {X.shape}")
    n_points, dim = X.shape
    if n_points < 1:
        raise ValueError("X must contain at least one point")
    if dim < 1:
        raise ValueError("X must have at least one feature")
    if not 1 <= k <= n_points:
        raise ValueError(f"k must satisfy 1 <= k <= n_points, got k={k}, n_points={n_points}")
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    return X


# --- 3. GPU K-MEANS DRIVER ---
def run_gpu_kmeans(X, k, max_iters=20, seed=42):
    """Run k-means on the GPU.

    Returns (centroids, labels, stats), where stats contains per-phase
    millisecond timings plus ``total_s``, ``iterations`` and ``converged``.
    """
    X = _validate_inputs(X, k, max_iters)
    n_points, dim = X.shape

    # Deterministic initialization: sample k distinct input points.
    rng = np.random.RandomState(seed)
    init_indices = rng.choice(n_points, k, replace=False)
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    centroids = X32[init_indices].copy()
    labels = np.zeros(n_points, dtype=np.int32)

    setup_t0 = time.perf_counter()
    d_X = cuda.to_device(X32)
    d_centroids = cuda.to_device(centroids)
    d_labels = cuda.to_device(labels)
    # Scratch buffers are allocated once and reused every iteration.
    d_new_centroids = cuda.device_array((k, dim), dtype=np.float32)
    d_cluster_counts = cuda.device_array(k, dtype=np.int32)
    cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_t0) * 1000.0

    threads = THREADS_PER_BLOCK
    blocks = min(65535, (n_points + threads - 1) // threads)
    zero_blocks = min(65535, (k * dim + threads - 1) // threads)

    ev_a0, ev_a1 = cuda.event(), cuda.event()
    ev_z0, ev_z1 = cuda.event(), cuda.event()
    ev_u0, ev_u1 = cuda.event(), cuda.event()

    stats = {
        "setup_ms": setup_ms,
        "assign_ms": 0.0,
        "zero_ms": 0.0,
        "update_ms": 0.0,
        "d2h_ms": 0.0,
        "h2d_ms": 0.0,
        "host_ms": 0.0,
        "total_s": 0.0,
        "iterations": 0,
        "converged": False,
    }

    converged = False
    iterations = 0
    total_t0 = time.perf_counter()

    for it in range(max_iters):
        iterations = it + 1

        # 1. Assignment phase
        ev_a0.record()
        kmeans_assign_kernel[blocks, threads](d_X, d_centroids, d_labels)
        ev_a1.record()
        cuda.synchronize()
        stats["assign_ms"] += ev_a0.elapsed_time(ev_a1)

        # 2. Zero scratch buffers, then accumulate sums/counts
        ev_z0.record()
        zero_buffers_kernel[zero_blocks, threads](d_new_centroids, d_cluster_counts)
        ev_z1.record()
        ev_u0.record()
        kmeans_update_kernel[blocks, threads](d_X, d_labels, d_new_centroids, d_cluster_counts)
        ev_u1.record()
        cuda.synchronize()
        stats["zero_ms"] += ev_z0.elapsed_time(ev_z1)
        stats["update_ms"] += ev_u0.elapsed_time(ev_u1)

        # 3. Bring back block sums and counts, average on the host
        d2h_t0 = time.perf_counter()
        h_new_centroids = d_new_centroids.copy_to_host()
        h_counts = d_cluster_counts.copy_to_host()
        cuda.synchronize()
        stats["d2h_ms"] += (time.perf_counter() - d2h_t0) * 1000.0

        host_t0 = time.perf_counter()
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            if h_counts[j] > 0:
                new_centroids[j] = h_new_centroids[j] / h_counts[j]
            else:
                new_centroids[j] = centroids[j]  # empty cluster: keep old centroid
        stats["host_ms"] += (time.perf_counter() - host_t0) * 1000.0

        # 4. Convergence check
        if np.allclose(centroids, new_centroids, atol=1e-4):
            centroids = new_centroids
            converged = True
            break

        centroids = new_centroids
        h2d_t0 = time.perf_counter()
        d_centroids.copy_to_device(centroids)
        cuda.synchronize()
        stats["h2d_ms"] += (time.perf_counter() - h2d_t0) * 1000.0

    # Final assignment pass so the returned labels match the returned centroids.
    d_centroids.copy_to_device(centroids)
    cuda.synchronize()
    ev_a0.record()
    kmeans_assign_kernel[blocks, threads](d_X, d_centroids, d_labels)
    ev_a1.record()
    cuda.synchronize()
    stats["assign_ms"] += ev_a0.elapsed_time(ev_a1)

    final_labels = d_labels.copy_to_host()

    stats["total_s"] = time.perf_counter() - total_t0
    stats["iterations"] = iterations
    stats["converged"] = converged

    return centroids, final_labels, stats


# --- 4. DEMO ---
if __name__ == "__main__":
    print("GPU information:")
    try:
        print(cuda.detect())
    except Exception as exc:
        print("CUDA is not available. Make sure a GPU runtime is enabled.")
        raise exc

    print("\nGenerating dataset...")
    N, D, K = 200_000, 8, 8
    rng = np.random.RandomState(42)
    X_data = rng.randn(N, D).astype(np.float32)

    print(f"Parameters: N={N:,}, D={D}, K={K}. Running GPU k-means...")
    # Warm-up so the reported phase timings exclude JIT compilation.
    run_gpu_kmeans(X_data, K, max_iters=1, seed=42)
    centroids, labels, stats = run_gpu_kmeans(X_data, K, max_iters=15)

    print(f"Done. GPU total time: {stats['total_s']:.4f} s")
    print(
        "  assign: {assign_ms:.3f} ms | zero: {zero_ms:.3f} ms | "
        "update: {update_ms:.3f} ms".format(**stats)
    )
    print(
        "  d2h: {d2h_ms:.3f} ms | h2d: {h2d_ms:.3f} ms | host: {host_ms:.3f} ms".format(
            **stats
        )
    )
    print(f"  iterations: {stats['iterations']}, converged: {stats['converged']}")
