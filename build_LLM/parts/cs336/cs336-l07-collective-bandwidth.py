def collective_bandwidth(payload_bytes: int, world_size: int, duration_seconds: float, collective: str) -> dict:
    """
    Returns a dict of floats: algorithm_bytes and bandwidth_bytes_per_second.
    """
    if collective == "all_reduce":
        passes = 2
    elif collective in ("all_gather", "reduce_scatter", "all_to_all"):
        passes = 1
    else:
        raise ValueError(f"Unsupported collective: {collective}")

    # When world_size == 1, (world_size - 1) / world_size evaluates to 0.0
    algorithm_bytes = float(passes * payload_bytes * (world_size - 1) / world_size)
    bandwidth_bytes_per_second = float(algorithm_bytes / duration_seconds)

    return {
        "algorithm_bytes": algorithm_bytes,
        "bandwidth_bytes_per_second": bandwidth_bytes_per_second,
    }
