from fractions import Fraction

def topology_aware_parallel_plan(
    num_gpus: int,
    gpus_per_node: int,
    model_state_bytes: int,
    activation_bytes_per_sample: int,
    global_batch_size: int,
    sequence_length: int,
    memory_per_gpu: int,
    min_context_parallel: int,
) -> dict | None:
    """
    Returns the best feasible plan dict, or None if no plan fits.
    """
    def ceil_div(numerator: int, denominator: int) -> int:
        return (numerator + denominator - 1) // denominator

    # Collect all positive divisors of num_gpus (G <= 64, so at most 12 divisors)
    divisors = [x for x in range(1, num_gpus + 1) if num_gpus % x == 0]

    best_score = None
    best_plan = None

    for d in divisors:
        if global_batch_size % d != 0:
            continue
        rem_after_d = num_gpus // d

        for t in divisors:
            if rem_after_d % t != 0:
                continue
            rem_after_dt = rem_after_d // t

            for p in divisors:
                if rem_after_dt % p != 0:
                    continue
                c = rem_after_dt // p

                # Check context parallel and intra-node topology constraints
                if c < min_context_parallel or sequence_length % c != 0:
                    continue
                if t * c > gpus_per_node:
                    continue

                # Compute exact integer per-GPU memory usage
                m_s = ceil_div(model_state_bytes, t * p)
                local_batch = global_batch_size // d
                m_a = ceil_div(activation_bytes_per_sample * local_batch, t * c)
                total_mem = m_s + m_a

                if total_mem > memory_per_gpu:
                    continue

                # Use exact Fraction arithmetic for tie-breaking comparison
                i_out_exact = (
                    Fraction(2 * m_s * (d - 1), d)
                    + Fraction(m_a * (p - 1), p)
                )
                i_in_exact = Fraction(
                    2 * m_a * (t * (c - 1) + c * (t - 1)),
                    t * c,
                )

                score = (i_out_exact, i_in_exact, p, -d, t, c)

                if best_score is None or score < best_score:
                    best_score = score
                    best_plan = {
                        "data_parallel": d,
                        "tensor_parallel": t,
                        "pipeline_parallel": p,
                        "context_parallel": c,
                        "estimated_memory_bytes": int(total_mem),
                        "inter_node_bytes": float(i_out_exact),
                        "intra_node_bytes": float(i_in_exact),
                    }

    return best_plan
