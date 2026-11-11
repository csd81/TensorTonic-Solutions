def pipeline_bubble_utilization(num_stages: int, num_microbatches: int, include_backward: bool) -> dict:
    """
    Returns a dict: schedule, bubble_slots, total_slots, utilization.
    """
    p = num_stages
    m = num_microbatches
    T = m + p - 1

    schedule = []

    # Forward wave: t in [0, T - 1]
    for t in range(T):
        slot = []
        for s in range(p):
            i_fwd = t - s
            if 0 <= i_fwd < m:
                slot.append(f"F{i_fwd}")
            else:
                slot.append(None)
        schedule.append(slot)

    # Backward wave: u in [0, T - 1]
    if include_backward:
        for u in range(T):
            slot = []
            for s in range(p):
                i_bwd = m - 1 - (u - (p - 1 - s))
                if 0 <= i_bwd < m:
                    slot.append(f"B{i_bwd}")
                else:
                    slot.append(None)
            schedule.append(slot)

    q = 2 if include_backward else 1
    useful_slots = q * p * m
    total_slots = q * T * p
    bubble_slots = total_slots - useful_slots
    utilization = float(useful_slots / total_slots)

    return {
        "schedule": schedule,
        "bubble_slots": bubble_slots,
        "total_slots": total_slots,
        "utilization": utilization,
    }
