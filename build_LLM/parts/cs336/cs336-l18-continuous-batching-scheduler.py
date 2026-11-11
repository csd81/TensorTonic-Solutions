def continuous_batching_scheduler(requests: list, kv_capacity: int) -> list:
    """
    Returns step dictionaries with step, admitted, active, finished, and kv_tokens_in_use.
    """
    if not requests:
        return []

    # Sort requests by (arrival_step, original_index) without modifying inputs
    pending = [
        {
            "id": req["id"],
            "arrival_step": req["arrival_step"],
            "prompt_tokens": req["prompt_tokens"],
            "max_new_tokens": req["max_new_tokens"],
            "footprint": req["prompt_tokens"] + req["max_new_tokens"],
            "generated": 0,
            "order": idx,
        }
        for idx, req in enumerate(requests)
    ]
    pending.sort(key=lambda r: (r["arrival_step"], r["order"]))

    waiting = []
    active = []
    reserved_used = 0
    pending_idx = 0
    n_requests = len(pending)
    step = pending[0]["arrival_step"]
    history = []

    def run_admission_pass(admitted_list: list, finished_list: list) -> None:
        nonlocal reserved_used
        still_waiting = []
        for req in waiting:
            if reserved_used + req["footprint"] <= kv_capacity:
                admitted_list.append(req["id"])
                if req["max_new_tokens"] == 0:
                    # Finishes immediately on admission without consuming ongoing capacity
                    finished_list.append(req["id"])
                else:
                    reserved_used += req["footprint"]
                    active.append(req)
            else:
                still_waiting.append(req)
        waiting[:] = still_waiting

    while pending_idx < n_requests or waiting or active:
        # Skip idle gaps when nothing is waiting or active
        if not waiting and not active:
            step = pending[pending_idx]["arrival_step"]

        # Move all requests arriving at or before the current step into waiting
        while pending_idx < n_requests and pending[pending_idx]["arrival_step"] <= step:
            waiting.append(pending[pending_idx])
            pending_idx += 1

        admitted_this_step = []
        finished_this_step = []

        # 1. Pre-decode admission pass
        run_admission_pass(admitted_this_step, finished_this_step)

        # 2. Decode phase: each active request generates 1 token
        still_active = []
        for req in active:
            req["generated"] += 1
            if req["generated"] == req["max_new_tokens"]:
                reserved_used -= req["footprint"]
                finished_this_step.append(req["id"])
            else:
                still_active.append(req)
        active = still_active

        # 3. Post-completion admission pass
        run_admission_pass(admitted_this_step, finished_this_step)

        kv_tokens_in_use = sum(
            req["prompt_tokens"] + req["generated"] for req in active
        )

        history.append(
            {
                "step": step,
                "admitted": admitted_this_step,
                "active": [req["id"] for req in active],
                "finished": finished_this_step,
                "kv_tokens_in_use": kv_tokens_in_use,
            }
        )

        step += 1

    return history
