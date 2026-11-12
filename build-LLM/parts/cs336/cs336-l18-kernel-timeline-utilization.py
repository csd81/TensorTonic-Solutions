def kernel_timeline_utilization(separate_intervals, fused_intervals, separate_sm_count, fused_sm_count):
    """
    Returns: dictionary containing timeline metrics and speedup
    """
    def analyze_timeline(intervals, sm_count):
        per_sm = [[] for _ in range(sm_count)]
        
        for sm_id, start, end in intervals:
            if end > start:
                per_sm[sm_id].append((float(start), float(end)))
                
        merged_per_sm = []
        for sm_intervals in per_sm:
            merged = []
            for start, end in sorted(sm_intervals):
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            merged_per_sm.append(merged)
            
        makespan = max(
            (end for merged in merged_per_sm for _, end in merged), default=0.0
        )
        busy_time = sum(
            end - start for merged in merged_per_sm for start, end in merged
        )
        utilization = busy_time / (sm_count * makespan) if makespan else 0.0
        idle_tails = [
            makespan - merged[-1][1] if merged else makespan
            for merged in merged_per_sm
        ]
        
        return {
            "makespan": makespan,
            "busy_time": busy_time,
            "utilization": utilization,
            "idle_tails": idle_tails,
        }

    sep_metrics = analyze_timeline(separate_intervals, separate_sm_count)
    fused_metrics = analyze_timeline(fused_intervals, fused_sm_count)
    
    speedup = (
        1.0
        if fused_metrics["makespan"] == 0
        else sep_metrics["makespan"] / fused_metrics["makespan"]
    )
        
    return {
        "separate": sep_metrics,
        "fused": fused_metrics,
        "relative_makespan_speedup": speedup,
    }
