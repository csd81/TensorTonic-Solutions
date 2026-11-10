import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a Python float.
    """
    trace_sum = 0.0
    for i in range(len(A)):
        trace_sum += A[i][i]

    return float(trace_sum)