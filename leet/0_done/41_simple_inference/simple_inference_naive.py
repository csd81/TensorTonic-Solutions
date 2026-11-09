"""Naive baseline for the Linear inference challenge.

Same result as simple_inference.py, but computed one output column at a
time with broadcast multiply + reduction instead of a single cuBLAS GEMM:

    for o in range(output_size):
        output[:, o] = (input * weight[o]).sum(dim=1) + bias[o]

This is a legitimate "naive" implementation in the sense that matters here:
it uses no matrix-multiply primitive, launches O(output_size) kernels, and
re-reads the entire input once per output column (output_size x the required
traffic).  At batch = input_size = output_size = 1000 that is 1000 kernels
over 4 MB tensors instead of one GEMM.
"""

import torch


def solve(input: torch.Tensor, model: torch.nn.Module, output: torch.Tensor):
    with torch.no_grad():
        w = model.weight
        b = model.bias
        for o in range(w.shape[0]):
            output[:, o] = (input * w[o]).sum(dim=1) + b[o]
