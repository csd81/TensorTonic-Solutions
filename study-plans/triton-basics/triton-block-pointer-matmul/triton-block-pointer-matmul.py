import torch
import triton
import triton.language as tl


@triton.jit
def block_ptr_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Identify which 2D tile block of matrix C this program instance computes
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Initialize the accumulation register tile as zeros
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Construct the initial block pointers for matrices A and B
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # Iterate over the inner dimension chunks
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load block chunks using implicit internal boundary checking
        
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")


        # Perform fused dot product accumulation
        acc = tl.dot(a, b, acc, input_precision="ieee")


        # Advance block pointers efficiently along the shared K axis
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # Construct the destination block pointer for matrix C
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Save the computed accumulated matrix tile back into memory
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))



def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch block_ptr_matmul_kernel using tl.make_block_ptr and tl.advance."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    block_ptr_matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )