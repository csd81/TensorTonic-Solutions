# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Csörföly Dániel's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/csorfolydaniel.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Argmax | Implement a parallel CUDA argmax reduction that returns the lowest index when multiple elements share the maximum. | https://www.tensortonic.com/problems/argmax |
| Argmin | Implement a parallel CUDA argmin reduction that returns the lowest index when multiple elements share the minimum. | https://www.tensortonic.com/problems/argmin |
| 1D Convolution | Implement 1D cross-correlation in CUDA with one thread per output element and register accumulation over the kernel. | https://www.tensortonic.com/problems/conv-1d |
| 2D Convolution | Implement 2D cross-correlation in CUDA with a two-dimensional grid, row-major indexing, and per-output accumulation. | https://www.tensortonic.com/problems/conv-2d |
| 3D Convolution | Implement naive 3D cross-correlation in CUDA with a three-dimensional launch grid and row-major flattened indexing. | https://www.tensortonic.com/problems/conv-3d |
| Implement Cosine Similarity | Implement cosine similarity from scratch with NumPy. Covers dot products, Euclidean norms and zero-vector handling, with tests in the browser. | https://www.tensortonic.com/problems/cosine-similarity |
| Cross-Entropy Loss | Implement numerically stable mean cross-entropy loss in CUDA using row-wise log-sum-exp and batch reduction. | https://www.tensortonic.com/problems/cross-entropy |
| Blockwise Online Softmax | Implement stable blockwise online softmax in CUDA for contiguous or strided rows across float32, float16, and bfloat16 inputs. | https://www.tensortonic.com/problems/cs336-l05-blockwise-online-softmax |
| Fused Tiled Matmul and ReLU | Fuse tiled matrix multiplication and ReLU in Triton for strided float16 or bfloat16 CUDA inputs and preallocated outputs. | https://www.tensortonic.com/problems/cs336-l06-triton-matmul-relu |
| Tiled Triton Row Sum | Implement tiled row reduction in Triton for strided CUDA tensors, fixed-width tiles, and float32 output sums. | https://www.tensortonic.com/problems/cs336-l06-triton-tiled-row-sum |
| Cosine Similarity | Compute cosine similarity in CUDA by reducing the dot product and both vector norms into one scalar result. | https://www.tensortonic.com/problems/cuda-cosine-similarity |
| Dot Product | Implement a multi-block CUDA dot-product reduction that combines partial sums into one scalar output. | https://www.tensortonic.com/problems/cuda-dot-product |
| GELU | Implement exact GELU activation in CUDA with one thread per element and the device error-function intrinsic. | https://www.tensortonic.com/problems/cuda-gelu |
| Leaky ReLU | Implement Leaky ReLU activation in CUDA with one thread per element, bounds checks, and a configurable negative slope. | https://www.tensortonic.com/problems/cuda-leaky-relu |
| Matrix Transpose | Implement matrix transpose in CUDA with a two-dimensional launch grid, row-major buffers, and bounds-checked writes. | https://www.tensortonic.com/problems/cuda-matrix-transpose |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Dropout | Implement inverted dropout in CUDA using a supplied binary mask and scaling retained elements by the keep probability. | https://www.tensortonic.com/problems/dropout |
| Matrix-Vector Multiplication | Implement row-major CUDA matrix-vector multiplication with one thread computing each output row. | https://www.tensortonic.com/problems/gemv |
| Hadamard Product | Implement elementwise matrix multiplication in CUDA using a two-dimensional grid and row-major bounds-checked indexing. | https://www.tensortonic.com/problems/hadamard-product |
| L1 Normalization | Implement CUDA L1 vector normalization by reducing absolute values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l1-normalize |
| L2 Normalization | Implement CUDA L2 vector normalization by reducing squared values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l2-normalize |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/problems/la-dot-product |
| Layer Normalization | Implement fused row-wise LayerNorm in CUDA with shared-memory mean and variance reduction, affine scale, and bias. | https://www.tensortonic.com/problems/layer-norm |
| Matrix Addition | Implement elementwise matrix addition in CUDA with a two-dimensional grid, row-major indexing, and bounds checks. | https://www.tensortonic.com/problems/matrix-addition |
| Matrix Multiplication | Implement row-major matrix multiplication in CUDA with one thread per output element and inner-product accumulation. | https://www.tensortonic.com/problems/matrix-multiplication |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max of Array | Implement a multi-block CUDA maximum reduction that combines block-local maxima into one scalar output. | https://www.tensortonic.com/problems/max-of-array |
| Max Pool 2D | Implement 2D max pooling in CUDA with one thread per output, configurable windows and strides, and row-major indexing. | https://www.tensortonic.com/problems/max-pool-2d |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Mean and Variance | Compute population mean and variance in CUDA with device-side reductions and synchronized scalar outputs. | https://www.tensortonic.com/problems/mean-variance |
| Min of Array | Implement a multi-block CUDA minimum reduction that combines block-local minima into one scalar output. | https://www.tensortonic.com/problems/min-of-array |
| Outer Product | Compute a vector outer product in CUDA with a two-dimensional grid, row-major output, and bounds-checked indexing. | https://www.tensortonic.com/problems/outer-product |
| Boolean Indexing | Filter pandas rows by a numeric column threshold and return the matching records with their original column order. | https://www.tensortonic.com/problems/pandas-boolean-indexing |
| Change Data Types | Create a DataFrame, convert the specified column to the target type, and return the dtypes before and after conversion. | https://www.tensortonic.com/problems/pandas-change-dtypes |
| Column Selection | Create a pandas DataFrame from dictionary data and extract one named column as an ordered list. | https://www.tensortonic.com/problems/pandas-column-selection |
| Data Types Overview | Create a pandas DataFrame and report each column dtype together with counts for every unique dtype. | https://www.tensortonic.com/problems/pandas-data-types |
| Drop Duplicates | Create a DataFrame, remove duplicate rows, and return the cleaned result along with counts of rows before and after deduplication. | https://www.tensortonic.com/problems/pandas-drop-duplicates |
| GroupBy Basics | Create a DataFrame and compute the sum, mean, and count of the value column for each group. | https://www.tensortonic.com/problems/pandas-groupby-basics |
| Handle Missing Values | Create a pandas DataFrame, count missing entries per column, and replace every null with a supplied fill value. | https://www.tensortonic.com/problems/pandas-handle-missing |
| Head and Tail Operations | Create a pandas DataFrame and return the requested first and last rows as record-oriented dictionaries. | https://www.tensortonic.com/problems/pandas-head-tail |
| Inspect DataFrame Shape | Create a DataFrame and return its structural properties: row count, column count, column names, data types, and total number of values. | https://www.tensortonic.com/problems/pandas-inspect-shape |
| Loc vs iLoc | Create a DataFrame and use positional indexing to extract: the single element, the full row, and the full column. | https://www.tensortonic.com/problems/pandas-loc-iloc |
| Multi-Column Selection | Create a pandas DataFrame and select an ordered subset of named columns without changing row order. | https://www.tensortonic.com/problems/pandas-multi-column-selection |
| Create DataFrame from Dict | Create a pandas DataFrame from dictionary data and report its records, shape, and ordered column names. | https://www.tensortonic.com/problems/pandas-read-csv |
| Rename Columns | Rename selected pandas DataFrame columns from an old-to-new mapping and return the updated records. | https://www.tensortonic.com/problems/pandas-rename-columns |
| Replace Values | Create a DataFrame, replace all occurrences of the old value with the new value in the specified column, and count how many replacements were made. | https://www.tensortonic.com/problems/pandas-replace-values |
| Resetting Index | Set a pandas column as the index, then restore the default integer index while retaining the original values. | https://www.tensortonic.com/problems/pandas-resetting-index |
| Setting Index | Set a named pandas DataFrame column as the index and report the resulting records and index metadata. | https://www.tensortonic.com/problems/pandas-setting-index |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/problems/relu |
| RMS Normalization | Implement row-wise RMS normalization in CUDA with sum-of-squares reduction, numerical stability, and learnable scaling. | https://www.tensortonic.com/problems/rms-norm |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Scaled Dot-Product Attention | Implement single-head scaled dot-product attention in CUDA with stable row-wise softmax and value aggregation. | https://www.tensortonic.com/problems/scaled-dot-product-attention |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/problems/sigmoid |
| Softmax | Implement numerically stable CUDA softmax over a vector using global maximum and normalization reductions. | https://www.tensortonic.com/problems/softmax |
| Sum of Array | Implement a multi-block CUDA sum reduction that combines partial block sums into one scalar output. | https://www.tensortonic.com/problems/sum-of-array |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/problems/swish |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/problems/tanh |
| Tiled Matrix Multiplication | Implement shared-memory tiled matrix multiplication in CUDA with synchronized tile loads and partial-tile handling. | https://www.tensortonic.com/problems/tiled-matmul |
| FlashAttention Forward | Implement fused FlashAttention forward in Triton with streamed key-value blocks and online softmax statistics. | https://www.tensortonic.com/problems/triton-flash-attention-fwd |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/problems/triton-fused-multiply-add |
| Grouped Program-ID Matmul | Implement grouped program-ID matrix multiplication in Triton to improve L2 reuse while preserving tail-safe tiled computation. | https://www.tensortonic.com/problems/triton-grouped-matmul |
| LayerNorm Backward | Implement LayerNorm backward in Triton with row-wise statistics and atomic gradients for scale and bias. | https://www.tensortonic.com/problems/triton-layernorm-backward |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/problems/triton-logsumexp |
| Online Softmax | Implement chunked online softmax in Triton with running maxima and denominators followed by a normalized output pass. | https://www.tensortonic.com/problems/triton-online-softmax |
| Split-K Matmul | Implement split-K matrix multiplication in Triton with parallel reduction slices, atomic output accumulation, and tail masks. | https://www.tensortonic.com/problems/triton-split-k-matmul |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/problems/triton-vector-addition |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-subtract |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/csorfolydaniel)
<!-- tensortonic:end -->
