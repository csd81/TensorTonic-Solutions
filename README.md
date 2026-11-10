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
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Run Deterministic Neural-Guided MCTS | Run deterministic AlphaGo Zero search over a finite two-player game tree using neural priors, values, PUCT, and backup. | https://www.tensortonic.com/problems/alphago-l05-neural-guided-mcts |
| Argmax | Implement a parallel CUDA argmax reduction that returns the lowest index when multiple elements share the maximum. | https://www.tensortonic.com/problems/argmax |
| Argmin | Implement a parallel CUDA argmin reduction that returns the lowest index when multiple elements share the minimum. | https://www.tensortonic.com/problems/argmin |
| Train a Tiny Micrograd MLP | Train a supplied scalar-output MLP with manual reverse-mode differentiation and deterministic gradient descent. | https://www.tensortonic.com/problems/autograd-l08-train-tiny-micrograd-mlp |
| Continuity of Activation Functions | Classify continuity and nondifferentiable points for common activation functions at supplied scalar inputs. | https://www.tensortonic.com/problems/calculus-activation-continuity |
| Derivatives of Common Activation Functions | Compute the analytical derivative of four activation functions: Sigmoid, Tanh, ReLU, and Swish. | https://www.tensortonic.com/problems/calculus-activation-derivatives |
| Backpropagation Through a Computation Graph | Run forward evaluation and reverse-mode backpropagation through a supplied scalar computation graph. | https://www.tensortonic.com/problems/calculus-backprop-graph |
| Chain Rule Through a 3-Layer Network | Apply the chain rule through a three-layer scalar network to compute intermediate activations and parameter gradients. | https://www.tensortonic.com/problems/calculus-chain-rule-3layer |
| Directional Derivative & Steepest Descent Direction | Compute a quadratic loss gradient, its directional derivative, and the normalized steepest-descent direction. | https://www.tensortonic.com/problems/calculus-directional-derivative |
| Forward-Mode Autodiff with Dual Numbers | Implement forward-mode automatic differentiation with dual numbers carrying both values and derivatives. | https://www.tensortonic.com/problems/calculus-dual-number-autodiff |
| Gradient of Multivariate Gaussian Log-PDF | Compute the gradient of a multivariate Gaussian log density with respect to its mean vector. | https://www.tensortonic.com/problems/calculus-gaussian-mean-gradient |
| Gradient Descent on a Non-Convex Loss | Run gradient descent on a nonconvex scalar loss and report the optimization path and final stationary behavior. | https://www.tensortonic.com/problems/calculus-gd-nonconvex |
| Finite Difference Gradient Checking | Estimate scalar derivatives with forward, backward, and central finite differences and compare their numerical errors. | https://www.tensortonic.com/problems/calculus-gradient-check |
| Gradient Clipping: Global Norm Rescaling | Compute the global gradient norm, rescale all gradients if the norm exceeds the threshold. | https://www.tensortonic.com/problems/calculus-gradient-clipping |
| Hessian Matrix & Convexity of Linear Regression Loss | Construct the linear-regression loss Hessian and determine convexity from its eigenvalues. | https://www.tensortonic.com/problems/calculus-hessian-convexity |
| Jacobian of a Vector-Valued Polynomial | Implement Jacobian of a Vector-Valued Polynomial, and return a 3x3 list of lists, with each entry rounded to 6 decimals. | https://www.tensortonic.com/problems/calculus-jacobian-computation |
| Jacobian of a Neural Network Layer | Implement Jacobian of a Neural Network Layer, and compute both analytically and via finite differences. | https://www.tensortonic.com/problems/calculus-jacobian-layer |
| JVP and VJP for a 2-Layer Network | Compute Jacobian-vector and vector-Jacobian products through a two-layer ReLU network without forming the full Jacobian. | https://www.tensortonic.com/problems/calculus-jvp-vjp |
| Lagrange Multipliers: Regularization as Constrained Optimization | Show that L2 regularization and constrained optimization (bounding the weight norm) produce the same solution. | https://www.tensortonic.com/problems/calculus-lagrange-regularization |
| Gradient of Multivariate Linear Regression Loss | Compute the multivariate linear-regression sum-of-squared-errors gradient with respect to weights and bias. | https://www.tensortonic.com/problems/calculus-linreg-gradient |
| Higher-Order Derivatives & Loss Curvature | Compute first and second derivatives for MSE, cross-entropy, and Huber losses to compare their local curvature. | https://www.tensortonic.com/problems/calculus-loss-curvature |
| Limit of a Learning Rate Schedule | Evaluate the long-run limit of an inverse-time learning-rate schedule from its initial rate and decay constant. | https://www.tensortonic.com/problems/calculus-lr-schedule-limit |
| Learning Rate Stability & Convergence Bounds | Derive the maximum stable learning rate for a quadratic objective and classify convergence from the chosen rate. | https://www.tensortonic.com/problems/calculus-lr-stability |
| Gradient of a Matrix-Valued Function | Differentiate every entry of a polynomial matrix-valued function and return its derivative matrix at a scalar point. | https://www.tensortonic.com/problems/calculus-matrix-function-gradient |
| Second-Order Partial Derivatives | Compute both pure second-order partial derivatives of a supplied bivariate polynomial at a requested point. | https://www.tensortonic.com/problems/calculus-second-order-partials |
| Sigmoid Boundedness via Squeeze Theorem | Verify sigmoid bounds numerically with a squeeze-theorem construction and expose its saturation behavior. | https://www.tensortonic.com/problems/calculus-sigmoid-squeeze |
| Softmax + Cross-Entropy Gradient | Compute the softmax probabilities, the cross-entropy loss, and the gradient two ways. | https://www.tensortonic.com/problems/calculus-softmax-ce-gradient |
| Softmax Numerical Stability & Log-Sum-Exp | Implement both naive softmax and the numerically stable version using the log-sum-exp trick. | https://www.tensortonic.com/problems/calculus-softmax-stability |
| Subgradients for Non-Smooth Functions | Compute valid subgradients for absolute value, ReLU, and maximum functions at smooth and nonsmooth points. | https://www.tensortonic.com/problems/calculus-subgradients |
| Taylor Approximation of a Loss Landscape | Build first-order (linear) and second-order (quadratic) Taylor approximations of a loss function at a given point. | https://www.tensortonic.com/problems/calculus-taylor-approximation |
| Hessian Tensor of a Vector-Valued Function | Compute the full Hessian tensor for a three-output polynomial function of three scalar variables. | https://www.tensortonic.com/problems/calculus-vector-valued-hessian |
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
| 2D Sinusoidal Positional Embedding | Build the 2D sin-cos positional embedding used by ViT-MAE, DINOv2, and similar models. | https://www.tensortonic.com/problems/cv-2d-sincos-pos-embed |
| Adaptive Average Pool 2D | Implement adaptive two-dimensional average pooling with variable, possibly overlapping regions for arbitrary output sizes. | https://www.tensortonic.com/problems/cv-adaptive-avg-pool-2d |
| Anchor Box Generation | Generate axis-aligned anchor boxes for a feature map of shape (feature h, feature w) at stride s. | https://www.tensortonic.com/problems/cv-anchor-generation |
| Average Pool 2D | Apply two-dimensional average pooling to an image with the requested kernel and stride, without padding. | https://www.tensortonic.com/problems/cv-avg-pool-2d |
| Bilinear Resize | Resize a grayscale image with corner-aligned bilinear interpolation matching PyTorch interpolation semantics. | https://www.tensortonic.com/problems/cv-bilinear-resize |
| Box Encode and Decode | Implement Faster-RCNN style box regression target encoding and decoding, using two arrays of axis-aligned boxes in xyxy format. | https://www.tensortonic.com/problems/cv-box-encode-decode |
| Per-Channel Mean and Std | Compute population means and standard deviations for every channel across a batch of images in NHWC layout. | https://www.tensortonic.com/problems/cv-channel-statistics |
| Complete IoU Loss | Compute Complete IoU loss for aligned boxes using overlap, center-distance, and aspect-ratio penalties. | https://www.tensortonic.com/problems/cv-ciou-loss |
| CLIP Cosine Retrieval | Rank corpus embeddings for each CLIP-style query by stable cosine similarity, including safe handling of zero vectors. | https://www.tensortonic.com/problems/cv-clip-cosine-retrieval |
| Multichannel 2D Convolution | Implement multichannel two-dimensional cross-correlation with output filters, optional bias, stride, and padding. | https://www.tensortonic.com/problems/cv-conv2d-multichannel |
| 2D Transposed Convolution | Implement two-dimensional transposed convolution matching PyTorch output layout, stride, padding, and optional bias. | https://www.tensortonic.com/problems/cv-conv2d-transpose |
| CutMix Augmentation | Apply deterministic CutMix by pasting a supplied image region and mixing labels according to the retained pixel area. | https://www.tensortonic.com/problems/cv-cutmix |
| Depthwise Separable Convolution | Factor image convolution into per-channel spatial filters and a pointwise convolution that mixes output channels. | https://www.tensortonic.com/problems/cv-depthwise-separable-conv |
| Dice Loss | Compute binary segmentation Dice loss from probability and target masks with the specified smoothing term. | https://www.tensortonic.com/problems/cv-dice-loss |
| Dilated 2D Convolution | Implement a 2D dilated cross-correlation (no kernel flip) on a single multi-channel image. | https://www.tensortonic.com/problems/cv-dilated-conv2d |
| Sigmoid Focal Loss | Implement mean binary sigmoid focal loss from raw logits with alpha balancing, gamma modulation, and stable probabilities. | https://www.tensortonic.com/problems/cv-focal-loss |
| FPN Top-Down Fusion | Fuse multiscale feature maps through top-down nearest-neighbor upsampling and aligned lateral addition. | https://www.tensortonic.com/problems/cv-fpn-fusion |
| Gaussian Blur 2D | Build a normalized separable Gaussian kernel and apply it to a grayscale image with the specified padding behavior. | https://www.tensortonic.com/problems/cv-gaussian-blur |
| Generalized IoU Loss | Compute mean Generalized IoU loss for aligned predicted and target boxes, including the smallest enclosing-box penalty. | https://www.tensortonic.com/problems/cv-giou-loss |
| Hungarian Matching | Find the minimum-cost one-to-one assignment between rows and columns of a rectangular cost matrix. | https://www.tensortonic.com/problems/cv-hungarian-matching |
| Image Normalize | Normalize each image channel by its supplied mean and standard deviation to produce standardized vision-model inputs. | https://www.tensortonic.com/problems/cv-image-normalize |
| Pairwise IoU | Compute the complete pairwise intersection-over-union matrix between two sets of axis-aligned boxes. | https://www.tensortonic.com/problems/cv-iou-pairwise |
| Max Pool 2D | Apply two-dimensional max pooling to an image with a configurable window and stride, without padding. | https://www.tensortonic.com/problems/cv-max-pool-2d |
| Mean IoU for Segmentation | Compute the mean Intersection-over-Union (mIoU) for a semantic segmentation prediction. | https://www.tensortonic.com/problems/cv-miou |
| MixUp Augmentation | MixUp augmentation forms a new training sample by taking a convex combination of two images and their labels using a mixing weight [0, 1]. | https://www.tensortonic.com/problems/cv-mixup |
| Non-Maximum Suppression | Implement greedy non-maximum suppression matching torchvision selection order and strict IoU threshold behavior. | https://www.tensortonic.com/problems/cv-nms |
| ViT Patchify | Split image batches into row-major non-overlapping patches and flatten each patch in spatial-then-channel order. | https://www.tensortonic.com/problems/cv-patchify |
| RGB to Grayscale | Convert an RGB image to grayscale with luminance-weighted color channels for classical computer vision preprocessing. | https://www.tensortonic.com/problems/cv-rgb-to-grayscale |
| ROI Align | Implement aligned ROI pooling with bilinear sampling over feature-map coordinates and fixed-size output bins. | https://www.tensortonic.com/problems/cv-roi-align |
| Sobel Edge Detection | Apply horizontal and vertical Sobel filters to a grayscale image with one-pixel zero padding and return edge gradients. | https://www.tensortonic.com/problems/cv-sobel-edges |
| Soft Non-Maximum Suppression | Apply Gaussian Soft-NMS by repeatedly selecting the strongest box and decaying overlapping box scores by IoU. | https://www.tensortonic.com/problems/cv-soft-nms |
| Top-K Classification Accuracy | Given logits of shape (N, K) and integer targets of shape (N,) with each target in [0, K), compute the top-k classification accuracy. | https://www.tensortonic.com/problems/cv-topk-accuracy |
| ViT Multi-Head Self-Attention | Implement Vision Transformer multi-head self-attention with combined QKV projection, scaled softmax, and output projection. | https://www.tensortonic.com/problems/cv-vit-attention-block |
| Window Partition and Reverse | Partition BHWC feature maps into row-major Swin Transformer windows and reconstruct the original tensor exactly. | https://www.tensortonic.com/problems/cv-window-partition |
| Zero Pad and Center Crop | Zero-pad a two-dimensional grayscale image on every side, then extract a centered crop with the requested dimensions. | https://www.tensortonic.com/problems/cv-zero-pad-and-center-crop |
| Tiny U-Net Forward Pass | Assemble a time-conditioned U-Net from scratch with residual blocks, downsampling, upsampling and skip connections. | https://www.tensortonic.com/problems/diffusion-l03-tiny-unet-forward |
| Transformer Decoder Block | Implement a NumPy Transformer decoder block with masked self-attention, cross-attention, and feed-forward layers. | https://www.tensortonic.com/problems/dl-decoder-block |
| Transformer Encoder Block | Implement a NumPy Transformer encoder block with self-attention, residual connections, normalization, and feed-forward layers. | https://www.tensortonic.com/problems/dl-encoder-block |
| GAN Training Step | Implement a single forward-pass training step for a Generative Adversarial Network (GAN). | https://www.tensortonic.com/problems/dl-gan-training |
| LSTM Cell | Implement a Long Short-Term Memory (LSTM) cell that processes a sequence using four gates: forget, input, output, and a candidate cell state. | https://www.tensortonic.com/problems/dl-lstm-cell |
| Multi-Head Attention | Implement NumPy multi-head attention with QKV projections, optional masking, stable softmax, and output projection. | https://www.tensortonic.com/problems/dl-multi-head-attention |
| Sequence-to-Sequence Forward Pass | Implement the forward pass of a sequence-to-sequence (seq2seq) model using a simple RNN for both the encoder and decoder. | https://www.tensortonic.com/problems/dl-sequence-to-sequence |
| Vision Transformer Patch Embedding | Convert an image into non-overlapping Vision Transformer patch embeddings with a learned linear projection. | https://www.tensortonic.com/problems/dl-vision-transformer |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Dropout | Implement inverted dropout in CUDA using a supplied binary mask and scaling retained elements by the keep probability. | https://www.tensortonic.com/problems/dropout |
| Gaussian Naive Bayes | Fit Gaussian Naive Bayes class statistics and predict labels from priors and feature likelihoods. | https://www.tensortonic.com/problems/gaussian-naive-bayes |
| Matrix-Vector Multiplication | Implement row-major CUDA matrix-vector multiplication with one thread computing each output row. | https://www.tensortonic.com/problems/gemv |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Hadamard Product | Implement elementwise matrix multiplication in CUDA using a two-dimensional grid and row-major bounds-checked indexing. | https://www.tensortonic.com/problems/hadamard-product |
| Implement Autoregressive Decoding with a KV Cache | Decode autoregressively with an append-only key-value cache so previously processed tokens are never recomputed. | https://www.tensortonic.com/problems/inference-autoregressive-kv-cache |
| Implement Block-Wise FP8 Quantization | Compute one scale per block from that block's own absolute maximum, and scale each block into the [-fp8 max, fp8 max] range with saturation. | https://www.tensortonic.com/problems/inference-blockwise-fp8-quantization |
| Simulate Chunked Prefill Scheduling | Simulate chunked prefill scheduling with strict decode priority, bounded token budgets, and request arrival order. | https://www.tensortonic.com/problems/inference-chunked-prefill-scheduling |
| Simulate Continuous Batching | Simulate continuous batching by admitting requests in arrival order and replacing completed sequences during decoding. | https://www.tensortonic.com/problems/inference-continuous-batching |
| Simulate Disaggregated Prefill and Decode Serving | Simulate Disaggregated Prefill and Decode Serving, and return selected prefill and decode replica indices, TTFT, and completion time for every request. | https://www.tensortonic.com/problems/inference-disaggregated-prefill-decode |
| Implement Dynamic Request Batching | Simulate a dynamic batching scheduler: requests arrive over time and are grouped into a forming batch. | https://www.tensortonic.com/problems/inference-dynamic-batching |
| Implement Expert Parallel Routing Across GPUs | Route token-expert assignments to destination GPU ranks with stable ordering, send counts, and an inverse permutation. | https://www.tensortonic.com/problems/inference-expert-parallel-routing |
| Implement FlashAttention with Online Softmax | Implement tiled attention with online softmax statistics that agrees numerically with dense scaled attention. | https://www.tensortonic.com/problems/inference-flash-attention-online-softmax |
| Implement Grouped-Query Attention (GQA) | Implement grouped-query attention by mapping query heads onto fewer key-value heads with validated head divisibility. | https://www.tensortonic.com/problems/inference-grouped-query-attention |
| Implement Group-Wise INT4 Quantization | Quantize two-dimensional weights in contiguous groups with signed INT4 values and one scale per group. | https://www.tensortonic.com/problems/inference-groupwise-int4-quantization |
| Calculate KV Cache Memory for MHA, MQA, GQA, and MLA | Compute the total KV-cache memory, in bytes, for a full sequence under four attention variants: MHA, MQA, GQA, and MLA. | https://www.tensortonic.com/problems/inference-kv-cache-memory |
| Calculate P50, P95, and P99 Inference Latency | Compute the 50th, 95th, and 99th percentile of a set of observed inference latencies, using linear interpolation between the two closest ranks. | https://www.tensortonic.com/problems/inference-latency-percentiles |
| Estimate Model Memory and Minimum GPU Count | Estimate model-serving memory from weights, key-value cache, activations, overhead, and per-GPU usable capacity. | https://www.tensortonic.com/problems/inference-model-memory-gpu-count |
| Implement MoE Token Dispatch and Expert Aggregation | Run each expert's feed-forward computation only on the tokens routed to it, and scatter-add the routing-weighted results back into token order. | https://www.tensortonic.com/problems/inference-moe-dispatch-aggregation |
| Implement Sparse MoE Top-k Expert Routing | Select the top k highest-scoring experts per token and compute routing weights as a softmax over only those selected logits. | https://www.tensortonic.com/problems/inference-moe-top-k-routing |
| Implement Multi-Head Attention (MHA) | Split into h heads, run scaled dot-product attention per head with an optional causal mask, concatenate the heads, and apply an output projection. | https://www.tensortonic.com/problems/inference-multi-head-attention |
| Implement Multi-Head Latent Attention (MLA) | Implement Multi-Head Latent Attention (MLA), and return both the projected attention output and the compressed latent tensor. | https://www.tensortonic.com/problems/inference-multi-head-latent-attention |
| Implement Multi-Query Attention (MQA) | Implement multi-query attention with separate query heads, shared key-value heads, optional masking, and output projection. | https://www.tensortonic.com/problems/inference-multi-query-attention |
| Implement PagedAttention Block Allocation | Assign each sequence enough fixed-size blocks to hold its tokens, drawn from the free pool in order. | https://www.tensortonic.com/problems/inference-paged-attention-allocation |
| Implement Per-Channel Weight Quantization | Quantize weights independently along a selected channel axis, then dequantize with channel-specific scales. | https://www.tensortonic.com/problems/inference-per-channel-quantization |
| Analyze Prefill and Decode with the Roofline Model | Analyze the prefill and decode phases of transformer inference using the roofline performance model. | https://www.tensortonic.com/problems/inference-prefill-decode-roofline |
| Implement Prefix Cache Matching and Reuse | Find the longest exactly-matching prefix, measured in whole blocks, among all candidates. | https://www.tensortonic.com/problems/inference-prefix-cache-reuse |
| Implement Rotary Position Embeddings for Decoding | Apply rotary position embeddings to query and key tensors at supplied absolute decoding positions. | https://www.tensortonic.com/problems/inference-rotary-position-embeddings |
| Implement Scaled Dot-Product Attention | Implement batched scaled dot-product attention for self- and cross-attention with optional masks and stable softmax. | https://www.tensortonic.com/problems/inference-scaled-dot-product-attention |
| Compute TTFT, TPOT, ITL, and Token Throughput | Compute TTFT, TPOT, inter-token latency, and token throughput from request arrival and token timestamp data. | https://www.tensortonic.com/problems/inference-serving-metrics |
| Implement Speculative Decoding and Token Verification | Verify speculative draft tokens against target probabilities and sample the corrected or bonus continuation exactly. | https://www.tensortonic.com/problems/inference-speculative-decoding-verification |
| Implement Symmetric INT8 Quantization | Quantize finite tensors to symmetric INT8 values with an absolute-maximum scale and reconstruct dequantized outputs. | https://www.tensortonic.com/problems/inference-symmetric-int8-quantization |
| Calculate Tensor Parallel Communication Cost | Calculate tensor-parallel collective communication volume and latency from message size, world size, bandwidth, and calls. | https://www.tensortonic.com/problems/inference-tensor-parallel-communication |
| Optimize Throughput and Cost Under a Latency SLO | Select the cheapest configuration that meets both a required throughput and a maximum P95 latency SLO. | https://www.tensortonic.com/problems/inference-throughput-cost-slo |
| Implement Temperature, Top-k, and Top-p Sampling | Sample tokens from batched logits using temperature scaling, top-k filtering, top-p filtering, and deterministic randomness. | https://www.tensortonic.com/problems/inference-token-sampling |
| Functional Array Update | Build a weighted histogram in JAX with indexed updates that return new arrays, so repeated bins accumulate and out-of-range indices are ignored under jit. | https://www.tensortonic.com/problems/jax-l01-functional-array-update |
| Beam Search Step | Write one beam search step in JAX that adds log probs, takes the top beams over all candidates and reorders histories and caches. | https://www.tensortonic.com/problems/jax-l08-beam-search-step |
| Ring All-Reduce with ppermute | Implement ring all-reduce in JAX as reduce-scatter then all-gather with lax.ppermute inside shard_map, matching psum exactly on 8 devices. | https://www.tensortonic.com/problems/jax-l09-ring-all-reduce |
| L1 Normalization | Implement CUDA L1 vector normalization by reducing absolute values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l1-normalize |
| L2 Normalization | Implement CUDA L2 vector normalization by reducing squared values and dividing each element by the resulting norm. | https://www.tensortonic.com/problems/l2-normalize |
| Cholesky Decomposition | Factor a symmetric positive-definite NumPy matrix into a lower-triangular matrix and its transpose. | https://www.tensortonic.com/problems/la-cholesky |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with explicit handling for zero-norm inputs. | https://www.tensortonic.com/problems/la-cosine-similarity |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/problems/la-dot-product |
| Eigendecomposition | Compute eigenvalues and aligned eigenvectors for a square NumPy matrix and verify the decomposition relationship. | https://www.tensortonic.com/problems/la-eigendecomposition |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors from the square root of summed squared differences. | https://www.tensortonic.com/problems/la-euclidean-distance |
| Gram-Schmidt Orthogonalization | Given k linearly independent vectors in R^n, the Gram-Schmidt process builds an orthonormal basis that spans exactly the same space. | https://www.tensortonic.com/problems/la-gram-schmidt |
| Hadamard Product | Compute elementwise multiplication between two same-shaped NumPy matrices to produce their Hadamard product. | https://www.tensortonic.com/problems/la-hadamard-product |
| Least Squares Solution | Solve a full-column-rank least-squares system and return the coefficient vector minimizing residual norm. | https://www.tensortonic.com/problems/la-least-squares |
| Linear Combination | Compute a weighted linear combination of equal-length NumPy vectors using one aligned scalar coefficient per vector. | https://www.tensortonic.com/problems/la-linear-combination |
| Low-Rank Approximation | Construct the best rank-k matrix approximation from truncated singular values and singular vectors. | https://www.tensortonic.com/problems/la-low-rank-approx |
| LU Decomposition | Factor a square matrix into lower and upper triangular matrices using the specified no-pivot LU procedure. | https://www.tensortonic.com/problems/la-lu-decomposition |
| Mahalanobis Distance | Compute Mahalanobis distance between a point and a distribution using its mean and covariance matrix. | https://www.tensortonic.com/problems/la-mahalanobis |
| Matrix Determinant | Compute the determinant of a square NumPy matrix as a scalar measure of invertibility and volume scaling. | https://www.tensortonic.com/problems/la-matrix-determinant |
| Matrix Multiply | Multiply compatible NumPy matrices and preserve the dtype produced by NumPy type-promotion rules. | https://www.tensortonic.com/problems/la-matrix-multiply |
| Matrix Rank | Compute the rank of a rectangular NumPy matrix from its number of linearly independent directions. | https://www.tensortonic.com/problems/la-matrix-rank |
| Matrix Trace | Compute the trace of a square NumPy matrix by summing its main-diagonal elements with numeric dtype support. | https://www.tensortonic.com/problems/la-matrix-trace |
| Matrix Transpose | Transpose a rectangular NumPy matrix by swapping its row and column axes without changing element values. | https://www.tensortonic.com/problems/la-matrix-transpose |
| Matrix-Vector Multiply | Multiply a NumPy matrix by a compatible vector, producing one row-wise dot product per output element. | https://www.tensortonic.com/problems/la-matrix-vector-multiply |
| Orthogonal Projection Matrix | Construct the orthogonal projection matrix onto the column space of a full-column-rank matrix. | https://www.tensortonic.com/problems/la-orthogonal-projection |
| Outer Product | Compute the NumPy outer product of two vectors as a matrix containing every pairwise element multiplication. | https://www.tensortonic.com/problems/la-outer-product |
| PCA from Scratch | Implement PCA from scratch by centering data, decomposing covariance, and projecting onto leading components. | https://www.tensortonic.com/problems/la-pca-from-scratch |
| Moore-Penrose Pseudoinverse | Compute the Moore-Penrose pseudoinverse of rectangular or singular matrices using singular-value decomposition. | https://www.tensortonic.com/problems/la-pseudoinverse |
| QR Decomposition | Compute its QR decomposition: factor A into an orthogonal matrix Q and an upper triangular matrix R. | https://www.tensortonic.com/problems/la-qr-decomposition |
| RBF Kernel Matrix | Compute the pairwise radial-basis-function kernel matrix from sample vectors and a positive bandwidth. | https://www.tensortonic.com/problems/la-rbf-kernel |
| Scaled Dot-Product Attention | Implement the scaled dot-product attention mechanism from the Transformer architecture. | https://www.tensortonic.com/problems/la-scaled-attention |
| Solve Linear System | Solve an invertible square linear system for the unique vector satisfying the matrix equation. | https://www.tensortonic.com/problems/la-solve-linear-system |
| SVD Components | Compute singular values and aligned singular vectors for a possibly rank-deficient matrix using NumPy. | https://www.tensortonic.com/problems/la-svd-components |
| Vector Norms | Compute L1, L2, and infinity norms for a one-dimensional NumPy vector and return them in a float64 array. | https://www.tensortonic.com/problems/la-vector-norms |
| Vector Projection | The vector projection of u onto v is the component of u that lies exactly along the direction of v. | https://www.tensortonic.com/problems/la-vector-projection |
| Whitening Transform | Center and whiten a data matrix so the transformed features have zero mean and identity covariance. | https://www.tensortonic.com/problems/la-whitening |
| Layer Normalization | Implement fused row-wise LayerNorm in CUDA with shared-memory mean and variance reduction, affine scale, and bias. | https://www.tensortonic.com/problems/layer-norm |
| L-BFGS Two-Loop Recursion | Implement the L-BFGS two-loop recursion to transform a gradient using stored correction-vector history. | https://www.tensortonic.com/problems/lbfgs-two-loop |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Matrix Addition | Implement elementwise matrix addition in CUDA with a two-dimensional grid, row-major indexing, and bounds checks. | https://www.tensortonic.com/problems/matrix-addition |
| Matrix Multiplication | Implement row-major matrix multiplication in CUDA with one thread per output element and inner-product accumulation. | https://www.tensortonic.com/problems/matrix-multiplication |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Max of Array | Implement a multi-block CUDA maximum reduction that combines block-local maxima into one scalar output. | https://www.tensortonic.com/problems/max-of-array |
| Max Pool 2D | Implement 2D max pooling in CUDA with one thread per output, configurable windows and strides, and row-major indexing. | https://www.tensortonic.com/problems/max-pool-2d |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Mean and Variance | Compute population mean and variance in CUDA with device-side reductions and synchronized scalar outputs. | https://www.tensortonic.com/problems/mean-variance |
| Min of Array | Implement a multi-block CUDA minimum reduction that combines block-local minima into one scalar output. | https://www.tensortonic.com/problems/min-of-array |
| Distance Metrics | Compute Euclidean, Manhattan, cosine, Chebyshev, and Minkowski distances between numeric vectors. | https://www.tensortonic.com/problems/ml-distance-metrics |
| Feature Scaling | Implement column-wise min-max scaling and z-score standardization while handling constant features safely. | https://www.tensortonic.com/problems/ml-feature-scaling |
| Gaussian Mixture Models (EM) | Implement a diagonal-covariance Gaussian Mixture Model using EM updates for responsibilities, means, variances, and mixture weights. | https://www.tensortonic.com/problems/ml-gmm |
| Linear Regression from Scratch | Train linear regression from scratch with mean squared error gradients for weights and bias. | https://www.tensortonic.com/problems/ml-linear-regression-from-scratch |
| PCA from Scratch | Implement PCA by centering data, eigendecomposing the covariance matrix, and projecting onto the leading components. | https://www.tensortonic.com/problems/ml-pca |
| Stacking Ensemble | Build a stacking classifier using cross-validated predictions from decision-stump and KNN base models to train a logistic meta-learner. | https://www.tensortonic.com/problems/ml-stacking |
| t-SNE Affinity Matrix | Compute the symmetric t-SNE affinity matrix using pairwise distances, perplexity-matched bandwidths, and conditional probabilities. | https://www.tensortonic.com/problems/ml-tsne |
| Bag of Words | Build an alphabetical vocabulary and document-term count matrix from a corpus of tokenized documents. | https://www.tensortonic.com/problems/nlp-bag-of-words |
| Beam Search Decoding | Decode token sequences with beam search over supplied next-token probabilities, cumulative scores, and stable tie handling. | https://www.tensortonic.com/problems/nlp-beam-search-decoding |
| Byte Pair Encoding | Byte Pair Encoding (BPE) is a subword tokenization algorithm that iteratively merges the most frequent pair of adjacent symbols in a corpus. | https://www.tensortonic.com/problems/nlp-bpe-algorithm |
| Greedy Sequence Tagging | Assign the highest-scoring tag independently at each token position from emission scores and ordered tag names. | https://www.tensortonic.com/problems/nlp-crf-layer |
| Document Ranking | Score each document by combining term frequency saturation, inverse document frequency, and document length normalization. | https://www.tensortonic.com/problems/nlp-document-ranking-bm25 |
| Edit Distance | Compute Levenshtein distance as the minimum insertions, deletions, and substitutions needed to transform one string into another. | https://www.tensortonic.com/problems/nlp-edit-distance |
| Embedding Similarity | Rank candidate embeddings against a query using the selected similarity metric and return the top-k indices. | https://www.tensortonic.com/problems/nlp-embedding-similarity |
| Evaluation Metrics (BLEU, ROUGE) | Compute corpus BLEU and ROUGE overlap metrics for generated text using the specified token and n-gram rules. | https://www.tensortonic.com/problems/nlp-evaluation-metrics-bleu-rouge |
| FastText | Learn FastText-style word representations from tokenized documents and compose embeddings from character n-grams. | https://www.tensortonic.com/problems/nlp-fasttext |
| GloVe | Build a vocabulary V sorted alphabetically, and construct a co-occurrence matrix X with 1/d distance weighting within window size w. | https://www.tensortonic.com/problems/nlp-glove |
| Language Modeling | Build a Laplace-smoothed bigram language model from tokenized sentences and score supplied token sequences. | https://www.tensortonic.com/problems/nlp-language-modeling |
| Masked Language Modeling | In masked language modeling, certain tokens in a sentence are replaced with a special [MASK] token. | https://www.tensortonic.com/problems/nlp-masked-language-modeling |
| Named Entity Recognition | Tag tokenized sentences with dictionary-backed named entity labels while preserving sentence and token order. | https://www.tensortonic.com/problems/nlp-ner |
| Next Sentence Prediction | Implement Next Sentence Prediction, and return True if the Jaccard similarity is greater than or equal to the given threshold, and False otherwise. | https://www.tensortonic.com/problems/nlp-next-sentence-prediction |
| N-gram Language Model | Build a Laplace-smoothed n-gram language model from tokenized training sentences and score test sequences. | https://www.tensortonic.com/problems/nlp-ngram-language-model |
| Padding and Truncation | Pad or truncate integer token sequences to a fixed length with configurable padding value and alignment direction. | https://www.tensortonic.com/problems/nlp-padding-truncation |
| Perplexity Computation | Compute sequence perplexity from token IDs and precomputed next-token logits using stable log probabilities. | https://www.tensortonic.com/problems/nlp-perplexity-computation |
| POS Tagging | Assign part-of-speech tags using each word's most frequent training tag and a deterministic fallback for unseen words. | https://www.tensortonic.com/problems/nlp-pos-tagging |
| Regex Patterns for NLP | Implement Regex Patterns for NLP, and return the list of matched strings in order of appearance. | https://www.tensortonic.com/problems/nlp-regex-patterns |
| Sentence Embeddings | Create fixed-size sentence embeddings with mean, maximum, or weighted pooling over aligned token vectors. | https://www.tensortonic.com/problems/nlp-sentence-embeddings |
| Sentence Segmentation | Segment text into sentences with rule-based punctuation boundaries while preserving supported abbreviations. | https://www.tensortonic.com/problems/nlp-sentence-segmentation |
| Span Extraction | Select the highest-scoring valid answer span from token-level start and end logits under a length limit. | https://www.tensortonic.com/problems/nlp-span-extraction |
| Spell Correction | Correct misspelled words by minimizing Levenshtein distance, then breaking ties by frequency and lexical order. | https://www.tensortonic.com/problems/nlp-spell-correction |
| Stemming vs Lemmatization | Produce a rule-based stem and dictionary-backed lemma for a supplied word with deterministic fallback behavior. | https://www.tensortonic.com/problems/nlp-stemming-lemmatization |
| Subword Tokenization | Segment a word with greedy longest-match WordPiece tokenization against a supplied subword vocabulary. | https://www.tensortonic.com/problems/nlp-subword-tokenization |
| Text Classification | Train a Laplace-smoothed Multinomial Naive Bayes text classifier and predict labels for test documents. | https://www.tensortonic.com/problems/nlp-text-classification |
| Text Deduplication (MinHash) | Detect near-duplicate documents with token shingles, deterministic MinHash signatures, and a similarity threshold. | https://www.tensortonic.com/problems/nlp-text-deduplication-minhash |
| Text Generation Sampling | Apply temperature scaling to control randomness, convert to probabilities via softmax, and optionally filter to keep only the top-k most likely tokens. | https://www.tensortonic.com/problems/nlp-text-generation-sampling |
| Text Normalization | Normalize text by applying an ordered pipeline of lowercase, punctuation, whitespace, digit, and accent transformations. | https://www.tensortonic.com/problems/nlp-text-normalization |
| TF-IDF | Build a vocabulary V sorted alphabetically and compute the TF-IDF score for every term in every document. | https://www.tensortonic.com/problems/nlp-tfidf |
| Token Classification | In token classification, each token in a sequence is assigned a label from a predefined set. | https://www.tensortonic.com/problems/nlp-token-classification |
| Tokenization | Tokenize raw text by scanning characters into words, punctuation marks, and supported special token forms. | https://www.tensortonic.com/problems/nlp-tokenization |
| Tokenizer Training | Build a subword vocabulary by iteratively merging the most frequent adjacent token pair. | https://www.tensortonic.com/problems/nlp-tokenizer-training-bpe |
| Vocabulary Building | Build a bounded token vocabulary from a corpus and encode every sentence as an integer sequence. | https://www.tensortonic.com/problems/nlp-vocabulary-building |
| Word2Vec | Train Skip-gram with Negative Sampling word embeddings from a corpus of tokenized documents. | https://www.tensortonic.com/problems/nlp-word2vec |
| Gradient Clipping | Train a linear regression model using full-batch gradient descent under three different gradient clipping strategies, and compare their loss curves. | https://www.tensortonic.com/problems/optim-grad-clipping |
| L-BFGS | Compare gradient descent and L-BFGS on identical MSE linear-regression data from zero-initialized weights. | https://www.tensortonic.com/problems/optim-lbfgs |
| Minimum of a Univariate Quadratic | Compute the minimizer and minimum value of a strictly convex univariate quadratic from its coefficients. | https://www.tensortonic.com/problems/optim-quadratic-minimum |
| Optimizer Showdown | Implement and compare three optimizers on binary classification using logistic regression with mini-batch training. | https://www.tensortonic.com/problems/optim-showdown |
| Outer Product | Compute a vector outer product in CUDA with a two-dimensional grid, row-major output, and bounds-checked indexing. | https://www.tensortonic.com/problems/outer-product |
| Aggregation Functions | Implement Aggregation Functions, and return a dict mapping each function name to a dict of group label to aggregated value. | https://www.tensortonic.com/problems/pandas-aggregation-functions |
| Apply Custom Transforms | Apply a named transformation to one pandas column and store the result in a new derived column. | https://www.tensortonic.com/problems/pandas-apply-custom-functions |
| Boolean Indexing | Filter pandas rows by a numeric column threshold and return the matching records with their original column order. | https://www.tensortonic.com/problems/pandas-boolean-indexing |
| Change Data Types | Create a DataFrame, convert the specified column to the target type, and return the dtypes before and after conversion. | https://www.tensortonic.com/problems/pandas-change-dtypes |
| Column Selection | Create a pandas DataFrame from dictionary data and extract one named column as an ordered list. | https://www.tensortonic.com/problems/pandas-column-selection |
| Concatenate DataFrames | Concatenate multiple pandas DataFrames vertically and return the combined records with a reset index. | https://www.tensortonic.com/problems/pandas-concatenate-dataframes |
| Cross Tabulation | Create a DataFrame and compute a cross-tabulation (frequency table) showing how often each combination of values co-occurs. | https://www.tensortonic.com/problems/pandas-cross-tabulation |
| Data Types Overview | Create a pandas DataFrame and report each column dtype together with counts for every unique dtype. | https://www.tensortonic.com/problems/pandas-data-types |
| Drop Duplicates | Create a DataFrame, remove duplicate rows, and return the cleaned result along with counts of rows before and after deduplication. | https://www.tensortonic.com/problems/pandas-drop-duplicates |
| GroupBy Basics | Create a DataFrame and compute the sum, mean, and count of the value column for each group. | https://www.tensortonic.com/problems/pandas-groupby-basics |
| Handle Missing Values | Create a pandas DataFrame, count missing entries per column, and replace every null with a supplied fill value. | https://www.tensortonic.com/problems/pandas-handle-missing |
| Head and Tail Operations | Create a pandas DataFrame and return the requested first and last rows as record-oriented dictionaries. | https://www.tensortonic.com/problems/pandas-head-tail |
| Inspect DataFrame Shape | Create a DataFrame and return its structural properties: row count, column count, column names, data types, and total number of values. | https://www.tensortonic.com/problems/pandas-inspect-shape |
| Loc vs iLoc | Create a DataFrame and use positional indexing to extract: the single element, the full row, and the full column. | https://www.tensortonic.com/problems/pandas-loc-iloc |
| Melt Wide to Long | Reshape a pandas DataFrame from wide to long format using selected identifier and value columns. | https://www.tensortonic.com/problems/pandas-melt-wide-to-long |
| Merge DataFrames | Use two dictionaries of column data and a key column present in both, create two DataFrames and merge them on the key column using a specified join type. | https://www.tensortonic.com/problems/pandas-merge-dataframes |
| Multi-Column Selection | Create a pandas DataFrame and select an ordered subset of named columns without changing row order. | https://www.tensortonic.com/problems/pandas-multi-column-selection |
| Multi-Level GroupBy | Create a DataFrame, group by all specified columns, apply the aggregation, and return the result as a flat table. | https://www.tensortonic.com/problems/pandas-multi-level-groupby |
| Pivot Tables | Build a pandas pivot table with selected index, columns, values, aggregation, and zero-filled missing combinations. | https://www.tensortonic.com/problems/pandas-pivot-tables |
| Create DataFrame from Dict | Create a pandas DataFrame from dictionary data and report its records, shape, and ordered column names. | https://www.tensortonic.com/problems/pandas-read-csv |
| Rename Columns | Rename selected pandas DataFrame columns from an old-to-new mapping and return the updated records. | https://www.tensortonic.com/problems/pandas-rename-columns |
| Replace Values | Create a DataFrame, replace all occurrences of the old value with the new value in the specified column, and count how many replacements were made. | https://www.tensortonic.com/problems/pandas-replace-values |
| Resetting Index | Set a pandas column as the index, then restore the default integer index while retaining the original values. | https://www.tensortonic.com/problems/pandas-resetting-index |
| Setting Index | Set a named pandas DataFrame column as the index and report the resulting records and index metadata. | https://www.tensortonic.com/problems/pandas-setting-index |
| Unstack Long to Wide | Implement Unstack Long to Wide, and return a dict of lists representing the wide-format DataFrame. | https://www.tensortonic.com/problems/pandas-unstack-long-to-wide |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions, with the formula and tests in the browser. | https://www.tensortonic.com/problems/positional-encoding |
| A/B Test Significance | Perform a two-proportion z-test to determine if the treatment conversion rate is significantly different from the control. | https://www.tensortonic.com/problems/probstat-ab-test-significance |
| ANOVA F-Statistic | Perform a one-way ANOVA across multiple numeric groups to test whether any population means differ. | https://www.tensortonic.com/problems/probstat-anova-f-test |
| Basic Probability Rules | Compute union, complement, and exclusive-event probabilities from two event probabilities and their intersection. | https://www.tensortonic.com/problems/probstat-basic-probability-rules |
| Bayes' Theorem | Compute a posterior probability from a prior, likelihood, and false-positive rate using Bayes' theorem. | https://www.tensortonic.com/problems/probstat-bayes-theorem |
| Bayesian Updating | Update a Beta prior with Binomial observations and report posterior parameters plus prior and posterior means. | https://www.tensortonic.com/problems/probstat-bayesian-updating |
| Binomial Distribution | Compute the full probability mass function, summary statistics, and the probability of at least a given number of successes. | https://www.tensortonic.com/problems/probstat-binomial-distribution |
| Bootstrap Confidence Interval | Estimate a sample mean and percentile bootstrap confidence interval from deterministic resamples. | https://www.tensortonic.com/problems/probstat-bootstrap-ci |
| Chi-Square Goodness of Fit | Perform a chi-square goodness-of-fit test to determine whether the observed distribution differs significantly from the expected one. | https://www.tensortonic.com/problems/probstat-chi-square-test |
| Central Limit Theorem | Compute the sample mean, standard error, and a confidence interval for the population mean using the CLT normal approximation. | https://www.tensortonic.com/problems/probstat-clt-confidence-interval |
| Conditional Probability | Compute both directional conditional probabilities from two event probabilities and their intersection. | https://www.tensortonic.com/problems/probstat-conditional-probability |
| Correlation Significance Test | Compute Pearson correlation and test whether the observed relationship differs significantly from zero. | https://www.tensortonic.com/problems/probstat-correlation-significance |
| Discrete Moments | Given the values and probabilities of a discrete random variable, compute its first two moments, variance, and standard deviation. | https://www.tensortonic.com/problems/probstat-discrete-moments |
| Exponential Distribution | Compute the PDF, CDF, survival probability, mean, and variance of an exponential distribution. | https://www.tensortonic.com/problems/probstat-exponential-distribution |
| Independence Testing | Determine whether two events are independent from their marginal probabilities and intersection probability. | https://www.tensortonic.com/problems/probstat-independence-testing |
| Mean, Median, Mode | Compute mean, median, and a deterministically selected mode for a one-dimensional numeric sample. | https://www.tensortonic.com/problems/probstat-mean-median-mode |
| Maximum Likelihood Estimation | Estimate the maximum-likelihood mean and standard deviation for observations from a normal distribution. | https://www.tensortonic.com/problems/probstat-mle-normal |
| Monte Carlo Pi Estimation | Estimate pi by sampling points in a unit square and measuring the fraction inside the quarter circle. | https://www.tensortonic.com/problems/probstat-monte-carlo-pi |
| Normal Distribution | Compute the z-score, cumulative probability (CDF), probability density (PDF), and the probability of falling within one standard deviation of the mean. | https://www.tensortonic.com/problems/probstat-normal-distribution |
| P-Values and Significance | Compute one-sided or two-sided p-values from the requested reference distribution and classify significance thresholds. | https://www.tensortonic.com/problems/probstat-p-value-analysis |
| Compute Pearson Correlation Matrix | Compute the complete Pearson correlation matrix across dataset features with stable handling of numeric inputs. | https://www.tensortonic.com/problems/probstat-pearson-correlation |
| Percentiles / Quantiles | Compute requested data percentiles with linear interpolation using NumPy-compatible quantile semantics. | https://www.tensortonic.com/problems/probstat-percentiles |
| Permutations and Combinations | Compute factorial, permutation, and combination counts for supplied nonnegative integers. | https://www.tensortonic.com/problems/probstat-permutations-combinations |
| Poisson Distribution | Compute Poisson probabilities, cumulative mass through a cutoff, and the probability of zero events. | https://www.tensortonic.com/problems/probstat-poisson-distribution |
| Sample Variance & Standard Deviation | Compute unbiased sample variance and standard deviation with Bessel correction for a numeric sample. | https://www.tensortonic.com/problems/probstat-sample-var-std |
| Sampling Distribution of the Mean | Compute the parameters of the sampling distribution of the sample mean and the probability that the sample mean falls below a threshold. | https://www.tensortonic.com/problems/probstat-sampling-distribution |
| Skewness and Kurtosis | Compute sample skewness and excess kurtosis, then classify the distribution's asymmetry and tail shape. | https://www.tensortonic.com/problems/probstat-skewness-kurtosis |
| Standard Error Comparison | Given multiple samples, compute the standard error of the mean for each sample and the average standard error across all samples. | https://www.tensortonic.com/problems/probstat-standard-errors |
| Stratified Sampling | Compute proportional allocation of a total sample and the resulting stratified estimator's mean and standard error. | https://www.tensortonic.com/problems/probstat-stratified-sampling |
| T-Test (One Sample) | Given raw sample data and a hypothesized mean, perform a one-sample t-test when the population standard deviation is unknown. | https://www.tensortonic.com/problems/probstat-t-test-one-sample |
| Z-Test (One Sample) | Perform a two-tailed one-sample z-test from a sample mean, hypothesized mean, known deviation, and sample size. | https://www.tensortonic.com/problems/probstat-z-test-one-sample |
| ReLU | Implement ReLU activation in CUDA with one thread per element, bounds checks, and branch-efficient rectification. | https://www.tensortonic.com/problems/relu |
| Actor-Critic (A2C) Loss with Entropy | Advantage Actor-Critic (A2C) jointly trains a policy and a value function with a single composite loss. | https://www.tensortonic.com/problems/rl-actor-critic-a2c-loss |
| Bellman Expectation Equation | The Bellman expectation equation is the cornerstone of policy evaluation in reinforcement learning. | https://www.tensortonic.com/problems/rl-bellman-expectation-equation |
| Bellman Optimality Equation | Apply one Bellman optimality backup by maximizing expected immediate reward plus discounted next-state value. | https://www.tensortonic.com/problems/rl-bellman-optimality-equation |
| Bradley-Terry Reward Model Loss | Compute Bradley-Terry pairwise preference loss for chosen and rejected reward-model scores in RLHF. | https://www.tensortonic.com/problems/rl-bradley-terry-reward-loss |
| Discounted Returns | Compute reverse-time discounted returns from a reward sequence using a configurable discount factor and terminal bootstrap. | https://www.tensortonic.com/problems/rl-discounted-returns |
| Double DQN Target | Compute Double DQN targets by selecting actions with the online network and evaluating them with the target network. | https://www.tensortonic.com/problems/rl-double-dqn-target |
| DPO Closed-Form Loss | Direct Preference Optimization (DPO) replaces the reward-model + PPO pipeline of RLHF with a single supervised loss. | https://www.tensortonic.com/problems/rl-dpo-closed-form-loss |
| DQN Loss with Target Network | Compute DQN targets and mean squared temporal-difference loss with terminal masking and a separate target network. | https://www.tensortonic.com/problems/rl-dqn-loss-with-target-network |
| Dueling DQN Decomposition | Dueling DQN splits the action-value function into a state value head V(s) and an advantage head A(s, a), then aggregates them back into Q(s, a). | https://www.tensortonic.com/problems/rl-dueling-dqn-decomposition |
| Epsilon-Greedy Action Selection | Choose actions with epsilon-greedy exploration, deterministic greedy tie handling, and controlled random sampling. | https://www.tensortonic.com/problems/rl-epsilon-greedy-action-selection |
| Expected SARSA | Apply Expected SARSA updates using the epsilon-greedy expectation over next-state action values. | https://www.tensortonic.com/problems/rl-expected-sarsa |
| Experience Replay Buffer | Implement a fixed-capacity circular replay buffer with deterministic overwrites and sampled transition batches. | https://www.tensortonic.com/problems/rl-experience-replay-buffer |
| Generalized Advantage Estimation (GAE) | Compute generalized advantage estimates by recursively combining rewards, value predictions, discounts, and GAE decay. | https://www.tensortonic.com/problems/rl-generalized-advantage-estimation |
| GRPO Group-Relative Advantage | Normalize rewards within each GRPO group using its mean and standard deviation, including zero-variance handling. | https://www.tensortonic.com/problems/rl-grpo-group-relative-advantage |
| LinUCB Contextual Bandit | LinUCB extends the UCB algorithm to contextual bandits by assuming the expected reward of each arm is a linear function of the context. | https://www.tensortonic.com/problems/rl-linucb-contextual-bandit |
| Monte Carlo Policy Evaluation | Estimate policy state values from complete sampled returns without requiring an environment transition model. | https://www.tensortonic.com/problems/rl-monte-carlo-policy-evaluation |
| Policy Iteration | Policy iteration solves a finite Markov Decision Process by alternating two steps until the greedy policy stops changing. | https://www.tensortonic.com/problems/rl-policy-iteration |
| PPO Clipped Surrogate Objective | Proximal Policy Optimization (PPO) optimizes a clipped surrogate objective that limits how far the new policy may drift from the data-collecting policy. | https://www.tensortonic.com/problems/rl-ppo-clipped-surrogate |
| Prioritized Experience Replay | Sample replay items from priority-weighted probabilities and compute normalized importance-sampling weights. | https://www.tensortonic.com/problems/rl-prioritized-experience-replay |
| Q-Learning Update | Apply the off-policy Q-learning update using the maximum next-state action value and terminal handling. | https://www.tensortonic.com/problems/rl-q-learning-update |
| REINFORCE Gradient | Implement REINFORCE Gradient, and return the scalar loss as a float rounded to 4 decimals. | https://www.tensortonic.com/problems/rl-reinforce-gradient |
| REINFORCE with Baseline | Implement REINFORCE with Baseline, and return the scalar loss as a Python float, rounded to 4 decimals. | https://www.tensortonic.com/problems/rl-reinforce-with-baseline |
| PPO with Reference-Policy KL | Compute a PPO policy objective with reference-policy KL regularization for stable RLHF optimization. | https://www.tensortonic.com/problems/rl-rlhf-ppo-kl-loss |
| RLOO (REINFORCE Leave-One-Out) | RLOO (REINFORCE Leave-One-Out) is a policy gradient variance-reduction trick used in modern RLHF. | https://www.tensortonic.com/problems/rl-rloo-reinforce-leave-one-out |
| SARSA Update | Apply the on-policy SARSA temporal-difference update from the current transition and next selected action. | https://www.tensortonic.com/problems/rl-sarsa-update |
| Softmax (Boltzmann) Exploration | Convert action values into a numerically stable Boltzmann exploration distribution controlled by temperature. | https://www.tensortonic.com/problems/rl-softmax-exploration |
| TD(0) Value Update | Apply one TD(0) state-value update using immediate reward, discounted next-state value, and learning rate. | https://www.tensortonic.com/problems/rl-td-zero-update |
| Thompson Sampling (Beta-Bernoulli) | Thompson sampling for a Bernoulli multi-armed bandit keeps a Beta posterior over each arm's success probability. | https://www.tensortonic.com/problems/rl-thompson-sampling-beta-bernoulli |
| UCB1 for Multi-Armed Bandits | In a multi-armed bandit, an agent must repeatedly choose one of K arms and only observes the reward of the arm it pulls. | https://www.tensortonic.com/problems/rl-ucb1-multi-armed-bandit |
| Value Iteration | Run value iteration to convergence for a finite MDP, then extract the greedy optimal policy with stable tie handling. | https://www.tensortonic.com/problems/rl-value-iteration |
| RMS Normalization | Implement row-wise RMS normalization in CUDA with sum-of-squares reduction, numerical stability, and learnable scaling. | https://www.tensortonic.com/problems/rms-norm |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Scaled Dot-Product Attention | Implement single-head scaled dot-product attention in CUDA with stable row-wise softmax and value aggregation. | https://www.tensortonic.com/problems/scaled-dot-product-attention |
| Sigmoid | Implement sigmoid activation in CUDA with one thread per element, device exponential math, and bounds-checked memory access. | https://www.tensortonic.com/problems/sigmoid |
| Softmax | Implement numerically stable CUDA softmax over a vector using global maximum and normalization reductions. | https://www.tensortonic.com/problems/softmax |
| Basic SELECT | Write a SQL SELECT query that aliases product names and calculates inventory value from unit price and stock quantity. | https://www.tensortonic.com/problems/sql-basic-select |
| CASE Statements | Classify user activity and platform type with SQL CASE expressions using session counts and mobile platform values. | https://www.tensortonic.com/problems/sql-case-statements |
| Conditional Aggregation | Summarize support tickets by department with conditional SQL counts for open, in-progress, and closed statuses. | https://www.tensortonic.com/problems/sql-conditional-aggregation |
| COUNT, SUM, AVG | Aggregate sales by category with SQL COUNT, SUM, and AVG while handling NULL discounts and deterministic ordering. | https://www.tensortonic.com/problems/sql-count-sum-avg |
| Cross Join | Generate every segment and metric combination with a SQL CROSS JOIN for a complete reporting grid. | https://www.tensortonic.com/problems/sql-cross-join |
| Common Table Expressions | Use a SQL CTE to calculate customer order counts and spending, then filter repeat customers by total spend. | https://www.tensortonic.com/problems/sql-cte |
| Date Functions | Extract signup year, month, quarter, and cohort month with SQL date functions for cohort analysis. | https://www.tensortonic.com/problems/sql-date-functions |
| DISTINCT Values | Return each customer and their distinct product count with SQL aggregation and deterministic sorting. | https://www.tensortonic.com/problems/sql-distinct-values |
| GROUP BY | Group orders by customer in SQL to calculate total order count and spending, ordered by highest spend. | https://www.tensortonic.com/problems/sql-group-by |
| HAVING Clause | Use SQL GROUP BY and HAVING to find customers with at least two orders and summarize their total spending. | https://www.tensortonic.com/problems/sql-having-clause |
| INNER JOIN | Join employees to matching departments with SQL INNER JOIN and return employee name, salary, and department. | https://www.tensortonic.com/problems/sql-inner-join |
| LAG and LEAD | Use SQL LAG to compare monthly revenue with the previous month and calculate month-over-month change. | https://www.tensortonic.com/problems/sql-lag-lead |
| LEFT JOIN | Use SQL LEFT JOIN to include every customer and calculate total spending, returning zero for customers without orders. | https://www.tensortonic.com/problems/sql-left-join |
| LIMIT and OFFSET | Use SQL ORDER BY, LIMIT, and OFFSET to return the second through fourth highest-revenue sales with tie-breaking. | https://www.tensortonic.com/problems/sql-limit-offset |
| Multiple Joins | Join users, experiment assignments, and conversion events in SQL to report converted users, variants, and revenue. | https://www.tensortonic.com/problems/sql-multiple-joins |
| Nested Aggregations | Use a SQL subquery or CTE to compute daily order totals, average daily revenue, and the busiest day. | https://www.tensortonic.com/problems/sql-nested-aggregations |
| NULL Handling | Handle SQL NULL values with COALESCE and conditional status logic while filtering customers without phone numbers. | https://www.tensortonic.com/problems/sql-null-handling |
| ORDER BY | Sort student exam results in SQL by descending score and ascending name for deterministic ties. | https://www.tensortonic.com/problems/sql-order-by |
| RANK and DENSE_RANK | Rank ML models within each dataset using SQL RANK and DENSE_RANK over descending accuracy. | https://www.tensortonic.com/problems/sql-rank-dense-rank |
| ROW_NUMBER | Assign deterministic per-segment activity ranks with SQL ROW_NUMBER ordered by engagement score and username. | https://www.tensortonic.com/problems/sql-row-number |
| Running Totals | Compute per-account running transaction totals with a partitioned SQL window ordered by date and transaction ID. | https://www.tensortonic.com/problems/sql-running-totals |
| Self Join | Use a SQL self join to pair users with their referrers while labeling organic signups without a referral. | https://www.tensortonic.com/problems/sql-self-join |
| String Functions | Clean survey data with SQL string functions for trimmed lowercase names, answer lengths, and parsed source URLs. | https://www.tensortonic.com/problems/sql-string-functions |
| Subqueries | Use SQL subqueries to compare product prices with the overall average and include only products with recorded sales. | https://www.tensortonic.com/problems/sql-subqueries |
| WHERE Clauses | Filter employees by department and salary with SQL WHERE conditions, returning only qualifying names and salaries. | https://www.tensortonic.com/problems/sql-where-clauses |
| Sum of Array | Implement a multi-block CUDA sum reduction that combines partial block sums into one scalar output. | https://www.tensortonic.com/problems/sum-of-array |
| Swish | Implement fused Swish or SiLU activation in CUDA with one thread per element and device exponential math. | https://www.tensortonic.com/problems/swish |
| Tanh | Implement hyperbolic tangent activation in CUDA with one thread per element, device intrinsic math, and bounds checks. | https://www.tensortonic.com/problems/tanh |
| Tiled Matrix Multiplication | Implement shared-memory tiled matrix multiplication in CUDA with synchronized tile loads and partial-tile handling. | https://www.tensortonic.com/problems/tiled-matmul |
| Block-Pointer Matmul | Implement tiled Triton matrix multiplication with block pointers, boundary checks, and tail-safe loads and stores. | https://www.tensortonic.com/problems/triton-block-pointer-matmul |
| Cross Entropy Loss (Mean Reduction) | Implement mean categorical cross-entropy in Triton with stable row-wise log-sum-exp and atomic loss accumulation. | https://www.tensortonic.com/problems/triton-cross-entropy |
| Dropout (Inverted Scaling) | Implement inverted dropout in Triton with a supplied mask, register scaling, and tail-safe tiled memory access. | https://www.tensortonic.com/problems/triton-dropout |
| FlashAttention Forward | Implement fused FlashAttention forward in Triton with streamed key-value blocks and online softmax statistics. | https://www.tensortonic.com/problems/triton-flash-attention-fwd |
| Fused Matmul + Bias + ReLU | Fuse tiled matrix multiplication, per-column bias, and ReLU in one Triton kernel with tail-safe memory access. | https://www.tensortonic.com/problems/triton-fused-matmul-bias-relu |
| Fused Multiply-Add | Implement a Triton fused multiply-add kernel with contiguous tiles, hardware FMA, and masked tail handling. | https://www.tensortonic.com/problems/triton-fused-multiply-add |
| Fused Row-Wise Softmax | Implement fused row-wise softmax in Triton with stable max subtraction, register reductions, and masked column tails. | https://www.tensortonic.com/problems/triton-fused-softmax |
| GELU | Implement exact GELU activation in Triton with device error-function math and masked contiguous tiles. | https://www.tensortonic.com/problems/triton-gelu |
| GEMV: Matrix Vector Product | Implement Triton matrix-vector multiplication with row-block programs, float32 accumulation, and masked matrix tails. | https://www.tensortonic.com/problems/triton-gemv |
| Grouped Program-ID Matmul | Implement grouped program-ID matrix multiplication in Triton to improve L2 reuse while preserving tail-safe tiled computation. | https://www.tensortonic.com/problems/triton-grouped-matmul |
| KV Cache Append | Append one autoregressive decoding row to key and value caches in Triton without modifying other cache positions. | https://www.tensortonic.com/problems/triton-kv-append |
| L2 Vector Norm | Compute a Triton L2 vector norm with tiled sum-of-squares reduction, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-l2-norm |
| LayerNorm Forward | Implement LayerNorm forward in Triton with per-row mean and variance reductions, affine parameters, and masked tails. | https://www.tensortonic.com/problems/triton-layernorm |
| LayerNorm Backward | Implement LayerNorm backward in Triton with row-wise statistics and atomic gradients for scale and bias. | https://www.tensortonic.com/problems/triton-layernorm-backward |
| Row-Wise LogSumExp | Implement numerically stable row-wise LogSumExp in Triton with max subtraction and masked register reductions. | https://www.tensortonic.com/problems/triton-logsumexp |
| Tiled Matrix Multiplication | Implement tiled matrix multiplication in Triton with a two-dimensional grid, float32 accumulation, and tail masks. | https://www.tensortonic.com/problems/triton-matmul |
| Autotuned Matrix Multiplication | Autotune Triton matrix multiplication across tile and pipeline configurations while preserving masked boundary handling. | https://www.tensortonic.com/problems/triton-matmul-autotune |
| Vector Max Reduction | Compute a vector maximum with one Triton reduction program and masked tail lanes that cannot win comparisons. | https://www.tensortonic.com/problems/triton-max |
| Single-Pass Mean and Variance | Compute population mean and variance in Triton with single-pass statistics, atomic partials, and masked tails. | https://www.tensortonic.com/problems/triton-mean-variance |
| Online Softmax | Implement chunked online softmax in Triton with running maxima and denominators followed by a normalized output pass. | https://www.tensortonic.com/problems/triton-online-softmax |
| ReLU | Implement ReLU activation in Triton with contiguous program tiles, branch-free rectification, and masked tails. | https://www.tensortonic.com/problems/triton-relu |
| RMSNorm Forward | Implement RMSNorm forward in Triton with per-row square reduction, numerical stability, scaling, and masked tails. | https://www.tensortonic.com/problems/triton-rmsnorm |
| Rotary Position Embedding | Implement Rotary Position Embeddings in Triton with per-token pair rotations, precomputed sine and cosine, and tail masks. | https://www.tensortonic.com/problems/triton-rope |
| Scaled Dot-Product Attention | Implement three-stage scaled dot-product attention in Triton with tiled scores, stable softmax, and value aggregation. | https://www.tensortonic.com/problems/triton-scaled-dot-product-attention |
| SiLU | Implement fused SiLU or Swish activation in Triton with contiguous tiles, sigmoid weighting, and masked tails. | https://www.tensortonic.com/problems/triton-silu |
| Split-K Matmul | Implement split-K matrix multiplication in Triton with parallel reduction slices, atomic output accumulation, and tail masks. | https://www.tensortonic.com/problems/triton-split-k-matmul |
| Vector Sum Reduction | Implement tiled vector sum reduction in Triton with register partials, atomic accumulation, and masked tail lanes. | https://www.tensortonic.com/problems/triton-sum |
| Tiled Transpose | Implement tiled matrix transpose in Triton by swapping load and store strides with masked boundary tiles. | https://www.tensortonic.com/problems/triton-transpose |
| Vector Addition | Implement elementwise vector addition in Triton with contiguous program tiles and safe masking for partial tails. | https://www.tensortonic.com/problems/triton-vector-addition |
| Vectorized Vector Add | Implement vector addition in Triton with larger per-program tiles to reduce launch overhead while masking the final tail. | https://www.tensortonic.com/problems/triton-vectorized-load |
| Vector Addition | Implement bounds-checked pointwise vector addition in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-addition |
| Vector Subtraction | Implement bounds-checked pointwise vector subtraction in CUDA with one thread per output element. | https://www.tensortonic.com/problems/vector-subtract |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/csorfolydaniel)
<!-- tensortonic:end -->
