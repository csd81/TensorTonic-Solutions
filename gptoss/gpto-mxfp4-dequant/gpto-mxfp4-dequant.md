# <span style="font-size: 20px;">MXFP4 Block Dequantization</span>

<span style="font-size: 14px;">MXFP4 (Microscaling FP4) is the block-scaled 4-bit floating-point format defined in the Open Compute Project Microscaling specification. It is the storage format used for roughly 90 percent of the parameters in gpt-oss, specifically the mixture-of-experts (MoE) weight tensors. Dequantization is the operation that reconstructs an approximate float32 weight from a packed 4-bit value and its shared per-block FP8 exponent.</span>

---

## <span style="font-size: 16px;">Why MXFP4 in gpt-oss</span>

<span style="font-size: 14px;">gpt-oss inherits the design pressure of every large MoE model: most of the parameter budget lives in the expert feed-forward weights, not the attention projections. Storing those experts in bf16 or fp16 burns memory bandwidth and inflates the model footprint past what is comfortable on a single GPU. The 120B variant has roughly 117B of its 120B parameters concentrated in MoE expert layers, so any saving on that tensor multiplies directly into VRAM headroom.</span>

<span style="font-size: 14px;">MXFP4 attacks both problems at once:</span>

* <span style="font-size: 14px;">**Storage**: each weight uses 4 bits for the value plus 8 bits of shared scale amortized over 32 elements, giving $32 \cdot 4 + 8 = 136$ bits per block, or **4.25 bits per parameter**. This is roughly 4x smaller than fp16.</span>
* <span style="font-size: 14px;">**Numerical range**: a per-block FP8-E8M0 scale lets each 32-element block float over a huge dynamic range (powers of two from $2^{-127}$ to $2^{+127}$) without spending any mantissa bits on the scale itself.</span>
* <span style="font-size: 14px;">**Hardware**: H100 and B100 GPUs have native MXFP4 GEMM kernels. Even when running on older hardware, the dequant step is cheap and bandwidth-bound, not compute-bound.</span>

<span style="font-size: 14px;">The gpt-oss release specifically states that MXFP4 quantization on the MoE weights is what makes the 120B variant fit in 80 GB of VRAM. Attention layers stay in bf16 because they are smaller and more sensitive to quantization noise.</span>

---

## <span style="font-size: 16px;">Format Definition</span>

<span style="font-size: 14px;">MXFP4 is a **block format**, not a per-element format. The unit of quantization is a contiguous block of 32 elements that share a single scale.</span>

* <span style="font-size: 14px;">**Element format**: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit). 4 bits total per element. 16 representable values.</span>
* <span style="font-size: 14px;">**Block size**: 32 elements. Smaller would cost more scale overhead; larger would lose dynamic-range adaptivity inside the block.</span>
* <span style="font-size: 14px;">**Scale format**: E8M0 (8 unsigned exponent bits, 0 mantissa bits, 0 sign bit). The byte value $s$ encodes the multiplier $2^{s - 127}$.</span>
* <span style="font-size: 14px;">**Packing**: two 4-bit elements share one byte. The convention in gpt-oss and the OCP spec is **low nibble first**, meaning element index 0 lives in bits 0-3 of the byte and element index 1 lives in bits 4-7.</span>

<span style="font-size: 14px;">A weight tensor of shape $(\text{rows}, \text{cols})$ becomes a packed tensor of shape $(\text{rows}, \text{cols} / 2)$ in uint8, plus a scale tensor of shape $(\text{rows}, \text{cols} / 32)$ in uint8.</span>

---

## <span style="font-size: 16px;">The Dequantization Equation</span>

<span style="font-size: 14px;">For block $b$ with packed bytes $P_b \in \mathbb{R}^{16}$ and scale byte $s_b$, the dequantized values $W_b \in \mathbb{R}^{32}$ are:</span>

$$
W_{b, 2i} = \text{LUT}\big[P_{b, i} \,\,\&\,\, 0\text{x}0\text{F}\big] \cdot 2^{s_b - 127}
$$

$$
W_{b, 2i+1} = \text{LUT}\big[(P_{b, i} \gg 4) \,\,\&\,\, 0\text{x}0\text{F}\big] \cdot 2^{s_b - 127}
$$

<span style="font-size: 14px;">where $i \in \{0, 1, \ldots, 15\}$ and LUT is the 16-entry E2M1 table.</span>

---

## <span style="font-size: 16px;">The E2M1 Lookup Table</span>

<span style="font-size: 14px;">E2M1 has 1 sign bit, 2 exponent bits, 1 mantissa bit. With only 16 possible bit patterns, it is faster to use a lookup table than to decode the IEEE-style fields at runtime. The 16 values are:</span>

* <span style="font-size: 14px;">**0b0000** to **0b0111**: $\{+0, +0.5, +1, +1.5, +2, +3, +4, +6\}$</span>
* <span style="font-size: 14px;">**0b1000** to **0b1111**: $\{-0, -0.5, -1, -1.5, -2, -3, -4, -6\}$</span>

<span style="font-size: 14px;">Two structural facts worth noticing:</span>

* <span style="font-size: 14px;">The top bit is the sign. Flipping bit 3 negates the value, with $+0$ flipping to $-0$ (both represent zero in float arithmetic).</span>
* <span style="font-size: 14px;">There is **no NaN or Inf** at the element level. E2M1 spends all its codes on finite values. NaN handling is delegated to the scale byte.</span>

---

## <span style="font-size: 16px;">The E8M0 Scale Format</span>

<span style="font-size: 14px;">E8M0 is an 8-bit unsigned biased exponent. It carries no mantissa and no sign bit, which makes it tiny but limits the scale to powers of two. The interpretation is:</span>

* <span style="font-size: 14px;">**Byte value $s$ in $[1, 254]$**: multiplier is $2^{s - 127}$. So $s = 127$ means $1.0$, $s = 128$ means $2.0$, $s = 125$ means $0.25$.</span>
* <span style="font-size: 14px;">**Byte value $s = 0$**: multiplier is $2^{-127}$ (subnormal flush region). Some implementations treat this as zero.</span>
* <span style="font-size: 14px;">**Byte value $s = 0\text{xFF}$ (255)**: NaN. The entire block is poisoned. Hardware kernels propagate NaN; software dequant should as well.</span>

<span style="font-size: 14px;">Storing only powers of two is what keeps the scale to a single byte. The mantissa is implicitly contributed by the E2M1 values themselves: any in-block dynamic-range adjustment happens through the choice of which 16 E2M1 levels the block uses, not by a fractional scale.</span>

---

## <span style="font-size: 16px;">Packing Convention</span>

<span style="font-size: 14px;">The OCP spec and the gpt-oss reference both pack **low nibble first**:</span>

* <span style="font-size: 14px;">Bits 0-3 of byte $i$ contain element index $2i$.</span>
* <span style="font-size: 14px;">Bits 4-7 of byte $i$ contain element index $2i + 1$.</span>

<span style="font-size: 14px;">Concretely, if a block starts with the values $[1, 4, 0, 0, \ldots]$ in nibble form (so $+0.5, +2.0, +0, +0, \ldots$ after the LUT), then byte 0 of the packed buffer equals $(4 \ll 4) \,|\, 1 = 0\text{x}41$. Swapping low and high is the single most common bug, because both orderings look superficially correct in code review.</span>

---

## <span style="font-size: 16px;">Special Scale Encodings</span>

<span style="font-size: 14px;">Two scale-byte values are reserved by the OCP spec for special semantics:</span>

* <span style="font-size: 14px;">**$s = 0\text{x}00$**: the spec calls this the "subnormal" or "flush" region. Strictly it represents the multiplier $2^{-127}$, which is so small that most kernels treat the entire block as zero. Production code typically checks this case and writes zeros directly, skipping the multiplication.</span>
* <span style="font-size: 14px;">**$s = 0\text{x}FF$**: NaN encoding. The block is poisoned, usually because an upstream training step produced NaN gradients that propagated into the quantizer. Hardware MXFP4 GEMM kernels propagate NaN through the output, and software dequant should do the same. Naive code that computes $2^{255 - 127}$ overflows to $+\infty$ and quietly corrupts the rest of the GEMM.</span>

<span style="font-size: 14px;">The standard byte range $[1, 254]$ covers multipliers from $2^{-126}$ to $2^{+127}$, which is enough dynamic range that no real weight tensor saturates either end after training.</span>

---

## <span style="font-size: 16px;">Bits per Parameter Accounting</span>

<span style="font-size: 14px;">A common talking point for MXFP4 is the "4.25 bits per parameter" figure. The arithmetic:</span>

* <span style="font-size: 14px;">**Element bits**: $32 \cdot 4 = 128$ bits per block of values.</span>
* <span style="font-size: 14px;">**Scale bits**: $8$ bits per block.</span>
* <span style="font-size: 14px;">**Total per block**: $136$ bits.</span>
* <span style="font-size: 14px;">**Per element**: $136 / 32 = 4.25$ bits.</span>

<span style="font-size: 14px;">For comparison, INT4 with a per-row fp16 scale lands at roughly $4 + 16/k$ bits where $k$ is the group size; for $k = 128$ that is about $4.125$ bits. MXFP4 trades a tiny extra fraction of a bit for a more expressive value set (E2M1 covers $\{0.5, 1.5, 3, 6\}$ which INT4 cannot represent without an extra zero-point).</span>

---

## <span style="font-size: 16px;">How Dequant Fits into the MoE Forward Pass</span>

<span style="font-size: 14px;">In gpt-oss the MoE expert weights live on disk and in HBM as MXFP4. During inference the engine has two strategies:</span>

* <span style="font-size: 14px;">**Eager dequant**: dequantize the routed expert's weight blocks to bf16 just in time, run a standard bf16 GEMM, throw the bf16 buffer away. This is what the reference PyTorch path does.</span>
* <span style="font-size: 14px;">**Fused MXFP4 GEMM**: on Hopper and Blackwell, a single kernel reads packed bytes plus scales, multiplies on the fly, and accumulates in fp32. No materialized bf16 tensor ever exists.</span>

<span style="font-size: 14px;">Either way the math is the same: the function in this problem is the building block. A real implementation also broadcasts the per-block scale efficiently and may fuse the LUT into a small constant SMEM table on the GPU, but the semantics match the numpy reference exactly.</span>

---

## <span style="font-size: 16px;">Quantization Source: How Weights Become MXFP4</span>

<span style="font-size: 14px;">This problem only implements the dequantization direction. The forward quantization step is conceptually simple but worth understanding for context.</span>

<span style="font-size: 14px;">Given a 32-element fp32 block to be quantized:</span>

* <span style="font-size: 14px;">**1. Find absolute maximum.** Let $M = \max_i |x_i|$. This sets the dynamic range the block needs.</span>
* <span style="font-size: 14px;">**2. Choose the scale exponent.** Pick $e$ so that $M / 2^e$ falls inside the representable range of E2M1, which is $[0, 6]$. The natural choice is $e = \lceil \log_2(M / 6) \rceil$, then encode as the byte $s = e + 127$.</span>
* <span style="font-size: 14px;">**3. Round each element.** Compute $x_i / 2^e$, then snap to the nearest E2M1 grid point. Standard rounding is round-to-nearest-even, the same rule IEEE 754 uses.</span>
* <span style="font-size: 14px;">**4. Pack.** Place the nibble for element $2i$ in the low half of byte $i$, the nibble for element $2i+1$ in the high half.</span>

<span style="font-size: 14px;">Errors are introduced in step 3, where the round-to-grid is lossy. Group-of-32 makes the error roughly self-limiting per block, which is the whole point of microscaling.</span>

---

## <span style="font-size: 16px;">Comparison with Other 4-Bit Formats</span>

* <span style="font-size: 14px;">**INT4 with per-group scale (GPTQ, AWQ)**: stores a signed integer plus an fp16 or bf16 scale (and sometimes a zero-point) per group. Higher resolution near zero, but no built-in handling of large outliers without a separate outlier tensor.</span>
* <span style="font-size: 14px;">**NF4 (QLoRA)**: stores 4-bit codes that index a Gaussian-quantile table. Optimized for normally distributed weights and trains well, but has no hardware GEMM support, only dequant-then-bf16 GEMM.</span>
* <span style="font-size: 14px;">**FP4-E2M1 raw (no microscaling)**: would have catastrophic dynamic range with a global scale. Microscaling is what makes the format usable.</span>
* <span style="font-size: 14px;">**FP8-E4M3**: 8 bits per element, much higher precision, but 2x the storage. Used in gpt-oss for activations during quantized MatMul.</span>

<span style="font-size: 14px;">MXFP4 is the sweet spot for static, well-distributed weight tensors when you also want hardware GEMM throughput, which is exactly the MoE expert use case.</span>

---

## <span style="font-size: 16px;">Worked Dequantization Example</span>

<span style="font-size: 14px;">Take a single block of 32 elements. Suppose the first four nibbles are $[2, 10, 4, 0]$ (so logical values $+1.0, -1.0, +2.0, +0.0$ before scaling) and the remaining 28 nibbles are $[0, 0, \ldots]$. The packed bytes are:</span>

* <span style="font-size: 14px;">**Byte 0**: low nibble $= 2$, high nibble $= 10$. Byte $= (10 \ll 4) \,|\, 2 = 0\text{xA2}$.</span>
* <span style="font-size: 14px;">**Byte 1**: low nibble $= 4$, high nibble $= 0$. Byte $= 0\text{x04}$.</span>
* <span style="font-size: 14px;">**Bytes 2-15**: all zero.</span>

<span style="font-size: 14px;">Let the scale byte be $s = 128$, so the multiplier is $2^{128 - 127} = 2.0$.</span>

<span style="font-size: 14px;">Dequant proceeds as:</span>

<span style="font-size: 14px;">1. **Unpack byte 0**: low $= 0\text{xA2} \,\&\, 0\text{x0F} = 2$, high $= (0\text{xA2} \gg 4) \,\&\, 0\text{x0F} = 10$. Elements 0 and 1 are nibble 2 and nibble 10.</span>

<span style="font-size: 14px;">2. **LUT lookup**: nibble $2 \to +1.0$, nibble $10 \to -1.0$, nibble $4 \to +2.0$, nibble $0 \to +0.0$.</span>

<span style="font-size: 14px;">3. **Apply scale**: multiply by $2.0$. Element 0 becomes $+2.0$, element 1 becomes $-2.0$, element 2 becomes $+4.0$, element 3 becomes $+0.0$, and the rest are zero.</span>

<span style="font-size: 14px;">The resulting block in fp32 is $[+2, -2, +4, 0, 0, \ldots, 0]$, a 32-vector.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong scale exponent base.** The most common error is using $2^s$ directly instead of $2^{s - 127}$. With small scale bytes this looks fine (the values are huge and divergent), and with byte values near 127 it looks roughly correct but is off by enormous factors. Always subtract the bias of 127.</span>
* <span style="font-size: 14px;">**Swapping high and low nibbles.** Putting the high nibble at even positions and the low nibble at odd positions is structurally identical code that produces silent corruption. The OCP spec and gpt-oss both use **low nibble first** (element 0 in bits 0-3). Verify with a hand-rolled byte like $0\text{x}41$ that decodes to $[0.5, 2.0]$.</span>
* <span style="font-size: 14px;">**Forgetting sign extension.** E2M1 is sign-magnitude: bit 3 of the nibble is the sign. A LUT that treats nibble 8 through 15 as positive will silently turn negative weights into positive ones. The model accuracy collapses but does not crash.</span>
* <span style="font-size: 14px;">**Treating $-0$ specially.** The LUT entry for nibble 8 is $-0.0$. In IEEE float arithmetic $-0$ equals $+0$, so this is fine, but tests that compare bit patterns instead of values will flag it as a mismatch. Compare with $\text{allclose}$, not bit equality.</span>
* <span style="font-size: 14px;">**Wrong dtype on output.** Returning int32 or float64 instead of float32 breaks downstream kernels that expect contiguous fp32 buffers. The cast is not free at scale; either cast at the end or compute in fp32 throughout.</span>
* <span style="font-size: 14px;">**Ignoring the NaN scale.** A scale byte of $0\text{xFF}$ poisons the whole block. Production kernels propagate NaN through the block; quick numpy code that just does $2^{255 - 127} = 2^{128}$ will overflow to $+\infty$ and corrupt the GEMM output silently. Check for $s = 0\text{xFF}$ explicitly when robustness matters.</span>
* <span style="font-size: 14px;">**Block-size off-by-one.** Confusing "32 elements per block" with "32 bytes per block" (which would be 64 elements) is a classic mistake. Each byte holds 2 elements, so the packed shape is $(n\_\text{blocks}, 16)$ for $\text{block\_size} = 32$.</span>
* <span style="font-size: 14px;">**Broadcasting the scale wrong.** The scale tensor has shape $(n\_\text{blocks},)$ and must broadcast over the 32 elements of each block. A common bug is to broadcast it over the block dimension instead, applying every scale to every block.</span>

---
