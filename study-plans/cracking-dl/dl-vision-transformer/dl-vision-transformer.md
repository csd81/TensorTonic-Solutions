# <span style="font-size: 20px;">Vision Transformer (ViT)</span>

<span style="font-size: 14px;">The Vision Transformer (Dosovitskiy et al., 2021) demonstrated that a pure Transformer architecture, applied directly to sequences of image patches, can achieve excellent image classification results. ViT challenged the dominance of CNNs in computer vision and opened the door for unified architectures across vision and language. It is now a standard interview topic for both CV and NLP positions.</span>

---

## <span style="font-size: 16px;">From CNNs to Patch Embeddings</span>

<span style="font-size: 14px;">CNNs process images through local receptive fields that grow layer by layer. Transformers, in contrast, have global receptive fields from the first layer - every token can attend to every other token. The key insight of ViT is that an image can be treated as a sequence of "visual tokens" by splitting it into non-overlapping patches.</span>

<span style="font-size: 14px;">**Patch embedding**: an image of size $H \times W \times C$ is divided into a grid of $P \times P$ patches, giving $N = (H/P) \times (W/P)$ patches. Each patch is flattened to a vector of size $P^2 \cdot C$ and linearly projected to dimension $d_{\text{model}}$.</span>

<span style="font-size: 14px;">**Conv2d shortcut**: `nn.Conv2d(C, d, kernel_size=P, stride=P)` achieves the same result as flatten + linear, since each convolution kernel covers exactly one non-overlapping patch. This is computationally equivalent but more efficient.</span>

<span style="font-size: 14px;">**[CLS] token**: a learnable embedding prepended to the patch sequence. After passing through the Transformer, this token's output serves as the aggregate image representation for classification - analogous to BERT's [CLS] token for sentence-level tasks.</span>

---

## <span style="font-size: 16px;">ViT Architecture</span>

<span style="font-size: 14px;">The full forward pass:</span>

<span style="font-size: 14px;">1. **Patch projection**: image $(B, C, H, W) \to$ patches $(B, N, d)$ via Conv2d</span>
<span style="font-size: 14px;">2. **Prepend [CLS]**: $(B, N, d) \to (B, N{+}1, d)$</span>
<span style="font-size: 14px;">3. **Add positional embedding**: learned $(1, N{+}1, d)$ added to the sequence</span>
<span style="font-size: 14px;">4. **Transformer encoder**: $L$ Pre-LN blocks with bidirectional self-attention + GELU FFN</span>
<span style="font-size: 14px;">5. **Classification**: extract CLS token $\to$ LayerNorm $\to$ Linear$(d, \text{num\_classes})$</span>

<span style="font-size: 14px;">ViT uses **Pre-LN** (LayerNorm before each sublayer), following the improved Transformer convention that provides more stable training. The attention is **bidirectional** (no causal mask) since all patches should attend to all other patches simultaneously.</span>

<span style="font-size: 14px;">The CLS token and positional embeddings are `nn.Parameter` tensors (raw learnable parameters), not `nn.Embedding` modules. This is because the CLS token is a single vector and the positional embedding has a fixed number of positions determined by the image/patch size.</span>

---

## <span style="font-size: 16px;">Positional Encoding in ViT</span>

<span style="font-size: 14px;">Unlike NLP Transformers where token order is inherently sequential, image patches have 2D spatial relationships. ViT uses **1D learned positional embeddings** - one vector per position in the flattened patch sequence (including the CLS position). Surprisingly, this simple 1D encoding works well; the model learns the 2D spatial structure from data.</span>

<span style="font-size: 14px;">**Why not 2D positional encoding?** Experiments showed that explicit 2D encodings provide minimal benefit over 1D learned embeddings. The attention mechanism is powerful enough to discover spatial relationships. However, for tasks requiring fine-grained spatial understanding (like detection or segmentation), 2D-aware position encodings can help.</span>

<span style="font-size: 14px;">**Interpolation for different resolutions**: since positional embeddings are learned for a fixed number of patches, using ViT at a different resolution than training requires interpolating the positional embeddings (typically via bicubic interpolation after reshaping to 2D).</span>

---

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">For ViT with image size $I$, patch size $P$, classes $K$, $d$ = d_model, $d_f$ = d_ff, $L$ layers, $C$ channels:</span>

- <span style="font-size: 14px;">**Patch projection** (Conv2d): $C \cdot P^2 \cdot d + d$</span>
- <span style="font-size: 14px;">**CLS token**: $d$</span>
- <span style="font-size: 14px;">**Positional embedding**: $((I/P)^2 + 1) \cdot d$</span>
- <span style="font-size: 14px;">**Per encoder block**: $4(d^2 + d) + 2d \cdot d_f + d_f + d + 4d$</span>
- <span style="font-size: 14px;">**Final LayerNorm**: $2d$</span>
- <span style="font-size: 14px;">**Classification head**: $d \cdot K + K$</span>

<span style="font-size: 14px;">ViT-Base/16: $d = 768$, $h = 12$, $d_f = 3072$, $L = 12$, $P = 16$, image 224. ~86M parameters.</span>

<span style="font-size: 14px;">ViT-Large/16: $d = 1024$, $h = 16$, $d_f = 4096$, $L = 24$. ~307M parameters.</span>

---

## <span style="font-size: 16px;">ViT Variants and Modern Extensions</span>

<span style="font-size: 14px;">**DeiT (Data-efficient Image Transformer)**: adds a distillation token alongside CLS, trained with knowledge distillation from a CNN teacher. Makes ViT competitive with CNNs even on smaller datasets like ImageNet-1K (without JFT pre-training).</span>

<span style="font-size: 14px;">**Swin Transformer**: uses shifted windows for local attention, creating a hierarchical feature map like CNNs. More efficient than ViT for high-resolution images and supports dense prediction tasks (detection, segmentation).</span>

<span style="font-size: 14px;">**MAE (Masked Autoencoders)**: applies BERT-style masked pre-training to ViT. Random patches are masked (75%) and the model reconstructs pixel values. Extremely effective for self-supervised pre-training.</span>

<span style="font-size: 14px;">**DINO/DINOv2**: self-supervised ViT training using self-distillation. Produces features that naturally capture semantic segmentation without any labels.</span>

<span style="font-size: 14px;">**Hybrid architectures**: use CNN stems (e.g., ResNet stages) for initial feature extraction, then ViT for global reasoning. Often more efficient than pure ViT for smaller-scale settings.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why does ViT need large datasets for pre-training, while CNNs work well on smaller datasets?**</span>
  <span style="font-size: 14px;">A: CNNs have strong inductive biases: locality (convolution kernels), translation equivariance, and hierarchical feature extraction. These biases act as implicit regularization, helping CNNs learn efficiently from limited data. ViT lacks these biases - its attention is global from the first layer, and patch embeddings have no built-in notion of spatial locality. With enough data (JFT-300M, ~300M images), ViT learns these patterns from scratch and can outperform CNNs. On smaller datasets (ImageNet-1K), tricks like DeiT's distillation or data augmentation are needed to close the gap.</span>

- <span style="font-size: 14px;">**Q: How does the CLS token aggregate information from all patches?**</span>
  <span style="font-size: 14px;">A: The CLS token is initialized randomly and attends to all patch tokens through self-attention in every layer. By the final layer, it has accumulated information from every patch through repeated attention. An alternative approach (used in some variants) is global average pooling over all patch tokens instead of a CLS token. Both approaches work well, but CLS is the standard ViT convention following BERT's design.</span>

- <span style="font-size: 14px;">**Q: What is the computational complexity of ViT compared to CNNs?**</span>
  <span style="font-size: 14px;">A: ViT's self-attention has $O(N^2 \cdot d)$ complexity where $N = (H/P)^2$ is the number of patches. For a 224x224 image with P=16, N=196 tokens. Doubling the image size quadruples N, making attention $16\times$ more expensive. CNNs scale linearly with pixel count. This is why Swin Transformer uses local windows and why larger ViT models use larger patch sizes (fewer tokens). For standard resolutions, ViT is competitive in throughput but attention cost dominates at high resolutions.</span>

- <span style="font-size: 14px;">**Q: Can ViT be used for tasks beyond classification?**</span>
  <span style="font-size: 14px;">A: Yes. For object detection: use all patch token outputs (not just CLS) with a detection head like DETR. For segmentation: use patch outputs reshaped to 2D with an upsampling decoder (SegFormer, Segmenter). For image generation: replace classification head with a pixel decoder (DiT for diffusion). ViT features are also used in multimodal models (CLIP, LLaVA) where image patches become visual tokens in a joint vision-language model.</span>

- <span style="font-size: 14px;">**Q: Why use Conv2d for patch embedding instead of nn.Linear?**</span>
  <span style="font-size: 14px;">A: They are mathematically equivalent: Conv2d(C, d, kernel_size=P, stride=P) applied to (B, C, H, W) produces the same result as reshaping to (B, N, C*P*P) and applying Linear(C*P*P, d). Conv2d is preferred because: (1) it handles the spatial reshape implicitly, (2) it is more memory-efficient for large images, and (3) GPU implementations of convolution are highly optimized. The parameter count is identical: C*P*P*d + d.</span>

---