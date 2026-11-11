# <span style="font-size: 20px;">Pooling Layers</span>

## <span style="font-size: 16px;">What Pooling Does</span>

<span style="font-size: 14px;">Pooling layers reduce the spatial dimensions of feature maps while retaining the most salient information. They serve three purposes:</span>

- <span style="font-size: 14px;">**Dimensionality reduction**: shrinking the spatial size reduces computation and parameters in subsequent layers</span>
- <span style="font-size: 14px;">**Translation invariance**: small shifts in the input do not significantly change the pooled output, making the network more robust to spatial perturbations</span>
- <span style="font-size: 14px;">**Noise suppression**: aggregating over a region smooths out small fluctuations in activation values</span>

<span style="font-size: 14px;">Pooling operates independently on each channel: it reduces</span> $(C, H, W)$ <span style="font-size: 14px;">to</span> $(C, H_{\text{out}}, W_{\text{out}})$ <span style="font-size: 14px;">without changing the number of channels. The output spatial dimensions are:</span>

$$
\begin{aligned}
H_{\text{out}} = \left\lfloor \frac{H - k}{s} \right\rfloor + 1, \\
W_{\text{out}} = \left\lfloor \frac{W - k}{s} \right\rfloor + 1
\end{aligned}
$$

<span style="font-size: 14px;">where</span> $k$ <span style="font-size: 14px;">is the pool size and</span> $s$ <span style="font-size: 14px;">is the stride.</span>

## <span style="font-size: 16px;">Max Pooling</span>

<span style="font-size: 14px;">Max pooling selects the largest value in each window:</span>

$$
y_{c,i,j} = \max_{0 \le m,n < k} x_{c,\, i \cdot s + m,\, j \cdot s + n}
$$

<span style="font-size: 14px;">Properties of max pooling:</span>
- <span style="font-size: 14px;">**Preserves dominant features**: the strongest activation survives, which is useful for detecting the presence of a feature regardless of its exact position</span>
- <span style="font-size: 14px;">**Sparse gradient**: only the position of the maximum receives gradient during backpropagation; all other positions get zero gradient</span>
- <span style="font-size: 14px;">**Non-linear**: max is a non-linear operation, adding representational capacity to the network</span>
- <span style="font-size: 14px;">**Most common choice**: used in AlexNet, VGG, and the vast majority of CNN architectures</span>

<span style="font-size: 14px;">The standard configuration is 2x2 with stride 2, which halves the spatial dimensions. 3x3 with stride 2 is used in some architectures (Inception) for slightly overlapping pooling.</span>

## <span style="font-size: 16px;">Average Pooling</span>

<span style="font-size: 14px;">Average pooling computes the mean of each window:</span>

$$
y_{c,i,j} = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} x_{c,\, i \cdot s + m,\, j \cdot s + n}
$$

<span style="font-size: 14px;">Properties of average pooling:</span>
- <span style="font-size: 14px;">**Smooth aggregation**: considers all values equally rather than just the maximum</span>
- <span style="font-size: 14px;">**Dense gradient**: every position in the window receives gradient during backpropagation (each gets</span> $1/k^2$ <span style="font-size: 14px;">of the upstream gradient)</span>
- <span style="font-size: 14px;">**Less common for spatial downsampling**: max pooling generally outperforms average pooling in classification tasks</span>
- <span style="font-size: 14px;">**Used as Global Average Pooling (GAP)**: averaging over the entire feature map is the standard final layer in modern CNNs</span>

## <span style="font-size: 16px;">Backward Pass: Gradient Routing</span>

<span style="font-size: 14px;">The backward pass for pooling distributes the upstream gradient</span> $\frac{\partial L}{\partial y_{c,i,j}}$ <span style="font-size: 14px;">back to the input elements.</span>

<span style="font-size: 14px;">**Max pooling backward:**</span>

$$
\frac{\partial L}{\partial x_{c,m,n}} = \sum_{(i,j)} \frac{\partial L}{\partial y_{c,i,j}} \cdot \mathbb{1}[x_{c,m,n} = \max(\text{window}_{c,i,j})]
$$

<span style="font-size: 14px;">The gradient flows only to the position of the maximum. During the forward pass, you must record which element was the maximum in each window (the "argmax index"). If two elements tie, the gradient is split equally among them.</span>

<span style="font-size: 14px;">**Average pooling backward:**</span>

$$
\frac{\partial L}{\partial x_{c,m,n}} = \sum_{(i,j)} \frac{\partial L}{\partial y_{c,i,j}} \cdot \frac{1}{k^2}
$$

<span style="font-size: 14px;">Each element receives an equal share of the upstream gradient. This is simpler than max pooling backward because no index tracking is needed.</span>

<span style="font-size: 14px;">When windows overlap (stride < pool_size), a single input element may contribute to multiple output positions. The gradient from each window is **accumulated** using addition. This is a direct application of the multivariate chain rule: the total gradient is the sum of partial contributions from all downstream consumers.</span>

## <span style="font-size: 16px;">Global Average Pooling (GAP)</span>

<span style="font-size: 14px;">Global Average Pooling applies average pooling with</span> $k = H$<span style="font-size: 14px;">, collapsing each channel's feature map into a single scalar:</span>

$$
y_c = \frac{1}{H \times W} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} x_{c,i,j}
$$

<span style="font-size: 14px;">The output shape is</span> $(C,)$<span style="font-size: 14px;">. This replaces the traditional flatten + fully-connected layers at the end of a CNN with several advantages:</span>
- <span style="font-size: 14px;">**No extra parameters**: eliminates the large FC layer that dominated parameter counts in early CNNs (e.g., VGG had 123M FC parameters vs 15M conv parameters)</span>
- <span style="font-size: 14px;">**Spatial invariance**: works regardless of input spatial size</span>
- <span style="font-size: 14px;">**Regularization effect**: forces the network to produce meaningful per-channel averages rather than relying on spatial patterns in the flattened representation</span>

<span style="font-size: 14px;">Introduced in Network-in-Network (Lin et al., 2013) and adopted by GoogLeNet, ResNet, EfficientNet, and virtually every modern CNN.</span>

## <span style="font-size: 16px;">Pooling vs Strided Convolutions</span>

<span style="font-size: 14px;">Modern architectures increasingly replace pooling with strided convolutions (stride 2) for downsampling:</span>

- <span style="font-size: 14px;">**Strided convolution**: learnable downsampling; the network decides what information to retain</span>
- <span style="font-size: 14px;">**Max pooling**: fixed rule (keep the max); no parameters, less computation</span>
- <span style="font-size: 14px;">**Trend**: ResNet uses strided convolutions for most downsampling but keeps one max pool at the input. Many recent architectures (ConvNeXt) eliminate pooling entirely.</span>

<span style="font-size: 14px;">The key tradeoff: strided convolutions add parameters and computation but give the network more flexibility. Pooling is parameter-free and fast but imposes a fixed inductive bias.</span>

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why does max pooling work better than average pooling for classification?**</span>
  <span style="font-size: 14px;">A: In classification, the presence of a feature is more important than its average response. Max pooling preserves strong activations (feature present) while suppressing weak ones. Average pooling dilutes strong activations with zeros, reducing signal strength.</span>

- <span style="font-size: 14px;">**Q: How does the backward pass handle overlapping windows?**</span>
  <span style="font-size: 14px;">A: When stride < pool_size, windows overlap and a single input element contributes to multiple outputs. During backprop, gradients from all contributing windows are summed at each input position. This is gradient accumulation, the same principle as in autograd when a variable has multiple consumers.</span>

- <span style="font-size: 14px;">**Q: What is the difference between max pooling and max unpooling?**</span>
  <span style="font-size: 14px;">A: Max unpooling (used in segmentation networks like SegNet) reverses max pooling by placing values at the recorded argmax positions and filling the rest with zeros. It requires storing the argmax indices from the forward pass, which increases memory usage.</span>

- <span style="font-size: 14px;">**Q: Why did ResNet keep one max pool layer?**</span>
  <span style="font-size: 14px;">A: ResNet uses a 3x3 max pool (stride 2) right after the initial 7x7 convolution to aggressively reduce spatial dimensions early. This is computationally efficient: reducing from 112x112 to 56x56 before the expensive residual blocks saves significant computation.</span>

- <span style="font-size: 14px;">**Q: Does pooling have learnable parameters?**</span>
  <span style="font-size: 14px;">A: Standard max and average pooling have zero parameters. However, learnable pooling variants exist: Stochastic Pooling (samples from multinomial distribution weighted by activations), Fractional Max Pooling (randomized pool sizes), and SoftPool (softmax-weighted average). These are rarely used in practice.</span>

---