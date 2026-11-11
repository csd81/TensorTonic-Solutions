Convolution is the fundamental operation behind convolutional neural networks (CNNs). Instead of connecting every input to every output as a fully connected layer does, a convolution uses a small **kernel** (also called a filter) that slides across the spatial dimensions of the input, computing a weighted sum at each position. This dramatically reduces the number of parameters and exploits the spatial structure of data like images.

## From Fully Connected to Convolution

Consider a grayscale image of size $H \times W$. A fully connected layer mapping this to an output of the same size would need $(H \times W)^2$ parameters. For a modest 28x28 image, that is over 600,000 weights for a single layer. A convolution with a 3x3 kernel needs just 9 weights (plus a bias), regardless of the image size. This works because the same kernel is applied at every spatial position: a property called **weight sharing**.

Weight sharing also introduces **translation equivariance**: if the input shifts, the output shifts by the same amount. This is exactly the right inductive bias for visual data, where a feature (like an edge) should be detected the same way regardless of where it appears.

## The Convolution Operation

For a single input channel and a single output channel, 2D convolution works as follows. Given:

* Input tensor $x$ of shape $(H, W)$
* Kernel $w$ of shape $(k, k)$

The output at position $(i, j)$ is:

$$
y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} x_{i+m,\, j+n} \cdot w_{m,n} + b
$$

where $b$ is a scalar bias. The kernel slides across all valid positions, producing an output of shape:

$$
H_{\text{out}} = H - k + 1
$$

$$
W_{\text{out}} = W - k + 1
$$

This is called "valid" convolution because the kernel only visits positions where it fits entirely within the input. No padding is added.

## Multiple Channels

Real inputs typically have multiple channels. A color image has 3 channels (RGB), and intermediate layers in a CNN can have hundreds. The convolution generalizes naturally:

* Input has $C_{\text{in}}$ channels, so $x$ has shape $(C_{\text{in}}, H, W)$
* Each output channel has its own kernel of shape $(C_{\text{in}}, k, k)$, which spans all input channels
* With $C_{\text{out}}$ output channels, the full weight tensor has shape $(C_{\text{out}}, C_{\text{in}}, k, k)$
* Each output channel also has its own scalar bias, giving a bias vector of shape $(C_{\text{out}},)$

The output at channel $c$ and position $(i, j)$ is:

$$
\begin{aligned}
y_{c,i,j} = \sum_{c'=0}^{C_{\text{in}}-1}
&\sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \\
&x_{c',\, i+m,\, j+n} \cdot w_{c,c',m,n} + b_c
\end{aligned}
$$

Each output channel looks at **all** input channels through its own set of weights. This is what allows the network to learn complex cross-channel features.

## Batched Inputs

In practice, inputs come in batches. The full input tensor has shape $(N, C_{\text{in}}, H, W)$ where $N$ is the batch size. The convolution applies independently to each sample in the batch, producing output of shape $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$. The weights are shared across all samples.

## The Sliding Window Viewpoint

The most intuitive way to implement convolution is with a sliding window:

* For each output position $(i, j)$, extract the patch $x[:, :, i:i+k, j:j+k]$ of shape $(N, C_{\text{in}}, k, k)$
* Flatten the patch to shape $(N, C_{\text{in}} \cdot k \cdot k)$
* Flatten the weight to shape $(C_{\text{out}}, C_{\text{in}} \cdot k \cdot k)$
* Compute $\text{output}[:, :, i, j] = \text{patch\_flat} \times \text{weight\_flat}^T + b$ using matrix multiplication

This approach loops over spatial positions but vectorizes across the batch and channel dimensions. For small kernels and inputs it is straightforward and educational.

## The im2col Approach

Production implementations avoid the spatial loop entirely using a technique called **im2col** (image to column). The idea is to extract all patches at once and stack them into a single large matrix:

* Extract all $H_{\text{out}} \times W_{\text{out}}$ patches from the input
* Reshape each patch from $(C_{\text{in}}, k, k)$ to a column of length $C_{\text{in}} \cdot k^2$
* Stack all columns into a matrix of shape $(C_{\text{in}} \cdot k^2,\ H_{\text{out}} \cdot W_{\text{out}})$

Then the entire convolution becomes a single matrix multiplication:

$$
Y_{\text{flat}} = W_{\text{flat}} \cdot X_{\text{col}} + b
$$

where $W_{\text{flat}}$ has shape $(C_{\text{out}}, C_{\text{in}} \cdot k^2)$ and $X_{\text{col}}$ has shape $(C_{\text{in}} \cdot k^2, H_{\text{out}} \cdot W_{\text{out}})$. The output $Y_{\text{flat}}$ has shape $(C_{\text{out}}, H_{\text{out}} \cdot W_{\text{out}})$ and is reshaped back to $(C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$.

This trades memory for speed: the column matrix duplicates input data (overlapping patches share elements), but the operation maps directly to optimized BLAS routines.

## Learnable Parameters

In a convolution layer, the kernel weights and biases are learnable. In PyTorch, this means wrapping them in nn.Parameter:

* self.weight of shape $(C_{\text{out}}, C_{\text{in}}, k, k)$, initialized randomly
* self.bias of shape $(C_{\text{out}},)$, often initialized to zero

Using nn.Parameter tells PyTorch that these tensors should be updated by the optimizer during training. They also appear when you call model.parameters() or print the model.

## Output Dimensions

For a convolution without padding or stride (the simplest case), the output spatial dimensions are:

$$
H_{\text{out}} = H - k + 1
$$

$$
W_{\text{out}} = W - k + 1
$$

With padding $p$ and stride $s$, the general formula is:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

For this problem, we use no padding ($p = 0$) and unit stride ($s = 1$), so only the first formula applies.

## What Each Output Channel Learns

Each output channel can be thought of as a different feature detector. In the first layer of a CNN:

* Some channels might detect horizontal edges
* Some might detect vertical edges
* Some might detect corners or textures

In deeper layers, the features become more abstract: parts of objects, shapes, semantic patterns. Each output channel has its own kernel that spans all input channels, allowing it to combine low-level features into higher-level ones.

## Receptive Field

The **receptive field** of an output neuron is the region of the original input that influences its value. For a single convolution layer with kernel size $k$, each output neuron sees a $k \times k$ patch. Stacking multiple convolution layers increases the receptive field: two 3x3 layers give an effective receptive field of 5x5, three give 7x7. This is why deep CNNs can capture large-scale patterns despite using small kernels.

## Convolution vs Cross-Correlation

Technically, what deep learning frameworks call "convolution" is actually **cross-correlation**. True convolution flips the kernel before sliding:

$$
y_{i,j} = \sum_{m} \sum_{n} x_{i+m,\, j+n} \cdot w_{k-1-m,\, k-1-n}
$$

In practice, since the kernel weights are learned, flipping makes no difference: the network simply learns the flipped version. All major frameworks (PyTorch, TensorFlow, JAX) use cross-correlation and call it "convolution." This is a universal convention.

## Parameter Count

The total number of learnable parameters in a convolution layer is:

$$
\text{params} = C_{\text{out}} \times C_{\text{in}} \times k^2 + C_{\text{out}}
$$

The first term is the weights, the second is the biases. For example, a layer with 3 input channels, 16 output channels, and a 3x3 kernel has $16 \times 3 \times 9 + 16 = 432 + 16 = 448$ parameters. Compare this to a fully connected layer between the same flattened input and output, which would need millions of parameters.

## Implementing as nn.Module

To implement a convolution layer as an nn.Module:

* In __init__, create self.weight and self.bias as nn.Parameter with the correct shapes
* In forward, implement the sliding window operation
* The weight shape must be $(C_{\text{out}}, C_{\text{in}}, k, k)$ following PyTorch's convention
* The bias shape must be $(C_{\text{out}},)$

The forward pass extracts patches from the input at each spatial position, flattens them, and computes a matrix multiplication with the flattened weights. This is the core of what happens inside torch.nn.Conv2d.

## Comparison with PyTorch Built-in

PyTorch's nn.Conv2d supports many additional features beyond the basic operation:

* **Padding**: adds zeros (or other values) around the input to control output size
* **Stride**: skips positions to downsample the output
* **Dilation**: spaces out kernel elements to increase receptive field without adding parameters
* **Groups**: splits channels into independent groups for efficiency (depthwise convolution uses groups = in_channels)
* **No bias**: option to remove the bias term

The from-scratch version in this problem covers the simplest case: no padding, stride 1, no dilation, no groups, with bias. This captures the essence of convolution while keeping the implementation focused.

## Role in CNN Architectures

Convolution layers are the building blocks of virtually all image-processing neural networks:

* **LeNet** (1998): The original CNN, used 5x5 convolutions for digit recognition
* **AlexNet** (2012): Popularized deep CNNs with 11x11, 5x5, and 3x3 kernels
* **VGGNet** (2014): Showed that stacking many 3x3 convolutions is more effective than using large kernels
* **ResNet** (2015): Added skip connections between convolution blocks, enabling much deeper networks
* **EfficientNet** (2019): Systematically scaled width, depth, and resolution of convolution networks

Even in the era of transformers, convolutions remain essential in vision tasks (ConvNeXt, hybrid architectures) and appear in speech, audio, and time-series models.

## Computational Complexity

The number of multiply-add operations for a single convolution layer is:

$$
\text{FLOPs} = C_{\text{out}} \times C_{\text{in}} \times k^2 \times H_{\text{out}} \times W_{\text{out}}
$$

This grows with both the number of channels and the spatial resolution. Techniques to reduce this cost include:

* **Depthwise separable convolution**: splits into a per-channel spatial convolution followed by a 1x1 pointwise convolution, reducing FLOPs by a factor of roughly $k^2$
* **Strided convolution**: reduces spatial resolution, cutting FLOPs proportionally
* **1x1 convolutions (pointwise)**: mix channels without spatial computation, used as bottleneck layers