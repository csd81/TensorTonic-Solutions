## Residual Blocks and Skip Connections

Residual blocks are the core building blocks of ResNets (Residual Networks), introduced by He et al. in 2015. They address the degradation problem: as neural networks get deeper, training accuracy can saturate and then degrade, not because of overfitting, but because deeper networks become harder to optimize.

### The Core Idea

Instead of learning a direct mapping $H(x)$, a residual block learns the residual function:

$$
F(x) = H(x) - x
$$

The output is then:

$$
y = F(x) + x
$$

This is implemented by adding a skip connection (also called a shortcut connection) that bypasses the convolutional layers and adds the input directly to the output.

### Why Residuals Help

- If the optimal transformation is close to identity, the network only needs to learn small residual adjustments $F(x) \approx 0$, which is easier than learning the full mapping
- Gradients flow directly through the skip connection during backpropagation, mitigating the vanishing gradient problem
- The identity path ensures that adding more layers never hurts performance: in the worst case, extra layers learn $F(x) = 0$

### Basic Block Architecture

A standard residual block with two convolutional layers follows this structure:

- $\text{out} = \text{Conv}_1(x)$: first 3x3 convolution
- $\text{out} = \text{ReLU}(\text{BN}_1(\text{out}))$: batch normalization then activation
- $\text{out} = \text{Conv}_2(\text{out})$: second 3x3 convolution
- $\text{out} = \text{BN}_2(\text{out})$: batch normalization (no activation yet)
- $y = \text{ReLU}(\text{out} + x)$: add skip connection, then activate

### Batch Normalization

Each convolution is followed by batch normalization, which normalizes each channel across batch and spatial positions during training:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance, and $\gamma$, $\beta$ are learnable parameters. During evaluation, the stored running mean and variance replace the batch statistics. This problem evaluates the block in that mode, with epsilon 0.00001.

### When Dimensions Match

The skip connection $x + F(x)$ requires that $x$ and $F(x)$ have the same shape. This holds when:

- Input and output channels are the same
- Spatial dimensions are preserved (using padding=1 with 3x3 kernels)

When dimensions do not match (e.g., downsampling or changing channels), a 1x1 convolution projection is used on the skip path to align shapes. This problem focuses on the simpler case where no projection is needed.