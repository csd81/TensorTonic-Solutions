# <span style="font-size: 20px;">Multi-Layer Perceptron (Forward Pass)</span>

<span style="font-size: 14px;">A multi-layer perceptron (MLP) is the foundational feedforward neural network. It consists of an input layer, one or more hidden layers, and an output layer. Each layer performs a linear transformation followed by a non-linear activation. The forward pass propagates input data through the network to produce predictions.</span>

---

## <span style="font-size: 16px;">Architecture</span>

<span style="font-size: 14px;">An MLP with</span> $L$ <span style="font-size: 14px;">layers is defined by weight matrices</span> $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ <span style="font-size: 14px;">and bias vectors</span> $b^{(l)} \in \mathbb{R}^{n_l}$ <span style="font-size: 14px;">for</span> $l = 1, \dots, L$<span style="font-size: 14px;">, where</span> $n_l$ <span style="font-size: 14px;">is the width of layer</span> $l$<span style="font-size: 14px;">:</span>

* <span style="font-size: 14px;">**Input layer** (</span>$a^{(0)}$<span style="font-size: 14px;">): the raw input vector, no parameters</span>
* <span style="font-size: 14px;">**Hidden layers**: linear transformation + non-linear activation</span>
* <span style="font-size: 14px;">**Output layer**: linear transformation only (activation depends on the task, applied separately)</span>

---

## <span style="font-size: 16px;">Forward Pass Equations</span>

<span style="font-size: 14px;">For each layer</span> $l = 1, \dots, L$<span style="font-size: 14px;">:</span>

$$
z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \begin{cases} \text{ReLU}(z^{(l)}) & l < L \text{ (hidden layers)} \\ z^{(l)} & l = L \text{ (output layer)} \end{cases}
$$

<span style="font-size: 14px;">The key distinction between</span> $z$ <span style="font-size: 14px;">(pre-activation) and</span> $a$ <span style="font-size: 14px;">(activation) matters for backpropagation. The pre-activation is needed to compute the activation function's derivative, and the previous layer's activation is needed to compute the weight gradient.</span>

---

## <span style="font-size: 16px;">Matrix Dimensions</span>

<span style="font-size: 14px;">For a network with layer widths</span> $[n_0, n_1, \dots, n_L]$<span style="font-size: 14px;">:</span>

* $W^{(l)}$ <span style="font-size: 14px;">has shape</span> $(n_l, n_{l-1})$ <span style="font-size: 14px;">: rows = output neurons, columns = input neurons</span>
* $b^{(l)}$ <span style="font-size: 14px;">has shape</span> $(n_l,)$
* $z^{(l)}$ <span style="font-size: 14px;">and</span> $a^{(l)}$ <span style="font-size: 14px;">have shape</span> $(n_l,)$
* <span style="font-size: 14px;">Total parameters:</span> $\sum_{l=1}^{L} (n_l \cdot n_{l-1} + n_l)$

<span style="font-size: 14px;">Example: a 3-4-2-1 network has</span> $W^{(1)}: 4 \times 3$<span style="font-size: 14px;">,</span> $W^{(2)}: 2 \times 4$<span style="font-size: 14px;">,</span> $W^{(3)}: 1 \times 2$<span style="font-size: 14px;">. Total: 12+4+8+2+2+1 = 29 parameters.</span>

---

## <span style="font-size: 16px;">Why Store Intermediate Values</span>

<span style="font-size: 14px;">During training, backpropagation computes gradients layer by layer in reverse. Each layer's gradient computation requires:</span>

* $a^{(l-1)}$<span style="font-size: 14px;">: to compute</span> $\partial L / \partial W^{(l)} = \delta^{(l)} \cdot (a^{(l-1)})^T$
* $z^{(l)}$<span style="font-size: 14px;">: to compute the activation derivative</span> $f'(z^{(l)})$

<span style="font-size: 14px;">Without storing these, you would need to recompute the entire forward pass for each layer's gradient, making training</span> $O(L)$ <span style="font-size: 14px;">times slower. This memory-for-speed tradeoff is fundamental to efficient neural network training.</span>

---

## <span style="font-size: 16px;">Universal Approximation</span>

<span style="font-size: 14px;">The Universal Approximation Theorem states that an MLP with a single hidden layer of sufficient width can approximate any continuous function on a compact set to arbitrary accuracy. However:</span>

* <span style="font-size: 14px;">Width sufficient for approximation can be exponentially large</span>
* <span style="font-size: 14px;">Deep networks (more layers, less width) are exponentially more parameter-efficient for many function classes</span>
* <span style="font-size: 14px;">The theorem guarantees existence but not learnability via gradient descent</span>

---

## <span style="font-size: 16px;">Parameter Counting</span>

<span style="font-size: 14px;">A critical interview skill is quickly computing the number of parameters in a network:</span>

* <span style="font-size: 14px;">Each layer has</span> $n_l \times n_{l-1}$ <span style="font-size: 14px;">weights plus</span> $n_l$ <span style="font-size: 14px;">biases</span>
* <span style="font-size: 14px;">Convolutional layers use weight sharing, so the count is kernel_size</span> $\times$ <span style="font-size: 14px;">channels, not input_size</span> $\times$ <span style="font-size: 14px;">output_size</span>
* <span style="font-size: 14px;">Batch norm adds</span> $2n$ <span style="font-size: 14px;">parameters per layer (scale and shift)</span>
* <span style="font-size: 14px;">Embedding layers have vocabulary_size</span> $\times$ <span style="font-size: 14px;">embedding_dim parameters</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why use ReLU in hidden layers but not the output layer?** The output layer must match the task: linear for regression, sigmoid for binary classification, softmax for multi-class. ReLU in the output layer would clip negative predictions to zero, which is wrong for regression</span>
* <span style="font-size: 14px;">**Why store activations separately from pre-activations?** During backpropagation, the weight gradient at layer</span> $l$ <span style="font-size: 14px;">is</span> $\delta^{(l)} \cdot (a^{(l-1)})^T$<span style="font-size: 14px;">, requiring the previous layer's activation. The activation derivative</span> $f'(z^{(l)})$ <span style="font-size: 14px;">requires the pre-activation. Both are needed, and they differ whenever the activation clips values (as ReLU does)</span>
* <span style="font-size: 14px;">**What is the computational complexity of the forward pass?** Each layer</span> $l$ <span style="font-size: 14px;">requires a matrix-vector product:</span> $O(n_l \times n_{l-1})$<span style="font-size: 14px;">. The total is</span> $O(\sum_l n_l \cdot n_{l-1})$<span style="font-size: 14px;">, which is dominated by the widest layers. For batch processing, matrix-vector becomes matrix-matrix, and GPU parallelism makes wider layers relatively cheaper</span>
* <span style="font-size: 14px;">**Deep vs wide networks?** Depth enables hierarchical feature learning (low-level to high-level), while width increases the number of features at each level. Deep networks are more parameter-efficient but harder to train (vanishing gradients, harder optimization). Residual connections and normalization techniques address the training difficulty</span>
* <span style="font-size: 14px;">**Can you do the forward pass without storing intermediates?** Yes, but then backpropagation requires recomputing the forward pass for each gradient, making training much slower. Gradient checkpointing is a middle ground: store only some intermediate values and recompute the rest, trading compute for memory</span>

---