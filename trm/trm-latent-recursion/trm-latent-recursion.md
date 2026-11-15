# Latent Recursion

Latent recursion is the inner loop of the Tiny Recursive Model (Jolicoeur-Martineau, 2025). It takes an embedded question $x$, a current answer embedding $y$ and a latent reasoning state $z$, all with shape $(B, L, D)$, and improves the answer by thinking in the latent space first. This problem covers one recursion process, which is $n$ latent updates followed by one answer update, all computed by the same tiny network $f$.

## One network instead of two

The Hierarchical Reasoning Model (Wang et al., 2025) that TRM simplifies keeps two networks and two latents. A low-level module $f_L$ updates a fast latent $z_L$ many times, and a high-level module $f_H$ updates a slow latent $z_H$ once per group of low-level steps. HRM justifies the two frequencies with biological arguments and needs twice the parameters of one network.

TRM reads the slow latent $z_H$ as the current answer $y$, since the output head reads it directly, and the fast latent $z_L$ as the reasoning state $z$. The two modules then do two different jobs.

- **Latent update.** The network sees the question, the answer and the latent, and produces a new latent, which is reasoning about the problem.
- **Answer update.** The network sees only the answer and the latent, and produces a new answer, which is writing down what the reasoning concluded.

Because the question is present in the first job and absent in the second, the network can tell which job it is doing from its input alone. Section 4.3 uses this to replace $f_L$ and $f_H$ with one network, and Table 1 reports that the single network raises Sudoku-Extreme accuracy from 82.4% to 87.4% while halving the parameters. That section names the frequent module $f_H$ once by mistake, and the frequent module is $f_L$.

## Why the answer and the latent are carried separately

The paper argues that both states are needed. Without $z$, the model would forget how it reached its previous answer, since $z$ plays the role of a hidden chain of thought. Without $y$, the model would have to store the answer inside $z$ and lose room for reasoning. Table 2 compares both alternatives on Sudoku-Extreme.

- **Two features.** Carrying $y$ and $z$ gives 87.4%.
- **Multi-scale latent.** Splitting $z$ into $n$ separate latents, one per update, gives 77.6% with $n + 1 = 7$ features.
- **Single latent.** Carrying only one feature gives 71.9%.

In this problem, $z$ is rebound after every update, so only the final latent $z^{(n)}$ is carried forward. The history of intermediate latents makes the update order visible.

## How the inputs are combined

The pseudocode in Figure 3 writes the calls as a network of three or two arguments, while Section 4.3 writes $f_L(x + y + z)$ and $f_H(y + z)$. The released code adds the tensors element-wise before the first layer, and this problem fixes that convention. Each latent update therefore evaluates $f(x + y + z^{(i-1)})$, and the answer update evaluates $f(y + z^{(n)})$.

The supplied network is a stand-in that keeps the post-norm SwiGLU channel layer and the gain-free RMS normalization of the TRM block but omits token mixing, so the recursion schedule can be checked cheaply. The Tiny Mixer Network problem covers the full block. In the complete model, latent recursion is repeated $T$ times per supervision step inside deep recursion, with $n = 6$ and $T = 3$ in the paper's main setting, and the later stages build on these $n + 1$ calls.

## Two updates traced through a one-layer network

Take $B = L = 1$, $D = 2$ and one layer with hidden width $H = 1$, gate weights $(0, 1)$, up weights $(1, 1)$ and down weights $(1, 1)$. Let $x = (1, 0)$, $y = (0, 0.5)$, $z = (0.5, 0)$ and $n = 2$.

- The first latent update sees $x + y + z = (1.5, 0.5)$. The gate is $\operatorname{SiLU}(0.5) = 0.3112$ and the up branch is $2.0$, so the product $0.6225$ is added to both channels, giving $(2.1225, 1.1225)$. RMS normalization divides by $1.6978$, so $z^{(1)} = (1.2502, 0.6611)$.
- The second latent update sees $x + y + z^{(1)} = (2.2502, 1.1611)$ and produces $z^{(2)} = (1.1080, 0.8789)$.
- The answer update sees $y + z^{(2)} = (1.1080, 1.3789)$, which contains no question term, and produces $y' = (0.9654, 1.0334)$.

The history is $z^{(1)}$ followed by $z^{(2)}$, from three network evaluations.

## Common mistakes to avoid

- **Adding the question to the answer update.** The answer update must see only $y + z^{(n)}$, because the missing question is what tells the shared network to write the answer.
- **Updating the answer inside the loop.** The answer changes once per recursion process, after all $n$ latent updates.
- **Updating the answer from the initial latent.** The answer must read $z^{(n)}$, the result of the reasoning, and reading $z^{(0)}$ ignores every latent update.
- **Dropping the answer from the latent update.** Each latent update needs $x + y + z^{(i-1)}$, since the reasoning has to know the current answer it is trying to improve.
- **Recording the history wrongly.** The history starts at $z^{(1)}$, ends at $z^{(n)}$, and excludes the incoming latent.
- **Turning off gradients.** Deep recursion backpropagates through the final recursion process, so this function must leave every evaluation attached to the graph.