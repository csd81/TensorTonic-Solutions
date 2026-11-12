# Recover an IsoFLOP Compute-Optimal Frontier

With a fixed compute budget you can train a large model on few tokens or a small model on many tokens, and only one balance gives the lowest loss. The Chinchilla study (Hoffmann et al., 2022) located that balance in several ways, one of which was IsoFLOP curves, sweeping model size at several fixed budgets. The heart of this problem is recovering the best model size for each budget and fitting a power law that predicts how the best size grows with compute.

## The trade-off at fixed compute

Training a model with $N$ parameters on $D$ tokens costs about $C = 6ND$ floating-point operations, two for the forward pass and four for the backward pass per parameter per token. For a fixed $C$, choosing $N$ also fixes $D = C / (6N)$.

A model that is too small cannot represent the data well however many tokens it sees. A model that is too large sees too few tokens to learn its many parameters. Somewhere between lies the model size with the lowest final loss for that budget, the compute-optimal size.

## One budget, many model sizes

An IsoFLOP curve fixes one compute budget $C$, trains several model sizes with exactly that budget, and plots their final losses against $x = \log N$, the natural log of the model size. The losses form a valley.

The log axis matters. Trained sizes usually grow by a constant factor, such as 2.5 times each, so on a log axis they are evenly spaced and the valley looks roughly symmetric. On a raw axis the same points crowd together at the small end. Near its bottom the valley is well described by a parabola

$$
L_C(x) = a_C x^2 + b_C x + c_C
$$

where $L_C$ is the fitted loss and $a_C$, $b_C$ and $c_C$ are fitted to that budget's observed losses by least squares. A positive $a_C$ means the parabola opens upward and has a minimum.

## From a parabola to the best model

A parabola's minimum sits where its slope is zero, at

$$
x_C^* = -\frac{b_C}{2 a_C}
$$

so the optimal model size is $N_C = e^{x_C^*}$, the token count follows from the compute model as $D_C = C / (6 N_C)$, and the fitted loss at the minimum is $L_C(x_C^*)$.

The fitted vertex usually falls between the sizes that were actually trained. Fitting a parabola uses every point in the sweep, so it can locate the optimum between trained sizes instead of snapping to the best trained size.

## Optimal sizes fall on a line in log space

Repeating this at several budgets gives one optimal size per budget. Plotted on log-log axes against compute, these optima fall close to a straight line

$$
\log N_C = \log k + \alpha \log C
$$

and exponentiating both sides gives the power law $N_C = k C^{\alpha}$. Least squares over the optima gives the scale $k$ and exponent $\alpha$, and the fitted line predicts the best model size for budgets far larger than any that were tested.

Chinchilla found $\alpha$ close to 0.5, which means model size and training tokens should grow at the same rate. The resulting optima work out to roughly 20 training tokens per parameter, as with Chinchilla's 70 billion parameters trained on 1.4 trillion tokens. That finding showed that many earlier large models had been trained on too little data for their size.

## A two-budget example

Take a budget of $3 \times 10^6$ FLOPs with model sizes 100, 250, 1000 and 2500, whose losses are 2.259, 2.048, 2.048 and 2.259. The sizes pair up around 500 on a log scale, since $100 \times 2500 = 250 \times 1000 = 500^2$, and the paired losses are equal. The parabola is therefore symmetric about $\log 500$, so the optimal size is 500 parameters, the optimal token count is $3 \times 10^6 / (6 \times 500) = 1000$, and the loss at the minimum is 2.0.

A second budget of $1.2 \times 10^7$ FLOPs, four times larger, uses sizes 160, 400, 1600 and 4000 with losses 2.159, 1.948, 1.948 and 2.159. Here $160 \times 4000 = 400 \times 1600 = 800^2$, so the vertex is at 800 parameters, with 2500 tokens and a loss of 1.9.

With only two budgets, the frontier passes exactly through both optima. Its exponent is

$$
\alpha = \frac{\log(800 / 500)}{\log 4} \approx 0.339
$$

and its scale follows from $\log k = \log 500 - \alpha \log(3 \times 10^6)$, giving $k \approx 3.184$. In this made-up sweep, quadrupling the compute grows the best model by a factor of 1.6 and the tokens by a factor of 2.5, so most of the extra compute goes to data.

## Common mistakes to avoid

- **Fitting the parabola against raw model size.** The quadratic is in the natural log of the parameter count, where the trained sizes are evenly spaced.
- **Picking the best trained size instead of the vertex.** The optimum comes from the fitted parabola and is usually between trained sizes.
- **Getting the vertex sign wrong.** The minimum is at $-b_C / (2 a_C)$.
- **Forgetting the factor of 6.** Tokens at the optimum are $C / (6 N_C)$.
- **Fitting the frontier on linear axes.** The frontier is a straight line between log compute and log parameters.
- **Reporting the observed minimum loss.** The loss for each budget is the parabola's value at its vertex.
