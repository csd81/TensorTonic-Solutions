# Gated Delta Rule as Online Learning

The gated delta rule is the memory update at the heart of Gated DeltaNet, and it decides how a fixed-size matrix state changes when one key and value arrive. This problem covers the paper's online learning view, which derives it and four earlier rules from one template.

## State orientation and notation

The state $S \in \mathbb{R}^{d_v \times d_k}$ has one row per value channel and one column per key channel, where $d_k$ is the key width and $d_v$ is the value width. A key $k \in \mathbb{R}^{d_k}$ retrieves $S k \in \mathbb{R}^{d_v}$, and a value $v \in \mathbb{R}^{d_v}$ is written as $v k^\top$, with transitions multiplying on the right. The official code stores the transposed state, but this problem uses the paper's orientation. The decay $\alpha \in (0, 1]$ shrinks the old state, and the write strength $\beta \in (0, 1]$ controls how strongly the new association is written.

## One template, five objectives

Following Liu et al. (2024), Table 1 of Yang, Kautz and Hatamizadeh, 2025 writes every update as the exact minimizer of an objective with two parts. A Frobenius term $\lVert S_t - S_{t-1} \rVert_F^2$ keeps the new state near the old one, and a second term fits the current pair.

- **Linear attention.** The fitting term is the negative inner product $-2 \langle S_t k_t, v_t \rangle$, which rewards recalling anything aligned with $v_t$, and the minimizer simply adds $v_t k_t^\top$.
- **Mamba2.** The anchor becomes $\alpha_t S_{t-1}$, so the minimizer adds $v_t k_t^\top$ to a uniformly shrunk state.
- **Longhorn.** The fitting term is the regression loss $\beta_t \lVert S_t k_t - v_t \rVert^2$, and the closed form uses the step size $\epsilon_t = \beta_t / (1 + \beta_t k_t^\top k_t)$.
- **DeltaNet.** The inner product is taken with the prediction error $\beta_t (v_t - S_{t-1} k_t)$, which yields the delta rule.
- **Gated DeltaNet.** Both the anchor and the prediction error use the decayed state $\alpha_t S_{t-1}$.

Table 1 prints the Longhorn objective with a minus sign in front of the regression term. With a minus sign the minimizer would push the recall away from the value, and the listed closed form is the minimizer only when the regression term is added, so this problem treats the printed sign as a typo and uses a plus sign.

## Regression turns writing into replacing

The additive rules pile associations on top of each other. Linear attention turns the recall $S k$ into $S k + v (k^\top k)$, so the old content never leaves. The delta rule first subtracts $\beta S k k^\top$, which removes a fraction $\beta$ of the current recall, and then writes $\beta v k^\top$. For a unit-length key with $\beta = 1$, the recall becomes exactly $v$, so one association is replaced while every direction orthogonal to $k$ is left alone.

Longhorn solves the regression problem implicitly and exactly, so its step size shrinks when $k^\top k$ is large, while DeltaNet takes one explicit gradient step with learning rate $\beta$. The two agree only when $\beta k^\top k$ is small.

## Decay as weight decay

The gated delta rule reads naturally as stochastic gradient descent with weight decay. Starting from the decayed state $\alpha S$ and taking one gradient step on $\frac{1}{2} \lVert S k - v \rVert_2^2$ gives

$$
S_{\text{GDN}} = \alpha S - \beta \left( \alpha S k - v \right) k^\top = S \left( \alpha \left( I - \beta k k^\top \right) \right) + \beta v k^\top
$$

Section 3.2 of the paper explains why both halves matter. The decay can clear the whole memory quickly when $\alpha \to 0$, which helps when the context switches, while the delta rule replaces one association precisely when $\alpha \to 1$. DeltaNet alone cannot forget quickly, and Mamba2 forgets every association at the same rate.

## A key that is not axis aligned

Take $d_v = 1$ and $d_k = 2$ with $S = (2, 1)$, $k = (0.6, 0.8)$, $v = 3$, $\alpha = 0.5$ and $\beta = 1$. The state recalls $S k = 2.0$, and the erase term is $S k k^\top = (1.2, 1.6)$.

- Linear attention gives $(3.8, 3.4)$, which recalls 5.0, a residual of 2.
- Mamba2 gives $(2.8, 2.9)$, which recalls 4.0, a residual of 1.
- Longhorn has $\epsilon = 0.5$ and gives $(2.3, 1.4)$, which recalls 2.5.
- DeltaNet gives $(2.6, 1.8)$ and Gated DeltaNet gives $(2.2, 2.1)$, and both recall exactly 3.0.

The last two differ only along $(0.8, -0.6)$, which the key never touches, where DeltaNet keeps the old component at 1.0 and Gated DeltaNet halves it to 0.5.

## Common mistakes to avoid

- **Writing $k v^\top$.** In the paper's orientation the write is $v k^\top$, which has shape $d_v \times d_k$ and matches the state.
- **Decaying the write.** The gated rule multiplies only the old state's transition by $\alpha$, so the new association enters at full strength $\beta$.
- **Decaying only the erase term.** The update $S (I - \alpha \beta k k^\top)$ keeps the untouched directions at full strength and misses the forgetting entirely.
- **Using $\beta$ inside Longhorn.** Longhorn's step size is $\epsilon$, which shrinks as the key norm grows, and it applies to both the erase and the write.
- **Dropping the write strength from the erase.** DeltaNet removes a fraction $\beta$ of the current recall, so erasing the full recall overshoots when $\beta < 1$.
- **Measuring the residual on the old state.** Each residual uses the new state.