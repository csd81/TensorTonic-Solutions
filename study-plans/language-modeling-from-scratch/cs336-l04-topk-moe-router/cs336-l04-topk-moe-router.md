# Top-k MoE Router with Load Statistics

A mixture-of-experts (MoE) layer replaces one large MLP with many smaller expert MLPs and sends each token to only a few of them. Mixtral, DeepSeek-V3 and many other large models use it to grow their parameter count without growing the compute spent per token. The heart of this problem is choosing each token's experts deterministically, weighting their outputs, and measuring how evenly the load is spread.

## Why route tokens to experts

In a dense model every token passes through every parameter of the MLP. An MoE layer with $E$ experts holds roughly $E$ times as many MLP parameters, but each token uses only $k$ of the experts, often 1, 2 or 8. The model gains capacity while the compute per token stays close to that of a dense MLP only $k$ experts wide.

The router is the small piece that decides which experts a token visits. It is a single linear layer that produces one score per expert.

## The highest scores win, and ties go to the lower index

For token $t$ with vector $x_t$ and expert $e$ with router column $w_e$, the score is

$$
s_{t,e} = x_t^{\mathsf T} w_e
$$

The token goes to the $k$ experts with the highest scores. When two experts have exactly equal scores, the lower expert index wins. Exact ties are common in practice, for example when the router weights start at zero or when bfloat16 rounds nearby scores to the same value. Without a fixed rule, two runs could send the same token to different experts.

## Renormalizing over the chosen experts

The selected experts' outputs are combined with weights from a softmax over their scores only. If $e_{t,j}$ is the expert in selected position $j$, its weight is

$$
\alpha_{t,j} = \frac{\exp(s_{t,e_{t,j}})}{\sum_{r=1}^{k} \exp(s_{t,e_{t,r}})}
$$

The weights of the $k$ selected experts sum to 1, and unselected experts contribute nothing. This renormalize-after-selection scheme is the one Mixtral uses. Some models instead take the softmax over all experts first and keep those weights for the selected ones, which gives weights that sum to at most 1.

Many recent models, following DeepSeekMoE, also include a shared expert that every token visits. The shared expert learns knowledge every token needs, which frees the routed experts to specialize. Its output is added to the weighted sum:

$$
y_t = u_t^{\text{shared}} + \sum_{j=1}^{k} \alpha_{t,j} \, u_{t,e_{t,j}}
$$

Here $u_{t,e}$ is expert $e$'s output for token $t$. In a real layer only the $k$ selected experts run on each token, which is where the compute saving comes from. Here the outputs of every expert are given so that the routing step can be studied on its own.

## Why load statistics matter

A router left alone tends to collapse. A few experts get slightly better early in training, the router sends them more tokens, they improve further, and the rest of the experts go unused. In distributed training an overloaded expert also becomes a bottleneck, because the GPU holding it has more work than the others.

Two statistics describe the load over $T$ tokens. The token fraction $f_e$ is the share of tokens that selected expert $e$, and the probability mass $m_e$ is the average probability the router gives expert $e$ under a softmax over all $E$ experts:

$$
f_e = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[e \text{ is selected for } t], \qquad m_e = \frac{1}{T} \sum_{t=1}^{T} \operatorname{softmax}(s_t)_e
$$

The Switch Transformer (Fedus et al., 2021) adds an auxiliary loss proportional to $E \sum_e f_e m_e$. Selection counts have no gradient, but $m_e$ does, so the loss pushes probability away from experts that are already chosen too often. With $k$ experts per token the fractions $f_e$ sum to $k$, while the masses $m_e$ always sum to 1.

## A two-token example

Take three experts, $k = 2$, one-dimensional tokens $x_1 = 2$ and $x_2 = -1.5$, and router weights $[1, 0, -1]$. The scores are $[2, 0, -2]$ for the first token and $[-1.5, 0, 1.5]$ for the second.

The first token selects experts 0 and 1, with weights $\operatorname{softmax}(2, 0) = [0.880797, 0.119203]$. The second selects experts 2 and 1, in that order, with weights $[0.817574, 0.182426]$. If the first token's experts output 10 and 20 and its shared output is 1, its result is $1 + 0.880797 \times 10 + 0.119203 \times 20 = 12.192029$.

Expert 1 was picked by both tokens and the others by one each, so the token fractions are $[0.5, 1, 0.5]$. The full softmaxes are $[0.866813, 0.117310, 0.015876]$ and $[0.039113, 0.175290, 0.785597]$, so the probability masses are their average, $[0.452963, 0.146300, 0.400737]$. Expert 1 is selected most often even though it never has the highest score and holds the least probability mass.

## Common mistakes to avoid

- **Breaking ties by higher index.** Equal scores keep the lower expert index first.
- **Normalizing over all experts.** The combining weights come from a softmax over the $k$ selected scores only.
- **Using the selected softmax for the probability mass.** The mass uses a softmax over all $E$ experts.
- **Forgetting the shared expert.** Its output is added to every token's weighted sum.
- **Reordering the selections.** Selected experts and weights stay in descending score order.
- **Routing in bfloat16.** Scores and softmaxes are computed in float32, since small score differences decide which experts are chosen.
