# Low-Confidence Remasking

Low-confidence remasking is the sampler behind every LLaDA generation result in the paper's main text. It runs the reverse process of a masked diffusion model in a fixed number of steps, and at each step it unmasks the predictions the mask predictor is most sure about while every other position stays masked. This problem covers the deterministic transfer schedule and the confidence ranking that together decide which tokens appear at each step.

## Revealing tokens instead of drawing them at random

LLaDA's reverse process starts from a response of $L$ mask tokens and moves from time $t = 1$ to $t = 0$ in $N$ steps. At each step the mask predictor sees the prompt and the current partially masked response and predicts every masked token at once. The theoretically faithful sampler, Algorithm 4, then sends each new prediction back to the mask independently with probability $s/t$, where $s = t - 1/N$ is the next time. That keeps the reverse process aligned with the forward masking process, but which tokens survive is pure chance.

The paper notes that "in principle, the remasking strategy should be purely random" and then replaces chance with confidence, citing the annealing tricks used when sampling from autoregressive models and the approach of Chang et al. in MaskGIT. Here the confidence of a masked position is the softmax probability of its greedy prediction. Revealing the most confident positions first lets easy tokens anchor the response, and the harder positions are predicted later with more context. Table 9 of the paper shows how large the effect is. With the same model and 256 steps, GSM8K rises from 21.3 with random remasking to 70.0 with low-confidence remasking, and the paper attributes this to reduced diversity, much like annealed sampling.

## A schedule fixed before sampling

Random remasking reveals a random number of tokens per step. Low-confidence remasking needs an exact count $c_k$ for each step $k$, and the paper and the official code compute it differently. Algorithm 5 sets $n_{un} = \lfloor L(1-s) \rfloor$, calls it the number of unmasked tokens at time $s$, and in the next line says the $n_{un}$ least confident positions are remasked. Those two statements cannot both hold, because remasking $n_{un}$ positions would leave $L - n_{un}$ unmasked.

The official sampler avoids the question. Its schedule divides the number of masked tokens $n_0$ by $N$ and gives the remainder to the earliest steps:

$$
c_k = \left\lfloor \frac{n_0}{N} \right\rfloor + \mathbb{1}\left[k < (n_0 \bmod N)\right]
$$

At each step it reveals exactly $c_k$ of the highest-confidence masked positions. The floor rule read as a running total gives $2, 3$ for $L = 5$ and $N = 2$, while the code gives $3, 2$. This exercise follows the code's even split, and the paper's floor schedule is one of the mistakes listed below.

## What stays fixed in this exercise

A few conventions remove randomness that is not part of the mechanism.

- **Greedy predictions.** The official sampler can perturb logits with Gumbel noise at a temperature. Algorithm 5 is greedy, and so is this exercise, which is the code's behavior at temperature zero.
- **Scripted predictor.** The step logits stand for the mask predictor's output on the state before each step, so the answer depends only on the stored logits.
- **No end-of-sequence heuristic.** Appendix B.4 lowers the confidence of the end-of-sequence token on some benchmarks. That heuristic is not applied here.
- **One block.** The whole response is a single candidate window. The semi-autoregressive sampler splits it into blocks, which a later problem covers.

## Two steps on four positions

Take $L = 4$ and $N = 3$, so $n_0 = 4$ and the schedule is $c = (2, 1, 1)$. Suppose the confidences at step 0 are $(0.62, 0.91, 0.40, 0.77)$. The two largest belong to positions 1 and 3, which are revealed at step 0.

At step 1 the new logits give confidences $(0.55, 0.99, 0.81, 0.97)$. Positions 1 and 3 are already revealed, so their confidences become $-\infty$ even though they are the largest values. Among positions 0 and 2, position 2 wins with $0.81$ and is revealed. At step 2 only position 0 remains, and it is revealed with whatever the last logits predict. The reveal steps are $(2, 0, 1, 0)$.

## Common mistakes to avoid

- **Revealing the least confident positions.** Algorithm 5 speaks of remasking the lowest confidences, which means keeping the highest ones. Selecting the lowest reverses the sampler.
- **Giving the remainder to the last steps.** The code front-loads it, so for five tokens in two steps the schedule is $3, 2$ and never $2, 3$.
- **Using the paper's floor count.** Differencing $\lfloor L(1-s) \rfloor$ across steps produces a back-loaded schedule that disagrees with the official sampler whenever $N$ does not divide $L$.
- **Ranking by the raw logit.** A large maximum logit can still have a small probability when a second logit is close to it, so the softmax is required.
- **Letting revealed positions back in.** A revealed token is final. If its confidence is not excluded, it can win the ranking, waste part of the step's count, or be overwritten by a newer prediction.
- **Indexing response logits from the start of the sequence.** The logits cover response positions only, so position $i$ of the logits belongs to sequence position $P + i$.