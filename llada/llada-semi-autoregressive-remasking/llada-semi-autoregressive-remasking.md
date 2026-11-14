# Semi-Autoregressive Remasking

Semi-autoregressive remasking, which the paper calls block diffusion LLaDA, is a sampler that sits between pure diffusion sampling and autoregressive decoding. It keeps the response length fixed and masked from the start, but it only lets one block of positions be revealed at a time, moving from left to right. This problem covers how the response is partitioned into blocks, how the step budget is shared among them, and how the candidate window restricts the low-confidence step that the previous problem already taught.

## Between pure diffusion and autoregression

LLaDA's reverse process starts from a prompt followed by $L$ mask tokens and reveals them over $N$ steps. In pure diffusion sampling the whole response is one window, so any masked position can be revealed at any step, and the order of the answer emerges from the model's confidence. Autoregressive sampling is the other extreme, where the next token is always the leftmost one and the sequence grows until an end-of-sequence token appears.

Block sampling with block length $b$ interpolates between the two. The response is cut into $K = L/b$ blocks, and the sampler finishes block $j$ before it touches block $j+1$, while inside each block it is free to choose any order. With $b = L$ it is pure diffusion, and with $b = 1$ it reveals one position per block from left to right, which is an autoregressive order over a fixed-length canvas.

The paper finds that pure diffusion is the best default for LLaDA 8B Base. Table 7 shows it at 49.7 on BBH against 48.0 to 48.6 for block diffusion LLaDA. For LLaDA 8B Instruct, however, Table 8 shows block diffusion LLaDA ahead on GSM8K, 77.5 against 69.4, and a block length of 8 raises it to 78.6. The paper attributes this to the heavy end-of-sequence padding in the SFT data, which causes early termination under low-confidence remasking. Restricting candidates to the leftmost unfinished block keeps the sampler from filling the end of the response before its content.

## Fixed-length blocks versus growing blocks

The original block diffusion of Arriola et al. grows the sequence block by block and stops at an end token, so the length is dynamic. LLaDA's variant, shown in Figure 5(c) of the paper, keeps every block in place from the start. Later blocks stay masked but remain visible to the bidirectional mask predictor as mask tokens, so the model always knows how much room is left. No retraining is needed, since the predictor still sees a partially masked sequence.

## Splitting the step budget

The total budget $N$ is divided evenly, so each block gets $N_b = N/K$ steps. Block $j$ then builds its own schedule from its masked count $b$, following the official sampler's even split with the remainder given to the earliest steps:

$$
c_i^{(j)} = \left\lfloor \frac{b}{N_b} \right\rfloor + \mathbb{1}\left[i < (b \bmod N_b)\right]
$$

Local step $i$ of block $j$ is global step $g = jN_b + i$, and the logits for step $g$ stand for the predictor's output on the state before that step. This exercise fixes the official code's conventions. The block length divides $L$, the number of blocks divides $N$, the schedule is the code's even split rather than Algorithm 5's floor rule, predictions are greedy with no temperature, and the end-of-sequence confidence heuristic of Appendix B.4 is not applied. The confidence ranking itself is supplied as a helper, so the work here is choosing the window, the count and the global step.

## Eight tokens in two blocks

Take $L = 8$, $b = 4$ and $N = 6$. There are $K = 2$ blocks and $N_b = 3$ steps per block. Each block's schedule splits 4 tokens over 3 steps, giving $2, 1, 1$, so the stacked schedules are two identical rows. Block 0 uses global steps 0, 1 and 2, and block 1 uses global steps 3, 4 and 5.

A single schedule over the whole response would instead split 8 tokens over 6 steps as $2, 2, 1, 1, 1, 1$. Cutting that into rows of three gives $2, 2, 1$ for block 0, which tries to reveal 5 tokens in a 4-token block, and $1, 1, 1$ for block 1, which leaves a token masked. The per-block schedule is what guarantees that every block is complete before the next one starts.

## Common mistakes to avoid

- **Leaving the window open.** If every response position is a candidate, confident positions in later blocks are revealed early and the sampler becomes pure diffusion.
- **Giving every block the full budget.** Each block owns $N/K$ steps, and a schedule built from $N$ steps reveals too few tokens per step.
- **Splitting one global schedule.** The remainder of $L \bmod N$ falls on different steps than the per-block remainder, so blocks finish too early or too late.
- **Offsetting the window by the prompt.** The candidate window covers response positions only, so adding the prompt length shifts it onto the wrong tokens.
- **Reading logits by local step.** Each global step has its own logits, and the local index replays the first block's predictions in every later block.
- **Recording local steps.** The reveal order must use global step indices so that it increases across blocks.
- **Reimplementing the ranking.** The supplied step already handles softmax confidence, candidate masking and selection, and a second copy invites inconsistencies.