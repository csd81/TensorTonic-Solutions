# Measure Pipeline Bubble Utilization

Pipeline parallelism splits a model's layers into consecutive stages on different GPUs, so a batch flows through them like an assembly line. Each stage must wait for the one before it, which leaves GPUs idle at the start and end of every step. This problem centres on laying out the schedule used by GPipe, one of the first pipeline systems, one time step at a time, and measuring how much of it is wasted in those idle gaps, called bubbles.

## Why split a model into stages

When a model is too large for one GPU, one option is to place the first few layers on GPU 0, the next few on GPU 1, and so on. Each GPU then stores only its own layers, and the only data passed between GPUs are the activations at the stage boundaries and their gradients, which is far less than the weights. This makes pipeline parallelism attractive across nodes, where links are slow.

The difficulty is dependency. Stage 1 cannot start until stage 0 has produced its output, so with a single batch only one GPU works at a time.

## Microbatches fill the pipeline

GPipe (Huang et al., 2019) splits each batch into $m$ microbatches. As soon as stage 0 finishes microbatch 0 and passes it on, it starts microbatch 1, so several stages work at once on different microbatches.

Time is divided into equal slots, with every stage operation taking one slot. A forward wave is the stretch of time in which every microbatch passes through every stage once. With $p$ stages it lasts

$$
T = m + p - 1
$$

slots, because the last stage starts at slot $p - 1$ and needs $m$ more slots to finish every microbatch. Microbatch $i$ enters stage 0 at slot $i$ and moves one stage per slot, so it reaches stage $s$ at slot $i + s$. A stage with no microbatch arriving in a slot sits idle.

## The backward wave runs in reverse

Gradients flow from the last layer to the first, so the backward wave starts at the last stage. In GPipe the backward wave begins only after the whole forward wave has finished. The last stage begins with the last microbatch, because that is the one whose forward pass completed most recently, and counts down to microbatch 0.

Stage $s$ sits $p - 1 - s$ stages before the last one, so its backward work begins $p - 1 - s$ slots into the backward wave. From there it counts down through the microbatches starting at $m - 1$, so at offset $u$ it works on microbatch

$$
i_{\text{backward}} = m - 1 - \left( u - (p - 1 - s) \right)
$$

and it is idle whenever that index falls outside 0 to $m - 1$. The backward wave also lasts $T$ slots.

## Counting the bubble

Each cell of the schedule is one stage during one slot. A forward-only step has one wave and a step with backward has two. Each wave has $Tp$ cells, and exactly $pm$ of them hold real work, because every stage processes every microbatch once. Utilization is therefore

$$
\frac{pm}{Tp} = \frac{m}{m + p - 1}
$$

and the bubble fraction is $(p - 1)/(m + p - 1)$, roughly $(p - 1)/m$ when there are many more microbatches than stages. The bubble shrinks as the number of microbatches grows relative to the number of stages, which is why pipeline training uses many microbatches. The cost is that each stage must keep the activations of all $m$ in-flight microbatches until the backward wave starts, which adds up to the activations of the whole batch. GPipe reduces this by recomputing activations during the backward pass.

## A three-stage example

Take $p = 3$ stages and $m = 2$ microbatches, forward only. In the table, F0 means the forward pass of microbatch 0. The wave lasts $T = 4$ slots:

| Slot | Stage 0 | Stage 1 | Stage 2 |
|---|---|---|---|
| 0 | F0 | idle | idle |
| 1 | F1 | F0 | idle |
| 2 | idle | F1 | F0 |
| 3 | idle | idle | F1 |

Six of the twelve cells do work, so utilization is $6 / 12 = 0.5$ and half the GPU time is bubble. With $m = 8$ microbatches on the same three stages, the wave lasts 10 slots and utilization rises to $8 / 10 = 0.8$.

## Better schedules

The schedule in this problem keeps the forward and backward waves strictly separate, which makes the bubble easy to see. Later schedules such as 1F1B, introduced with PipeDream and used in synchronous form by Megatron-LM, interleave one forward and one backward step per stage. They have the same bubble size but hold far fewer activations at once, and interleaved variants that give each GPU several smaller stages shrink the bubble further.

## Common mistakes to avoid

- **Running the backward wave forward.** The backward wave starts at the last stage with the last microbatch.
- **Overlapping the two waves.** In this schedule the backward wave begins only after the forward wave has finished.
- **Measuring the bubble in time.** A slot with one busy stage out of three is still two-thirds idle, so waste is counted per stage-slot cell.
- **Forgetting the ramp-down.** The forward wave lasts $m + p - 1$ slots, including the slots where only later stages are busy.
- **Assuming more stages always helps.** With $m$ fixed, each extra stage adds another idle slot per wave, so utilization drops.
