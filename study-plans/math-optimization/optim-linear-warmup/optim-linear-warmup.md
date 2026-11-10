# <span style="font-size: 20px;">Linear Warmup</span>

## Motivation

Starting training with a large learning rate can cause instability: the initial random weights produce large, noisy gradients that push parameters in poor directions. If the first few updates are too aggressive, the model can diverge or settle into a bad region of the loss landscape.

Linear warmup addresses this by starting with a small learning rate and gradually increasing it to the target value over a fixed number of "warmup" steps or epochs.

## The Schedule

Given a base learning rate $\alpha$ and a warmup period of $W$ epochs, the learning rate at epoch $e$ (0-indexed) is:

$$
\text{lr}(e) = \begin{cases} \alpha \cdot \frac{e + 1}{W} & \text{if } e < W \\ \alpha & \text{if } e \geq W \end{cases}
$$

At epoch 0 the LR is $\alpha / W$, at epoch $W-1$ it reaches $\alpha$, and it stays constant thereafter.

## Why Warmup Helps

- **Stable early updates**: small LR prevents large, poorly-directed updates when the model knows nothing about the data
- **Adaptive optimizer interaction**: Adam and similar optimizers need several steps to build up accurate moment estimates. Large LR during this calibration phase amplifies estimation errors
- **Batch norm and residual networks**: these architectures are especially sensitive to early gradient magnitudes; warmup provides time for internal statistics to stabilize

## When to Use Warmup

Warmup is most important for:
- Large learning rates (where initial instability risk is highest)
- Transformers and large models (standard practice: warmup over 1-10% of total steps)
- Transfer learning / fine-tuning (prevent destroying pretrained features)

For small models or conservative learning rates, warmup may have little effect since the initial updates are already small enough to be stable.

## Warmup + Decay Combinations

In practice, warmup is almost always combined with a subsequent decay schedule:
- **Warmup + constant**: simplest; used when training duration is short
- **Warmup + linear decay**: LR rises then falls linearly to zero
- **Warmup + cosine decay**: standard for transformer training
