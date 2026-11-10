# <span style="font-size: 20px;">LR Range Test and One-Cycle Policy</span>

## The LR Range Test

Smith (2017) proposed a simple diagnostic: train for a few hundred steps while exponentially increasing the learning rate from a very small value to a very large one. Plot loss vs. LR. The loss initially drops as the LR reaches a useful range, then eventually diverges when the LR is too high. The optimal `max_lr` for training is typically in the region where the loss is still decreasing steeply, before the minimum.

## The One-Cycle Policy

Smith and Topin (2018) showed that a single cycle of LR warmup followed by decay - the "1cycle" policy - can train models in significantly fewer epochs than conventional schedules. The schedule has two phases:

**Phase 1** (first `pct_start` fraction of epochs): cosine interpolation from `initial_lr` up to `max_lr`

$$
\begin{aligned}
\text{lr}(e) &= \text{initial\_lr} \\
&\quad + (\text{max\_lr} - \text{initial\_lr}) \cdot \frac{1 - \cos(\pi \cdot e / E_{\text{up}})}{2}
\end{aligned}
$$

**Phase 2** (remaining epochs): cosine interpolation from `max_lr` down to `min_lr`

$$
\begin{aligned}
\text{lr}(e) &= \text{min\_lr} \\
&\quad + (\text{max\_lr} - \text{min\_lr}) \cdot \frac{1 + \cos\!\left(\frac{\pi (e - E_{\text{up}})}{E_{\text{down}}}\right)}{2}
\end{aligned}
$$

where $E_{\text{up}}$ is the number of warmup epochs and $E_{\text{down}} = T - E_{\text{up}}$.

## Derived Learning Rates

The three LR levels are derived from two parameters:

$$
\begin{aligned}
\text{initial\_lr} &= \frac{\text{max\_lr}}{\text{div\_factor}} \\[6pt]
\text{min\_lr} &= \frac{\text{initial\_lr}}{\text{final\_div}}
\end{aligned}
$$

Typical values: `div_factor=25` gives `initial_lr = max_lr / 25`, and `final_div=1e4` gives a `min_lr` four orders of magnitude below `initial_lr`.

## Why One-Cycle Works

- **Super-convergence**: the large LR in the middle of training acts as regularization, preventing the model from settling in sharp minima
- **Fast training**: because the LR reaches high values quickly, the model makes large progress early, then fine-tunes with the decay
- **Implicit regularization**: high learning rates have been shown to favor flatter minima, which generalize better
