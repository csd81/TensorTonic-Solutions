# Fit Learning-Rate and Batch-Size Scaling

The best learning rate and batch size change as models grow, and searching for them directly at full scale is far too expensive. Teams instead sweep both hyperparameters on several small models, find the best pair at each size, and fit power laws that extrapolate to the target model. The heart of this problem is picking each scale's best pair from a grid of measured losses and fitting a separate log-log line for each hyperparameter.

## Why hyperparameters need scaling laws

A learning rate that works well for a 100 million parameter model is usually too large for a 10 billion parameter one, which can diverge or train poorly with it. The best batch size tends to grow with scale, because as the loss falls the useful signal in each gradient shrinks relative to its noise, so averaging over more examples per step keeps paying off. Guessing these values for a large run risks wasting the whole compute budget.

Small sweeps are cheap. If the best values follow a smooth trend across small scales, a power law fitted to them predicts good values at the target scale. Scale is usually parameter count or training compute, and any positive measure works because only ratios matter in log space. DeepSeek LLM (2024) chose its learning rate and batch size in a similar way, fitting power laws of the best values against compute budget from small-scale sweeps.

## Finding the best pair at each scale

For each model scale there is a grid of measured losses, one for every combination of learning rate and batch size. The best pair is the cell with the lowest loss, and its two values at scale $S_i$ are $\eta_i^*$ and $B_i^*$. On a flat loss surface several cells can share the lowest loss, so a fixed rule for picking one keeps the fit reproducible.

## Two independent power laws

Each hyperparameter is fitted against scale with its own power law, $\eta^* = a_\eta S^{b_\eta}$ and $B^* = a_B S^{b_B}$, where $a_\eta$ and $a_B$ are positive coefficients and $b_\eta$ and $b_B$ are exponents. Taking natural logs turns each into a straight line:

$$
\log \eta_i^* = \log a_\eta + b_\eta \log S_i
$$

$$
\log B_i^* = \log a_B + b_B \log S_i
$$

Ordinary least squares on each line gives an intercept equal to $\log a$ and a slope equal to the exponent $b$. A negative exponent means the hyperparameter falls as models grow, which is typical for the learning rate, and a positive one means it rises, which is typical for the batch size. The fits are independent, so the learning-rate trend does not influence the batch-size trend.

Plugging the target scale into a fitted law, for example $\hat{\eta} = a_\eta S_{\text{target}}^{b_\eta}$, turns a handful of cheap small runs into a starting point for the expensive one.

## What residuals tell you

A residual is the observed log value minus the fitted log value at one scale. Small residuals mean the optima really do follow a power law. A large residual at one scale often means the grid was too coarse there, so the true optimum fell between grid values, or that the loss surface was flat and noise decided the winner. Because the fit is extrapolated far past the measured scales, one bad optimum can move the prediction a lot.

## A three-scale example

Take model scales 1, 4 and 16, learning rates 0.004, 0.002 and 0.001, and batch sizes 32, 64 and 128. Suppose the lowest losses fall at $(0.004, 32)$ for scale 1, $(0.002, 64)$ for scale 4 and $(0.001, 64)$ for scale 16.

The learning rate halves each time the scale grows fourfold, so its fit is exact, with exponent

$$
b_\eta = \frac{\log(1/2)}{\log 4} = -0.5
$$

and coefficient 0.004. At a target scale of 64 it predicts $0.004 \times 64^{-0.5} = 0.0005$.

The batch sizes 32, 64 and 64 do not lie on one line in log space. The log scales are $0$, $\log 4$ and $2 \log 4$, centered on $\log 4$, so the least-squares slope only compares the two outer points, giving $b_B = (\log 64 - \log 32) / (2 \log 4) = 0.25$. The coefficient comes out at about 35.92, and the residuals are about $-0.116$, $0.231$ and $-0.116$. The middle scale sits above the line and the outer scales below it, a sign that the batch-size grid is too coarse to pin the trend down. The predicted batch size at scale 64 is about 101.6, which in practice would be rounded to a hardware-friendly value.

## Common mistakes to avoid

- **Fitting on raw values.** Both lines are fitted between natural logs of scale and hyperparameter.
- **Fitting the two hyperparameters jointly.** Each gets its own independent line.
- **Computing residuals on the original scale.** Residuals are differences of log values.
- **Forgetting to exponentiate the intercept.** The intercept is the log of the coefficient.
- **Mixing log bases.** Using base 10 for some values and natural logs for others makes the coefficient come out wrong.
- **Trusting a fit with large residuals.** A poorly fitted trend can extrapolate to a badly wrong value at the target scale.
