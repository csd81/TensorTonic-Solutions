# Remove Response-Length Bias from Judge Scores

When a language model or a person judges which of two responses is better, longer responses tend to win even when they are not more helpful. That bias lets a model climb a leaderboard just by being verbose. The heart of this problem is fitting a model that explains each vote with both model quality and length difference, then reporting win probabilities with the length effect removed.

## Why judges prefer long answers

Longer responses look more thorough, include more caveats and cover more cases, so judges, human or automated, often reward them. AlpacaEval, a benchmark where a strong language model judges responses against a reference, found that its judge preferred longer outputs strongly enough to change rankings. Length-controlled AlpacaEval (Dubois et al., 2024) was introduced to correct for it, and the correction in this problem follows the same idea.

## A model with two causes for a win

In comparison $n$, call model $a_n$'s response A and model $b_n$'s response B, and let $\Delta_n$ be A's length minus B's length, in tokens or whatever unit the data uses. Each model has a utility $u_i$ that measures its true quality, and a single coefficient $\beta$ measures how much the judge rewards extra length. The predicted probability that A wins is

$$
p_n = \sigma(u_{a_n} - u_{b_n} + \beta \Delta_n)
$$

where $\sigma(z) = 1 / (1 + e^{-z})$ is the logistic sigmoid. A positive $\beta$ means longer responses get a boost.

The outcome $y_n$ is 1 if A won, 0 if B won and 0.5 for a tie. The parameters are fitted by minimizing the mean binary cross-entropy over the $M$ comparisons:

$$
L = -\frac{1}{M} \sum_n \left[ y_n \log p_n + (1 - y_n) \log (1 - p_n) \right]
$$

Because $L$ is the cross-entropy of a sigmoid, its gradient with respect to comparison $n$'s sigmoid input is $(p_n - y_n)/M$. The chain rule multiplies that by $\Delta_n$ for $\beta$, by $+1$ for model $a_n$'s utility and by $-1$ for model $b_n$'s. Adding a constant to every utility changes no prediction, which is why the utilities are re-centered to mean 0 after every gradient step.

## Asking what would happen at equal length

Once fitted, the length term has absorbed whatever the judge rewarded for verbosity. Setting $\Delta_n = 0$ gives the probability that A would have won if both responses had the same length:

$$
p_n^{(0)} = \sigma(u_{a_n} - u_{b_n})
$$

A model's debiased win rate averages $p_n^{(0)}$ over the comparisons where it was A and $1 - p_n^{(0)}$ over those where it was B.

## Length can hide quality, too

Take two models and five comparisons, fitted with learning rate 0.1. Model 0 is A in the first two, with length differences 3 and $-3$, and wins both. Model 1 is A in the next two, with length differences 3 and $-3$, and wins only when its response is longer. In the fifth, model 0 is A at equal length and the vote is a tie.

The first step starts with every $p_n = 0.5$, so the residuals $y_n - p_n$ are $0.5$, $0.5$, $0.5$, $-0.5$ and $0$. The length gradient averages residual times length difference, $(1.5 - 1.5 + 1.5 + 1.5 + 0) / 5 = 0.6$, so $\beta$ moves to 0.06. Model 0 collects $+1$ from the comparisons where it was A and $0$ from those where it was B, so its utility gradient is $1/5 = 0.2$, and model 1's is $-0.2$. After centering, the utilities are 0.02 and $-0.02$.

After 200 steps, $\beta$ is about 0.482 and model 0's equal-length win probability against model 1 is about 0.764, close to their converged values. Fitting the same 200 steps with every length difference set to 0 gives only about 0.700.

The length model noticed that model 1's only win came when its answer was longer, so it credits that win to length and gives model 1 less credit for quality. Model 0 won even when it was shorter, which length cannot explain, so it looks stronger once length is accounted for. Debiasing can raise a model as well as lower it, because each vote is credited to whichever cause the data supports.

## Limits of the correction

A single linear length term cannot tell a long, useful answer from a long, padded one, and its estimate becomes unreliable when the models with the longest answers are also the best ones, because the two causes are then hard to separate. Balanced comparisons, where each model sometimes writes the longer response and sometimes the shorter, make the coefficient much easier to estimate.

## Common mistakes to avoid

- **Reporting probabilities with the length term.** Debiased probabilities set $\Delta_n = 0$ after fitting.
- **Fitting without length and subtracting it afterward.** The length coefficient must be learned jointly with the utilities.
- **Forgetting the $\Delta_n$ factor in the $\beta$ gradient.** Without it, $\beta$ moves on every miss whether or not length was involved.
- **Dropping the $1/M$ from the gradient.** Steps then grow with the number of comparisons.
- **Using raw lengths instead of the difference.** The model needs A's length minus B's, and swapping the sides flips the sign.
