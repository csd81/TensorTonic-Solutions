# Fit Elo Ratings from Pairwise Votes

Leaderboards such as Chatbot Arena rank language models from human votes, where each vote says which of two anonymous responses was better. Elo ratings turn those many pairwise outcomes into one number per model. The heart of this problem is fitting the ratings by gradient ascent on the votes, ranking the models, and checking which models were ever compared at all.

## Ratings that predict wins

The Elo system, first used for chess, gives each model a rating. In comparison $n$, call model $a_n$ side A and model $b_n$ side B. The predicted probability that A wins depends only on the rating difference:

$$
p_n = \frac{1}{1 + 10^{(r_{b_n} - r_{a_n})/400}}
$$

Equal ratings give a probability of 0.5. A 400-point lead gives about 0.91, so the leader is expected to win 10 times as often as it loses. The 400 is a scale chosen so ratings look like chess ratings, and only differences matter.

This formula equals $\sigma\left(\frac{\ln 10}{400}(r_{a_n} - r_{b_n})\right)$, where $\sigma(z) = 1 / (1 + e^{-z})$ is the logistic sigmoid. Fitting ratings to all votes at once is therefore logistic regression on rating differences, the model statisticians call Bradley-Terry.

## Why fit all votes at once

Chess Elo updates a player's rating after every game, so the final ratings depend on the order in which games were played. For a leaderboard that collects votes in no meaningful order, that order dependence is noise. Fitting the ratings to the whole set of votes together removes it, which is why Chatbot Arena moved from online Elo updates to this kind of batch fit.

## Gradient ascent on the log likelihood

The outcome $y_n$ is 1 if A won, 0 if B won and 0.5 for a tie. The ratings are chosen to maximize the average log likelihood of the observed outcomes. Because $p_n$ is a sigmoid of $\frac{\ln 10}{400}$ times the rating difference, the derivative of that average with respect to model $i$'s rating over $M$ comparisons is

$$
g_i = \frac{\ln 10}{400 M} \sum_n (y_n - p_n) \left( \mathbf{1}[a_n = i] - \mathbf{1}[b_n = i] \right)
$$

where the indicator $\mathbf{1}$ is 1 when its condition holds. The residual $y_n - p_n$ says how much better A did than predicted, so a model that keeps outperforming its predictions gets a positive gradient. Starting from zero, all ratings are updated together with $r_i \leftarrow r_i + \eta g_i$, where $\eta$ is the learning rate and every $g_i$ uses the ratings from before the step.

Shifting every rating by the same amount changes no prediction, so the ratings have no natural zero. After each step the mean rating is subtracted, which pins the average at 0.

## Which models can be compared

Ratings are only comparable between models that are linked by comparisons. If models 0 and 1 were only compared with each other, and models 2 and 3 only with each other, nothing tells you whether 0 is better than 2.

Treating each comparison as an edge between two models, a connected component is a group of models where you can walk from any model to any other through a chain of comparisons. Each component can be ranked internally, and a model that never appears in a comparison forms a component of its own. A leaderboard is only fully meaningful when there is a single component.

## One step on a split graph

Take four models and three votes, where model 0 beats model 1 twice and model 2 beats model 1 once. Model 3 never plays. Use one step with learning rate 400.

All ratings start at 0, so every prediction is 0.5 and every residual is 0.5. Model 0 is side A twice with residual $+0.5$ each, for a sum of 1. Model 1 is side B three times, for $-1.5$. Model 2 is side A once, for $+0.5$, and model 3 has no comparisons. The step changes the ratings by

$$
400 \times \frac{\ln 10}{400 \times 3} \times [1, -1.5, 0.5, 0] \approx [0.767528, -1.151293, 0.383764, 0]
$$

which already has mean 0. Ranking by rating gives $[0, 2, 3, 1]$. Model 3 sits third only because its rating stayed at 0, and the component labels $[0, 0, 0, 1]$ show that the graph is not connected, a warning that model 3's place in the ranking means nothing.

## Common mistakes to avoid

- **Updating ratings one comparison at a time.** Every step uses the full batch of comparisons and updates all ratings together.
- **Using the natural sigmoid without the $\ln 10 / 400$ factor.** Probabilities and step sizes then come out on the wrong scale.
- **Skipping the centering.** Subtracting the mean after each step keeps the ratings pinned at an average of 0.
- **Forgetting the $1/M$ average.** The gradient averages over comparisons, so more votes do not mean bigger steps.
- **Trusting ranks across components.** Models in different components were never compared, so their relative order carries no information.
