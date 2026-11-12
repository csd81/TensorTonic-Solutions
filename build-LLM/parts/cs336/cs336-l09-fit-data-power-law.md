# Fit a Data Power Law with a Noise Floor

Scaling laws predict how a model's loss will fall as you give it more data, so a team can decide how much data to collect before spending the compute to train on it. The standard form is a power law that levels off at a floor. This problem centres on fitting that curve to a few small runs by trying candidate floors, fitting a straight line in log space for each, and keeping the floor that matches the original losses best.

## Why loss follows a power law

Across many experiments, the loss of a language model falls smoothly as the training data grows, and each tenfold increase in data removes roughly the same fraction of the remaining reducible loss. Kaplan et al. (2020) found that loss falls as a power law in data size, and the Chinchilla study (Hoffmann et al., 2022) added an irreducible term in its joint law over parameters and data. Holding model size fixed gives the form

$$
L(D) = E + A D^{-\alpha}
$$

where $D$ is the data size, $E$ is the loss floor, $A$ is a positive coefficient and $\alpha$ is a positive exponent. The floor captures loss that more data cannot remove, the genuine unpredictability of text plus, for a fixed model size, the limit set by that model's capacity. The term $A D^{-\alpha}$ shrinks toward zero as $D$ grows.

## Why the floor makes fitting hard

Without a floor, a power law is a straight line on a log-log plot, and fitting it is ordinary linear regression. With a floor, the curve bends and flattens toward $E$, so no single straight line fits it.

The usual approach is to try a set of candidate floors and fit each one separately. Let $D_i$ and $L_i$ be the data size and measured loss of the $i$-th small run. Subtracting a candidate $E$ from every loss and taking natural logs gives

$$
\log(L_i - E) = \log A - \alpha \log D_i
$$

which is a straight line in $\log D_i$ with intercept $\log A$ and slope $-\alpha$. Least squares finds the best intercept and slope, and exponentiating the intercept recovers $A$.

A candidate floor is only usable if it lies below every observed loss, because the logarithm of zero or a negative number is undefined. A fit is also discarded unless both $A$ and $\alpha$ come out positive and finite, since a negative exponent would mean loss grows with data.

## Choosing among candidate floors

Each eligible floor gives a different curve. They are compared by root mean squared error on the original loss scale,

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=0}^{n-1} \left( L(D_i) - L_i \right)^2 }
$$

where $n$ is the number of runs and $L(D_i)$ is the fitted curve's loss at $D_i$. The comparison happens on the original scale because the log transform stretches small differences near the floor, and each candidate floor would distort log-space errors differently.

## A curved example

Take data sizes 1, 10 and 100 with losses 5, 2.948683 and 2.3. These come from a floor of 2, coefficient 3 and exponent 0.5. Try the candidate floors 0 and 2.

With floor 2, the shifted losses are 3, 0.948683 and 0.3, which lie on a line in log-log space. The fit recovers $A = 3$ and $\alpha = 0.5$ with an RMSE of essentially zero, and predicts a loss of $2 + 3 / \sqrt{1000} \approx 2.094868$ at a data size of 1000.

With floor 0, the points are curved in log space, and the best straight line gives $A \approx 4.772$ and $\alpha \approx 0.169$ with an RMSE of about 0.220. Its prediction at 1000 is about 1.489, well below the true floor of 2. Floor 2 has the smaller RMSE, so it is the fit that is kept.

## Why the floor matters for extrapolation

A curve without a floor keeps falling forever and promises gains that more data cannot deliver. In practice these fits are made on small, cheap runs and then used to extrapolate one or two orders of magnitude, so a wrong floor can turn a sensible data budget into a costly mistake. Small errors in the fitted exponent also grow quickly once the curve is pushed far past the data, which is why the fit is done in float64.

## Common mistakes to avoid

- **Using a floor at or above an observed loss.** The logarithm needs every shifted loss to be positive.
- **Selecting the fit by log-space error.** Fits are compared by RMSE on the original losses.
- **Forgetting the sign of the slope.** The exponent is the negative of the fitted slope.
- **Keeping fits with nonpositive coefficient or exponent.** Those curves are not valid scaling laws and are skipped.
- **Fitting without a floor.** A plain log-log line keeps falling forever and predicts losses below what any amount of data can reach.
