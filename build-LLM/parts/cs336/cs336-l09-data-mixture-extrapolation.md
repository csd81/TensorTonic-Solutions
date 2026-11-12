# Extrapolate Data-Mixture Scaling Curves

A language model's training data is a mixture of sources, such as web pages, code, books and papers, and the proportions change how well the model learns. Testing every mixture at full scale is far too expensive, so teams run each candidate at small scale and extrapolate. This problem centres on fitting a scaling curve to each mixture and choosing the one whose curve predicts the lowest loss at the target size, even when it is not the best at any size that was actually tested.

## Why mixtures need scaling curves

Two mixtures can trade places as the data grows. A mixture heavy in easy, repetitive text may reach a low loss quickly and then flatten out, while a mixture of harder, more varied text starts slower but keeps improving for longer. Comparing them at the small sizes you can afford to test may pick the wrong one for the size you actually plan to train at.

Fitting a curve to each mixture captures both how fast its loss falls and where it levels off, so the comparison can be made at the target size instead.

## One curve per mixture

Each mixture $k$ gets its own scaling law:

$$
L_k(D) = E_k + A_k D^{-\alpha_k}
$$

Here $D$ is the data size, $E_k$ is the mixture's loss floor, $A_k$ is a positive coefficient and $\alpha_k$ is a positive exponent. The floor is where the mixture's loss would settle with unlimited data, and the exponent sets how quickly it gets there. A lower floor wins in the long run, while a larger exponent closes the gap faster.

Each curve is fitted with the same floor search used for a single power law, run on that mixture's own row of observations. For each candidate floor below all of the mixture's losses, the losses minus the floor are fitted with a straight line in log space,

$$
\log(L_{k,i} - E_k) = \log A_k - \alpha_k \log D_{k,i}
$$

where $D_{k,i}$ and $L_{k,i}$ are the size and loss of the mixture's $i$-th run. The candidate whose curve has the lowest error on the original losses is kept. Because every mixture gets its own floor, a mixture that the model can learn more completely can level off at a lower loss than one that stays harder to predict.

## Comparing at the target size

With every curve fitted, each mixture's predicted loss at the target size comes straight from its formula, and the mixture with the lowest prediction wins.

## A crossover example

Take two mixtures, each observed at data sizes 1, 10 and 100, with candidate floors 0.25 and 1.5. The target size is 10,000.

The web mixture has losses 3.5, 2.502374 and 2.002377. The floor of 1.5 fits them almost exactly, with coefficient 2 and exponent 0.3. The math mixture has losses 20.25, 6.574555 and 2.25. Its floor of 0.25 fits almost exactly, with coefficient 20 and exponent 0.5, while the floor of 1.5 leaves a curved residual and a larger error.

The web mixture has the lower loss at every tested size, including 100, where it leads by about 0.25. At the target size the predictions are

$$
L_{\text{web}}(10{,}000) = 1.5 + 2 \times 10{,}000^{-0.3} \approx 1.626191, \qquad L_{\text{math}}(10{,}000) = 0.25 + 20 \times 10{,}000^{-0.5} = 0.45
$$

so the math mixture wins clearly. Its larger exponent and lower floor more than make up for its poor start. Judging by every observed size would have picked the web mixture.

## Limits of extrapolation

These curves are trustworthy only near the range they were fitted on, and a mixture's floor and exponent are estimated from a handful of small runs. Extrapolating two orders of magnitude, as in the example, amplifies any error in the fitted exponent. Real mixture studies fit at several scales, check that the ranking is stable, and often confirm the winner with one larger run before committing the full budget.

## Common mistakes to avoid

- **Sharing one fit across mixtures.** Each mixture gets its own floor, coefficient and exponent from its own row of data.
- **Comparing at the largest observed size.** A mixture that leads there can still lose at the target if its floor is higher or its exponent smaller.
- **Comparing fits by log-space error.** The best floor for a mixture is chosen by error on the original losses.
- **Ignoring the floor when ranking.** Two mixtures with similar exponents can still end far apart if their floors differ.
- **Trusting a fit far past its data.** An extrapolation many orders of magnitude beyond the fitted sizes can reverse the ranking, so real studies confirm with a larger run.
