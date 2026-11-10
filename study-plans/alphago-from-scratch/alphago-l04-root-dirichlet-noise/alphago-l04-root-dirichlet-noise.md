# Understanding Root Dirichlet Noise Mixing

During self-play, repeatedly following the network's strongest prior can make games too predictable. AlphaGo Zero encourages broader exploration at the root by mixing the policy priors with a supplied Dirichlet noise sample. This problem performs that mixture while keeping all probability on legal actions.

## The three aligned inputs

The prior array contains the network's action weights, the noise array contains one sampled nonnegative weight per action, and the legal mask identifies actions allowed in the current position. All three use the complete action order, including pass at its usual final index.

The supplied prior and noise values are weights, not necessarily normalized probabilities. Their legal entries may sum to any positive amount. Before mixing, each source must be normalized independently over the legal-action set.

Illegal entries are excluded from both normalizations. They must remain exact zeros in the final result regardless of how large their supplied prior or noise weights are.

## Legal-only prior normalization

Let $\mathcal{L}$ be the legal-action set. For a legal action $a$, the normalized network prior is:

$$
\widehat{P}_a=\frac{P_a}{\sum_{b\in\mathcal{L}}P_b}
$$

For an illegal action, $\widehat{P}_a$ is zero. The constraints guarantee positive prior mass among legal actions, so the legal denominator is nonzero.

This step is still necessary when the complete prior array already appears normalized. Some of its mass may sit on illegal actions. Renormalizing only the legal entries ensures that the legal prior distribution sums to one.

## Legal-only noise normalization

The noise weights are normalized separately:

$$
\widehat{\eta}_a=\frac{\eta_a}{\sum_{b\in\mathcal{L}}\eta_b}
$$

Again, illegal actions receive zero. The legal noise mass is guaranteed positive.

Independent normalization is essential. Combining raw prior and noise weights before normalizing would make their relative total magnitudes influence the result. The requested noise fraction would no longer describe the actual fraction of noise in the mixture.

For example, if the raw priors sum to ten and the raw noise sums to one thousand, mixing the raw arrays would let noise dominate even when the requested noise fraction is small. Normalizing each source first puts them on the same probability scale.

## The convex mixture

For each legal action, the final prior is:

$$
P'_a=(1-\varepsilon)\widehat{P}_a+\varepsilon\widehat{\eta}_a
$$

The noise fraction $\varepsilon$ lies between zero and one. The two coefficients therefore also lie between zero and one and add to one. This is a convex combination of two legal probability distributions.

The endpoints make the meaning clear:

- When $\varepsilon=0$, the result is the normalized network prior.
- When $\varepsilon=1$, the result is the normalized noise distribution.
- Between those values, the result retains both sources in the stated proportions.

Because each source sums to one over legal actions, their mixture also sums to one:

$$
\sum_{a\in\mathcal{L}}P'_a
=(1-\varepsilon)+\varepsilon
=1
$$

This property is not an extra cleanup step. It follows directly from independent normalization and the mixture coefficients.

## Why this happens only at the root

The operation in this problem is specifically root prior mixing. The root represents the current self-play decision, where broader action exploration creates varied games and training examples. The function does not add noise to every node in the tree and does not sample a move itself.

The noise sample is already supplied, so this function also does not generate random Dirichlet values. Its responsibility is deterministic: normalize the two legal weight sets and combine them using the given fraction.

Keeping sampling separate makes the result reproducible for a fixed input and lets tests focus on the mixture rules.

## Alignment and exact illegal zeros

The result has the same length and action order as the inputs. A clear conceptual approach is to create complete zero arrays, place normalized values only at legal indices, mix those complete arrays, and finally ensure illegal entries are zero.

Exact zeros matter. Multiplying an illegal raw value by a coefficient or assigning it a tiny approximation would leave probability outside the legal set. Search expects an illegal action to have no prior at all.

If only one action is legal, both independently normalized distributions assign it probability one. Their mixture is also one at that action, regardless of the raw weights or noise fraction. Every other entry stays zero.

## Numeric type and input safety

The output uses 64-bit floating point so normalization and mixing retain fractional precision. Converting the supplied weights to new NumPy arrays of the requested type allows arithmetic without editing the caller's inputs.

The prior array, noise array, and legal mask must remain unchanged. The function returns a new mixed array. This is important because the original network priors may still be needed for inspection or for another search computation.

## Useful invariants

A correct result satisfies several conditions:

- Every entry is nonnegative.
- Every illegal action is exactly zero.
- Legal entries sum to one within floating-point precision.
- A zero noise fraction reproduces normalized legal priors.
- A full noise fraction reproduces normalized legal noise.
- With one legal action, that action receives probability one.

## Common mistakes to avoid

- Normalizing across illegal actions wastes probability mass outside the game rules.
- Mixing before independent normalization changes the meaning of the noise fraction.
- Generating a new random sample ignores the supplied noise weights.
- Adding noise at every search node expands the scope beyond root prior mixing.
- Mutating the prior array destroys the original network output.
- Returning only legal entries loses alignment with the complete action space.
- Leaving illegal values as tiny nonzero numbers violates the exact-zero requirement.

The operation is a controlled probability mixture: normalize network priors and noise separately over legal actions, blend them with the requested fraction, and preserve the complete action layout with exact zeros everywhere illegal.

---