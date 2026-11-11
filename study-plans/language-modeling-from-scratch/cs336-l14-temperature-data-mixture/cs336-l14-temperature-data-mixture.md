# Compute Temperature-Based Data Mixtures

A training corpus combines sources of very different sizes, such as a huge web crawl, a smaller code collection and a tiny set of math problems. Sampling each source in proportion to its size would let the largest one drown out the rest. Temperature-based mixing raises each size to a power before normalizing, which flattens the mixture toward smaller sources. The heart of this problem is computing those sampling probabilities stably and translating them into how many tokens and how many passes each source gets.

## Why proportional sampling is not enough

If every token in the corpus is equally likely to be drawn, each source's share of training equals its share of the data. A source that is 1 percent of the corpus gets 1 percent of training, even if its content, such as code, math or a low-resource language, matters far more than that.

Upweighting small sources means seeing them more often. That helps the model learn skills that are rare in the raw data, at the cost of repeating those sources several times and seeing less of the large ones. Choosing these weights is one of the main decisions in a pretraining data pipeline, made after the sources are cleaned and deduplicated and before training starts.

## The temperature exponent

Let $n_i$ be the number of tokens in source $i$ and $\alpha$ a nonnegative exponent. The sampling probability of source $i$ is

$$
p_i = \frac{n_i^{\alpha}}{\sum_{j : n_j > 0} n_j^{\alpha}}
$$

where the sum covers only sources that contain tokens. The exponent controls how flat the mixture is. With $\alpha = 1$ sampling is proportional to size. With $\alpha = 0$ every non-empty source gets the same probability, however small. Values in between, such as 0.3 used for languages in multilingual models like XLM-R and mT5, move part of the way from proportional toward uniform. The exponent is often written as $1/\tau$ for a temperature $\tau$, which is where the method's name comes from.

Empty sources need care. A zero count raised to the power 0 evaluates to 1, so a naive power would hand an empty source a share of the probability. Empty sources are therefore left out of the weights entirely and get zero probability, zero tokens and zero epochs.

## Turning probabilities into tokens and epochs

With a training budget of $T$ tokens, source $i$ is expected to supply

$$
t_i = T p_i
$$

tokens, and the number of passes, or epochs, through it is

$$
e_i = \frac{t_i}{n_i}
$$

An epoch count well above 1 means the source is repeated many times, which risks memorizing it. An epoch count below 1 means the source is not fully covered even once in expectation. Looking at epochs is how teams check whether a chosen exponent is reasonable.

## Computing powers without overflow

Raising large token counts to a power can overflow, since a count of $10^{12}$ raised to the power 30 is far beyond the largest float64. Writing each weight as $\exp(\alpha \log n_i)$ moves the problem into log space. Subtracting the largest $\log n_i$ from every log count before multiplying by $\alpha$ scales every weight by the same factor, which cancels in the normalization, and it keeps the product finite even when $\alpha$ is enormous.

## Two sources at three temperatures

Take two sources with 900 and 100 tokens and a training budget of 1000 tokens.

With $\alpha = 0.5$, the weights are $\sqrt{900} = 30$ and $\sqrt{100} = 10$, so the probabilities are

$$
p = \left[ \frac{30}{40}, \ \frac{10}{40} \right] = [0.75, \ 0.25]
$$

The sources supply 750 and 250 tokens, which is about 0.833 epochs of the large source and 2.5 epochs of the small one.

With $\alpha = 1$ the probabilities are 0.9 and 0.1, and both sources get exactly one epoch. With $\alpha = 0$ the two sources are sampled equally, so the small source gets 5 epochs while the large one gets only about 0.56. At $\alpha = 0.5$ the small source is repeated 2.5 times, half of what uniform sampling would do.

## Common mistakes to avoid

- **Letting empty sources into the weights.** A zero count raised to the power 0 gives 1, so empty sources must be dropped before normalizing or they take probability mass.
- **Raising counts to the power directly.** Large counts and exponents overflow unless the weights are computed in log space.
- **Multiplying by $\alpha$ before subtracting the offset.** With a huge exponent the products become infinite and their difference is undefined.
- **Normalizing before applying the exponent.** Raising already-normalized proportions to $\alpha$ without renormalizing gives probabilities that no longer sum to 1.
- **Dividing epochs by the budget.** Epochs are expected tokens divided by the source's own size.
