## <span style="font-size: 20px;">Bayes' Theorem</span>

Bayes' theorem is arguably the most important result in probability theory for machine learning. It provides a principled mechanism for updating beliefs in light of new evidence, connecting prior knowledge with observed data to produce refined posterior beliefs.

### The Formula

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

Each term has a specific name and interpretation:

| Term | Name | Meaning |
|------|------|---------|
| $P(A \mid B)$ | **Posterior** | Updated belief about A after observing B |
| $P(B \mid A)$ | **Likelihood** | How probable the evidence is if A is true |
| $P(A)$ | **Prior** | Initial belief about A before seeing evidence |
| $P(B)$ | **Evidence** | Total probability of observing B (normalizing constant) |

The theorem says: posterior is proportional to likelihood times prior, with the evidence serving as a normalizing constant to ensure probabilities sum to 1.

### Law of Total Probability

The evidence $P(B)$ is computed by marginalizing over all possible states of $A$:

$$P(B) = P(B \mid A) \cdot P(A) + P(B \mid A') \cdot P(A')$$

This is the **law of total probability**. It enumerates all pathways through which $B$ can occur, weighting each by its probability. More generally, if $A_1, A_2, \ldots, A_n$ partition the sample space:

$$P(B) = \sum_{i=1}^{n} P(B \mid A_i) \cdot P(A_i)$$

The law of total probability is crucial because we often know conditional probabilities (likelihoods) but need the marginal probability.

### Medical Testing Example

Consider a disease affecting 1% of the population ($P(D) = 0.01$). A test has:
- **Sensitivity** (true positive rate): $P(+ \mid D) = 0.95$
- **Specificity** (true negative rate): $P(- \mid D') = 0.90$, so $P(+ \mid D') = 0.10$

Applying Bayes' theorem:

$$P(D \mid +) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = \frac{0.0095}{0.1085} \approx 0.088$$

Despite a 95% sensitive test, a positive result only means an 8.8% chance of disease. The low prior (1%) dominates because there are so many more healthy people generating false positives than sick people generating true positives. This counterintuitive result demonstrates why base rates matter enormously in diagnostic reasoning.

### Bayesian vs. Frequentist Perspectives

The **frequentist** interpretation treats probabilities as long-run frequencies of repeatable experiments. Parameters are fixed but unknown, and inference is based on the sampling distribution.

The **Bayesian** interpretation treats probabilities as degrees of belief. Parameters themselves have probability distributions. Bayes' theorem provides the update rule:

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \cdot P(\theta)}{P(\text{data})}$$

The posterior $P(\theta \mid \text{data})$ combines prior beliefs $P(\theta)$ with the data likelihood $P(\text{data} \mid \theta)$. As more data arrives, the posterior concentrates around the true parameter value, and the influence of the prior diminishes.

### The Naive Bayes Classifier

For classification with features $x_1, \ldots, x_n$:

$$P(C_k \mid x_1, \ldots, x_n) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$

The "naive" assumption of conditional independence between features given the class makes computation tractable. Despite this strong assumption, Naive Bayes often performs surprisingly well in text classification, spam filtering, and sentiment analysis, particularly when the number of features is large relative to training data.

### Bayesian Neural Networks

Standard neural networks produce point estimates of weights. Bayesian neural networks place distributions over weights:

$$P(w \mid \mathcal{D}) \propto P(\mathcal{D} \mid w) \cdot P(w)$$

This provides calibrated uncertainty estimates, which are critical in safety-sensitive applications like medical imaging and autonomous driving. The posterior is typically intractable, so variational inference or Monte Carlo dropout are used as approximations.

### A/B Testing

In Bayesian A/B testing, instead of computing p-values, we directly compute $P(\text{variant B is better} \mid \text{data})$. Starting with prior beliefs about conversion rates and updating with observed clicks and conversions gives a posterior distribution that directly answers the business question, with intuitive interpretation that frequentist methods do not readily provide.