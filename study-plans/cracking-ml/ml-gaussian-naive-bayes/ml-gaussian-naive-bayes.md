# <span style="font-size: 20px;">Gaussian Naive Bayes</span>

<span style="font-size: 14px;">Gaussian Naive Bayes is a generative classifier that models the class-conditional feature distributions as Gaussians, then applies Bayes' theorem to compute posterior probabilities. Despite the "naive" conditional independence assumption, it works surprisingly well in practice and is one of the fastest classifiers available.</span>

---

## <span style="font-size: 16px;">Generative vs Discriminative</span>

- <span style="font-size: 14px;">**Discriminative classifiers** (logistic regression, SVM) learn the decision boundary directly by modeling</span> $P(y \mid x)$
- <span style="font-size: 14px;">**Generative classifiers** (Naive Bayes, LDA) model</span> $P(x \mid y)$ <span style="font-size: 14px;">and</span> $P(y)$<span style="font-size: 14px;">, then use Bayes' rule to get</span> $P(y \mid x)$
- <span style="font-size: 14px;">Generative models can generate synthetic data, handle missing features, and work well with small datasets</span>
- <span style="font-size: 14px;">Discriminative models typically achieve higher accuracy when given enough data</span>

---

## <span style="font-size: 16px;">Bayes' Theorem</span>

$$
P(c \mid x) = \frac{P(x \mid c) \, P(c)}{P(x)} \propto P(x \mid c) \, P(c)
$$

- $P(c)$ <span style="font-size: 14px;">is the **prior**: the probability of class</span> $c$ <span style="font-size: 14px;">before seeing the features, estimated as the class frequency</span>
- $P(x \mid c)$ <span style="font-size: 14px;">is the **likelihood**: how likely the features are given the class</span>
- $P(c \mid x)$ <span style="font-size: 14px;">is the **posterior**: what we want to compute for classification</span>
- $P(x)$ <span style="font-size: 14px;">is the **evidence**: constant across classes, so we can ignore it for classification</span>

---

## <span style="font-size: 16px;">The Naive Independence Assumption</span>

<span style="font-size: 14px;">Computing</span> $P(x \mid c)$ <span style="font-size: 14px;">for a</span> $d$<span style="font-size: 14px;">-dimensional feature vector requires modeling the joint distribution of all features, which needs exponentially many parameters. The "naive" assumption is:</span>

$$
P(x \mid c) = \prod_{j=1}^{d} P(x_j \mid c)
$$

- <span style="font-size: 14px;">This assumes features are conditionally independent given the class</span>
- <span style="font-size: 14px;">Rarely true in practice (e.g., height and weight are correlated)</span>
- <span style="font-size: 14px;">Despite this, Naive Bayes still makes good classification decisions because it only needs to rank classes correctly, not estimate exact probabilities</span>

---

## <span style="font-size: 16px;">Gaussian Model</span>

<span style="font-size: 14px;">For continuous features, each</span> $P(x_j \mid c)$ <span style="font-size: 14px;">is modeled as a Gaussian:</span>

$$
P(x_j \mid c) = \frac{1}{\sqrt{2\pi\sigma_{c,j}^2}} \exp\left(-\frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}\right)
$$

- $\mu_{c,j}$ <span style="font-size: 14px;">is the mean of feature</span> $j$ <span style="font-size: 14px;">for class</span> $c$<span style="font-size: 14px;">, estimated from training data</span>
- $\sigma_{c,j}^2$ <span style="font-size: 14px;">is the variance of feature</span> $j$ <span style="font-size: 14px;">for class</span> $c$
- <span style="font-size: 14px;">Each class-feature pair has its own mean and variance (</span>$2 \times d \times K$ <span style="font-size: 14px;">parameters total for</span> $K$ <span style="font-size: 14px;">classes)</span>

---

## <span style="font-size: 16px;">Log-Space Computation</span>

<span style="font-size: 14px;">Working with log-probabilities avoids numerical underflow from multiplying many small probabilities:</span>

$$
\log P(c \mid x) \propto \log P(c) + \sum_{j=1}^{d} \log P(x_j \mid c)
$$

$$
\log P(x_j \mid c) = -\frac{1}{2}\log(2\pi\sigma_{c,j}^2) - \frac{(x_j - \mu_{c,j})^2}{2\sigma_{c,j}^2}
$$

<span style="font-size: 14px;">The predicted class is:</span>

$$
\hat{y} = \arg\max_c \left[\log P(c) + \sum_{j=1}^{d} \log P(x_j \mid c)\right]
$$

---

## <span style="font-size: 16px;">Numerical Stability</span>

- <span style="font-size: 14px;">If a feature has zero variance for a class (all training values identical), the Gaussian density becomes a Dirac delta, causing division by zero</span>
- <span style="font-size: 14px;">Add a small constant</span> $\epsilon$ <span style="font-size: 14px;">(e.g.,</span> $10^{-9}$<span style="font-size: 14px;">) to all variances to prevent this</span>
- <span style="font-size: 14px;">This is called "variance smoothing" and is standard practice</span>

---

## <span style="font-size: 16px;">Other Naive Bayes Variants</span>

- <span style="font-size: 14px;">**Multinomial NB**: for count/frequency data (e.g., word counts in text). Uses multinomial distribution instead of Gaussian</span>
- <span style="font-size: 14px;">**Bernoulli NB**: for binary features (e.g., word presence/absence). Uses Bernoulli distribution</span>
- <span style="font-size: 14px;">**Complement NB**: a variant of Multinomial NB that handles class imbalance better</span>
- <span style="font-size: 14px;">The choice of variant depends on the feature type: Gaussian for continuous, Multinomial for counts, Bernoulli for binary</span>

---

## <span style="font-size: 16px;">Strengths and Weaknesses</span>

- <span style="font-size: 14px;">**Fast**: training is</span> $O(nd)$ <span style="font-size: 14px;">(one pass through data), prediction is</span> $O(Kd)$ <span style="font-size: 14px;">per point</span>
- <span style="font-size: 14px;">**Works with small data**: few parameters to estimate, so less prone to overfitting with limited data</span>
- <span style="font-size: 14px;">**Handles many classes well**: linear in the number of classes</span>
- <span style="font-size: 14px;">**Poor probability estimates**: the independence assumption makes the posterior probabilities unreliable, but class rankings are often correct</span>
- <span style="font-size: 14px;">**Fails with correlated features**: redundant features are double-counted, distorting the posterior</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why "naive"?**</span>
  <span style="font-size: 14px;">A: The conditional independence assumption $P(x \mid c) = \prod_j P(x_j \mid c)$ is almost never true in real data. Despite this, the classifier works well because classification only requires ranking classes correctly, not calibrated probabilities</span>

- <span style="font-size: 14px;">**Q: When is Naive Bayes better than logistic regression?**</span>
  <span style="font-size: 14px;">A: With very small training sets, high-dimensional features, or when fast training is needed. In NLP with bag-of-words features, Multinomial NB is competitive with logistic regression and much faster</span>

- <span style="font-size: 14px;">**Q: What is Laplace smoothing?**</span>
  <span style="font-size: 14px;">A: Adding a pseudocount to prevent zero probabilities in Multinomial NB. For Gaussian NB, the analogous technique is variance smoothing (adding $\epsilon$ to variances)</span>

- <span style="font-size: 14px;">**Q: How does NB handle missing features?**</span>
  <span style="font-size: 14px;">A: Simply omit the missing feature's likelihood term from the product. This is a natural advantage of the factored model</span>

- <span style="font-size: 14px;">**Q: What happens with correlated features?**</span>
  <span style="font-size: 14px;">A: Correlated features are effectively double-counted in the likelihood product, giving them outsized influence. This biases the posterior but often does not change the predicted class</span>

---