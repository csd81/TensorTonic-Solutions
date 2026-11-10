## <span style="font-size: 20px;">Bootstrap Confidence Intervals</span>

The **bootstrap** is a resampling method that estimates the sampling distribution of a statistic by repeatedly sampling with replacement from the observed data. It was introduced by Bradley Efron in 1979 and has become one of the most widely used tools in applied statistics.

### The Bootstrap Procedure

Given a sample of $n$ observations $x_1, x_2, \dots, x_n$:

1. Draw a bootstrap sample $x_1^*, x_2^*, \dots, x_n^*$ by sampling $n$ values with replacement from the original data
2. Compute the statistic of interest $\hat{\theta}^*$ on the bootstrap sample
3. Repeat steps 1-2 a total of $B$ times (typically $B = 1000$ to $10000$)
4. Use the distribution of $\hat{\theta}^*_1, \dots, \hat{\theta}^*_B$ to estimate properties of the estimator

Each bootstrap sample has the same size $n$ as the original data, but some observations appear multiple times while others may not appear at all. On average, about 63.2% of the original observations appear in each bootstrap sample (since the probability of not being selected in $n$ draws is $(1-1/n)^n \approx e^{-1} \approx 0.368$).

### Percentile Method for Confidence Intervals

The simplest bootstrap CI uses the percentiles of the bootstrap distribution. For a $(1-\alpha)$ confidence interval:

$$\text{CI} = \left[\hat{\theta}^*_{(\alpha/2)}, \; \hat{\theta}^*_{(1-\alpha/2)}\right]$$

For a 95% CI, we take the 2.5th and 97.5th percentiles of the sorted bootstrap statistics. This is intuitive: the middle 95% of bootstrap estimates defines the interval.

### Why It Works

The key insight is the **bootstrap principle**: the relationship between the sample and the bootstrap samples mirrors the relationship between the population and the sample. By studying how the statistic varies across bootstrap resamples, we learn how it would vary across actual repeated samples from the population. The empirical distribution of the data serves as a proxy for the true population distribution.

### Improved Methods

The percentile method can be biased. More sophisticated approaches include:

- **BCa (bias-corrected and accelerated)**: Adjusts for both median bias and skewness in the bootstrap distribution. It applies a correction to the percentile indices based on the proportion of bootstrap estimates below the original estimate and a jackknife-based acceleration constant. Generally preferred for publication-quality results.
- **Basic (or pivotal) bootstrap**: Uses $2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)}$ and $2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}$ as bounds. This reflects the bootstrap distribution around the original estimate, correcting for bias.
- **Studentized bootstrap**: Standardizes each bootstrap replicate by its own estimated SE, producing coverage closer to the nominal level.

### When Bootstrap Works Well

The bootstrap is reliable when:
- The statistic is a smooth function of the data (means, variances, regression coefficients, correlation)
- The sample reasonably represents the population structure
- The sample size is not extremely small ($n \geq 20$ is a reasonable minimum)
- The underlying distribution has finite variance

### When Bootstrap Fails

The bootstrap can break down for:
- **Extreme order statistics** (max, min) - the bootstrap cannot produce values beyond the observed range
- **Very small samples** ($n < 10$) - not enough data to resample meaningfully
- **Heavy-tailed distributions** - convergence may require very large $B$
- **Non-independent data** (time series) - standard bootstrap destroys temporal structure; block bootstrap is needed instead

### Applications in Machine Learning

**Bagging (Bootstrap Aggregating)**: Random Forests train each tree on a bootstrap sample of the training data. This reduces variance and improves generalization. The out-of-bootstrap samples serve as validation data (out-of-bag error estimate).

**Uncertainty estimation**: Bootstrap any metric (accuracy, AUC, F1) to get a confidence interval rather than a single point estimate. This is especially valuable for small test sets.

**Cross-validation connection**: Both bootstrap and cross-validation estimate out-of-sample performance. The .632 bootstrap estimator blends in-sample and out-of-sample error: $\hat{Err}_{.632} = 0.368 \cdot \overline{err} + 0.632 \cdot \hat{Err}^{(1)}$, balancing bias and variance.

**Feature importance**: Bootstrap the feature importance calculation to assess which features have reliably high importance versus those that fluctuate due to sampling variability.