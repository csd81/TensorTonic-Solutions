## <span style="font-size: 20px;">Chi-Square Goodness-of-Fit Test</span>

The **chi-square test** assesses whether observed frequencies differ significantly from expected frequencies. It answers the question: "Does the data fit a hypothesized distribution?"

### The Chi-Square Statistic

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ is the observed count in category $i$, $E_i$ is the expected count, and $k$ is the number of categories. Each term measures the squared deviation from expectation, normalized by the expected count. The normalization by $E_i$ is important: a deviation of 5 from an expected count of 10 is much more noteworthy than a deviation of 5 from an expected count of 1000. This gives each category appropriate weight.

### The Chi-Square Distribution

Under $H_0$ (the data follow the expected distribution), the test statistic approximately follows a chi-square distribution with $df = k - 1$ degrees of freedom. We lose one degree of freedom because the total count is fixed (the category counts must sum to $n$). If $m$ parameters were estimated from the data to compute expected counts, then $df = k - 1 - m$.

The chi-square distribution is:
- Always non-negative (it is a sum of squared terms)
- Right-skewed, becoming more symmetric as $df$ increases
- Has mean $df$ and variance $2 \cdot df$
- The sum of $df$ independent standard normal squared variables: $\chi^2_{df} = \sum_{i=1}^{df} Z_i^2$

### Hypotheses

$$H_0: O_i = E_i \text{ for all } i \qquad H_a: O_i \neq E_i \text{ for at least one } i$$

The test is always **one-tailed (right-sided)**: only large values of $\chi^2$ provide evidence against $H_0$. A $\chi^2$ near zero means the observed data closely match expectations. Note that an extremely small $\chi^2$ (near zero with many categories) might itself be suspicious, suggesting data fabrication.

### P-Value and Decision

$$p = P(\chi^2_{df} \geq \chi^2_{\text{obs}})$$

Reject $H_0$ if $p < \alpha$ (typically 0.05). The p-value represents the probability of seeing a fit this poor (or worse) if the data truly follow the expected distribution.

### Expected Frequency Requirement

The chi-square approximation requires that **all expected frequencies $E_i \geq 5$**. When this is violated:
- Combine adjacent categories to increase expected counts
- Use Fisher's exact test (for 2x2 tables)
- Use simulation-based p-values (permutation tests)

This requirement exists because the chi-square distribution is a continuous approximation to a discrete test statistic. With small expected counts, the approximation breaks down.

### Yates Correction

For $k = 2$ categories (or 2x2 contingency tables), **Yates' continuity correction** adjusts for the discrete-to-continuous approximation:

$$\chi^2_{\text{Yates}} = \sum \frac{(|O_i - E_i| - 0.5)^2}{E_i}$$

This produces a slightly more conservative test (larger p-value). It is most important when expected counts are small and $k = 2$.

### Test of Independence

In a **contingency table** with $r$ rows and $c$ columns, the chi-square test of independence tests whether two categorical variables are independent:

$$E_{ij} = \frac{(\text{row } i \text{ total}) \cdot (\text{col } j \text{ total})}{n}$$

with $df = (r-1)(c-1)$. This test asks: "Is knowing one variable informative about the other?" It is widely used for feature analysis in categorical data.

### Residual Analysis

When the test rejects $H_0$, examine the **standardized residuals** to identify which categories deviate most:

$$r_i = \frac{O_i - E_i}{\sqrt{E_i}}$$

Categories with $|r_i| > 2$ are the primary contributors to the rejection.

### Applications in Data Science

**Feature independence testing**: Testing whether two categorical features are independent helps with feature selection and understanding relationships in the data.

**Model calibration assessment**: Comparing predicted probability bins against observed frequencies tests whether a classifier is well-calibrated (the Hosmer-Lemeshow test is a variant).

**Distribution fitting**: After fitting a distribution (Poisson, normal bins, etc.), the chi-square test validates whether the fit is adequate.

**Categorical data analysis**: Testing whether a categorical outcome variable has the expected distribution across groups - for example, whether error types are uniformly distributed across model versions.