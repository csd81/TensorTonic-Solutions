## <span style="font-size: 20px;">Testing the Significance of Correlation</span>

Pearson's correlation coefficient $r$ measures the strength and direction of the linear relationship between two variables. But observing a non-zero $r$ does not automatically mean the true population correlation $\rho$ is non-zero - we need a formal significance test to determine whether the observed correlation could have arisen by chance.

### The Pearson Correlation Coefficient

For paired observations $(x_1, y_1), \ldots, (x_n, y_n)$, the sample Pearson correlation is:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

The value $r$ ranges from $-1$ (perfect negative linear relationship) to $+1$ (perfect positive linear relationship), with $0$ indicating no linear association. Importantly, $r$ measures only linear relationships - a strong non-linear pattern can produce $r \approx 0$.

### Hypothesis Test for Correlation

To test $H_0: \rho = 0$ against $H_a: \rho \neq 0$, we use the test statistic:

$$t = r\sqrt{\frac{n-2}{1-r^2}}$$

Under $H_0$, this follows a $t$-distribution with $df = n - 2$ degrees of freedom. The p-value for a two-sided test is $2 \cdot P(T > |t|)$, and we reject $H_0$ at significance level $\alpha = 0.05$ if $p < 0.05$.

This transformation works because when $\rho = 0$, the sampling distribution of $r$ can be converted to a $t$-distribution through this formula. The two degrees of freedom lost ($n-2$) correspond to estimating the two means $\bar{x}$ and $\bar{y}$. The relationship between the correlation test and the t-test is deep: testing $r = 0$ is exactly equivalent to testing whether the slope in simple linear regression is zero.

### Coefficient of Determination

The quantity $r^2$ (coefficient of determination) has a direct interpretation: it is the fraction of variance in $y$ that is explained by the linear relationship with $x$. For example, $r = 0.7$ gives $r^2 = 0.49$, meaning 49% of the variability in $y$ is accounted for by its linear relationship with $x$. This connects correlation to regression: in simple linear regression, $R^2 = r^2$.

### Beyond Pearson: Spearman Rank Correlation

When the relationship is monotonic but not necessarily linear, or when the data contains outliers, **Spearman's rank correlation** $r_s$ is more appropriate. It applies Pearson's formula to the ranks of the data rather than the raw values, making it robust to non-linear monotonic relationships and less sensitive to extreme values. For example, an exponential relationship $y = e^x$ would show $r_s = 1$ but $r < 1$.

### Partial Correlation

In multivariate settings, **partial correlation** measures the association between two variables after controlling for the effect of other variables. This is critical for distinguishing direct from spurious associations. For instance, ice cream sales and drowning rates are correlated, but the partial correlation controlling for temperature reveals no direct association. Partial correlation is computed by regressing both variables on the confounders and correlating the residuals.

### Assumptions

The significance test for Pearson's $r$ requires: (1) linearity of the relationship, (2) bivariate normality (or at least marginal normality for approximate validity), (3) homoscedasticity (constant variance of residuals), and (4) independence of observations. Violations of these assumptions can inflate Type I error rates or reduce statistical power. When assumptions are violated, consider Spearman's correlation or permutation-based tests as alternatives.

### Applications in Machine Learning

Correlation analysis is fundamental to feature selection: features with high absolute correlation with the target are good predictive candidates, while highly correlated feature pairs indicate **multicollinearity** that can destabilize coefficient estimates in linear models and inflate variance. In exploratory data analysis, correlation matrices and heatmaps provide a rapid overview of pairwise relationships in high-dimensional datasets. Correlation-based distance measures are also used in hierarchical clustering, feature grouping, and dimensionality reduction methods.