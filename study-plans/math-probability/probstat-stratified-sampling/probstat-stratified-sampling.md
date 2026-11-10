## <span style="font-size: 20px;">Stratified Sampling</span>

**Stratified sampling** divides a population into non-overlapping subgroups called **strata** and samples independently from each stratum. This design exploits known structure in the population to produce more precise estimates than simple random sampling (SRS).

### Why Stratify?

If the population consists of groups that differ substantially in their means but are relatively homogeneous within each group, stratification reduces the overall variance of the estimator. The key insight is that within-stratum variance is typically smaller than the total population variance, and stratified sampling eliminates the between-stratum component of variance entirely. The total population variance can be decomposed as:

$$\sigma^2_{\text{total}} = \sigma^2_{\text{within}} + \sigma^2_{\text{between}}$$

Stratification removes $\sigma^2_{\text{between}}$ from the sampling error, which can be a substantial reduction when strata are well-chosen.

### Proportional Allocation

The most common allocation scheme assigns samples proportionally to stratum size:

$$n_h = n \cdot \frac{N_h}{N}$$

where $n_h$ is the number of samples from stratum $h$, $N_h$ is the stratum population size, $N = \sum N_h$ is the total population size, and $n$ is the total sample size. This ensures each stratum is represented in proportion to its share of the population. Proportional allocation is self-weighting: the stratified mean simplifies to the ordinary sample mean.

### Neyman Allocation

When the goal is to minimize the variance of the stratified mean, **Neyman (optimal) allocation** accounts for both stratum size and variability:

$$n_h = n \cdot \frac{N_h \sigma_h}{\sum_k N_k \sigma_k}$$

Strata with larger populations or higher variability receive more samples. This can be substantially more efficient than proportional allocation when strata have very different variances. For example, if one stratum is highly variable and another is nearly constant, Neyman allocation concentrates samples where they are most needed.

### The Stratified Mean

The stratified estimate of the population mean is a weighted average of stratum means:

$$\bar{x}_{st} = \sum_{h=1}^{H} W_h \bar{x}_h \qquad \text{where } W_h = \frac{N_h}{N}$$

Each stratum mean $\bar{x}_h$ is weighted by its population share $W_h$. This is an unbiased estimator of the population mean regardless of the allocation scheme used. The weights ensure that larger strata contribute more to the overall estimate.

### Variance of the Stratified Mean

Under proportional allocation, assuming sampling with replacement (or with negligible sampling fraction):

$$\text{Var}(\bar{x}_{st}) = \sum_{h=1}^{H} W_h^2 \cdot \frac{\sigma_h^2}{n_h}$$

The stratified standard error is $SE_{st} = \sqrt{\text{Var}(\bar{x}_{st})}$. This is typically smaller than the SRS standard error $\sigma / \sqrt{n}$ because it only reflects within-stratum variability.

### Variance Reduction over SRS

Stratification is most effective when:
- **Between-group variance is large**: The stratum means differ substantially from each other
- **Within-group variance is small**: Observations within each stratum are relatively similar
- **Strata are chosen based on variables correlated with the outcome**: The stronger the correlation, the greater the variance reduction
- The extreme case: if all within-stratum variances were zero, a single observation per stratum would perfectly estimate the population mean

The efficiency gain can be expressed as the design effect $DEFF = \text{Var}_{SRS} / \text{Var}_{stratified}$. Values greater than 1 indicate that stratification is beneficial.

### Practical Considerations

When implementing proportional allocation, the raw allocations $n \cdot N_h / N$ are generally not integers. Common rounding approaches include rounding to nearest integer and adjusting the largest stratum to ensure the total equals $n$. Every stratum should receive at least 2 observations (to estimate within-stratum variance).

### Applications in Data Science

**Training/test splits by class**: In classification, stratified splits ensure each class is proportionally represented in both training and test sets. This prevents unlucky splits where rare classes are underrepresented, which is especially important for imbalanced datasets.

**Oversampling minority classes**: Techniques like SMOTE can be viewed as adjusting the sampling allocation to give rare classes more representation, analogous to deviating from proportional allocation toward a more balanced scheme.

**Survey design**: Large-scale surveys (census, market research) use stratification by demographics to ensure precise estimates for each subgroup while controlling total sample size.

**Experimental design**: Blocking in experimental design is analogous to stratification - subjects are grouped by a known confounder, and treatment is randomized within blocks to reduce confounding variance.