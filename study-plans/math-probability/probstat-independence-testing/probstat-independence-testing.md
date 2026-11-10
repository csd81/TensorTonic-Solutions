## <span style="font-size: 20px;">Independence Testing</span>

Statistical independence is one of the most fundamental concepts in probability and machine learning. Understanding whether events or variables are independent drives model design, feature engineering, and inference algorithms.

### Definition of Independence

Two events $A$ and $B$ are **independent** if and only if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently, independence means that conditioning on one event does not change the probability of the other:

$$P(A \mid B) = P(A) \quad \text{and} \quad P(B \mid A) = P(B)$$

Independence is a **symmetric** property: if $A$ is independent of $B$, then $B$ is independent of $A$.

### Testing Independence

To test whether events are independent given their probabilities:

1. Compute the **expected joint probability** under independence: $P(A) \cdot P(B)$.
2. Compare with the **actual joint probability**: $P(A \cap B)$.
3. If they are equal (within floating-point tolerance), the events are independent.

When the expected joint exceeds the actual joint, the events are **negatively associated** (knowing one occurred makes the other less likely). When the actual exceeds expected, they are **positively associated**.

### Conditional Independence

Events $A$ and $B$ are **conditionally independent** given $C$ if:

$$P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)$$

This is written $A \perp\!\!\perp B \mid C$. Crucially, marginal independence does not imply conditional independence, and vice versa. Classic examples:

- **Explaining away**: Two independent causes of a common effect become dependent once the effect is observed. If a fire alarm goes off (effect), learning that cooking smoke triggered it (cause 1) makes an actual fire (cause 2) less likely.
- **Confounding**: Two marginally dependent variables may become independent after conditioning on a confounder.

### Pairwise vs. Mutual Independence

For three events $A$, $B$, $C$:

- **Pairwise independence**: Each pair is independent ($A \perp\!\!\perp B$, $A \perp\!\!\perp C$, $B \perp\!\!\perp C$).
- **Mutual independence**: All subsets are independent, including $P(A \cap B \cap C) = P(A) P(B) P(C)$.

Pairwise independence does **not** imply mutual independence. A classic counterexample: roll two fair dice. Let $A$ = "first die is odd", $B$ = "second die is odd", $C$ = "sum is odd". Any two of these are pairwise independent, but they are not mutually independent since knowing any two determines the third.

### Chi-Square Test for Independence

In practice, with observed data, we use the **chi-square test of independence**. Given a contingency table with observed counts $O_{ij}$ and expected counts $E_{ij}$ (under independence):

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Under the null hypothesis of independence, this statistic follows a chi-square distribution with $(r-1)(c-1)$ degrees of freedom.

### Applications in Machine Learning

**Feature selection**: Independent features provide non-redundant information. Mutual information $I(X; Y) = 0$ if and only if $X$ and $Y$ are independent. Features with high mutual information with the target are selected, while features independent of the target are discarded.

**Graphical models**: Bayesian networks and Markov random fields encode conditional independence assumptions as graph structure. The **d-separation** criterion in directed graphs determines conditional independence from the graph topology alone, enabling efficient probabilistic inference.

**Causal inference**: Independence and conditional independence tests are used to discover causal structure from observational data. The PC algorithm and IC algorithm build causal graphs by systematically testing conditional independence relationships.

**Naive Bayes assumption**: The assumption that features are conditionally independent given the class label dramatically reduces the number of parameters from exponential to linear in the number of features, making the model tractable even with high-dimensional data.