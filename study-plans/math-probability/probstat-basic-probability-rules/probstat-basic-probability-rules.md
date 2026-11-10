## <span style="font-size: 20px;">Basic Probability Rules</span>

Probability theory provides a mathematical framework for quantifying uncertainty. Every event $A$ in a sample space $\Omega$ is assigned a number $P(A) \in [0, 1]$ satisfying Kolmogorov's three axioms:

1. **Non-negativity**: $P(A) \geq 0$ for every event $A$.
2. **Normalization**: $P(\Omega) = 1$.
3. **Additivity**: For mutually exclusive events $A$ and $B$, $P(A \cup B) = P(A) + P(B)$.

These three axioms generate all the probability rules we use in practice. They constrain probabilities to behave consistently, and every formula below can be derived from them.

### The Addition Rule (Inclusion-Exclusion)

When events are **not** mutually exclusive, they can overlap. Naive addition would double-count the elements in $A \cap B$. The inclusion-exclusion principle corrects for this:

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

This can be visualized with a Venn diagram: the two circles overlap in the middle, and subtracting the intersection removes the double-counted region. For mutually exclusive events, $P(A \cap B) = 0$, and the formula reduces to simple addition.

The principle generalizes to three or more events:

$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

The alternating pattern of addition and subtraction ensures each outcome in the union is counted exactly once.

### Complement Rule

The complement $A'$ (also written $A^c$ or $\bar{A}$) contains all outcomes **not** in $A$. Since $A$ and $A'$ are mutually exclusive and exhaustive:

$$P(A') = 1 - P(A)$$

This is extremely useful in practice. When computing $P(\text{at least one})$ is hard, it is often easier to compute $P(\text{none})$ and subtract from 1. For example, the probability of getting at least one head in 10 coin flips is $1 - (0.5)^{10} = 0.999$. The complement approach turns a sum of many terms into a single calculation.

### Intersection with Complements

The event $A \cap B'$ represents outcomes that are in $A$ but not in $B$. Since $A$ can be partitioned into $(A \cap B)$ and $(A \cap B')$:

$$P(A \cap B') = P(A) - P(A \cap B)$$

This identity is a direct consequence of the additivity axiom applied to the partition of $A$. It answers the question: "What is the probability that A occurs without B also occurring?"

### Venn Diagrams and Set Operations

Venn diagrams provide geometric intuition for probability rules. Each region corresponds to a distinct combination of membership:

| Region | Meaning | Formula |
|--------|---------|---------|
| $A \cap B$ | Both A and B occur | Given directly |
| $A \cap B'$ | Only A occurs | $P(A) - P(A \cap B)$ |
| $A' \cap B$ | Only B occurs | $P(B) - P(A \cap B)$ |
| $(A \cup B)'$ | Neither occurs | $1 - P(A \cup B)$ |

The four regions are mutually exclusive and their probabilities sum to 1. This partition provides a complete accounting of all possible outcomes for two events.

### Mutually Exclusive Events

Two events are **mutually exclusive** (disjoint) if $A \cap B = \emptyset$, meaning $P(A \cap B) = 0$. This simplifies the addition rule to $P(A \cup B) = P(A) + P(B)$. Note that mutually exclusive events with positive probabilities can never be independent, since independence requires $P(A \cap B) = P(A)P(B) > 0$. This distinction between "mutually exclusive" and "independent" is a common source of confusion.

### Boole's Inequality (Union Bound)

A useful upper bound that avoids needing the intersection:

$$P(A \cup B) \leq P(A) + P(B)$$

This generalizes to $P(\bigcup_i A_i) \leq \sum_i P(A_i)$ and is known as the **union bound**. It is widely used in theoretical machine learning, particularly in PAC learning bounds and uniform convergence arguments, where computing exact intersection probabilities is intractable.

### Applications in Machine Learning

These rules form the foundation of the **Naive Bayes** classifier. The classifier computes posterior probabilities for each class using Bayes' theorem, which itself relies on the addition rule (through the law of total probability) and the complement rule. Feature probabilities are combined under the naive independence assumption, where the joint probability of features given a class is the product of individual conditional probabilities.

In **ensemble methods**, the probability that at least one model in a committee makes a correct prediction uses inclusion-exclusion. If $k$ independent models each have accuracy $p$, the ensemble error rate (all wrong) is $(1-p)^k$, giving ensemble accuracy $1 - (1-p)^k$ via the complement rule.

Understanding these building blocks is essential before progressing to conditional probability, Bayes' theorem, and the distributional concepts that underpin modern statistical learning.