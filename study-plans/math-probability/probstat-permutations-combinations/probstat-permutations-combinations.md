## <span style="font-size: 20px;">Permutations and Combinations</span>

Counting is the foundation of discrete probability. The number of ways to arrange or select items determines the size of sample spaces and event sets, enabling probability calculations for discrete experiments.

### Factorial

The factorial $n!$ counts the number of ways to arrange $n$ distinct objects in a sequence:

$$n! = n \times (n-1) \times (n-2) \times \cdots \times 1$$

By convention, $0! = 1$. Factorials grow extremely fast: $10! = 3{,}628{,}800$ and $20! \approx 2.43 \times 10^{18}$. Stirling's approximation provides a useful estimate for large $n$:

$$n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n$$

This approximation is remarkably accurate even for moderate $n$ and is used when exact computation is intractable or when asymptotic analysis is needed.

### Permutations (Order Matters)

A **permutation** is an ordered arrangement of $r$ objects chosen from $n$ distinct objects:

$$P(n, r) = \frac{n!}{(n-r)!}$$

For each of the $r$ positions, we choose from a shrinking pool: $n$ choices for the first, $n-1$ for the second, and so on, giving $n(n-1)(n-2) \cdots (n-r+1)$.

Example: How many 3-letter "words" can be formed from 26 letters (no repeats)? $P(26, 3) = 26 \times 25 \times 24 = 15{,}600$.

When $r = n$, we get the total number of arrangements: $P(n, n) = n!$.

### Combinations (Order Does Not Matter)

A **combination** selects $r$ objects from $n$ without regard to order:

$$C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!} = \frac{P(n,r)}{r!}$$

The division by $r!$ removes the overcounting: each group of $r$ items can be arranged in $r!$ ways, all of which correspond to the same combination.

Example: How many 5-card poker hands from a 52-card deck? $\binom{52}{5} = 2{,}598{,}960$.

A useful symmetry property: $\binom{n}{r} = \binom{n}{n-r}$. Choosing $r$ items to include is equivalent to choosing $n-r$ items to exclude.

### Relationship Between Permutations and Combinations

$$P(n, r) = C(n, r) \times r!$$

Combinations count the groups; permutations count the ordered arrangements within each group. This relationship is a key tool for solving counting problems: first count the groups, then multiply by the number of orderings if order matters.

### Pascal's Triangle and the Binomial Theorem

The binomial coefficients $\binom{n}{r}$ form **Pascal's triangle**, where each entry is the sum of the two entries above it:

$$\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$$

This identity has a combinatorial proof: the $r$ items either include the $n$-th element (choose the remaining $r-1$ from $n-1$) or exclude it (choose all $r$ from $n-1$).

The **binomial theorem** connects combinations to algebra:

$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^k b^{n-k}$$

Setting $a = b = 1$ gives $2^n = \sum \binom{n}{k}$, the total number of subsets of an $n$-element set.

### Stars and Bars

The **stars and bars** method counts the number of ways to distribute $n$ identical items into $k$ distinct bins:

$$\binom{n + k - 1}{k - 1}$$

This technique appears when counting multisets or distributing resources across categories. For example, the number of ways to distribute 10 identical balls into 3 distinct boxes is $\binom{12}{2} = 66$.

### Permutations with Repetition

When objects are not all distinct, the number of arrangements of $n$ objects where object type $i$ appears $n_i$ times is the **multinomial coefficient**:

$$\frac{n!}{n_1! \cdot n_2! \cdots n_k!}$$

For example, the number of distinct arrangements of the letters in "MISSISSIPPI" is $\frac{11!}{1! \cdot 4! \cdot 4! \cdot 2!} = 34{,}650$.

### Applications in Machine Learning

**Hypothesis space sizes**: For a binary classifier on $n$ Boolean features, the hypothesis space contains $2^{2^n}$ distinct functions. Combinatorics quantifies these spaces and informs sample complexity bounds through the VC dimension theory.

**Ensemble methods**: Random forests select random subsets of features at each split. The number of possible subsets of size $m$ from $p$ features is $\binom{p}{m}$, which determines the diversity of the ensemble.

**Data augmentation counting**: When augmenting data by combining $k$ transformations from $n$ available options, the number of distinct augmentation pipelines is $\binom{n}{k}$ (if order does not matter) or $P(n, k)$ (if it does). This helps analyze the coverage of the augmentation strategy.