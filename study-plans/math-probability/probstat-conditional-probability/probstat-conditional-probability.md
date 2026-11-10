## <span style="font-size: 20px;">Conditional Probability</span>

Conditional probability answers a fundamental question: how does the probability of an event change when we learn that another event has occurred? This concept is at the heart of probabilistic reasoning and underpins virtually every machine learning algorithm that makes predictions from observed data.

### Definition

The conditional probability of $A$ given $B$ is defined as:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

Intuitively, once we know $B$ has occurred, the sample space shrinks from $\Omega$ to $B$. We then ask: what fraction of $B$ also belongs to $A$? The division by $P(B)$ re-normalizes the probabilities so they sum to 1 within this restricted space.

It is worth verifying that conditional probability satisfies the axioms of probability: $P(A \mid B) \geq 0$, $P(B \mid B) = 1$, and conditional probabilities of mutually exclusive events add. This confirms that conditioning produces a valid probability measure.

### The Multiplication (Chain) Rule

Rearranging the definition gives the **multiplication rule**:

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

This generalizes to chains of events:

$$P(A \cap B \cap C) = P(A) \cdot P(B \mid A) \cdot P(C \mid A \cap B)$$

And more generally for $n$ events:

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \prod_{i=2}^{n} P(A_i \mid A_1 \cap \cdots \cap A_{i-1})$$

The chain rule is essential in sequence modeling. In language models, the probability of a sentence $w_1, w_2, \ldots, w_n$ is decomposed as:

$$P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \cdots$$

### Asymmetry of Conditioning

A crucial insight is that $P(A \mid B) \neq P(B \mid A)$ in general. Confusing the two is called the **prosecutor's fallacy** or the **confusion of the inverse**. For example:

- $P(\text{fever} \mid \text{flu})$ is high: most flu patients have fever.
- $P(\text{flu} \mid \text{fever})$ is much lower: most fevers are caused by other illnesses.

Both values use the same numerator $P(\text{fever} \cap \text{flu})$, but the denominators differ. The relationship between the two directions is given by Bayes' theorem, which makes the role of the base rates explicit.

### Updating Beliefs with Evidence

Conditional probability provides a mechanism for **updating beliefs** as new evidence arrives. Starting with a prior belief $P(A)$, observing event $B$ updates our belief to $P(A \mid B)$. This update can increase, decrease, or leave unchanged our belief in $A$:

- If $P(A \mid B) > P(A)$: observing $B$ makes $A$ more likely. $B$ is **positive evidence** for $A$.
- If $P(A \mid B) < P(A)$: observing $B$ makes $A$ less likely. $B$ is **negative evidence** for $A$.
- If $P(A \mid B) = P(A)$: $B$ provides no information about $A$ (independence).

The ratio $P(A \mid B) / P(A)$ quantifies the **evidential strength** of $B$ regarding $A$.

### Conditional Independence

Events $A$ and $B$ are **conditionally independent** given $C$ if:

$$P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)$$

Events can be marginally dependent yet conditionally independent (or vice versa). This distinction is critical in graphical models and Bayesian networks, where conditional independence assumptions encode the structure of the model and enable efficient inference algorithms.

### Applications in Machine Learning

**Spam filtering**: The probability $P(\text{spam} \mid \text{words})$ is estimated using conditional word frequencies. Each word contributes evidence for or against the spam hypothesis through its conditional probability given each class.

**Medical diagnosis**: Given test results (symptoms, lab values), the conditional probability of each disease is computed. The differential diagnosis ranks conditions by $P(\text{disease}_i \mid \text{evidence})$.

**Language models**: Autoregressive models like GPT compute $P(\text{next token} \mid \text{context})$ at each step, directly implementing the chain rule decomposition of joint probability over token sequences. The quality of these conditional estimates determines the fluency and coherence of generated text.

**Hidden Markov Models**: State transitions and emissions are both defined through conditional probabilities, enabling sequence labeling tasks like part-of-speech tagging and speech recognition.