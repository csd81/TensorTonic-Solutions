# <span style="font-size: 20px;">Text Classification</span>

<span style="font-size: 14px;">Text classification assigns a category label to a document based on its content. It is one of the most common NLP tasks, used for spam detection, topic categorization, language identification, and content moderation. The standard interview approach is to implement a Naive Bayes classifier from scratch, which demonstrates understanding of probabilistic modeling, text representation, and the full train-predict pipeline.</span>

---

## <span style="font-size: 16px;">Naive Bayes for Text</span>

<span style="font-size: 14px;">Naive Bayes applies Bayes' theorem with the "naive" assumption that features (words) are conditionally independent given the class:</span>

$$P(c \mid d) \propto P(c) \prod_{w \in d} P(w \mid c)$$

<span style="font-size: 14px;">where $c$ is a class label, $d$ is a document, and $w$ ranges over the words in the document. The predicted class is the one with the highest posterior probability.</span>

---

## <span style="font-size: 16px;">Training: Estimating Parameters</span>

<span style="font-size: 14px;">From a labeled training corpus, estimate two sets of parameters:</span>

### <span style="font-size: 14px;">Prior Probability</span>

$$P(c) = \frac{N_c}{N}$$

<span style="font-size: 14px;">where $N_c$ is the number of documents with class $c$ and $N$ is the total number of documents.</span>

### <span style="font-size: 14px;">Likelihood (Word Probability)</span>

$$P(w \mid c) = \frac{\text{count}(w, c) + \alpha}{\sum_{w'} \text{count}(w', c) + \alpha V}$$

<span style="font-size: 14px;">where $\text{count}(w, c)$ is how many times word $w$ appears in documents of class $c$, $V$ is the vocabulary size (number of unique words across all classes), and $\alpha$ is the smoothing parameter (Laplace smoothing when $\alpha = 1$).</span>

---

## <span style="font-size: 16px;">Prediction: Log-space Computation</span>

<span style="font-size: 14px;">To classify a new document, compute the log-posterior for each class and pick the maximum:</span>

$$\hat{c} = \arg\max_c \left[ \log P(c) + \sum_{w \in d} \log P(w \mid c) \right]$$

<span style="font-size: 14px;">Working in log-space converts products into sums, preventing numerical underflow when multiplying many small probabilities. Unknown words (not seen in training) are handled by the smoothing term.</span>

---

## <span style="font-size: 16px;">Vocabulary Construction</span>

<span style="font-size: 14px;">The vocabulary is the set of all unique words seen across the entire training corpus (all classes combined). At prediction time, include every occurrence of an unseen word using a token count of zero, and keep the training vocabulary size fixed. The smoothing denominator uses the token total for each class, so unseen-word probabilities can differ between classes and can change the predicted label.</span>

---

## <span style="font-size: 16px;">Multinomial vs Bernoulli Naive Bayes</span>

* <span style="font-size: 14px;">**Multinomial**: Uses word counts (how many times each word appears). Standard for text classification. Each word occurrence contributes to the likelihood</span>
* <span style="font-size: 14px;">**Bernoulli**: Uses binary word presence (appears or not). Better for short documents where word frequency is less informative</span>

<span style="font-size: 14px;">The multinomial model is the standard interview implementation.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why is the independence assumption "naive"?** Words are obviously not independent: "New York" has a different meaning than "New" and "York" separately. Despite this, Naive Bayes works surprisingly well because classification only needs the correct ranking of classes, not calibrated probabilities</span>
* <span style="font-size: 14px;">**Why Laplace smoothing?** Without smoothing, a single unseen word gives zero probability to an entire class, dominating the prediction. Smoothing ensures every word has non-zero probability in every class</span>
* <span style="font-size: 14px;">**Naive Bayes vs logistic regression?** Naive Bayes is generative (models $P(d, c)$) while logistic regression is discriminative (models $P(c \mid d)$ directly). Logistic regression typically has higher accuracy on large datasets, but Naive Bayes trains faster and needs less data</span>

---