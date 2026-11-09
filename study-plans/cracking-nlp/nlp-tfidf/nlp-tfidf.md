# <span style="font-size: 20px;">TF-IDF</span>

<span style="font-size: 14px;">TF-IDF (Term Frequency - Inverse Document Frequency) is one of the most widely used text representations in information retrieval and NLP. It improves upon raw Bag of Words counts by downweighting words that appear in many documents and upweighting words that are distinctive to specific documents.</span>

---

## <span style="font-size: 16px;">The Intuition</span>

* <span style="font-size: 14px;">A word that appears frequently in a document is important to that document (high TF)</span>
* <span style="font-size: 14px;">A word that appears in every document is not distinctive and carries little discriminative power (low IDF)</span>
* <span style="font-size: 14px;">TF-IDF multiplies these two signals: words that are both frequent in a document AND rare across the corpus get the highest scores</span>
* <span style="font-size: 14px;">Common words like "the", "is", "and" get low TF-IDF because they appear in nearly every document</span>

---

## <span style="font-size: 16px;">Term Frequency (TF)</span>

<span style="font-size: 14px;">The simplest definition of term frequency is the raw count of term $t$ in document $d$:</span>

$$\text{TF}(t, d) = \text{count of } t \text{ in } d$$

<span style="font-size: 14px;">Variants include:</span>

* <span style="font-size: 14px;">**Normalized TF**: Divide by document length to prevent bias toward longer documents</span>
* <span style="font-size: 14px;">**Log-scaled TF**: $1 + \log(\text{TF})$ to reduce the impact of very high counts</span>
* <span style="font-size: 14px;">**Boolean TF**: 1 if the term is present, 0 otherwise</span>

---

## <span style="font-size: 16px;">Inverse Document Frequency (IDF)</span>

<span style="font-size: 14px;">IDF measures how rare a term is across the entire corpus:</span>

$$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)$$

<span style="font-size: 14px;">where $N$ is the total number of documents and $\text{DF}(t)$ is the number of documents containing term $t$.</span>

* <span style="font-size: 14px;">A term in every document: $\text{IDF} = \log(N/N) = \log(1) = 0$</span>
* <span style="font-size: 14px;">A term in one document: $\text{IDF} = \log(N/1) = \log(N)$, the maximum value</span>
* <span style="font-size: 14px;">Scikit-learn uses a smoothed variant: $\log((1 + N) / (1 + \text{DF})) + 1$ to avoid division by zero and zero IDF</span>

---

## <span style="font-size: 16px;">TF-IDF Score</span>

<span style="font-size: 14px;">The TF-IDF score for term $t$ in document $d$ is simply:</span>

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

<span style="font-size: 14px;">This produces a document-term matrix where each entry reflects both local importance (TF) and global rarity (IDF).</span>

---

## <span style="font-size: 16px;">Applications</span>

* <span style="font-size: 14px;">**Information retrieval**: Search engines rank documents by TF-IDF similarity to the query</span>
* <span style="font-size: 14px;">**Text classification**: TF-IDF vectors with linear classifiers (SVM, logistic regression) are strong baselines</span>
* <span style="font-size: 14px;">**Keyword extraction**: The highest TF-IDF terms in a document are its most distinctive keywords</span>
* <span style="font-size: 14px;">**Document similarity**: Cosine similarity between TF-IDF vectors measures topical overlap</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why use log in IDF?** Without the log, IDF values span a huge range (1 to N). The log compresses this range, preventing rare terms from dominating the representation</span>
* <span style="font-size: 14px;">**What happens to stop words?** High-frequency words like "the" appear in most documents, so their IDF approaches zero, naturally downweighting them without an explicit stop word list</span>
* <span style="font-size: 14px;">**TF-IDF vs word embeddings?** TF-IDF captures lexical importance but not semantics. Word embeddings capture meaning but not document-level importance. BM25 and dense retrieval models attempt to combine both signals</span>

---