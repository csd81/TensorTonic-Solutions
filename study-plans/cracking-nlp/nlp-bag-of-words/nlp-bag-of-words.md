# <span style="font-size: 20px;">Bag of Words</span>

<span style="font-size: 14px;">The Bag of Words (BoW) model is the simplest and most foundational text representation in NLP. It converts a document into a fixed-length numerical vector by counting how many times each vocabulary word appears, completely ignoring word order and grammar.</span>

---

## <span style="font-size: 16px;">The Core Idea</span>

* <span style="font-size: 14px;">A document is represented as a "bag" (multiset) of its words - position and order are discarded</span>
* <span style="font-size: 14px;">Given a vocabulary of size $V$, each document becomes a vector of length $V$ where entry $i$ is the count of vocabulary word $i$ in the document</span>
* <span style="font-size: 14px;">Two documents with the same word frequencies produce identical vectors, regardless of word arrangement</span>
* <span style="font-size: 14px;">"The cat sat on the mat" and "The mat sat on the cat" produce the same BoW vector</span>

---

## <span style="font-size: 16px;">Building a BoW Representation</span>

### <span style="font-size: 14px;">Step 1: Build the Vocabulary</span>

<span style="font-size: 14px;">Collect all unique tokens across the corpus. Assign each token an index (typically sorted alphabetically for determinism).</span>

### <span style="font-size: 14px;">Step 2: Count Occurrences</span>

<span style="font-size: 14px;">For each document, count how many times each vocabulary word appears. The result is a sparse vector of length $V$.</span>

### <span style="font-size: 14px;">Step 3: Construct the Matrix</span>

<span style="font-size: 14px;">Stack all document vectors to form a document-term matrix of shape $(D, V)$ where $D$ is the number of documents.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Corpus: ["the cat sat", "the dog sat", "a cat ran"]</span>

<span style="font-size: 14px;">Vocabulary (sorted): {a: 0, cat: 1, dog: 2, ran: 3, sat: 4, the: 5}</span>

<span style="font-size: 14px;">Document vectors:</span>

* <span style="font-size: 14px;">"the cat sat" -> [0, 1, 0, 0, 1, 1]</span>
* <span style="font-size: 14px;">"the dog sat" -> [0, 0, 1, 0, 1, 1]</span>
* <span style="font-size: 14px;">"a cat ran"   -> [1, 1, 0, 1, 0, 0]</span>

---

## <span style="font-size: 16px;">Variants</span>

* <span style="font-size: 14px;">**Binary BoW**: Use 1/0 instead of counts - only records whether a word is present, not how many times</span>
* <span style="font-size: 14px;">**Normalized BoW**: Divide by document length to handle documents of different sizes</span>
* <span style="font-size: 14px;">**TF-IDF**: Weight counts by inverse document frequency to downweight common words. Covered in the next problem</span>
* <span style="font-size: 14px;">**N-gram BoW**: Use bigrams or trigrams as vocabulary entries to capture some word order</span>

---

## <span style="font-size: 16px;">Strengths and Limitations</span>

* <span style="font-size: 14px;">**Strengths**: Simple, fast, interpretable. Works well for text classification with linear models. Easy to compute and store</span>
* <span style="font-size: 14px;">**Loses word order**: "not good" and "good not" are identical. This is the fundamental limitation</span>
* <span style="font-size: 14px;">**Sparse and high-dimensional**: With a vocabulary of 100k words, each vector has 100k dimensions but most entries are zero</span>
* <span style="font-size: 14px;">**No semantic similarity**: "happy" and "joyful" are as different as "happy" and "table" - the model has no notion of meaning</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**When would you use BoW over embeddings?** When interpretability matters, when you have limited data, or as a baseline. BoW with logistic regression is a strong baseline for text classification</span>
* <span style="font-size: 14px;">**How does scikit-learn implement BoW?**</span> <span style="font-family:monospace; font-size:13px;">CountVectorizer</span> <span style="font-size: 14px;">builds the vocabulary and transforms documents into sparse count vectors in a single pipeline</span>
* <span style="font-size: 14px;">**What is the relationship between BoW and TF-IDF?** TF-IDF starts with BoW counts (term frequency) and multiplies by inverse document frequency to downweight words that appear in many documents</span>

---