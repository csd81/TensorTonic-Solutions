# <span style="font-size: 20px;">Vocabulary Building</span>

<span style="font-size: 14px;">A vocabulary maps tokens to integer IDs, converting text into numbers that models can process. Every NLP model - from bag-of-words to transformers - requires a vocabulary as the bridge between raw text and numerical computation.</span>

---

## <span style="font-size: 16px;">Why Build a Vocabulary?</span>

* <span style="font-size: 14px;">Neural networks operate on numbers, not strings. A vocabulary provides the mapping from tokens to integer indices</span>
* <span style="font-size: 14px;">Embedding layers use vocabulary IDs to look up dense vector representations for each token</span>
* <span style="font-size: 14px;">A fixed vocabulary defines the set of tokens the model can understand - anything outside it becomes unknown</span>
* <span style="font-size: 14px;">Vocabulary size directly impacts model size: each token needs its own embedding vector</span>

---

## <span style="font-size: 16px;">The Vocabulary Building Process</span>

### <span style="font-size: 14px;">Step 1: Count Token Frequencies</span>

<span style="font-size: 14px;">Iterate through the corpus and count how many times each token appears. This tells us which tokens are most important to include.</span>

### <span style="font-size: 14px;">Step 2: Sort and Select</span>

<span style="font-size: 14px;">Sort tokens by descending frequency and keep the top</span> $k$ <span style="font-size: 14px;">tokens (where</span> $k = \texttt{max\_vocab\_size} - 2$ <span style="font-size: 14px;">to leave room for special tokens). Alphabetical tiebreaking ensures deterministic ordering.</span>

### <span style="font-size: 14px;">Step 3: Assign IDs</span>

<span style="font-size: 14px;">Map each selected token to a unique integer index. Special tokens get reserved indices (typically 0 and 1).</span>

### <span style="font-size: 14px;">Step 4: Encode</span>

<span style="font-size: 14px;">Convert each sentence to a list of integer IDs using the vocabulary mapping. Unknown tokens map to the</span> `<unk>` <span style="font-size: 14px;">ID.</span>

---

## <span style="font-size: 16px;">Special Tokens</span>

<span style="font-size: 14px;">Special tokens serve specific purposes in the model architecture:</span>

* `<pad>` <span style="font-size: 14px;">(ID 0): Padding token. Used to fill sequences to equal length in a batch. The model learns to ignore padding through attention masks</span>
* `<unk>` <span style="font-size: 14px;">(ID 1): Unknown token. Replaces any token not in the vocabulary. Frequent</span> `<unk>` <span style="font-size: 14px;">usage indicates the vocabulary is too small</span>
* `<bos>` / `<eos>` <span style="font-size: 14px;">: Beginning/end of sequence markers. Used by sequence-to-sequence models and language models</span>
* `[CLS]` / `[SEP]` <span style="font-size: 14px;">: BERT-specific tokens for classification and segment separation</span>

---

## <span style="font-size: 16px;">Vocabulary Size Tradeoffs</span>

* <span style="font-size: 14px;">**Too small:** Many tokens map to</span> `<unk>`<span style="font-size: 14px;">, losing information. The model cannot distinguish between different rare words</span>
* <span style="font-size: 14px;">**Too large:** The embedding matrix becomes huge (vocabulary_size x embedding_dim parameters), increasing memory and computation. Rare tokens have few training examples, leading to poor embeddings</span>
* <span style="font-size: 14px;">**Typical sizes:** BERT uses 30,522 tokens, GPT-2 uses 50,257, GPT-4 uses around 100,000</span>
* <span style="font-size: 14px;">**Zipf's law:** Token frequencies follow a power law - a small vocabulary covers most occurrences. The top 10,000 words typically cover 95%+ of English text</span>

---

## <span style="font-size: 16px;">Frequency Thresholds vs Fixed Size</span>

<span style="font-size: 14px;">Two common strategies for vocabulary selection:</span>

* <span style="font-size: 14px;">**Fixed size** (this problem): Keep the top N most frequent tokens. Simple and predictable model size</span>
* <span style="font-size: 14px;">**Frequency threshold**: Keep all tokens appearing at least K times. Vocabulary size varies with corpus</span>
* <span style="font-size: 14px;">In practice, both are used: start with a frequency threshold, then cap at a maximum size</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why is pad at index 0?** Convention. Many frameworks default to 0 for padding. PyTorch's</span> `nn.Embedding` <span style="font-size: 14px;">has a</span> `padding_idx` <span style="font-size: 14px;">parameter that zeros out the gradient for that index</span>
* <span style="font-size: 14px;">**What about subword vocabularies?** BPE and WordPiece build vocabularies bottom-up from characters, avoiding the</span> `<unk>` <span style="font-size: 14px;">problem entirely. Modern models use subword vocabularies instead of word-level ones</span>
* <span style="font-size: 14px;">**How do you handle new tokens at inference time?** They map to</span> `<unk>`<span style="font-size: 14px;">. This is a fundamental limitation of word-level vocabularies and a key motivation for subword tokenization</span>

---