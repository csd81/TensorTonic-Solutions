# <span style="font-size: 20px;">Tokenization</span>

<span style="font-size: 14px;">Tokenization is the process of breaking a string of text into meaningful units called **tokens**. It is the first step in virtually every NLP pipeline, from classical bag-of-words models to modern transformer-based language models.</span>

---

## <span style="font-size: 16px;">Why Tokenization Matters</span>

* <span style="font-size: 14px;">Models operate on discrete units, not raw character streams</span>
* <span style="font-size: 14px;">The choice of tokenization strategy directly affects vocabulary size, model performance, and handling of unseen words</span>
* <span style="font-size: 14px;">Downstream tasks (NER, classification, translation) depend on correct token boundaries</span>

---

## <span style="font-size: 16px;">Word-Level Tokenization</span>

<span style="font-size: 14px;">The simplest approach splits text on whitespace. However, naive whitespace splitting fails because punctuation remains attached to words:</span>

* <span style="font-size: 14px;">**Input:** "Hello, world!"</span>
* <span style="font-size: 14px;">**Whitespace split:** ["Hello,", "world!"] (punctuation stuck to words)</span>
* <span style="font-size: 14px;">**Correct tokenization:** ["Hello", ",", "world", "!"] (punctuation separated)</span>

<span style="font-size: 14px;">The key insight is that we need to treat punctuation characters as their own tokens while keeping word characters grouped together.</span>

---

## <span style="font-size: 16px;">Rule-Based Approach</span>

<span style="font-size: 14px;">A robust word tokenizer uses pattern matching to extract two kinds of tokens from the input string:</span>

* <span style="font-size: 14px;">**Word tokens:** maximal sequences of word characters (letters, digits, underscore)</span>
* <span style="font-size: 14px;">**Punctuation tokens:** individual characters that are neither word characters nor whitespace</span>

<span style="font-size: 14px;">This approach handles multiple spaces, leading/trailing whitespace, and adjacent punctuation naturally, since whitespace is simply not matched and gets discarded.</span>

---

## <span style="font-size: 16px;">Regular Expression Matching</span>

<span style="font-size: 14px;">Regular expressions provide an elegant solution. The key character classes are:</span>

* <span style="font-family:monospace; font-size:13px;">\w</span> <span style="font-size: 14px;">matches any word character (letter, digit, underscore)</span>
* <span style="font-family:monospace; font-size:13px;">\W</span> <span style="font-size: 14px;">matches any non-word character</span>
* <span style="font-family:monospace; font-size:13px;">\s</span> <span style="font-size: 14px;">matches any whitespace character</span>

<span style="font-size: 14px;">By alternating between "one or more word characters" and "a single non-word, non-whitespace character," we capture both word tokens and individual punctuation marks in a single pass through the string.</span>

---

## <span style="font-size: 16px;">Edge Cases</span>

* <span style="font-size: 14px;">**Empty or whitespace-only input:** should return an empty list</span>
* <span style="font-size: 14px;">**Contractions:** "I'm" tokenizes as ["I", "'", "m"] since the apostrophe is punctuation</span>
* <span style="font-size: 14px;">**Hyphenated words:** "state-of-the-art" becomes ["state", "-", "of", "-", "the", "-", "art"]</span>
* <span style="font-size: 14px;">**Multiple punctuation:** "..." becomes [".", ".", "."] since each is a separate token</span>
* <span style="font-size: 14px;">**Numbers with punctuation:** "$9.99" becomes ["$", "9", ".", "99"]</span>

---

## <span style="font-size: 16px;">Tokenization Hierarchy</span>

<span style="font-size: 14px;">Modern NLP uses multiple levels of tokenization, each building on the previous:</span>

* <span style="font-size: 14px;">**Word-level** (this problem): splits on whitespace and punctuation. Simple, but creates large vocabularies and cannot handle out-of-vocabulary words</span>
* <span style="font-size: 14px;">**Subword-level** (BPE, WordPiece, Unigram): splits rare words into frequent subword pieces. Balances vocabulary size with coverage. Used by BERT, GPT, T5</span>
* <span style="font-size: 14px;">**Character-level:** each character is a token. Tiny vocabulary but very long sequences, making it impractical for most models</span>

<span style="font-size: 14px;">Word-level tokenization remains important as the conceptual foundation and as a preprocessing step before subword tokenization in many pipelines.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why not just split on whitespace?** Punctuation stays attached to words, creating spurious vocabulary entries like "hello," and "hello" as different tokens. This inflates vocabulary size and fragments the embedding space</span>
* <span style="font-size: 14px;">**How would you handle contractions like "don't"?** In this simple tokenizer, "don't" becomes ["don", "'", "t"]. More sophisticated tokenizers (like NLTK's TreebankWordTokenizer) use rules to produce ["do", "n't"]. The choice depends on downstream task requirements</span>
* <span style="font-size: 14px;">**What is the time complexity?** A regex-based tokenizer runs in</span> $O(n)$ <span style="font-size: 14px;">where</span> $n$ <span style="font-size: 14px;">is the length of the input string, since the regex engine scans each character at most a constant number of times</span>
* <span style="font-size: 14px;">**Why do modern LLMs use subword tokenization instead?** Word-level tokenization creates a fixed vocabulary that cannot handle unseen words. Subword methods like BPE decompose rare words into known subword units, giving a compact vocabulary with full coverage. This is covered in the BPE Algorithm problem</span>

---