# <span style="font-size: 20px;">Stemming vs Lemmatization</span>

## <span style="font-size: 16px;">Overview</span>

<span style="font-size: 14px;">Stemming and lemmatization are two fundamental text normalization techniques in Natural Language Processing (NLP). Both aim to reduce words to a common base form, but they differ significantly in their approach, accuracy, and computational cost.</span>

---

## <span style="font-size: 16px;">Stemming</span>

<span style="font-size: 14px;">Stemming is a rule-based, heuristic process that strips suffixes from words to produce a **stem** - which may or may not be a valid word in the language. The most well-known stemming algorithm is the **Porter Stemmer**, published by Martin Porter in 1980.</span>

### <span style="font-size: 14px;">How the Porter Stemmer Works</span>

<span style="font-size: 14px;">The Porter Stemmer applies a series of suffix-stripping rules in multiple phases. Each phase targets specific types of suffixes:</span>

<span style="font-size: 14px;">**Phase 1:** Handle plural and past-tense suffixes (`-s`, `-ed`, `-ing`)</span>

<span style="font-size: 14px;">**Phase 2:** Handle derivational suffixes (`-ational` to `-ate`, `-tion` to `-te`)</span>

<span style="font-size: 14px;">**Phase 3-5:** Further reduction rules based on the measure (number of consonant-vowel sequences) of the remaining stem.</span>

<span style="font-size: 14px;">A simplified version of these rules might include:</span>

<span style="font-size: 14px;">1. Remove `-ing` if the remaining stem has length >= 3 (e.g., "running" to "runn")</span>

<span style="font-size: 14px;">2. Remove `-ly` if the remaining stem has length >= 3 (e.g., "quickly" to "quick")</span>

<span style="font-size: 14px;">3. Remove `-ed` if the remaining stem has length >= 3 (e.g., "walked" to "walk")</span>

<span style="font-size: 14px;">4. Remove `-s` (but not `-ss`) if the remaining stem has length >= 2 (e.g., "cats" to "cat", but "lass" stays "lass")</span>

<span style="font-size: 14px;">5. Remove `-er` if the remaining stem has length >= 3 (e.g., "faster" to "fast")</span>

<span style="font-size: 14px;">6. Replace `-tion` with `-te` if the resulting stem has length >= 3 (e.g., "creation" to "create")</span>

### <span style="font-size: 14px;">Why Stemming Is Lossy</span>

<span style="font-size: 14px;">Stemming is inherently a lossy operation because it uses crude heuristics rather than linguistic knowledge:</span>

<span style="font-size: 14px;">- **Over-stemming:** Different words collapse to the same stem. For example, "universal" and "university" might both stem to "univers", even though they have different meanings.</span>

<span style="font-size: 14px;">- **Under-stemming:** Related words fail to converge. "Absorb" and "absorption" have different stems because the rules do not capture the morphological relationship.</span>

<span style="font-size: 14px;">- **Invalid stems:** The output may not be a real word (e.g., "running" becomes "runn"), which is acceptable since the stem is only used as a canonical key for grouping related words.</span>

---

## <span style="font-size: 16px;">Lemmatization</span>

<span style="font-size: 14px;">Lemmatization reduces a word to its **lemma** - the dictionary or canonical form of the word. Unlike stemming, the result is always a valid word.</span>

<span style="font-size: 14px;">Examples:</span>

<span style="font-size: 14px;">- "running" lemmatizes to "run"</span>
<span style="font-size: 14px;">- "better" lemmatizes to "good"</span>
<span style="font-size: 14px;">- "mice" lemmatizes to "mouse"</span>
<span style="font-size: 14px;">- "was" lemmatizes to "be"</span>

### <span style="font-size: 14px;">Why Lemmatization Requires Linguistic Knowledge</span>

<span style="font-size: 14px;">Lemmatization cannot be done with simple suffix rules. It requires:</span>

<span style="font-size: 14px;">1. **A lexicon or dictionary** that maps inflected forms to their base forms (e.g., "went" -> "go")</span>

<span style="font-size: 14px;">2. **Part-of-speech (POS) information** to disambiguate words. For example, "better" as an adjective lemmatizes to "good", but "better" as a verb lemmatizes to "better" (meaning to improve).</span>

<span style="font-size: 14px;">3. **Morphological analysis** to handle irregular forms like "mice" -> "mouse" or "geese" -> "goose".</span>

<span style="font-size: 14px;">Popular lemmatizers include WordNet Lemmatizer (NLTK) and spaCy's built-in lemmatizer. Both rely on curated linguistic databases.</span>

---

## <span style="font-size: 16px;">Comparison</span>

<span style="font-size: 14px;">| Aspect | Stemming | Lemmatization |</span>
<span style="font-size: 14px;">|---|---|---|</span>
<span style="font-size: 14px;">| Output | May not be a real word | Always a valid word |</span>
<span style="font-size: 14px;">| Speed | Fast (rule-based) | Slower (dictionary lookup + POS) |</span>
<span style="font-size: 14px;">| Accuracy | Lower (heuristic errors) | Higher (linguistically informed) |</span>
<span style="font-size: 14px;">| Resources | No external data needed | Requires lexicon/dictionary |</span>
<span style="font-size: 14px;">| Language support | Easy to port (new rules) | Hard to port (new dictionary) |</span>

---

## <span style="font-size: 16px;">Use Cases</span>

<span style="font-size: 14px;">**Stemming is preferred when:**</span>

<span style="font-size: 14px;">- Speed matters more than precision (e.g., search engine indexing)</span>
<span style="font-size: 14px;">- The exact base form is not important, only grouping related words</span>
<span style="font-size: 14px;">- Working with languages that lack good lemmatization resources</span>

<span style="font-size: 14px;">**Lemmatization is preferred when:**</span>

<span style="font-size: 14px;">- Semantic correctness matters (e.g., text generation, question answering)</span>
<span style="font-size: 14px;">- You need valid words for downstream processing</span>
<span style="font-size: 14px;">- Part-of-speech information is available</span>

---

## <span style="font-size: 16px;">Modern NLP: Subword Tokenization</span>

<span style="font-size: 14px;">In modern deep learning-based NLP, subword tokenization methods like **Byte-Pair Encoding (BPE)**, **WordPiece**, and **SentencePiece** have largely replaced both stemming and lemmatization as preprocessing steps.</span>

<span style="font-size: 14px;">These methods split words into learned subword units (e.g., "unhappiness" -> ["un", "happiness"] or ["un", "hap", "pi", "ness"]) that capture morphological patterns implicitly through training data statistics. Transformer models like BERT, GPT, and T5 all use subword tokenization.</span>

<span style="font-size: 14px;">Key advantages of subword tokenization:</span>

<span style="font-size: 14px;">- Handles unseen words by decomposing them into known subword pieces</span>
<span style="font-size: 14px;">- Learns language-specific morphology from data rather than hand-crafted rules</span>
<span style="font-size: 14px;">- Eliminates the need for separate stemming or lemmatization pipelines</span>
<span style="font-size: 14px;">- Works across languages without language-specific dictionaries</span>

<span style="font-size: 14px;">However, stemming and lemmatization remain relevant in classical NLP pipelines, information retrieval systems, and scenarios where interpretability of the normalization step is important.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**When would you choose stemming over lemmatization?** Stemming is preferred when speed matters more than precision (e.g., search engine indexing of millions of documents), when exact word forms are unimportant, or when no lemmatization resources exist for the target language</span>
* <span style="font-size: 14px;">**Why have modern LLMs mostly replaced stemming and lemmatization?** Subword tokenization (BPE, WordPiece) learns morphological patterns from data, handles unseen words by decomposing them into known subword units, and works across languages without hand-crafted rules or dictionaries</span>
* <span style="font-size: 14px;">**What are the failure modes of rule-based stemming?** Over-stemming merges unrelated words (e.g., "universe" and "university" both stem to "univers"). Under-stemming fails to merge related words (e.g., "alumnus" and "alumni" produce different stems). These errors are systematic and hard to fix without abandoning the rule-based approach</span>
* <span style="font-size: 14px;">**How does POS tagging improve lemmatization?** The same word form can have different lemmas depending on part of speech: "better" as an adjective lemmatizes to "good", but as a verb it lemmatizes to "better" (meaning to improve). Without POS information, the lemmatizer must guess or use context heuristics</span>

---