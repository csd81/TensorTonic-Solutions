# <span style="font-size: 20px;">Named Entity Recognition</span>

<span style="font-size: 14px;">Named Entity Recognition (NER) identifies and classifies named entities in text into predefined categories such as PERSON, ORGANIZATION, LOCATION, and DATE. It is a token-level classification task: each token in a sentence receives a tag indicating whether it is part of a named entity and, if so, what type. NER is foundational for information extraction, question answering, and knowledge graph construction.</span>

<span style="font-size: 14px;">This exercise uses case-sensitive dictionary lookup for each token, preserves the supplied tag strings, and assigns O to absent entries. It does not match phrases or add BIO prefixes; the methods below are optional background beyond this baseline.</span>

---

## <span style="font-size: 16px;">Optional Background: BIO Tagging Scheme</span>

<span style="font-size: 14px;">The BIO (Beginning-Inside-Outside) scheme encodes entity boundaries:</span>

* <span style="font-size: 14px;">**B-TYPE**: Beginning of an entity of the given type (e.g., B-PER, B-ORG)</span>
* <span style="font-size: 14px;">**I-TYPE**: Inside (continuation of) an entity of the given type</span>
* <span style="font-size: 14px;">**O**: Outside any entity</span>

<span style="font-size: 14px;">Example: "Barack Obama visited New York"</span>

| <span style="font-size: 14px;">Token</span> | <span style="font-size: 14px;">Tag</span> |
|---|---|
| <span style="font-size: 14px;">Barack</span> | <span style="font-size: 14px;">B-PER</span> |
| <span style="font-size: 14px;">Obama</span> | <span style="font-size: 14px;">I-PER</span> |
| <span style="font-size: 14px;">visited</span> | <span style="font-size: 14px;">O</span> |
| <span style="font-size: 14px;">New</span> | <span style="font-size: 14px;">B-LOC</span> |
| <span style="font-size: 14px;">York</span> | <span style="font-size: 14px;">I-LOC</span> |

---

## <span style="font-size: 16px;">Feature-Based NER</span>

<span style="font-size: 14px;">A classic approach to NER extracts features for each token and uses them for classification. Common features include:</span>

* <span style="font-size: 14px;">**Word identity**: The token itself (lowercased)</span>
* <span style="font-size: 14px;">**Capitalization**: Whether the token starts with an uppercase letter</span>
* <span style="font-size: 14px;">**All caps**: Whether the entire token is uppercase (e.g., "USA")</span>
* <span style="font-size: 14px;">**Contains digit**: Whether the token contains any digit character</span>
* <span style="font-size: 14px;">**Word shape**: Pattern of character types (e.g., "Obama" has shape "Xxxxx", "U.S." has shape "X.X.")</span>
* <span style="font-size: 14px;">**Prefix/suffix**: First and last 2-3 characters</span>
* <span style="font-size: 14px;">**Context window**: Features of neighboring tokens (previous and next words)</span>

---

## <span style="font-size: 16px;">Window-Based Classification</span>

<span style="font-size: 14px;">A window-based approach classifies each token independently using features from a fixed-size context window. For each token at position $t$, extract features from tokens in the window $[t-w, t+w]$ and predict the BIO tag. The window captures local context without modeling the full sequence.</span>

<span style="font-size: 14px;">An optional contextual extension can combine a gazetteer (dictionary of known entities and their types) with context and capitalization features to handle multi-word entities and disambiguation.</span>

---

## <span style="font-size: 16px;">Gazetteer-Based NER</span>

<span style="font-size: 14px;">A gazetteer is a dictionary mapping known entity phrases to their types. A phrase-matching extension performs longest-match lookup:</span>

* <span style="font-size: 14px;">At each position, try to match the longest sequence of tokens that appears in the gazetteer</span>
* <span style="font-size: 14px;">If a match is found, tag the first token as B-TYPE and subsequent tokens as I-TYPE</span>
* <span style="font-size: 14px;">If no match is found, tag the token as O</span>
* <span style="font-size: 14px;">Continue from the position after the matched entity</span>

<span style="font-size: 14px;">This greedy approach is simple but effective for closed-domain NER where entities are known in advance.</span>

---

## <span style="font-size: 16px;">Entity Extraction from BIO Tags</span>

<span style="font-size: 14px;">To extract entities from a BIO-tagged sequence:</span>

* <span style="font-size: 14px;">A B-TYPE tag starts a new entity</span>
* <span style="font-size: 14px;">Following I-TYPE tags (of the same type) extend the entity</span>
* <span style="font-size: 14px;">An O tag, a B tag, or an I tag of a different type ends the current entity</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**BIO vs BIOES?** BIOES adds S (singleton entity) and E (end of entity) tags. It provides more precise boundary information and typically improves model performance by distinguishing single-token entities from the start of multi-token entities</span>
* <span style="font-size: 14px;">**Why not just classify each token independently?** Independent classification ignores dependencies between adjacent tags. For example, I-PER should never follow B-ORG. Sequence models like CRFs or BiLSTM-CRF enforce such constraints</span>
* <span style="font-size: 14px;">**Gazetteer vs learned NER?** Gazetteers are precise for known entities but cannot generalize to unseen ones. Learned models (BiLSTM-CRF, BERT) generalize based on context and morphological features but may hallucinate entities. Production systems often combine both</span>

---