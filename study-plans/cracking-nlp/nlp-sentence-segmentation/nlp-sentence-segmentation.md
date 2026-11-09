# <span style="font-size: 20px;">Sentence Segmentation</span>

## <span style="font-size: 16px;">What Is Sentence Segmentation?</span>

<span style="font-size: 14px;">Sentence segmentation (also called sentence boundary detection or sentence splitting) is the task of dividing a text into individual sentences. While it sounds simple, it is one of the trickiest low-level NLP tasks because punctuation marks that typically end sentences can also appear in other contexts.</span>

---

## <span style="font-size: 16px;">Why It Is Hard</span>

<span style="font-size: 14px;">The period character "." is the most ambiguous punctuation mark in English. It can serve as:</span>

<span style="font-size: 14px;">1. **Sentence terminator** - "The cat sat down. It was tired."</span>

<span style="font-size: 14px;">2. **Abbreviation marker** - "Dr. Smith", "U.S. Army", "etc."</span>

<span style="font-size: 14px;">3. **Decimal point** - "The value is 3.14."</span>

<span style="font-size: 14px;">4. **Ellipsis** - "He thought... and then left."</span>

<span style="font-size: 14px;">5. **Part of URLs or emails** - "Visit www.example.com for details."</span>

<span style="font-size: 14px;">Question marks and exclamation marks are less ambiguous but can still appear inside quotations or in rhetorical constructions where they do not mark a boundary.</span>

---

## <span style="font-size: 16px;">Rule-Based Approaches</span>

<span style="font-size: 14px;">The simplest approach uses handcrafted rules:</span>

<span style="font-size: 14px;">1. **Abbreviation lists** - maintain a set of known abbreviations like `"Mr"`, `"Dr"`, `"U.S"` and skip periods that follow them.</span>

<span style="font-size: 14px;">2. **Contextual patterns** - a period followed by a space and an uppercase letter likely marks a sentence boundary. A period followed by a lowercase letter likely does not.</span>

<span style="font-size: 14px;">3. **Regular expressions** - use regex patterns to identify sentence-ending punctuation while excluding known non-boundary patterns.</span>

<span style="font-size: 14px;">Rule-based systems are fast and interpretable but brittle. They fail on edge cases not covered by the rules and need manual maintenance of abbreviation lists for each domain.</span>

---

## <span style="font-size: 16px;">Machine Learning Approaches</span>

<span style="font-size: 14px;">Modern NLP tools often use supervised ML models trained on annotated corpora:</span>

<span style="font-size: 14px;">- **Punkt tokenizer (NLTK)** - an unsupervised algorithm by Kiss and Strunk (2006) that learns abbreviations, collocations, and sentence starters from raw text. It uses log-likelihood ratios to distinguish abbreviation periods from sentence boundaries.</span>

<span style="font-size: 14px;">- **spaCy's sentencizer** - uses a dependency parser or a trained sentence segmentation model. The parser-based approach identifies sentence roots in the dependency tree, which implicitly determines boundaries. spaCy also provides a rule-based `Sentencizer` component for simpler use cases.</span>

<span style="font-size: 14px;">- **Neural approaches** - transformer-based models can be fine-tuned on sentence boundary detection as a token classification task, achieving high accuracy even on noisy text like social media posts.</span>

---

## <span style="font-size: 16px;">Common Abbreviation Handling</span>

<span style="font-size: 14px;">The core challenge in rule-based segmentation is abbreviation handling. Common strategies include:</span>

<span style="font-size: 14px;">1. **Curated lists** - keep a set of abbreviations (`"Mr"`, `"Mrs"`, `"Dr"`, `"Jr"`, `"Sr"`, `"vs"`, `"etc"`, `"i.e"`, `"e.g"`, `"U.S"`). When a period is found, check if the preceding word matches an abbreviation.</span>

<span style="font-size: 14px;">2. **Heuristic patterns** - single uppercase letters followed by a period (e.g., "J. K. Rowling") are likely initials, not sentence boundaries.</span>

<span style="font-size: 14px;">3. **Dual-role detection** - some abbreviations can also end a sentence (e.g., "He lives in the U.S." at end of text). In such cases, the period serves double duty as both abbreviation marker and sentence terminator.</span>

---

## <span style="font-size: 16px;">Evaluation Metrics</span>

<span style="font-size: 14px;">Sentence segmentation is typically evaluated using:</span>

<span style="font-size: 14px;">- **Precision** - of all predicted boundaries, how many are correct?</span>

<span style="font-size: 14px;">- **Recall** - of all true boundaries, how many were detected?</span>

<span style="font-size: 14px;">- **F1 score** - harmonic mean of precision and recall.</span>

<span style="font-size: 14px;">- **Error rate** - percentage of incorrectly placed or missed boundaries.</span>

<span style="font-size: 14px;">On clean, well-formatted English text, rule-based systems can achieve F1 scores above 95%. On noisy text (social media, OCR output, legal documents), ML-based approaches significantly outperform rule-based ones.</span>

---

## <span style="font-size: 16px;">Practical Importance</span>

<span style="font-size: 14px;">Sentence segmentation is a critical preprocessing step for many NLP tasks. Machine translation, text summarization, sentiment analysis, and named entity recognition all operate at the sentence level. Errors in sentence segmentation propagate downstream and degrade the quality of all subsequent processing.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**How do you handle abbreviations like "Dr.", "U.S.", "e.g."?** Maintain a curated set of known abbreviations and check if a period belongs to one before treating it as a sentence boundary. Some abbreviations at the end of a sentence serve double duty as both abbreviation marker and terminator</span>
* <span style="font-size: 14px;">**What makes sentence segmentation hard on noisy text?** Social media text has inconsistent punctuation, missing periods, run-on sentences, and non-standard abbreviations. OCR output introduces spurious periods and removes legitimate ones. ML-based approaches significantly outperform rule-based methods on noisy text</span>
* <span style="font-size: 14px;">**Why is sentence segmentation important for downstream NLP tasks?** Machine translation, summarization, sentiment analysis, and NER all operate at the sentence level. Errors in segmentation propagate downstream - splitting a sentence too early can break coreference chains, and failing to split can exceed model context windows</span>
* <span style="font-size: 14px;">**How do modern NLP systems handle sentence boundaries?** Many transformer models process text as continuous token sequences without explicit sentence boundaries. However, BERT uses [SEP] tokens between sentences, and most training pipelines still segment text into sentences for creating training examples</span>

---