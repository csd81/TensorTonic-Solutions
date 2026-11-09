# <span style="font-size: 20px;">Text Normalization</span>

<span style="font-size: 14px;">Text normalization is the process of transforming raw text into a cleaner, more consistent form. It is typically the first step in any NLP pipeline, applied before tokenization, embedding, or model inference.</span>

---

## <span style="font-size: 16px;">Why Normalize Text?</span>

* <span style="font-size: 14px;">Raw text is messy: mixed case, extra whitespace, punctuation, numbers, Unicode variants</span>
* <span style="font-size: 14px;">Models treat "Hello" and "hello" as different tokens unless case is normalized. This inflates the vocabulary: for a vocabulary of $|V|$ word types, case variants can multiply the effective size by a factor of 2-3x</span>
* <span style="font-size: 14px;">Punctuation and digits add noise for tasks like topic classification where they carry little signal</span>
* <span style="font-size: 14px;">Inconsistent whitespace breaks tokenizers and downstream feature extraction</span>
* <span style="font-size: 14px;">Normalization reduces vocabulary size and improves model generalization</span>

---

## <span style="font-size: 16px;">Common Normalization Operations</span>

### <span style="font-size: 14px;">Lowercasing</span>

<span style="font-size: 14px;">The simplest and most universal operation. Converting to lowercase maps every character $c$ to $\text{lower}(c)$, reducing vocabulary size significantly - "The", "the", "THE" all become one token. However, lowercasing destroys information: "US" (country) becomes indistinguishable from "us" (pronoun).</span>

### <span style="font-size: 14px;">Punctuation Removal</span>

<span style="font-size: 14px;">Removes non-alphanumeric, non-whitespace characters. The regex pattern</span> <span style="font-family:monospace; font-size:13px;">[^\w\s]|_</span> <span style="font-size: 14px;">matches any character $c$ where $c \notin \{\text{word characters}\} \cup \{\text{whitespace}\}$, or an underscore. This removes periods, commas, exclamation marks, special symbols, and underscores, even though underscores are word characters in regex. The substitution replaces each match with the empty string $\varepsilon$.</span>

### <span style="font-size: 14px;">Digit Removal</span>

<span style="font-size: 14px;">Strips all numeric characters matching the pattern</span> <span style="font-family:monospace; font-size:13px;">\d</span> <span style="font-size: 14px;">(Unicode decimal digits, including non-ASCII digits but excluding superscript digits). Useful when numbers are irrelevant to the task (e.g., sentiment analysis). Some pipelines replace digits with a placeholder token like `<NUM>` instead of removing them entirely, preserving the information that a number was present.</span>

### <span style="font-size: 14px;">Whitespace Collapsing</span>

<span style="font-size: 14px;">Replaces runs of one or more whitespace characters with a single space. The regex</span> <span style="font-family:monospace; font-size:13px;">\s+</span> <span style="font-size: 14px;">matches the pattern $s^+$ where $s \in \{\text{space}, \text{tab}, \text{newline}, \dots\}$. This is essential after punctuation or digit removal, which can leave gaps in the text.</span>

### <span style="font-size: 14px;">Stripping</span>

<span style="font-size: 14px;">Removes leading and trailing whitespace. A final cleanup step to ensure the output has no trailing spaces. Formally, for string $s = w_1 \cdot s_{\text{core}} \cdot w_2$ where $w_1, w_2$ are whitespace-only prefixes and suffixes, stripping returns $s_{\text{core}}$.</span>

---

## <span style="font-size: 16px;">Order of Operations</span>

<span style="font-size: 14px;">The order in which operations are applied matters significantly. Given a pipeline of operations $f_1, f_2, \dots, f_k$, the result is the composition $f_k \circ \dots \circ f_2 \circ f_1$, and in general $f_i \circ f_j \neq f_j \circ f_i$:</span>

* <span style="font-size: 14px;">Removing punctuation before collapsing whitespace avoids leftover double spaces where punctuation was</span>
* <span style="font-size: 14px;">Lowercasing before punctuation removal is usually safe, but not always - some Unicode characters change category when case-folded</span>
* <span style="font-size: 14px;">Stripping should typically be the last step to clean up any trailing whitespace introduced by earlier operations</span>

---

## <span style="font-size: 16px;">Advanced Normalization</span>

<span style="font-size: 14px;">Beyond the basic operations in this problem, production pipelines often include:</span>

* <span style="font-size: 14px;">**Unicode normalization** (NFC/NFKD): Converts equivalent Unicode representations to a canonical form. For example, the character "e" ($\text{U+0065}$) + combining acute accent ($\text{U+0301}$) maps to the single codepoint "e" ($\text{U+00E9}$) under NFC</span>
* <span style="font-size: 14px;">**Accent/diacritic removal**: Strips accents so "cafe" matches "cafe". Done via NFKD decomposition followed by removing combining characters (Unicode category $M$)</span>
* <span style="font-size: 14px;">**HTML entity decoding**: Converts "&amp;" to "&", "&lt;" to "<"</span>
* <span style="font-size: 14px;">**Emoji handling**: Remove, replace with text descriptions, or keep as features</span>

---

## <span style="font-size: 16px;">When Not to Normalize</span>

* <span style="font-size: 14px;">**Named Entity Recognition:** Case carries signal - "Apple" (company) vs "apple" (fruit)</span>
* <span style="font-size: 14px;">**Sentiment Analysis:** Punctuation can carry meaning - "great" vs "great!!!"</span>
* <span style="font-size: 14px;">**Machine Translation:** All surface forms matter for faithful translation</span>
* <span style="font-size: 14px;">**Modern transformer models:** BERT and GPT handle raw text well and often perform better without aggressive normalization</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why does the order of normalization steps matter?** Removing punctuation before collapsing whitespace avoids leftover double spaces. Lowercasing before punctuation removal is generally safe, but some Unicode characters change category when case-folded, so order can affect correctness</span>
* <span style="font-size: 14px;">**When should you skip normalization?** For named entity recognition (case carries signal), sentiment analysis (punctuation like "!!!" carries meaning), and modern transformer models that handle raw text well. Over-normalizing can destroy useful signal</span>
* <span style="font-size: 14px;">**What is Unicode normalization and why does it matter?** Unicode allows multiple representations of the same character (e.g., accented "e" can be one codepoint or two). NFC/NFKD normalization converts to a canonical form so string comparison works correctly across different encodings</span>
* <span style="font-size: 14px;">**How does text normalization differ for search engines vs language models?** Search engines aggressively normalize (lowercase, remove accents, stem) to maximize recall. Language models often prefer minimal normalization to preserve the full signal in the text</span>

---
