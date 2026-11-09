# <span style="font-size: 20px;">Byte Pair Encoding</span>

<span style="font-size: 14px;">Byte Pair Encoding (BPE) builds larger symbols by repeatedly merging a frequent adjacent pair. This exercise learns an ordered list of merges from words and their frequencies. The frequencies count how many times each word occurs, so the algorithm can work on the compact dictionary without expanding the corpus.</span>

---

## <span style="font-size: 16px;">Initial Symbols and Word Boundaries</span>

<span style="font-size: 14px;">Split each word into characters and attach the end-of-word marker &lt;/w&gt; to its final character. For example, low starts as ["l", "o", "w&lt;/w&gt;"]. The marker belongs to the final symbol and remains attached when that symbol is merged. Pairs never cross word boundaries.</span>

## <span style="font-size: 16px;">Frequency-Weighted Pair Counts</span>

$$
C(a,b)=\sum_{w\in W} f(w)\sum_{j=0}^{m_w-2}\mathbf{1}[s_{w,j}=a\text{ and }s_{w,j+1}=b]
$$

<span style="font-size: 14px;">W is the set of input words, f(w) is a word's supplied frequency, and the sequence s contains its current symbols. The sequence length is m. C(a,b) counts adjacent occurrences of symbols a and b, weighted by word frequency. The indicator contributes one when both neighboring symbols match.</span>

<span style="font-size: 14px;">Count every occurrence, including overlaps. Select the pair with the largest count. For tied counts, compare the first symbol alphabetically, then the second symbol. This tie rule makes the learned merge order deterministic.</span>

## <span style="font-size: 16px;">Applying a Merge</span>

<span style="font-size: 14px;">Replace the chosen pair by its concatenation in every word. Scan each word from left to right and consume both symbols when a match is replaced. Replacements within one pass are non-overlapping. Recount pairs after each merge because the new symbols change which adjacent pairs exist.</span>

<span style="font-size: 14px;">For example, the word aaaa initially has symbols ["a", "a", "a", "a&lt;/w&gt;"]. The pair ("a", "a") occurs twice when counting overlaps, but one left-to-right replacement pass produces ["aa", "a", "a&lt;/w&gt;"]. Counting and replacement therefore use different overlap rules.</span>

## <span style="font-size: 16px;">A Small Worked Example</span>

<span style="font-size: 14px;">With frequencies {"low": 5, "lower": 2}, the initial splits are ["l", "o", "w&lt;/w&gt;"] and ["l", "o", "w", "e", "r&lt;/w&gt;"]. The pair ("l", "o") has weighted count 7, so it merges first. The resulting pair ("lo", "w&lt;/w&gt;") has count 5 and merges next.</span>

<span style="font-size: 14px;">After two merges, the ordered merge list is [["l", "o"], ["lo", "w&lt;/w&gt;"]]. The symbols remaining in the two word splits form the sorted vocabulary ["e", "lo", "low&lt;/w&gt;", "r&lt;/w&gt;", "w"].</span>

## <span style="font-size: 16px;">Stopping and Returned Vocabulary</span>

<span style="font-size: 14px;">Stop after num_merges merge steps, or earlier when no adjacent pair remains. With zero merges, return an empty merge list and the sorted initial symbol set. A one-character word is already a single symbol containing its end marker.</span>

<span style="font-size: 14px;">Return a dictionary containing merges as ordered two-string lists and vocab as the sorted unique symbols still present in the final word splits. Removed symbols are excluded from this exercise's returned vocabulary. Keep the merge order because later merges depend on symbols created by earlier ones.</span>

## <span style="font-size: 16px;">Common Implementation Errors</span>

- <span style="font-size: 14px;">Counting a pair only once per word loses repeated occurrences and changes its frequency.</span>
- <span style="font-size: 14px;">Adding &lt;/w&gt; as a separate initial symbol changes the available pairs.</span>
- <span style="font-size: 14px;">Ignoring the tie rule can produce a different valid-looking merge sequence.</span>
- <span style="font-size: 14px;">Collecting every historical symbol produces a cumulative vocabulary instead of the requested final symbol set.</span>
