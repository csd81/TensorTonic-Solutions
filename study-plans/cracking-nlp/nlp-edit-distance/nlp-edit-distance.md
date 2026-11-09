# <span style="font-size: 20px;">Edit Distance (Levenshtein Distance)</span>

<span style="font-size: 14px;">Edit distance measures the minimum number of single-character operations needed to transform one string into another. It is one of the most fundamental algorithms in NLP and computational biology.</span>

---

## <span style="font-size: 16px;">The Three Operations</span>

<span style="font-size: 14px;">The Levenshtein edit distance allows exactly three operations, each costing 1:</span>

* <span style="font-size: 14px;">**Insert** a character into the source string</span>
* <span style="font-size: 14px;">**Delete** a character from the source string</span>
* <span style="font-size: 14px;">**Substitute** one character for another in the source string</span>

<span style="font-size: 14px;">For example, transforming</span> `"kitten"` <span style="font-size: 14px;">into</span> `"sitting"` <span style="font-size: 14px;">requires 3 edits: substitute k->s, substitute e->i, insert g.</span>

---

## <span style="font-size: 16px;">Dynamic Programming Formulation</span>

<span style="font-size: 14px;">Let</span> `source` <span style="font-size: 14px;">have length m and</span> `target` <span style="font-size: 14px;">have length n. Build a matrix</span> `dp` <span style="font-size: 14px;">of size (m+1) x (n+1) where</span> `dp[i][j]` <span style="font-size: 14px;">is the minimum edit distance between</span> `source[:i]` <span style="font-size: 14px;">and</span> `target[:j]`<span style="font-size: 14px;">.</span>

### <span style="font-size: 14px;">Base Cases</span>

* `dp[i][0] = i` <span style="font-size: 14px;">- deleting all i characters from source prefix</span>
* `dp[0][j] = j` <span style="font-size: 14px;">- inserting all j characters of target prefix</span>

### <span style="font-size: 14px;">Recurrence</span>

<span style="font-size: 14px;">For each cell</span> `dp[i][j]`<span style="font-size: 14px;">:</span>

* <span style="font-size: 14px;">If</span> `source[i-1] == target[j-1]`<span style="font-size: 14px;">: characters match, no edit needed:</span> `dp[i][j] = dp[i-1][j-1]`
* <span style="font-size: 14px;">Otherwise, take the minimum of:</span>
  * `dp[i-1][j] + 1` <span style="font-size: 14px;">(delete from source)</span>
  * `dp[i][j-1] + 1` <span style="font-size: 14px;">(insert into source)</span>
  * `dp[i-1][j-1] + 1` <span style="font-size: 14px;">(substitute)</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Transform</span> `"cat"` <span style="font-size: 14px;">into</span> `"car"`<span style="font-size: 14px;">:</span>

```
        ""    c     a     r
  ""  [  0    1     2     3 ]
   c  [  1    0     1     2 ]
   a  [  2    1     0     1 ]
   t  [  3    2     1     1 ]
```

* `dp[1][1] = 0` <span style="font-size: 14px;">: c == c, no edit</span>
* `dp[2][2] = 0` <span style="font-size: 14px;">: a == a, no edit</span>
* `dp[3][3] = 1` <span style="font-size: 14px;">: t != r, min(dp[2][3]+1, dp[3][2]+1, dp[2][2]+1) = min(2, 2, 1) = 1</span>

<span style="font-size: 14px;">Result: edit distance is 1 (substitute t->r).</span>

---

## <span style="font-size: 16px;">Complexity</span>

* <span style="font-size: 14px;">**Time:** O(m * n) where m and n are the lengths of the two strings</span>
* <span style="font-size: 14px;">**Space:** O(m * n) for the full DP matrix</span>

---

## <span style="font-size: 16px;">Space Optimization</span>

<span style="font-size: 14px;">Since each row of the DP matrix only depends on the previous row, you can reduce space to O(min(m, n)) by keeping only two rows (or even a single row with a temporary variable). Iterate over the shorter string as columns to minimize the row width:</span>

* <span style="font-size: 14px;">Maintain</span> `prev_row` <span style="font-size: 14px;">and</span> `curr_row` <span style="font-size: 14px;">of length min(m, n) + 1</span>
* <span style="font-size: 14px;">After processing each row, swap</span> `prev_row = curr_row`

---

## <span style="font-size: 16px;">Applications in NLP</span>

* <span style="font-size: 14px;">**Spell correction:** Suggest corrections by finding dictionary words with small edit distance to the misspelled word</span>
* <span style="font-size: 14px;">**Fuzzy matching:** Match user queries to database entries even when they contain typos (e.g., "Jhon" matching "John")</span>
* <span style="font-size: 14px;">**Entity resolution:** Determine if two entity mentions refer to the same real-world entity despite spelling variations</span>
* <span style="font-size: 14px;">**DNA sequence alignment:** In bioinformatics, edit distance measures similarity between genetic sequences (the Needleman-Wunsch algorithm is a generalization)</span>
* <span style="font-size: 14px;">**Machine translation evaluation:** Metrics like TER (Translation Edit Rate) use edit distance at the word level to evaluate translation quality</span>

---

## <span style="font-size: 16px;">Variants</span>

* <span style="font-size: 14px;">**Damerau-Levenshtein:** Also allows transposition of two adjacent characters as a single edit (cost 1)</span>
* <span style="font-size: 14px;">**Weighted edit distance:** Assign different costs to different operations or character substitutions (e.g., substituting 'a' for 'e' costs less than 'a' for 'z')</span>
* <span style="font-size: 14px;">**Hamming distance:** Only counts substitutions; requires strings of equal length</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Can you compute edit distance in less than O(mn) space?** Yes - since each row only depends on the previous row, you can use two rows of size min(m,n)+1, reducing space from O(mn) to O(min(m,n)). You cannot recover the edit operations without the full matrix though</span>
* <span style="font-size: 14px;">**How is edit distance used in spell correction?** Given a misspelled word, compute edit distance to all dictionary words and suggest the closest matches. In practice, you generate all strings within edit distance 1 or 2 and check which ones are valid dictionary words, which is much faster than comparing against every word</span>
* <span style="font-size: 14px;">**What is the difference between Levenshtein and Damerau-Levenshtein distance?** Damerau-Levenshtein also allows transposition of two adjacent characters as a single edit operation. This is useful because transpositions account for a large fraction of real typos (e.g., "teh" for "the")</span>
* <span style="font-size: 14px;">**How would you use edit distance at the word level?** Replace characters with words: insertion/deletion of a word costs 1, substitution costs 1 (or uses word similarity). This is the basis of the TER (Translation Edit Rate) metric for evaluating machine translation quality</span>

---