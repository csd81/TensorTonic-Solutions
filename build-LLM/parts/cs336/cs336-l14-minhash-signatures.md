# Build Deterministic MinHash Signatures

Exact deduplication misses documents that are almost the same, such as a page with a different footer or a story with one sentence changed. MinHash finds these near-duplicates by turning each document into a short signature whose agreement with another signature estimates how much the two documents overlap. The heart of this problem is building those signatures from token shingles with a fixed hash recipe and comparing them to estimate similarity.

## Measuring overlap with Jaccard similarity

A document can be described by its set of shingles, the consecutive windows of $k$ tokens. With $k = 2$, the tokens "the cat sat down" give the shingles "the cat", "cat sat" and "sat down". Repeated shingles count once, since this is a set.

The Jaccard similarity of two sets is the size of their intersection divided by the size of their union. Two documents that share most of their shingles have a similarity close to 1, and unrelated documents have a similarity close to 0. Near-duplicate detection usually flags pairs above a threshold such as 0.8.

Computing exact Jaccard similarity for every pair of documents in a web-scale corpus is impossible, because the number of pairs grows with the square of the corpus size. MinHash makes each document's summary small and fixed in size. Combined with locality-sensitive hashing, which groups documents whose signatures agree on a block of columns, it lets similar documents be found without comparing every pair.

## The MinHash trick

Pick a hash function and apply it to every shingle of a document, then keep the smallest hash value. Now consider two documents together. A random hash function is equally likely to give any shingle in their union the smallest value. The two documents end up with the same minimum exactly when that winning shingle is in both sets, so the chance of a match is the intersection size divided by the union size, which is the Jaccard similarity. This result is due to Broder (1997).

One hash function gives a yes-or-no match, which is a very noisy estimate. Using $H$ independent hash functions gives a signature of $H$ minimums per document, and the fraction of matching positions estimates the similarity:

$$
\widehat{J}(i, j) = \frac{1}{H} \sum_{h=0}^{H-1} \mathbf{1}[s_{i,h} = s_{j,h}]
$$

where $s_{i,h}$ is document $i$'s minimum in hash column $h$ and $\mathbf{1}$ is 1 when its condition holds. The estimate's error shrinks like $1 / \sqrt{H}$, so a few hundred hash functions give an accuracy of a few percent.

## Why the hash recipe is fixed

Python's built-in hash is salted differently in every process, so two runs would produce different signatures. A cryptographic hash such as SHA-256 over a fixed byte encoding gives the same value on every machine. Each shingle is written as a compact JSON array of its tokens, and the hashed text for column $h$ is the seed, a colon, the column number, another colon and the encoded shingle. Putting the column number into the hashed text gives each column its own, effectively independent hash function from the same recipe.

Empty documents have no shingles to estimate from, so their similarities are fixed by convention instead.

## A toy trace, then the real hashes

Take the documents "the cat sat down" and "the cat sat up" with shingle size 2. They share "the cat" and "cat sat", and each has one shingle the other lacks, so the union has four shingles and the exact Jaccard similarity is $2 / 4 = 0.5$.

To see the mechanism, imagine two hash columns with made-up values. In the first column the four shingles hash to "the cat" 7, "cat sat" 3, "sat down" 9 and "sat up" 5. The smallest value overall, 3, belongs to the shared shingle "cat sat", so both documents get 3 and the column matches. In the second column suppose "sat down" gets the smallest value, 1. The first document's minimum is 1, while the second document does not contain "sat down" and gets its own minimum from another shingle, so the column does not match. Two columns, one match, an estimate of 0.5.

With the real SHA-256 recipe, 8 hash columns and seed 1, the two signatures match in 4 of the 8 columns, an estimate of exactly 0.5. With 256 columns the estimate is about 0.523. Small signatures can land on the true value by luck, while larger ones are reliably close to it.

## Common mistakes to avoid

- **Counting repeated shingles.** Shingles form a set, so duplicates within a document count once.
- **Building shingles from characters instead of tokens.** The sets change, and so does every similarity.
- **Changing the hashed text.** Leaving out the colons, adding spaces to the JSON or escaping Unicode changes every hash value.
- **Taking the maximum hash.** Each signature entry is the smallest hash value over the document's shingles.
- **Comparing minimum values numerically.** Only exact equality in each column matters, and a closer number means nothing.
