# Perform Stable Exact Deduplication

Web-scale training data is full of repeated documents, from mirrored pages to copied boilerplate, and training on duplicates wastes compute and encourages memorization. Exact deduplication removes documents whose text is the same after normalization. The heart of this problem is normalizing text consistently, bucketing it with a hash for speed, and still deciding duplicates by exact text equality so the earliest copy is kept.

## Why remove duplicates

A document that appears a thousand times in a training set gets a thousand times the weight of a unique one. The model spends capacity memorizing it, which increases the chance of reproducing it word for word, and evaluation data that leaked into the training set becomes much more likely to be memorized. Lee et al. (2022) found that deduplicating training data cut memorized output sharply and reached the same accuracy in fewer training steps.

## Normalization decides what counts as the same

Two pages can differ in ways no reader would care about, such as letter case, extra spaces or different Unicode encodings of the same character. Normalization maps those variants to one canonical string before comparison.

Unicode NFKC normalization comes first. It composes characters, so an "e" followed by a combining accent becomes the single character "é", and it replaces compatibility characters with their ordinary forms, for example the full-width letter "Ａ" with "A" and the ligature "ﬁ" with the two letters "fi". Optional case folding then maps text to a caseless form using Unicode's language-independent rules, which cover more cases than plain lowercasing. Optional whitespace collapsing trims the ends and replaces each run of whitespace with a single space.

The order matters. NFKC turns a full-width "Ａ" into an ordinary uppercase "A", and that letter only matches a lowercase "a" if case folding runs afterward. Each step removes a kind of difference, so turning one off can only keep more documents as distinct, never fewer.

## Hash buckets for speed

Comparing every document with every other is far too slow for billions of documents. Instead, each normalized text is hashed with SHA-256, and the low-order bits of the digest, the last few bits read as a number, form a bucket key. Only documents in the same bucket can be duplicates, so each new document is compared with a handful of earlier texts instead of all of them.

A shorter key means more collisions, where different texts land in the same bucket. A collision must never cause a deletion. Inside a bucket, the actual normalized texts are compared, and only an exact match counts as a duplicate. Bucketing limits how many texts must be compared, and the exact comparison inside the bucket guarantees that no two different texts are merged.

## Keeping the earliest copy

Documents are processed in their original order. The first time a normalized text appears, that document is retained. Every later document with the same normalized text is removed and mapped to the retained document it duplicates. Processing in order makes the result stable, so the same input always keeps the same documents, and the retained documents stay in their original order.

## Three ways to write a café menu

Take three documents with case folding and whitespace collapsing turned on. Document d1 has the text "Café  menu" with two spaces. Document d2 has "cafe" with a combining acute accent on the final e, a single space, then "MENU". Document d3 has "Cafe menu" with no accent.

NFKC composes d2's accent into "é", case folding lowercases all three, and whitespace collapsing removes d1's extra space. The normalized texts are "café menu", "café menu" and "cafe menu".

Document d1 is kept as the first occurrence. Document d2 matches it exactly after normalization, so it is removed and mapped to d1. Document d3 differs by its unaccented "e" and is kept. Without NFKC, d2 would have survived as a separate document, even though it displays identically to d1.

## Where exact deduplication fits

Exact deduplication is cheap and certain, so it runs early in a data pipeline. It cannot catch pages that differ by a changed date, a different footer or one edited sentence. Near-duplicate methods such as MinHash run afterward to find those, comparing documents by how much of their text overlaps.

## Common mistakes to avoid

- **Treating a hash match as a duplicate.** Collisions are possible, so only exact normalized-text equality removes a document.
- **Normalizing in the wrong order.** Case folding before NFKC misses letters that NFKC turns into uppercase.
- **Using lowercase instead of case folding.** Simple lowercasing misses equivalences that Unicode case folding handles.
- **Keeping the last copy.** The earliest occurrence is retained and later copies point to it.
- **Splitting only on spaces.** Whitespace collapsing must treat every Unicode whitespace character as whitespace.
