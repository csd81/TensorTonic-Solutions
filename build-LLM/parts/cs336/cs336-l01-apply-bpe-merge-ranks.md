# Apply Ranked BPE Merges

Once a BPE vocabulary has been trained, the tokenizer must use it on new text. Encoding replays the learned merge rules on the bytes of the input, and decoding turns token IDs back into a string. The heart of this problem is applying the rules in their ranked order and decoding the bytes as one sequence, so that encoding and decoding stay exact inverses.

## What the merge list stores

Byte-pair encoding (BPE) starts with one token for each of the 256 possible byte values, IDs 0 to 255. Training repeatedly finds the most frequent adjacent pair of tokens and gives it a new ID, starting at 256. The result is an ordered list of rules. Rule $k$ says that wherever token $l_k$ is immediately followed by token $r_k$, the pair becomes the single token $n_k$:

$$
(l_k, r_k) \rightarrow n_k
$$

The position $k$ is the rule's rank. Rank 0 was the most frequent pair in the training corpus, rank 1 was the most frequent pair after that merge, and so on. The rank order is part of the tokenizer, just like the rules themselves.

Each rule can only fire on tokens that already exist. A rule such as $(256, 99) \rightarrow 257$ needs token 256 to have been created by an earlier rule. This is why the rules must be replayed in exactly the order they were learned.

## Encoding replays the rules in rank order

Encoding starts from the UTF-8 bytes of the text, one token per byte, and then applies the rules from rank 0 to the last rank. Each rule replaces every occurrence of its pair before the next rule is considered.

Within one rule, matches never share a token, so a run of identical bytes pairs up from the left and any odd one out stays single. In the sequence $[120, 120, 120, 120, 120]$, the rule $(120, 120) \rightarrow 300$ gives $[300, 300, 120]$, and the last byte is left alone because its neighbour was already used.

Each rule gets exactly one pass. After a rule has been applied, encoding moves to the next rank and never returns to an earlier rule.

## Why the order matters

Take the text "abcab", whose bytes are $[97, 98, 99, 97, 98]$, and the two rules

$$
(97, 98) \rightarrow 256, \qquad (256, 99) \rightarrow 257
$$

The first rule turns both copies of "ab" into token 256, giving $[256, 99, 256]$. The second rule then finds 256 followed by 99 and produces $[257, 256]$. The text "abcab" becomes two tokens, one for "abc" and one for "ab".

Now swap the rules so that $(256, 99) \rightarrow 257$ comes first. Token 256 does not exist yet, so the first rule finds nothing. The second rule then produces $[256, 99, 256]$, which is three tokens instead of two. The same rules in a different order give a different output, so a tokenizer that ignores rank order cannot reproduce the tokens a model was trained on.

## Decoding joins bytes before reading text

The vocabulary maps every token ID to the list of bytes it stands for. Decoding looks up each ID, joins all the byte lists in token order, and only then interprets the complete byte sequence as UTF-8.

The joining step must come before the text conversion because a token does not have to hold a whole character. The character "é" is the two bytes $[195, 169]$ in UTF-8. A vocabulary may contain one token for byte 195 and another for byte 169. Neither byte on its own is valid UTF-8, so converting each token separately would fail or produce replacement characters. Joined together, they form a valid "é".

## A small decoding example

Suppose the vocabulary contains

$$
256 \mapsto [104], \qquad 257 \mapsto [195], \qquad 258 \mapsto [169, 33]
$$

and the IDs are $[256, 257, 258]$. Joining the byte lists gives $[104, 195, 169, 33]$. Read as UTF-8, byte 104 is "h", bytes 195 and 169 together are "é", and byte 33 is "!". The decoded text is "hé!".

Token 257 holds only the first byte of "é" and token 258 starts with its second byte, so neither can be converted to text on its own.

## Where this sits in a language model

Encoding runs on every prompt before the model sees it, and decoding runs on every token the model generates. The model only ever sees token IDs, so any difference between how text was encoded during training and how it is encoded later changes what the model receives. Exact, deterministic encoding is what keeps those two in agreement.

## Common mistakes to avoid

- **Applying rules in a different order.** Reordering the rules, for example by pair value, can break chains where a later rule needs a token made by an earlier one, so always follow the order of the list.
- **Overlapping replacements.** Replacing at every position where a pair starts would let one token join two merges, so a run of three identical bytes would share its middle byte.
- **Decoding token by token.** Converting each token's bytes to text separately breaks characters whose bytes are split across tokens.
- **Decoding with error replacement.** Silently substituting replacement characters hides split characters instead of joining their bytes.
- **Mishandling empty inputs.** An empty string, an empty merge list and an empty ID list are all valid and should not raise errors.
