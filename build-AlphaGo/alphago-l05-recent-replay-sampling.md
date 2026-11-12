# Understanding Recent Replay Sampling

Self-play produces games, and each game contains a sequence of training records. A replay sampler chooses positions from this stored experience. This problem limits eligibility to the most recent games, preserves the exact requested draws, and returns independent copies so training cannot alter replay storage.

## The replay structure

The outer collection is ordered from older games to newer games. Within each game, records are ordered by position. Every record contains a state, a policy target, and a value target.

The sampler does not reinterpret those fields. Its job is to select complete records while preserving their contents. State arrays, policy arrays, and any nested containers are part of the record that must be copied.

Two different counts are involved:

- Window size counts games.
- Position indices refer to records after the recent games are flattened.

Confusing these levels changes which experiences are eligible.

## Select recent games before flattening

If the window size is $w$, keep the last $w$ games. When fewer than $w$ games exist, keep all available games. A positive window size never requires padding or synthetic games.

The recent-game window must be formed before positions are flattened. This is the central ordering rule of the problem. Selecting the last number of positions from the entire replay would define a position window rather than a game window.

The difference matters because games can have different lengths. A window of two games may include many positions from a long older game and only a few from the newest game. The requested rule keeps both complete games.

## Flattening the eligible records

After choosing the recent games, flatten their records in game order and position order. The oldest game still inside the window contributes its first record first, followed by its later records. Newer games follow in the same way.

Empty games contribute no records but remain valid members of the recent window. They do not interrupt ordering and do not create synthetic positions.

The number of records in this flattened collection is the eligible position count returned as the second result. This count describes the full sampling pool, not the number of requested samples.

For example, if the recent window contains one empty game, one game with two records, and one game with three records, the eligible count is five. An empty request still returns that count because the pool exists even when no draws are requested.

## Position indices are supplied draws

Each supplied zero-based index selects one record from the flattened recent pool. The indices are already valid, so the function does not generate random numbers or validate probability distributions.

Selection must preserve both order and multiplicity. If the requested indices refer to a later position and then an earlier position, the returned records follow that requested order. The sampler does not sort indices into replay order.

Repeated indices represent repeated draws with replacement. If the same index appears three times, the output contains three records. Deduplicating the indices or converting them to a set would change the sample.

This design separates random sampling from replay access. A caller can generate indices using any desired random process, while this function deterministically materializes the corresponding batch.

## Why every sample needs a deep copy

Replay storage is meant to remain stable while training consumes a batch. A sampled record may contain nested mutable values such as a board state or policy collection. Returning references to the stored records would let later modifications alter replay history.

A shallow copy of the record is not enough because nested state and policy objects would still be shared. A deep copy recursively creates independent nested contents.

Repeated draws require separate deep copies for each occurrence. If one deep-copied object were created and inserted multiple times, modifying one batch entry would also modify the other repeated entry. Copying during each indexed selection isolates the batch from storage and isolates repeated samples from each other.

## Window boundaries

When the window size is one, only the newest game is eligible. Records from every older game must be excluded even if the newest game contains very few positions.

When the window size exceeds the number of stored games, all games are eligible. Taking the smaller of the window size and game count expresses this clearly and avoids unusual slice behaviour.

When there are no stored games, the recent window and eligible pool are empty. Under the stated constraints, the supplied index collection must then also be empty. The result contains an empty sample and an eligible count of zero.

## Empty samples and empty games

An empty position-index collection is valid even when eligible positions exist. The sampled-record result is empty, but the eligible count still reports the size of the recent pool.

Empty games can appear anywhere. If the newest game is empty and the window size is one, no positions are eligible even though an older game contains records. Expanding the window may bring those older records into eligibility.

These cases follow naturally when windowing and flattening remain separate operations.

## Common mistakes to avoid

- Flattening all games before applying the window creates a position-based window.
- Counting requested samples instead of eligible positions returns the wrong second value.
- Sorting indices loses the caller's draw order.
- Deduplicating repeated indices changes sampling with replacement.
- Returning original records lets training mutate replay storage.
- Making one copy per unique index causes repeated outputs to share nested objects.
- Skipping empty games by changing window counting can pull in an older game incorrectly.
- Generating random indices inside the function ignores the supplied draws.

Recent replay sampling is precise data selection: choose the newest games, flatten their positions without changing order, materialize every requested index including repetitions, and deep-copy each result so stored experience and batch entries remain independent.

---