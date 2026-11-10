# Prefix Cache Matching and Reuse

Many requests begin with the same tokens. They may share a system instruction, a document template, or another long prompt prefix. If a previous request has already produced key-value cache entries for that exact prefix, a new request can reuse those entries instead of calculating them again.

This problem represents cached sequences as fixed-size token blocks. It finds the candidate with the longest consecutive match from the beginning of the new request and returns the physical block IDs holding that reusable cache.

The word prefix is essential. Matching blocks later in a sequence do not help when an earlier block differs, because a token’s cached representation depends on the context that came before it.

## Matching begins at token zero

Suppose the block size is four and the request begins with

$$
[10,11,12,13,20,21,22,23,30,31]
$$

Its complete blocks are

$$
[10,11,12,13]
$$

and

$$
[20,21,22,23]
$$

The final tokens $[30,31]$ form only half a block, so they are not eligible for reuse in this exercise.

A candidate must match the first complete block before its second block can matter. If its first block differs, its reusable prefix length is zero even if a later block happens to equal the request’s second block.

## A block match is exact

Every token inside a block must match in the same order. For a request block $[10,11,12,13]$:

- $[10,11,12,13]$ is a match,
- $[10,11,99,13]$ is not,
- $[13,12,11,10]$ is not.

One differing token invalidates the complete block. The function does not return partial credit for the matching tokens before that difference.

Exact comparison is appropriate because the cached keys and values were created from a specific token history. Similar text or a nearly matching token sequence does not guarantee the same cached state.

## Stop at the first mismatch

For candidate $c$, let $m_c$ be the number of consecutive complete blocks matching from the start. The comparison proceeds in logical order:

1. compare request block 0 with candidate block 0,
2. continue only if they are equal,
3. stop permanently at the first unequal block or when either side has no next complete block.

If blocks 0 and 1 match but block 2 differs, then $m_c=2$. A matching block 3 cannot extend the prefix because the chain was already broken at block 2.

This gives the reusable token count

$$
R_c=m_cP
$$

where $P$ is the block size.

## Choosing among several candidates

Each cached candidate may match a different number of blocks. The selected candidate is the one with the largest $m_c$.

Consider a request with three complete blocks:

- candidate 0 matches the first block and then differs,
- candidate 1 matches the first two blocks and then differs,
- candidate 2 also matches the first two blocks.

Candidates 1 and 2 have the longest match. The required tie rule selects candidate 1 because it has the lower candidate index.

A simple way to preserve that rule is to replace the current best only when a candidate has a strictly greater match length. An equal length leaves the earlier winner unchanged.

## Returning physical block IDs

Cached token blocks describe what each physical block contains. The aligned physical-ID lists describe where those blocks live.

Suppose the chosen candidate has token blocks $A$, $B$, and $C$ stored in physical blocks $[40,7,19]$. If only $A$ and $B$ match the request prefix, return $[40,7]$.

Do not return the candidate’s third physical ID merely because it exists. Only the IDs aligned with the matched prefix are reusable.

Physical IDs need not be sorted or contiguous. Their order follows the candidate’s logical block order and must be preserved in the result.

If no candidate matches the first complete request block, the reusable ID list is empty and the reused token count is zero.

## Why only complete request blocks count

The number of eligible request blocks is found with floor division:

$$
N_{req}=\left\lfloor\frac{|R|}{P}\right\rfloor
$$

For ten request tokens and block size four, only two complete blocks are considered. The two remaining tokens cannot form a reusable block under this contract.

This rule gives the allocator a clean ownership boundary. Reused physical blocks are complete, and new cache computation begins after the last reused block.

A production system can define more elaborate handling for partial blocks, but this exercise explicitly excludes it. Inventing partial reuse would return a token count that is not a multiple of the block size and would conflict with the stored-row requirements.

## Candidate lengths can differ

A cached candidate may contain fewer blocks than the request. Its match ends when the candidate ends, even if all its available blocks matched.

A candidate may also contain more blocks than the request has complete blocks. Extra candidate blocks are irrelevant because there is no eligible request block to compare with them.

The maximum possible match for a candidate is therefore limited by both complete request blocks and candidate block count.

The token-block structure and physical-ID structure are aligned by contract. Candidate $c$ has one physical ID for each of its token blocks, allowing a matched block index to select the corresponding location directly.

## What reuse saves

When the first $m$ blocks match, their key-value states have already been calculated and stored. Reusing their physical IDs lets the new request begin after $mP$ tokens of cached context.

The function itself does not copy cache tensors or run the remaining model computation. It identifies the reusable region and its locations. A later component can attach those physical blocks to the new sequence’s block table.

This separation keeps matching deterministic and small. Eviction policies, cache hashing, reference counts, and memory allocation are outside the requested function.

## Complexity and memory

Let $C$ be the number of candidates, $N$ the number of complete request blocks, and $P$ the block size. In the worst case, each candidate matches until one side ends, so token comparison costs approximately $O(CNP)$.

Early mismatches reduce the actual work because comparison stops immediately for that candidate.

The result uses $O(m)$ memory for the selected physical IDs, where $m$ is the number of matched blocks. Apart from the returned list, the search needs only counters and the current best candidate information.

## Common mistakes to avoid

- Comparing every matching block anywhere in a candidate ignores the consecutive-prefix requirement.
- Continuing after the first mismatch can falsely count a later block as reusable.
- Treating a partially matching block as reusable violates the whole-block boundary.
- Including a partial final request block uses ceiling instead of the required floor division.
- Updating the best candidate on equal match lengths breaks the lowest-index tie rule.
- Returning all physical IDs from the winning candidate includes blocks beyond the matched prefix.
- Sorting physical IDs destroys their alignment with logical block order.
- Returning a matching token count that is not a multiple of block size contradicts full-block reuse.

The safe mental model is a chain of complete blocks starting at the first token. Each equal block extends the chain by exactly one block size, the first mismatch ends it, and the longest unbroken chain identifies the cache entries that can be reused.
