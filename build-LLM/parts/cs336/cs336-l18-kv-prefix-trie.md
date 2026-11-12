# Find Reusable Prefixes in a KV Trie

Many requests to a language model share their opening tokens, such as the same system prompt, the same few-shot examples or earlier turns of the same conversation. The keys and values for those shared tokens are identical, so recomputing them for every request wastes work. A prefix trie stores cached sequences so a new request can quickly find the longest prefix whose KV cache already exists. The heart of this problem is building that trie and walking it to find the longest cached prefix of each request.

## Why share prefixes

Prefill, the step that processes a prompt and fills its KV cache, costs time that grows with the prompt length. A chatbot whose every request begins with the same 2,000-token system prompt would redo that work on every call. Reusing the cached keys and values for the shared part means only the new suffix needs prefill, which cuts both latency and GPU time. SGLang's RadixAttention (Zheng et al., 2024) is one well-known system built on this idea.

The reuse is only valid for an exact prefix that starts at the first token. A key or value at position $t$ depends on that token and every token before it, so a cache entry can be reused only if all earlier tokens match.

## A tree of token edges

A trie stores sequences as paths from a root. Each edge is labeled with one token, and each node stands for the prefix spelled by the path that reaches it. The root, node 0, is the empty prefix.

Cached sequences are inserted in order, tokens left to right. When a token edge already exists, the insertion follows it. When it does not, a new node is created. Shared prefixes therefore reuse the same nodes, so a popular system prompt is stored once however many requests used it. In a real server each node would point to the stored keys and values for its prefix.

## Walking the trie finds the longest cached prefix

A request walks down from the root, following one edge per token, until it reaches a token with no matching edge or runs out of tokens. The number of tokens followed is the matched length. The node reached marks where the stored keys and values for the matched prefix live, and the tokens after the match form the uncached suffix that still needs prefill.

A match can stop partway along a longer cached sequence, since any prefix of a cached sequence is also cached. The walk takes one step per matched token, so finding the reusable prefix costs time proportional to the request length, however many sequences the cache holds.

## Three requests against two cached sequences

Insert the cached sequences $[5, 6, 7]$ and $[5, 8]$. The first creates node 1 for token 5, node 2 for 6 and node 3 for 7. The second follows the existing edge for 5 to node 1, then creates node 4 for token 8.

Now match three requests.

- The request $[5, 6, 7, 2]$ follows 5, 6 and 7 to node 3, then finds no edge for 2. It matches 3 tokens at node 3, and only $[2]$ remains uncached.
- The request $[5, 8]$ follows 5 and 8 to node 4 and runs out of tokens. It matches both tokens, and nothing remains to compute.
- The request $[6, 5]$ has no edge for 6 at the root, so it matches 0 tokens at node 0, and all of $[6, 5]$ is uncached. It contains the cached token 5 at its second position, which does not help, because a cache entry is only valid from position 0.

The first request reused a whole cached sequence and added one token, which is the common case when a conversation continues.

## Keeping the cache useful

Memory for cached prefixes is limited, so real systems evict nodes that have not been used recently, usually starting from the leaves so that widely shared prefixes near the root survive longest. Some also merge chains of single-child nodes into one edge holding several tokens, a radix tree, which saves memory without changing which prefixes match.

## Common mistakes to avoid

- **Creating new nodes for a repeated prefix.** Existing edges are followed, and only unseen tokens create nodes.
- **Using one global token-to-node map.** Token 6 after 5 and token 6 at the start would share a node even though their KV caches differ.
- **Requiring a match to end at a full cached sequence.** Any prefix of a cached sequence is reusable.
- **Continuing past the first mismatch.** Matching stops at the first token without an edge, since later tokens depend on it.
- **Matching a cached sequence anywhere inside the request.** KV entries are valid only for prefixes that start at position 0.
