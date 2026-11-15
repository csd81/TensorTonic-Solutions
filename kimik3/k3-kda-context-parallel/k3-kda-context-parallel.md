# KDA Context Parallelism

KDA Context Parallelism lets several sequence segments be processed in parallel even though KDA normally carries a recurrent state from one token to the next. The key is to summarize each segment as an affine state transformation, then combine those transformations in sequence order.

## Why a zero-state result is not enough

For simple additive linear attention, a segment can compute the state created from zero, and preceding segment states can be added later. KDA also transforms the state that enters a segment. Its local result therefore depends on both the segment's own contribution and the incoming memory.

Each segment is summarized by two tensors:

- A transition matrix $M_i$ describing how the segment transforms incoming state.
- A local contribution $U_i$ describing the state the segment creates when starting from zero.

Together they define an affine map

$$
F_i(S)=M_iS+U_i
$$

Once these two pieces are known, the segment can be applied to any incoming state without replaying its tokens.

## Compose segments in sequence order

Suppose segment 1 is followed by segment 2. Applying the first map and then the second gives

$$
F_2(F_1(S))=M_2(M_1S+U_1)+U_2
$$

After regrouping, their combined summary is

$$
(M_2,U_2)\circ(M_1,U_1)
=
(M_2M_1,\;M_2U_1+U_2)
$$

This composition is associative, so a parallel prefix scan can combine many segment summaries efficiently. It is not commutative. Swapping two segments usually changes both the transition product and the transformed local contribution.

The implementation in this problem can process segments in a clear loop. The important result is the same exclusive-prefix behavior that a parallel scan would produce.

## Incoming and outgoing states

The incoming state for segment 0 is the supplied initial state. Its outgoing state is

$$
S_0^{\mathrm{out}}=M_0S^{\mathrm{initial}}+U_0
$$

That outgoing state becomes the incoming state for segment 1, and the process continues. The incoming stack records the state before each segment. The outgoing stack records the state after each segment.

This is an exclusive prefix for incoming states because segment $i$ receives the result of preceding segments only. An inclusive prefix would incorrectly include the current segment in its own input.

## A scalar three-segment example

Use scalar states so every matrix is a number. Let the initial state be $1$, and let the three segment maps be

$$
F_0(S)=2S+1,\qquad F_1(S)=3S+0,\qquad F_2(S)=S-4
$$

Segment 0 receives $1$ and outputs $3$. Segment 1 receives $3$ and outputs $9$. Segment 2 receives $9$ and outputs $5$.

The incoming states are $[1,3,9]$, the outgoing states are $[3,9,5]$, and the final global state is $5$.

Composing the first two maps gives $F_{1\circ0}(S)=6S+3$. Applied to the initial state, this gives $9$, matching the second outgoing state. Reversing them would give $6S+1$, which demonstrates why order matters.

## Matrix order matters

With real KDA states, $M_i$ is a matrix and state multiplication is left-sided. The current transition multiplies the incoming state as $M_iS$. Do not reverse this to $SM_i$, and do not accumulate transition matrices in the opposite order.

Local contributions have the same shape as the recurrent state. When a later transition is composed with an earlier segment, that later matrix must also transform the earlier local contribution.

## Implementation order

- Start a working state from the supplied initial state.
- For each segment in sequence order, append the current working state to the incoming list.
- Compute the outgoing state as current transition times incoming state plus local contribution.
- Append that result to the outgoing list and use it as the next working state.
- Stack incoming states and outgoing states along the segment axis.
- Return both stacks followed by the final working state.

## Common mistakes to avoid

- **Using an inclusive prefix for inputs.** A segment's incoming state excludes its own transformation.
- **Adding local contributions without transitions.** Earlier contributions are transformed by later segment matrices.
- **Reversing matrix multiplication order.** Segment maps follow sequence order and do not generally commute.
- **Starting from zero.** The supplied initial state must enter the first segment.
- **Returning only the final state.** The task also requires every incoming and outgoing state.
