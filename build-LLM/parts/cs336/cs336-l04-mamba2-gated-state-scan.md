# Mamba 2 Gated State Scan

Mamba 2 is a sequence model that replaces softmax attention with a recurrent state. Like linear attention, it keeps a $D_k \times D_v$ state matrix and adds the outer product of each token's key and value to it, but it also applies a decay gate that shrinks the old state before each new write. The heart of this problem is applying that gate at the right moment in the update and reading the state after the current token has been written.

## The problem with a state that never forgets

In plain linear attention the state after $t$ tokens is the sum of every key-value outer product so far. A token from the start of a long document carries exactly as much weight as the token just before the current one. The state keeps filling up, and old information can never be cleared to make room for new information.

Recurrent networks solved a similar problem decades ago with gates that decide what to keep. Mamba 2 brings the same idea to the linear attention state with a single number per token.

## The gated update

At position $t$, let $q_t$, $k_t$ and $v_t$ be the query, key and value, with keys and queries of width $D_k$ and values of width $D_v$. Let $\gamma_t$ be a scalar gate between 0 and 1. Starting from a zero state, the scan computes

$$
S_t = \gamma_t S_{t-1} + k_t v_t^{\mathsf T}
$$

$$
y_t = q_t^{\mathsf T} S_t
$$

The state $S_t$ is a $D_k \times D_v$ matrix. The gate multiplies every entry of the old state by the same number, then the new outer product is added. The output reads the state after the new write, so each token can see its own key and value.

## What the gate means

The two extreme values of the gate have clear meanings.

- A gate of 1 keeps the old state unchanged, which turns the scan back into plain linear attention for that step.
- A gate of 0 erases the old state completely, so the state holds only the current token's write.

Values in between fade the past gradually. Since the gates multiply together over time, a write made at position $s$ is still present at position $t$ with weight $\gamma_{s+1} \gamma_{s+2} \cdots \gamma_t$. A long run of gates near 0.9 lets old information fade over a few dozen tokens, while gates very close to 1 keep it for much longer.

In Mamba 2 the gate is computed from the current token, so the model can learn to hold on to the past in some places and to reset at boundaries such as the start of a new document. Here the gates are given directly, so the exercise isolates the recurrence from the network that would produce them.

## Why a scalar gate matters

The first Mamba model allowed a different decay for every state entry, which made the update more expressive but hard to compute quickly on GPUs. Mamba 2 (Dao and Gu, 2024) restricted the decay to a single scalar per step. With that restriction the whole scan can also be written as one masked matrix product over the sequence, so training runs on fast matrix multiplication while generation uses the step-by-step form. This problem implements the step-by-step form.

## A two-step example

Take $D_k = 2$ and $D_v = 1$. The first token has key $[1, 0]$, value $[4]$ and query $[1, 0]$. The second has key $[1, 1]$, value $[2]$, query $[1, 1]$ and gate $\gamma_2 = 0.5$.

The state starts at zero, so the first gate has nothing to shrink, whatever its value. The first write gives

$$
S_1 = \begin{bmatrix} 4 \\ 0 \end{bmatrix}, \qquad y_1 = [1, 0] \, S_1 = 4
$$

At the second step the gate halves the old state before the new key writes into both rows:

$$
S_2 = 0.5 \begin{bmatrix} 4 \\ 0 \end{bmatrix} + \begin{bmatrix} 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \end{bmatrix}, \qquad y_2 = [1, 1] \, S_2 = 6
$$

With a gate of 1 instead, the state would be $[6, 2]$ and the second output would be 8. The gate removed half of the first token's contribution, while the second token's write entered at full strength.

## Carrying the state into the next token

Each sequence carries its own state, and the state after the last position is what a generation loop keeps for the next token. Because that state is updated at every step, small rounding errors pile up, which is why the recurrence is accumulated in float32.

## Common mistakes to avoid

- **Applying the gate after the write.** The gate shrinks the old state before the new outer product is added.
- **Reading the state before the update.** Each output uses the state that already includes the current token.
- **Gating the new write.** The current key-value product enters the state at full strength.
- **Writing $v_t k_t^{\mathsf T}$ instead of $k_t v_t^{\mathsf T}$.** The transposed product has shape $D_v \times D_k$ and does not match the state.
- **Applying a later token's gate to an earlier write.** Each $\gamma_t$ only shrinks the state that existed before position $t$.
- **Accumulating in bfloat16.** Repeated updates lose precision unless the scan runs in float32.
