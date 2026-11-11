# Gated DeltaNet State Update

Linear attention and Mamba 2 can only add to their state or fade all of it at once, so writing a new value for a key that was already stored leaves the old value mixed in. Gated DeltaNet fixes this with the delta rule, which first removes what the state currently stores for a key and then writes the new value, while a decay gate still fades the whole state over time. The heart of this problem is applying the erase, the decay and the write in the right order.

## The problem with purely additive memory

Think of the state as a memory that maps keys to values. Reading it with a unit key $k$ returns $k^{\mathsf T} S$, the value stored for that key. If the same key is written twice with values 3 and then 8, an additive update stores $3 + 8 = 11$, which is neither value. The memory cannot overwrite a single association.

The delta rule, introduced by Widrow and Hoff for training simple adaptive units, corrects the stored value instead of adding blindly. It looks up what the state currently returns for the key, compares it with the new value, and moves the stored value toward the new one.

## Erase, decay, then write

At position $t$, let $q_t$ be the query, $k_t$ the key and $v_t$ the value, with keys and queries of width $D_k$ and values of width $D_v$, so the state $S_t$ has shape $D_k \times D_v$. Let $\gamma_t$ and $\beta_t$ be two scalar gates between 0 and 1, and let $I$ be the $D_k \times D_k$ identity matrix. The key is normalized to unit length. Starting from a zero state, the update is

$$
S_t = \gamma_t \left( I - \beta_t k_t k_t^{\mathsf T} \right) S_{t-1} + \beta_t k_t v_t^{\mathsf T}
$$

and the output reads the updated state, $y_t = q_t^{\mathsf T} S_t$.

The matrix $I - \beta_t k_t k_t^{\mathsf T}$ erases a fraction $\beta_t$ of whatever the old state stores along the direction $k_t$, and leaves directions orthogonal to $k_t$ untouched. The gate $\gamma_t$ then decays the whole erased state, and the last term writes a fraction $\beta_t$ of the new value along $k_t$. The decay applies to the old state only, so the new write enters at strength $\beta_t$.

## Why it is called a correction

Ignore the decay for a moment by setting $\gamma_t = 1$. Expanding the update gives

$$
S_t = S_{t-1} + \beta_t k_t \left( v_t - S_{t-1}^{\mathsf T} k_t \right)^{\mathsf T}
$$

Here $S_{t-1}^{\mathsf T} k_t$ is what the old state returns for the key, so $v_t - S_{t-1}^{\mathsf T} k_t$ is the prediction error. The update is one gradient step with learning rate $\beta_t$ on the squared error $\tfrac{1}{2} \lVert S^{\mathsf T} k_t - v_t \rVert^2$.

The expansion is pure algebra and holds for any key. Reading it as a correction depends on $k_t$ having unit length. Only then does reading with $k_t$ move the stored value exactly a fraction $\beta_t$ of the way toward the target, so that $\beta_t = 1$ replaces it completely and $\beta_t = 0$ leaves it alone.

## Editing one key versus fading everything

The update gate $\beta_t$ controls how strongly the current token edits the memory for its own key. The decay gate $\gamma_t$, as in Mamba 2, fades everything stored so far. Gated DeltaNet (Yang, Kautz and Hatamizadeh, 2024) combines them so the model can both overwrite specific associations precisely and clear large parts of the memory when the context changes.

## A partial correction example

Take $D_k = 2$, $D_v = 1$ and both decay gates equal to 1. The first token has key $[1, 0]$, value $[3]$ and $\beta_1 = 1$, so starting from zero it writes

$$
S_1 = \begin{bmatrix} 3 \\ 0 \end{bmatrix}
$$

The second token has unit key $k_2 = [0.6, 0.8]$, value $[5]$ and $\beta_2 = 0.5$. The old state returns $k_2^{\mathsf T} S_1 = 1.8$ for this key, so the error is $5 - 1.8 = 3.2$. Erasing half of the stored component and writing half of the new value gives

$$
S_2 = \begin{bmatrix} 3 \\ 0 \end{bmatrix} - 0.5 \times 1.8 \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix} + 0.5 \times 5 \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix} = \begin{bmatrix} 3.96 \\ 1.28 \end{bmatrix}
$$

Reading with the query $q_2 = k_2$ gives $0.6 \times 3.96 + 0.8 \times 1.28 = 3.4$, which is exactly $1.8 + 0.5 \times 3.2$. The stored value moved halfway from 1.8 toward 5.

## A state rewritten at every step

Each sequence keeps its own state, and every step multiplies that state by a new erase matrix before adding a write. Rounding errors from these repeated products accumulate, so the recurrence is computed in float32.

## Common mistakes to avoid

- **Decaying the new write.** The gate $\gamma_t$ multiplies only the erased old state.
- **Using $\beta_t$ in the erase but 1 in the write.** Mismatched strengths stop the update from moving the stored value toward the target.
- **Multiplying the erase matrix on the wrong side.** The $D_k \times D_k$ matrix multiplies the state from the left.
- **Renormalizing the keys.** The supplied keys are already unit length and must be used as given.
- **Reading the state before the update.** Each output uses the state after the current token has been written.
