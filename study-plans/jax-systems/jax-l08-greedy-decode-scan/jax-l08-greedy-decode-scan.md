# Greedy Decode Loop with Scan

A language model writes text one token at a time. It reads the token it just produced, updates its internal state, scores every possible next token, and picks one, and that choice becomes the input of the next step. This feedback loop is called autoregressive decoding, and the simplest version of it, greedy decoding, always picks the highest-scoring token. This page starts with a single step of a tiny model, then follows the loop for a few steps, and ends with the JAX tool that compiles the whole loop into one efficient program, lax.scan.

## One step of a tiny recurrent model

The model in this problem is a recurrent network, a model that carries a hidden state vector from one step to the next. Write $h$ for that state, a vector with H entries. One step needs three pieces of learned weights. The embedding table $E$ has one row per token, so row $E_y$ is the vector for token $y$. The recurrent matrix $W_{hh}$ and the bias $b_h$ mix the old state into the new one. The output matrix $W_{out}$ and bias $b_{out}$ turn the state into one score per vocabulary entry. These scores are called logits.

Take the smallest possible case, with hidden size H = 1 and a vocabulary of V = 3 tokens. Let the embeddings be $-1.0$, $1.5$ and $0.3$ for tokens 0, 1 and 2, let $W_{hh} = 0.5$ and $b_h = 0$, and let $W_{out} = [1, -1, 0.5]$ with $b_{out} = 0$. Start from token 0 and a zero state.

The step first adds the embedding of the current token to the transformed old state, which gives $-1.0 + 0.5 \cdot 0 = -1.0$. It then squashes this with tanh, so the new state is $\tanh(-1.0) \approx -0.7616$. Multiplying by $W_{out}$ gives the logits $[-0.7616, 0.7616, -0.3808]$.

## Turning logits into a choice and a log probability

Logits are raw scores. The softmax function turns a vector of logits $\ell$ into probabilities that are positive and sum to 1, by exponentiating each entry and dividing by the total. Taking the log of that result gives the log-softmax.

$$
\log \operatorname{softmax}(\ell)_v = \ell_v - \log \sum_{w=0}^{V-1} e^{\ell_w}
$$

Here $v$ is the token whose log probability you want and $w$ runs over all V tokens. For the logits above, the log probabilities are about $[-1.9531, -0.4299, -1.5723]$, which are the probabilities $0.1418$, $0.6506$ and $0.2076$.

Greedy decoding takes the token with the largest logit, which here is token 1. The softmax never changes which entry is largest, because it applies the same increasing function to every logit, so taking the argmax of the logits and taking it of the probabilities give the same token. The step reports token 1 together with its log probability $-0.4299$.

Decoders track log probabilities because they add up. The probability of a whole sequence is the product of its per-step probabilities, and a product of many numbers below 1 quickly becomes too small for a float to hold. Fifty steps at probability 0.9 already give $0.9^{50} \approx 0.005$, while the log of that product is a comfortable $-5.27$. Libraries compute the log-softmax in one stable operation that subtracts the largest logit first, so the exponentials never overflow.

## Feeding the choice back in

The second step reads token 1, the token just produced, and the state $-0.7616$. Its pre-activation is $1.5 + 0.5 \cdot (-0.7616) \approx 1.1192$, the new state is about $0.8073$, and the logits are about $[0.8073, -0.8073, 0.4036]$. Token 0 wins with log probability about $-0.6243$.

The third step reads token 0 and the state $0.8073$, giving a pre-activation of about $-0.5964$, a state of about $-0.5345$, and token 1 again with log probability about $-0.5833$. The three generated tokens are 1, 0 and 1, and the log probability of the whole sequence is the sum, about $-1.6374$.

Two things carry over from one step to the next, the hidden state and the token that was just chosen. Everything else, the weights in particular, stays the same at every step. That split between what changes and what stays fixed is exactly what the JAX loop needs to know.

## What happens to a Python loop under jit

jax.jit compiles a function by tracing it. JAX runs the Python code once with placeholder values called tracers, which carry a shape and a dtype but no actual numbers, and it records every array operation into a program called a jaxpr. The compiler then optimizes and runs that program.

A Python for loop that calls the step function T times is not recorded as a loop. Python simply runs the body T times during tracing, so the recorded program holds T copies of the step, one after another. In a small measurement, a toy step with three operations produced a jaxpr with 3 equations for one step, 30 for ten steps and 300 for a hundred. Compile time grows with that size, and in one CPU measurement compiling 200 unrolled steps took about 0.6 seconds, while the same loop written with scan compiled in about 0.05 seconds.

Python control flow on traced values is a second problem. A tracer has no concrete value, so an if statement on a traced token, or a loop that stops when some traced value reaches a threshold, raises an error during tracing. The decoding loop has to be expressed with JAX's own loop primitives.

## How lax.scan runs a loop

lax.scan is the JAX primitive for a loop with a fixed number of steps that carries state forward and collects one output per step. You give it a step function $f$, an initial carry $c_0$ and either a sequence of per-step inputs or a length. At step $t$ it computes

$$
(c_t, y_t) = f(c_{t-1}, x_t)
$$

where $c_t$ is the carry after step $t$, $x_t$ is the per-step input (None when you only pass a length), and $y_t$ is that step's output. When the loop ends, scan returns the final carry $c_T$ together with all the outputs $y_1, \dots, y_T$ stacked along a new leading axis of size T.

The key difference from a Python loop is that scan traces $f$ exactly once. The jaxpr holds a single scan equation whose body is the step, in the toy measurement above 1 equation whatever the number of steps, and the compiled program runs that body T times. Compile time no longer depends on T.

Both the carry and the outputs can be pytrees, which in JAX means any nesting of tuples, lists and dicts whose leaves are arrays. A carry made of a hidden state and a token is a tuple with two leaves, and an output made of a token and a log probability is another tuple with two leaves.

## The carry must keep its shape and dtype

Because the body is compiled once and reused, every step must hand back a carry with the same structure, shapes and dtypes it received. If the carry comes in as an int32 scalar and the body returns a float32 one, scan raises a TypeError saying the input carry has type int32[] while the output carry has type float32[]. A carry that grows from shape (1,) to shape (2,) fails the same way.

This rule matters for the token in the carry. The start tokens arrive as int32, and jnp.argmax also returns int32 when 64-bit mode is off, which is the JAX default. Converting the token to a float, or building it with a different integer width, breaks the match. The hidden state stays float32 throughout because tanh and matrix products of float32 arrays return float32.

Values that never change, such as the weights in params, do not belong in the carry. The step function can simply read them from the enclosing Python scope, and scan treats them as constants of the loop body.

## Time comes first in stacked outputs

scan stacks outputs along axis 0, so per-step outputs with shape (B,) come back with shape (T, B), time first. This layout is called time-major. A batch-first layout of shape (B, T), where each row is one sequence, is what most downstream code expects, so the two leading axes need to be swapped at the end.

The two layouts are easy to confuse when B equals T, because both shapes then look the same while the values sit in the wrong places. The safest habit is to name the axes as you go and check which one indexes the batch.

## What greedy decoding gives up

Greedy decoding is deterministic and cheap, and it is a common baseline for generation. It commits to the locally best token at every step, though, and a slightly worse token now can lead to a much more likely sequence later. Beam search, the subject of a later problem in this section, keeps several candidate sequences alive to soften that weakness, and sampling methods replace the argmax with a random draw. All of them share the same loop structure, with a carried state, one token per step and per-step outputs collected by scan.

## Common mistakes to avoid

- **Writing the loop as a Python for loop.** Under jit it unrolls into num_steps copies of the step, which makes the compiled program and its compile time grow with the number of tokens.
- **Returning the stacked outputs without swapping axes.** scan returns time-major arrays of shape (num_steps, B), and the result must be batch-first.
- **Scoring the old state.** The logits of step $t$ come from the state after reading token $y_{t-1}$, so computing logits before the state update shifts every token by one step.
- **Reporting raw logits as log probabilities.** A logit is only a score, and the log probability needs the log of the softmax normalizer subtracted.
- **Changing the carry dtype.** Casting the chosen token to float, or returning a carry of a different shape, makes scan reject the step function.
- **Ignoring the starting state.** The first update uses h0, and replacing it with zeros changes every step that follows.