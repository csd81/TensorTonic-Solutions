# Exponential Moving Average with Scan

An exponential moving average is a running average that gives recent values more weight than old ones. It appears all over machine learning, from the momentum and second-moment estimates inside Adam to the averaged weights that many image and language models use at evaluation time. Computing it means walking through a sequence and updating a small state at every step, which is exactly the kind of loop that JAX wants written with jax.lax.scan. This page builds the average from a three-step example, explains the bias that appears at the start and how to correct it, and shows why scan is the right way to express the loop inside compiled code.

## A running average in one line

Suppose a sequence of values $x_1, x_2, \dots, x_T$ arrives one step at a time. The exponential moving average keeps one running value $m$ that starts at 0. At each step $t$ it keeps a fraction $\beta$ of the old value and mixes in the rest from the new one,

$$
m_t = \beta \, m_{t-1} + (1 - \beta)\, x_t, \qquad m_0 = 0,
$$

where $\beta$ is the decay, a number from 0 up to but not including 1. A decay close to 1 makes the average change slowly and remember a long history, while a decay of 0 makes $m_t$ equal to the latest value.

Take $\beta = 0.8$ and a constant sequence of three 5s. The running value goes from 0 to 1 at step 1, then to 1.8 at step 2, then to 2.44 at step 3. Every input was 5, yet the average is far below 5 after three steps.

## Where the bias comes from

Unrolling the update shows what happened. After $t$ steps,

$$
m_t = (1 - \beta) \left( x_t + \beta\, x_{t-1} + \beta^2 x_{t-2} + \cdots + \beta^{t-1} x_1 \right),
$$

so $m_t$ is a weighted sum of the inputs whose weights add up to $(1 - \beta)(1 + \beta + \cdots + \beta^{t-1}) = 1 - \beta^t$. The missing weight $\beta^t$ belongs to the starting value of 0. In the example, the weights after three steps add up to $1 - 0.8^3 = 0.488$, so the average is less than half of the true level of 5.

The effect fades as $t$ grows, because $\beta^t$ shrinks towards 0. With a typical decay of 0.99, however, the weights add up to only about 0.0956 after 10 steps, and the missing share only falls below one half after about 69 steps. For an optimizer that reads the average from the very first step, that start-up bias is far too large to ignore.

## Correcting the bias

Since the weights add up to $1 - \beta^t$, dividing by that sum turns the running value into a proper weighted average,

$$
\hat m_t = \frac{m_t}{1 - \beta^t},
$$

where $\hat m_t$ is the bias-corrected average at step $t$. In the example, $1 / 0.2 = 5$ at step 1, $1.8 / 0.36 = 5$ at step 2 and $2.44 / 0.488 = 5$ at step 3, so the corrected average recovers the true level from the first step. Adam applies exactly this correction to both of its running averages.

Step numbering matters here. The first update is step 1, which is why the exponent is 1 at that step and the correction divides by $1 - \beta$. Counting from 0 would put $1 - \beta^0 = 0$ in the denominator. With $\beta = 0$ the correction is $1 - 0^t = 1$ for every step, so the corrected value equals the latest input, which matches the meaning of a decay of 0.

The correction is only applied when the average is read. The value carried forward to the next step is always the raw $m_t$, because the next update expects the uncorrected recurrence.

## Why a Python loop is a poor fit under jit

You could write the recurrence as a Python for loop over the rows of the input. Under jax.jit that loop runs once while the function is traced, and every pass appends a fresh copy of the update to the compiled program. JAX can show the recorded program with jax.make_jaxpr. Take a loop that applies the update and the bias correction at every step and stacks the corrected values, run on a sequence with two features and with decay passed in as a traced float32 value. Its program has 92 equations when the sequence has 10 steps and 902 equations when it has 100 steps. The exact count depends on how the loop is written, and it always grows in proportion to the number of steps. The written-out program grows with the sequence length, compile time grows with it, and a new length means a new, larger program.

The scan version of the same average has 2 top-level equations for both lengths, one to build the starting zeros and one scan, with the update written once inside it. Its size does not depend on the length of the sequence.

## How lax.scan works

jax.lax.scan takes a step function, an initial carry and an input array, and it runs the step function once for each slice along the input's leading axis. The step function receives the current carry and one slice, and it returns two things, the new carry and an output for that step.

$$
(\text{carry}_t,\ y_t) = f(\text{carry}_{t-1},\ x_t)
$$

When the loop ends, scan returns the final carry and all the per-step outputs stacked along a new leading axis. For an input of shape $(T, D)$, each slice $x_t$ has shape $(D,)$, and if each $y_t$ has shape $(D,)$ the stacked outputs have shape $(T, D)$.

The carry must keep the same structure, shapes and dtypes on every step, because the compiled loop body has fixed shapes. The carry can be a single array, or a tuple of several arrays and numbers. Starting the carry as an int32 array when the update produces float32, or starting a scalar carry that the update turns into a vector, makes scan raise a TypeError saying that the carry input and carry output must have equal types.

## What to keep in the carry

The running average $m_t$ is the obvious part of the state. The bias correction also needs to know which step it is on, or equivalently the current value of $\beta^t$, so that has to travel in the carry too. One option carries a step counter and raises $\beta$ to that power inside the step function. Another option carries the running power itself and multiplies it by $\beta$ at every step. Both give the same corrected values.

As a second example, take $\beta = 0.9$ and the inputs 4, 2 and 6. The raw values are 0.4, 0.56 and 1.104, the weight sums are 0.1, 0.19 and 0.271, and the corrected averages are 4, about 2.947368 and about 4.073801. The final carry holds the raw value 1.104, while the per-step outputs hold the three corrected values.

## The final state and the history

The two results of a scan answer two different questions. The final carry is the state you would save and resume from, for example the averaged weights at the end of an epoch, so it holds the raw running value. The stacked outputs are the history, one corrected average per step, which is what you would plot or feed to the next computation. Returning both from one scan costs nothing extra, since the loop computes each of them on the way.

## Common mistakes to avoid

- **Skipping the bias correction.** The raw average starts far below the true level, and with a decay near 1 it stays biased for many steps.
- **Counting steps from 0.** The first update is step 1, so an exponent that starts at 0 divides by zero at the first step.
- **Carrying the corrected value forward.** The recurrence must use the raw $m_{t-1}$, so feeding the corrected value back in changes every later step.
- **Starting the average at the first input.** The definition starts from 0 and relies on the correction, so seeding with $x_1$ gives different values whenever $\beta$ is above 0.
- **Writing a Python loop inside jit.** The loop is unrolled into one copy of the update per step, so the program and its compile time grow with the sequence length.
- **Changing the carry's dtype or shape.** scan requires the carry to look the same on every step, so a carry that starts as an int32 array or as a scalar where a vector is needed fails to trace.