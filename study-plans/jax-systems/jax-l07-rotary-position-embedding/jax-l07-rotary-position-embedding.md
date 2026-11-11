# Rotary Position Embedding

Attention compares every query with every key through a dot product, and a plain dot product has no idea where in the sentence each token sits. Rotary position embedding, usually shortened to RoPE, fixes this by turning each query and key vector by an angle that depends on the token's position. This page builds the idea from a single pair of numbers turning on a circle, then shows why the turn makes attention care about distance, and ends with the full rule for many pairs, many heads and positions that arrive as a JAX array inside a compiled function. That full rule is what you implement in this problem.

## Why attention needs positions at all

A query vector and a key vector are just lists of numbers computed from token embeddings. If the same word appears at position 2 and at position 40, it produces the same key both times, so a plain dot product gives the same score for both copies. The model would see a sentence as an unordered bag of words.

Older transformers added a position vector to each token embedding before attention. RoPE leaves the embeddings alone and rotates the query and key right before the dot product. Rotation is a good choice because it never changes the length of a vector, so it adds position information without inflating or shrinking the scores.

## Turning one pair of numbers

Start with the smallest possible case, a vector with two numbers $(x_0, x_1)$. Picture it as an arrow on a flat plane. Rotating it by an angle $a$ (measured in radians) gives a new arrow with the same length.

$$
x_0' = x_0 \cos a - x_1 \sin a, \qquad x_1' = x_1 \cos a + x_0 \sin a
$$

Take the arrow $(1, 0)$ and the angle $a = 0.5$. The cosine of 0.5 is about 0.8776 and the sine is about 0.4794, so the arrow becomes $(0.8776, 0.4794)$. Its length is still 1. With $a = 0$ the cosine is 1 and the sine is 0, so nothing moves, which is why a token at position 0 comes out of RoPE unchanged.

RoPE gives every position its own angle. With a speed $\theta$ (the angle added per position), a token at position $p$ turns by $a = p\,\theta$. So position 3 with $\theta = 0.5$ turns by 1.5 radians, and position 10 turns by 5 radians.

## Why rotation turns into relative position

The useful property appears when a rotated query meets a rotated key. Suppose the query sits at position $m$ and the key at position $n$, both rotated with the same speed $\theta$. Rotating the query by $m\theta$ and the key by $n\theta$, then taking their dot product, gives exactly the same number as leaving the key alone and rotating the query by $(m - n)\theta$. Rotations add up, and the dot product only sees the angle between the two arrows.

Here is a check with numbers. Let the query be $(1, 2)$, the key be $(0.5, -1)$ and $\theta = 0.5$. With the query at position 3 and the key at position 1, the score after rotation is about $-2.4934$. Move both tokens two places later, to positions 5 and 3, and the score is again $-2.4934$. Only the gap of 2 positions matters. This is the reason RoPE is popular, since a model trained on short sequences sees the same pattern for "two tokens apart" wherever it happens in a long sequence.

## Many pairs turning at different speeds

A real head has D numbers, where D is the head size, and D is even. RoPE splits the D numbers into $D/2$ pairs and turns each pair at its own speed. Call the number of pairs half, so half $= D/2$, and number the pairs with $i = 0, 1, \dots, \text{half} - 1$. Pair $i$ turns at the speed

$$
\theta_i = \text{base}^{-2i/D}
$$

where base is a fixed number, usually 10000. Pair 0 always has speed 1, and the speeds shrink geometrically after that. With $D = 8$ and base 10000 the speeds are 1, 0.1, 0.01 and 0.001.

The spread of speeds is the point. A fast pair changes a lot from one position to the next, so it can tell neighbours apart, but it wraps around the circle quickly and repeats. A slow pair barely moves between neighbours, but it takes a very long time to wrap. The slowest pair in the $D = 8$ example needs about 6283 positions to complete one full turn. Together the pairs act like the hands of a clock, where the fast ones give precision and the slow ones give range. A larger base makes the slow pairs even slower, which is one way long-context models stretch their range.

## Which numbers form a pair

Nothing in the rotation itself says which two of the D numbers belong together, so every implementation has to pick a convention, and a query rotated with one convention gives nonsense against a key rotated with another. Two conventions are common.

- **Interleaved pairs.** Neighbours form a pair, so dimension 0 goes with dimension 1, dimension 2 with dimension 3, and so on.
- **Half-split pairs.** Dimension $i$ goes with dimension $i + \text{half}$, so the first half of the vector holds the first member of every pair and the second half holds the second member.

This problem uses the half-split convention, which is the one used by many open model codebases. With $D = 8$, dimension 1 pairs with dimension 5, and that pair turns at speed $\theta_1$. The two conventions give the same model quality, so the only thing that matters is that queries, keys and any pretrained weights agree on one of them.

## A worked example with four numbers

Take $D = 4$, base 100 and a vector $x = (1, 2, 3, 4)$ at position $p = 2$. There are two pairs. The speeds are $\theta_0 = 100^{0} = 1$ and $\theta_1 = 100^{-2/4} = 0.1$, so the angles are $a_0 = 2$ and $a_1 = 0.2$.

With the half-split convention, pair 0 is $(x_0, x_2) = (1, 3)$ and pair 1 is $(x_1, x_3) = (2, 4)$. The cosine and sine of 2 are about $-0.4161$ and $0.9093$, and the cosine and sine of 0.2 are about $0.9801$ and $0.1987$. Rotating each pair gives

- **Pair 0.** It becomes $1 \cdot (-0.4161) - 3 \cdot 0.9093 \approx -3.1440$ and $3 \cdot (-0.4161) + 1 \cdot 0.9093 \approx -0.3391$.
- **Pair 1.** It becomes $2 \cdot 0.9801 - 4 \cdot 0.1987 \approx 1.1655$ and $4 \cdot 0.9801 + 2 \cdot 0.1987 \approx 4.3176$.

Putting the results back in their slots gives about $(-3.1440, 1.1655, -0.3391, 4.3176)$. The first pair had length $\sqrt{1^2 + 3^2} \approx 3.1623$ before the turn, and $\sqrt{3.1440^2 + 0.3391^2}$ is again about 3.1623, as a rotation should keep it.

## Heads, queries and keys share the angles

The angle depends only on the token's position and the pair index. It does not depend on the head, so every query head and every key head of one token turns by the same angles. In a grouped-query model the query block has Hq heads while the key block has fewer heads Hk, and both still use the same table of angles. That table has one row per token and one column per pair, so its shape is (T, half), where T is the number of tokens. Giving it a size-1 axis for heads makes it line up with a (T, H, half) block by broadcasting, which is NumPy's rule for stretching a size-1 axis to match the other array.

Values are never rotated. Position only needs to affect which keys a query attends to, and the value vectors carry content that should come out of attention unchanged.

## Positions are data inside a compiled function

In an inference server the positions are not a neat range starting at 0. During decoding each sequence in a batch sits at its own position, and after a cache is trimmed or a prompt is reused the positions can jump. So positions arrive as an int32 array, one entry per token.

When a function runs under jax.jit, JAX first traces it. Tracing replaces every array argument with a stand-in object called a tracer, which knows the array's shape and dtype but not its values, and records the operations applied to it into a program that XLA (the compiler behind JAX) then compiles. A traced position therefore has no concrete number. Code that calls int or float on a position, or converts the positions to a NumPy array, fails at trace time, while code written with jax.numpy operations on whole arrays traces into one compiled program that works for any positions of the same shape. base is different, since it is passed as a static argument, an ordinary Python number baked into the compiled program.

## Precision of the angles

The angle $p\,\theta_i$ grows with the position. In float32, which keeps about seven significant digits, neighbouring representable numbers near 10000 are about 0.001 apart, so a large angle carries a rounding error that goes straight into the cosine and sine. With D = 128 and base 10000, the largest gap between a cosine from a float32 angle table and the float64 cosine is about 0.000004 at position 255 and about 0.005 at position 100000. A bfloat16 table is far worse, since position 100000 rounds to 99840 in bfloat16 and some cosines at position 255 are already off by more than 0.5. Long-context code therefore builds the angle table once in float32 or higher precision from integer positions, and only the rotated vectors are cast to a smaller format afterwards.

## Common mistakes to avoid

- **Pairing neighbours instead of halves.** Interleaved pairing is a valid convention in general, but it rotates different numbers together and gives a different answer from the half-split rule used here.
- **Using the token index instead of the position.** Positions can be out of order or repeated, so an arange over the T tokens gives the wrong angles whenever they are not 0, 1, 2 and so on.
- **Dropping the factor of 2 in the exponent.** The speed of pair $i$ is base to the power $-2i/D$, and using $-i/D$ makes the speeds fall too slowly.
- **Turning the wrong way.** The first member of a pair loses the sine term and the second gains it, and swapping those signs rotates by the negative angle.
- **Converting positions to Python numbers.** Under jax.jit a position is a tracer, so calling int, float or np.asarray on it fails during tracing.
- **Rotating values or giving each head its own angle.** Only queries and keys are rotated, and all heads of one token share the same angle table.
