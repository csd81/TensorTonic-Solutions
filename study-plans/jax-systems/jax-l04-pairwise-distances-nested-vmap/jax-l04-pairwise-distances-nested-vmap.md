# Pairwise Distances with Nested vmap

Many methods need a number for every pair of points drawn from two sets. Nearest-neighbour search compares each query with every stored point, retrieval ranks documents by their cosine similarity to a query embedding, and contrastive losses such as the one in CLIP, a model that learns matching image and text embeddings (Radford et al., 2021), compare every image in a batch with every caption. For a set of N points and a set of M points these numbers form an (N, M) matrix. This page starts from the formulas for one pair, shows how one vmap turns them into a row of the matrix, and then how a second vmap turns the row into the full grid.

## Two numbers for one pair of points

Let $u$ and $w$ be two points written as vectors of width D. Their squared Euclidean distance is the sum of the squared differences of their coordinates,

$$
\lVert u - w \rVert^2 = \sum_{d=1}^{D} \left(u_d - w_d\right)^2 ,
$$

where $u_d$ is coordinate $d$ of $u$. It is 0 when the points coincide and grows as they move apart. Many methods use the squared distance in place of the distance, because it avoids a square root and ranks pairs in the same order.

Their cosine similarity is the cosine of the angle between them,

$$
\cos(u, w) = \frac{u \cdot w}{\lVert u \rVert \, \lVert w \rVert} ,
$$

where $u \cdot w$ is the dot product and $\lVert u \rVert$ is the Euclidean norm, the square root of $u \cdot u$. It ignores the lengths of the vectors and measures only their directions, from 1 for the same direction through 0 for perpendicular vectors to -1 for opposite directions.

Take $u = (1, 2)$ and $w = (3, 0)$. The difference is $(-2, 2)$, so the squared distance is $4 + 4 = 8$. The dot product is 3, the norms are $\sqrt{5} \approx 2.2361$ and 3, and the cosine is $3 / 6.7082 \approx 0.4472$.

## When a point is the zero vector

The zero vector has norm 0, so the cosine formula divides 0 by 0 and returns NaN. A zero vector has no direction, and the convention in this problem gives it a cosine of 0 with every point. Dividing by the larger of the norm product and a tiny floor such as $10^{-8}$ achieves this, because the dot product with a zero vector is exactly 0. For pairs of nonzero points of ordinary size the norm product is far above the floor, so the floor leaves their cosine unchanged.

## One vmap gives one row

Suppose the pair function takes two vectors and returns the two numbers. To compare one point $u$ against all M points of b, map the function over the rows of b while every call receives the same $u$. In terms of in_axes, the first argument is shared with None and the second is mapped over axis 0.

With $u = (1, 2)$ and the three points $(3, 0)$, $(1, 2)$ and $(-1, 1)$, this single vmap returns the squared distances (8, 0, 5) and the cosines (0.4472, 1, 0.3162). A vmapped function can return several outputs, such as a tuple of two numbers, and vmap returns the matching tuple of batched arrays, here two arrays of length 3.

## A second vmap gives the whole grid

The row function takes one point of a and all of b. Mapping it over the rows of a, with b shared, produces one row per point of a. This is the second vmap, wrapped around the first, with in_axes that map the first argument over axis 0 and share the second.

Each vmap adds its axis at position 0 of its output. The inner vmap adds the axis for b, so a row has shape (M). The outer vmap adds the axis for a in front of that, so the grid has shape (N, M), with the point of a selecting the row and the point of b selecting the column. Swapping which vmap maps which argument produces the transposed (M, N) grid.

With a holding $(1, 2)$ and $(0, 1)$ and b holding the three points above, the grid of squared distances is

$$
\begin{pmatrix} 8 & 0 & 5 \\ 10 & 2 & 1 \end{pmatrix} ,
$$

and the second row of cosines is 0, 0.8944 and 0.7071. The first row of the grid is exactly the row computed in the previous section.

## What the nested vmap compiles to

Tracing the nested squared distance for (4, 3) arrays a and b gives a program of five operations. Two of them insert unit axes to give arrays of shape (4, 1, 3) and (1, 4, 3), one subtracts them into a (4, 4, 3) array of differences, one squares it and one sums over the last axis. That is the broadcasting program an experienced NumPy user would write by hand, produced from a function that only knows about two vectors.

A Python double loop over the points produces the same numbers, but tracing unrolls it. A double loop that calls the pair function for every pair and stacks each row with jnp.stack records 117 operations for 4 by 4 points and 457 for 8 by 8 points, while the nested vmap records 5 operations for both sizes. Under jax.jit the compile time grows with the length of the traced program, so loops over data points make compilation slower as the data grows.

The intermediate array of differences has N times M times D entries. That is harmless for small sets, and for large sets it is the reason libraries switch to the expansion described next or process the grid in blocks.

## A faster formula that loses precision

Large-scale libraries often compute squared distances through the expansion

$$
\lVert u - w \rVert^2 = \lVert u \rVert^2 + \lVert w \rVert^2 - 2\, u \cdot w ,
$$

because the cross terms for all pairs form one matrix product. The expansion subtracts large numbers to get a small one, which loses precision in float32. For $u = (1000, 1000)$ and $w = (1000.01, 1000)$, the direct sum of squared differences gives about 0.0001002 in float32, which matches the stored coordinates, while the expansion gives exactly 0. The direct formula in the pair function avoids this cancellation, and for small point sets its extra cost is negligible.

## Integer inputs

Points sometimes arrive as integer arrays, for example coordinates on a grid. In JAX, subtracting, squaring and summing int32 arrays gives an int32 result, and the dot product of two int32 vectors is int32 as well. The norm returns float32, so a cosine computed from int32 inputs comes out float32 while the squared distance stays int32. Converting both inputs to float32 at the start gives both results the same float32 dtype.

## Common mistakes to avoid

- **Nesting the maps in the wrong order.** The outer vmap sets axis 0 of the result, so mapping b on the outside gives an (M, N) grid that is the transpose of the one wanted.
- **Mapping both arguments in one vmap.** With in_axes equal to (0, 0), point i of a is compared only with point i of b, which gives the diagonal of the grid and fails when N differs from M.
- **Taking the square root.** The squared distance has no square root, and taking one returns the ordinary Euclidean distance.
- **Dividing by a zero norm.** Without a floor on the norm product, a zero vector produces NaN cosines for its whole row or column.
- **Leaving integer inputs as integers.** The squared distance of int32 points stays int32 unless the inputs are converted to float32.
- **Looping over points in Python.** A double loop unrolls into a traced program that grows with N times M and compiles slowly.