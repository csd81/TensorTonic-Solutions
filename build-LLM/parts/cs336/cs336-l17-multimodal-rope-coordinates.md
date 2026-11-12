# Assign Multimodal RoPE Coordinates

A multimodal language model reads text, images and video as one long token sequence, but an image is a two-dimensional grid and a video adds time. Rotary position embeddings normally use a single position number per token, which throws away that structure. Multimodal RoPE, used in Qwen2-VL (Wang et al., 2024), gives every token three coordinates for time, height and width. The heart of this problem is assigning those coordinates segment by segment with one shared offset that keeps positions increasing across the sequence.

## Why one position number is not enough

If the patches of an image are numbered 0, 1, 2 and so on in reading order, two patches stacked vertically get positions far apart, while the last patch of one row and the first patch of the next get adjacent positions. Relative position then says almost nothing about where patches sit in the picture.

Giving each token separate height and width coordinates fixes this. Rotary embeddings already rotate pairs of a head's dimensions by an angle proportional to position. Multimodal RoPE splits those pairs into three groups and rotates each group by one coordinate, so attention sees relative time, relative row and relative column separately. This problem computes only the coordinates that feed those rotations.

## One offset runs through the whole sequence

Let $o$ be the first position not yet used by any earlier segment. It starts at 0, and every segment advances it past the coordinates it used, so later segments always land after earlier ones along every axis.

For text, all three coordinates are the same. A text token at position $j$ within a segment gets

$$
(t, h, w) = (o + j, \ o + j, \ o + j)
$$

so text behaves exactly like standard one-dimensional RoPE, and a separator is treated as a single text token.

## Images share one time step

An image of $H$ by $W$ patches is a single moment, so every patch gets the same time coordinate. The patch at row $r$ and column $c$ gets

$$
(t, h, w) = (o, \ o + r, \ o + c)
$$

so neighbors in the picture differ by exactly 1 in the matching coordinate. After the image, the offset advances by $\max(H, W)$, the largest coordinate span it used. That keeps the next segment's positions above every coordinate the image produced. Advancing by the patch count instead would push later text far away along every axis, making a caption look distant from the image it describes.

## Videos add time

A video has $F$ frames, each an $H$ by $W$ grid, and frame $f$ gets time coordinate $o + f s$ for a frame stride $s$:

$$
(t, h, w) = (o + f s, \ o + r, \ o + c)
$$

Qwen2-VL raises the time coordinate by one per frame. This exercise adds the stride $s$ so time coordinates can track real elapsed time when frames are sampled sparsely, an idea later models such as Qwen2.5-VL use. The offset then advances by the largest span on any axis, $\max((F - 1) s + 1, H, W)$, which covers the full time range as well as the spatial one.

## A separator, an image and some text

Take a separator, then an image of 1 row and 2 columns, then 2 text tokens.

The separator starts at offset 0 and gets $(0, 0, 0)$, moving the offset to 1. The image's two patches share time 1 and row 1, with columns 1 and 2, so they get $(1, 1, 1)$ and $(1, 1, 2)$. The offset advances by $\max(1, 2) = 2$, to 3. The text tokens get $(3, 3, 3)$ and $(4, 4, 4)$.

Stacking the coordinates as rows gives

$$
\begin{bmatrix} 0 & 1 & 1 & 3 & 4 \\ 0 & 1 & 1 & 3 & 4 \\ 0 & 1 & 2 & 3 & 4 \end{bmatrix}
$$

for time, height and width. The two image patches differ only in width, which is exactly what a horizontal neighbor should look like. The first text token starts at 3, past the image's largest coordinate of 2, and it is only one step beyond that edge, so text right after an image still counts as close to it.

## Common mistakes to avoid

- **Advancing the offset by the patch count.** Later text would then sit far from the image along the height and width axes.
- **Giving image patches different time coordinates.** Every patch of an image shares the same time step.
- **Using raw $f$ instead of $o + f s$ for video time.** Time coordinates would restart near zero and collide with earlier tokens.
- **Forgetting the time span when advancing past a video.** A long video can span more time positions than its height or width, and the offset must clear all of them.
- **Using one coordinate for all three axes on images.** That collapses the grid back into a line and loses the row and column structure.
