# Patchify a Batch of Images

A transformer works on a sequence of tokens, and an image is a grid of pixels. The Vision Transformer (Dosovitskiy et al., 2021) bridged the two by cutting each image into small square patches and treating every patch as one token. The heart of this problem is rearranging an image batch into a sequence of flattened patches in a fixed order, along with each patch's grid position.

## Why patches instead of pixels

Attention compares every token with every other token, so its cost grows with the square of the sequence length. A 224 by 224 image has over 50,000 pixels, and treating each pixel as a token would make attention far too expensive.

Grouping pixels into 16 by 16 patches turns the same image into a 14 by 14 grid of 196 tokens, a sequence length transformers handle easily. Each patch's pixels are flattened into one vector, and a linear layer then projects it to the model's width, playing the role a token embedding plays for words. Multimodal language models use the same idea to turn images into tokens.

Patch size is a trade-off. Halving the patch side to 8 pixels gives four times as many tokens, sixteen times the attention cost, and finer detail in each token.

## The patch grid

An image batch has shape $(B, C, H, W)$, for batch size, channels, height and width. With patches of height $p_h$ and width $p_w$ that divide the image exactly, there are $H / p_h$ patch rows and $W / p_w$ patch columns, so each image becomes

$$
N = \frac{H}{p_h} \times \frac{W}{p_w}
$$

patches. Patches never overlap, and together they cover every pixel exactly once.

Patches are ordered like reading text, row by row and left to right within a row. Each patch also gets its grid coordinate, its patch row and column, which position embeddings use to tell the model where the patch came from. These coordinates are the same for every image in the batch.

## Flattening one patch

Each patch is a small block of shape $(C, p_h, p_w)$, flattened into a vector of length $C p_h p_w$ with all of one channel's pixels in reading order before the next channel's. Any fixed order would let a model learn, but it must match the order the projection weights were trained with. Loading pretrained ViT weights with a different flattening order would scramble every patch.

## Why a plain reshape is not enough

In memory, a row of pixels runs straight across every patch in that row. Reading the image in storage order and chopping it into pieces of length $C p_h p_w$ would therefore mix pixels from neighboring patches. The patch structure has to be exposed as separate axes, for patch row and column and for position within a patch, and those axes reordered before any flattening. Tensor reshapes and axis reordering do this for the whole batch at once, with no loop over pixels.

## A two-channel example

Take one image with two channels, height 2 and width 4, and patches of height 2 and width 2. The first channel holds rows $[0, 1, 2, 3]$ and $[4, 5, 6, 7]$, and the second holds $[10, 11, 12, 13]$ and $[14, 15, 16, 17]$.

The image has one patch row and two patch columns, so $N = 2$ patches, each of length $2 \times 2 \times 2 = 8$. The left patch takes columns 0 and 1. Its first channel contributes 0, 1, 4 and 5 in reading order, and its second channel contributes 10, 11, 14 and 15:

$$
[0, 1, 4, 5, 10, 11, 14, 15]
$$

The right patch takes columns 2 and 3 and becomes $[2, 3, 6, 7, 12, 13, 16, 17]$. The coordinates are $[0, 0]$ and $[0, 1]$, one patch row with two columns. Reading the first channel in storage order would instead have produced 0, 1, 2, 3 as the start of the first patch, mixing both patches.

## Common mistakes to avoid

- **Flattening pixels in the wrong order.** Inside a patch the order is channel, then local row, then local column.
- **Ordering patches column by column.** Patches go row by row, left to right within each row.
- **Reshaping straight to patches without reordering.** A single reshape interleaves pixels from different patches.
- **Swapping height and width when splitting.** Patches come out transposed and pixels from different rows mix.
