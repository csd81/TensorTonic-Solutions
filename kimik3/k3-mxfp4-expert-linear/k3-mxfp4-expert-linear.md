# MXFP4 Routed Expert Linear

This problem performs a routed expert linear layer whose weights are stored in MXFP4. The storage format packs two tiny floating-point values into each byte and shares one scale across a group of 32 weights. Only the experts selected for each token should be reconstructed and applied.

## What is stored in one group

An ordinary floating-point matrix stores each weight directly. MXFP4 separates a group into two parts:

- Sixteen bytes containing 32 four-bit E2M1 value codes.
- One E8M0 scale byte shared by those 32 values.

Each byte has a low nibble and a high nibble. Decode the low nibble first, then the high nibble. This interleaved order is part of the format. Decoding all low nibbles followed by all high nibbles would scramble the reconstructed weight positions.

The four-bit code is an index into the supplied 16-entry E2M1 value table. Codes represent the magnitudes zero, one half, one, one and a half, two, three, four, and six, followed by their signed versions.

## Apply the shared group scale

If the scale byte is $s$, every decoded value in its group is multiplied by

$$
2^{s-127}
$$

The subtraction by 127 interprets the byte with an exponent bias. A scale byte of 127 gives a factor of one, 128 gives two, and 126 gives one half.

The scale belongs to one expert, one output row, and one group. Do not share it across output rows or neighboring groups. After scaling, concatenate the groups in order to reconstruct one complete row of the expert matrix.

## Reconstruct the matrix in the right orientation

For each selected expert, the decoded matrix has one row per output feature and one column per latent input feature. If there are $G$ groups, the input width is $32G$ because every group contributes 32 values.

The token is multiplied by the transpose relationship implied by this row layout: each output coordinate is the dot product between the input token and one reconstructed matrix row.

You can reason about one output row at a time. Decode its first group into input positions 0 through 31, decode its second group into positions 32 through 63, and continue until the full row is restored.

## Route, weight, and add the shared path

Each token supplies a list of selected expert indices and matching mixture weights. For selected expert $e$, compute its linear output $W_ex_t$, multiply by the route weight, and add it to the token's routed sum.

The final output is

$$
y_t = y_t^{\mathrm{shared}} + \sum_{e\in\mathcal{T}_t}p_{t,e}W_ex_t
$$

The shared output is already computed and is not quantized or decoded by this function. Start from it and add the selected routed contributions.

## A tiny decoding example

Consider one packed byte with hexadecimal value $0x21$. Its low nibble is code 1 and its high nibble is code 2. From the supplied table, these decode to $0.5$ and $1.0$ in that order.

If the group's scale byte is 128, the scale is $2^{128-127}=2$. The two reconstructed values become $1.0$ and $2.0$.

If the byte were decoded high first, those positions would become $2.0$ and $1.0$. The numbers are the same as a set, but the matrix is different, so the linear output would usually be wrong.

## Decode only selected experts

The prompt specifically asks for selected expert reconstruction. This matches sparse routing: weights belonging to experts that receive no token do not affect the result. A simple correct implementation can cache a reconstructed matrix after an expert is first selected so repeated selections do not decode it again.

The returned tensor should use the latent token's floating-point dtype and device. Packed bytes and scale bytes are storage data, so intermediate conversions must not accidentally force the final result onto the CPU or into an integer type.

## Implementation order

- Start from a non-mutating copy of the supplied shared output.
- For each selected expert that has not yet been decoded, split every packed byte into low and high nibbles.
- Map codes through the E2M1 table in low-then-high order.
- Apply the correct E8M0 scale to each 32-value group and concatenate groups into matrix rows.
- Multiply each token by its selected expert matrices, weight the results, and accumulate them.
- Add routed contributions to the shared output and return the floating-point result.

## Common mistakes to avoid

- **Decoding the high nibble first.** The required order is low, then high, for every byte.
- **Using one scale for a whole matrix.** Every group has its own scale byte.
- **Applying the exponent bias with the wrong sign.** The factor is $2^{s-127}$.
- **Reconstructing every expert.** Only selected experts are needed.
- **Quantizing the shared output.** It is already supplied in higher precision and should be added directly.
