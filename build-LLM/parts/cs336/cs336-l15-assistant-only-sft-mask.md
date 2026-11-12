# Build an Assistant-Only SFT Loss Mask

Supervised fine-tuning (SFT) teaches a pretrained model to answer like an assistant by training it on conversations. Each conversation mixes user turns, assistant turns and special formatting tokens, but the model should only learn to write the assistant's part. The heart of this problem is building next-token labels that keep exactly the assistant targets and mark everything else to be ignored by the loss.

## Why train only on assistant tokens

A chat transcript contains text the model will never be asked to produce, such as the user's questions and system instructions. Training on those tokens would teach the model to imitate users, to write their questions for them or to continue a conversation by speaking as the user. It also spends part of the training signal on predicting prompts, which does nothing for answer quality.

Masking the loss so only assistant tokens count keeps the training focused on the behavior you want. The model still reads the whole conversation as context. It is never graded on predicting the parts it did not write.

## Eligibility follows the token being predicted

A language model at position $t$ predicts the token at position $t + 1$. The label for position $t$ is therefore the next token in the sequence, and the last position has no next token to predict.

Each token carries a role, such as user, assistant or a special formatting role. A target is kept only when the token being predicted, the one at position $t + 1$, has the assistant role. The role of the token at position $t$, the one doing the predicting, does not matter. The model may be reading the last token of the user's question when it predicts the first token of the answer, and that prediction is exactly what SFT should teach.

Both positions must also be real tokens. Batches are padded to a common length, and an attention mask marks which positions are real. A padded target is not a real token to learn, and a padded source position carries no real context, so a prediction made from it is meaningless even if the next token is real. Either case excludes the transition.

## Marking ignored positions with -100

Every excluded position gets the label $-100$. This is the default ignore value for cross-entropy in PyTorch, which skips those positions entirely, so they contribute neither loss nor gradient. Included positions get the next token's ID as their label.

The loss mask is a Boolean version of the same decision, true exactly where a real label was kept. It is useful for counting how many tokens were trained on, for example to average the loss over assistant tokens only.

## A short conversation

Take one sequence of seven tokens with IDs $[5, 6, 7, 8, 9, 2, 0]$. Token 5 is a formatting token, 6 is the user's message, 7 is a formatting token that starts the assistant turn, 8 and 9 are the assistant's reply, 2 is an end-of-turn token marked as assistant, and 0 is padding. With formatting as role $-1$, user as role 0 and the assistant role as 1, the roles are $[-1, 0, -1, 1, 1, 1, -1]$, and the attention mask is true at the first six positions and false at the last.

Each label comes from checking the token one position to the right.

- Positions 0 and 1 predict tokens 6 and 7, which are not assistant tokens, so their labels are $-100$.
- Position 2 predicts token 8, an assistant token, so its label is 8.
- Positions 3 and 4 predict tokens 9 and 2, both assistant tokens, so their labels are 9 and 2.
- Position 5 predicts the padding token, which is unattended, so its label is $-100$.
- Position 6 is the last position, so its label is $-100$.

The labels are $[-100, -100, 8, 9, 2, -100, -100]$, and the loss mask is true at positions 2, 3 and 4. Position 2 is a formatting token predicting the first word of the reply, which shows why the source role does not matter. The end-of-turn token is trained too, which is how the model learns when to stop talking.

## Common mistakes to avoid

- **Checking the source token's role.** Eligibility depends on the role of the target token at position $t + 1$.
- **Forgetting the shift.** The label at position $t$ is the token at $t + 1$, which also means the last position has nothing to predict.
- **Training across padding.** A transition is excluded if either position is unattended.
- **Using 0 for ignored labels.** Token 0 is a real token, while $-100$ is what the loss skips.
- **Special-casing special tokens.** Eligibility comes only from the role, so an end-of-turn token marked as assistant is trained like any other assistant token.
