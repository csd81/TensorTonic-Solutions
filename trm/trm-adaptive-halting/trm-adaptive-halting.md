# Adaptive Halting

Adaptive halting decides after every supervision step whether an example may leave the batch. The Tiny Recursive Model (TRM) learns one halting probability, while the Hierarchical Reasoning Model (HRM) uses a Q-learning rule that needs a second forward pass. This problem implements one batched halting step as the released TRM code runs it, with the HRM rule available as a comparison mode.

## Deep supervision needs a way out

Deep supervision reuses the detached answer $y$ and latent $z$ across up to $N_{sup} = 16$ supervision steps, where each step is one optimizer update. Spending all 16 on every minibatch is expensive, so HRM introduced adaptive computational time (ACT), and Section 2.5 reports fewer than 2 steps per Sudoku-Extreme example on average during training.

Figure 3 of the paper writes the early stop as a single break for the whole minibatch. The released code instead halts row by row (trm.py lines 252 to 287). A halted row is reset to the fixed initial carry $(y_{init}, z_{init})$, receives a fresh example and counts from zero, while the other rows keep refining. This problem uses that per-row behavior, with a halted flag $r_b$ and step count $k_b$ per row.

## HRM learns halting with Q-learning

HRM reads two logits from its answer state, a halting logit $q^h$ and a continue logit $q^c$, and halts a row when $q^h > q^c$. The continue head is trained toward a bootstrapped target built from the next state, written with the logistic sigmoid $\sigma$. That target is $\sigma(\tilde{q}^h)$ at the last step and $\sigma(\max(\tilde{q}^h, \tilde{q}^c))$ before it, where $\tilde{q}$ comes from running the model again on the carry it just produced. Section 3.2 notes that this extra pass doubles the forward passes per optimization step.

The comparison mode here keeps the TRM inner forward and swaps only the halting rule, so the difference isolates ACT. HRM's own inner forward uses two networks at two frequencies and a 1-step gradient. The TRM code's HRM-style ACT flag raises an error at trm.py line 294, so this problem reproduces the target from hrm.py lines 288 to 292. As in that code, the second pass runs only in training with $N_{sup} > 1$.

## TRM keeps one halting probability

Section 4.6 drops the continue loss and trains the halting logit with a binary cross-entropy on whether the current answer is exactly correct, the target from the Supervision Loss problem. A row halts when $q^h > 0$, which is the same as $\sigma(q^h) > 0.5$. With no bootstrapped target there is no second pass, and Table 1 reports accuracy moving from 86.1% to 87.4%. The released head still outputs two logits, and TRM mode uses only the first.

## Halting only matters during training

Section 2.5 states that HRM halts only during training and runs all $N_{sup}$ steps at evaluation, and trm.py lines 275 to 278 apply the same rule to TRM. Outside training the halted flag is the limit indicator $\lambda_b = [k^1_b \ge N_{sup}]$, where $k^1_b$ is the new step count, which keeps every row in lockstep.

During training, the code picks about one row in ten and gives it a random minimum step count in $[2, N_{sup}]$ (trm.py line 286). This exploration floor $m_b$ stops the halting head from always quitting at step 1 and keeps showing the model longer trajectories. Here the floor arrives as concrete integers, with zero meaning no floor. The stand-in network in the supplied inner forward omits token mixing for cheap checking, and the Tiny Mixer Network problem covers the full block.

## Two rows at the edge of the budget

Take $N_{sup} = 3$ in training. Row A halted before, so it restarts and reaches count 1, while row B goes from 2 to 3, so $\lambda = (0, 1)$. Suppose the logits are $q^h = (0.4, -0.7)$ and $q^c = (0.9, 0.2)$.

- **TRM rule.** The signal is $[q^h > 0] = (1, 0)$, so with no floor both rows halt, row A by its signal and row B by the limit.
- **HRM rule.** The signal is $[q^h > q^c] = (0, 0)$, so only row B halts. With next logits $\tilde{q}^h = (0.5, -0.4)$ and $\tilde{q}^c = (-0.1, 1.2)$, row A gets $\sigma(\max(0.5, -0.1)) = 0.6225$ and row B, at the limit, gets $\sigma(-0.4) = 0.4013$.
- **Exploration floor.** A floor of 2 on row A blocks its TRM halt, because its count of 1 is below it.

## Common mistakes to avoid

- **Resetting rows that did not halt.** Only rows flagged on the previous step take the initial carry and new inputs.
- **Keeping the old step count.** A reset row must count from zero, or it hits the limit early.
- **Halting in evaluation mode.** Outside training only the step limit decides, even when the halting logit is large.
- **Using the HRM rule in TRM mode.** A positive halting logit halts a TRM row even when its continue logit is larger.
- **Comparing the floor strictly.** A row whose new count equals its floor is allowed to halt.
- **Running the second HRM pass on the reset carry.** The target looks one step ahead, so it must start from the carry the first pass returned.
- **Taking the maximum at the last step.** At the limit there is no future step, so the target uses the next halting logit alone.