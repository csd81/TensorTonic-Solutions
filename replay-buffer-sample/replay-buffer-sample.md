## What Is a Replay Buffer?

A replay buffer (also called experience replay memory) is a data structure that stores past experiences and allows **random sampling** for training.

Instead of learning from experiences in the order they occur, the agent samples random minibatches from the buffer. This technique dramatically improves learning stability and efficiency.

Introduced in the context of Deep Q-Networks (DQN) by Mnih et al. (2013).

---

## Why Experience Replay?

Training neural networks with reinforcement learning faces several challenges:

**1. Correlated data:**

Consecutive experiences are highly correlated (same region of state space). Neural networks learn poorly from correlated data.

**2. Non-stationary distribution:**

As the policy improves, the data distribution changes. This can cause catastrophic forgetting.

**3. Sample inefficiency:**

Each experience is used only once and then discarded.

Experience replay addresses all three issues.

---

## How It Works

**1. Store:** After each transition $(s, a, r, s', \text{done})$, add it to the buffer.

**2. Sample:** When training, randomly sample a minibatch of transitions from the buffer.

**3. Learn:** Update the neural network using the sampled minibatch.

**4. Manage:** When the buffer is full, remove old experiences (typically FIFO).

---

## The Transition Tuple

Each experience stored in the buffer is a tuple:

$$
(s_t, a_t, r_{t+1}, s_{t+1}, \text{done})
$$

**Components:**
- $s_t$: Current state
- $a_t$: Action taken
- $r_{t+1}$: Reward received
- $s_{t+1}$: Next state
- done: Boolean indicating if episode ended

This is all the information needed for a TD update.

---

## Buffer Operations

**Initialization:**
- Create buffer with maximum capacity $N$
- Buffer is initially empty

**Push (store):**
- Add new transition to buffer
- If buffer is full, remove oldest transition

**Sample:**
- Randomly select $B$ transitions (batch size)
- Return as arrays/tensors for training

**Size:**
- Return current number of stored transitions

---

## Uniform Random Sampling

The simplest approach samples each transition with equal probability:

$$
P(\text{select transition } i) = \frac{1}{|\text{buffer}|}
$$

**Process:**
1. Generate $B$ random indices from $[0, |\text{buffer}| - 1]$
2. Retrieve transitions at those indices
3. Return as minibatch

**Properties:**
- Simple to implement
- All experiences equally likely
- May oversample or undersample certain transitions

---

## Worked Example: Sampling

**Buffer contents (capacity 5):**

Position 0: $(s_1, a_1, r_1, s_2, \text{False})$

Position 1: $(s_2, a_2, r_2, s_3, \text{False})$

Position 2: $(s_3, a_3, r_3, s_4, \text{False})$

Position 3: $(s_4, a_4, r_4, s_5, \text{True})$

Position 4: $(s_5, a_5, r_5, s_6, \text{False})$

**Sample batch of size 3:**

Random indices: [2, 0, 4]

**Returned batch:**
- $(s_3, a_3, r_3, s_4, \text{False})$
- $(s_1, a_1, r_1, s_2, \text{False})$
- $(s_5, a_5, r_5, s_6, \text{False})$

---

## Buffer Size Considerations

**Small buffer (e.g., 10,000):**
- More recent experiences dominate
- Adapts quickly to policy changes
- May forget useful old experiences
- Lower memory usage

**Large buffer (e.g., 1,000,000):**
- More diverse experiences
- More stable training
- May contain outdated experiences
- Higher memory usage

**Typical values:** 100,000 to 1,000,000 transitions

DQN uses 1 million transitions.

---

## Circular Buffer Implementation

The most common implementation uses a circular (ring) buffer:

**Variables:**
- Array of size $N$ to store transitions
- Write pointer indicating next position to write
- Current size (until buffer is full)

**Push operation:**
1. Store transition at write pointer position
2. Increment write pointer: $\text{ptr} = (\text{ptr} + 1) \mod N$
3. Update size: $\text{size} = \min(\text{size} + 1, N)$

**Properties:**
- O(1) push
- O(B) sample for batch size B
- Fixed memory footprint

---

## Breaking Correlation

Without replay, consecutive samples are correlated:

$$
(s_1, a_1, r_1, s_2), (s_2, a_2, r_2, s_3), (s_3, a_3, r_3, s_4), ...
$$

State $s_2$ appears as both "next state" and "current state" in adjacent samples.

**With replay:** Random sampling decorrelates the data:

$$
(s_{47}, a_{47}, r_{47}, s_{48}), (s_{12}, a_{12}, r_{12}, s_{13}), (s_{89}, a_{89}, r_{89}, s_{90}), ...
$$

This is similar to i.i.d. sampling assumption in supervised learning.

---

## Data Efficiency

Without replay, each experience is used once:

**Updates per experience:** 1

With replay, each experience can be sampled multiple times:

**Updates per experience:** Many (until removed from buffer)

This dramatically improves sample efficiency, especially important when environment interactions are expensive.

---

## Prioritized Experience Replay

Not all experiences are equally valuable for learning. Prioritized replay samples important transitions more frequently.

**Priority:** Often based on TD error magnitude:

$$
p_i = |\delta_i| + \epsilon
$$

where $\delta_i$ is the TD error for transition $i$.

**Sampling probability:**

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

$\alpha$ controls how much prioritization affects sampling (0 = uniform, 1 = full prioritization).

---

## Importance Sampling Correction

Prioritized sampling introduces bias. To correct:

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
$$

where $\beta$ is annealed from 0 to 1 during training.

The loss is weighted:

$$
L = \frac{1}{B}\sum_{i=1}^{B} w_i \cdot (y_i - Q(s_i, a_i))^2
$$

---

## When to Start Learning

The buffer should have enough diverse experiences before training begins:

**Warm-up period:**
- Fill buffer with $N_{\text{min}}$ transitions before first update
- Use random policy during warm-up

**Typical values:** 10,000 to 50,000 initial transitions

This ensures the first minibatches are reasonably diverse.

---

## Update Frequency

**Every step:** Update network after each new transition
- Most responsive
- Higher computation

**Every N steps:** Update after N transitions
- More efficient
- May lag behind new experiences

**DQN approach:**
- Store every transition
- Update every 4 transitions
- Sample batch of 32

---

## Replay Buffer for Different Algorithms

**DQN:**
- Stores $(s, a, r, s', \text{done})$
- Samples for Q-learning updates

**DDPG (continuous actions):**
- Same structure
- Used for actor-critic updates

**SAC:**
- Same structure
- Samples for soft actor-critic updates

**PPO (on-policy):**
- Uses recent rollout data, not a large buffer
- Data discarded after each update

---

## Memory Considerations

**State representation matters:**

If states are images (84x84x4 = 28,224 floats):
- 1M transitions $\times$ 28,224 $\times$ 4 bytes $\approx$ 113 GB

**Optimizations:**
- Store states as uint8 (1 byte) instead of float32
- Store only current state, compute next state
- Compress states
- Use lazy frames (store frames once, reference multiple times)

---

## Handling Episode Boundaries

When sampling transitions across episode boundaries:

**Terminal transitions:** $(s, a, r, s', \text{done}=\text{True})$

The next state $s'$ is meaningless (episode ended). During training:

$$
\text{target} = r \quad \text{(not } r + \gamma \max_a Q(s', a) \text{)}
$$

The done flag tells us not to bootstrap from the next state.

---

## Advantages of Experience Replay

**1. Data efficiency:** Reuse experiences multiple times

**2. Decorrelation:** Random sampling breaks temporal correlation

**3. Stability:** Diverse minibatches stabilize neural network training

**4. Off-policy learning:** Can learn from old policy experiences

---

## Disadvantages and Limitations

**1. Memory usage:** Large buffers require significant RAM

**2. Stale data:** Old experiences may be from very different policies

**3. On-policy incompatibility:** Does not work with purely on-policy methods

**4. Delayed learning:** Cannot learn immediately from new experiences

---

## Variations and Extensions

**Hindsight Experience Replay (HER):**
- For sparse reward settings
- Relabels failed experiences as successes for different goals

**Combined Experience Replay (CER):**
- Always includes most recent transition in batch
- Combines recency with diversity

**Distributed Replay:**
- Multiple actors fill a shared buffer
- Learner samples from the combined buffer