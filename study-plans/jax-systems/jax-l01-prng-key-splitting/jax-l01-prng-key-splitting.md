# PRNG Key Splitting

Dropout, weight initialization and data shuffling all need random numbers, and those numbers have to be reproducible so that a run can be debugged or repeated. NumPy keeps its random state hidden inside a global generator that moves forward with every draw. JAX instead passes the random state around as an ordinary value called a key, and gives you a way to split one key into several new ones. This page explains why JAX works this way, how keys are created and split, and how to give every layer of a network its own dropout mask from a single seed.

## Hidden state and compiled code

A call to np.random.rand reads the global generator state and moves it forward. Two calls return different numbers because something outside the function changed between them. A change like that is called a side effect, and side effects clash with the way JAX compiles code.

When you wrap a function in jax.jit, JAX calls the Python function once with placeholder arrays called tracers, records the array operations, and compiles that record. Any NumPy random call runs during that one recording, and its result is stored in the compiled program as a fixed constant. After np.random.seed(0), a jitted function that adds np.random.rand() to its input returns 0.5488135 for an input of 0 on the first call, and again 0.5488135 on the second call. The draw was frozen at compile time.

The problem grows with the other transformations. jax.vmap runs a function over a batch axis, and sharded programs run on many devices at once. A single global stream gives no clean answer to which numbers belong to which example or which device.

## A key is a value

JAX makes the generator state an explicit input. The call jax.random.key(seed) builds a key from an integer, and every random function takes a key as its first argument. A random function is then a pure function of the key, the shape and its other arguments, which means the same key always gives the same numbers.

Drawing four uniform numbers between 0 and 1 from the key for seed 7 gives about [0.6741, 0.9746, 0.3028, 0.4440], and drawing again from the same key gives exactly the same four numbers. Inside, that key holds just two 32-bit unsigned integers, 0 and 7. Because a key is ordinary data, it can be passed into a jitted function as an argument, and each call can receive a different key.

## Reusing a key repeats the randomness

Purity has a direct consequence. If two draws use the same key and the same shape, they return the same numbers. Drawing dropout masks for two layers from the seed 7 key, each with keep probability 0.5 and four entries, gives [False, False, True, True] for both layers.

Dropout relies on making fresh random choices at every place it is applied. Identical masks drop the same positions in every layer, which correlates noise that is supposed to be independent and weakens the regularization. The working rule in JAX is that a key is used for at most one draw.

## Splitting one key into fresh keys

The function jax.random.split turns one key into new keys that are statistically independent of each other and of the original. By default it returns two keys, and split(key, n) returns n of them. Splitting the seed 7 key and drawing four uniform numbers from each half gives about [0.8674, 0.5563, 0.3007, 0.3325] for the first half and [0.6751, 0.8262, 0.3940, 0.4243] for the second. Neither matches the draw from the parent key.

After a split, the parent key counts as used. The usual pattern keeps one of the new keys as the running key, which will be split again later, and spends the other on a draw. Following that pattern, every draw in a program gets its own key and no key is ever used twice.

## How a counter-based generator makes splitting work

The default JAX generator, called Threefry, is counter-based. It produces random bits by applying a scrambling function to the key together with a position counter, so there is no state that has to move forward. Splitting applies the same kind of scrambling to derive new keys. Every random number is therefore a deterministic function of a key, so any device can compute its own numbers without coordination, and the result never depends on the order in which operations happen to run.

## Two splitting rules give different masks

Because every key comes from a fixed sequence of splits, the splitting rule decides every number. Two common rules both avoid key reuse but produce different masks.

- **Sequential splitting.** For each layer, split the running key, keep the first new key as the running key and use the second for the layer.
- **Splitting all at once.** Split the seed key into one key per layer in a single call and give key i to layer i.

Take the seed 7 key, two layers and a keep probability of 0.6. The sequential rule uses the second half of the first split for layer 0, whose uniform draws are about [0.6751, 0.8262, 0.3940, 0.4243] and whose mask is [False, False, True, True]. For layer 1 it splits the kept first half again, and the new layer key gives about [0.9876, 0.0197, 0.0451, 0.9365] and the mask [False, True, True, False]. The all-at-once rule gives layer 0 the mask [False, True, True, True] and layer 1 the mask [False, False, True, True]. For two keys, split(key, 2) is the same call as split(key), so the all-at-once rule hands out the same two keys in the other role, and its layer 1 mask equals the sequential layer 0 mask.

Both rules are sound, and a codebase only has to pick one and keep it, since changing the rule changes every random number. This problem fixes the sequential rule.

## From uniform numbers to a mask

A Bernoulli draw with probability $p$ returns True with probability $p$ and False otherwise. One way to produce it is to draw a uniform number $u$ in the range from 0 up to, but not including, 1 and return True when $u < p$. The function jax.random.bernoulli makes exactly this comparison. With the sequential layer 0 draws above and $p = 0.6$, the entries 0.6751 and 0.8262 are not below 0.6 and the entries 0.3940 and 0.4243 are, which gives [False, False, True, True].

Two edge values behave predictably. With $p = 1$ every uniform number is below $p$, so every entry is kept, and with $p = 0$ none is.

## Scaling the kept activations

A full dropout layer multiplies the activations by the mask and divides the survivors by the keep probability, a convention called inverted dropout. With a keep probability of 0.8, a kept activation of 2.0 becomes 2.5, and its average over many draws is $0.8 \cdot 2.5 + 0.2 \cdot 0 = 2.0$, the same as without dropout. This problem stops at the mask, which is the part the keys decide.

## Keys in a training loop

A training step usually receives a key as an argument, splits off one key for every random operation it performs, and passes a fresh key to the next step. The function jax.random.fold_in derives a new key by mixing an integer, such as the step number, into an existing key. Either way, every random choice in the program traces back to one seed through a fixed path of splits.

## Common mistakes to avoid

- **Reusing one key for every layer.** The same key gives the same draw, so every layer receives an identical mask.
- **Swapping the two halves of each split.** Drawing from the first key and keeping the second is a valid scheme, but it produces different masks from the stated rule.
- **Splitting all layer keys at once.** One call to split with a count produces a different set of layer keys than the sequential rule.
- **Adding the layer number to the seed.** With seed plus layer, seed 0 at layer 1 and seed 1 at layer 0 share a key, so neighbouring runs repeat each other's masks.
- **Calling NumPy random inside jitted code.** The draw runs once during tracing and becomes a constant in the compiled program.
- **Passing the drop rate as keep_prob.** A drop rate of 0.1 passed as the keep probability keeps only a tenth of the entries.