# AlphaGo Zero-lite

A runnable AlphaGo Zero-style Go engine assembled from the 18 solutions in
`study-plans/alphago-from-scratch/`. The copied solution files and their
problem write-ups are included unmodified. `alphago_system.py` adds the game
state, the on-the-fly MCTS tree, self-play, training, evaluation,
checkpointing, and a CLI.

The shipped model is a 5×5 Go agent with a 45k-parameter policy-value network.
It plays legal Go with captures, positional superko, passes, and Tromp-Taylor
scoring, and it wins **35/40 games (87.5%)** against random play.

## Quick start

```bash
cd /home/d/Desktop/TensorTonic-Solutions/build_AlphaGo
python alphago_system.py selftest          # checks all 18 copied parts + glue
python alphago_system.py play              # play against the shipped best model
python alphago_system.py play --ai-vs-ai   # watch a complete game
python alphago_system.py eval --games 40 --simulations 32
```

Training from scratch (CPU, about 5 minutes):

```bash
python alphago_system.py train --out runs/alphago
```

## What is copied and what is added

Copied unmodified from the study plan:

| Copied file | Role in the system |
|---|---|
| `alphago-l01-group-liberties.py` | group/liberty queries (self-test) |
| `alphago-l01-legal-action-mask.py` | legal moves, suicide, positional superko, pass |
| `alphago-l01-resolve-move.py` | apply a move, captures, pass, terminal move limit |
| `alphago-l01-tromp-taylor-result.py` | final area scoring with komi |
| `alphago-l02-encode-board-history.py` | 17-plane board history from the player to move |
| `alphago-l02-compact-policy-value-network.py` | full policy/value forward pass |
| `alphago-l02-policy-value-heads.py` | standalone heads (self-test) |
| `alphago-l02-residual-block.py` | standalone residual block (self-test) |
| `alphago-l03-puct-selection.py` | PUCT action selection |
| `alphago-l03-initialize-edge-statistics.py` | legal-softmax priors and edge arrays |
| `alphago-l03-backup-mcts-path.py` | alternating-perspective backup rule (mirrored in the fast loop) |
| `alphago-l04-policy-value-loss.py` | policy CE + value MSE + L2 loss |
| `alphago-l04-root-dirichlet-noise.py` | root exploration noise |
| `alphago-l04-visit-count-policy.py` | temperature-controlled move policy |
| `alphago-l05-dihedral-symmetry.py` | 8-way training augmentation |
| `alphago-l05-neural-guided-mcts.py` | fixed-tree MCTS (self-test; Go needs an on-the-fly tree) |
| `alphago-l05-policy-value-training-step.py` | head-only SGD step (reference) |
| `alphago-l05-recent-replay-sampling.py` | replay-window sampling |

Added glue:

- `GoGame`: immutable position, history, seen positions, pass rule, encoding.
- `PVNetwork`: parameter initialization for the copied compact network,
  checkpointing, and single-position evaluation.
- On-the-fly MCTS: expands Go positions as needed and reuses the copied
  PUCT/edge/visit/Dirichlet math. The copied `neural_guided_mcts` requires a
  pre-built finite tree, so it is exercised by the self-test instead.
- Self-play, replay, training, evaluation, best-checkpoint selection, and the
  `selftest` / `train` / `eval` / `play` commands.

## How a game is played

At each move the agent runs MCTS from the current position. Each simulation
selects an edge by PUCT, expands an unexpanded leaf with the network, and
backups the leaf value with alternating perspective. The root's visit counts
become the move policy. Self-play adds Dirichlet noise to the root priors and
stores `(encoded state, visit policy, game outcome)` for training.

The network is trained on one position at a time, which matches the
batch-size-one normalization used during MCTS. Training uses the copied
policy-value loss plus Adam, gradient clipping, and random dihedral transforms.

## The small-board pass rule

On a 5×5 board, letting either side pass at any time made self-play degenerate:
the trained policy collapsed onto early passing and lost 0/20 games to the
untrained MCTS baseline. The system therefore keeps pass illegal until
`pass_min_moves` moves have been played. The default is half the board
(12 moves on 5×5) and is stored in `config.json`.

```bash
python alphago_system.py train --pass-min-moves 0   # normal Go rules
```

The restriction is a training stabilizer for the tiny board, not a real Go
rule. The shipped model was trained with it, so keep the config's value when
playing against that checkpoint.

## Results

Shipped 5×5 model (45,237 parameters):

```
training loss:                 4.25 -> ~3.8 over 20 iterations
vs random, 40 games:           35 wins, 5 losses, 0 draws  (87.5%)
vs the untrained network:      5 wins / 10 games (even)
training time:                 about 5 minutes on CPU
```

The 10-game evaluations printed during training are noisy (they ranged
0.70–1.00); the 40-game number above is the reliable one. Self-play games
against the untrained MCTS baseline are an even match, so the search itself
accounts for much of the strength on this small board.

`runs/alphago/` contains:

| File | Contents |
|---|---|
| `model.pt` | final iteration parameters |
| `best_model.pt` | best 10-game evaluation checkpoint (used by `play`) |
| `config.json` | board size, network shape, search and training settings |
| `metrics.json` | loss curve, evaluations, final 40-game evaluation |
| `training.png` | loss and win-rate plots |

## Commands

```bash
# Play as black; input "row col", "pass", or "quit"
python alphago_system.py play --simulations 64

# Let two copies of the model play a full game
python alphago_system.py play --ai-vs-ai --simulations 32

# Play as white
python alphago_system.py play --human-color white

# Evaluate a checkpoint
python alphago_system.py eval --checkpoint runs/alphago/best_model.pt \
    --games 40 --simulations 32

# Train a larger 7x7 model (slower; 9x9 also works)
python alphago_system.py train --board-size 7 --channels 48 --blocks 3 \
    --iterations 30 --games-per-iter 8 --out runs/alphago7
```

## Limitations

- This is a small-board (5×5 by default) educational agent, not a strong Go
  program. It has no tree reuse, batching, or parallel self-play.
- Python MCTS with a fresh network evaluation per simulation is the bottleneck;
  7×7 and 9×9 work but need much longer training.
- The copied network normalizes over the batch and board with no running
  statistics, so training and inference both run one board at a time.
- The active `pass_min_moves` rule is a toy stabilizer described above.
- The copied `alphago-l05-policy-value-training-step.py` only updates linear
  heads; the full network is trained through the copied
  `alpha_go_policy_value_loss`.
- Runs on CPU by design. The installed system PyTorch cannot target this
  machine's sm_61 P2000.

## Provenance

Copied unmodified on 2026-11-11 from:

```
/home/d/Desktop/TensorTonic-Solutions/study-plans/alphago-from-scratch/
```

Each source directory contains the original problem write-up (`*.md`) and its
solution (`*.py`); both are copied here.
