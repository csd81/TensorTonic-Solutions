#!/usr/bin/env python3
"""AlphaGo Zero-lite built from the copied alphago-from-scratch solutions.

The Go rules, board encoding, policy-value network, MCTS edge/PUCT/backup
math, loss, Dirichlet noise, visit-count policy, dihedral augmentation, and
replay sampling all come from the ``alphago-*.py`` files copied next to this
script.  The glue here adds a mutable game state, an on-the-fly MCTS tree
(the copied ``neural_guided_mcts`` works only on a pre-built finite tree),
self-play, training, evaluation, checkpointing, and a CLI.

Quick start (5x5 board, a few minutes on CPU):

    python alphago_system.py selftest
    python alphago_system.py train --out runs/alphago
    python alphago_system.py play --checkpoint runs/alphago/model.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

# The system PyTorch build cannot target this machine's sm_61 P2000.  This
# workload is CPU-sized anyway, and hiding the GPU also silences the warning.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Load every copied solution as a module (file names contain hyphens)
# ---------------------------------------------------------------------------
def load_part(name: str):
    path = HERE / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing copied part: {path}")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARTS = {
    "liberties": load_part("alphago-l01-group-liberties"),
    "legal": load_part("alphago-l01-legal-action-mask"),
    "resolve": load_part("alphago-l01-resolve-move"),
    "score": load_part("alphago-l01-tromp-taylor-result"),
    "net": load_part("alphago-l02-compact-policy-value-network"),
    "encode": load_part("alphago-l02-encode-board-history"),
    "heads": load_part("alphago-l02-policy-value-heads"),
    "resblock": load_part("alphago-l02-residual-block"),
    "backup": load_part("alphago-l03-backup-mcts-path"),
    "edges": load_part("alphago-l03-initialize-edge-statistics"),
    "puct": load_part("alphago-l03-puct-selection"),
    "loss": load_part("alphago-l04-policy-value-loss"),
    "dirichlet": load_part("alphago-l04-root-dirichlet-noise"),
    "visit": load_part("alphago-l04-visit-count-policy"),
    "symmetry": load_part("alphago-l05-dihedral-symmetry"),
    "tree_mcts": load_part("alphago-l05-neural-guided-mcts"),
    "train_step": load_part("alphago-l05-policy-value-training-step"),
    "replay": load_part("alphago-l05-recent-replay-sampling"),
}


# ---------------------------------------------------------------------------
# Go game state, built on the copied l01 rules
# ---------------------------------------------------------------------------
class GoGame:
    """Immutable position.  ``play`` returns a new position."""

    __slots__ = (
        "size",
        "board",
        "player",
        "passes",
        "moves",
        "history",
        "seen",
        "komi",
        "pass_min_moves",
    )

    def __init__(self, size: int = 5, komi: float = 0.5, pass_min_moves: int = -1):
        self.size = int(size)
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.player = 1
        self.passes = 0
        self.moves = 0
        # Newest board first, at most eight boards are kept.
        self.history: tuple[np.ndarray, ...] = (self.board.copy(),)
        # Every board that has occurred, for positional superko.
        self.seen: tuple[np.ndarray, ...] = (self.board.copy(),)
        self.komi = float(komi)
        # On a small board, early passing makes self-play degenerate.  Keep the
        # pass action closed until at least this many moves have been played.
        # pass_min_moves=0 restores normal Go rules.
        self.pass_min_moves = (
            int(pass_min_moves)
            if int(pass_min_moves) >= 0
            else self.size * self.size // 2
        )

    @property
    def num_actions(self) -> int:
        return self.size * self.size + 1

    def legal_mask(self) -> np.ndarray:
        mask = PARTS["legal"].legal_go_action_mask(
            self.board,
            self.player,
            list(self.seen),
            self.passes,
            self.moves,
        )
        mask = np.asarray(mask, dtype=bool)
        if self.moves < self.pass_min_moves:
            mask[-1] = False
        return mask

    def is_terminal(self) -> bool:
        return self.passes >= 2 or self.moves >= 2 * self.size * self.size

    def play(self, action: int) -> "GoGame":
        next_board, next_player, _captured, passes, moves, _terminal = (
            PARTS["resolve"].resolve_go_action(
                self.board,
                self.player,
                int(action),
                self.passes,
                self.moves,
            )
        )
        new = GoGame.__new__(GoGame)
        new.size = self.size
        new.board = np.asarray(next_board, dtype=np.int8)
        new.player = int(next_player)
        new.passes = int(passes)
        new.moves = int(moves)
        new.komi = self.komi
        new.pass_min_moves = self.pass_min_moves
        new.history = (new.board.copy(),) + self.history[:7]
        new.seen = self.seen + (new.board.copy(),)
        return new

    def encode(self) -> torch.Tensor:
        boards = [torch.from_numpy(b.copy()) for b in self.history]
        return PARTS["encode"].encode_go_history(boards, self.player)

    def result(self) -> tuple[float, float, int]:
        terminal, black, white, winner = PARTS["score"].tromp_taylor_result(
            self.board, self.komi, self.passes, self.moves
        )
        if not terminal:
            raise ValueError("result() called on a non-terminal game")
        return float(black), float(white), int(winner)

    def value_for_player(self) -> float:
        _black, _white, winner = self.result()
        if winner == 0:
            return 0.0
        return 1.0 if winner == self.player else -1.0

    def render(self) -> str:
        chars = {0: ".", 1: "X", -1: "O"}
        header = "    " + " ".join(f"{c:2d}" for c in range(self.size))
        rows = [header]
        for r in range(self.size):
            rows.append(f"{r:2d}  " + "  ".join(chars[int(v)] for v in self.board[r]))
        return "\n".join(rows)

    def action_name(self, action: int) -> str:
        if action == self.size * self.size:
            return "pass"
        return f"{action // self.size} {action % self.size}"


# ---------------------------------------------------------------------------
# Policy-value network assembled by the copied l02 function
# ---------------------------------------------------------------------------
class PVNetwork:
    def __init__(
        self,
        board_size: int,
        channels: int = 32,
        blocks: int = 2,
        policy_channels: int = 2,
        value_hidden: int = 64,
        seed: int = 0,
    ):
        self.size = int(board_size)
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.policy_channels = int(policy_channels)
        self.value_hidden = int(value_hidden)
        self.device = torch.device("cpu")
        self._g = torch.Generator(device="cpu").manual_seed(seed)

        def conv(out_c: int, in_c: int, k: int) -> torch.Tensor:
            w = torch.randn(
                out_c, in_c, k, k, generator=self._g
            ) * math.sqrt(2.0 / (in_c * k * k))
            return w.to(self.device).requires_grad_(True)

        def linear(in_f: int, out_f: int):
            bound = 1.0 / math.sqrt(in_f)
            w = torch.empty(out_f, in_f).uniform_(
                -bound, bound, generator=self._g
            )
            b = torch.empty(out_f).uniform_(-bound, bound, generator=self._g)
            return (
                w.to(self.device).requires_grad_(True),
                b.to(self.device).requires_grad_(True),
            )

        self.trunk_weight = conv(channels, 17, 3)
        self.trunk_scale = torch.ones(channels, device=self.device).requires_grad_(True)
        self.trunk_shift = torch.zeros(channels, device=self.device).requires_grad_(True)

        self.res_blocks = []
        for _ in range(blocks):
            self.res_blocks.append(
                {
                    "conv1_weight": conv(channels, channels, 3),
                    "scale1": torch.ones(
                        channels, device=self.device
                    ).requires_grad_(True),
                    "shift1": torch.zeros(
                        channels, device=self.device
                    ).requires_grad_(True),
                    "conv2_weight": conv(channels, channels, 3),
                    "scale2": torch.ones(
                        channels, device=self.device
                    ).requires_grad_(True),
                    "shift2": torch.zeros(
                        channels, device=self.device
                    ).requires_grad_(True),
                }
            )

        n_actions = self.size * self.size + 1
        policy_flat = self.policy_channels * self.size * self.size
        value_flat = self.size * self.size
        p_lin_w, p_lin_b = linear(policy_flat, n_actions)
        v_hid_w, v_hid_b = linear(value_flat, value_hidden)
        v_out_w, v_out_b = linear(value_hidden, 1)

        self.head = {
            "policy_conv_weight": conv(self.policy_channels, channels, 1),
            "policy_scale": torch.ones(
                self.policy_channels, device=self.device
            ).requires_grad_(True),
            "policy_shift": torch.zeros(
                self.policy_channels, device=self.device
            ).requires_grad_(True),
            "policy_linear_weight": p_lin_w,
            "policy_linear_bias": p_lin_b,
            "value_conv_weight": conv(1, channels, 1),
            "value_scale": torch.ones(1, device=self.device).requires_grad_(True),
            "value_shift": torch.zeros(1, device=self.device).requires_grad_(True),
            "value_hidden_weight": v_hid_w,
            "value_hidden_bias": v_hid_b,
            "value_output_weight": v_out_w,
            "value_output_bias": v_out_b,
        }

    def forward(self, x: torch.Tensor):
        return PARTS["net"].compact_policy_value_network(
            x,
            self.trunk_weight,
            self.trunk_scale,
            self.trunk_shift,
            self.res_blocks,
            self.head,
            1e-5,
        )

    def flat_params(self) -> list[torch.Tensor]:
        params = [self.trunk_weight, self.trunk_scale, self.trunk_shift]
        for block in self.res_blocks:
            params.extend(block.values())
        params.extend(self.head.values())
        return params

    def reg_params(self) -> list[torch.Tensor]:
        """Only convolution/linear weights are L2-regularized."""
        params = [self.trunk_weight]
        for block in self.res_blocks:
            params.extend([block["conv1_weight"], block["conv2_weight"]])
        params.extend(
            [
                self.head["policy_conv_weight"],
                self.head["policy_linear_weight"],
                self.head["value_conv_weight"],
                self.head["value_hidden_weight"],
                self.head["value_output_weight"],
            ]
        )
        return params

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd = {
            "trunk_weight": self.trunk_weight.detach().clone(),
            "trunk_scale": self.trunk_scale.detach().clone(),
            "trunk_shift": self.trunk_shift.detach().clone(),
        }
        for i, block in enumerate(self.res_blocks):
            for key, value in block.items():
                sd[f"res{i}_{key}"] = value.detach().clone()
        for key, value in self.head.items():
            sd[f"head_{key}"] = value.detach().clone()
        return sd

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        def put(param: torch.Tensor, key: str):
            if key not in sd:
                raise KeyError(f"checkpoint is missing {key}")
            param.data.copy_(sd[key].to(self.device, dtype=torch.float32))

        put(self.trunk_weight, "trunk_weight")
        put(self.trunk_scale, "trunk_scale")
        put(self.trunk_shift, "trunk_shift")
        for i, block in enumerate(self.res_blocks):
            for key, value in block.items():
                put(value, f"res{i}_{key}")
        for key, value in self.head.items():
            put(value, f"head_{key}")

    @classmethod
    def from_checkpoint(cls, path: Path, config: dict) -> "PVNetwork":
        net = cls(
            board_size=config["board_size"],
            channels=config["channels"],
            blocks=config["blocks"],
            policy_channels=config.get("policy_channels", 2),
            value_hidden=config.get("value_hidden", 64),
            seed=config.get("seed", 0),
        )
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        net.load_state_dict(sd)
        return net

    @torch.no_grad()
    def evaluate(self, game: GoGame) -> tuple[np.ndarray, float]:
        state = game.encode().unsqueeze(0)
        logits, value = self.forward(state)
        return (
            logits[0].detach().cpu().numpy().astype(np.float64),
            float(value[0].detach().cpu()),
        )


# ---------------------------------------------------------------------------
# On-the-fly MCTS using the copied PUCT / edge / visit math
# ---------------------------------------------------------------------------
class MCTSNode:
    __slots__ = (
        "game",
        "legal_mask",
        "priors",
        "visits",
        "values",
        "children",
        "expanded",
        "value",
        "terminal",
    )

    def __init__(self, game: GoGame, legal_mask: np.ndarray | None = None):
        self.game = game
        self.legal_mask = game.legal_mask() if legal_mask is None else legal_mask
        self.priors = np.zeros(len(self.legal_mask), dtype=np.float64)
        self.visits = np.zeros(len(self.legal_mask), dtype=np.int64)
        self.values = np.zeros(len(self.legal_mask), dtype=np.float64)
        self.children: dict[int, MCTSNode] = {}
        self.expanded = False
        self.value = 0.0
        self.terminal = not bool(self.legal_mask.any())


def expand_node(node: MCTSNode, net: PVNetwork) -> float:
    if node.terminal:
        node.value = node.game.value_for_player()
        return node.value
    logits, value = net.evaluate(node.game)
    edges = PARTS["edges"].initialize_mcts_edges(logits, node.legal_mask)
    node.priors = np.asarray(edges["P"], dtype=np.float64)
    node.value = float(value)
    node.expanded = True
    return node.value


def simulate(root: MCTSNode, net: PVNetwork, cpuct: float) -> None:
    node = root
    path: list[tuple[MCTSNode, int]] = []
    while True:
        if node.terminal:
            leaf_value = node.game.value_for_player()
            leaf_player = node.game.player
            break
        if not node.expanded:
            leaf_value = expand_node(node, net)
            leaf_player = node.game.player
            break

        with np.errstate(divide="ignore", invalid="ignore"):
            action, _scores = PARTS["puct"].select_puct_action(
                node.visits,
                node.values,
                node.priors,
                node.legal_mask,
                cpuct,
            )
        path.append((node, int(action)))
        child = node.children.get(action)
        if child is None:
            child_game = node.game.play(action)
            child = MCTSNode(child_game, child_game.legal_mask())
            node.children[action] = child
        node = child

    # Mirrors alphago-l03-backup-mcts-path: alternate perspective per level.
    for parent, action in reversed(path):
        signed = leaf_value if parent.game.player == leaf_player else -leaf_value
        parent.visits[action] += 1
        parent.values[action] += signed


def run_mcts(
    game: GoGame,
    net: PVNetwork,
    simulations: int,
    cpuct: float,
    rng: np.random.Generator,
    dirichlet_alpha: float | None = None,
    dirichlet_epsilon: float = 0.0,
) -> MCTSNode:
    if game.is_terminal():
        raise ValueError("cannot search a terminal game")
    root = MCTSNode(game)
    expand_node(root, net)

    if dirichlet_alpha is not None:
        legal = root.legal_mask
        n_legal = int(legal.sum())
        if n_legal > 0:
            noise = np.zeros_like(root.priors)
            noise[legal] = rng.dirichlet([dirichlet_alpha] * n_legal)
            root.priors = PARTS["dirichlet"].mix_root_dirichlet_noise(
                root.priors, noise, legal, dirichlet_epsilon
            )

    for _ in range(max(0, int(simulations))):
        simulate(root, net, cpuct)
    return root


def root_action(
    root: MCTSNode, temperature: float, rng: np.random.Generator
) -> tuple[int, np.ndarray]:
    policy, action = PARTS["visit"].visit_count_policy(
        root.visits,
        root.legal_mask,
        float(temperature),
        float(rng.random()),
    )
    return int(action), np.asarray(policy, dtype=np.float64)


# ---------------------------------------------------------------------------
# Self-play, training, evaluation
# ---------------------------------------------------------------------------
def self_play_game(
    net: PVNetwork, args, rng: np.random.Generator, add_noise: bool = True
) -> tuple[list[dict], tuple[float, float, int], int]:
    game = GoGame(args.board_size, args.komi, args.pass_min_moves)
    records: list[dict] = []
    move_number = 0
    while not game.is_terminal():
        root = run_mcts(
            game,
            net,
            args.simulations,
            args.cpuct,
            rng,
            dirichlet_alpha=args.dirichlet_alpha if add_noise else None,
            dirichlet_epsilon=args.dirichlet_epsilon,
        )
        _, target_policy = root_action(root, 1.0, rng)
        legal = game.legal_mask()
        if args.policy_smoothing > 0.0 and legal.any():
            # With few simulations the visit policy is nearly one-hot.  A
            # little uniform smoothing keeps the trained policy explorable.
            target_policy = (1.0 - args.policy_smoothing) * target_policy
            target_policy[legal] += args.policy_smoothing / legal.sum()
        temperature = 1.0 if move_number < args.temperature_moves else args.temperature_late
        action, _ = root_action(root, temperature, rng)

        records.append(
            {
                "state": game.encode().numpy().astype(np.float32),
                "policy": target_policy.astype(np.float32),
                "player": int(game.player),
            }
        )
        game = game.play(action)
        move_number += 1

    black, white, winner = game.result()
    for record in records:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == record["player"] else -1.0
        record["value"] = float(value)
        record.pop("player", None)
    return records, (black, white, winner), move_number


def train_network(
    net: PVNetwork, optimizer: torch.optim.Optimizer, replay: list[dict], args, rng
) -> dict | None:
    if len(replay) < args.batch_size:
        return None
    indices = rng.integers(0, len(replay), size=args.batch_size).tolist()
    sampled, _eligible = PARTS["replay"].sample_recent_replay_batch(
        [replay], 1, indices
    )

    optimizer.zero_grad()
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "reg": 0.0}
    for record in sampled:
        transform = int(rng.integers(0, 8))
        state, policy, value = PARTS["symmetry"].apply_go_symmetry(
            record["state"], record["policy"], record["value"], transform
        )
        x = torch.from_numpy(state).unsqueeze(0)
        target_policy = torch.from_numpy(policy).unsqueeze(0)
        target_value = torch.tensor([value], dtype=torch.float32)

        # One board at a time: the copied network normalizes over the batch
        # and board, so this matches the batch-size-one MCTS inference path.
        logits, predicted_value = net.forward(x)
        loss, policy_loss, value_loss, reg_loss = PARTS["loss"].alpha_go_policy_value_loss(
            logits,
            predicted_value,
            target_policy,
            target_value,
            net.reg_params(),
            args.l2,
        )
        (loss / args.batch_size).backward()
        totals["loss"] += loss.detach().item() / args.batch_size
        totals["policy"] += policy_loss.detach().item() / args.batch_size
        totals["value"] += value_loss.detach().item() / args.batch_size
        totals["reg"] += reg_loss.detach().item() / args.batch_size

    torch.nn.utils.clip_grad_norm_(net.flat_params(), args.grad_clip)
    optimizer.step()
    return totals


def evaluate_vs_random(
    net: PVNetwork,
    board_size: int,
    komi: float,
    games: int,
    simulations: int,
    cpuct: float,
    seed: int,
    pass_min_moves: int = -1,
) -> dict:
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    for game_index in range(games):
        ai_color = 1 if game_index % 2 == 0 else -1
        game = GoGame(board_size, komi, pass_min_moves)
        while not game.is_terminal():
            if game.player == ai_color:
                root = run_mcts(game, net, simulations, cpuct, rng)
                action, _ = root_action(root, 0.0, rng)
            else:
                legal = np.flatnonzero(game.legal_mask())
                action = int(rng.choice(legal))
            game = game.play(action)
        _black, _white, winner = game.result()
        if winner == 0:
            draws += 1
        elif winner == ai_color:
            wins += 1
        else:
            losses += 1
    total = wins + losses + draws
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    out_dir: Path, net: PVNetwork, config: dict, filename: str = "model.pt"
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir / filename)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def load_checkpoint(path: Path) -> tuple[PVNetwork, dict]:
    config_path = path.parent / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Missing config next to checkpoint: {config_path}")
    net = PVNetwork.from_checkpoint(path, config)
    return net, config


def config_from_args(args) -> dict:
    keys = [
        "board_size",
        "channels",
        "blocks",
        "policy_channels",
        "value_hidden",
        "komi",
        "simulations",
        "eval_simulations",
        "cpuct",
        "dirichlet_alpha",
        "dirichlet_epsilon",
        "temperature_moves",
        "temperature_late",
        "iterations",
        "games_per_iter",
        "train_steps",
        "batch_size",
        "lr",
        "l2",
        "grad_clip",
        "replay_capacity",
        "seed",
        "pass_min_moves",
        "policy_smoothing",
    ]
    return {key: getattr(args, key) for key in keys}


def plot_metrics(metrics: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    losses = metrics.get("losses", [])
    if losses:
        axes[0].plot(losses, linewidth=0.8)
        axes[0].set_title("training loss")
        axes[0].set_xlabel("training step")
    else:
        axes[0].axis("off")

    history = metrics.get("evaluations", [])
    if history:
        xs = [item["iteration"] for item in history]
        ys = [item["win_rate"] for item in history]
        axes[1].plot(xs, ys, marker="o")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_title("win rate vs random")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("win rate")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    net = PVNetwork(
        args.board_size,
        args.channels,
        args.blocks,
        args.policy_channels,
        args.value_hidden,
        args.seed,
    )
    optimizer = torch.optim.Adam(net.flat_params(), lr=args.lr)
    replay: list[dict] = []
    out_dir = Path(args.out)
    if args.pass_min_moves < 0:
        args.pass_min_moves = args.board_size * args.board_size // 2
    config = config_from_args(args)
    config["out"] = str(out_dir)
    save_checkpoint(out_dir, net, config)

    print(
        f"AlphaGo-lite: {args.board_size}x{args.board_size}, "
        f"{args.channels} channels, {args.blocks} residual blocks, "
        f"{sum(p.numel() for p in net.flat_params())} parameters"
    )
    print(
        f"self-play sims={args.simulations}, eval sims={args.eval_simulations}, "
        f"{args.iterations} iterations x {args.games_per_iter} games"
    )

    metrics = {"config": config, "losses": [], "evaluations": []}
    best_win_rate = -1.0
    if args.eval_games > 0:
        print("evaluating the untrained network vs random...", flush=True)
        initial = evaluate_vs_random(
            net,
            args.board_size,
            args.komi,
            args.eval_games,
            args.eval_simulations,
            args.cpuct,
            args.seed + 1000,
            args.pass_min_moves,
        )
        metrics["evaluations"].append({"iteration": 0, **initial})
        print(f"  untrained win rate: {initial['win_rate']:.2f} ({initial})")

    start = time.time()
    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        self_play_wins = {"black": 0, "white": 0, "draw": 0}
        for _ in range(args.games_per_iter):
            records, (black, white, winner), moves = self_play_game(net, args, rng)
            replay.extend(records)
            if len(replay) > args.replay_capacity:
                replay = replay[-args.replay_capacity :]
            if winner == 0:
                self_play_wins["draw"] += 1
            elif winner == 1:
                self_play_wins["black"] += 1
            else:
                self_play_wins["white"] += 1

        step_losses = []
        for _ in range(args.train_steps):
            result = train_network(net, optimizer, replay, args, rng)
            if result is not None:
                step_losses.append(result)
                metrics["losses"].append(result["loss"])
        avg_loss = (
            sum(item["loss"] for item in step_losses) / len(step_losses)
            if step_losses
            else float("nan")
        )

        evaluation = None
        if args.eval_games > 0 and (
            iteration % args.eval_every == 0 or iteration == args.iterations
        ):
            evaluation = evaluate_vs_random(
                net,
                args.board_size,
                args.komi,
                args.eval_games,
                args.eval_simulations,
                args.cpuct,
                args.seed + 2000 + iteration,
                args.pass_min_moves,
            )
            metrics["evaluations"].append(
                {"iteration": iteration, **evaluation}
            )
            if evaluation["win_rate"] >= best_win_rate:
                best_win_rate = evaluation["win_rate"]
                save_checkpoint(out_dir, net, config, "best_model.pt")
                print("  saved new best checkpoint", flush=True)

        elapsed = time.time() - iter_start
        message = (
            f"iter {iteration:3d}/{args.iterations}  "
            f"loss {avg_loss:.4f}  replay {len(replay):5d}  "
            f"self-play B/W/D {self_play_wins['black']}/"
            f"{self_play_wins['white']}/{self_play_wins['draw']}  "
            f"{elapsed:.1f}s"
        )
        if evaluation is not None:
            message += (
                f"  vs-random {evaluation['win_rate']:.2f} "
                f"({evaluation['wins']}/{evaluation['losses']}/{evaluation['draws']})"
            )
        print(message, flush=True)

        save_checkpoint(out_dir, net, config)
        metrics["elapsed_seconds"] = time.time() - start
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    plot_metrics(metrics, out_dir / "training.png")
    metrics["best_win_rate"] = best_win_rate
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if best_win_rate < 0 and (out_dir / "model.pt").exists():
        save_checkpoint(out_dir, net, config, "best_model.pt")
    print(f"saved model to {out_dir / 'model.pt'}")
    print(f"saved metrics to {out_dir / 'metrics.json'}")


def cmd_eval(args) -> None:
    path = Path(args.checkpoint)
    net, config = load_checkpoint(path)
    board_size = config.get("board_size", args.board_size)
    komi = config.get("komi", args.komi)
    cpuct = config.get("cpuct", args.cpuct)
    result = evaluate_vs_random(
        net,
        board_size,
        komi,
        args.games,
        args.simulations,
        cpuct,
        args.seed,
        config.get("pass_min_moves", -1),
    )
    print(
        f"{board_size}x{board_size} vs random: "
        f"{result['wins']} wins, {result['losses']} losses, "
        f"{result['draws']} draws  ({result['win_rate']:.2f})"
    )


def cmd_play(args) -> None:
    checkpoint = args.checkpoint
    if checkpoint is None:
        for candidate in (
            HERE / "runs" / "alphago" / "best_model.pt",
            HERE / "runs" / "alphago" / "model.pt",
        ):
            if candidate.exists():
                checkpoint = str(candidate)
                break
    if checkpoint:
        net, config = load_checkpoint(Path(checkpoint))
        board_size = config.get("board_size", args.board_size)
        komi = config.get("komi", args.komi)
        cpuct = config.get("cpuct", args.cpuct)
    else:
        net = PVNetwork(
            args.board_size,
            args.channels,
            args.blocks,
            args.policy_channels,
            args.value_hidden,
            args.seed,
        )
        config = vars(args)
        board_size = args.board_size
        komi = args.komi
        cpuct = args.cpuct

    rng = np.random.default_rng(args.seed)
    game = GoGame(board_size, komi, config.get("pass_min_moves", args.pass_min_moves))
    human_color = 1 if args.human_color == "black" else -1

    print(f"AlphaGo-lite on {board_size}x{board_size}, komi {komi}. "
          f"X = black, O = white.")
    if not args.ai_vs_ai:
        print(f"you are {'black' if human_color == 1 else 'white'}.")

    while not game.is_terminal():
        print()
        print(game.render())
        if game.player == human_color and not args.ai_vs_ai:
            while True:
                raw = input("your move (row col), 'pass', or 'quit': ").strip().lower()
                if raw in {"quit", "q"}:
                    return
                if raw == "pass":
                    action = game.size * game.size
                else:
                    try:
                        row, col = [int(part) for part in raw.split()]
                    except Exception:
                        print("enter two integers, e.g. '2 3'")
                        continue
                    if not (0 <= row < game.size and 0 <= col < game.size):
                        print(f"row and col must be between 0 and {game.size - 1}")
                        continue
                    action = row * game.size + col
                if not game.legal_mask()[action]:
                    print("illegal move")
                    continue
                break
        else:
            root = run_mcts(
                game,
                net,
                args.simulations,
                cpuct,
                rng,
                dirichlet_alpha=None,
            )
            action, policy = root_action(root, 0.0, rng)
            legal = np.flatnonzero(game.legal_mask())
            order = sorted(legal, key=lambda a: -root.visits[a])[:3]
            top = ", ".join(
                f"{game.action_name(int(a))} ({int(root.visits[a])})" for a in order
            )
            print(f"AI plays {game.action_name(action)}  [top: {top}]")

        game = game.play(action)

    print()
    print(game.render())
    black, white, winner = game.result()
    who = "black" if winner == 1 else "white" if winner == -1 else "draw"
    print(f"final score: black {black}  white {white}  -> {who}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def cmd_selftest(args) -> None:
    for name, module in PARTS.items():
        if module is None:
            raise AssertionError(f"part {name} failed to load")

    # Rules: legal mask, pass, terminal scoring.
    game = GoGame(5, 0.5, 0)
    mask = game.legal_mask()
    assert mask.shape == (26,) and mask.sum() == 26
    center = 2 * 5 + 2
    game = game.play(center)
    assert game.board[2, 2] == 1 and game.player == -1 and game.moves == 1
    assert not bool(game.legal_mask()[center])
    pass_action = 25
    game = game.play(pass_action).play(pass_action)
    assert game.is_terminal()
    black, white, winner = game.result()
    assert black + white > 0 and winner in (-1, 0, 1)
    board, liberties = PARTS["liberties"].go_group_liberties(
        game.board, 2, 2
    )
    assert (2, 2) in {tuple(x) for x in board}

    # Encoding and network shapes.
    state = GoGame(5, 0.5, 0).encode()
    assert state.shape == (17, 5, 5) and state.dtype == torch.float32
    net = PVNetwork(5, channels=8, blocks=1, value_hidden=16, seed=0)
    logits, value = net.forward(state.unsqueeze(0))
    assert logits.shape == (1, 26) and value.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()

    # Full-network backward through the copied loss.
    target = torch.zeros(1, 26)
    target[0, center] = 1.0
    loss, policy_loss, value_loss, reg_loss = PARTS["loss"].alpha_go_policy_value_loss(
        logits, value, target, torch.zeros(1), net.reg_params(), 1e-4
    )
    loss.backward()
    assert all(p.grad is not None for p in net.flat_params())
    assert policy_loss.detach().item() >= 0.0 and value_loss.detach().item() >= 0.0

    # Residual block and heads run standalone too.
    x = torch.randn(2, 8, 5, 5)
    block = net.res_blocks[0]
    y = PARTS["resblock"].alpha_go_residual_block(
        x,
        block["conv1_weight"],
        block["scale1"],
        block["shift1"],
        block["conv2_weight"],
        block["scale2"],
        block["shift2"],
        1e-5,
    )
    assert y.shape == x.shape
    h = net.head
    p_logits, p_values = PARTS["heads"].policy_value_heads(
        x,
        h["policy_conv_weight"],
        h["policy_scale"],
        h["policy_shift"],
        h["policy_linear_weight"],
        h["policy_linear_bias"],
        h["value_conv_weight"],
        h["value_scale"],
        h["value_shift"],
        h["value_hidden_weight"],
        h["value_hidden_bias"],
        h["value_output_weight"],
        h["value_output_bias"],
        1e-5,
    )
    assert p_logits.shape == (2, 26) and p_values.shape == (2,)

    # MCTS on a real position.
    rng = np.random.default_rng(0)
    root = run_mcts(GoGame(5, 0.5, 0), net, 4, 1.5, rng, 0.3, 0.25)
    assert root.visits.sum() == 4
    assert root.priors[root.legal_mask].sum() > 0.99
    action, policy = root_action(root, 0.0, rng)
    assert 0 <= action < 26 and bool(root.legal_mask[action])
    assert abs(policy.sum() - 1.0) < 1e-9

    # Symmetry and replay sampling.
    state = GoGame(5, 0.5, 0).encode().numpy()
    policy = np.zeros(26)
    policy[:25] = 1 / 25
    s2, p2, v2 = PARTS["symmetry"].apply_go_symmetry(state, policy, 0.5, 3)
    assert s2.shape == state.shape and p2.shape == policy.shape and v2 == 0.5
    records = [{"state": state, "policy": policy, "value": 0.0}] * 4
    sampled, count = PARTS["replay"].sample_recent_replay_batch(
        [records], 1, [0, 2, 3]
    )
    assert count == 4 and len(sampled) == 3

    # The copied fixed-tree MCTS also runs.
    n, q, pol = PARTS["tree_mcts"].neural_guided_mcts(
        "A",
        {"A": {"0": "B", "1": "C"}},
        {"A": [0, 1], "B": [], "C": []},
        {"A": 1, "B": -1, "C": -1},
        {"A": [0.0, 0.0], "B": [0.0], "C": [0.0]},
        {"B": 1.0, "C": -1.0},
        {"B": 1.0, "C": -1.0},
        4,
        1.5,
        1.0,
    )
    assert n.shape == (2,) and q.shape == (2,) and abs(pol.sum() - 1.0) < 1e-9

    print("PASS: 18 parts loaded; rules, encoding, network, loss, MCTS, "
          "symmetry, replay, and fixed-tree MCTS all OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_network_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--policy-channels", type=int, default=2)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--komi", type=float, default=0.5)
    parser.add_argument(
        "--pass-min-moves",
        type=int,
        default=-1,
        help="pass is illegal until this many moves; -1 means board_size^2/2",
    )
    parser.add_argument("--seed", type=int, default=0)


def add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--simulations", type=int, default=24)
    parser.add_argument("--cpuct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=6)
    parser.add_argument("--temperature-late", type=float, default=0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AlphaGo Zero-lite assembled from the copied AlphaGo parts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    test = sub.add_parser("selftest", help="check every copied part and the glue")
    test.set_defaults(func=cmd_selftest)

    train = sub.add_parser("train", help="self-play and train a model")
    add_network_args(train)
    add_search_args(train)
    train.add_argument("--iterations", type=int, default=20)
    train.add_argument("--games-per-iter", type=int, default=6)
    train.add_argument("--train-steps", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--l2", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--policy-smoothing", type=float, default=0.1)
    train.add_argument("--replay-capacity", type=int, default=20000)
    train.add_argument("--eval-every", type=int, default=5)
    train.add_argument("--eval-games", type=int, default=10)
    train.add_argument("--eval-simulations", type=int, default=24)
    train.add_argument("--out", default=str(HERE / "runs" / "alphago"))
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="evaluate a checkpoint against random play")
    ev.add_argument(
        "--checkpoint",
        default=str(HERE / "runs" / "alphago" / "best_model.pt"),
    )
    ev.add_argument("--board-size", type=int, default=5)
    ev.add_argument("--komi", type=float, default=0.5)
    ev.add_argument("--cpuct", type=float, default=1.5)
    ev.add_argument("--games", type=int, default=20)
    ev.add_argument("--simulations", type=int, default=48)
    ev.add_argument("--seed", type=int, default=0)
    ev.set_defaults(func=cmd_eval)

    play = sub.add_parser("play", help="play against the model, or AI vs AI")
    add_network_args(play)
    play.add_argument("--checkpoint", default=None)
    play.add_argument("--simulations", type=int, default=64)
    play.add_argument("--cpuct", type=float, default=1.5)
    play.add_argument("--human-color", choices=["black", "white"], default="black")
    play.add_argument("--ai-vs-ai", action="store_true")
    play.set_defaults(func=cmd_play)

    args = parser.parse_args(argv)
    if getattr(args, "board_size", 5) < 3:
        parser.error("--board-size must be at least 3")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
