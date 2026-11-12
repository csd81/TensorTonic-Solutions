#!/usr/bin/env python3
"""A runnable tiny LLM built from the copied CS336 solutions.

Core pieces copied from ``study-plans/language-modeling-from-scratch``:

    BPE training/encoding      cs336-l01-*
    attention/board math       cs336-l02-*
    RMSNorm, SwiGLU, RoPE, GQA cs336-l03-*
    KV-cache decode            cs336-l10-*

Useful pieces copied from other tracks:

    warmup + cosine LR         pytorch-basics/pytorch-lr-warmup-scheduler
    temperature/top-k sampling cracking-nlp/nlp-text-generation-sampling
    perplexity                 cracking-nlp/nlp-perplexity-computation
    masked causal attention    pytorch-basics/pytorch-masked-causal-attention

The glue adds a Tokenizer wrapper, a decoder-only Transformer, data
preparation, training, generation, checkpointing, plots, and a CLI.

Quick start:

    python llm_system.py selftest
    python llm_system.py prepare
    python llm_system.py train
    python llm_system.py generate --prompt "The "
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

# The system PyTorch build cannot target this machine's sm_61 P2000.  Keep the
# GPU hidden unless the caller explicitly passes --device cuda.
_requested_device = "cpu"
for _index, _argument in enumerate(sys.argv):
    if _argument == "--device" and _index + 1 < len(sys.argv):
        _requested_device = sys.argv[_index + 1]
if _requested_device != "cuda":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CS336_DIR = HERE / "parts" / "cs336"
EXTRA_DIR = HERE / "parts" / "extra"


# ---------------------------------------------------------------------------
# Load the copied solutions
# ---------------------------------------------------------------------------
def load_part(directory: Path, name: str):
    path = directory / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing copied part: {path}")
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARTS = {
    "bpe_train": load_part(CS336_DIR, "cs336-l01-train-byte-pair-encoding"),
    "bpe_encode": load_part(CS336_DIR, "cs336-l01-apply-bpe-merge-ranks"),
    "attn_scores": load_part(CS336_DIR, "cs336-l02-einsum-attention-scores"),
    "rmsnorm": load_part(CS336_DIR, "cs336-l03-rmsnorm-forward"),
    "swiglu": load_part(CS336_DIR, "cs336-l03-parameter-matched-swiglu"),
    "rope": load_part(CS336_DIR, "cs336-l03-rotary-query-key-embeddings"),
    "gqa": load_part(CS336_DIR, "cs336-l03-causal-grouped-query-attention"),
    "kv": load_part(CS336_DIR, "cs336-l10-kv-cache-decode-step"),
}

EXTRA_PARTS = {
    "warmup": load_part(EXTRA_DIR, "pytorch-lr-warmup-scheduler"),
    "sampling": load_part(EXTRA_DIR, "nlp-text-generation-sampling"),
    "perplexity": load_part(EXTRA_DIR, "nlp-perplexity-computation"),
    "causal_attn": load_part(EXTRA_DIR, "pytorch-masked-causal-attention"),
}


# ---------------------------------------------------------------------------
# Tokenizer: character-level or BPE from the copied CS336 parts
# ---------------------------------------------------------------------------
class Tokenizer:
    def __init__(
        self,
        kind: str,
        stoi: dict[str, int] | None = None,
        itos: dict[int, str] | None = None,
        merges: list[list[int]] | None = None,
        bpe_vocab: dict[int, list[int]] | None = None,
        backend=None,
    ):
        self.kind = kind
        self.stoi = stoi or {}
        self.itos = itos or {}
        self.merges = merges or []
        self.bpe_vocab = bpe_vocab or {}
        self.backend = backend

    @classmethod
    def train(cls, text: str, kind: str, vocab_size: int) -> "Tokenizer":
        if kind == "char":
            chars = sorted(set(text))
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}
            return cls("char", stoi=stoi, itos=itos)
        if kind == "hf-bpe":
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers import decoders, models, pre_tokenizers, trainers

            backend = HFTokenizer(models.BPE(unk_token="<unk>"))
            backend.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False
            )
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["<unk>", "<pad>"],
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )
            backend.train_from_iterator([text], trainer=trainer)
            backend.decoder = decoders.ByteLevel()
            return cls("hf-bpe", backend=backend)
        if kind != "bpe":
            raise ValueError(f"unknown tokenizer {kind!r}")
        if vocab_size <= 256:
            raise ValueError("BPE vocab_size must be greater than 256")
        result = PARTS["bpe_train"].train_bpe([text], vocab_size)
        merges = [
            [int(left), int(right), int(new_id)]
            for left, right, new_id in result["merges"]
        ]
        vocab = {i: list(bytes([i])) for i in range(256)}
        for token_id, token_bytes in result["vocab"]:
            vocab[int(token_id)] = [int(b) for b in token_bytes]
        return cls("bpe", merges=merges, bpe_vocab=vocab)

    @property
    def vocab_size(self) -> int:
        if self.kind == "char":
            return len(self.stoi)
        if self.kind == "hf-bpe":
            return int(self.backend.get_vocab_size())
        return len(self.bpe_vocab)

    @property
    def pad_id(self) -> int:
        if self.kind == "hf-bpe":
            return int(self.backend.token_to_id("<pad>"))
        return 0

    def encode(self, text: str) -> list[int]:
        if self.kind == "char":
            missing = sorted({ch for ch in text if ch not in self.stoi})
            if missing:
                raise ValueError(
                    f"characters not in tokenizer vocabulary: {missing!r}"
                )
            return [self.stoi[ch] for ch in text]
        if self.kind == "hf-bpe":
            return [int(i) for i in self.backend.encode(text).ids]
        return [int(x) for x in PARTS["bpe_encode"].encode(text, self.merges)]

    def decode(self, ids: list[int]) -> str:
        if self.kind == "char":
            return "".join(self.itos[int(i)] for i in ids)
        if self.kind == "hf-bpe":
            return self.backend.decode([int(i) for i in ids])
        return PARTS["bpe_encode"].decode([int(i) for i in ids], self.bpe_vocab)

    def save(self, path: Path) -> None:
        if self.kind == "hf-bpe":
            self.backend.save(str(path))
            return
        payload = {"kind": self.kind}
        if self.kind == "char":
            payload["chars"] = [self.itos[i] for i in range(len(self.itos))]
        else:
            payload["merges"] = self.merges
            payload["vocab"] = {
                str(token_id): token_bytes
                for token_id, token_bytes in self.bpe_vocab.items()
            }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "kind" not in payload:
            from tokenizers import Tokenizer as HFTokenizer

            return cls("hf-bpe", backend=HFTokenizer.from_file(str(path)))
        if payload["kind"] == "char":
            chars = payload["chars"]
            return cls(
                "char",
                stoi={ch: i for i, ch in enumerate(chars)},
                itos={i: ch for i, ch in enumerate(chars)},
            )
        vocab = {
            int(token_id): [int(b) for b in token_bytes]
            for token_id, token_bytes in payload["vocab"].items()
        }
        return cls("bpe", merges=payload["merges"], bpe_vocab=vocab)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(
    corpus_path: Path,
    out_dir: Path,
    kind: str,
    vocab_size: int,
) -> tuple[Tokenizer, np.ndarray, dict]:
    text = corpus_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"empty corpus: {corpus_path}")
    print(
        f"training {kind} tokenizer on {len(text):,} characters "
        f"(target vocab {vocab_size})..."
    )
    start = time.time()
    tokenizer = Tokenizer.train(text, kind, vocab_size)
    print(
        f"  tokenizer ready: vocab={tokenizer.vocab_size} "
        f"({time.time() - start:.2f}s)"
    )
    print("encoding corpus...")
    start = time.time()
    ids = tokenizer.encode(text)
    tokens = np.asarray(ids, dtype=np.int32)
    split_index = max(1, int(len(tokens) * 0.9))
    meta = {
        "corpus": str(corpus_path),
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "tokenizer": kind,
        "vocab_size": tokenizer.vocab_size,
        "characters": len(text),
        "tokens": int(len(tokens)),
        "split_index": split_index,
        "compression_chars_per_token": len(text) / max(1, len(tokens)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    np.save(out_dir / "tokens.npy", tokens)
    (out_dir / "prepare.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"  {len(tokens):,} tokens, {meta['compression_chars_per_token']:.2f} "
        f"chars/token, split at {split_index:,} ({time.time() - start:.2f}s)"
    )
    return tokenizer, tokens, meta


def ensure_prepared(args):
    out_dir = Path(args.out)
    tok_path = out_dir / "tokenizer.json"
    tokens_path = out_dir / "tokens.npy"
    meta_path = out_dir / "prepare.json"
    if tok_path.exists() and tokens_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        same_kind = meta.get("tokenizer") == args.tokenizer
        same_vocab = (
            args.tokenizer == "char"
            or meta.get("vocab_size") == args.vocab_size
        )
        same_corpus = (
            Path(meta.get("corpus", "")).resolve()
            == Path(args.corpus).resolve()
            and meta.get("corpus_sha256")
            == hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest()
        )
        if same_kind and same_vocab and same_corpus:
            tokenizer = Tokenizer.load(tok_path)
            tokens = np.load(tokens_path)
            print(
                f"using prepared {tokenizer.kind} tokenizer: "
                f"vocab={tokenizer.vocab_size}, tokens={len(tokens):,}"
            )
            return tokenizer, tokens, meta
    return prepare_data(
        Path(args.corpus), out_dir, args.tokenizer, args.vocab_size
    )


# ---------------------------------------------------------------------------
# Transformer modules assembled from the copied forward functions
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return PARTS["rmsnorm"].rmsnorm(x, self.weight, self.epsilon)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w_g = nn.Parameter(torch.randn(dim, hidden) * 0.02)
        self.w_v = nn.Parameter(torch.randn(dim, hidden) * 0.02)
        self.w_o = nn.Parameter(torch.randn(hidden, dim) * 0.02)
        self.base_params = 3 * dim * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = PARTS["swiglu"].parameter_matched_swiglu(
            x, self.w_g, self.w_v, self.w_o, self.base_params
        )
        return out["output"]


class CausalGroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        block_size: int,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("d_model must be divisible by n_head")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.block_size = block_size

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.wq(x).view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.wk(x).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.wv(x).view(
            batch, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        positions = torch.arange(seq_len, device=x.device)
        # The copied RoPE helper expects q and k to have the same head count.
        # GQA has more query heads than KV heads, so rotate each separately.
        q = PARTS["rope"].rotary_embed(
            q, q, positions, self.inv_freq
        )["q_rotated"]
        k = PARTS["rope"].rotary_embed(
            k, k, positions, self.inv_freq
        )["k_rotated"]

        out = PARTS["gqa"].causal_gqa(q, k, v)
        out = out.transpose(1, 2).contiguous().view(
            batch, seq_len, self.dim
        )
        return self.wo(out)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden: int,
        block_size: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalGroupedQueryAttention(
            dim, n_heads, n_kv_heads, block_size
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        dim = config["d_model"]
        self.tok_emb = nn.Embedding(config["vocab_size"], dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    config["n_head"],
                    config["n_kv_head"],
                    config["hidden"],
                    config["block_size"],
                    config["dropout"],
                )
                for _ in range(config["n_layer"])
            ]
        )
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, config["vocab_size"], bias=False)
        self.apply(self._init_weights)
        if config.get("tie_weights", True):
            self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: int = 0,
    ) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        was_training = self.training
        self.eval()
        block_size = self.config["block_size"]
        vocab_size = self.config["vocab_size"]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]
            next_ids = []
            for row in next_logits:
                if temperature <= 0.0:
                    next_ids.append(int(torch.argmax(row).item()))
                    continue
                sampled = EXTRA_PARTS["sampling"].text_generation_sampling(
                    row.detach().cpu().tolist(),
                    list(range(vocab_size)),
                    float(temperature),
                    int(top_k),
                )
                top = sampled["top_tokens"]
                tokens = [int(token) for token, _prob in top]
                probs = np.asarray(
                    [float(prob) for _token, prob in top], dtype=np.float64
                )
                if probs.sum() <= 0:
                    probs = np.ones(len(tokens), dtype=np.float64) / len(tokens)
                else:
                    probs /= probs.sum()
                next_ids.append(int(rng.choice(tokens, p=probs)))
            idx = torch.cat(
                [
                    idx,
                    torch.tensor(
                        next_ids, dtype=torch.long, device=idx.device
                    ).unsqueeze(1),
                ],
                dim=1,
            )
        if was_training:
            self.train()
        return idx


# ---------------------------------------------------------------------------
# Batches, evaluation, checkpoints
# ---------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this Python. Use the torch126 venv: "
                "/home/d/.venvs/torch126/bin/python llm_system.py ... "
                "--device cuda"
            )
        try:
            torch.zeros(1, device="cuda")
        except Exception as exc:
            raise RuntimeError(
                "This PyTorch build cannot run on this GPU. Use the torch126 "
                "venv for the P2000 (sm_61)."
            ) from exc
        return torch.device("cuda")
    return torch.device("cpu")


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
):
    if len(data) <= block_size + 1:
        raise ValueError("dataset is too short for this block_size")
    starts = torch.randint(
        0, len(data) - block_size - 1, (batch_size,), generator=generator
    )
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(data, block_size, batch_size, generator)
        xb = xb.to(device)
        yb = yb.to(device)
        _logits, loss = model(xb, yb)
        losses.append(loss.detach().item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(
    out_dir: Path,
    model: TransformerLM,
    config: dict,
    filename: str = "model.pt",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / filename)
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def load_model(
    out_dir: Path,
    filename: str = "model.pt",
    device: torch.device | None = None,
):
    device = device or torch.device("cpu")
    config_path = out_dir / "config.json"
    tokenizer_path = out_dir / "tokenizer.json"
    checkpoint_path = out_dir / filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing tokenizer: {tokenizer_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = TransformerLM(config)
    state = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tokenizer = Tokenizer.load(tokenizer_path)
    return model, tokenizer, config


def clean_text(text: str) -> str:
    """Drop invalid UTF-8 replacement chars and control characters."""
    text = text.replace("�", "")
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def plot_training(metrics: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is not available; skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    train = metrics.get("train_loss", [])
    val = metrics.get("val_loss", [])
    steps = metrics.get("eval_steps", [])
    if steps:
        axes[0].plot(steps, train, marker="o", label="train")
        axes[0].plot(steps, val, marker="o", label="val")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("cross-entropy")
        axes[0].set_title("loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    else:
        axes[0].axis("off")
    if steps and len(metrics.get("learning_rate", [])) == len(steps):
        axes[1].plot(steps, metrics["learning_rate"], color="C2")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("learning rate")
        axes[1].set_title("learning-rate schedule")
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
def cmd_prepare(args) -> None:
    ensure_prepared(args)


def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer, tokens, meta = ensure_prepared(args)
    split_index = int(meta["split_index"])
    train_tokens = torch.from_numpy(tokens[:split_index].astype(np.int64))
    val_tokens = torch.from_numpy(tokens[split_index:].astype(np.int64))
    if len(train_tokens) <= args.block_size + 1:
        raise ValueError("training split is too small; use a larger corpus")
    if len(val_tokens) <= args.block_size + 1:
        raise ValueError("validation split is too small; use a larger corpus")

    config = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_kv_head": args.n_kv_head,
        "d_model": args.d_model,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "tie_weights": not args.no_tie,
        "tokenizer": tokenizer.kind,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
    }
    model = TransformerLM(config).to(device)
    print(
        f"model: {args.n_layer} layers, d_model={args.d_model}, "
        f"{args.n_head} heads, {args.n_kv_head} KV heads, "
        f"block={args.block_size}, vocab={tokenizer.vocab_size}, "
        f"params={model.num_params():,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    lr_schedule = EXTRA_PARTS["warmup"].warmup_cosine_schedule(
        args.lr, args.warmup, args.steps
    )
    batch_generator = torch.Generator().manual_seed(args.seed)
    eval_generator = torch.Generator().manual_seed(args.seed + 1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "eval_steps": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    best_val = float("inf")
    start = time.time()

    def evaluate_and_log(step: int) -> None:
        nonlocal best_val
        train_loss = estimate_loss(
            model,
            train_tokens,
            args.block_size,
            args.batch_size,
            args.eval_iters,
            eval_generator,
            device,
        )
        val_loss = estimate_loss(
            model,
            val_tokens,
            args.block_size,
            args.batch_size,
            args.eval_iters,
            eval_generator,
            device,
        )
        current_lr = (
            lr_schedule[min(step - 1, len(lr_schedule) - 1)]
            if step > 0
            else lr_schedule[0]
        )
        metrics["eval_steps"].append(step)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["learning_rate"].append(current_lr)
        print(
            f"step {step:4d}/{args.steps}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  ppl {math.exp(min(val_loss, 50)):.1f}  "
            f"lr {current_lr:.2e}  {time.time() - start:.1f}s"
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir, model, config, "best_model.pt")
            print("  saved new best checkpoint")
        if args.sample_tokens > 0:
            prompt_ids = tokenizer.encode(args.prompt) or [0]
            prompt = torch.tensor(
                [prompt_ids], dtype=torch.long, device=device
            )
            out = model.generate(
                prompt,
                args.sample_tokens,
                args.temperature,
                args.top_k,
                seed=args.seed + step,
            )
            text = clean_text(tokenizer.decode(out[0].tolist()))
            print("  sample: " + text.replace("\n", "\\n")[:400])

    evaluate_and_log(0)
    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_schedule[step - 1]
        for group in optimizer.param_groups:
            group["lr"] = lr
        xb, yb = get_batch(
            train_tokens, args.block_size, args.batch_size, batch_generator
        )
        xb = xb.to(device)
        yb = yb.to(device)
        _logits, loss = model(xb, yb)
        (loss / args.grad_accum).backward()
        if step % args.grad_accum == 0 or step == args.steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            print(
                f"  step {step:4d}  loss {loss.detach().item():.4f}  "
                f"lr {lr:.2e}  {time.time() - start:.1f}s"
            )
        if step % args.eval_interval == 0 or step == args.steps:
            evaluate_and_log(step)

    save_checkpoint(out_dir, model, config, "model.pt")
    if not (out_dir / "best_model.pt").exists():
        save_checkpoint(out_dir, model, config, "best_model.pt")
    metrics["best_val_loss"] = best_val
    metrics["elapsed_seconds"] = time.time() - start
    metrics["model"] = config
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if not args.no_plot:
        plot_training(metrics, out_dir / "training.png")
        print(f"plot written to {out_dir / 'training.png'}")
    print(f"model written to {out_dir / 'model.pt'}")
    print(f"best model written to {out_dir / 'best_model.pt'}")


def cmd_generate(args) -> None:
    out_dir = Path(args.out)
    device = resolve_device(args.device)
    filename = args.checkpoint or (
        "best_model.pt" if (out_dir / "best_model.pt").exists() else "model.pt"
    )
    model, tokenizer, config = load_model(out_dir, filename, device)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    start = time.time()
    out = model.generate(
        idx,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        seed=args.seed,
    )
    text = clean_text(tokenizer.decode(out[0].tolist()))
    print(text)
    print(
        f"\n[{len(out[0]) - len(prompt_ids)} new tokens, "
        f"{time.time() - start:.1f}s]"
    )
    if args.save:
        Path(args.save).write_text(text, encoding="utf-8")


def cmd_eval(args) -> None:
    out_dir = Path(args.out)
    device = resolve_device(args.device)
    filename = args.checkpoint or (
        "best_model.pt" if (out_dir / "best_model.pt").exists() else "model.pt"
    )
    model, tokenizer, config = load_model(out_dir, filename, device)
    tokens = np.load(out_dir / "tokens.npy")
    meta = json.loads((out_dir / "prepare.json").read_text(encoding="utf-8"))
    val_tokens = torch.from_numpy(
        tokens[int(meta["split_index"]) :].astype(np.int64)
    )
    generator = torch.Generator().manual_seed(args.seed)
    loss = estimate_loss(
        model,
        val_tokens,
        config["block_size"],
        config["batch_size"] if "batch_size" in config else 16,
        args.iters,
        generator,
        device,
    )
    print(f"validation loss {loss:.4f}  perplexity {math.exp(min(loss, 50)):.2f}")

    # Also run the copied single-sequence perplexity helper (eval mode).
    model.eval()
    xb, _yb = get_batch(
        val_tokens, config["block_size"], 1, generator
    )
    with torch.no_grad():
        logits, _ = model(xb.to(device), None)
    copied = EXTRA_PARTS["perplexity"].compute_perplexity(
        xb[0].tolist(), logits[0].cpu().tolist()
    )
    print(
        f"copied-perplexity helper: avg_nll={copied['avg_nll']} "
        f"ppl={copied['perplexity']}"
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def cmd_selftest(args) -> None:
    cs336_files = list(CS336_DIR.glob("*"))
    extra_files = list(EXTRA_DIR.glob("*"))
    assert len(cs336_files) >= 113, f"expected 113 CS336 files, got {len(cs336_files)}"
    assert len(extra_files) >= 4, f"expected 4 extra files, got {len(extra_files)}"

    # Tokenizer round-trips.
    char_tok = Tokenizer.train("hello world", "char", 0)
    assert char_tok.decode(char_tok.encode("hello world")) == "hello world"
    bpe_text = "hello hello world hello world " * 20
    bpe_tok = Tokenizer.train(bpe_text, "bpe", 260)
    assert bpe_tok.decode(bpe_tok.encode("hello world")) == "hello world"

    # Copied tokenizer/normalization/MLP pieces.
    x = torch.randn(2, 4, 8)
    g = torch.ones(8)
    y = PARTS["rmsnorm"].rmsnorm(x, g, 1e-5)
    assert y.shape == x.shape and torch.isfinite(y).all()
    w_g = torch.randn(8, 16)
    w_v = torch.randn(8, 16)
    w_o = torch.randn(16, 8)
    swiglu = PARTS["swiglu"].parameter_matched_swiglu(
        x, w_g, w_v, w_o, 3 * 8 * 16
    )
    assert swiglu["output"].shape == x.shape and swiglu["hidden_width"] == 16

    # RoPE: position zero is the identity.
    q = torch.randn(2, 2, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8)
    )
    rotated = PARTS["rope"].rotary_embed(q, k, torch.arange(5), inv_freq)
    assert rotated["q_rotated"].shape == q.shape
    assert torch.allclose(
        rotated["q_rotated"][:, :, 0], q[:, :, 0], atol=1e-6
    )

    # GQA is causal: changing V at the last position leaves earlier outputs.
    q4 = torch.randn(2, 4, 5, 8)
    k2 = torch.randn(2, 2, 5, 8)
    v2 = torch.randn(2, 2, 5, 8)
    attn = PARTS["gqa"].causal_gqa(q4, k2, v2)
    v2_changed = v2.clone()
    v2_changed[:, :, -1] += 1.0
    attn_changed = PARTS["gqa"].causal_gqa(q4, k2, v2_changed)
    assert attn.shape == (2, 4, 5, 8)
    assert torch.allclose(attn[:, :, :-1], attn_changed[:, :, :-1], atol=1e-5)
    assert not torch.allclose(attn[:, :, -1], attn_changed[:, :, -1])

    # KV-cache decode step.
    kv_out = PARTS["kv"].kv_cache_decode_step(
        torch.randn(2, 4, 8),
        torch.randn(2, 2, 8),
        torch.randn(2, 2, 8),
        torch.randn(2, 2, 3, 8),
        torch.randn(2, 2, 3, 8),
        4,
        2,
    )
    assert kv_out["output"].shape == (2, 4, 8)
    assert kv_out["new_key_cache"].shape == (2, 2, 4, 8)

    # EInsum attention scores.
    scores = PARTS["attn_scores"].attention_scores(
        torch.randn(2, 5, 8), torch.randn(2, 5, 8), 2
    )
    assert scores.shape == (2, 2, 5, 5)

    # Cross-track extras.
    schedule = EXTRA_PARTS["warmup"].warmup_cosine_schedule(1e-3, 10, 100)
    assert len(schedule) == 100
    assert abs(schedule[0] - 1e-4) < 1e-12 and schedule[-1] < 1e-5
    sampling = EXTRA_PARTS["sampling"].text_generation_sampling(
        [0.0, 1.0, 2.0], [0, 1, 2], 1.0, 2
    )
    assert len(sampling["top_tokens"]) == 2
    assert abs(sum(sampling["probabilities"]) - 1.0) < 1e-3
    copied_ppl = EXTRA_PARTS["perplexity"].compute_perplexity(
        [0, 1, 2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    assert copied_ppl["perplexity"] > 0.0
    causal_ref = EXTRA_PARTS["causal_attn"].causal_attention(
        torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    )
    assert causal_ref.shape == (2, 4, 8)

    # The assembled model: forward, backward, optimizer step, and generation.
    config = {
        "vocab_size": 64,
        "block_size": 16,
        "n_layer": 1,
        "n_head": 2,
        "n_kv_head": 1,
        "d_model": 32,
        "hidden": 64,
        "dropout": 0.0,
        "tie_weights": True,
    }
    model = TransformerLM(config)
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 64) and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    generated = model.generate(
        torch.randint(0, 64, (2, 4)), 5, temperature=0.8, top_k=10, seed=0
    )
    assert generated.shape == (2, 9)

    print(
        "PASS: 113 CS336 files + 4 cross-track files copied; BPE/char "
        "tokenizer, RMSNorm, SwiGLU, RoPE, causal GQA, KV cache, warmup, "
        "sampling, perplexity, transformer forward/backward, and generation "
        "all OK"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", default=str(HERE / "data" / "corpus.txt"))
    parser.add_argument("--tokenizer", choices=["bpe", "char", "hf-bpe"], default="bpe")
    parser.add_argument("--vocab-size", type=int, default=384)
    parser.add_argument("--out", default=str(HERE / "runs" / "llm"))


def add_train_args(parser: argparse.ArgumentParser) -> None:
    add_data_args(parser)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-kv-head", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-tie", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=150)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-plot", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A tiny LLM assembled from the copied CS336 parts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    test = sub.add_parser("selftest", help="check copied parts and the glue")
    test.set_defaults(func=cmd_selftest)

    prepare = sub.add_parser("prepare", help="train the tokenizer and encode the corpus")
    add_data_args(prepare)
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="train the transformer")
    add_train_args(train)
    train.set_defaults(func=cmd_train)

    gen = sub.add_parser("generate", help="generate text from a checkpoint")
    gen.add_argument("--out", default=str(HERE / "runs" / "llm"))
    gen.add_argument("--checkpoint", default=None)
    gen.add_argument("--prompt", default="The ")
    gen.add_argument("--max-new-tokens", type=int, default=200)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=40)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    gen.add_argument("--save", default=None)
    gen.set_defaults(func=cmd_generate)

    ev = sub.add_parser("eval", help="compute validation loss/perplexity")
    ev.add_argument("--out", default=str(HERE / "runs" / "llm"))
    ev.add_argument("--checkpoint", default=None)
    ev.add_argument("--iters", type=int, default=20)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    if getattr(args, "vocab_size", 384) <= 256 and args.tokenizer == "bpe":
        parser.error("--vocab-size must be greater than 256 for BPE")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
