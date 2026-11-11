#!/usr/bin/env python3
"""Instruction tuning with the copied CS336 assistant-only SFT mask.

The script loads a pretrained checkpoint from ``llm_system.py``, formats
Alpaca-style instruction/response examples, packs them into fixed blocks, and
trains only on response tokens using
``cs336-l15-assistant-only-sft-mask.py``.

Quick start (after the scaled LLM exists in runs/llm10m):

    python sft_system.py selftest
    python sft_system.py train --checkpoint-dir runs/llm10m --device cuda
    python llm_system.py generate --out runs/sft --prompt "### Instruction:\\nName three colors.\\n### Response:\\n"
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import llm_system as L

HERE = Path(__file__).resolve().parent
SFT_MASK = L.load_part(L.CS336_DIR, "cs336-l15-assistant-only-sft-mask")
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def format_example(example: dict, tokenizer) -> tuple[list[int], list[int]] | None:
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()
    if not output:
        return None
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:\n"
    response = output + "\n\n"
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    if not response_ids:
        return None
    return (
        prompt_ids + response_ids,
        [0] * len(prompt_ids) + [1] * len(response_ids),
    )


def pack_examples(
    examples: list[dict], tokenizer, block_size: int
) -> list[dict]:
    """Right-pad one example per block.

    Padding comes after the response, so causal attention on the response
    tokens never sees it.  Keeping examples separate avoids cross-example
    confusion in a small SFT run.
    """
    blocks = []
    pad_id = tokenizer.pad_id
    for example in examples:
        item = format_example(example, tokenizer)
        if item is None:
            continue
        ids, roles = item
        if len(ids) > block_size:
            ids = ids[:block_size]
            roles = roles[:block_size]
        if not any(roles):
            continue
        pad_len = block_size - len(ids)
        blocks.append(
            {
                "ids": ids + [pad_id] * pad_len,
                "roles": roles + [0] * pad_len,
                "attention": [1] * len(ids) + [0] * pad_len,
            }
        )
    return blocks


class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, blocks: list[dict]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int):
        block = self.blocks[index]
        return (
            torch.tensor(block["ids"], dtype=torch.long),
            torch.tensor(block["roles"], dtype=torch.long),
            torch.tensor(block["attention"], dtype=torch.bool),
        )


def collate_blocks(batch):
    ids = torch.stack([item[0] for item in batch])
    roles = torch.stack([item[1] for item in batch])
    attention = torch.stack([item[2] for item in batch])
    return ids, roles, attention


def aggregate_loss(labels: torch.Tensor, logits: torch.Tensor):
    """Sum loss over assistant labels and count the valid labels."""
    flat_labels = labels.reshape(-1)
    valid = flat_labels != -100
    if valid.sum() == 0:
        return torch.zeros((), device=logits.device), 0
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        flat_labels,
        ignore_index=-100,
        reduction="sum",
    )
    return loss, int(valid.sum().item())


def batch_loss(model, ids, roles, attention, device):
    ids = ids.to(device)
    roles = roles.to(device)
    attention = attention.to(device)
    masked = SFT_MASK.assistant_only_sft_mask(
        ids, roles, attention, assistant_role=1
    )
    logits, _ = model(ids, None)
    return aggregate_loss(masked["labels"], logits)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    tokens = 0
    for ids, roles, attention in loader:
        loss, count = batch_loss(model, ids, roles, attention, device)
        total += float(loss.detach())
        tokens += count
    model.train()
    return total / max(1, tokens)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def response_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n### Response:\n"


def generate_response(
    model,
    tokenizer,
    instruction: str,
    device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    prompt = response_prompt(instruction)
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens, temperature, top_k, seed=seed
    )
    text = L.clean_text(tokenizer.decode(out[0].tolist()))
    return text


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_prepare(args) -> None:
    alpaca = Path(args.alpaca)
    if not alpaca.exists():
        print(f"downloading Alpaca to {alpaca} ...")
        alpaca.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ALPACA_URL, alpaca)
    full = json.loads(alpaca.read_text(encoding="utf-8"))
    subset = [
        item
        for item in full
        if item.get("output", "").strip()
    ][: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    print(f"wrote {len(subset)} examples to {out}")


def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = L.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    model, tokenizer, model_config = L.load_model(
        checkpoint_dir, args.checkpoint, device
    )
    examples = json.loads(Path(args.data).read_text(encoding="utf-8"))
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    train_blocks = pack_examples(train_examples, tokenizer, args.block_size)
    val_blocks = pack_examples(val_examples, tokenizer, args.block_size)
    if len(train_blocks) < args.batch_size:
        raise ValueError(
            f"only {len(train_blocks)} training blocks; use more data or a "
            f"smaller block size"
        )
    if not val_blocks:
        val_blocks = train_blocks[-max(1, len(train_blocks) // 20) :]
    assistant_tokens = sum(
        sum(block["roles"]) for block in train_blocks
    )
    print(
        f"examples: {len(train_examples)} train / {len(val_examples)} val; "
        f"blocks: {len(train_blocks)} / {len(val_blocks)}; "
        f"assistant tokens: {assistant_tokens:,}; device={device}"
    )

    train_loader = torch.utils.data.DataLoader(
        BlockDataset(train_blocks),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_blocks,
    )
    val_loader = torch.utils.data.DataLoader(
        BlockDataset(val_blocks),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_blocks,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = L.EXTRA_PARTS["warmup"].warmup_cosine_schedule(
        args.lr, args.warmup, args.steps
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint_dir / "tokenizer.json", out_dir / "tokenizer.json")
    config = dict(model_config)
    config["sft"] = {
        "data": str(args.data),
        "examples": len(train_examples),
        "assistant_tokens": int(assistant_tokens),
        "pretrained": str(checkpoint_dir),
        "steps": args.steps,
        "lr": args.lr,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    metrics = {"train_loss": [], "val_loss": [], "steps": [], "samples": []}
    best_val = float("inf")
    start = time.time()
    train_iter = iter(train_loader)
    model.train()

    val_loss = evaluate(model, val_loader, device)
    print(f"step 0  val {val_loss:.4f}")

    for step in range(1, args.steps + 1):
        lr = schedule[step - 1]
        for group in optimizer.param_groups:
            group["lr"] = lr
        try:
            ids, roles, attention = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            ids, roles, attention = next(train_iter)

        loss, count = batch_loss(model, ids, roles, attention, device)
        if count == 0:
            continue
        (loss / (count * args.grad_accum)).backward()
        if step % args.grad_accum == 0 or step == args.steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            print(
                f"  step {step:4d}  loss {float(loss.detach()) / count:.4f}  "
                f"lr {lr:.2e}  {time.time() - start:.1f}s"
            )

        if step % args.eval_interval == 0 or step == args.steps:
            train_loss = float(loss.detach()) / count
            val_loss = evaluate(model, val_loader, device)
            metrics["steps"].append(step)
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            print(
                f"step {step:4d}/{args.steps}  train {train_loss:.4f}  "
                f"val {val_loss:.4f}  {time.time() - start:.1f}s"
            )
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                print("  saved new best checkpoint")
            if args.sample_tokens > 0:
                sample = generate_response(
                    model,
                    tokenizer,
                    args.sample_instruction,
                    device,
                    args.sample_tokens,
                    args.temperature,
                    args.top_k,
                    seed=args.seed + step,
                )
                metrics["samples"].append(
                    {"step": step, "instruction": args.sample_instruction,
                     "text": sample}
                )
                print("  sample: " + sample.replace("\n", "\\n")[:400])

    torch.save(model.state_dict(), out_dir / "model.pt")
    metrics["best_val_loss"] = best_val
    metrics["elapsed_seconds"] = time.time() - start
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"best val loss: {best_val:.4f}")
    print(f"model written to {out_dir / 'model.pt'}")


def cmd_eval(args) -> None:
    device = L.resolve_device(args.device)
    model, tokenizer, _config = L.load_model(
        Path(args.out), args.checkpoint, device
    )
    examples = json.loads(Path(args.data).read_text(encoding="utf-8"))
    random.seed(args.seed)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val_blocks = pack_examples(examples[:n_val], tokenizer, args.block_size)
    loader = torch.utils.data.DataLoader(
        BlockDataset(val_blocks),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_blocks,
    )
    loss = evaluate(model, loader, device)
    print(f"assistant-token validation loss: {loss:.4f}")


def cmd_selftest(args) -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    roles = torch.tensor([[0, 0, 1, 1, 0]])
    attention = torch.ones_like(ids, dtype=torch.bool)
    masked = SFT_MASK.assistant_only_sft_mask(ids, roles, attention, 1)
    assert masked["labels"].tolist() == [[-100, 3, 4, -100, -100]]
    assert masked["loss_mask"].tolist() == [[False, True, True, False, False]]

    tokenizer = L.Tokenizer.train(
        "### Instruction:\nName a color.\n### Response:\nblue\n\n", "char", 0
    )
    example = {
        "instruction": "Name a color.",
        "input": "",
        "output": "blue",
    }
    item = format_example(example, tokenizer)
    assert item is not None
    ids_list, roles_list = item
    assert 1 in roles_list and 0 in roles_list
    blocks = pack_examples([example] * 5, tokenizer, 64)
    assert blocks and all(any(block["roles"]) for block in blocks)

    config = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": 64,
        "n_layer": 1,
        "n_head": 2,
        "n_kv_head": 1,
        "d_model": 32,
        "hidden": 64,
        "dropout": 0.0,
        "tie_weights": True,
    }
    model = L.TransformerLM(config)
    block = blocks[0]
    batch_ids = torch.tensor([block["ids"]] * 2)
    batch_roles = torch.tensor([block["roles"]] * 2)
    batch_attn = torch.tensor([block["attention"]] * 2, dtype=torch.bool)
    loss, count = batch_loss(model, batch_ids, batch_roles, batch_attn, "cpu")
    assert count > 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    print(
        "PASS: copied assistant-only SFT mask, Alpaca formatting, block "
        "packing, model forward/backward, and response-token loss all OK"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Instruction tuning with the copied CS336 SFT mask."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    test = sub.add_parser("selftest")
    test.set_defaults(func=cmd_selftest)

    prepare = sub.add_parser("prepare", help="download Alpaca and build a subset")
    prepare.add_argument("--alpaca", default=str(HERE / "data" / "sft" / "alpaca_full.json"))
    prepare.add_argument("--out", default=str(HERE / "data" / "sft" / "alpaca_subset.json"))
    prepare.add_argument("--limit", type=int, default=2000)
    prepare.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="fine-tune a pretrained LLM")
    train.add_argument("--checkpoint-dir", default=str(HERE / "runs" / "llm10m"))
    train.add_argument("--checkpoint", default="best_model.pt")
    train.add_argument("--data", default=str(HERE / "data" / "sft" / "alpaca_subset.json"))
    train.add_argument("--out", default=str(HERE / "runs" / "sft"))
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--block-size", type=int, default=128)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup", type=int, default=20)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--val-fraction", type=float, default=0.05)
    train.add_argument("--eval-interval", type=int, default=100)
    train.add_argument("--log-every", type=int, default=50)
    train.add_argument("--sample-tokens", type=int, default=80)
    train.add_argument("--sample-instruction", default="Name three colors.")
    train.add_argument("--temperature", type=float, default=0.7)
    train.add_argument("--top-k", type=int, default=40)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="assistant-token validation loss")
    ev.add_argument("--out", default=str(HERE / "runs" / "sft"))
    ev.add_argument("--checkpoint", default="best_model.pt")
    ev.add_argument("--data", default=str(HERE / "data" / "sft" / "alpaca_subset.json"))
    ev.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ev.add_argument("--block-size", type=int, default=128)
    ev.add_argument("--batch-size", type=int, default=2)
    ev.add_argument("--val-fraction", type=float, default=0.05)
    ev.add_argument("--seed", type=int, default=0)
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
