#!/usr/bin/env python3
"""Sentence-embedding semantic search trained on STS-B.

This system is built from the copied NLP and linear-algebra parts:

    math-linear-algebra/la-cosine-similarity.py
    math-linear-algebra/la-vector-norms.py
    cracking-nlp/nlp-sentence-embeddings.py
    cracking-nlp/nlp-embedding-similarity.py
    cracking-nlp/nlp-document-ranking-bm25.py
    cracking-nlp/nlp-tokenization.py

The glue adds a word tokenizer, a small bidirectional Transformer bi-encoder,
a cosine-regression + in-batch-contrastive objective, Spearman and retrieval
evaluation, and a search CLI.

Quick start:

    python semantic_search.py download
    python semantic_search.py selftest
    python semantic_search.py train --device cuda
    python semantic_search.py eval --device cuda
    python semantic_search.py search --query "A plane is taking off." --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

# The semantic model can run on CPU with the system torch or on the P2000 with
# the torch126 venv; the caller chooses --device.
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PARTS_DIR = HERE / "parts" / "semantic"
DATA_DIR = HERE / "data"


# ---------------------------------------------------------------------------
# Copied solutions
# ---------------------------------------------------------------------------
def load_part(name: str):
    path = PARTS_DIR / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing copied part: {path}")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARTS = {
    "cosine": load_part("la-cosine-similarity"),
    "norms": load_part("la-vector-norms"),
    "sentence_emb": load_part("nlp-sentence-embeddings"),
    "emb_similarity": load_part("nlp-embedding-similarity"),
    "bm25": load_part("nlp-document-ranking-bm25"),
    "tokenization": load_part("nlp-tokenization"),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
STSB_URL = "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip"
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def download_stsb(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "STS-B"
    if (target / "train.tsv").exists():
        print(f"STS-B already present at {target}")
        return target
    archive = data_dir / "STS-B.zip"
    print(f"downloading {STSB_URL} ...")
    urllib.request.urlretrieve(STSB_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    archive.unlink(missing_ok=True)
    print(f"extracted to {target}")
    return target


def read_stsb(path: Path, labeled: bool = True) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    i1 = header.index("sentence1")
    i2 = header.index("sentence2")
    iscore = header.index("score") if labeled else None
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != len(header):
            continue
        rows.append(
            {
                "sentence1": fields[i1],
                "sentence2": fields[i2],
                "score": float(fields[iscore]) if labeled else None,
            }
        )
    return rows


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class WordTokenizer:
    PAD = 0
    UNK = 1

    def __init__(self, words: list[str]):
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def train(cls, sentences: list[str], vocab_size: int = 12000):
        counts = Counter()
        for sentence in sentences:
            counts.update(tokenize(sentence))
        words = [token for token, _ in counts.most_common(vocab_size - 2)]
        return cls(words)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(token, self.UNK) for token in tokenize(text)]
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        return " ".join(
            self.itos[int(i)] for i in ids if int(i) > self.UNK
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"words": self.itos[2:]}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8"))["words"])


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: list[dict], tokenizer: WordTokenizer, max_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return {
            "ids1": self.tokenizer.encode(pair["sentence1"], self.max_len),
            "ids2": self.tokenizer.encode(pair["sentence2"], self.max_len),
            "score": float(pair["score"]),
        }


def collate_pairs(batch: list[dict]):
    max_len = max(
        max(len(item["ids1"]), len(item["ids2"])) for item in batch
    )
    max_len = max(1, max_len)
    ids1 = torch.zeros(len(batch), max_len, dtype=torch.long)
    ids2 = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask1 = torch.zeros(len(batch), max_len, dtype=torch.bool)
    mask2 = torch.zeros(len(batch), max_len, dtype=torch.bool)
    scores = torch.zeros(len(batch), dtype=torch.float32)
    for i, item in enumerate(batch):
        a = item["ids1"]
        b = item["ids2"]
        ids1[i, : len(a)] = torch.tensor(a, dtype=torch.long)
        ids2[i, : len(b)] = torch.tensor(b, dtype=torch.long)
        mask1[i, : len(a)] = True
        mask2[i, : len(b)] = True
        scores[i] = item["score"]
    return ids1, mask1, ids2, mask2, scores


# ---------------------------------------------------------------------------
# Model and losses
# ---------------------------------------------------------------------------
class SentenceEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.embed = nn.Embedding(
            config["vocab_size"], config["d_model"], padding_idx=0
        )
        self.pos_embed = nn.Embedding(
            config["max_len"], config["d_model"]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["n_heads"],
            dim_feedforward=config["ff"],
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config["layers"],
            enable_nested_tensor=False,
        )
        self.projection = nn.Linear(config["d_model"], config["d_model"])

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.embed(ids) + self.pos_embed(positions)
        h = self.encoder(x, src_key_padding_mask=~mask)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        projected = torch.tanh(self.projection(pooled))
        return F.normalize(projected, dim=-1)


def contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    valid = scores >= threshold
    if valid.sum() < 2:
        return torch.zeros((), device=emb_a.device)
    anchors = valid.nonzero(as_tuple=True)[0]
    logits_ab = emb_a[valid] @ emb_b.T / temperature
    logits_ba = emb_b[valid] @ emb_a.T / temperature
    labels = anchors
    return 0.5 * (
        F.cross_entropy(logits_ab, labels)
        + F.cross_entropy(logits_ba, labels)
    )


def regression_loss(
    emb_a: torch.Tensor, emb_b: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    predicted = (emb_a * emb_b).sum(dim=-1)
    target = scores / 5.0 * 2.0 - 1.0
    return F.mse_loss(predicted, target)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    rx = average_ranks(np.asarray(x, dtype=np.float64))
    ry = average_ranks(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


@torch.no_grad()
def encode_texts(
    model: SentenceEncoder,
    tokenizer: WordTokenizer,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = [tokenizer.encode(text, max_len) for text in chunk]
        max_chunk = max(1, max(len(ids) for ids in encoded))
        ids = torch.zeros(len(chunk), max_chunk, dtype=torch.long)
        mask = torch.zeros(len(chunk), max_chunk, dtype=torch.bool)
        for i, items in enumerate(encoded):
            ids[i, : len(items)] = torch.tensor(items, dtype=torch.long)
            mask[i, : len(items)] = True
        emb = model(ids.to(device), mask.to(device))
        outputs.append(emb.cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1))


@torch.no_grad()
def predict_similarity(
    model: SentenceEncoder,
    tokenizer: WordTokenizer,
    pairs: list[dict],
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        ids1, mask1, ids2, mask2, _scores = collate_pairs(
            [
                {
                    "ids1": tokenizer.encode(p["sentence1"], max_len),
                    "ids2": tokenizer.encode(p["sentence2"], max_len),
                    "score": p["score"] if p["score"] is not None else 0.0,
                }
                for p in chunk
            ]
        )
        a = model(ids1.to(device), mask1.to(device))
        b = model(ids2.to(device), mask2.to(device))
        predictions.extend((a * b).sum(dim=-1).cpu().tolist())
    return np.asarray(predictions)


def retrieval_eval(
    model: SentenceEncoder,
    tokenizer: WordTokenizer,
    pairs: list[dict],
    max_len: int,
    batch_size: int,
    device: torch.device,
    max_queries: int = 0,
) -> dict:
    queries = [p["sentence1"] for p in pairs]
    positives = [p["sentence2"] for p in pairs]
    candidate_texts = list(dict.fromkeys(positives))
    candidate_index = {text: i for i, text in enumerate(candidate_texts)}

    if max_queries and len(queries) > max_queries:
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(queries), size=max_queries, replace=False)
        queries = [queries[i] for i in chosen]
        positives = [positives[i] for i in chosen]

    query_emb = encode_texts(
        model, tokenizer, queries, max_len, batch_size, device
    )
    cand_emb = encode_texts(
        model, tokenizer, candidate_texts, max_len, batch_size, device
    )
    sims = query_emb @ cand_emb.T
    target = np.asarray([candidate_index[p] for p in positives])
    ranks = (sims > sims[np.arange(len(target)), target][:, None]).sum(axis=1) + 1
    return {
        "queries": int(len(target)),
        "candidates": int(len(candidate_texts)),
        "accuracy_at_1": float(np.mean(ranks == 1)),
        "accuracy_at_5": float(np.mean(ranks <= 5)),
        "mrr_at_10": float(np.mean(1.0 / np.minimum(ranks, 10))),
        "random_baseline_at_1": 1.0 / max(1, len(candidate_texts)),
    }


def bm25_baseline(
    pairs: list[dict], max_queries: int = 100, candidates_per_query: int = 200
) -> dict:
    if not pairs:
        return {"accuracy_at_1": 0.0}
    rng = np.random.default_rng(1)
    n = min(max_queries, len(pairs))
    chosen = rng.choice(len(pairs), size=n, replace=False)
    all_positives = [p["sentence2"] for p in pairs]
    hits = 0
    for idx in chosen:
        positive = pairs[int(idx)]["sentence2"]
        negative_pool = [text for text in all_positives if text != positive]
        rng.shuffle(negative_pool)
        candidates = [positive] + negative_pool[: candidates_per_query - 1]
        rng.shuffle(candidates)
        docs = [tokenize(text) for text in candidates]
        query = tokenize(pairs[int(idx)]["sentence1"])
        ranking = PARTS["bm25"].document_ranking(docs, query)["ranking"]
        if ranking and candidates[ranking[0]] == positive:
            hits += 1
    return {"accuracy_at_1": hits / n, "queries": n, "candidates": candidates_per_query}


# ---------------------------------------------------------------------------
# Training / evaluation / search commands
# ---------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available in this Python. Use the torch126 venv: "
                "/home/d/.venvs/torch126/bin/python semantic_search.py ..."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)

    stsb_dir = Path(args.data_dir)
    train_pairs = read_stsb(stsb_dir / "train.tsv", labeled=True)
    dev_pairs = read_stsb(stsb_dir / "dev.tsv", labeled=True)
    tokenizer = WordTokenizer.train(
        [p["sentence1"] for p in train_pairs]
        + [p["sentence2"] for p in train_pairs]
        + [p["sentence1"] for p in dev_pairs]
        + [p["sentence2"] for p in dev_pairs],
        args.vocab_size,
    )
    config = {
        "vocab_size": tokenizer.vocab_size,
        "max_len": args.max_len,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "ff": args.ff,
        "layers": args.layers,
        "dropout": args.dropout,
    }
    model = SentenceEncoder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = torch.utils.data.DataLoader(
        PairDataset(train_pairs, tokenizer, args.max_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
    )
    print(
        f"STS-B: {len(train_pairs)} train pairs, {len(dev_pairs)} dev pairs; "
        f"vocab={tokenizer.vocab_size}; device={device}"
    )
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    best_rho = -2.0
    history = {"loss": [], "dev_spearman": [], "epoch_seconds": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batches = 0
        for ids1, mask1, ids2, mask2, scores in train_loader:
            ids1 = ids1.to(device)
            mask1 = mask1.to(device)
            ids2 = ids2.to(device)
            mask2 = mask2.to(device)
            scores = scores.to(device)
            emb_a = model(ids1, mask1)
            emb_b = model(ids2, mask2)
            loss = regression_loss(emb_a, emb_b, scores)
            loss = loss + args.con_weight * contrastive_loss(
                emb_a,
                emb_b,
                scores,
                args.positive_threshold,
                args.temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1

        dev_pred = predict_similarity(
            model,
            tokenizer,
            dev_pairs,
            args.max_len,
            args.batch_size,
            device,
        )
        rho = spearman(dev_pred, [p["score"] for p in dev_pairs])
        avg_loss = total_loss / max(1, batches)
        history["loss"].append(avg_loss)
        history["dev_spearman"].append(rho)
        history["epoch_seconds"].append(time.time() - epoch_start)
        print(
            f"epoch {epoch:2d}/{args.epochs}  loss {avg_loss:.4f}  "
            f"dev Spearman {rho:.4f}  {time.time() - epoch_start:.1f}s"
        )
        if rho > best_rho:
            best_rho = rho
            torch.save(model.state_dict(), out_dir / "model.pt")
            print("  saved new best model")

    history["best_dev_spearman"] = best_rho
    history["elapsed_seconds"] = time.time() - start
    (out_dir / "metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"best dev Spearman: {best_rho:.4f}")
    print(f"model written to {out_dir / 'model.pt'}")


def load_semantic_model(checkpoint_dir: Path, device: torch.device):
    config = json.loads((checkpoint_dir / "config.json").read_text())
    tokenizer = WordTokenizer.load(checkpoint_dir / "tokenizer.json")
    model = SentenceEncoder(config).to(device)
    state = torch.load(
        checkpoint_dir / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, config


def cmd_eval(args) -> None:
    device = resolve_device(args.device)
    model, tokenizer, config = load_semantic_model(Path(args.checkpoint_dir), device)
    stsb_dir = Path(args.data_dir)
    split = "dev" if args.split == "dev" else "test"
    labeled = split == "dev"
    pairs = read_stsb(stsb_dir / f"{split}.tsv", labeled=labeled)

    if labeled:
        pred = predict_similarity(
            model,
            tokenizer,
            pairs,
            config["max_len"],
            args.batch_size,
            device,
        )
        rho = spearman(pred, [p["score"] for p in pairs])
        print(f"{split} Spearman: {rho:.4f}")
    else:
        print(f"{split} has no labels; running retrieval only")

    retrieval_pairs = read_stsb(stsb_dir / "dev.tsv", labeled=True)
    retrieval = retrieval_eval(
        model,
        tokenizer,
        retrieval_pairs,
        config["max_len"],
        args.batch_size,
        device,
        max_queries=args.max_queries,
    )
    print(f"retrieval: {retrieval}")
    bm25 = bm25_baseline(
        retrieval_pairs,
        max_queries=args.bm25_queries,
        candidates_per_query=args.bm25_candidates,
    )
    print(f"BM25 baseline: {bm25}")

    # spot-check the copied cosine function
    example = retrieval_pairs[0]
    emb = encode_texts(
        model,
        tokenizer,
        [example["sentence1"], example["sentence2"]],
        config["max_len"],
        args.batch_size,
        device,
    )
    copied_cos = PARTS["cosine"].cosine_similarity(
        emb[0].tolist(), emb[1].tolist()
    )
    print(f"copied cosine on first pair: {copied_cos:.4f}")

    results = {"retrieval": retrieval, "bm25": bm25}
    if labeled:
        results["spearman"] = rho
    (Path(args.checkpoint_dir) / "eval.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def cmd_search(args) -> None:
    device = resolve_device(args.device)
    model, tokenizer, config = load_semantic_model(Path(args.checkpoint_dir), device)
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        if corpus_path == DATA_DIR / "stsb_sentences.txt":
            corpus_path = build_search_corpus()
        else:
            raise FileNotFoundError(f"missing corpus: {corpus_path}")
    sentences = [
        line.strip()
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_emb = encode_texts(
        model, tokenizer, [args.query], config["max_len"], args.batch_size, device
    )[0]
    cand_emb = encode_texts(
        model, tokenizer, sentences, config["max_len"], args.batch_size, device
    )
    # Use the copied linear-algebra cosine function for the final ranking.
    scores = [
        PARTS["cosine"].cosine_similarity(query_emb.tolist(), vector.tolist())
        for vector in cand_emb
    ]
    order = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[
        : args.top_k
    ]
    print(f'query: {args.query}')
    for rank, idx in enumerate(order, 1):
        print(f"{rank:2d}. {scores[idx]:+.4f}  {sentences[idx]}")


def build_search_corpus(
    stsb_dir: Path = DATA_DIR / "STS-B",
    out: Path = DATA_DIR / "stsb_sentences.txt",
) -> Path:
    pairs = read_stsb(stsb_dir / "dev.tsv", labeled=True)
    seen = []
    seen_set = set()
    for pair in pairs:
        for text in (pair["sentence1"], pair["sentence2"]):
            if text not in seen_set:
                seen_set.add(text)
                seen.append(text)
    out.write_text("\n".join(seen) + "\n", encoding="utf-8")
    print(f"wrote {len(seen)} search sentences to {out}")
    return out


def cmd_download(args) -> None:
    download_stsb()
    build_search_corpus()


def cmd_selftest(args) -> None:
    # Data-independent checks.
    assert len(list(PARTS_DIR.glob("*.py"))) >= 6
    assert abs(PARTS["cosine"].cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(PARTS["cosine"].cosine_similarity([1, 0], [0, 1])) < 1e-9

    tok = WordTokenizer.train(["The cat sat.", "A dog ran."], 100)
    assert tok.decode(tok.encode("the cat", 10)) == "the cat"
    assert tok.encode("unknownword", 10)[0] == WordTokenizer.UNK

    pooled = PARTS["sentence_emb"].sentence_embeddings(
        [["a", "b"]], {"a": [1.0, 0.0], "b": [0.0, 1.0]}, method="mean"
    )
    assert np.allclose(pooled[0], [0.5, 0.5])
    ranked = PARTS["emb_similarity"].embedding_similarity(
        [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], k=1
    )
    assert ranked == [0]
    bm25 = PARTS["bm25"].document_ranking(
        [["cat", "sat"], ["dog", "ran"]], ["cat"]
    )
    assert bm25["ranking"][0] == 0
    assert spearman([1, 2, 3], [1, 2, 3]) > 0.999
    assert spearman([1, 2, 3], [3, 2, 1]) < -0.999

    # Model forward/backward and one optimizer step.
    config = {
        "vocab_size": 100,
        "max_len": 16,
        "d_model": 32,
        "n_heads": 2,
        "ff": 64,
        "layers": 1,
        "dropout": 0.0,
    }
    model = SentenceEncoder(config)
    ids1 = torch.randint(2, 100, (4, 8))
    ids2 = torch.randint(2, 100, (4, 8))
    mask1 = torch.ones_like(ids1, dtype=torch.bool)
    mask2 = torch.ones_like(ids2, dtype=torch.bool)
    scores = torch.tensor([5.0, 4.2, 3.8, 1.0])
    emb_a = model(ids1, mask1)
    emb_b = model(ids2, mask2)
    loss = regression_loss(emb_a, emb_b, scores) + 0.5 * contrastive_loss(
        emb_a, emb_b, scores, 3.5, 0.05
    )
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    assert torch.isfinite(loss)
    assert abs(float((emb_a[0] * emb_a[0]).sum()) - 1.0) < 1e-4

    print(
        "PASS: 6 copied semantic/linear-algebra parts; word tokenizer, "
        "sentence pooling, cosine/ranking, BM25, Spearman, encoder, "
        "regression + contrastive losses, and optimizer all OK"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sentence-embedding semantic search trained on STS-B."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    down = sub.add_parser("download", help="download and extract STS-B")
    down.set_defaults(func=cmd_download)

    test = sub.add_parser("selftest", help="check copied parts and the semantic model")
    test.set_defaults(func=cmd_selftest)

    train = sub.add_parser("train", help="train the bi-encoder on STS-B")
    train.add_argument("--data-dir", default=str(DATA_DIR / "STS-B"))
    train.add_argument("--out", default=str(HERE / "runs" / "semantic"))
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--max-len", type=int, default=64)
    train.add_argument("--vocab-size", type=int, default=12000)
    train.add_argument("--d-model", type=int, default=192)
    train.add_argument("--n-heads", type=int, default=4)
    train.add_argument("--ff", type=int, default=384)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--positive-threshold", type=float, default=3.5)
    train.add_argument("--temperature", type=float, default=0.05)
    train.add_argument("--con-weight", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="evaluate Spearman and retrieval")
    ev.add_argument("--checkpoint-dir", default=str(HERE / "runs" / "semantic"))
    ev.add_argument("--data-dir", default=str(DATA_DIR / "STS-B"))
    ev.add_argument("--split", choices=["dev", "test"], default="dev")
    ev.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--max-queries", type=int, default=0)
    ev.add_argument("--bm25-queries", type=int, default=100)
    ev.add_argument("--bm25-candidates", type=int, default=200)
    ev.set_defaults(func=cmd_eval)

    search = sub.add_parser("search", help="rank a sentence corpus for a query")
    search.add_argument("--checkpoint-dir", default=str(HERE / "runs" / "semantic"))
    search.add_argument("--corpus", default=str(DATA_DIR / "stsb_sentences.txt"))
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    search.add_argument("--batch-size", type=int, default=64)
    search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
