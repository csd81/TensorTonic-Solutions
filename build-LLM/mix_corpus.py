#!/usr/bin/env python3
"""Build a mixed, deduplicated text corpus for the scaled LLM demo.

Sources:

    TinyStories (simple English stories)
    Tiny Shakespeare (public-domain plays)
    Project Gutenberg: Pride and Prejudice, Frankenstein

The copied CS336 l14 exact-deduplication solution is used to remove duplicate
documents after normalization.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "data" / "sources"
PARTS = HERE / "parts" / "cs336"

URLS = {
    "tinystories": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "pride": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "frank": "https://www.gutenberg.org/files/84/84-0.txt",
}


def load_dedup():
    path = PARTS / "cs336-l14-stable-exact-deduplication.py"
    spec = importlib.util.spec_from_file_location("cs336_dedup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download(name: str) -> Path:
    SOURCES.mkdir(parents=True, exist_ok=True)
    target = SOURCES / f"{name}.txt"
    if target.exists() and target.stat().st_size > 1000:
        return target
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(URLS[name], target)
    return target


def strip_gutenberg(text: str) -> str:
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        newline = text.find("\n", start)
        text = text[newline + 1 :] if newline != -1 else text
    if end != -1:
        text = text[:end]
    return text


def split_tinystories(text: str) -> list[str]:
    return [
        piece.strip()
        for piece in text.split("<|endoftext|>")
        if len(piece.strip()) > 80
    ]


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for block in text.split("\n\n"):
        block = block.strip()
        if len(block) > 80:
            paragraphs.append(block)
    return paragraphs


def take_documents(documents: list[str], max_chars: int) -> list[str]:
    kept = []
    total = 0
    for doc in documents:
        if total + len(doc) > max_chars:
            break
        kept.append(doc)
        total += len(doc)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "data" / "mixed.txt"))
    parser.add_argument("--tinystories-chars", type=int, default=12_000_000)
    parser.add_argument("--shakespeare-chars", type=int, default=1_500_000)
    parser.add_argument("--gutenberg-chars", type=int, default=3_000_000)
    parser.add_argument("--max-chars", type=int, default=20_000_000)
    args = parser.parse_args()

    start = time.time()
    source_docs = {}
    for name in ("tinystories", "shakespeare", "pride", "frank"):
        path = download(name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        if name in {"pride", "frank"}:
            text = strip_gutenberg(text)
        if name == "tinystories":
            docs = split_tinystories(text)
            docs = take_documents(docs, args.tinystories_chars)
        elif name == "shakespeare":
            docs = split_paragraphs(text)
            docs = take_documents(docs, args.shakespeare_chars)
        else:
            docs = split_paragraphs(text)
            docs = take_documents(docs, args.gutenberg_chars)
        source_docs[name] = docs
        print(f"{name:12s} {len(docs):6d} docs, {sum(map(len, docs)):9,d} chars")

    documents = []
    texts_by_id = {}
    for name, docs in source_docs.items():
        for index, text in enumerate(docs):
            doc_id = f"{name}_{index}"
            documents.append({"id": doc_id, "text": text})
            texts_by_id[doc_id] = text

    print(f"deduplicating {len(documents):,} documents ...")
    dedup = load_dedup()
    result = dedup.stable_exact_deduplication(documents)
    retained = [texts_by_id[doc_id] for doc_id in result["retained_ids"]]

    mixed = "\n\n".join(retained)
    if len(mixed) > args.max_chars:
        mixed = mixed[: args.max_chars]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mixed, encoding="utf-8")

    manifest = {
        "sources": {
            name: {
                "documents": len(docs),
                "characters": sum(map(len, docs)),
            }
            for name, docs in source_docs.items()
        },
        "documents_before_dedup": len(documents),
        "documents_retained": len(retained),
        "duplicates_removed": len(documents) - len(retained),
        "characters": len(mixed),
        "elapsed_seconds": time.time() - start,
    }
    (out.parent / "mixed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
