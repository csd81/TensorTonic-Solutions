#!/usr/bin/env python3
"""Question answering with a pretrained SQuAD-fine-tuned T5-small.

Reader: ``mrm8488/t5-small-finetuned-squadv2`` (T5-small fine-tuned on SQuAD
v2).  Retrieval: the copied BM25 solution from
``cracking-nlp/nlp-document-ranking-bm25.py`` over SQuAD dev passages plus a
small set of hand-written fact passages.

Quick start:

    python qa_system.py download
    python qa_system.py selftest --device cuda
    python qa_system.py ask --device cuda --question "Who wrote Frankenstein?"
    python qa_system.py eval --device cuda --limit 200 --retrieval-limit 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import string
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "qa"
MODEL_ID = "mrm8488/t5-small-finetuned-squadv2"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
BM25_PATH = HERE / "parts" / "semantic" / "nlp-document-ranking-bm25.py"


def load_bm25():
    spec = importlib.util.spec_from_file_location("bm25", BM25_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BM25 = load_bm25()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "dev-v1.1.json"
    if not target.exists():
        print(f"downloading {SQUAD_URL} ...")
        urllib.request.urlretrieve(SQUAD_URL, target)
    print(f"SQuAD dev: {target}")
    build_corpus()
    return target


def load_squad_examples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                examples.append(
                    {
                        "title": article["title"],
                        "question": qa["question"],
                        "context": paragraph["context"],
                        "answers": [a["text"] for a in qa["answers"]],
                        "id": qa["id"],
                    }
                )
    return examples


def build_corpus() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus = []
    seen = set()
    squad_path = DATA_DIR / "dev-v1.1.json"
    if squad_path.exists():
        data = json.loads(squad_path.read_text(encoding="utf-8"))
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if context not in seen:
                    seen.add(context)
                    corpus.append(
                        {"title": article["title"], "context": context}
                    )
    extra_path = DATA_DIR / "extra_passages.json"
    if extra_path.exists():
        for passage in json.loads(extra_path.read_text(encoding="utf-8")):
            if passage["context"] not in seen:
                seen.add(passage["context"])
                corpus.append(passage)
    (DATA_DIR / "corpus.json").write_text(
        json.dumps(corpus, indent=2), encoding="utf-8"
    )
    print(f"retrieval corpus: {len(corpus)} passages")
    return corpus


def load_corpus() -> list[dict]:
    path = DATA_DIR / "corpus.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_corpus()


# ---------------------------------------------------------------------------
# Reader and retrieval
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_model(model_id: str, device: str):
    print(f"loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def answer_batch(
    questions: list[str],
    contexts: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 32,
    num_beams: int = 4,
) -> list[str]:
    texts = [
        f"question: {question} context: {context}"
        for question, context in zip(questions, contexts)
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_length=max_length,
            num_beams=num_beams,
        )
    return [
        text.strip()
        for text in tokenizer.batch_decode(output, skip_special_tokens=True)
    ]


def retrieve(
    question: str, corpus: list[dict], top_k: int
) -> list[tuple[int, dict]]:
    query = tokenize(question)
    documents = [tokenize(passage["context"]) for passage in corpus]
    result = BM25.document_ranking(documents, query)
    scores = result["scores"]
    query_terms = set(query)
    # Favour passages whose title contains the question's entity: "capital of
    # England" should prefer the England article over a broader UK article.
    adjusted = []
    for index, passage in enumerate(corpus):
        title_terms = set(tokenize(passage["title"]))
        title_overlap = len(query_terms & title_terms)
        adjusted.append((scores[index] + 1.0 * title_overlap, index))
    adjusted.sort(key=lambda item: (-item[0], item[1]))
    return [(index, corpus[index]) for _score, index in adjusted[:top_k]]


def clean_answer(text: str):
    text = text.strip()
    if text.lower() in {"", "none", "no answer"}:
        return None
    return text


def ask(question: str, model, tokenizer, device: str, top_k: int = 3) -> list[dict]:
    corpus = load_corpus()
    hits = retrieve(question, corpus, top_k)
    contexts = [passage["context"] for _index, passage in hits]
    answers = answer_batch(
        [question] * len(contexts), contexts, model, tokenizer, device
    )
    results = []
    for (rank, (index, passage)), answer in zip(enumerate(hits, 1), answers):
        results.append(
            {
                "rank": rank,
                "index": index,
                "title": passage["title"],
                "context": passage["context"],
                "answer": clean_answer(answer),
                "raw_answer": answer,
            }
        )
    print(f"Q: {question}")
    for result in results:
        snippet = " ".join(result["context"].split())[:220]
        shown = result["answer"] or "(no answer in this passage)"
        print(f"  [{result['rank']}] {result['title']}")
        print(f"      answer: {shown}")
        print(f"      passage: {snippet}...")
    best = next((result for result in results if result["answer"]), None)
    if best is not None:
        print(f"  best answer: {best['answer']}  (from {best['title']})")
    else:
        print(
            "  no answer found: add the fact to data/qa/extra_passages.json "
            "or use a larger retrieval corpus"
        )
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> int:
    normalized = normalize_answer(prediction)
    return int(any(normalized == normalize_answer(g) for g in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        common = {}
        for token in pred_tokens:
            common[token] = common.get(token, 0) + 1
        overlap = 0
        for token in gold_tokens:
            if common.get(token, 0) > 0:
                common[token] -= 1
                overlap += 1
        if not pred_tokens or not gold_tokens:
            score = float(pred_tokens == gold_tokens)
        elif overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_download(args) -> None:
    download()


def cmd_selftest(args) -> None:
    device = args.device
    model, tokenizer = load_model(args.model, device)
    context = "The capital of France is Paris. Paris is known for the Eiffel Tower."
    prediction = answer_batch(
        ["What is the capital of France?"],
        [context],
        model,
        tokenizer,
        device,
    )[0]
    print(f"reader: {prediction!r}")
    assert "paris" in normalize_answer(prediction), prediction

    corpus = [
        {"title": "France", "context": context},
        {"title": "Japan", "context": "The capital of Japan is Tokyo."},
    ]
    hits = retrieve("What is the capital of Japan?", corpus, 1)
    assert hits and hits[0][1]["title"] == "Japan", hits
    print("PASS: T5 reader, copied BM25 retrieval, and answer normalization OK")


def cmd_ask(args) -> None:
    model, tokenizer = load_model(args.model, args.device)
    if args.context:
        answer = answer_batch(
            [args.question], [args.context], model, tokenizer, args.device
        )[0]
        print(f"Q: {args.question}")
        print(f"A: {clean_answer(answer) or '(no answer in this passage)'}")
    else:
        ask(args.question, model, tokenizer, args.device, args.top_k)


def cmd_eval(args) -> None:
    data_path = DATA_DIR / "dev-v1.1.json"
    if not data_path.exists():
        download()
    examples = load_squad_examples(data_path)
    model, tokenizer = load_model(args.model, args.device)

    reader_examples = examples[: args.limit]
    questions = [ex["question"] for ex in reader_examples]
    contexts = [ex["context"] for ex in reader_examples]
    predictions = []
    for start in range(0, len(questions), args.batch_size):
        predictions.extend(
            answer_batch(
                questions[start : start + args.batch_size],
                contexts[start : start + args.batch_size],
                model,
                tokenizer,
                args.device,
            )
        )
    em = sum(
        exact_match(pred, ex["answers"])
        for pred, ex in zip(predictions, reader_examples)
    ) / len(reader_examples)
    f1 = sum(
        token_f1(pred, ex["answers"])
        for pred, ex in zip(predictions, reader_examples)
    ) / len(reader_examples)
    print(f"reader with gold context: EM {em:.4f}  F1 {f1:.4f}  "
          f"({len(reader_examples)} questions)")

    corpus = load_corpus()
    e2e_examples = examples[: args.retrieval_limit]
    e2e_em = 0
    recall_at_k = 0
    for ex in e2e_examples:
        hits = retrieve(ex["question"], corpus, args.top_k)
        contexts = [passage["context"] for _index, passage in hits]
        if ex["context"] in contexts:
            recall_at_k += 1
        prediction = answer_batch(
            [ex["question"]], [contexts[0]], model, tokenizer, args.device
        )[0]
        e2e_em += exact_match(prediction, ex["answers"])
    n = len(e2e_examples)
    print(
        f"retrieval+reader: recall@{args.top_k} {recall_at_k / n:.4f}  "
        f"end-to-end EM {e2e_em / n:.4f}  ({n} questions)"
    )

    results = {
        "model": args.model,
        "reader_em": em,
        "reader_f1": f1,
        "reader_questions": len(reader_examples),
        "retrieval_recall_at_k": recall_at_k / n,
        "end_to_end_em": e2e_em / n,
        "retrieval_questions": n,
        "top_k": args.top_k,
    }
    (DATA_DIR / "eval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"results written to {DATA_DIR / 'eval_results.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="T5-small SQuAD question answering with BM25 retrieval."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    down = sub.add_parser("download", help="download SQuAD dev and build corpus")
    down.set_defaults(func=cmd_download)

    test = sub.add_parser("selftest", help="test reader and retrieval")
    test.add_argument("--model", default=MODEL_ID)
    test.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    test.set_defaults(func=cmd_selftest)

    ask_parser = sub.add_parser("ask", help="answer a question")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--context", default=None)
    ask_parser.add_argument("--top-k", type=int, default=3)
    ask_parser.add_argument("--model", default=MODEL_ID)
    ask_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ask_parser.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", help="SQuAD exact match and retrieval evaluation")
    ev.add_argument("--model", default=MODEL_ID)
    ev.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ev.add_argument("--limit", type=int, default=200)
    ev.add_argument("--retrieval-limit", type=int, default=50)
    ev.add_argument("--top-k", type=int, default=3)
    ev.add_argument("--batch-size", type=int, default=16)
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is not available; use the torch126 venv or --device cpu"
        )
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
