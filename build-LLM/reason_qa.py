#!/usr/bin/env python3
"""Retrieval-augmented chain-of-thought QA with Flan-T5-base.

Components:

    Flan-T5-base (google/flan-t5-base, 250M) as the generative reasoner
    BM25 retrieval of similar solved exemplars (copied cracking-nlp part)
    chain-of-thought prompts
    a safe arithmetic calculator tool for CALC: expressions
    the ch03 SymPy grader from reasoning-from-scratch for answer checking

Tasks:

    GSM8K      grade-school math word problems, final numeric answer
    StrategyQA yes/no commonsense questions with facts

Quick start:

    python reason_qa.py download
    /home/d/.venvs/torch126/bin/python reason_qa.py selftest --device cuda
    /home/d/.venvs/torch126/bin/python reason_qa.py eval --device cuda \
        --task both --gsm8k-limit 30 --strategyqa-limit 100
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "reason"
BM25_PATH = HERE / "parts" / "semantic" / "nlp-document-ranking-bm25.py"
MODEL_ID = "google/flan-t5-base"

# Use the ch03 symbolic grader from the vendored reasoning-from-scratch repo.
# It handles LaTeX, fractions, decimals, and mathematical equivalence via SymPy.
REASONING_REPO = HERE / "reasoning"
if str(REASONING_REPO) not in sys.path:
    sys.path.insert(0, str(REASONING_REPO))
try:
    from reasoning_from_scratch.ch03 import (
        extract_final_candidate as extract_ch03_candidate,
        grade_answer as grade_with_sympy,
    )
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "reason_qa.py uses the ch03 grader from reasoning-from-scratch. "
        f"Expected the package under {REASONING_REPO}; install its requirements "
        "(including sympy) or restore that directory."
    ) from exc

URLS = {
    "gsm8k_train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "strategyqa_train": "https://huggingface.co/datasets/voidful/StrategyQA/resolve/main/strategyqa_train.json",
    "strategyqa_test": "https://huggingface.co/datasets/voidful/StrategyQA/resolve/main/strategyqa_test.json",
}


def load_bm25():
    spec = importlib.util.spec_from_file_location("bm25", BM25_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BM25 = load_bm25()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        suffix = ".jsonl" if name.startswith("gsm8k") else ".json"
        target = DATA_DIR / f"{name}{suffix}"
        if target.exists() and target.stat().st_size > 1000:
            print(f"have {target}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, target)
        print(f"  {target} ({target.stat().st_size:,} bytes)")


def load_gsm8k(path: Path) -> list[dict]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        examples.append(
            {"question": item["question"], "answer": item["answer"]}
        )
    return examples


def load_strategyqa(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for item in data:
        if "answer" not in item:
            continue
        facts = [str(fact) for fact in item.get("facts", []) if fact]
        examples.append(
            {
                "question": item["question"],
                "answer": bool(item["answer"]),
                "facts": facts,
            }
        )
    return examples


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
_ALLOWED = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def calculator(expression: str) -> float | None:
    expression = expression.replace("^", "**").replace(",", "")
    expression = expression.strip().rstrip("=.")
    if not expression or len(expression) > 200:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return None
    try:
        value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


CALC_RE = re.compile(r"(?:CALC:|<<)\s*([0-9+\-*/().%^,\s]+?)(?:>>|\n|=|$)", re.I)


def evaluate_calcs(text: str) -> list[tuple[str, float]]:
    results = []
    for match in CALC_RE.finditer(text):
        expression = match.group(1).strip()
        value = calculator(expression)
        if value is not None:
            results.append((expression, value))
    return results


def normalize_number(value: str) -> str:
    value = value.replace("$", "").replace(",", "").strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def extract_final_answer(text: str) -> str | None:
    # The prompt asks for the GSM8K marker; honor it first.
    match = re.search(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return normalize_number(match.group(1))
    match = re.search(
        r"(?:answer is|final answer is|answer:)\s*(-?\$?\d[\d,]*(?:\.\d+)?)",
        text,
        re.I,
    )
    if match:
        return normalize_number(match.group(1))
    # Fall back to the ch03 extractor: it understands \boxed{...}, nested
    # braces, and LaTeX fractions that the old last-number regex mangled
    # (e.g. \frac{3}{4} used to become "4").
    candidate = extract_ch03_candidate(text, fallback="number_only")
    if candidate:
        return normalize_number(candidate)
    return None


def answers_match(predicted: str | None, gold: str | None) -> bool:
    """Compare two answers with the ch03 SymPy grader.

    ``grade_answer`` normalizes LaTeX and checks mathematical equivalence, so
    ``18``, ``18.0``, and ``\\frac{36}{2}`` all match.
    """
    if predicted is None or gold is None:
        return False
    return bool(grade_with_sympy(predicted, gold))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, examples: list[dict], key: str = "question"):
        self.examples = examples
        self.documents = [tokenize(item[key]) for item in examples]

    def retrieve(self, question: str, k: int) -> list[dict]:
        if not self.examples or k <= 0:
            return []
        ranking = BM25.document_ranking(
            self.documents, tokenize(question)
        )["ranking"]
        return [self.examples[index] for index in ranking[:k]]


# ---------------------------------------------------------------------------
# Prompting and generation
# ---------------------------------------------------------------------------
def gsm8k_prompt(
    question: str, exemplars: list[dict], use_calculator: bool
) -> str:
    lines = [
        "Solve the math problem step by step.",
        "Show the reasoning, then write the final answer as #### <number>.",
    ]
    if use_calculator:
        lines.append(
            "For arithmetic you may write CALC: <expression>, for example "
            "CALC: 16 - 3 - 4."
        )
    lines.append("")
    for example in exemplars:
        lines.append(f"Q: {example['question']}")
        lines.append(f"A: {example['answer'].strip()}")
        lines.append("")
    lines.append(f"Q: {question}")
    lines.append("A: Let's think step by step.")
    return "\n".join(lines)


def strategyqa_prompt(
    question: str, facts: list[str], exemplars: list[dict]
) -> str:
    lines = [
        "Answer the yes/no question. Think step by step and end with "
        "'The answer is yes.' or 'The answer is no.'.",
        "",
    ]
    for example in exemplars:
        lines.append(f"Q: {example['question']}")
        if example.get("facts"):
            lines.append("Facts: " + " ".join(example["facts"][:3]))
        answer = "yes" if example["answer"] else "no"
        lines.append(
            f"A: Let's think step by step. The answer is {answer}."
        )
        lines.append("")
    lines.append(f"Q: {question}")
    if facts:
        lines.append("Facts: " + " ".join(facts[:6]))
    lines.append("A: Let's think step by step.")
    return "\n".join(lines)


def generate(
    prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def solve_gsm8k(
    question: str,
    retriever: Retriever,
    model,
    tokenizer,
    device: str,
    retrieved_k: int,
    max_new_tokens: int,
    num_beams: int,
    use_calculator: bool,
) -> dict:
    exemplars = retriever.retrieve(question, retrieved_k)
    prompt = gsm8k_prompt(question, exemplars, use_calculator)
    output = generate(
        prompt, model, tokenizer, device, max_new_tokens, num_beams
    )
    calcs = evaluate_calcs(output) if use_calculator else []
    final = extract_final_answer(output)
    if final is None and calcs:
        final = normalize_number(str(calcs[-1][1]))
    return {
        "question": question,
        "prompt": prompt,
        "output": output,
        "final": final,
        "calcs": calcs,
        "exemplars": [ex["question"] for ex in exemplars],
    }


def solve_strategyqa(
    question: str,
    facts: list[str],
    retriever: Retriever,
    model,
    tokenizer,
    device: str,
    retrieved_k: int,
    max_new_tokens: int,
    num_beams: int,
) -> dict:
    exemplars = retriever.retrieve(question, retrieved_k)
    prompt = strategyqa_prompt(question, facts, exemplars)
    output = generate(
        prompt, model, tokenizer, device, max_new_tokens, num_beams
    )
    lowered = output.lower()
    if re.search(r"\bthe answer is\s+yes\b", lowered):
        prediction = "yes"
    elif re.search(r"\bthe answer is\s+no\b", lowered):
        prediction = "no"
    elif re.search(r"\byes\b", lowered):
        prediction = "yes"
    elif re.search(r"\bno\b", lowered):
        prediction = "no"
    else:
        prediction = None
    return {
        "question": question,
        "prompt": prompt,
        "output": output,
        "prediction": prediction,
        "exemplars": [ex["question"] for ex in exemplars],
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def load_model(model_id: str, device: str):
    print(f"loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def cmd_download(args) -> None:
    download()


def cmd_selftest(args) -> None:
    assert calculator("16 - 3 - 4") == 9.0
    assert calculator("9 * 2") == 18.0
    assert calculator("2 ** 5") == 32.0
    assert calculator("__import__('os').system('echo hi')") is None
    assert extract_final_answer("... #### 18") == "18"
    assert extract_final_answer("The answer is 1,234.") == "1234"
    # The ch03 grader: LaTeX, fractions, and numeric equivalence.
    assert answers_match("18", "18")
    assert answers_match("18.0", "18")
    assert answers_match("0.5", "1/2")
    assert answers_match("\\frac{3}{4}", "0.75")
    assert not answers_match("18", "19")
    assert not answers_match(None, "18")
    assert answers_match(extract_final_answer("so \\boxed{\\frac{3}{4}}"), "3/4")

    train = [
        {"question": "Tom has 2 apples and buys 3 more. How many?", "answer": "5"},
        {"question": "A box has 4 red balls. How many red balls?", "answer": "4"},
    ]
    retriever = Retriever(train)
    hits = retriever.retrieve("How many apples does Tom have?", 1)
    assert hits and "apples" in hits[0]["question"]

    device = args.device
    model, tokenizer = load_model(args.model, device)
    result = solve_gsm8k(
        "What is 2 + 2?",
        Retriever([]),
        model,
        tokenizer,
        device,
        retrieved_k=0,
        max_new_tokens=64,
        num_beams=2,
        use_calculator=True,
    )
    print("model output:", result["output"][:300])
    assert result["final"] is not None
    print(
        "PASS: calculator, ch03 SymPy grader, BM25 retrieval, "
        "and Flan-T5 generation OK"
    )


def cmd_eval(args) -> None:
    download()
    device = args.device
    model, tokenizer = load_model(args.model, device)
    results = {"model": args.model, "tool": args.tool}

    if args.task in {"gsm8k", "both"}:
        train = load_gsm8k(DATA_DIR / "gsm8k_train.jsonl")
        test = load_gsm8k(DATA_DIR / "gsm8k_test.jsonl")[: args.gsm8k_limit]
        retriever = Retriever(train)
        correct = 0
        records = []
        for index, example in enumerate(test, 1):
            record = solve_gsm8k(
                example["question"],
                retriever,
                model,
                tokenizer,
                device,
                args.gsm8k_retrieved,
                args.gsm8k_max_new_tokens,
                args.num_beams,
                args.tool == "calculator",
            )
            gold = extract_final_answer(example["answer"])
            ok = answers_match(record["final"], gold)
            correct += int(ok)
            record["gold"] = gold
            record["correct"] = bool(ok)
            records.append(record)
            if index % 5 == 0 or index == len(test):
                print(
                    f"  GSM8K {index}/{len(test)}  accuracy {correct / index:.3f}"
                )
        accuracy = correct / max(1, len(test))
        results["gsm8k"] = {
            "questions": len(test),
            "accuracy": accuracy,
            "correct": correct,
            "retrieved_k": args.gsm8k_retrieved,
        }
        print(f"GSM8K exact match: {correct}/{len(test)} = {accuracy:.4f}")
        (DATA_DIR / "gsm8k_results.json").write_text(
            json.dumps({"summary": results["gsm8k"], "records": records}, indent=2),
            encoding="utf-8",
        )

    if args.task in {"strategyqa", "both"}:
        data = load_strategyqa(DATA_DIR / "strategyqa_train.json")
        if len(data) <= args.strategyqa_limit:
            raise ValueError("StrategyQA needs more training examples")
        held_out = data[-args.strategyqa_limit :]
        pool = data[: -args.strategyqa_limit]
        retriever = Retriever(pool)
        correct = 0
        records = []
        for index, example in enumerate(held_out, 1):
            record = solve_strategyqa(
                example["question"],
                example["facts"],
                retriever,
                model,
                tokenizer,
                device,
                args.strategyqa_retrieved,
                args.strategyqa_max_new_tokens,
                args.num_beams,
            )
            gold = "yes" if example["answer"] else "no"
            ok = record["prediction"] == gold
            correct += int(ok)
            record["gold"] = gold
            record["correct"] = bool(ok)
            records.append(record)
            if index % 10 == 0 or index == len(held_out):
                print(
                    f"  StrategyQA {index}/{len(held_out)}  accuracy "
                    f"{correct / index:.3f}"
                )
        accuracy = correct / max(1, len(held_out))
        results["strategyqa"] = {
            "questions": len(held_out),
            "accuracy": accuracy,
            "correct": correct,
            "retrieved_k": args.strategyqa_retrieved,
        }
        print(
            f"StrategyQA accuracy: {correct}/{len(held_out)} = {accuracy:.4f}"
        )
        (DATA_DIR / "strategyqa_results.json").write_text(
            json.dumps({"summary": results["strategyqa"], "records": records}, indent=2),
            encoding="utf-8",
        )

    (DATA_DIR / "reason_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"results written to {DATA_DIR / 'reason_results.json'}")


def cmd_ask(args) -> None:
    download()
    device = args.device
    model, tokenizer = load_model(args.model, device)
    if args.task == "gsm8k":
        train = load_gsm8k(DATA_DIR / "gsm8k_train.jsonl")
        retriever = Retriever(train)
        record = solve_gsm8k(
            args.question,
            retriever,
            model,
            tokenizer,
            device,
            args.gsm8k_retrieved,
            args.gsm8k_max_new_tokens,
            args.num_beams,
            args.tool == "calculator",
        )
        print(record["output"])
        print(f"\nfinal answer: {record['final']}")
        print(f"retrieved exemplars: {record['exemplars']}")
        print(f"calculator uses: {record['calcs']}")
    else:
        data = load_strategyqa(DATA_DIR / "strategyqa_train.json")
        retriever = Retriever(data)
        record = solve_strategyqa(
            args.question,
            [],
            retriever,
            model,
            tokenizer,
            device,
            args.strategyqa_retrieved,
            args.strategyqa_max_new_tokens,
            args.num_beams,
        )
        print(record["output"])
        print(f"\nprediction: {record['prediction']}")
        print(f"retrieved exemplars: {record['exemplars']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval-augmented CoT QA with Flan-T5-base."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    down = sub.add_parser("download")
    down.set_defaults(func=cmd_download)

    test = sub.add_parser("selftest")
    test.add_argument("--model", default=MODEL_ID)
    test.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    test.set_defaults(func=cmd_selftest)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--task", choices=["gsm8k", "strategyqa", "both"], default="both")
    evaluate.add_argument("--model", default=MODEL_ID)
    evaluate.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    evaluate.add_argument("--tool", choices=["calculator", "none"], default="calculator")
    evaluate.add_argument("--num-beams", type=int, default=4)
    evaluate.add_argument("--gsm8k-limit", type=int, default=30)
    evaluate.add_argument("--gsm8k-retrieved", type=int, default=2)
    evaluate.add_argument("--gsm8k-max-new-tokens", type=int, default=192)
    evaluate.add_argument("--strategyqa-limit", type=int, default=100)
    evaluate.add_argument("--strategyqa-retrieved", type=int, default=3)
    evaluate.add_argument("--strategyqa-max-new-tokens", type=int, default=64)
    evaluate.set_defaults(func=cmd_eval)

    ask = sub.add_parser("ask")
    ask.add_argument("--task", choices=["gsm8k", "strategyqa"], default="gsm8k")
    ask.add_argument("--question", required=True)
    ask.add_argument("--model", default=MODEL_ID)
    ask.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ask.add_argument("--tool", choices=["calculator", "none"], default="calculator")
    ask.add_argument("--num-beams", type=int, default=4)
    ask.add_argument("--gsm8k-retrieved", type=int, default=2)
    ask.add_argument("--gsm8k-max-new-tokens", type=int, default=192)
    ask.add_argument("--strategyqa-retrieved", type=int, default=3)
    ask.add_argument("--strategyqa-max-new-tokens", type=int, default=64)
    ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use the torch126 venv or --device cpu")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
