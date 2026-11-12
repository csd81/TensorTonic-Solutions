#!/usr/bin/env python3
"""Run the reasoning-from-scratch pipeline with smaller base models.

Supported checkpoints:

    smollm2-360m            Llama layout, 0.36B, no QKV/MLP biases, eps=1e-5
    smollm2-135m            same layout, 0.135B
    qwen2.5-0.5b-instruct   Qwen2 layout, 0.49B, QKV biases, chat template

SmolLM2 matches the from-scratch ``Qwen3Model`` almost exactly (no QKV/MLP
biases, GQA, SwiGLU, RMSNorm, split-halves RoPE); only the RMSNorm epsilon,
tokenizer, and dimensions differ. Qwen2.5 needs QKV-bias support, which this
script adds before loading the HF checkpoint.

The script evaluates on GSM8K or MATH-500 and grades with the ch03 SymPy
verifier.

Examples:

    /home/d/.venvs/torch126/bin/python small_reasoner.py eval \
        --model smollm2-360m --task gsm8k --limit 10 --device cuda

    /home/d/.venvs/torch126/bin/python small_reasoner.py eval \
        --model qwen2.5-0.5b-instruct --task gsm8k --limit 50 \
        --device cuda --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RUN_DIR = HERE / "runs" / "small_reasoner"
REASONING_REPO = HERE / "reasoning"
if str(REASONING_REPO) not in sys.path:
    sys.path.insert(0, str(REASONING_REPO))

from reasoning_from_scratch.ch02 import (  # noqa: E402
    generate_text_basic_stream_cache,
)
from reasoning_from_scratch.ch03 import (  # noqa: E402
    extract_final_candidate,
    grade_answer,
    load_math500_test,
    render_prompt,
)
from reasoning_from_scratch.ch04 import (  # noqa: E402
    generate_text_top_p_stream_cache,
)
from reasoning_from_scratch.qwen3 import (  # noqa: E402
    Qwen3Model,
    Qwen3Tokenizer,
    load_hf_weights_into_qwen,
)

DEFAULT_SYSTEM = (
    "You are a helpful assistant running locally on the user's computer. "
    "You have no internet access, no web search, and no external tools. "
    "Never claim to search, browse, or retrieve current information. "
    "If the conversation contains sources retrieved by the local harness, use "
    "them. If a source is a list, table, or timeline, summarize the events it "
    "describes; never treat day numbers, dates, or list indices as events. "
    "If asked for news or real-time data and no sources were provided, say you "
    "cannot access it. If you are unsure, say so instead of inventing details."
)

SEARCH_SYSTEM = (
    "You are a helpful assistant. The local harness retrieved the sources in "
    "the user's message. Answer the user's question directly in 1-3 sentences "
    "using only those sources. Put citations inline like [1]. Never repeat the "
    "source list, URLs, or titles. If a source is a list, table, or timeline, "
    "summarize the events it describes; never treat day numbers, dates, or list "
    "indices as events. Do not claim to browse or search the web yourself. If "
    "the sources do not contain the answer, say you don't know."
)

MODELS = {
    "smollm2-360m": {
        "repo_id": "HuggingFaceTB/SmolLM2-360M",
        "rms_norm_eps": 1e-5,
        "cfg": {
            "vocab_size": 49_152,
            "context_length": 8_192,
            "emb_dim": 960,
            "n_heads": 15,
            "n_layers": 32,
            "hidden_dim": 2_560,
            "head_dim": 64,
            "qk_norm": False,
            "n_kv_groups": 5,
            "rope_base": 100_000.0,
            "dtype": torch.bfloat16,
        },
    },
    "smollm2-135m": {
        "repo_id": "HuggingFaceTB/SmolLM2-135M",
        "rms_norm_eps": 1e-5,
        "cfg": {
            "vocab_size": 49_152,
            "context_length": 8_192,
            "emb_dim": 576,
            "n_heads": 9,
            "n_layers": 30,
            "hidden_dim": 1_536,
            "head_dim": 64,
            "qk_norm": False,
            "n_kv_groups": 3,
            "rope_base": 100_000.0,
            "dtype": torch.bfloat16,
        },
    },
    "qwen2.5-0.5b-instruct": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "rms_norm_eps": 1e-6,
        "qkv_bias": True,
        "chat": True,
        "cfg": {
            "vocab_size": 151_936,
            "context_length": 32_768,
            "emb_dim": 896,
            "n_heads": 14,
            "n_layers": 24,
            "hidden_dim": 4_864,
            "head_dim": 64,
            "qk_norm": False,
            "n_kv_groups": 2,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        },
    },
}


class HFTokenizerAdapter:
    """Minimal tokenizer interface over a Hugging Face tokenizer."""

    def __init__(self, repo_id: str):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(repo_id)
        self.eos_token_id = self._tok.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def chat(self, user_message: str, system_message: str | None = None) -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat_messages(messages)

    def chat_messages(self, messages: list[dict]) -> str:
        return self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def add_qkv_bias(model, cfg) -> None:
    """Qwen2.5 has QKV biases; the Qwen3 blocks are built with bias=False."""
    for block in model.trf_blocks:
        att = block.att
        att.W_query = torch.nn.Linear(
            att.W_query.in_features,
            att.W_query.out_features,
            bias=True,
            dtype=cfg["dtype"],
        )
        att.W_key = torch.nn.Linear(
            att.W_key.in_features,
            att.W_key.out_features,
            bias=True,
            dtype=cfg["dtype"],
        )
        att.W_value = torch.nn.Linear(
            att.W_value.in_features,
            att.W_value.out_features,
            bias=True,
            dtype=cfg["dtype"],
        )


@torch.no_grad()
def load_qkv_biases(model, params, n_layers: int) -> None:
    for layer in range(n_layers):
        att = model.trf_blocks[layer].att
        prefix = f"model.layers.{layer}.self_attn"
        att.W_query.bias.copy_(params[f"{prefix}.q_proj.bias"])
        att.W_key.bias.copy_(params[f"{prefix}.k_proj.bias"])
        att.W_value.bias.copy_(params[f"{prefix}.v_proj.bias"])


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo search via the keyless ``ddgs`` package."""
    try:
        from ddgs import DDGS
    except ImportError:
        print("(ddgs is missing; install it with: pip install ddgs)")
        return []
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        print(f"(web search failed: {type(exc).__name__}: {exc})")
        return []
    results = []
    for row in rows:
        title = str(row.get("title") or "").strip()
        url = str(row.get("href") or row.get("url") or "").strip()
        snippet = str(row.get("body") or row.get("snippet") or "").strip()
        if title and url:
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": " ".join(snippet.split())[:500],
                    "source": "web",
                }
            )
    return results


def search_wikipedia(query: str, max_results: int = 3) -> list[dict]:
    """Wikipedia search + intro extracts via the public MediaWiki API."""
    headers = {"User-Agent": "small-reasoner/1.0 (local demo)"}
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results,
                "utf8": 1,
            },
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
    except Exception as exc:
        print(f"(wikipedia search failed: {type(exc).__name__}: {exc})")
        return []

    results = []
    for hit in hits:
        title = str(hit.get("title") or "").strip()
        if not title:
            continue
        snippet = _strip_html(str(hit.get("snippet") or ""))
        extract = ""
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": 1,
                    "explaintext": 1,
                    "titles": title,
                    "format": "json",
                    "redirects": 1,
                },
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = str(page.get("extract") or "").strip()
                break
        except Exception:
            extract = ""
        body = extract or snippet
        results.append(
            {
                "title": title,
                "url": "https://en.wikipedia.org/wiki/"
                + quote(title.replace(" ", "_")),
                "snippet": " ".join(body.split())[:800],
                "source": "wikipedia",
            }
        )
    return results


def format_sources(results: list[dict], limit: int = 500) -> str:
    lines = []
    for index, result in enumerate(results, 1):
        snippet = result["snippet"][:limit]
        lines.append(f"[{index}] {result['title']} ({result['source']})")
        lines.append(f"    URL: {result['url']}")
        if snippet:
            lines.append(f"    {snippet}")
    return "\n".join(lines)


def load_model(name: str, device: str, tokenizer_only: bool = False):
    spec = MODELS[name]
    snapshot = Path(
        snapshot_download(
            spec["repo_id"],
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "generation_config.json",
            ],
        )
    )
    print(f"snapshot: {snapshot}")

    if spec.get("chat"):
        tokenizer = HFTokenizerAdapter(spec["repo_id"])
    else:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer_path = RUN_DIR / f"{name}-tokenizer-base.json"
        if not tokenizer_path.exists():
            shutil.copyfile(snapshot / "tokenizer.json", tokenizer_path)
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    if tokenizer_only:
        return None, tokenizer

    model = Qwen3Model(spec["cfg"])
    if spec.get("qkv_bias"):
        add_qkv_bias(model, spec["cfg"])
    weights = load_file(str(snapshot / "model.safetensors"))
    load_hf_weights_into_qwen(
        model,
        {"n_layers": spec["cfg"]["n_layers"], "hidden_dim": spec["cfg"]["hidden_dim"]},
        weights,
    )
    if spec.get("qkv_bias"):
        load_qkv_biases(model, weights, spec["cfg"]["n_layers"])
    # The from-scratch Qwen3 blocks hardcode 1e-6; SmolLM2 was trained with 1e-5.
    for block in model.trf_blocks:
        block.norm1.eps = spec["rms_norm_eps"]
        block.norm2.eps = spec["rms_norm_eps"]
    model.final_norm.eps = spec["rms_norm_eps"]
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {name}: {n_params:,} parameters on {device}")
    return model, tokenizer


def gsm8k_examples(limit: int) -> list[dict]:
    path = DATA_DIR / "reason" / "gsm8k_test.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        match = re.search(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)", item["answer"])
        if not match:
            continue
        rows.append(
            {
                "problem": item["question"],
                "answer": match.group(1).replace("$", "").replace(",", ""),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def task_examples(task: str, limit: int) -> list[dict]:
    if task == "gsm8k":
        return gsm8k_examples(limit)
    return load_math500_test()[:limit]


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int]:
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    generated = []
    if temperature and temperature > 0:
        generator = generate_text_top_p_stream_cache(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        generator = generate_text_basic_stream_cache(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    for token in generator:
        generated.append(token.squeeze(0).item())
    return tokenizer.decode(generated), len(generated)


def cmd_download(args) -> None:
    load_model(args.model, args.device, tokenizer_only=False)


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int, float]:
    """Stream a reply to the terminal and return (text, token_count, seconds)."""
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    if temperature > 0:
        generator = generate_text_top_p_stream_cache(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        generator = generate_text_basic_stream_cache(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    print("bot> ", end="", flush=True)
    generated_ids: list[int] = []
    printed = ""
    start = time.time()
    for token in generator:
        generated_ids.append(token.squeeze(0).item())
        text = tokenizer.decode(generated_ids)
        common = 0
        for a, b in zip(printed, text):
            if a != b:
                break
            common += 1
        print(text[common:], end="", flush=True)
        printed = text
    print()
    elapsed = time.time() - start
    return tokenizer.decode(generated_ids).strip(), len(generated_ids), elapsed


def cmd_chat(args) -> None:
    """Interactive multi-turn chat REPL."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use the torch126 venv or --device cpu")
    spec = MODELS[args.model]
    if not spec.get("chat"):
        raise SystemExit(
            "chat requires a model with a chat template; "
            "use --model qwen2.5-0.5b-instruct"
        )

    model, tokenizer = load_model(args.model, device)
    context_length = spec["cfg"]["context_length"]
    system = args.system or DEFAULT_SYSTEM
    history: list[dict] = [{"role": "system", "content": system}]
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    top_p = args.top_p

    print(
        "Chat ready. Commands: /search <query>, /reset, /system <text>, "
        "/temp <value>, /tokens <n>, /status, /exit"
    )
    print(f"system: {system}")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            print("bye")
            break
        if user == "/help":
            print(
                "Commands: /search <query>, /reset, /system <text>, "
                "/temp <value>, /tokens <n>, /status, /exit"
            )
            continue
        if user == "/reset":
            history = [{"role": "system", "content": system}]
            print("(history reset)")
            continue
        if user == "/system" or user.startswith("/system "):
            new_system = user[len("/system"):].strip()
            if new_system:
                system = new_system
                history = [{"role": "system", "content": system}]
                print("(system set; history reset)")
            else:
                print(f"system: {system}")
            continue
        if user == "/temp" or user.startswith("/temp "):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"(temperature = {temperature})")
                except ValueError:
                    print("usage: /temp 0.7")
            else:
                print(f"temperature: {temperature}")
            continue
        if user == "/tokens" or user.startswith("/tokens "):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    max_new_tokens = int(parts[1])
                    print(f"(max new tokens = {max_new_tokens})")
                except ValueError:
                    print("usage: /tokens 1024")
            else:
                print(f"max new tokens: {max_new_tokens}")
            continue
        if user == "/status":
            probe = tokenizer.chat_messages(history)
            used = len(tokenizer.encode(probe))
            budget = context_length - max_new_tokens
            turns = max(0, len(history) - 1) // 2
            print(
                f"context: {used}/{budget} prompt tokens, "
                f"{max(0, budget - used)} left; turns={turns}; "
                f"max_new_tokens={max_new_tokens}"
            )
            continue

        if user == "/search" or user.startswith("/search "):
            query = user[len("/search"):].strip()
            if not query:
                print("usage: /search <query>")
                continue
            print(f"(searching web + wikipedia for: {query})")
            results = search_web(query, max_results=5) + search_wikipedia(
                query, max_results=3
            )
            seen = set()
            unique = []
            for result in results:
                if result["url"] in seen:
                    continue
                seen.add(result["url"])
                unique.append(result)
            if not unique:
                print("(no search results; no answer generated)")
                continue

            sources = format_sources(unique)
            print("sources:")
            print(sources)
            search_user = (
                f"Question: {query}\n\n"
                f"Sources:\n{sources}\n\n"
                "Answer the question directly in 1-3 sentences with inline "
                "citations. Do not repeat the source list or URLs."
            )
            messages = [{"role": "system", "content": SEARCH_SYSTEM}]
            messages.extend(history[1:])
            messages.append({"role": "user", "content": search_user})
            prompt = tokenizer.chat_messages(messages)
            prompt_ids = tokenizer.encode(prompt)
            budget = context_length - max_new_tokens
            while len(prompt_ids) > budget and len(messages) > 2:
                if len(messages) > 3:
                    del messages[1:3]
                else:
                    messages.pop(1)
                prompt = tokenizer.chat_messages(messages)
                prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) > budget:
                print(
                    f"(search context too long: {len(prompt_ids)} tokens "
                    f"> {budget})"
                )
                continue

            response, n_tokens, elapsed = stream_generate(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens,
                temperature,
                top_p,
            )
            print(
                f"[{n_tokens} tokens, "
                f"{n_tokens / elapsed if elapsed else 0:.1f} tok/s]"
            )
            if response:
                history.append({"role": "user", "content": search_user})
                history.append({"role": "assistant", "content": response})
            if n_tokens >= max_new_tokens:
                print(
                    f"(hit the {max_new_tokens}-token cap; use "
                    f"/tokens {max_new_tokens * 2} for a longer reply)"
                )
            print("(model summary from the sources above; verify before relying on it)")
            continue

        if re.search(r"\b(search|browse|look\s+up)\b", user, re.I) and re.search(
            r"\b(web|internet|online|news)\b", user, re.I
        ):
            print(
                "(note: this model has no web access; "
                "use /search <query> to retrieve web results)"
            )

        history.append({"role": "user", "content": user})
        prompt = tokenizer.chat_messages(history)
        prompt_ids = tokenizer.encode(prompt)
        budget = context_length - max_new_tokens
        while len(prompt_ids) > budget and len(history) > 2:
            if len(history) > 3:
                del history[1:3]
            else:
                history.pop(1)
            prompt = tokenizer.chat_messages(history)
            prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) > budget:
            print(f"(input too long: {len(prompt_ids)} tokens > {budget})")
            history.pop()
            continue

        response, n_tokens, elapsed = stream_generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens,
            temperature,
            top_p,
        )
        if response:
            history.append({"role": "assistant", "content": response})
        print(
            f"[{n_tokens} tokens, "
            f"{n_tokens / elapsed if elapsed else 0:.1f} tok/s]"
        )
        if n_tokens >= max_new_tokens:
            print(
                f"(hit the {max_new_tokens}-token cap; use "
                f"/tokens {max_new_tokens * 2} for a longer reply)"
            )


def cmd_ask(args) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use the torch126 venv or --device cpu")
    spec = MODELS[args.model]
    model, tokenizer = load_model(args.model, device)

    if spec.get("chat"):
        user_message = args.question
        if args.cot:
            user_message += "\n\nExplain step by step."
        prompt = tokenizer.chat(
            user_message, system_message=args.system or DEFAULT_SYSTEM
        )
    else:
        prompt = render_prompt(args.question)
        if args.cot:
            prompt += "\n\nExplain step by step."

    output, n_tokens = generate(
        model,
        tokenizer,
        prompt,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    print(output)
    print(f"\n[{n_tokens} generated tokens]")


def self_consistency_vote(outputs: list[str]) -> tuple[str | None, list[str]]:
    """Majority-vote the extracted answers; ties go to the first occurrence."""
    candidates = [
        extract_final_candidate(output, fallback="number_then_full")
        for output in outputs
    ]
    counts = Counter(candidate for candidate in candidates if candidate)
    if not counts:
        return None, candidates
    best_count = max(counts.values())
    for candidate in candidates:
        if candidate and counts[candidate] == best_count:
            return candidate, candidates
    return None, candidates


def cmd_eval(args) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use the torch126 venv or --device cpu")

    examples = task_examples(args.task, args.limit)
    if not examples:
        raise SystemExit(f"no examples found for task {args.task}")
    spec = MODELS[args.model]
    model, tokenizer = load_model(args.model, device)

    num_samples = max(1, args.num_samples)
    temperature = args.temperature
    if num_samples > 1 and temperature <= 0:
        temperature = 0.7
        print(
            "(num-samples > 1 with greedy decoding would repeat; "
            "using temperature=0.7)"
        )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"-sc{num_samples}" if num_samples > 1 else ""
    out_path = RUN_DIR / f"{args.model}-{args.task}-{args.limit}{suffix}.jsonl"
    correct = 0
    total_tokens = 0
    start = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for index, example in enumerate(examples, 1):
            if spec.get("chat"):
                user_message = example["problem"]
                if args.cot:
                    user_message += "\n\nExplain step by step."
                prompt = tokenizer.chat(
                    user_message,
                    system_message=(
                        "You are a helpful math assistant. Solve the problem "
                        "step by step and write the final answer in \\boxed{}."
                    ),
                )
            else:
                prompt = render_prompt(example["problem"])
                if args.cot:
                    prompt += "\n\nExplain step by step."
            outputs = []
            n_tokens = 0
            for _ in range(num_samples):
                output, sample_tokens = generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    args.max_new_tokens,
                    temperature,
                    args.top_p,
                )
                outputs.append(output)
                n_tokens += sample_tokens

            if num_samples > 1:
                extracted, candidates = self_consistency_vote(outputs)
            else:
                extracted = extract_final_candidate(
                    outputs[0], fallback="number_then_full"
                )
                candidates = [extracted]

            if extracted is not None:
                winner_index = next(
                    (i for i, candidate in enumerate(candidates)
                     if candidate == extracted),
                    0,
                )
            else:
                winner_index = 0
            output = outputs[winner_index]

            total_tokens += n_tokens
            ok = bool(extracted) and grade_answer(extracted, example["answer"])
            correct += int(ok)
            record = {
                "index": index,
                "problem": example["problem"],
                "gold": example["answer"],
                "output": output,
                "outputs": outputs,
                "candidates": candidates,
                "extracted": extracted,
                "correct": ok,
                "tokens": n_tokens,
                "num_samples": num_samples,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            elapsed = time.time() - start
            print(
                f"[{index}/{len(examples)}] correct={correct} "
                f"acc={correct / index:.3f} "
                f"tokens={n_tokens} samples={num_samples} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    accuracy = correct / len(examples)
    elapsed = time.time() - start
    print(
        f"\n{args.model} on {args.task} ({num_samples} sample(s)): "
        f"{correct}/{len(examples)} = {accuracy:.4f} in {elapsed:.1f}s "
        f"({total_tokens / elapsed:.1f} tok/s)"
    )
    print(f"results: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download")
    download.add_argument("--model", choices=MODELS, default="smollm2-360m")
    download.add_argument("--device", default="cpu")
    download.set_defaults(func=cmd_download)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--model", choices=MODELS, default="smollm2-360m")
    evaluate.add_argument("--task", choices=["gsm8k", "math500"], default="gsm8k")
    evaluate.add_argument("--limit", type=int, default=10)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--max-new-tokens", type=int, default=256)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--top-p", type=float, default=0.9)
    evaluate.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Self-consistency samples per question (1 disables).",
    )
    evaluate.add_argument("--cot", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    ask = sub.add_parser("ask")
    ask.add_argument("--model", choices=MODELS, default="qwen2.5-0.5b-instruct")
    ask.add_argument("--question", required=True)
    ask.add_argument("--system", default=None)
    ask.add_argument("--device", default="cuda")
    ask.add_argument("--max-new-tokens", type=int, default=256)
    ask.add_argument("--temperature", type=float, default=0.0)
    ask.add_argument("--top-p", type=float, default=0.9)
    ask.add_argument("--cot", action="store_true")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat")
    chat.add_argument("--model", choices=MODELS, default="qwen2.5-0.5b-instruct")
    chat.add_argument("--system", default=None)
    chat.add_argument("--device", default="cuda")
    chat.add_argument("--max-new-tokens", type=int, default=1024)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
