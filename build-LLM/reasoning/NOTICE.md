# Vendored subset of reasoning-from-scratch

Source: https://github.com/rasbt/reasoning-from-scratch
Book: *Build a Reasoning Model (From Scratch)*, Sebastian Raschka (Manning, 2025)
License: Apache License 2.0 — see `LICENSE`

Copied on 2026-11-12, unmodified, because these are the only modules used by
`build_LLM/small_reasoner.py` and `build_LLM/reason_qa.py`:

- `reasoning_from_scratch/__init__.py`
- `reasoning_from_scratch/utils.py`
- `reasoning_from_scratch/qwen3.py`
- `reasoning_from_scratch/ch02.py`
- `reasoning_from_scratch/ch03.py`
- `reasoning_from_scratch/ch04.py`

Not included: `ch05`–`ch08` (scoring, GRPO, distillation), the appendices,
tests, notebooks, chapter scripts, and `book.txt`. The book text is
copyrighted and is not part of this vendored subset.

The original `build_LLM/reasoning-from-scratch/` directory is left intact.
