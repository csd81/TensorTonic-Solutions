# Minimal reasoning-from-scratch subset

This directory contains only the modules from
[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) that
the local code actually imports. See `NOTICE.md` for provenance and `LICENSE`
for the Apache-2.0 license.

## Contents

- `qwen3.py` — Qwen3-style transformer, tokenizer wrapper, HF weight loader,
  KV cache
- `ch02.py` — KV-cached greedy streaming generation
- `ch03.py` — `\boxed{}` extraction, LaTeX normalization, SymPy grading,
  MATH-500 loader, prompt template
- `ch04.py` — temperature/top-p streaming generation
- `utils.py` — download helper used by `qwen3.py`

## Use

From `build_LLM`, the scripts already add this directory to `sys.path`:

```python
from reasoning_from_scratch.ch03 import extract_final_candidate, grade_answer
```

Or explicitly:

```python
import sys
sys.path.insert(0, "reasoning")
```

## Dependencies

`torch`, `tokenizers`, `sympy`, `requests` — see `requirements.txt`.
