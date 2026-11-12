# LLM, Semantic Search, SFT, QA, and Reasoning Systems

A build around the 113 files in
`study-plans/language-modeling-from-scratch/` (58 Python solutions, 54
write-ups, 1 CUDA kernel), plus useful pieces from other tracks and a
vendored subset of `reasoning-from-scratch`.

It now contains seven working systems:

1. **Tiny LLM** — a 640k-parameter decoder-only transformer on a bundled
   Shakespeare excerpt, CPU-friendly.
2. **Scaled LLM** — a 12.6M-parameter model trained on a 14 MB mixed,
   deduplicated corpus with a fast 8k HuggingFace BPE tokenizer, on the P2000.
3. **Semantic search** — a sentence bi-encoder trained on STS-B, evaluated with
   Spearman and retrieval against a BM25 baseline.
4. **Instruction tuning** — Alpaca + focused instruction data, trained with the
   copied CS336 assistant-only SFT mask.
5. **Question answering** — a pretrained SQuAD-fine-tuned T5-small reader plus
   copied BM25 retrieval.
6. **Retrieval-augmented reasoning QA** — Flan-T5-base with BM25 exemplar
   retrieval, chain-of-thought prompts, a calculator, and the ch03 SymPy
   grader.
7. **Small reasoner** — Qwen2.5-0.5B-Instruct served through the from-scratch
   Qwen3 model, with a chat REPL, web search, GSM8K evaluation, and
   self-consistency code.

## Quick start

```bash
cd /home/d/Desktop/TensorTonic-Solutions/build-LLM

# Tiny CPU model
python llm_system.py selftest
python llm_system.py train
python llm_system.py generate --prompt "The " --max-new-tokens 200

# Fast mixed-corpus LLM on the P2000
python mix_corpus.py --out data/mixed.txt
/home/d/.venvs/torch126/bin/python llm_system.py prepare \
    --corpus data/mixed.txt --tokenizer hf-bpe --vocab-size 8192 \
    --out runs/llm10m
/home/d/.venvs/torch126/bin/python llm_system.py train \
    --corpus data/mixed.txt --tokenizer hf-bpe --vocab-size 8192 \
    --out runs/llm10m --device cuda \
    --d-model 384 --n-layer 6 --n-head 6 --n-kv-head 2 --hidden 1024 \
    --block-size 128 --batch-size 4 --grad-accum 8 --steps 2000 \
    --lr 3e-4 --warmup 100 --eval-interval 250 --eval-iters 10 \
    --sample-tokens 40 --log-every 100 --no-plot

# Semantic search
python semantic_search.py download
/home/d/.venvs/torch126/bin/python semantic_search.py train --device cuda
/home/d/.venvs/torch126/bin/python semantic_search.py eval --device cuda
/home/d/.venvs/torch126/bin/python semantic_search.py search --device cuda \
    --query "A plane is taking off." --top-k 5

# Instruction tuning
/home/d/.venvs/torch126/bin/python sft_system.py train \
    --checkpoint-dir runs/llm10m --device cuda --out runs/sft
python llm_system.py generate --out runs/sft_tiny \
    --prompt "### Instruction:\nWhat is 2 + 2?\n### Response:\n"

# Question answering (pretrained T5-small SQuAD reader + BM25)
python qa_system.py download
/home/d/.venvs/torch126/bin/python qa_system.py selftest --device cuda
/home/d/.venvs/torch126/bin/python qa_system.py ask --device cuda \
    --question "Who wrote Frankenstein?"
/home/d/.venvs/torch126/bin/python qa_system.py eval --device cuda \
    --limit 200 --retrieval-limit 50

# Retrieval-augmented reasoning QA (Flan-T5-base + BM25 + ch03 grader)
python reason_qa.py download
/home/d/.venvs/torch126/bin/python reason_qa.py selftest --device cuda
/home/d/.venvs/torch126/bin/python reason_qa.py eval --device cuda \
    --task both --gsm8k-limit 30 --strategyqa-limit 100
/home/d/.venvs/torch126/bin/python reason_qa.py ask --device cuda \
    --task gsm8k --question "A train travels 60 mph for 2.5 hours. How far does it go?"

# Small reasoner (Qwen2.5-0.5B-Instruct through the from-scratch Qwen3 model)
/home/d/.venvs/torch126/bin/python small_reasoner.py eval \
    --model qwen2.5-0.5b-instruct --task gsm8k --limit 100 \
    --device cuda --max-new-tokens 512
/home/d/.venvs/torch126/bin/python small_reasoner.py chat \
    --model qwen2.5-0.5b-instruct --device cuda
/home/d/.venvs/torch126/bin/python small_reasoner.py ask \
    --model qwen2.5-0.5b-instruct --device cuda \
    --question "What is 2 + 2?"

# Self-consistency code is added but has not been run yet:
# small_reasoner.py eval ... --num-samples 3 --temperature 0.7
```

The P2000 is `sm_61`; the system Python's `torch 2.15.0+cu132` cannot run it.
Use `/home/d/.venvs/torch126/bin/python` (torch 2.10.0+cu126). NumPy,
`tokenizers`, `sympy`, `requests`, `ddgs`, `huggingface_hub`, and `safetensors`
were installed into that venv for these runs.

## 1. Tiny CPU LLM

640,128 parameters, 200 KB Tiny Shakespeare excerpt, 600 steps, CPU:

| Metric | Value |
|---|---|
| Loss | 5.9701 -> **3.3509** |
| Validation loss / perplexity | 5.9730 -> **3.5940** / **36.4** |
| Training time | about 2 m 38 s |
| Generation | 200 tokens in about 2.3 s |

The output captures Shakespearean dialogue formatting and rhythm, not meaning.

## 2. Scaled mixed-corpus LLM

`mix_corpus.py` combines TinyStories, Tiny Shakespeare, Pride and Prejudice,
and Frankenstein, then deduplicates with the copied
`cs336-l14-stable-exact-deduplication.py`.

| Item | Value |
|---|---|
| Corpus | 14,091,120 chars, 21,172 documents, 1 duplicate removed |
| Tokenizer | HuggingFace byte-level BPE, vocab **8,192**, 3.97 chars/token |
| Tokens | **3,545,230** (90/10 train/val split) |
| Model | 12,587,904 params, 6 layers, d_model 384, 6 heads, 2 KV heads, hidden 1024 |
| Training | 2,000 steps, batch 4 x grad-accum 8, **1 m 50 s** on the P2000 |
| Loss | 9.0444 -> **4.4060** |
| Validation loss | 9.0493 -> **6.73** (best 6.7288 at step 1,750) |

Generated sample (`"Once upon a time"`, temperature 0.8, top-k 40):

```
Once upon a time, there was a little boy named Tim. The bird was very happy
and would her friend.
One day, a girl named Tim to play. The bird had an friends and the girl with
the new cat. Tim was so very happy, but he was happy. It was so happy.
...
```

Two thousand steps is a demo, not a fully trained model. It learns simple
story-like English but is far from a general LLM.

## 3. Semantic search (STS-B)

A 2.95M-parameter word-level Transformer bi-encoder trained on 5,749 STS-B
pairs with cosine regression + in-batch contrastive loss. Evaluation uses the
copied `la-cosine-similarity.py`, `nlp-embedding-similarity.py`,
`nlp-sentence-embeddings.py`, and `nlp-document-ranking-bm25.py` parts.

| Metric | Result |
|---|---|
| Dev Spearman | **0.6841** (best epoch 2 of 15) |
| Retrieval, 500 queries / 1,467 candidates | **acc@1 0.35**, acc@5 0.468, MRR@10 0.449 |
| Random retrieval baseline | 0.00068 |
| BM25, same 1,467-candidate pool (20 queries) | acc@1 **0.55** |
| BM25, easier 200-candidate pool (100 queries) | acc@1 0.74 |

Honest read: the bi-encoder learns a strong similarity correlation (0.68
Spearman) and retrieves far above random, but BM25 is stronger on exact lexical
matching in STS-B. Search example:

```
query: A plane is taking off.
 1. +0.4381  A plane in the sky.
 2. +0.4347  Soggy solstice: Stonehenge gets soaking
 3. +0.4166  A person is burning cameras.
```

`semantic_search.py search` uses the copied cosine function for the final
ranking. The BM25 baseline is reported by `semantic_search.py eval`.

## 4. Instruction tuning (SFT)

Uses the copied `cs336-l15-assistant-only-sft-mask.py`: labels are shifted
next-token targets only where the next token belongs to the assistant response,
then cross-entropy is applied over those labels.

Alpaca run: 2,000 usable examples, 1,900 train / 100 val, one example per
128-token block, 94,315 assistant tokens. 2,000 steps, batch 2, grad-accum 8:

| Metric | Value |
|---|---|
| Validation assistant loss | 7.4709 -> **5.9204** |
| Training time | about 1 m 44 s |
| Behavior | still story-like; the 12.6M base is too small/undertrained for general Alpaca |

Focused instruction run: 105 short examples (colors, arithmetic, capitals,
greetings, shapes), 100 train / 5 val, 64-token blocks. 2,000 steps:

| Prompt | Response |
|---|---|
| What is 2 + 2? | **4.** |
| What color is grass? | **Green.** |
| Name a shape. | **Square.** |
| What is the capital of France? | **Paris.** |
| What sound does a cat make? | **A cat says meow.** |
| What is 2 + 3? (held out) | 4. (wrong) |

This shows the model learning the response format and memorizing the small
training set; it is not general reasoning.

## 5. Question answering (T5-small SQuAD)

`qa_system.py` uses the pretrained
`mrm8488/t5-small-finetuned-squadv2` checkpoint as an extractive reader, plus
the copied `nlp-document-ranking-bm25.py` part for retrieval over SQuAD dev
passages and 20 hand-written fact passages.

Reader input: `question: {question} context: {context}` (lowercase). The
checkpoint is sensitive to this format; uppercase reduces exact match.

| Metric | Result |
|---|---|
| Reader with gold context, 200 SQuAD dev questions | **EM 0.705**, F1 0.734 |
| BM25 retrieval + reader, 50 questions | recall@3 **0.620**, end-to-end EM **0.360** |
| Reader parameters | ~77M (T5-small fine-tuned on SQuAD v2) |
| Evaluation time | about 28 s on the P2000 |

Examples:

```
Q: Who wrote Frankenstein?
A: Mary Shelley

Q: Which NFL team represented the AFC at Super Bowl 50?
A: Denver Broncos

Q: What is the capital of France?   (with the matching context)
A: Paris
```

Commands:

```bash
python qa_system.py download
/home/d/.venvs/torch126/bin/python qa_system.py selftest --device cuda
/home/d/.venvs/torch126/bin/python qa_system.py ask --device cuda \
    --question "Who wrote Frankenstein?" --top-k 3
/home/d/.venvs/torch126/bin/python qa_system.py ask --device cuda \
    --question "What is the capital of France?" \
    --context "The capital of France is Paris."
/home/d/.venvs/torch126/bin/python qa_system.py eval --device cuda \
    --limit 200 --retrieval-limit 50
```

This is extractive QA: it answers from a supplied or retrieved passage. It is
not an open-domain reasoning chatbot. The end-to-end 0.36 EM is limited by
BM25 recall and by T5-small itself; the reader alone reaches 0.705 EM.

## 6. Retrieval-augmented reasoning QA (Flan-T5-base)

`reason_qa.py` combines:

- `google/flan-t5-base` (250M parameters) as the generative reasoner;
- BM25 retrieval of solved exemplars from the copied
  `nlp-document-ranking-bm25.py`;
- chain-of-thought prompts;
- a safe `CALC:` arithmetic calculator;
- the ch03 SymPy grader from `reasoning/` for answer checking.

Tasks are GSM8K (grade-school math word problems) and StrategyQA (yes/no
commonsense questions with facts). The grader was validated against all 8,792
GSM8K gold answers (1,319 test + 7,473 train) with zero extraction mismatches;
end-to-end Flan-T5 accuracy is not reported here.

```bash
python reason_qa.py download
/home/d/.venvs/torch126/bin/python reason_qa.py selftest --device cuda
/home/d/.venvs/torch126/bin/python reason_qa.py eval --device cuda \
    --task both --gsm8k-limit 30 --strategyqa-limit 100
/home/d/.venvs/torch126/bin/python reason_qa.py ask --device cuda \
    --task gsm8k --question "A train travels 60 mph for 2.5 hours. How far does it go?"
```

Flan-T5-base weights download on first use (~1 GB). The system does not
update any weights; it is prompting plus retrieval, not RL or distillation.

## 7. Small reasoner (Qwen2.5-0.5B-Instruct)

`small_reasoner.py` loads Hugging Face checkpoints into the from-scratch
`Qwen3Model` from `reasoning/`:

- **Qwen2.5-0.5B-Instruct** (494M parameters, 988 MB bf16) with QKV-bias
  support and the Hugging Face chat template;
- **SmolLM2-360M / 135M** (Llama layout, no QKV/MLP biases, RMSNorm eps 1e-5).

It evaluates on GSM8K with the ch03 SymPy grader, and provides a multi-turn
chat REPL with streaming, `/search`, and context accounting.

Measured on the P2000 with the chat prompt, greedy decoding, 512 max new
tokens, and the ch03 grader:

| Model | Questions | Result | Time |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 100 | **47/100 = 47.0%** | 789 s (~13.2 min), ~37.7 tok/s |
| Qwen2.5-0.5B-Instruct | 50 | 25/50 = 50.0% | 401 s |
| Qwen2.5-0.5B-Instruct | 10 | 6/10 = 60% (small sample) | 73 s |
| SmolLM2-360M | 10 | 0/10 zero-shot; 0/10 two-shot | below the capability floor |

The 100-question failures are reasoning/arithmetic, not format: all 100
produced an extractable answer (3 lacked a `\boxed{}` wrapper and used the
numeric fallback), only 3 hit the 512-token cap, and wrong answers averaged
321 generated tokens vs 271 for correct ones. The 50-question 50% was a
high-variance sample; 47% is the stable estimate. SmolLM2-360M is useful for
testing the loader, not for reasoning.

```bash
# GSM8K evaluation
/home/d/.venvs/torch126/bin/python small_reasoner.py eval \
    --model qwen2.5-0.5b-instruct --task gsm8k --limit 100 \
    --device cuda --max-new-tokens 512

# Interactive chat
/home/d/.venvs/torch126/bin/python small_reasoner.py chat \
    --model qwen2.5-0.5b-instruct --device cuda

# One-shot question
/home/d/.venvs/torch126/bin/python small_reasoner.py ask \
    --model qwen2.5-0.5b-instruct --device cuda \
    --question "What is 2 + 2?"

# Self-consistency code is present but not yet run
/home/d/.venvs/torch126/bin/python small_reasoner.py eval \
    --model qwen2.5-0.5b-instruct --task gsm8k --limit 100 \
    --device cuda --max-new-tokens 512 --num-samples 3 --temperature 0.7
```

Chat commands: `/reset`, `/system <text>`, `/temp <value>`, `/tokens <n>`,
`/status`, `/search <query>`, `/exit`.

`/search` uses the keyless `ddgs` DuckDuckGo package plus the Wikipedia API.
It prints numbered sources, feeds them into the prompt, and asks the model to
answer from them with citations. The model itself has no web access; the
harness retrieves the sources. Treat the generated summary as unverified and
check the printed sources. `/status` reports the context budget; when it is
exceeded, the oldest user/assistant pairs are dropped, not summarized.

Model files live in the Hugging Face cache, not the repo:

- `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct` — 988 MB
- `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M` — 724 MB

Qwen2.5-0.5B has only 2 KV heads × 64 dimensions, so its KV cache is about
12 KB/token; the full 32K context needs ~384 MB, which fits alongside the ~1 GB
of weights on the 4.2 GB P2000. Full-parameter training does not fit: weights,
gradients, and AdamW state are ~4 GB before activations.

## Vendored reasoning-from-scratch subset

`build-LLM/reasoning/` is a 272 KB Apache-2.0 subset of
[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch),
copied 2026-11-12. It contains only the modules used by `small_reasoner.py`
and `reason_qa.py`:

| Module | Role |
|---|---|
| `qwen3.py` | Qwen3-style model, tokenizer wrapper, HF weight loader, KV cache |
| `ch02.py` | greedy KV-cached streaming generation |
| `ch03.py` | `\boxed{}` extraction, LaTeX normalization, SymPy grading, MATH-500 loader |
| `ch04.py` | temperature/top-p streaming generation |
| `utils.py` | download helper used by `qwen3.py` |

`LICENSE`, `NOTICE.md`, `README.md`, and `requirements.txt` are included. The
full upstream copy at `build-LLM/reasoning-from-scratch/` (6.5 MB, including
the copyrighted `book.txt`) stays local and is gitignored. The book's training
chapters (`ch05`–`ch08`: scoring, GRPO, distillation) are not wired into this
build.

## Which CS336 files are actually used?

**No, not all 113 are on an active path.** All are copied verbatim into
`parts/cs336/`, but a coherent system uses a subset. The active CS336 Python
solutions are:

| Lesson | Active files |
|---|---|
| l01 | `train-byte-pair-encoding`, `apply-bpe-merge-ranks` |
| l02 | `einsum-attention-scores` (self-test) |
| l03 | `rmsnorm-forward`, `parameter-matched-swiglu`, `rotary-query-key-embeddings`, `causal-grouped-query-attention` |
| l10 | `kv-cache-decode-step` (self-test) |
| l14 | `stable-exact-deduplication` (mixed corpus) |
| l15 | `assistant-only-sft-mask` (SFT) |

That is **10 of 58 Python solutions**. The large-corpus tokenizer is the
HuggingFace `tokenizers` byte-level BPE, not the CS336 l01 BPE; the CS336 BPE
is still available with `--tokenizer bpe` for small corpora.

The remaining 48 Python solutions are copied reference material. They cover
l02 FLOP/memory accounting, l04 DeltaNet/linear attention/Mamba2/MoE, l05 GPU
analysis, l06 Triton, l07 collectives, l08 parallelism, l09 scaling laws, l11
Muon/μP, l12 evaluation, l14 minhash/temperature mixing, l15 DPO/preference/KL,
l16 GRPO, l17 multimodal, and l18 serving. They are useful for extending the
system, but they are not required to run it.

## Cross-track parts used

| Copied file | Role |
|---|---|
| `pytorch-lr-warmup-scheduler.py` | warmup + cosine LR |
| `nlp-text-generation-sampling.py` | temperature/top-k sampling |
| `nlp-perplexity-computation.py` | single-sequence perplexity |
| `pytorch-masked-causal-attention.py` | attention reference (self-test) |
| `la-cosine-similarity.py` | semantic scoring |
| `la-vector-norms.py` | vector utilities |
| `nlp-sentence-embeddings.py` | sentence-pooling reference |
| `nlp-embedding-similarity.py` | ranking reference |
| `nlp-document-ranking-bm25.py` | BM25 baseline and QA retrieval |
| `nlp-tokenization.py` | tokenization reference |

## Data

- `data/corpus.txt` — 200 KB Tiny Shakespeare excerpt (public domain), bundled.
- `data/mixed.txt` — generated by `mix_corpus.py` from TinyStories, Tiny
  Shakespeare, and two Project Gutenberg books; 14 MB after exact dedup.
- `data/STS-B/` — downloaded by `semantic_search.py download`; GLUE STS-B.
- `data/sft/alpaca_subset.json` — 2,000 Stanford Alpaca examples
  (CC BY-NC 4.0, research use).
- `data/sft/tiny_instructions.json` — 105 hand-written short instruction/answer
  pairs for the focused SFT demo.
- `data/reason/` — GSM8K and StrategyQA JSON downloaded by
  `reason_qa.py download`.
- `data/qa/` — SQuAD dev passages and corpus downloaded by
  `qa_system.py download`.
- `runs/small_reasoner/` — GSM8K evaluation JSONL and tokenizer copies.

Generated corpora and downloaded data are ignored by git; `runs/` is ignored.

## Environment

- CPU path: system Python, `torch 2.15.0+cu132`, NumPy 2.5.4, matplotlib.
- GPU path: `/home/d/.venvs/torch126` with `torch 2.10.0+cu126`, CUDA on the
  Quadro P2000 (4.2 GB). NumPy 2.5.4, `tokenizers`, `sympy`, `requests`,
  `ddgs`, `huggingface_hub`, and `safetensors` were installed into that venv.
- Use `--device cuda` only with the torch126 venv. The system torch build does
  not contain sm_61 kernels.
- `requirements.txt`: `numpy`, `torch`, `matplotlib`, `tokenizers`,
  `transformers`, `sympy`, `requests`, `ddgs`, `huggingface_hub`,
  `safetensors`, `sentencepiece`, `protobuf`.
- Hugging Face checkpoints are cached under `~/.cache/huggingface/hub`; they
  are not in the repo. `reason_qa.py` downloads Flan-T5-base on first use.

## Limitations

- The scaled LLM is trained for 2,000 steps (~2 epochs of 3.5M tokens); it
  learns simple story patterns, not general meaning.
- The semantic bi-encoder is small and trained only on STS-B; BM25 beats it on
  exact retrieval, though its Spearman correlation is strong.
- SFT on Alpaca with the 12.6M base still produces story-like text. The focused
  run answers its training prompts and fails held-out arithmetic.
- QA is extractive and retrieval-bound: 0.705 EM with the gold passage, 0.36
  end-to-end EM with BM25 retrieval. It is not open-domain reasoning.
- A real meaning/instruction-following system needs many orders of magnitude
  more data and compute; use a modern GPU and a much larger base model.
- The tiny and scaled LLM loops recompute the full context per token; the
  copied KV-cache path is self-tested there but not used in their loops. The
  small reasoner does use KV caching.
- The small reasoner is a 0.5B instruct model: **47% on 100 GSM8K test
  questions**, with genuine arithmetic/reasoning failures. It is not a general
  reasoner. SmolLM2-360M is below the capability floor.
- Self-consistency code is present (`--num-samples`) but has not been run; its
  expected gain is therefore not claimed here.
- `/search` retrieval is real (DDGS + Wikipedia), but the answer is a
  model-generated summary. Verify the printed sources; the model has no
  autonomous web access.
- Full-parameter GRPO or distillation does not fit on the P2000: weights,
  gradients, and AdamW state are ~4 GB before activations against a 3.9 GiB
  budget. The book's training runs need a 24 GB+ GPU and are not wired into
  this build.
- `reason_qa.py` is retrieval plus prompting with Flan-T5-base; it updates no
  weights. End-to-end Flan-T5 accuracy is not reported here.

## Provenance

Copied unmodified on 2026-11-11 from
`study-plans/language-modeling-from-scratch/` and the relevant `pytorch-basics`,
`cracking-nlp`, and `math-linear-algebra` exercises.

The vendored reasoning subset in `reasoning/` was copied unmodified on
2026-11-12 from https://github.com/rasbt/reasoning-from-scratch (Apache-2.0;
see `reasoning/LICENSE` and `reasoning/NOTICE.md`). The full upstream copy is
kept locally and gitignored; its `book.txt` is not redistributed.
