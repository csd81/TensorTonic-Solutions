# LLM, Semantic Search, and SFT from the CS336 Parts

A build around the 113 files in
`study-plans/language-modeling-from-scratch/` (58 Python solutions, 54
write-ups, 1 CUDA kernel), plus useful pieces from other tracks.

It now contains four working systems:

1. **Tiny LLM** — a 640k-parameter decoder-only transformer on a bundled
   Shakespeare excerpt, CPU-friendly.
2. **Scaled LLM** — a 12.6M-parameter model trained on a 14 MB mixed,
   deduplicated corpus with a fast 8k HuggingFace BPE tokenizer, on the P2000.
3. **Semantic search** — a sentence bi-encoder trained on STS-B, evaluated with
   Spearman and retrieval against a BM25 baseline.
4. **Instruction tuning** — Alpaca + focused instruction data, trained with the
   copied CS336 assistant-only SFT mask.

## Quick start

```bash
cd /home/d/Desktop/TensorTonic-Solutions/build_LLM

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
```

The P2000 is `sm_61`; the system Python's `torch 2.15.0+cu132` cannot run it.
Use `/home/d/.venvs/torch126/bin/python` (torch 2.10.0+cu126). NumPy and
`tokenizers` were installed into that venv for these runs.

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
| `nlp-document-ranking-bm25.py` | BM25 baseline |
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

Generated corpora and downloaded data are ignored by git; `runs/` is ignored.

## Environment

- CPU path: system Python, `torch 2.15.0+cu132`, NumPy 2.5.4, matplotlib.
- GPU path: `/home/d/.venvs/torch126` with `torch 2.10.0+cu126`, CUDA on the
  Quadro P2000 (4.2 GB). NumPy 2.5.4 and `tokenizers` 0.23.3 were installed
  into that venv.
- Use `--device cuda` only with the torch126 venv. The system torch build does
  not contain sm_61 kernels.
- `requirements.txt`: `numpy`, `torch`, `matplotlib`, `tokenizers`.

## Limitations

- The scaled LLM is trained for 2,000 steps (~2 epochs of 3.5M tokens); it
  learns simple story patterns, not general meaning.
- The semantic bi-encoder is small and trained only on STS-B; BM25 beats it on
  exact retrieval, though its Spearman correlation is strong.
- SFT on Alpaca with the 12.6M base still produces story-like text. The focused
  run answers its training prompts and fails held-out arithmetic.
- A real meaning/instruction-following system needs many orders of magnitude
  more data and compute; use a modern GPU and a much larger base model.
- Generation recomputes the full context per token; the copied KV-cache path is
  self-tested but not used in the loop.

## Provenance

Copied unmodified on 2026-11-11 from
`study-plans/language-modeling-from-scratch/` and the relevant `pytorch-basics`,
`cracking-nlp`, and `math-linear-algebra` exercises.
