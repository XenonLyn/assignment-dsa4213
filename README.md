# DSA4213 Assignment 1 — Part I: next-token language models

Four next-token LMs trained from scratch on the same corpus and evaluated under one common setup:
an interpolated Kneser-Ney n-gram model, a vanilla RNN, an LSTM and a causal Transformer.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with PyTorch's MPS backend on Apple Silicon; CUDA and CPU also work (`--device cpu|cuda|mps`).

## Data
WikiText-2 (tokenised version `wikitext-2-v1`, CC BY-SA), downloaded automatically from the
HuggingFace Hub (`Salesforce/wikitext`) into `data/wikitext-2/{train,valid,test}.txt` on first run.
Any corpus works if you put three plain-text files with those names into `--data_dir`.

Preprocessing (`common.py`): lowercase, whitespace tokenisation, `<eos>` at the end of every non-empty
line, empty lines dropped. The vocabulary is built on the training split only; out-of-vocabulary
tokens map to `<unk>`. Specials: `<pad>`=0, `<unk>`=1, `<eos>`=2.

## Common evaluation setup
Each split is cut into non-overlapping chunks of `--eval_seq_len` (default 128) targets.
Every token except the first is a target exactly once, and each target may only use the preceding
tokens *inside its chunk* — for all four models, including the n-gram (no hidden state is carried
across chunks). Padded targets in the last chunk are excluded from the loss.
Metric: average next-token cross-entropy (nats) and perplexity = exp(CE).
The n-gram order is selected on the validation set; neural models use the best-validation checkpoint.
`models.check_causality` asserts at start-up that no neural model can see future tokens.

## Commands (main results)
```bash
python ngram.py        --max_order 4                        --out_dir runs/ngram
python train_neural.py --model rnn                          --out_dir runs/rnn
python train_neural.py --model lstm                         --out_dir runs/lstm
python train_neural.py --model transformer --lr 5e-4 --dropout 0.2 --out_dir runs/transformer
python summarize.py --runs runs/ngram runs/rnn runs/lstm runs/transformer
```
Quick pipeline check: add `--max_train_tokens 200000 --epochs 1` to any command.

Each run writes `results.json` (test CE/PPL, parameter counts, training time, inference throughput,
hardware, generations), and neural runs also `history.json` (per-epoch train/val loss, grad norms)
and `best.pt`. `summarize.py` writes `runs/summary/{loss_curves.png,table.md,generations.md}`.

## Default configuration
| | RNN / LSTM | Transformer |
|---|---|---|
| embedding dim | 300 (random init U(-0.1, 0.1), trainable) | 300 |
| body | 2 layers, hidden 512 | 4 pre-LN blocks, 6 heads, FFN 1200, learned positions |
| dropout | 0.3 | 0.2 (recommended) |
| output layer | untied linear | untied linear |

Shared: train context 128, batch 32, 10 epochs, AdamW (weight decay 0.01), 200 warm-up steps then
cosine decay to 10 % of the peak LR, gradient clipping 1.0, seed 42.

Decoding (identical for all four models, see `common.sample_next`): temperature 0.8, top-k 40,
50 new tokens, `<pad>` and `<unk>` banned, the same RNG seed per prompt (`--gen_seed` + prompt index).
Prompts are in `prompts.txt`.

## Timing procedure
* Training time: sum of the training loops only (validation excluded), with device synchronisation.
* Inference workload (common to all models): scoring the full test set under the common setup
  (tokens/s; n-gram on CPU, neural models with batch 32 on the training device), plus
  generation throughput (batch 1). RNN/LSTM generate incrementally with their hidden state;
  the Transformer re-encodes its context window every step (no KV cache).
* On a fanless MacBook Air, run timing on a plugged-in, cooled-down machine and report this.

## Reproducibility notes
Seeds are fixed (`--seed`, `--gen_seed`), but MPS kernels are not fully deterministic, so repeated
runs may differ slightly in the last digits.

## Attribution
Written with assistance from Claude (Anthropic). Cite any course notebooks you adapt here.
