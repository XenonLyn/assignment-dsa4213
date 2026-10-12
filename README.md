# DSA4213 Assignment 1 — Language models, embeddings and representations

* **Part I** — four next-token LMs trained from scratch on the same corpus and evaluated under one common
  setup: an interpolated Kneser-Ney n-gram model, a vanilla RNN, an LSTM and a causal Transformer.
  Results and discussion: [`PART1_RESULTS.md`](PART1_RESULTS.md).
* **Part II** — the Part I LSTM with three input-embedding settings: random/trainable, own Skip-gram
  (frozen), public GloVe (frozen). Results and discussion: [`PART2_RESULTS.md`](PART2_RESULTS.md).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested on an Apple M1 (16 GB), macOS, Python 3.11, PyTorch 2.5.1 with the MPS backend; CUDA and CPU also
work (`--device cpu|cuda|mps`).

## Data
* **WikiText-2** (tokenised version `wikitext-2-v1`, CC BY-SA), downloaded automatically from the
  HuggingFace Hub (`Salesforce/wikitext`) into `data/wikitext-2/{train,valid,test}.txt` on first run.
  Any corpus works if you put three plain-text files with those names into `--data_dir`.
* **GloVe 6B 300d** (Part II; Pennington et al., 2014; Wikipedia 2014 + Gigaword 5), downloaded by
  `embeddings.py glove` through `gensim.downloader` (`glove-wiki-gigaword-300`, ~394 MB, cached in
  `~/gensim-data/`). Not redistributed here.

Preprocessing (`common.py`): lowercase, whitespace tokenisation, `<eos>` at the end of every non-empty
line, empty lines dropped. The vocabulary (28,913 types) is built on the training split only;
out-of-vocabulary tokens map to `<unk>`. Specials: `<pad>`=0, `<unk>`=1, `<eos>`=2.

## Common evaluation setup
Each split is cut into non-overlapping chunks of `--eval_seq_len` (default 128) targets.
Every token except the first is a target exactly once, and each target may only use the preceding
tokens *inside its chunk* — for all models, including the n-gram (no hidden state is carried
across chunks). Padded targets in the last chunk are excluded from the loss.
Metric: average next-token cross-entropy (nats) and perplexity = exp(CE).
The n-gram order, the best epoch and the learning rate are selected on the validation set only.
Start-up checks in `train_neural.py`: `models.check_causality` (no model can see future tokens) and
`models.check_dropout_and_eval` (dropout resamples in train mode; eval output matches a CPU copy).

## Commands
Every model is trained on its own (never two at once), so the timings are not distorted.
```bash
# ---- Part I
python ngram.py        --max_order 4                                 --out_dir runs/ngram
python train_neural.py --model rnn  --lr 1e-3                        --out_dir runs/rnn
python train_neural.py --model rnn  --lr 3e-3                        --out_dir runs/rnn_lr3e-3
python train_neural.py --model lstm --lr 1e-3                        --out_dir runs/lstm_lr1e-3
python train_neural.py --model lstm --lr 3e-3                        --out_dir runs/lstm_lr3e-3
python train_neural.py --model transformer --lr 5e-4 --dropout 0.2   --out_dir runs/transformer
# selected on validation loss: RNN lr 1e-3, LSTM lr 3e-3 (Transformer: one value tried)
python summarize.py --runs ngram=runs/ngram rnn=runs/rnn lstm=runs/lstm_lr3e-3 transformer=runs/transformer
python summarize.py --out runs/summary/lr_check --runs rnn_lr1e-3=runs/rnn rnn_lr3e-3=runs/rnn_lr3e-3 \
    lstm_lr1e-3=runs/lstm_lr1e-3 lstm_lr3e-3=runs/lstm_lr3e-3 transformer_lr5e-4=runs/transformer
# common inference workload, all checkpoints back to back (median of 3)
python bench_inference.py --runs ngram=runs/ngram rnn=runs/rnn lstm=runs/lstm_lr3e-3 \
    transformer=runs/transformer lstm_w2v_frozen=runs/lstm_w2v_frozen lstm_glove_frozen=runs/lstm_glove_frozen

# ---- Part II (setting A = runs/lstm_lr3e-3 from Part I)
python embeddings.py w2v   --out emb/w2v_sg300.pt  --ref_ckpt runs/lstm_lr3e-3/best.pt   # own Skip-gram
python embeddings.py glove --out emb/glove6b300.pt --ref_ckpt runs/lstm_lr3e-3/best.pt   # public GloVe
python train_neural.py --model lstm --lr 3e-3 --embedding_path emb/w2v_sg300.pt  --freeze_embeddings \
    --out_dir runs/lstm_w2v_frozen
python train_neural.py --model lstm --lr 3e-3 --embedding_path emb/glove6b300.pt --freeze_embeddings \
    --out_dir runs/lstm_glove_frozen
python analyze_part2.py --runs random=runs/lstm_lr3e-3 w2v_frozen=runs/lstm_w2v_frozen \
    glove_frozen=runs/lstm_glove_frozen
```
Shell scripts that ran these in sequence: `run_part1.sh` (n-gram, RNN, Transformer), `run_part1_lr.sh`
(RNN lr 3e-3), `run_lstm_fix.sh` (all LSTM runs incl. Part II, LR selection, embedding tables, and a
fresh-process re-evaluation check). Quick pipeline check: add `--max_train_tokens 200000 --epochs 1`.

Outputs: each run writes `results.json` (test CE/PPL, parameter counts, training time, inference
throughput, hardware, generations); neural runs also `history.json` (per-epoch train/val loss, grad
norms, step log) and `best.pt`. `summarize.py` → `runs/summary/{loss_curves.png,table.md,generations.md}`;
`analyze_part2.py` → `runs/part2/{curves.png,freq_buckets.md,neighbours.md,stability.md,diversity.md}`;
`embeddings.py` → `emb/*.stats.json` (coverage, missing words, build time, rescaling).
Checkpoints, embedding tables and data are not in the repository (see `.gitignore`); rerun the commands.

## Configuration
| | RNN | LSTM | Transformer |
|---|---|---|---|
| embedding | 300-d, random U(−0.1, 0.1), trainable (Part II: frozen Skip-gram / GloVe) | same | same |
| body | 2-layer tanh `nn.RNN`, hidden 512 | 2 layers, hidden 512: one `nn.LSTM` per layer + `nn.Dropout` between | 4 pre-LN blocks, 6 heads, FFN 1200, learned positions |
| dropout | 0.3 (embedding, between layers, before softmax) | 0.3 (same places) | 0.2 |
| output layer | untied linear | untied linear | untied linear |
| peak LR | 1e-3 | 3e-3 | 5e-4 |

Shared: train context 128, batch 32, 10 epochs (5,070 updates), AdamW (weight decay 0.01), 200 warm-up
steps then cosine decay to 10% of the peak LR, gradient clipping 1.0, seed 42 (`--seed`).
n-gram: interpolated Kneser-Ney, orders 1–4 scored on validation, order 4 selected.
Skip-gram (`embeddings.py`): dim 300, window 5, 5 negatives, subsampling 1e-3, min_count 1, 10 epochs,
8 workers, seed 42. Frozen tables are rescaled by one scalar to the mean row norm of the learned table
of `runs/lstm_lr3e-3`; GloVe-missing words get N(μ_d, σ_d) vectors (seed 42), `<pad>` = 0.

Decoding (identical for all models, `common.sample_next`): temperature 0.8, top-k 40, 50 new tokens,
`<pad>` and `<unk>` banned, the same RNG seed per prompt (`--gen_seed` 1234 + prompt index).
Prompts are in `prompts.txt`.

## Timing procedure
* Training time: sum of the training loops only (validation excluded), with device synchronisation.
* Inference workload (common to all models, `bench_inference.py`): scoring the full test set under the
  common setup (tokens/s; n-gram on CPU, neural models with batch 32 on MPS) and generating 50 tokens for
  each of the 5 prompts (batch 1); one warm-up, median of 3. RNN/LSTM generate incrementally with their
  hidden state; the Transformer re-encodes its context window every step (no KV cache).
* Run timing on a plugged-in, awake machine (`caffeinate -dims`); training runs were done over several
  days on a laptop, so training times differ by up to ~10% for identical configurations.

## Reproducibility notes
* Seeds are fixed, but MPS kernels and multi-threaded gensim are not fully deterministic, so repeated
  runs may differ slightly.
* **PyTorch 2.5.1 / MPS issue with multi-layer `nn.LSTM`**: its built-in inter-layer dropout is not
  resampled between calls, and evaluation at the same tensor shape right after training can reuse the
  train-mode graph, giving wrong validation/test losses. The LSTM is therefore built from single-layer
  `nn.LSTM`s with explicit `nn.Dropout` (same parameters), and `models.check_dropout_and_eval` guards
  against the problem at start-up. Runs trained with the affected version are kept in
  `runs/archive_mps_lstm_bug/` for reference only.

## Attribution
* Data: WikiText-2 (Merity et al., 2016, CC BY-SA); GloVe (Pennington et al., 2014).
* Libraries: PyTorch, HuggingFace `datasets`, gensim, NumPy, matplotlib.
* Code written with assistance from Claude (Anthropic). Cite any course notebooks you adapt here.
