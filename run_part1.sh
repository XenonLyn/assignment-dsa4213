#!/usr/bin/env bash
# Part I main results. Models run one after another so timings are not
# distorted by contention (n-gram on CPU, neural models on the GPU/MPS).
set -euo pipefail
export PYTHONUNBUFFERED=1  # live logs through tee
cd "$(dirname "$0")"
mkdir -p runs/logs

python ngram.py --max_order 4 --out_dir runs/ngram 2>&1 | tee runs/logs/ngram.log
python train_neural.py --model rnn --out_dir runs/rnn 2>&1 | tee runs/logs/rnn.log
# LSTM runs: see run_lstm_fix.sh
python train_neural.py --model transformer --lr 5e-4 --dropout 0.2 \
    --out_dir runs/transformer 2>&1 | tee runs/logs/transformer.log
python summarize.py --runs ngram=runs/ngram rnn=runs/rnn lstm=runs/lstm_lr3e-3 transformer=runs/transformer
