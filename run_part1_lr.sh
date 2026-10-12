#!/usr/bin/env bash
# Learning-rate check for the RNN: also train it with lr 3e-3 (everything else
# unchanged) and pick the LR on VALIDATION loss only. The LSTM's LR check is in
# run_lstm_fix.sh.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
mkdir -p runs/logs

python train_neural.py --model rnn  --lr 3e-3 --out_dir runs/rnn_lr3e-3  2>&1 | tee runs/logs/rnn_lr3e-3.log
