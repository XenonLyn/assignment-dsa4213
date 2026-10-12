#!/usr/bin/env bash
# Retrain every LSTM after the MPS nn.LSTM fix (models.py: per-layer nn.LSTM with
# explicit dropout). Covers the Part I learning-rate choice and all of Part II.
#   1. LSTM, random trainable embeddings, lr 1e-3 and 3e-3 (pick on validation loss)
#   2. rebuild the fixed embedding tables, rescaled to the selected run's table norm
#   3. LSTM with frozen Skip-gram / frozen GloVe embeddings at the selected lr
#   4. re-evaluate every checkpoint in a fresh process and compare with the logs
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
mkdir -p runs/logs
# (LSTM runs made before the fix were moved once, by hand, to runs/archive_mps_lstm_bug/)

for lr in 1e-3 3e-3; do
    python train_neural.py --model lstm --lr $lr --out_dir runs/lstm_lr$lr \
        2>&1 | tee runs/logs/lstm_lr$lr.log
done

LR=$(python - <<'EOF'
import json
v = {lr: json.load(open(f"runs/lstm_lr{lr}/results.json"))["best_val_loss"] for lr in ("1e-3", "3e-3")}
print(min(v, key=v.get))
EOF
)
echo "[select] LSTM lr = $LR (by validation loss)" | tee runs/logs/lstm_lr_selection.log
echo "$LR" > runs/lstm_selected_lr.txt

python embeddings.py w2v   --out emb/w2v_sg300.pt  --ref_ckpt runs/lstm_lr$LR/best.pt
python embeddings.py glove --out emb/glove6b300.pt --ref_ckpt runs/lstm_lr$LR/best.pt

for e in w2v_sg300:lstm_w2v_frozen glove6b300:lstm_glove_frozen; do
    python train_neural.py --model lstm --lr $LR --embedding_path emb/${e%%:*}.pt \
        --freeze_embeddings --out_dir runs/${e##*:} 2>&1 | tee runs/logs/${e##*:}.log
done

# fresh-process re-evaluation must reproduce the logged test loss
python - <<'EOF' 2>&1 | tee runs/logs/lstm_fresh_eval_check.log
import json, torch
from argparse import Namespace
from common import load_corpus, make_chunks
from models import build_model
from train_neural import evaluate
vocab, splits, _ = load_corpus("data/wikitext-2")
X, Y = make_chunks(splits["test"], 128)
lr = open("runs/lstm_selected_lr.txt").read().strip()
for run in ["runs/lstm_lr1e-3", "runs/lstm_lr3e-3", "runs/lstm_w2v_frozen", "runs/lstm_glove_frozen"]:
    r = json.load(open(run + "/results.json"))
    m = build_model("lstm", len(vocab), Namespace(**r["args"]))
    m.load_state_dict(torch.load(run + "/best.pt", map_location="cpu", weights_only=True))
    fresh = evaluate(m, X, Y, 32, torch.device("cpu"))[0]
    print(f"{run}: logged {r['test_loss']:.4f} | fresh (CPU) {fresh:.4f} | "
          f"{'OK' if abs(fresh - r['test_loss']) < 1e-3 else 'MISMATCH'}")
EOF
