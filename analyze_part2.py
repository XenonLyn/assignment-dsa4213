"""Part II analysis: embedding settings of one LSTM.

  python analyze_part2.py --runs random=runs/lstm_lr3e-3 \
      w2v_frozen=runs/lstm_w2v_frozen glove_frozen=runs/lstm_glove_frozen

Writes runs/part2/:
  freq_buckets.md      test CE by training frequency of the INPUT token (the token
                       right before the target) and of the target, common chunked setup
  neighbours.md        cosine nearest neighbours of query words in each input table
  stability.md         per-run gradient-norm / loss-curve statistics
  diversity.md         distinct-n / repeated-trigram rate of the generations
  curves.png           validation loss per epoch + logged training loss per step
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import (IGNORE_INDEX, SPECIALS, UNK, EOS, get_device, load_corpus,
                    make_chunks)
from models import build_model

BUCKETS = [(1, 10), (11, 100), (101, 1000), (1001, 10000), (10001, 10**9)]
QUERIES = ["king", "france", "january", "album", "war", "river", "three", "played",
           "guitar", "scientists", "hurricane", "quickly"]


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def token_losses(model, X, Y, device, bs=32):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x, y = X[i: i + bs].to(device), Y[i: i + bs].to(device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                              ignore_index=IGNORE_INDEX, reduction="none")
        out.append(nll.view_as(y).cpu())
    return torch.cat(out)


def bucket_of(counts):
    b = np.full(len(counts), -1)
    for k, (lo, hi) in enumerate(BUCKETS):
        b[(counts >= lo) & (counts <= hi)] = k
    return b


def neighbours(E, vocab, word, k=6):
    if word not in vocab.stoi:
        return "(not in vocab)"
    En = F.normalize(E, dim=1)
    sims = En @ En[vocab.stoi[word]]
    sims[: len(SPECIALS)] = -2  # never return specials
    sims[vocab.stoi[word]] = -2
    return ", ".join(vocab.itos[i] for i in sims.topk(k).indices.tolist())


def diversity(gens):
    d1 = d2 = rep = 0.0
    for g in gens:
        t = g["continuation"].replace("\n", " <eos> ").split()
        bi, tri = list(zip(t, t[1:])), list(zip(t, t[1:], t[2:]))
        d1 += len(set(t)) / len(t)
        d2 += len(set(bi)) / len(bi)
        rep += 1 - len(set(tri)) / len(tri)
    n = len(gens)
    return d1 / n, d2 / n, rep / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir")
    ap.add_argument("--glove_stats", default="emb/glove6b300.stats.json")
    ap.add_argument("--out", default="runs/part2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data_dir", default="data/wikitext-2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    vocab, splits, _ = load_corpus(args.data_dir)
    V = len(vocab)
    train_counts = np.bincount(splits["train"].numpy(), minlength=V)
    runs = {}
    for spec in args.runs:
        label, _, d = spec.rpartition("=")
        runs[label] = Path(d)

    # ------------------------------------------------ per-token test losses
    X, Y = make_chunks(splits["test"], 128)
    mask = (Y != IGNORE_INDEX)
    inp, tgt = X[mask].numpy(), Y[mask].numpy()
    in_b, tg_b = bucket_of(train_counts[inp]), bucket_of(train_counts[tgt])
    glove_found = None  # vocab rows that received a real GloVe vector
    if Path(args.glove_stats).exists():
        import gensim.downloader as api
        from embeddings import GLOVE_ALIASES
        kv = api.load(load(args.glove_stats)["args"]["glove_name"])
        glove_found = np.array([GLOVE_ALIASES.get(w, w) in kv.key_to_index for w in vocab.itos])
        del kv
    losses, tables, results = {}, {}, {}
    for label, d in runs.items():
        r = load(d / "results.json")
        results[label] = r
        a = Namespace(**r["args"])
        model = build_model(a.model, V, a).to(device)
        model.load_state_dict(torch.load(d / "best.pt", map_location=device, weights_only=True))
        losses[label] = token_losses(model, X, Y, device)[mask].numpy()
        tables[label] = model.embedding.weight.detach().cpu()
        del model
        print(f"{label}: test CE {losses[label].mean():.4f} (results.json {r['test_loss']:.4f})")

    labels = list(runs)
    lines = ["# Test CE by training frequency (common chunked setup)\n",
             "Input token = the token right before the target (what the input embedding encodes).\n",
             "| input-token train freq | share of test targets | "
             + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    special = {"<unk>": inp == UNK, "<eos>": inp == EOS}
    rows = [(f"{lo}-{hi if hi < 10**9 else '∞'}", (in_b == k) & (inp != UNK) & (inp != EOS))
            for k, (lo, hi) in enumerate(BUCKETS)] + list(special.items())
    if glove_found is not None:
        words = (inp != UNK) & (inp != EOS)
        rows += [("word not in GloVe (random frozen row)", words & ~glove_found[inp]),
                 ("word in GloVe", words & glove_found[inp])]
    for name, m in rows:
        if m.sum() == 0:
            continue
        lines.append(f"| {name} | {m.mean():.3f} | "
                     + " | ".join(f"{losses[l][m].mean():.3f}" for l in labels) + " |")
    lines.append(f"| all | 1.000 | " + " | ".join(f"{losses[l].mean():.3f}" for l in labels) + " |")
    lines += ["\n| target-token train freq | share | " + " | ".join(labels) + " |",
              "|---|---|" + "---|" * len(labels)]
    for k, (lo, hi) in enumerate(BUCKETS):
        m = (tg_b == k) & (tgt != UNK) & (tgt != EOS)
        if m.sum():
            lines.append(f"| {lo}-{hi if hi < 10**9 else '∞'} | {m.mean():.3f} | "
                         + " | ".join(f"{losses[l][m].mean():.3f}" for l in labels) + " |")
    for name, m in {"<unk>": tgt == UNK, "<eos>": tgt == EOS}.items():
        lines.append(f"| {name} | {m.mean():.3f} | "
                     + " | ".join(f"{losses[l][m].mean():.3f}" for l in labels) + " |")
    (out / "freq_buckets.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ------------------------------------------------ nearest neighbours
    nn_lines = ["# Cosine nearest neighbours in each input-embedding table\n",
                "| word (train freq) | " + " | ".join(labels) + " |",
                "|---|" + "---|" * len(labels)]
    for w in QUERIES:
        if w not in vocab.stoi:
            continue
        nn_lines.append(f"| {w} ({train_counts[vocab.stoi[w]]}) | "
                        + " | ".join(neighbours(tables[l], vocab, w) for l in labels) + " |")
    (out / "neighbours.md").write_text("\n".join(nn_lines) + "\n", encoding="utf-8")

    # ------------------------------------------------ stability + diversity
    st_lines = ["| setting | trainable params | best val PPL (epoch) | test PPL | first epoch val PPL < 250 / < 200 "
                "| grad-norm mean (ep1 / ep10) | grad-norm max (any epoch) | nonfinite steps "
                "| final train-val gap | train time (s) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    dv_lines = ["| setting | distinct-1 | distinct-2 | repeated-trigram |", "|---|---|---|---|"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, d in runs.items():
        r, h = results[label], load(d / "history.json")
        ep = h["epoch"]
        first = lambda thr: next((e["epoch"] for e in ep if e["val_ppl"] < thr), "-")
        st_lines.append(
            f"| {label} | {r['size']['trainable']/1e6:.2f}M | {np.exp(r['best_val_loss']):.1f} "
            f"({r['best_epoch']}) | {r['test_ppl']:.1f} | {first(250)} / {first(200)} | "
            f"{ep[0]['grad_norm_mean']:.2f} / {ep[-1]['grad_norm_mean']:.2f} | "
            f"{max(e['grad_norm_max'] for e in ep):.2f} | {sum(e['nonfinite_steps'] for e in ep)} | "
            f"{ep[-1]['train_loss'] - ep[-1]['val_loss']:+.3f} | {r['train_time_s']:.0f} |")
        d1, d2, rep = diversity(r["generations"])
        dv_lines.append(f"| {label} | {d1:.2f} | {d2:.2f} | {rep:.3f} |")
        ax[0].plot([e["epoch"] for e in ep], [e["val_loss"] for e in ep], marker="o", label=label)
        ax[1].plot([s["step"] for s in h["step"]], [s["train_loss"] for s in h["step"]],
                   lw=1, label=label)
    ax[0].set_title("validation loss")
    ax[0].set_xlabel("epoch")
    ax[1].set_title("training loss (logged every 100 steps, dropout on)")
    ax[1].set_xlabel("step")
    for a in ax:
        a.grid(alpha=0.3)
        a.set_ylabel("cross-entropy (nats/token)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=200)
    (out / "stability.md").write_text("\n".join(st_lines) + "\n", encoding="utf-8")
    (out / "diversity.md").write_text("\n".join(dv_lines) + "\n", encoding="utf-8")
    print("\n".join(st_lines))
    print("\n".join(dv_lines))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
