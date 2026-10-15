"""Figures for the report (vector PDF), drawn from the saved run results.

  python report/make_figures.py      # from the repository root -> report/figures/*.pdf

Colour follows the model across all figures (validated categorical slots, light mode):
LSTM blue, Transformer orange, RNN aqua, LSTM+Skip-gram green, LSTM+GloVe violet;
non-LM references (4-gram, TF-IDF) in neutral grey. Lines are direct-labelled.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"

C = {"lstm": "#2a78d6", "transformer": "#eb6834", "rnn": "#1baf7a",
     "lstm_w2v": "#008300", "lstm_glove": "#4a3aa7"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
NAME = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer",
        "lstm_w2v": "LSTM + Skip-gram (frozen)", "lstm_glove": "LSTM + GloVe (frozen)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(p):
    with open(ROOT / p, encoding="utf-8") as f:
        return json.load(f)


def epochs(run):
    return load(f"runs/{run}/history.json")["epoch"]


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center",
                fontsize=7, color=INK2)
    ax.plot([x], [y], "o", color=color, ms=4, zorder=4)


def fig_part1():
    runs = {"rnn": "rnn", "lstm": "lstm_lr3e-3", "transformer": "transformer"}
    ngram = load("runs/ngram/results.json")
    ng_val = ngram["valid_by_order"][str(ngram["best_order"])]["loss"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.15))
    offs = [{"rnn": 8, "lstm": -6, "transformer": 6},   # (a) LSTM ends just below Transformer
            {"rnn": 0, "lstm": 5, "transformer": -5}]   # (b) LSTM ends just above Transformer
    for key, run in runs.items():
        ep = epochs(run)
        e = [x["epoch"] for x in ep]
        for a, k in ((ax[0], "val_loss"), (ax[1], "train_loss")):
            ys = [x[k] for x in ep]
            a.plot(e, ys, color=C[key], label=NAME[key])
            end_label(a, e[-1], ys[-1], NAME[key], C[key], offs[0 if a is ax[0] else 1][key])
    ax[0].axhline(ng_val, color=MUTED, ls="--", lw=1.0)
    ax[0].annotate("4-gram KN (order 4)", (4.6, ng_val), xytext=(0, -8), textcoords="offset points",
                   fontsize=7, color=INK2)
    ax[0].set_title("(a) Validation cross-entropy")
    ax[1].set_title("(b) Training cross-entropy (epoch mean, dropout on)")
    for a in ax:
        a.set_xlabel("epoch")
        a.set_xticks(range(1, 11))
        a.set_xlim(0.7, 12.6)
    ax[0].set_ylabel("nats / token")
    ax[0].legend(loc="upper right", handlelength=1.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig1_part1_curves.pdf")


def fig_part2():
    runs = {"lstm": "lstm_lr3e-3", "lstm_w2v": "lstm_w2v_frozen", "lstm_glove": "lstm_glove_frozen"}
    label = {"lstm": "A: random, trainable", "lstm_w2v": "B: Skip-gram, frozen",
             "lstm_glove": "C: GloVe, frozen"}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.15), gridspec_kw={"width_ratios": [1, 1.15]})
    for key, run in runs.items():
        ep = epochs(run)
        e = [x["epoch"] for x in ep]
        ys = [x["val_loss"] for x in ep]
        ax[0].plot(e, ys, color=C[key], label=label[key])
        end_label(ax[0], e[-1], ys[-1], label[key].split(":")[0], C[key],
                  {"lstm": 0, "lstm_w2v": -6, "lstm_glove": 5}[key])
    ax[0].set_title("(a) Validation cross-entropy")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("nats / token")
    ax[0].set_xticks(range(1, 11))
    ax[0].set_xlim(0.7, 11.2)
    ax[0].legend(loc="upper right", handlelength=1.5)

    # test CE change vs. setting A, by training frequency of the input token
    md = (ROOT / "runs/part2/freq_buckets.md").read_text().splitlines()
    rows = [l for l in md if l.startswith("| ") and "share" not in l and "---" not in l]
    first_table = rows[: rows.index(next(r for r in rows if r.startswith("| all")))]
    buckets = [r for r in first_table if r.split("|")[1].strip()[0].isdigit()]
    names = [r.split("|")[1].strip().replace("-∞", "+").replace("10001", "10k").replace("1001-10000", "1k-10k")
             for r in buckets]
    names = [n.replace("101-1000", "101-1k") for n in names]
    vals = np.array([[float(x) for x in r.split("|")[3:6]] for r in buckets])
    dB, dC = vals[:, 1] - vals[:, 0], vals[:, 2] - vals[:, 0]
    x = np.arange(len(names))
    w = 0.36
    ax[1].bar(x - w / 2 - 0.01, dB, w, color=C["lstm_w2v"], label="B: Skip-gram − A")
    ax[1].bar(x + w / 2 + 0.01, dC, w, color=C["lstm_glove"], label="C: GloVe − A")
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_xticks(x, names)
    ax[1].set_xlabel("training frequency of the input token")
    ax[1].set_ylabel("Δ test CE vs. A (nats)")
    ax[1].set_title("(b) Gain over random embeddings, by input frequency")
    ax[1].set_ylim(-0.40, 0.0)
    ax[1].legend(loc="lower center", ncol=2, handlelength=1.2)
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig2_part2.pdf")


def fig_part3():
    r = load("runs/part3/results.json")
    res, curve = r["results"], r["curve"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.35))
    lms = ["rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove"]
    short = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer",
             "lstm_w2v": "LSTM+SG", "lstm_glove": "LSTM+GloVe"}
    for lm in lms:
        m, l = res[f"{lm}/mean"], res[f"{lm}/last"]
        ax[0].plot([m["lm_test_ppl"]] * 2, [l["test_acc"], m["test_acc"]], color=GRID, lw=1.2, zorder=1)
        ax[0].scatter(m["lm_test_ppl"], m["test_acc"], s=22, color=INK, zorder=3)
        ax[0].scatter(l["lm_test_ppl"], l["test_acc"], s=22, facecolor="white", edgecolor=INK,
                      marker="s", linewidths=1.0, zorder=3)
        dx, dy = {"lstm_w2v": (6, -7), "lstm_glove": (6, 0), "lstm": (-6, 0),
                  "transformer": (6, 2), "rnn": (-6, 4)}[lm]
        ax[0].annotate(short[lm], (m["lm_test_ppl"], m["test_acc"]), xytext=(dx, dy),
                       textcoords="offset points", fontsize=7, color=INK2, va="center",
                       ha="right" if dx < 0 else "left")
    ax[0].scatter([], [], s=22, color=INK, label="mean pooling")
    ax[0].scatter([], [], s=22, facecolor="white", edgecolor=INK, marker="s", label="last token")
    for key, ls, txt in (("tfidf_bow", "--", "TF-IDF (no LM)"),
                         ("emb_glove/mean", ":", "GloVe average (no context)")):
        ax[0].axhline(res[key]["test_acc"], color=MUTED, ls=ls, lw=1.0)
        ax[0].annotate(txt, (220, res[key]["test_acc"]), xytext=(0, 3 if key == "tfidf_bow" else -8),
                       textcoords="offset points", fontsize=6.5, color=INK2, ha="right")
    ax[0].set_xlabel("LM test perplexity (lower = better LM)")
    ax[0].set_ylabel("AG News test accuracy")
    ax[0].set_title("(a) Language-model quality vs. downstream accuracy")
    ax[0].set_ylim(0.37, 0.93)
    ax[0].set_xlim(100, 222)
    ax[0].legend(loc="lower left", handletextpad=0.3)

    ns = [int(k) for k in curve["tfidf_bow"]]
    series = [("tfidf_bow", MUTED, "-", "TF-IDF"),
              ("emb_glove/mean", C["lstm_glove"], "--", "GloVe avg."),
              ("lstm_glove/mean", C["lstm_glove"], "-", "LSTM+GloVe"),
              ("transformer/mean", C["transformer"], "-", "Transformer"),
              ("lstm/mean", C["lstm"], "-", "LSTM")]
    for key, col, ls, lab in series:
        ys = [curve[key][str(n)] for n in ns]
        ax[1].plot(ns, ys, color=col, ls=ls, marker="o", ms=3.5, label=lab)
    ax[1].set_xscale("log")
    ax[1].set_xticks(ns, [f"{n:,}" for n in ns])
    ax[1].minorticks_off()
    ax[1].set_xlabel("labelled training documents (log scale)")
    ax[1].set_ylabel("test accuracy")
    ax[1].set_title("(b) Accuracy vs. labelled data (mean pooling)")
    ax[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=2.0)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig3_part3.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_part1()
    fig_part2()
    fig_part3()
    print("saved to", OUT)
