"""Collect Part I results: loss-curve figure, comparison table, generations side by side.

  python summarize.py --runs runs/ngram runs/rnn runs/lstm runs/transformer
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="runs/summary")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = {Path(r).name: Path(r) for r in args.runs}
    results = {n: load(d / "results.json") for n, d in run_dirs.items()}

    # ---------- loss curves (neural models only)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for name, r in results.items():
        h = run_dirs[name] / "history.json"
        if not h.exists():
            continue
        ep = load(h)["epoch"]
        e = [x["epoch"] for x in ep]
        ax[0].plot(e, [x["train_loss"] for x in ep], marker="o", label=name)
        ax[1].plot(e, [x["val_loss"] for x in ep], marker="o", label=name)
    for a, t in zip(ax, ["train loss (epoch avg, dropout on)", "validation loss"]):
        a.set_title(t)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("cross-entropy (nats/token)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=200)

    # ---------- comparison table
    rows = ["| model | test CE | test PPL | params / entries | train time (s) | "
            "score tok/s | gen tok/s | device |",
            "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        size = r["size"]
        size_str = (f"{size['total']/1e6:.1f}M params" if "trainable" in size
                    else f"{size['total']/1e6:.1f}M n-grams")
        inf = r["inference"]
        rows.append(f"| {name} | {r['test_loss']:.3f} | {r['test_ppl']:.1f} | {size_str} | "
                    f"{r['train_time_s']:.0f} | {inf['score_tokens_per_s']:.0f} | "
                    f"{inf['gen_tokens_per_s']:.1f} | {r['device']} |")
    table = "\n".join(rows)
    print(table)

    # ---------- generations side by side
    lines = ["# Generations (same prompts, seeds and decoding settings)\n"]
    prompts = [g["prompt"] for g in next(iter(results.values()))["generations"]]
    for i, p in enumerate(prompts):
        lines.append(f"## Prompt: {p}\n")
        for name, r in results.items():
            cont = r["generations"][i]["continuation"].replace("\n", " / ")
            lines.append(f"**{name}**: {cont}\n")
    (out / "generations.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "table.md").write_text(table + "\n", encoding="utf-8")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
