"""Collect Part I results: loss-curve figure, comparison table, generations side by side.

  python summarize.py --runs runs/ngram runs/rnn runs/lstm runs/transformer
  python summarize.py --runs lstm=runs/lstm_lr3e-3 ...   # label=path
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
    ap.add_argument("--bench", default="runs/summary/inference_bench.json",
                    help="bench_inference.py output; its numbers replace the training-time ones")
    args = ap.parse_args()
    bench = load(args.bench)["models"] if Path(args.bench).exists() else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = {}
    for r in args.runs:
        name, _, path = r.rpartition("=")
        run_dirs[name or Path(path).name] = Path(path)
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
    for name, r in results.items():  # n-gram has no curve: reference line
        if "valid_by_order" in r:
            ax[1].axhline(r["valid_by_order"][str(r["best_order"])]["loss"], ls="--",
                          color="gray", label=f"{name} (order {r['best_order']})")
    for a, t in zip(ax, ["train loss (epoch avg, dropout on)", "validation loss"]):
        a.set_title(t)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("cross-entropy (nats/token)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=200)

    # ---------- comparison table
    rows = ["| model | test CE | test PPL | params / entries | size (MB) | best epoch | "
            "final train-val gap | train time (s) | score tok/s | gen tok/s | device |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        size = r["size"]
        size_str = (f"{size['total']/1e6:.1f}M params" if "trainable" in size
                    else f"{size['total']/1e6:.1f}M n-grams")
        inf = r["inference"]
        bench_run = next((v for v in bench.values()
                          if Path(v["run_dir"]) == run_dirs[name]), None)
        if bench_run:  # warmed-up, back-to-back timing (see bench_inference.py)
            inf = {"score_tokens_per_s": bench_run["score"]["tokens_per_s"],
                   "gen_tokens_per_s": bench_run["generate"]["tokens_per_s"]}
        mb = size.get("size_mb_fp32", r.get("model_file_mb", float("nan")))
        h = run_dirs[name] / "history.json"
        if h.exists():
            last = load(h)["epoch"][-1]
            gap = f"{last['train_loss'] - last['val_loss']:+.3f}"
            best_ep = str(r["best_epoch"])
        else:
            gap = best_ep = "-"
        rows.append(f"| {name} | {r['test_loss']:.3f} | {r['test_ppl']:.1f} | {size_str} | "
                    f"{mb:.0f} | {best_ep} | {gap} | "
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
