"""Common inference benchmark: re-time every trained model back to back, in one
process, on the same machine state (Part I cost comparison).

Workload per model (median of --repeats runs, after one warm-up):
  * scoring: the full test set under the common chunked setup (batch 32 for the
    neural models on --device; the n-gram on CPU), reported as tokens/s
  * generation: every prompt in prompts.txt, 50 new tokens, batch 1, same
    decoding settings as training-time generation, reported as tokens/s

  python bench_inference.py --runs ngram=runs/ngram rnn=runs/rnn \
      lstm=runs/lstm_lr3e-3 transformer=runs/transformer
"""
import argparse
import json
import pickle
import statistics
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from common import (PAD, UNK, Timer, get_device, hardware_info, load_corpus,
                    load_prompts, make_chunks, save_json, tokenize_line)
from models import build_model
from ngram import KneserNeyLM
from train_neural import evaluate, generate


class _Unpickler(pickle.Unpickler):
    """ngram.py pickles its model from __main__; map it back to the module."""
    def find_class(self, module, name):
        if name == "KneserNeyLM":
            return KneserNeyLM
        return super().find_class(module, name)


def bench(fn, repeats):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t, n = fn()
        times.append((t, n))
    t_med = statistics.median(t for t, _ in times)
    return {"time_s": t_med, "tokens": times[0][1], "tokens_per_s": times[0][1] / t_med,
            "all_times_s": [t for t, _ in times]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir")
    ap.add_argument("--data_dir", default="data/wikitext-2")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="runs/summary/inference_bench.json")
    args = ap.parse_args()
    device = get_device(args.device)
    torch.set_grad_enabled(False)

    vocab, splits, _ = load_corpus(args.data_dir)
    test = splits["test"]
    prompts = [vocab.encode(tokenize_line(p, add_eos=False)) for p in load_prompts(args.prompts)]
    results = {"hardware": hardware_info(), "device": str(device),
               "repeats": args.repeats, "models": {}}

    for spec in args.runs:
        label, _, run_dir = spec.rpartition("=")
        run_dir = Path(run_dir)
        cfg = json.load(open(run_dir / "results.json"))
        a = Namespace(**cfg["args"])
        assert json.load(open(run_dir / "vocab.json")) == vocab.itos, "vocab mismatch"
        banned = (PAD,) if a.allow_unk else (PAD, UNK)

        if (run_dir / "model.pkl").exists():  # n-gram, CPU
            with open(run_dir / "model.pkl", "rb") as f:
                lm = _Unpickler(f).load()
            order, ids = cfg["best_order"], test.tolist()

            def score():
                with Timer() as t:
                    _, n = lm.score_chunks(ids, a.eval_seq_len, order)
                return t.elapsed, n

            def gen():
                n = 0
                with Timer() as t:
                    for i, p in enumerate(prompts):
                        rng = np.random.default_rng(a.gen_seed + i)
                        n += len(lm.generate(p, a.gen_len, a.temperature, a.top_k,
                                             rng, banned, order))
                return t.elapsed, n
            dev = "cpu"
        else:
            model = build_model(a.model, len(vocab), a).to(device)
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
            model.eval()
            X, Y = make_chunks(test, a.eval_seq_len)

            def score():
                with Timer(device) as t:
                    _, n = evaluate(model, X, Y, a.eval_batch_size, device)
                return t.elapsed, n

            def gen():
                n = 0
                with Timer(device) as t:
                    for i, p in enumerate(prompts):
                        rng = np.random.default_rng(a.gen_seed + i)
                        n += len(generate(model, p, a.gen_len, a.temperature, a.top_k,
                                          rng, banned, device))
                return t.elapsed, n
            dev = str(device)

        res = {"run_dir": str(run_dir), "device": dev,
               "score": bench(score, args.repeats), "generate": bench(gen, args.repeats)}
        results["models"][label] = res
        print(f"{label:12s} {dev:4s} score {res['score']['tokens_per_s']:9.0f} tok/s | "
              f"generate {res['generate']['tokens_per_s']:7.1f} tok/s")
        if dev != "cpu":
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    save_json(results, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
