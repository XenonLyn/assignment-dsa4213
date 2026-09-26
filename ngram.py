"""Interpolated Kneser-Ney n-gram language model (Part I, model 1).

P_KN(w | h) = max(c(h,w) - D, 0) / c(h)  +  D * N1+(h,.) / c(h) * P_KN(w | h')
  - highest order in use: raw counts c(.)
  - lower orders: continuation counts N1+(., h', w)
  - base case: uniform 1/V, so every word in the vocabulary has P > 0
  - one discount per order and count type, D = n1 / (n1 + 2 n2)  (Ney et al.)
  - unseen history at some order -> that level is skipped (pure back-off)

Usage:
  python ngram.py --max_order 4 --out_dir runs/ngram
"""
import argparse
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (EOS, UNK, Timer, detokenize, hardware_info, load_corpus,
                    load_prompts, sample_next, save_json, set_seed, tokenize_line)


class KneserNeyLM:
    def __init__(self, max_order: int, vocab_size: int):
        assert max_order >= 1
        self.N = max_order
        self.V = vocab_size

    # ------------------------------------------------------------------ fit
    def fit(self, ids):
        N = self.N
        # raw[k][ctx][w] = count of the k-gram (ctx, w), len(ctx) = k-1
        self.raw = {k: defaultdict(dict) for k in range(1, N + 1)}
        for k in range(1, N + 1):
            table = self.raw[k]
            for i in range(len(ids) - k + 1):
                ctx = tuple(ids[i: i + k - 1])
                w = ids[i + k - 1]
                d = table[ctx]
                d[w] = d.get(w, 0) + 1
        # cont[k][ctx][w] = N1+(., ctx, w): distinct left extensions of (ctx, w)
        self.cont = {k: defaultdict(dict) for k in range(1, N)}
        for k in range(1, N):
            table = self.cont[k]
            for full_ctx, followers in self.raw[k + 1].items():
                ctx = full_ctx[1:]
                d = table[ctx]
                for w in followers:
                    d[w] = d.get(w, 0) + 1
        # convert to plain dicts + precompute (denominator, #types) per context
        self.raw = {k: dict(v) for k, v in self.raw.items()}
        self.cont = {k: dict(v) for k, v in self.cont.items()}
        self.raw_stats = {k: {c: (sum(d.values()), len(d)) for c, d in t.items()}
                          for k, t in self.raw.items()}
        self.cont_stats = {k: {c: (sum(d.values()), len(d)) for c, d in t.items()}
                           for k, t in self.cont.items()}
        self.raw_D = {k: self._discount(t) for k, t in self.raw.items()}
        self.cont_D = {k: self._discount(t) for k, t in self.cont.items()}
        return self

    @staticmethod
    def _discount(table):
        n1 = n2 = 0
        for d in table.values():
            for c in d.values():
                n1 += c == 1
                n2 += c == 2
        return n1 / (n1 + 2 * n2) if n1 + n2 > 0 else 0.75

    def num_entries(self):
        """Stored (context, word) entries - our 'model size' for the n-gram."""
        raw = sum(len(d) for t in self.raw.values() for d in t.values())
        cont = sum(len(d) for t in self.cont.values() for d in t.values())
        return {"raw_entries": raw, "continuation_entries": cont, "total": raw + cont}

    # ------------------------------------------------------------ helpers
    def _levels(self, history, order):
        """Yield (table, stats, D, ctx) from order 1 up to the top order in use."""
        h = tuple(history[len(history) - (order - 1):]) if order > 1 else ()
        top = len(h) + 1  # shorter history (e.g. chunk start) -> lower top order
        for k in range(1, top + 1):
            ctx = h[len(h) - (k - 1):]
            if k == top:
                yield self.raw[k], self.raw_stats[k], self.raw_D[k], ctx
            else:
                yield self.cont[k], self.cont_stats[k], self.cont_D[k], ctx

    def prob(self, w, history, order=None):
        order = order or self.N
        p = 1.0 / self.V
        for table, stats, D, ctx in self._levels(history, order):
            st = stats.get(ctx)
            if st is None:
                continue
            denom, ntypes = st
            c = table[ctx].get(w, 0)
            p = max(c - D, 0.0) / denom + D * ntypes / denom * p
        return p

    def distribution(self, history, order=None):
        order = order or self.N
        p = np.full(self.V, 1.0 / self.V)
        for table, stats, D, ctx in self._levels(history, order):
            st = stats.get(ctx)
            if st is None:
                continue
            denom, ntypes = st
            foll = table[ctx]
            idx = np.fromiter(foll.keys(), dtype=np.int64, count=len(foll))
            cnt = np.fromiter(foll.values(), dtype=np.float64, count=len(foll))
            p = (D * ntypes / denom) * p
            p[idx] += np.maximum(cnt - D, 0.0) / denom
        return p

    # ----------------------------------------------------------- scoring
    def score_chunks(self, ids, seq_len, order=None):
        """Average NLL under the common chunked evaluation (see common.py)."""
        order = order or self.N
        n_targets = len(ids) - 1
        nll, n = 0.0, 0
        for s in range(0, n_targets, seq_len):
            e = min(s + seq_len, n_targets)
            for t in range(s + 1, e + 1):  # t = index of target token
                hist = ids[max(s, t - (order - 1)): t]
                nll -= math.log(self.prob(ids[t], hist, order))
                n += 1
        return nll / n, n

    def generate(self, prompt_ids, n_tokens, temperature, top_k, rng,
                 banned, order=None):
        order = order or self.N
        seq = list(prompt_ids) if prompt_ids else [EOS]
        for _ in range(n_tokens):
            p = self.distribution(seq[-(order - 1):] if order > 1 else [], order)
            with np.errstate(divide="ignore"):
                seq.append(sample_next(np.log(p), temperature, top_k, rng, banned))
        return seq[len(prompt_ids) if prompt_ids else 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/wikitext-2")
    ap.add_argument("--out_dir", default="runs/ngram")
    ap.add_argument("--max_order", type=int, default=4)
    ap.add_argument("--eval_seq_len", type=int, default=128)
    ap.add_argument("--max_train_tokens", type=int, default=None)
    ap.add_argument("--no_lowercase", action="store_true")
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--gen_len", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--gen_seed", type=int, default=1234)
    ap.add_argument("--allow_unk", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab, splits, data_stats = load_corpus(args.data_dir, not args.no_lowercase,
                                            max_train_tokens=args.max_train_tokens)
    vocab.save(out / "vocab.json")
    print("[data]", data_stats)
    train, valid, test = (splits[k].tolist() for k in ("train", "valid", "test"))

    with Timer() as t_fit:
        lm = KneserNeyLM(args.max_order, len(vocab)).fit(train)
    print(f"[fit] {t_fit.elapsed:.1f}s, entries={lm.num_entries()}")

    # choose the order on the validation set (never on test)
    val_by_order = {}
    for order in range(1, args.max_order + 1):
        loss, _ = lm.score_chunks(valid, args.eval_seq_len, order)
        val_by_order[order] = {"loss": loss, "ppl": math.exp(loss)}
        print(f"[valid] order={order} loss={loss:.4f} ppl={math.exp(loss):.2f}")
    best_order = min(val_by_order, key=lambda o: val_by_order[o]["loss"])

    with Timer() as t_test:
        test_loss, n_test = lm.score_chunks(test, args.eval_seq_len, best_order)
    print(f"[test] order={best_order} loss={test_loss:.4f} ppl={math.exp(test_loss):.2f}")

    # generation with the same decoding settings as the neural models
    banned = (0,) if args.allow_unk else (0, UNK)
    gens, gen_tokens, gen_time = [], 0, 0.0
    for i, prompt in enumerate(load_prompts(args.prompts)):
        p_ids = vocab.encode(tokenize_line(prompt, not args.no_lowercase, add_eos=False))
        rng = np.random.default_rng(args.gen_seed + i)
        with Timer() as t_gen:
            new = lm.generate(p_ids, args.gen_len, args.temperature, args.top_k,
                              rng, banned, best_order)
        gen_time += t_gen.elapsed
        gen_tokens += len(new)
        gens.append({"prompt": prompt, "continuation": detokenize(vocab.decode(new))})

    with open(out / "model.pkl", "wb") as f:
        pickle.dump(lm, f)
    results = {
        "model": f"ngram (interpolated Kneser-Ney, order={best_order})",
        "args": vars(args),
        "data": data_stats,
        "hardware": hardware_info(),
        "device": "cpu",
        "size": lm.num_entries(),
        "model_file_mb": (out / "model.pkl").stat().st_size / 2**20,
        "discounts": {"raw": lm.raw_D, "cont": lm.cont_D},
        "valid_by_order": val_by_order,
        "best_order": best_order,
        "train_time_s": t_fit.elapsed,
        "test_loss": test_loss,
        "test_ppl": math.exp(test_loss),
        "test_tokens": n_test,
        "inference": {
            "score_time_s": t_test.elapsed,
            "score_tokens_per_s": n_test / t_test.elapsed,
            "gen_tokens_per_s": gen_tokens / gen_time if gen_time else None,
        },
        "generations": gens,
    }
    save_json(results, out / "results.json")
    print(f"[done] results saved to {out/'results.json'}")


if __name__ == "__main__":
    main()
