"""Part III: AG News topic classification with frozen LM representations.

  python downstream.py --runs rnn=runs/rnn lstm=runs/lstm_lr3e-3 \
      lstm_w2v=runs/lstm_w2v_frozen lstm_glove=runs/lstm_glove_frozen transformer=runs/transformer

Every language model stays frozen (eval mode, no dropout); only a classifier is trained.
Each document becomes a fixed-length vector in two ways:
  mean  = average of the LM's top-layer token states over the document's real tokens (no padding)
  last  = top-layer state at the document's last token (<eos>); with causal models this is the
          only position that has read the whole document
Context-free baselines: the average of the LM's input-embedding rows (learned / Skip-gram / GloVe)
and a TF-IDF bag of words. Classifier for all: standardised features + multinomial logistic
regression, C chosen on a held-out validation split; metric: test accuracy and macro-F1.
"""
import argparse
import html
import json
import re
import time
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import EOS, PAD, UNK, Vocab, get_device, hardware_info, save_json, set_seed
from models import build_model

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
CS = [0.01, 0.1, 1.0, 10.0]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_agnews(data_dir: Path):
    """AG News (fancyzhx/ag_news on the HuggingFace Hub), cached as jsonl."""
    if not (data_dir / "train.jsonl").exists():
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news")
        data_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            with open(data_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for ex in ds[split]:
                    f.write(json.dumps({"text": ex["text"], "label": ex["label"]}) + "\n")
    out = {}
    for split in ("train", "test"):
        with open(data_dir / f"{split}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        out[split] = ([r["text"] for r in rows], np.array([r["label"] for r in rows]))
    return out


def clean(text):
    text = re.sub(r"#(\d+);", lambda m: chr(int(m.group(1))), text)  # "#39;" -> "'"
    text = re.sub(r"(?<![&\w])(quot|amp|lt|gt);", r"&\1;", text)    # "quot;" -> "&quot;"
    text = html.unescape(text)
    text = re.sub(r"<[^>]*>", " ", text)                             # stray HTML tags (incl. long hrefs)
    return text.replace("\\", " ")


TOKEN_RE = re.compile(r"@[-,.]@|(?:\w\.){2,}|'(?:s|t|re|ve|ll|d|m)\b|\w+|[^\w\s]")


def tokenize(text):
    """Mimic WikiText-2's tokenisation (lowercased like our LM corpus):
    1,000 -> 1 @,@ 000 ; 3.5 -> 3 @.@ 5 ; long-term -> long @-@ term ;
    wasn't -> wasn 't ; game's -> game 's ; punctuation and $ % ( ) " as separate tokens ;
    abbreviations such as u.s. stay one token."""
    t = clean(text).lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", " @,@ ", t)
    t = re.sub(r"(?<=\d)\.(?=\d)", " @.@ ", t)
    t = re.sub(r"(?<=\w)-(?=\w)", " @-@ ", t)
    t = re.sub(r"(\w)'(s|t|re|ve|ll|d|m)\b", r"\1 '\2", t)
    return TOKEN_RE.findall(t)


def stratified_split(y, n_train, n_val, seed):
    """Disjoint, class-balanced train / validation indices from the official training set.
    The train indices are ordered so that every prefix of k * n_classes is balanced too
    (used for the low-resource learning curve)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_tr, per_va = n_train // len(classes), n_val // len(classes)
    tr, va = [], []
    for c in classes:
        idx = rng.permutation(np.where(y == c)[0])
        tr.append(idx[:per_tr])
        va.append(idx[per_tr: per_tr + per_va])
    tr = np.stack(tr, 1).reshape(-1)  # interleave classes: c0, c1, c2, c3, c0, ...
    return tr, np.concatenate(va)


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #
def encode(tok_docs, vocab, max_len):
    """LM ids: tokens (truncated) + <eos>, exactly like a line of the LM corpus."""
    ids = [vocab.encode(t[: max_len - 1]) + [EOS] for t in tok_docs]
    return ids


@torch.no_grad()
def lm_features(model, ids, device, bs=128):
    model.eval()
    order = np.argsort([len(x) for x in ids])  # length-sorted batches, less padding
    H = None
    for s in range(0, len(ids), bs):
        b = order[s: s + bs]
        lens = torch.tensor([len(ids[i]) for i in b])
        x = torch.full((len(b), int(lens.max())), PAD, dtype=torch.long)
        for j, i in enumerate(b):
            x[j, : lens[j]] = torch.tensor(ids[i])
        h, _ = model.hidden_states(x.to(device))           # [B, T, H], padding only at the end
        mask = (torch.arange(x.size(1))[None] < lens[:, None]).to(device)
        mean = (h * mask[..., None]).sum(1) / lens.to(device)[:, None]
        last = h[torch.arange(len(b), device=device), (lens - 1).to(device)]
        if H is None:
            H = {"mean": np.zeros((len(ids), h.size(-1)), np.float32),
                 "last": np.zeros((len(ids), h.size(-1)), np.float32)}
        H["mean"][b] = mean.float().cpu().numpy()
        H["last"][b] = last.float().cpu().numpy()
    return H


def emb_mean(E, ids):
    E = E.numpy() if torch.is_tensor(E) else E
    return np.stack([E[x].mean(0) for x in ids]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Classifier (identical for every representation)
# --------------------------------------------------------------------------- #
def fit_select(Xtr, ytr, Xva, yva, sparse=False, seed=42):
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
        if not sparse:
            clf = make_pipeline(StandardScaler(), clf)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yva, clf.predict(Xva))
        if best is None or acc > best[0]:
            best = (acc, C, clf)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=run_dir (LMs)")
    ap.add_argument("--data_dir", default="data/ag_news")
    ap.add_argument("--out", default="runs/part3")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--max_len", type=int, default=128)  # = the LMs' context length
    ap.add_argument("--curve", type=int, nargs="+", default=[200, 1000, 5000, 20000])
    ap.add_argument("--glove", default="emb/glove6b300.pt")
    ap.add_argument("--w2v", default="emb/w2v_sg300.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    # ---------------- data
    data = load_agnews(Path(args.data_dir))
    X_text, y_all = data["train"]
    te_text, y_te = data["test"]
    tr_idx, va_idx = stratified_split(y_all, args.n_train, args.n_val, args.seed)
    texts = {"train": [X_text[i] for i in tr_idx], "val": [X_text[i] for i in va_idx],
             "test": te_text}
    ys = {"train": y_all[tr_idx], "val": y_all[va_idx], "test": y_te}
    toks = {k: [tokenize(t) for t in v] for k, v in texts.items()}

    runs = {}
    for spec in args.runs:
        label, _, d = spec.rpartition("=")
        runs[label] = Path(d)
    vocab = Vocab.load(next(iter(runs.values())) / "vocab.json")
    for d in runs.values():
        assert Vocab.load(d / "vocab.json").itos == vocab.itos, "all LMs must share the vocab"
    ids = {k: encode(v, vocab, args.max_len) for k, v in toks.items()}
    lens = {k: np.array([len(x) for x in v]) for k, v in ids.items()}
    unk_frac = {k: np.array([np.mean(np.array(x) == UNK) for x in v]) for k, v in ids.items()}
    data_stats = {
        "n": {k: len(v) for k, v in ids.items()},
        "class_counts": {k: np.bincount(v, minlength=4).tolist() for k, v in ys.items()},
        "tokens_per_doc_mean": {k: float(v.mean()) for k, v in lens.items()},
        "truncated_docs": {k: int(sum(len(t) > args.max_len - 1 for t in toks[k])) for k in toks},
        "unk_token_rate": {k: float(np.mean(np.concatenate(v) == UNK)) for k, v in ids.items()},
    }
    print("[data]", data_stats)
    print("[example]", texts["test"][0], "->", " ".join(toks["test"][0]))

    # ---------------- features
    feats, meta = {}, {}
    for label, d in runs.items():
        r = json.load(open(d / "results.json"))
        a = Namespace(**r["args"])
        model = build_model(a.model, len(vocab), a).to(device)
        model.load_state_dict(torch.load(d / "best.pt", map_location=device, weights_only=True))
        t0 = time.perf_counter()
        H = {k: lm_features(model, ids[k], device) for k in ids}
        t_ext = time.perf_counter() - t0
        for pool in ("mean", "last"):
            feats[f"{label}/{pool}"] = {k: H[k][pool] for k in ids}
            meta[f"{label}/{pool}"] = {"lm": label, "pool": pool, "lm_test_ppl": r["test_ppl"],
                                       "lm_test_ce": r["test_loss"], "dim": H["test"][pool].shape[1],
                                       "extract_time_s": t_ext}
        if label == "lstm":  # its own learned input-embedding table (context-free baseline)
            E_learned = model.embedding.weight.detach().cpu()
        print(f"[features] {label}: {t_ext:.1f}s")
        del model
    for name, E in [("emb_learned", E_learned if "lstm" in runs else None),
                    ("emb_w2v", torch.load(args.w2v)), ("emb_glove", torch.load(args.glove))]:
        if E is None:
            continue
        feats[f"{name}/mean"] = {k: emb_mean(E, ids[k]) for k in ids}
        meta[f"{name}/mean"] = {"lm": None, "pool": "mean of input embeddings", "dim": E.shape[1]}

    # ---------------- classifiers (main setting: full 20k train)
    results, preds = {}, {}
    for name, F in feats.items():
        acc_va, C, clf = fit_select(F["train"], ys["train"], F["val"], ys["val"])
        p = clf.predict(F["test"])
        preds[name] = p
        results[name] = {**meta[name], "C": C, "val_acc": acc_va,
                         "test_acc": accuracy_score(y_te, p),
                         "test_macro_f1": f1_score(y_te, p, average="macro"),
                         "per_class_f1": f1_score(y_te, p, average=None).tolist()}
        print(f"{name:22s} val {acc_va:.4f} test {results[name]['test_acc']:.4f} (C={C})")
    tfidf = TfidfVectorizer(analyzer=lambda d: d + [f"{a} {b}" for a, b in zip(d, d[1:])],
                            min_df=2, sublinear_tf=True)
    T = {k: (tfidf.fit_transform(toks[k]) if k == "train" else tfidf.transform(toks[k]))
         for k in ("train", "val", "test")}
    acc_va, C, clf = fit_select(T["train"], ys["train"], T["val"], ys["val"], sparse=True)
    p = clf.predict(T["test"])
    preds["tfidf_bow"] = p
    results["tfidf_bow"] = {"lm": None, "pool": "tf-idf uni+bigrams (original words, no <unk>)",
                            "dim": T["train"].shape[1], "C": C, "val_acc": acc_va,
                            "test_acc": accuracy_score(y_te, p),
                            "test_macro_f1": f1_score(y_te, p, average="macro"),
                            "per_class_f1": f1_score(y_te, p, average=None).tolist()}
    print(f"{'tfidf_bow':22s} val {acc_va:.4f} test {results['tfidf_bow']['test_acc']:.4f} (C={C})")

    # ---------------- low-resource learning curve (prefixes of the balanced train order)
    curve = {}
    for name, F in list(feats.items()) + [("tfidf_bow", None)]:
        curve[name] = {}
        for n in args.curve:
            if name == "tfidf_bow":
                vec = TfidfVectorizer(analyzer=tfidf.analyzer, min_df=1, sublinear_tf=True)
                Xtr = vec.fit_transform(toks["train"][:n])
                _, _, clf = fit_select(Xtr, ys["train"][:n], vec.transform(toks["val"]),
                                       ys["val"], sparse=True)
                pt = clf.predict(vec.transform(toks["test"]))
            else:
                _, _, clf = fit_select(F["train"][:n], ys["train"][:n], F["val"], ys["val"])
                pt = clf.predict(F["test"])
            curve[name][n] = accuracy_score(y_te, pt)
        print(f"[curve] {name:22s} " + " ".join(f"{n}:{v:.3f}" for n, v in curve[name].items()))

    # ---------------- analysis
    lm_names = [n for n in results if results[n].get("lm")]
    corr = {}
    for pool in ("mean", "last"):
        ns = [n for n in lm_names if results[n]["pool"] == pool]
        rho, pv = spearmanr([results[n]["lm_test_ppl"] for n in ns],
                            [results[n]["test_acc"] for n in ns])
        corr[pool] = {"spearman_ppl_vs_acc": rho, "p": pv, "n": len(ns)}
    best = max(lm_names, key=lambda n: results[n]["test_acc"])
    worst = min(lm_names, key=lambda n: results[n]["test_acc"])

    lines = ["| representation | LM test PPL | dim | C | val acc | **test acc** | test macro-F1 | "
             + " | ".join(f"F1 {l}" for l in LABELS) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(LABELS)]
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["test_acc"]):
        ppl = f"{r['lm_test_ppl']:.1f}" if r.get("lm") else "–"
        lines.append(f"| {n} | {ppl} | {r['dim']} | {r['C']} | {r['val_acc']:.4f} | "
                     f"{r['test_acc']:.4f} | {r['test_macro_f1']:.4f} | "
                     + " | ".join(f"{x:.3f}" for x in r["per_class_f1"]) + " |")
    lines += ["", "Spearman(LM test PPL, test accuracy) over the LMs: "
              + "; ".join(f"{p}: ρ = {c['spearman_ppl_vs_acc']:.2f} (n={c['n']}, p={c['p']:.2f})"
                          for p, c in corr.items())]
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cl = ["| train examples | " + " | ".join(map(str, args.curve)) + " |",
          "|---|" + "---|" * len(args.curve)]
    for n, c in sorted(curve.items(), key=lambda kv: -kv[1][args.curve[-1]]):
        cl.append(f"| {n} | " + " | ".join(f"{c[k]:.3f}" for k in args.curve) + " |")
    (out / "learning_curve.md").write_text("\n".join(cl) + "\n", encoding="utf-8")

    # accuracy vs <unk> share of the document
    bins = [(-1, 0.0, "0%"), (0.0, 0.05, "(0, 5%]"), (0.05, 0.10, "(5, 10%]"), (0.10, 1.0, "> 10%")]
    show = [best, worst, "tfidf_bow"]
    ub = ["| <unk> share of doc | share of test | " + " | ".join(show) + " |",
          "|---|---|" + "---|" * len(show)]
    for lo, hi, lab in bins:
        m = (unk_frac["test"] > lo) & (unk_frac["test"] <= hi)
        ub.append(f"| {lab} | {m.mean():.3f} | "
                  + " | ".join(f"{np.mean(preds[s][m] == y_te[m]):.3f}" for s in show) + " |")
    (out / "unk_bins.md").write_text("\n".join(ub) + "\n", encoding="utf-8")

    cm_lines = []
    for n in [best, worst, "tfidf_bow"]:
        cm = confusion_matrix(y_te, preds[n])
        cm_lines += [f"### {n} (rows = true, columns = predicted)", "",
                     "| | " + " | ".join(LABELS) + " |", "|---|" + "---|" * 4]
        cm_lines += [f"| {LABELS[i]} | " + " | ".join(map(str, cm[i])) + " |" for i in range(4)]
        cm_lines.append("")
    (out / "confusion.md").write_text("\n".join(cm_lines), encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    def examples(mask, k=8):
        idx = np.where(mask)[0]
        return rng.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else []
    all_lm_wrong = np.all([preds[n] != y_te for n in lm_names], axis=0)
    groups = [
        (f"best LM representation ({best}) wrong, TF-IDF right", (preds[best] != y_te) & (preds["tfidf_bow"] == y_te)),
        (f"best LM representation ({best}) right, TF-IDF wrong", (preds[best] == y_te) & (preds["tfidf_bow"] != y_te)),
        ("every LM representation wrong", all_lm_wrong),
    ]
    er = ["# Representative test errors\n"]
    for title, mask in groups:
        er.append(f"## {title} ({mask.sum()} docs)\n")
        er += ["| true | " + best + " | tfidf | <unk> share | text |", "|---|---|---|---|---|"]
        for i in examples(mask):
            t = te_text[i].replace("|", "/").replace("\n", " ")
            er.append(f"| {LABELS[y_te[i]]} | {LABELS[preds[best][i]]} | "
                      f"{LABELS[preds['tfidf_bow'][i]]} | {unk_frac['test'][i]:.2f} | {t[:220]} |")
        er.append("")
    (out / "errors.md").write_text("\n".join(er), encoding="utf-8")

    # figures: PPL vs accuracy, learning curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for pool, mk in (("mean", "o"), ("last", "s")):
        ns = [n for n in lm_names if results[n]["pool"] == pool]
        ax[0].scatter([results[n]["lm_test_ppl"] for n in ns], [results[n]["test_acc"] for n in ns],
                      marker=mk, label=f"{pool} pooling")
        for n in ns:
            ax[0].annotate(results[n]["lm"], (results[n]["lm_test_ppl"], results[n]["test_acc"]),
                           fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].axhline(results["tfidf_bow"]["test_acc"], ls="--", color="gray", label="TF-IDF")
    ax[0].set_xlabel("LM test perplexity (lower = better LM)")
    ax[0].set_ylabel("AG News test accuracy")
    ax[0].legend(fontsize=8)
    for n in [best, worst, "tfidf_bow", "emb_glove/mean", "emb_w2v/mean"]:
        if n in curve:
            ax[1].plot(args.curve, [curve[n][k] for k in args.curve], marker="o", label=n)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("labelled training examples")
    ax[1].set_ylabel("test accuracy")
    ax[1].legend(fontsize=7)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "ppl_vs_acc_and_curve.png", dpi=200)

    save_json({"data": data_stats, "results": results, "curve": curve, "spearman": corr,
               "best_lm_rep": best, "worst_lm_rep": worst, "args": vars(args),
               "hardware": hardware_info(), "device": str(device)}, out / "results.json")
    print("\n".join(lines))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
