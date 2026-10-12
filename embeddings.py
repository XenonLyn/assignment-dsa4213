"""Part II: build fixed input-embedding tables aligned with the LM vocabulary.

  python embeddings.py w2v   --out emb/w2v_sg300.pt     # Skip-gram on the LM training split
  python embeddings.py glove --out emb/glove6b300.pt    # public GloVe 6B 300d (gensim-data)

Both tables are [V, 300] tensors in vocab order (row i = vocab.itos[i]), ready for
`train_neural.py --embedding_path ... --freeze_embeddings`.

Rescaling: each table is multiplied by ONE scalar so that its mean row L2 norm
(over all rows except <pad>) equals that of the end-to-end learned table of the
reference LM (--ref_ckpt, Part I LSTM). A global scalar leaves cosine geometry
unchanged; it only puts every setting on the same input scale for the shared
LSTM initialisation and learning rate. <pad> is always the zero vector.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import (PAD, SPECIALS, hardware_info, load_corpus, save_json,
                    set_seed, tokenize_line)

# WikiText splits "-", "," and "." inside words into these tokens; GloVe has the plain symbols.
GLOVE_ALIASES = {"@-@": "-", "@,@": ",", "@.@": "."}


def mean_row_norm(E: torch.Tensor):
    rows = torch.ones(len(E), dtype=torch.bool)
    rows[PAD] = False
    return float(E[rows].norm(dim=1).mean())


def rescale(E: torch.Tensor, ref_ckpt):
    raw = mean_row_norm(E)
    sd = torch.load(ref_ckpt, map_location="cpu", weights_only=True)
    target = mean_row_norm(sd["embedding.weight"])
    E = E * (target / raw)
    E[PAD] = 0.0
    return E, {"raw_mean_row_norm": raw, "target_mean_row_norm": target,
               "scale": target / raw, "ref_ckpt": str(ref_ckpt)}


def train_sentences(data_dir, lowercase=True):
    """One sentence per non-empty line, exactly the LM's tokens (incl. <eos>)."""
    with open(Path(data_dir) / "train.txt", encoding="utf-8") as f:
        return [t for t in (tokenize_line(l, lowercase) for l in f) if t]


def build_w2v(args, vocab, splits):
    from gensim.models import Word2Vec
    sents = train_sentences(args.data_dir)
    assert sum(map(len, sents)) == len(splits["train"])
    t0 = time.perf_counter()
    model = Word2Vec(sents, vector_size=args.dim, window=args.window, min_count=1,
                     sg=1, negative=args.negative, sample=args.sample,
                     epochs=args.epochs, workers=args.workers, seed=args.seed)
    train_time = time.perf_counter() - t0
    E = torch.zeros(len(vocab), args.dim)
    missing = []
    for i, w in enumerate(vocab.itos):
        if w in model.wv.key_to_index:
            E[i] = torch.from_numpy(model.wv[w].copy())
        elif i != PAD:
            missing.append(w)
    assert missing == [], missing  # LM vocab is built from the same tokens, min_freq 1
    stats = {"method": "word2vec skip-gram (gensim)",
             "params": {"dim": args.dim, "window": args.window, "negative": args.negative,
                        "sample": args.sample, "epochs": args.epochs, "min_count": 1,
                        "workers": args.workers, "seed": args.seed},
             "train_tokens": len(splits["train"]), "sentences": len(sents),
             "train_time_s": train_time, "device": "cpu",
             "coverage": {"types": 1.0, "missing": []},
             "note": "multi-threaded gensim training is not bit-reproducible"}
    return E, stats


def build_glove(args, vocab, splits):
    import gensim.downloader as api
    t0 = time.perf_counter()
    kv = api.load(args.glove_name)
    load_time = time.perf_counter() - t0
    G = kv.vectors.astype(np.float32)
    mu, sigma = G.mean(0), G.std(0)
    rng = np.random.default_rng(args.seed)

    E = torch.zeros(len(vocab), kv.vector_size)
    found = np.zeros(len(vocab), dtype=bool)
    aliased, missing = [], []
    for i, w in enumerate(vocab.itos):
        if i == PAD:
            continue
        key = GLOVE_ALIASES.get(w, w)
        if key in kv.key_to_index:
            E[i] = torch.from_numpy(kv[key].copy())
            found[i] = True
            if key != w:
                aliased.append(w)
        else:  # missing word: random vector with GloVe's per-dimension mean / std
            E[i] = torch.from_numpy((mu + sigma * rng.standard_normal(kv.vector_size))
                                    .astype(np.float32))
            missing.append(w)

    n_words = len(vocab) - len(SPECIALS)
    found_words = int(found[len(SPECIALS):].sum())
    token_cov = {k: float(found[v.numpy()].mean()) for k, v in splits.items()}
    train_counts = np.bincount(splits["train"].numpy(), minlength=len(vocab))
    top_missing = sorted(missing, key=lambda w: -train_counts[vocab.stoi[w]])[:30]
    stats = {"method": f"GloVe ({args.glove_name}): Wikipedia 2014 + Gigaword 5, "
                       "6B tokens, 400k uncased vocab",
             "glove_vocab": len(kv.key_to_index), "load_time_s": load_time,
             "aliases": GLOVE_ALIASES, "aliased_tokens": aliased,
             "coverage": {
                 "types_found": found_words, "types_total": n_words,
                 "type_coverage": found_words / n_words,
                 "token_coverage": token_cov,  # fraction of tokens with a GloVe row
                 "missing_types": len(missing),
                 "top_missing_by_train_freq": [(w, int(train_counts[vocab.stoi[w]]))
                                               for w in top_missing]},
             "missing_init": "N(mu_d, sigma_d) per dimension from all GloVe vectors, "
                             f"seed {args.seed}; <pad> = 0; <unk>, <eos> are missing words"}
    return E, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["w2v", "glove"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default="data/wikitext-2")
    ap.add_argument("--ref_ckpt", default="runs/lstm_lr3e-3/best.pt")
    ap.add_argument("--no_rescale", action="store_true")
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--negative", type=int, default=5)
    ap.add_argument("--sample", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--glove_name", default="glove-wiki-gigaword-300")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    vocab, splits, _ = load_corpus(args.data_dir)
    E, stats = (build_w2v if args.method == "w2v" else build_glove)(args, vocab, splits)
    if not args.no_rescale:
        E, stats["rescale"] = rescale(E, args.ref_ckpt)
    E[PAD] = 0.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(E, out)
    vocab.save(out.with_suffix(".vocab.json"))
    stats.update({"shape": list(E.shape), "hardware": hardware_info(), "args": vars(args)})
    save_json(stats, out.with_suffix(".stats.json"))
    print({k: v for k, v in stats.items() if k in ("coverage", "train_time_s", "load_time_s",
                                                   "rescale")})


if __name__ == "__main__":
    main()
