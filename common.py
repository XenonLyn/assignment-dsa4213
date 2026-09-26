"""Shared utilities for DSA4213 Assignment 1, Part I.

Data: WikiText-2 (word level, already <unk>-ified by the dataset authors).
Preprocessing: lowercase (helps GloVe coverage in Part II), whitespace split,
<eos> appended to every non-empty line, empty lines dropped.

Common evaluation setup (used by ALL four models):
  the test token stream is split into non-overlapping chunks of `eval_seq_len`
  targets. Each chunk is scored independently: the model may only use tokens
  inside the same chunk that precede the target (no hidden state carried across
  chunks). The last chunk is padded; padded targets are excluded from the loss.
"""
import json
import math
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SPECIALS = ["<pad>", "<unk>", "<eos>"]
PAD, UNK, EOS = 0, 1, 2
IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Reproducibility / device / timing
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device):
    """MPS/CUDA run asynchronously; synchronise before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:  # macOS chip name, e.g. "Apple M1"
        info["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["cpu"] = platform.processor()
    return info


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download_wikitext2(data_dir: Path):
    """Download WikiText-2 (tokenised v1) from HuggingFace into plain text files."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, name in [("train", "train"), ("validation", "valid"), ("test", "test")]:
        with open(data_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            for line in ds[split]["text"]:
                f.write(line.rstrip("\n") + "\n")


def tokenize_line(line: str, lowercase: bool = True, add_eos: bool = True):
    line = line.strip()
    if not line:
        return []
    if lowercase:
        line = line.lower()
    toks = line.split()
    return toks + ["<eos>"] if add_eos else toks


def read_tokens(path: Path, lowercase: bool, max_tokens=None):
    toks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            toks.extend(tokenize_line(line, lowercase))
            if max_tokens is not None and len(toks) >= max_tokens:
                return toks[:max_tokens]
    return toks


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, tokens, min_freq: int = 1, max_size=None):
        counter = Counter(tokens)
        for s in SPECIALS:
            counter.pop(s, None)
        words = [w for w, c in sorted(counter.items(), key=lambda x: (-x[1], x[0]))
                 if c >= min_freq]
        if max_size is not None:
            words = words[: max_size - len(SPECIALS)]
        return cls(SPECIALS + words)

    def encode(self, toks):
        return [self.stoi.get(t, UNK) for t in toks]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        save_json(self.itos, path)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def load_corpus(data_dir="data/wikitext-2", lowercase=True, min_freq=1,
                max_train_tokens=None):
    """Returns vocab and train/valid/test id tensors. Vocab is built on train only."""
    data_dir = Path(data_dir)
    if not (data_dir / "train.txt").exists():
        print(f"[data] {data_dir}/train.txt not found, downloading WikiText-2 ...")
        download_wikitext2(data_dir)
    train_tok = read_tokens(data_dir / "train.txt", lowercase, max_train_tokens)
    valid_tok = read_tokens(data_dir / "valid.txt", lowercase)
    test_tok = read_tokens(data_dir / "test.txt", lowercase)
    vocab = Vocab.build(train_tok, min_freq=min_freq)

    def enc(toks):
        return torch.tensor(vocab.encode(toks), dtype=torch.long)

    splits = {"train": enc(train_tok), "valid": enc(valid_tok), "test": enc(test_tok)}
    stats = {k: {"tokens": len(v), "unk_rate": float((v == UNK).float().mean())}
             for k, v in splits.items()}
    stats["vocab_size"] = len(vocab)
    return vocab, splits, stats


def make_chunks(ids: torch.Tensor, seq_len: int):
    """Non-overlapping (input, target) chunks. Every token except the very first
    is a target exactly once. Padding: input=<pad>, target=IGNORE_INDEX."""
    n_targets = len(ids) - 1
    n_chunks = math.ceil(n_targets / seq_len)
    X = torch.full((n_chunks, seq_len), PAD, dtype=torch.long)
    Y = torch.full((n_chunks, seq_len), IGNORE_INDEX, dtype=torch.long)
    for i in range(n_chunks):
        s = i * seq_len
        e = min(s + seq_len, n_targets)
        X[i, : e - s] = ids[s:e]
        Y[i, : e - s] = ids[s + 1: e + 1]
    return X, Y


# --------------------------------------------------------------------------- #
# Decoding (identical for n-gram and neural models)
# --------------------------------------------------------------------------- #
def sample_next(logprobs: np.ndarray, temperature: float, top_k: int,
                rng: np.random.Generator, banned=(PAD,)):
    """Temperature + top-k sampling on a vector of log-probabilities.
    temperature == 0 -> greedy."""
    logits = np.asarray(logprobs, dtype=np.float64).copy()
    for b in banned:
        logits[b] = -np.inf
    if temperature == 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k and top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def detokenize(tokens):
    text = " ".join("\n" if t == "<eos>" else t for t in tokens)
    for a, b in [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" \n ", "\n")]:
        text = text.replace(a, b)
    return text


class Timer:
    def __init__(self, device=None):
        self.device = device

    def __enter__(self):
        if self.device is not None:
            sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None:
            sync(self.device)
        self.elapsed = time.perf_counter() - self.t0
