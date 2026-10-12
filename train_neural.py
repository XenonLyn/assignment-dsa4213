"""Train and evaluate RNN / LSTM / Transformer language models (Part I).

Examples:
  python train_neural.py --model rnn         --out_dir runs/rnn
  python train_neural.py --model lstm        --out_dir runs/lstm
  python train_neural.py --model transformer --out_dir runs/transformer --lr 5e-4

Part II hook: --embedding_path emb.pt (a [V, emb_dim] tensor aligned with
vocab.json) and --freeze_embeddings.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (IGNORE_INDEX, PAD, UNK, Timer, detokenize, get_device,
                    hardware_info, load_corpus, load_prompts, make_chunks,
                    sample_next, save_json, set_seed, sync, tokenize_line)
from models import build_model, check_causality, check_dropout_and_eval, count_params


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rnn", "lstm", "transformer"])
    ap.add_argument("--data_dir", default="data/wikitext-2")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_train_tokens", type=int, default=None)
    ap.add_argument("--no_lowercase", action="store_true")
    # architecture
    ap.add_argument("--emb_dim", type=int, default=300)  # matches GloVe-300d for Part II
    ap.add_argument("--hidden_dim", type=int, default=512)  # RNN / LSTM
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=6)  # Transformer
    ap.add_argument("--ff_dim", type=int, default=1200)  # Transformer
    ap.add_argument("--dropout", type=float, default=0.3)
    # training
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--eval_seq_len", type=int, default=128)  # keep equal across models!
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    # embeddings (Part II)
    ap.add_argument("--embedding_path", default=None)
    ap.add_argument("--freeze_embeddings", action="store_true")
    # generation / benchmark
    ap.add_argument("--prompts", default="prompts.txt")
    ap.add_argument("--gen_len", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--gen_seed", type=int, default=1234)
    ap.add_argument("--allow_unk", action="store_true")
    ap.add_argument("--eval_batch_size", type=int, default=32)
    args = ap.parse_args()
    args.out_dir = args.out_dir or f"runs/{args.model}"
    return args


def lr_lambda(step, warmup, total):
    """Linear warm-up then cosine decay to 10% of the peak LR."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, X, Y, batch_size, device):
    """Token-level average NLL; padded targets (IGNORE_INDEX) are excluded."""
    model.eval()
    nll, n = 0.0, 0
    for i in range(0, len(X), batch_size):
        x, y = X[i: i + batch_size].to(device), Y[i: i + batch_size].to(device)
        logits, _ = model(x)
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=IGNORE_INDEX, reduction="sum").item()
        n += (y != IGNORE_INDEX).sum().item()
    return nll / n, n


@torch.no_grad()
def generate(model, prompt_ids, n_tokens, temperature, top_k, rng, banned, device):
    """Recurrent models carry their hidden state token by token; the Transformer
    re-reads its (cropped) window every step (no KV cache)."""
    model.eval()
    seq = list(prompt_ids) if prompt_ids else [2]  # 2 = <eos> as a neutral start
    if model.is_recurrent:
        logits, state = model(torch.tensor([seq], device=device))
        for _ in range(n_tokens):
            logp = F.log_softmax(logits[0, -1].float(), -1).cpu().numpy()
            nxt = sample_next(logp, temperature, top_k, rng, banned)
            seq.append(nxt)
            logits, state = model(torch.tensor([[nxt]], device=device), state)
    else:
        for _ in range(n_tokens):
            window = seq[-model.max_len:]
            logits, _ = model(torch.tensor([window], device=device))
            logp = F.log_softmax(logits[0, -1].float(), -1).cpu().numpy()
            seq.append(sample_next(logp, temperature, top_k, rng, banned))
    return seq[len(prompt_ids) if prompt_ids else 1:]


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")

    vocab, splits, data_stats = load_corpus(args.data_dir, not args.no_lowercase,
                                            max_train_tokens=args.max_train_tokens)
    vocab.save(out / "vocab.json")
    V = len(vocab)
    print("[data]", data_stats)
    X_tr, Y_tr = make_chunks(splits["train"], args.seq_len)
    X_va, Y_va = make_chunks(splits["valid"], args.eval_seq_len)
    X_te, Y_te = make_chunks(splits["test"], args.eval_seq_len)

    model = build_model(args.model, V, args).to(device)
    if args.embedding_path:
        emb = torch.load(args.embedding_path)
        assert emb.shape == model.embedding.weight.shape, (emb.shape, model.embedding.weight.shape)
        model.embedding.weight.data.copy_(emb.to(device))
    if args.freeze_embeddings:
        model.embedding.weight.requires_grad_(False)
    assert check_causality(model, V, device), "model can see future tokens!"
    checks = check_dropout_and_eval(model, V, device)
    print("[checks]", checks)
    assert all(checks.values()), f"dropout / eval-mode check failed on {device}: {checks}"
    params = count_params(model)
    print("[params]", params)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(X_tr) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup_steps, total_steps))

    history = {"step": [], "epoch": []}
    best_val, best_epoch, train_time, step = float("inf"), -1, 0.0, 0
    g = torch.Generator().manual_seed(args.seed)
    wall0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=g)
        tok_loss, tok_n, grad_norms, nonfinite = 0.0, 0, [], 0
        sync(device)
        t0 = time.perf_counter()
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size: (b + 1) * args.batch_size]
            x, y = X_tr[idx].to(device), Y_tr[idx].to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                   ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
            if not math.isfinite(gn):
                nonfinite += 1
                opt.zero_grad(set_to_none=True)  # skip the bad update
            else:
                opt.step()
            sched.step()
            step += 1
            n_tok = (y != IGNORE_INDEX).sum().item()
            tok_loss += loss.item() * n_tok
            tok_n += n_tok
            grad_norms.append(gn)
            if step % args.log_interval == 0:
                history["step"].append({"step": step, "train_loss": loss.item(),
                                        "grad_norm": gn, "lr": sched.get_last_lr()[0]})
                print(f"  ep {epoch} step {step}/{total_steps} loss {loss.item():.3f} "
                      f"gnorm {gn:.2f}")
        sync(device)
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        val_loss, _ = evaluate(model, X_va, Y_va, args.eval_batch_size, device)
        gns = np.array([x for x in grad_norms if math.isfinite(x)])
        rec = {"epoch": epoch, "train_loss": tok_loss / tok_n, "val_loss": val_loss,
               "val_ppl": math.exp(val_loss), "epoch_time_s": epoch_time,
               "grad_norm_mean": float(gns.mean()) if len(gns) else None,
               "grad_norm_max": float(gns.max()) if len(gns) else None,
               "nonfinite_steps": nonfinite}
        history["epoch"].append(rec)
        print(f"[epoch {epoch}] train {rec['train_loss']:.4f} | val {val_loss:.4f} "
              f"(ppl {rec['val_ppl']:.1f}) | {epoch_time:.0f}s")
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), out / "best.pt")
        save_json(history, out / "history.json")
    total_wall = time.perf_counter() - wall0

    # ---------------- test + common inference workload (best checkpoint)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    evaluate(model, X_te[:2], Y_te[:2], args.eval_batch_size, device)  # warm-up
    with Timer(device) as t_test:
        test_loss, n_test = evaluate(model, X_te, Y_te, args.eval_batch_size, device)
    print(f"[test] loss {test_loss:.4f} ppl {math.exp(test_loss):.2f}")

    banned = (PAD,) if args.allow_unk else (PAD, UNK)
    gens, gen_tokens, gen_time = [], 0, 0.0
    for i, prompt in enumerate(load_prompts(args.prompts)):
        p_ids = vocab.encode(tokenize_line(prompt, not args.no_lowercase, add_eos=False))
        rng = np.random.default_rng(args.gen_seed + i)  # same seeds for every model
        with Timer(device) as t_gen:
            new = generate(model, p_ids, args.gen_len, args.temperature, args.top_k,
                           rng, banned, device)
        gen_time += t_gen.elapsed
        gen_tokens += len(new)
        gens.append({"prompt": prompt, "continuation": detokenize(vocab.decode(new))})
        print(f"\n>>> {prompt}\n{gens[-1]['continuation']}")

    results = {
        "model": args.model,
        "args": vars(args),
        "data": data_stats,
        "hardware": hardware_info(),
        "device": str(device),
        "size": params,
        "train_time_s": train_time,          # training loops only, synced
        "total_wall_s": total_wall,          # incl. validation + checkpointing
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
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
