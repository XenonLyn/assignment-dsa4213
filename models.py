"""Neural next-token language models (Part I, models 2-4).

All three models:
  - randomly initialised, trainable input embeddings (Part II can swap these)
  - untied output layer, so freezing the input embeddings in Part II does not
    freeze the softmax layer as well
  - `hidden_states(x)` returns [B, T, H] token states for Part III
"""
import torch
import torch.nn as nn

from common import PAD


class RecurrentLM(nn.Module):
    is_recurrent = True

    def __init__(self, vocab_size, emb_dim=300, hidden_dim=512, n_layers=2,
                 dropout=0.3, cell="lstm"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        if cell == "rnn":
            self.rnn = nn.RNN(emb_dim, hidden_dim, n_layers, nonlinearity="tanh",
                              batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        elif cell == "lstm":
            # One single-layer nn.LSTM per layer with an explicit nn.Dropout in between.
            # PyTorch 2.5's MPS kernel for a multi-layer nn.LSTM does not resample its
            # built-in inter-layer dropout and can reuse the train-mode graph in eval
            # mode (wrong validation/test loss); see check_dropout_and_eval below.
            self.layers = nn.ModuleList(
                nn.LSTM(emb_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True)
                for i in range(n_layers))
        else:
            raise ValueError(cell)
        self.cell = cell
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def hidden_states(self, x, state=None):
        h = self.drop(self.embedding(x))
        if self.cell == "rnn":
            return self.rnn(h, state)
        new_state = []  # LSTM state: one (h, c) tuple per layer
        for i, layer in enumerate(self.layers):
            if i > 0:
                h = self.drop(h)
            h, s = layer(h, None if state is None else state[i])
            new_state.append(s)
        return h, new_state

    def forward(self, x, state=None):
        out, state = self.hidden_states(x, state)
        return self.head(self.drop(out)), state


class TransformerLM(nn.Module):
    """Decoder-only Transformer (pre-LN) with learned positions and a causal mask."""
    is_recurrent = False

    def __init__(self, vocab_size, emb_dim=300, n_layers=4, n_heads=6, ff_dim=1200,
                 dropout=0.2, max_len=256):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, emb_dim)
        layer = nn.TransformerEncoderLayer(emb_dim, n_heads, ff_dim, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(emb_dim, vocab_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD].zero_()

    def hidden_states(self, x, state=None):
        T = x.size(1)
        assert T <= self.max_len, f"sequence length {T} > max_len {self.max_len}"
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.embedding(x) + self.pos(pos))
        # position i may attend to positions <= i only. Padding sits only at the
        # END of a chunk, so real tokens never attend to it -> no key padding mask.
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.norm(h), None

    def forward(self, x, state=None):
        h, _ = self.hidden_states(x)
        return self.head(self.drop(h)), None


def build_model(name, vocab_size, args):
    if name in ("rnn", "lstm"):
        return RecurrentLM(vocab_size, args.emb_dim, args.hidden_dim, args.n_layers,
                           args.dropout, cell=name)
    if name == "transformer":
        return TransformerLM(vocab_size, args.emb_dim, args.n_layers, args.n_heads,
                             args.ff_dim, args.dropout,
                             max_len=max(args.seq_len, args.eval_seq_len))
    raise ValueError(name)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.embedding.weight.numel()
    head = sum(p.numel() for p in model.head.parameters())
    return {"total": total, "trainable": trainable, "input_embedding": emb,
            "output_layer": head, "body": total - emb - head,
            "size_mb_fp32": total * 4 / 2**20}


@torch.no_grad()
def check_dropout_and_eval(model, vocab_size, device, B=4, T=16):
    """Train mode must resample dropout on every call, and eval mode must match a CPU
    reference even right after a train-mode forward of the same shape (guards
    against the MPS multi-layer nn.LSTM bug described in RecurrentLM)."""
    import copy
    was_training = model.training
    x = torch.randint(3, vocab_size, (B, T), device=device)
    model.train()
    a, _ = model(x)
    b, _ = model(x)
    resampled = not torch.allclose(a, b)
    model.eval()
    e, _ = model(x)
    ref = copy.deepcopy(model).cpu().eval()
    r, _ = ref(x.cpu())
    eval_ok = torch.allclose(e.float().cpu(), r, atol=1e-3, rtol=1e-3)
    model.train(was_training)
    return {"dropout_resampled": bool(resampled), "eval_matches_cpu": bool(eval_ok)}


@torch.no_grad()
def check_causality(model, vocab_size, device, T=16):
    """Changing tokens after position t must not change the logits at <= t."""
    was_training = model.training
    model.eval()
    x = torch.randint(3, vocab_size, (2, T), device=device)
    y = x.clone()
    y[:, T // 2:] = torch.randint(3, vocab_size, (2, T - T // 2), device=device)
    a, _ = model(x)
    b, _ = model(y)
    ok = torch.allclose(a[:, : T // 2], b[:, : T // 2], atol=1e-4)
    model.train(was_training)
    return bool(ok)
