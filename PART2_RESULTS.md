# Part II — Input-embedding variants (draft for the report)

Architecture: the Part I **LSTM** (2 × 512, per-layer `nn.LSTM` with explicit dropout 0.3, untied
softmax), 300-d input embeddings. Everything except the input-embedding table is identical across the three
settings: data and preprocessing, vocabulary (28,913), lr 3e-3 (selected on validation for setting A in
Part I and reused unchanged), 10 epochs, batch 32 × 128, AdamW, schedule, clipping, seed 42, common chunked
evaluation, decoding. Setting A *is* the Part I LSTM run (`runs/lstm_lr3e-3`); B and C are the two
additional LM runs. Single seed per setting. Numbers from `runs/*/results.json`; analyses from
`analyze_part2.py` → `runs/part2/`.

| setting | input embedding | trainable? | trained on |
|---|---|---|---|
| A random | U(−0.1, 0.1) init | yes (jointly with the LM) | the LM objective, WikiText-2 train (2.08M tokens) |
| B own (Skip-gram) | word2vec Skip-gram, gensim | **no — entire table frozen** | WikiText-2 train only (same 2.08M tokens) |
| C public (GloVe) | GloVe 6B 300d (Wikipedia 2014 + Gigaword 5) | **no — entire table frozen** | ~6B tokens, i.e. ~2,900× our corpus |

## Embedding construction (`embeddings.py`)
* **Dimension 300**, chosen so that the public 300-d GloVe vectors can be used without projection.
* **Skip-gram (B)**: the exact token stream the LM sees (lowercased, `<eos>` per line, the dataset's
  `<unk>`), one sentence per line; window 5, 5 negative samples, subsampling 1e-3, min_count 1, 10 epochs,
  8 threads, seed 42. Coverage of the LM vocabulary: **100%** (built from the same tokens).
  **Cost (separate from LM training): 27–38 s on the M1 CPU** (two builds; multi-threaded gensim is not
  bit-reproducible), i.e. ~1% of one LM training run.
* **GloVe (C)**: `glove-wiki-gigaword-300` from gensim-data (394 MB download, 400k uncased words; loading
  33 s; no training by us — the cost of GloVe's own pre-training on 6B tokens is not ours but is far larger
  than everything in this assignment). WikiText's split-symbol tokens are mapped `@-@`→`-`, `@,@`→`,`,
  `@.@`→`.`.
  * Coverage: **96.2% of vocabulary types** (27,812 / 28,910 words); **token coverage 95.8% train /
    93.3% valid / 92.5% test**. Most uncovered tokens are the dataset's `<unk>` (2.6% of train tokens) and
    `<eos>` (1.1%); the remaining 1,098 missing types are rare names and symbols (`'t` 592×, `°` 258×,
    `−`, `'brien`, `×`, `magadheera`, …) and cover only **0.1%** of test inputs.
  * Missing words (incl. `<unk>`, `<eos>`) get a random vector N(μ_d, σ_d) with GloVe's per-dimension mean
    and std (seed 42); `<pad>` = 0. These rows are frozen too.
* **Scale**: each frozen table is multiplied by one scalar so that its mean row norm equals that of the
  end-to-end learned table of setting A (2.87; raw Skip-gram 2.29 → ×1.25, raw GloVe 6.48 → ×0.44).
  A global scalar keeps all cosine similarities; it only puts the three settings on the same input scale
  for the shared LSTM initialisation and learning rate.
* Freezing: `requires_grad=False` on the whole table, excluded from the optimizer (so also from weight
  decay). Trainable parameters drop from 27.27M to 18.60M; the model is otherwise identical.

## Results

| setting | test CE | **test PPL** | best val PPL | val PPL after epoch 1 / 2 / 3 | first epoch with val PPL < 200 | final train − val (nats) | grad-norm max | non-finite steps | LM train time |
|---|---|---|---|---|---|---|---|---|---|
| A random, trainable | 4.930 | 138.4 | 148.2 | 527 / 276 / 221 | 4 | −0.27 | 2.18 | 0 | 3,510 s |
| B Skip-gram, frozen | **4.688** | **108.6** | **114.4** | 452 / 195 / 151 | 2 | −0.47 | 1.95 | 0 | 3,255 s |
| C GloVe, frozen | 4.701 | 110.1 | 115.9 | 790 / 234 / 166 | 3 | −0.41 | 1.78 | 0 | 3,564 s |

(`runs/part2/curves.png`; all three best at epoch 10; a fresh CPU re-evaluation of every checkpoint
reproduces the logged test loss.)

**Prediction quality.** Both frozen pre-trained tables beat the jointly learned one by a wide margin
(−0.24 / −0.23 nats, i.e. 20–22% lower PPL). B and C are within 0.013 nats of each other — not distinguishable with
one seed.

**Convergence.** B is ahead from the first epoch. C starts *slowest* (val PPL 790 after epoch 1) but
overtakes A during epoch 2; its gap to B shrinks every epoch (0.56 → 0.09 → 0.045 → 0.013 nats after
epochs 1 / 3 / 5 / 10). A converges slowest after epoch 1 and is still clearly behind at epoch 10.
All three are still improving slightly at epoch 10 (by 0.009, 0.001 and 0.004 nats for A / B / C), B is
essentially converged.

**Training stability.** No setting had non-finite gradients or loss spikes; maximum gradient norms are 1.8–2.2
(clip at 1.0) and the step-level training-loss curves have the same noise level. Stability does not
distinguish the settings in this setup.

**Fit vs. generalisation.** The frozen models reach a much *lower* training loss (4.27 / 4.35 vs 4.73)
despite having 8.7M fewer trainable parameters, and they also generalise better. Their train − val gap is
larger (−0.47 / −0.41 vs −0.27), so they are overfitting more, not less. Freezing therefore did not help
mainly as a regulariser here: it made the LM easier to optimise.

### Where the gain comes from (`runs/part2/freq_buckets.md`)
Test CE grouped by how often the **input** token (the one right before the target) occurs in training:

| input-token train freq | share of test | A random | B Skip-gram | C GloVe |
|---|---|---|---|---|
| 1–10 | 3.4% | 4.731 | 4.627 | **4.446** |
| 11–100 | 13.6% | 4.534 | 4.392 | **4.362** |
| 101–1,000 | 23.9% | 4.320 | **4.102** | 4.150 |
| 1,001–10,000 | 15.4% | 5.030 | **4.738** | 4.785 |
| > 10,000 | 36.3% | 5.480 | **5.184** | 5.199 |
| `<unk>` | 6.2% | 4.898 | 4.681 | 4.667 |
| word not in GloVe (frozen random row in C) | 0.1% | 4.519 | 4.936 | 5.012 |

* The frozen tables help in **every** frequency band; the absolute gain is largest for frequent inputs
  (−0.30 nats for > 10k), not concentrated on rare words.
* **GloVe is best for the rarest inputs** (−0.18 nats vs Skip-gram for words seen ≤ 10 times), where its
  ~2,900× larger training corpus provides vectors that 2M tokens cannot. **Skip-gram is best for frequent
  and mid-frequency inputs**, where in-domain co-occurrence statistics are plentiful and match the LM's
  data. The two effects roughly cancel overall.
* The handful of inputs without a GloVe vector (0.1%) are predicted worst by C, as expected for a frozen
  random row (bucket too small to weigh much).

### What the tables encode (`runs/part2/neighbours.md`, cosine nearest neighbours)
* **A (learned through the LM)**: good for closed classes — months (january → november, october,
  february …), numbers (three → two, four, six …), hurricane → storm, cyclone — but noisy for many
  content words: war → *fastra, ahk, mahdi, bishopric*; guitar → *dj, asylums, saiban*; france →
  *townsville, barron, invincible*. 10 epochs of sparse LM gradients do not organise these rows well.
* **B (Skip-gram on WikiText-2)**: topical/co-occurrence neighbours tied to this corpus: album →
  *yankovic, daydream, slayer*; river → *missouri, mississippi, platte*; king → *æthelred, esarhaddon*.
* **C (GloVe)**: clean general-English semantic neighbours: war → *wars, conflict, battle*; guitar →
  *guitars, bass, vocals*; scientists → *researchers, experts, biologists*.

### Generated text (same 5 prompts, seeds, decoding as Part I)
| setting | distinct-1 | distinct-2 | repeated-trigram rate |
|---|---|---|---|
| A random | 0.65 | 0.93 | 0.004 |
| B Skip-gram | 0.68 | 0.94 | 0.013 |
| C GloVe | 0.65 | 0.89 | 0.038 |

* The frozen-embedding models stay closer to the prompt's topic: "scientists have found that the" →
  B: "presence of the species … the noisy miner"; C: "kakapo … common species of allosaurus"; A drifts to
  a film soundtrack. "the history of the city" → B/C: school, buildings, cathedral, ruins, national
  register of historic places; A: album reviews.
* C repeats most ("the church of the church", "the same time … the same of … the same species"); B is the
  most varied. Local fluency and markup artefacts (`= = = … = = =`) are similar for all three.

## Interpretation: trainability vs. extra data
* **A vs B — same data, different trainability.** With exactly the same 2.08M tokens, a 30-second
  Skip-gram pass followed by freezing beats learning the embeddings jointly with the LM by 0.24 nats.
  So, under our 10-epoch budget, end-to-end trainability did not pay off; what mattered was starting
  from embeddings that already encode distributional similarity. Supporting observations: lower training
  loss and faster convergence for B; the noisy neighbours of A's learned table.
  Possible explanations (not tested): each embedding row only gets gradient when its word occurs, so most
  of the 28.9k rows receive few, noisy updates in 5k steps, while Skip-gram's objective is designed for
  this and sees each word in many contexts per epoch; trainable embeddings also share the LSTM's
  aggressive lr 3e-3 and weight decay.
* **B vs C — both frozen, different data.** GloVe adds ~2,900× more (general-domain) text but brings
  domain mismatch and 7.5% of test tokens without a real vector (mostly `<unk>`/`<eos>`). The net effect
  is a tie; the extra data shows up exactly where it should — rare words — while the in-domain Skip-gram
  wins on frequent words. More data is not automatically better when the smaller corpus is the target
  domain.
* What we **cannot** separate with these three settings: whether the gain comes from the pre-trained
  *initialisation* or from *freezing*. A fourth run — Skip-gram initialisation, embeddings trainable —
  would answer this (optional, not run).

## Limitations
* One seed per setting; B vs C (0.013 nats) is within plausible seed variance; A vs B/C (0.23–0.24 nats)
  is far larger than the seed-to-seed differences one would expect.
* The LR was selected for setting A and reused; the frozen settings might prefer another LR (this, if
  anything, favours A).
* The rescaling to a common mean norm is a design choice; raw-scale tables were not tried.
* Timing: LM training times differ by < 10% (noise on a laptop); freezing saves only the embedding-table
  update, which is negligible next to the 28.9k-way softmax. Inference cost is identical by construction
  (same network); the benchmark measured 20.0k–25.5k scoring tok/s for the three, which is thermal drift
  of the fanless laptop, not a real difference.
