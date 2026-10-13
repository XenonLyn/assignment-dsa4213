# Part III — Downstream topic classification with frozen LM representations (draft for the report)

**Question.** Does better next-token prediction (lower LM perplexity) give better document
representations for a downstream task? We use all five language models from Parts I–II, whose test PPL
ranges from 108.6 to 214.2, and two pooling methods per model. Code: `downstream.py`; outputs:
`runs/part3/`; log: `runs/logs/part3.log`. Single seed (42).

## Task and data
* **AG News** (`fancyzhx/ag_news` on the HuggingFace Hub; Zhang et al., 2015; academic/non-commercial
  use): news topic classification, 4 balanced classes (World, Sports, Business, Sci/Tech).
* Splits: from the official 120k training set, a class-balanced **20,000-document training set** and a
  disjoint balanced **5,000-document validation set** (seed 42); the official **7,600-document test set**
  is used only for the final numbers.
* Preprocessing mimics the LM corpus (WikiText-2): HTML entities/tags cleaned (`#39;` → `'`, `<a href…>`
  removed), lowercased, `1,000` → `1 @,@ 000`, `long-term` → `long @-@ term`, `wasn't` → `wasn 't`,
  punctuation/`$`/`%` as separate tokens, `<eos>` appended. Words outside the LM vocabulary become `<unk>`:
  **6.8% of tokens** (e.g. *afp, corp, oracle, peoplesoft, darfur, arafat, putin, ipod*), comparable to
  the 6.2% `<unk>` rate of the WikiText-2 test set. Documents average 45 tokens; 17 of 7,600 test
  documents exceed the 128-token LM context and are truncated.

## Representations (all language models frozen, eval mode)
| name | how a document becomes one vector | dim |
|---|---|---|
| `<lm>/mean` | average of the LM's top-layer token states over all real tokens (padding excluded) | 512 (RNN/LSTM), 300 (Transformer) |
| `<lm>/last` | top-layer state at the last token (`<eos>`) — for these causal models the only position that has read the whole document | same |
| `emb_*/mean` | context-free baseline: average of the input-embedding rows (the LSTM's learned table / own Skip-gram / GloVe) | 300 |
| `tfidf_bow` | reference without any LM: TF-IDF of unigrams + bigrams over the *original* words (no `<unk>`), min_df 2 | 110,746 (sparse) |

LMs: `rnn` (PPL 214.2), `lstm` (random trainable embeddings, 138.4), `lstm_w2v` (frozen Skip-gram, 108.6),
`lstm_glove` (frozen GloVe, 110.1), `transformer` (140.5).
**Classifier (identical for all):** standardised features (TF-IDF: none) + multinomial logistic
regression; C ∈ {0.01, 0.1, 1, 10} chosen on validation accuracy; the selected classifier is evaluated once
on the test set. Metrics: accuracy and macro-F1 (classes are balanced, so the two agree closely).

## Results (20k training documents)

| representation | LM test PPL | val acc | **test acc** | test macro-F1 | F1 World / Sports / Business / Sci-Tech |
|---|---|---|---|---|---|
| tfidf_bow (no LM) | – | 0.900 | **0.900** | 0.900 | 0.909 / 0.959 / 0.860 / 0.871 |
| emb_glove/mean | – | 0.883 | 0.878 | 0.878 | 0.885 / 0.955 / 0.826 / 0.846 |
| **lstm_glove/mean** | 110.1 | 0.843 | **0.848** | 0.847 | 0.866 / 0.934 / 0.783 / 0.806 |
| emb_w2v/mean | – | 0.823 | 0.828 | 0.827 | 0.840 / 0.921 / 0.761 / 0.788 |
| lstm_glove/last | 110.1 | 0.818 | 0.820 | 0.820 | 0.831 / 0.929 / 0.736 / 0.783 |
| lstm_w2v/mean | 108.6 | 0.808 | 0.809 | 0.808 | 0.827 / 0.909 / 0.732 / 0.765 |
| lstm_w2v/last | 108.6 | 0.778 | 0.785 | 0.785 | 0.792 / 0.904 / 0.699 / 0.744 |
| transformer/mean | 140.5 | 0.788 | 0.780 | 0.779 | 0.787 / 0.888 / 0.701 / 0.742 |
| transformer/last | 140.5 | 0.770 | 0.769 | 0.768 | 0.784 / 0.890 / 0.678 / 0.720 |
| emb_learned/mean | – | 0.754 | 0.761 | 0.760 | 0.758 / 0.852 / 0.701 / 0.729 |
| lstm/mean | 138.4 | 0.730 | 0.719 | 0.718 | 0.725 / 0.835 / 0.638 / 0.672 |
| rnn/mean | 214.2 | 0.663 | 0.661 | 0.659 | 0.674 / 0.747 / 0.602 / 0.614 |
| lstm/last | 138.4 | 0.621 | 0.617 | 0.615 | 0.640 / 0.761 / 0.464 / 0.593 |
| rnn/last | 214.2 | 0.392 | 0.399 | 0.394 | 0.315 / 0.461 / 0.349 / 0.449 |

(chance = 0.25; validation and test accuracy agree within 1.2 points for every representation.)

### 1. Does the LM ranking agree with the downstream ranking?
Only partly. Spearman correlation between LM test PPL and test accuracy over the five LMs is
**ρ = −0.80 for both poolings** (n = 5, p = 0.10). The direction is right — the worst LM (RNN) gives the
worst features and the two frozen-embedding LSTMs give the best — but the ranking breaks exactly where
perplexities are tied:
* **LSTM (138.4) vs Transformer (140.5)**: equal PPL, but the Transformer's representations are
  **6.1 points** better (mean pooling: 78.0% vs 71.9%) and 15 points better with last-token pooling.
* **LSTM+Skip-gram (108.6) vs LSTM+GloVe (110.1)**: equal PPL (GloVe slightly *worse*), but GloVe
  is **3.9 points** better downstream (84.8% vs 80.9%), and the gap holds for last-token pooling (+3.5).

The downstream order follows the **source of the input embeddings** more closely than perplexity:
GloVe > Skip-gram > (Transformer) > LM-learned table > RNN.

### 2. Context-free averages beat the LM states built on top of them
For every LSTM, simply averaging its *input* embeddings beats averaging its contextual hidden states:
GloVe 87.8% vs 84.8%, Skip-gram 82.8% vs 80.9%, learned table 76.1% vs 71.9% — although the hidden
states are larger (512-d vs 300-d) and the classifier is the same. The recurrent layers, trained only to
predict the next word, transform the lexical (topic-bearing) information into something less linearly
useful for topic classification. The Transformer loses less: its mean-pooled states (78.0%) beat the
LSTM's learned-table average (76.1%).

### 3. Pooling
Mean pooling beats last-token pooling for every LM, by very different margins: RNN 26.2 points, LSTM 10.1,
GloVe-LSTM 2.7, Skip-gram-LSTM 2.4, Transformer 1.1. The RNN's final state is nearly useless (39.9%, close
to chance): after 45 tokens it retains almost nothing about the document's topic, matching its poor use of
long context in Part I. The Transformer's last position can attend to every token directly, so its last
state is almost as good as the average.

### 4. Amount of labelled data (`runs/part3/learning_curve.md`, right panel of `ppl_vs_acc_and_curve.png`)

| labelled training docs | 200 | 1,000 | 5,000 | 20,000 |
|---|---|---|---|---|
| tfidf_bow | 0.732 | 0.827 | 0.869 | **0.897** |
| emb_glove/mean | **0.838** | **0.855** | **0.871** | 0.878 |
| lstm_glove/mean | 0.786 | 0.806 | 0.831 | 0.848 |
| emb_w2v/mean | 0.770 | 0.791 | 0.811 | 0.828 |
| lstm_w2v/mean | 0.746 | 0.764 | 0.793 | 0.809 |
| transformer/mean | 0.685 | 0.716 | 0.768 | 0.780 |
| lstm/mean | 0.595 | 0.657 | 0.703 | 0.719 |
| rnn/mean | 0.482 | 0.570 | 0.635 | 0.661 |

(Same C selection on the 5k validation set for every point. The 20k column differs from the main table
for TF-IDF only because the learning-curve vectoriser uses min_df 1.)
With **200 labelled documents**, GloVe averages (83.8%) and even the GloVe-LSTM states (78.6%) beat
TF-IDF (73.2%) by a wide margin; TF-IDF catches up around 5,000 documents and wins at 20,000. Pre-trained
dense representations matter most when labels are scarce; with enough labels, sparse lexical features
that keep every original word win this keyword-driven task.

### 5. Error analysis (`runs/part3/confusion.md`, `runs/part3/errors.md`, `runs/part3/unk_bins.md`)
* **Business ↔ Sci/Tech** is the dominant confusion for every representation: 515 of the 1,159 errors of
  `lstm_glove/mean` (258 + 257), 338 of 759 for TF-IDF. Typical cases are technology-company business news:
  "AOL to cut 700 jobs" (Business → predicted Sci/Tech), "Dell May Soon Unveil More Consumer Goods"
  (Sci/Tech → Business). Some test labels are themselves debatable (the Sony/IBM/Toshiba "Cell" chip is
  labelled Business; a cricket test match and an Olympic hockey result are labelled World) — 224 test
  documents are misclassified by *every* LM representation, many of this kind.
* **Where TF-IDF beats the best LM representation (621 docs vs 221 the other way)**: the decisive cue is
  often one specific word or source tag — ESPN.com, NBA, VoIP, FCC, AOL, Nuggets, or the agency tag
  (Reuters/AP/AFP). TF-IDF keeps every original word (including ones the LM maps to `<unk>`, such as *afp*)
  and weights rare, discriminative words highly; averaging 45 contextual states dilutes a single cue word.
* **Where the LM representation wins**: no strong keyword, but general vocabulary with a clear topic,
  where GloVe-style similarity generalises: "Canada defends submarine fleet …" (World; TF-IDF said
  Sports), "… pass interference on a pivotal interception" (Sports; TF-IDF: World), "Sony looks under the
  Christmas tree … gadget shopping" (Sci/Tech; TF-IDF: Business).
* **`<unk>` share** has little effect: `lstm_glove/mean` accuracy is 83.4–85.2% in every bin from 0% to
  >10% `<unk>` tokens; the remaining words carry enough topic signal.

## Which representation works better, and why (observations vs. hypotheses)
Observed:
1. Best LM-based representation: **mean-pooled states of the frozen-GloVe LSTM (84.8% accuracy, 0.847
   macro-F1)**, but the context-free GloVe average (87.8%) and TF-IDF (90.0%) are better with 20k labels.
2. Mean pooling > last-token pooling for every LM; the gap shrinks as the architecture gets better at
   carrying document-level information to the last position (RNN ≫ LSTM > Transformer).
3. LM perplexity is a coarse predictor of representation quality (ρ = −0.8): it separates a bad LM from
   good ones, but among LMs of similar perplexity the downstream differences are large (4–6 points) and
   are explained by the embedding source and the architecture.

Hypotheses (not tested):
* GloVe was trained on ~6B tokens of general text including newswire (Gigaword), so its geometry encodes
  topical similarity of news words; a frozen GloVe table passes that through the LSTM unchanged, which
  helps classification but not next-word prediction on Wikipedia (Part II: equal PPL).
* The Transformer's residual connections carry the input embedding of each token up to the top layer, so
  its states keep lexical identity; the LSTM must rewrite everything through its gates, optimised for
  predicting the next word.
* Next-word prediction rewards local syntactic information; topic is a global property, so better
  perplexity need not mean more linearly accessible topic information.

## Limitations
* One seed, and one 20k subset; differences of ~1 point between representations should not be
  over-interpreted (validation and test accuracies agree within 1.2 points, which suggests the larger gaps
  are stable).
* Linear classifier only, frozen LMs only; fine-tuning or a non-linear classifier could change the ranking.
* Domain and vocabulary mismatch: the LMs were trained on lowercased Wikipedia with a 28.9k vocabulary
  (6.8% of AG News tokens become `<unk>`), whereas TF-IDF sees every original word — this favours TF-IDF.
* Only 5 LMs for the correlation (p = 0.10), and they differ in several factors at once (architecture,
  embedding source, trainable parameters).
* Feature-extraction times in the log (11–152 s) include per-shape MPS graph compilation and are not a
  cost comparison.
