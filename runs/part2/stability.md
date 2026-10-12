| setting | trainable params | best val PPL (epoch) | test PPL | first epoch val PPL < 250 / < 200 | grad-norm mean (ep1 / ep10) | grad-norm max (any epoch) | nonfinite steps | final train-val gap | train time (s) |
|---|---|---|---|---|---|---|---|---|---|
| random | 27.27M | 148.2 (10) | 138.4 | 3 / 4 | 0.53 / 0.16 | 2.18 | 0 | -0.271 | 3510 |
| w2v_frozen | 18.60M | 114.4 (10) | 108.6 | 2 / 2 | 0.52 / 0.22 | 1.95 | 0 | -0.474 | 3255 |
| glove_frozen | 18.60M | 115.9 (10) | 110.1 | 2 / 3 | 0.54 / 0.22 | 1.78 | 0 | -0.407 | 3564 |
