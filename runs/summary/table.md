| model | test CE | test PPL | params / entries | size (MB) | best epoch | final train-val gap | train time (s) | score tok/s | gen tok/s | device |
|---|---|---|---|---|---|---|---|---|---|---|
| ngram | 5.379 | 216.7 | 5.8M n-grams | 81 | - | - | 5 | 301407 | 470.3 | cpu |
| rnn | 5.367 | 214.2 | 24.4M params | 93 | 10 | +0.101 | 3366 | 19953 | 146.0 | mps |
| lstm | 4.930 | 138.4 | 27.3M params | 104 | 10 | -0.271 | 3510 | 25472 | 135.6 | mps |
| transformer | 4.945 | 140.5 | 19.6M params | 75 | 10 | -0.359 | 3172 | 29345 | 62.5 | mps |
