| model | test CE | test PPL | params / entries | size (MB) | best epoch | final train-val gap | train time (s) | score tok/s | gen tok/s | device |
|---|---|---|---|---|---|---|---|---|---|---|
| rnn_lr1e-3 | 5.367 | 214.2 | 24.4M params | 93 | 10 | +0.101 | 3366 | 19953 | 146.0 | mps |
| rnn_lr3e-3 | 5.417 | 225.2 | 24.4M params | 93 | 10 | +0.045 | 3754 | 19594 | 183.3 | mps |
| lstm_lr1e-3 | 5.230 | 186.7 | 27.3M params | 104 | 10 | -0.033 | 3226 | 23288 | 188.8 | mps |
| lstm_lr3e-3 | 4.930 | 138.4 | 27.3M params | 104 | 10 | -0.271 | 3510 | 25472 | 135.6 | mps |
| transformer_lr5e-4 | 4.945 | 140.5 | 19.6M params | 75 | 10 | -0.359 | 3172 | 29345 | 62.5 | mps |
