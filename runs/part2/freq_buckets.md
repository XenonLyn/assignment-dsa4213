# Test CE by training frequency (common chunked setup)

Input token = the token right before the target (what the input embedding encodes).

| input-token train freq | share of test targets | random | w2v_frozen | glove_frozen |
|---|---|---|---|---|
| 1-10 | 0.034 | 4.731 | 4.627 | 4.446 |
| 11-100 | 0.136 | 4.534 | 4.392 | 4.362 |
| 101-1000 | 0.239 | 4.320 | 4.102 | 4.150 |
| 1001-10000 | 0.154 | 5.030 | 4.738 | 4.785 |
| 10001-∞ | 0.363 | 5.480 | 5.184 | 5.199 |
| <unk> | 0.062 | 4.898 | 4.681 | 4.667 |
| <eos> | 0.012 | 4.386 | 4.268 | 4.313 |
| word not in GloVe (random frozen row) | 0.001 | 4.519 | 4.936 | 5.012 |
| word in GloVe | 0.925 | 4.939 | 4.694 | 4.709 |
| all | 1.000 | 4.930 | 4.688 | 4.701 |

| target-token train freq | share | random | w2v_frozen | glove_frozen |
|---|---|---|---|---|
| 1-10 | 0.034 | 12.287 | 11.681 | 11.734 |
| 11-100 | 0.136 | 9.071 | 8.645 | 8.632 |
| 101-1000 | 0.239 | 6.566 | 6.155 | 6.137 |
| 1001-10000 | 0.154 | 4.519 | 4.304 | 4.343 |
| 10001-∞ | 0.363 | 2.181 | 2.103 | 2.122 |
| <unk> | 0.062 | 3.270 | 3.231 | 3.291 |
| <eos> | 0.012 | 1.337 | 1.207 | 1.258 |
