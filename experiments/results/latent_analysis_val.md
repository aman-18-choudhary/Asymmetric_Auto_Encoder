| Representation | features | probe val accuracy ± s.e. | probe train acc | L2 |
|---|---|---|---|---|
| raw_pixels (3072-d) | 3072 | 0.4116 ± 0.0070 | 0.4593 | 0.01 |
| raw_pixels PCA-64 | 64 | 0.3800 ± 0.0069 | 0.3916 | 0.01 |
| baseline (trained) | 64 | 0.3788 ± 0.0069 | 0.3810 | 0.01 |
| baseline (random init) | 64 | 0.3238 ± 0.0066 | 0.3381 | 0.0001 |
| baseline_matched (trained) | 64 | 0.3748 ± 0.0068 | 0.3838 | 0.01 |
| baseline_matched (random init) | 64 | 0.3336 ± 0.0067 | 0.3465 | 0.001 |
| asymmetric (trained) | 64 | 0.3794 ± 0.0069 | 0.3834 | 0.01 |
| asymmetric (random init) | 64 | 0.3242 ± 0.0066 | 0.3336 | 0.001 |

| Model | eff. rank | participation ratio | dims for 90/95/99% var | low-var dims | mean dim std | random-init eff. rank |
|---|---|---|---|---|---|---|
| baseline | 29.3 | 13.5 | 39/49/61 | 0 | 0.757 | 20.2 |
| baseline_matched | 37.2 | 18.4 | 45/54/62 | 0 | 0.392 | 16.0 |
| asymmetric | 31.3 | 14.5 | 41/50/61 | 0 | 5.052 | 14.9 |

| Model | k=0 | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=48 | k=64 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.06254 | 0.04438 | 0.03742 | 0.03076 | 0.02364 | 0.01741 | 0.01186 | 0.00924 | 0.00759 |
| baseline_matched | 0.06246 | 0.04474 | 0.03746 | 0.03086 | 0.02369 | 0.01751 | 0.01183 | 0.00918 | 0.00749 |
| asymmetric | 0.06246 | 0.04505 | 0.03808 | 0.03168 | 0.02624 | 0.02160 | 0.01634 | 0.01137 | 0.00766 |
