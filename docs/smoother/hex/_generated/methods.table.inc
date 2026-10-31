| method | inverted | min SJ | mean SJ | worst 5 % SJ | F/hex | time (s) |
|---|---:|---:|---:|---:|---:|---:|
| input | 0 | 0.2856 | 0.7754 | 0.5496 | 0.9317 | — |
| Laplace | **0** | 0.9372 | 0.9773 | 0.9553 | 0.4692 | **0.02** |
| optimizer | **0** | **0.9802** | **0.9894** | **0.9821** | **0.4385** | 0.16 |
| schedule | **0** | **0.9802** | 0.9893 | **0.9821** | **0.4385** | 0.50 |

**Bold**: the best value in each column among the results (highest scaled Jacobian; lowest count, objective, time, change or distance); the input row is for reference.

Times: a laptop, running serial.
