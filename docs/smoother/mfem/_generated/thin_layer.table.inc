| variant | first layer, mean change (%) | max abs. change (%) | inverted | min SJ | mean SJ | time (s) |
|---|---:|---:|---:|---:|---:|---:|
| input | — | — | 0 | 0.2129 | 0.7560 | — |
| native optimizer | +137.4 | 271.1 | **0** | 0.7591 | 0.9004 | 0.57 |
| TMOP ideal | +173.6 | 291.4 | **0** | 0.7129 | 0.8841 | **0.41** |
| TMOP input + designed | **-2.3** | **4.7** | **0** | **0.9597** | **0.9904** | 0.65 |

**Bold**: the best value in each column among the results (highest scaled Jacobian; lowest count, objective, time, change or distance); the input row is for reference.

Times: a laptop, running serial.
