| method | stages | inverted | min SJ | mean SJ | time (s) |
|---|---:|---:|---:|---:|---:|
| input | — | 12 | -0.9649 | 0.8681 | — |
| Laplace | **1** | **0** | 0.8497 | 0.9845 | **0.02** |
| optimize δ=0 | **1** | 12 | -0.6455 | 0.8290 | 1.45 |
| untangle preset | 5 | **0** | **0.9802** | **0.9893** | 0.49 |
| suggested | 27 | **0** | **0.9802** | **0.9893** | 0.90 |

**Bold**: the best value in each column among the results (highest scaled Jacobian; lowest count, objective, time, change or distance); the input row is for reference.

Times: a laptop, running serial.
