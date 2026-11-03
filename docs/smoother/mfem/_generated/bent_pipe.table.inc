| variant | order | min SJ, nodes | mean SJ, nodes | min SJ, order 7 | inverted, order 7 | min SJ, corners | wall off the torus, max (% of R) | time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| input | 3 | 0.3273 | 0.6866 | 0.2429 | 0 | 0.6757 | 0.020 | — |
| order-1 route | 1 | **0.8507** | 0.8767 | **0.8507** | **0** | **0.8507** | 9.324 | **0.19** |
| native optimizer | 3 | 0.4779 | 0.7932 | 0.4699 | **0** | 0.8064 | 0.022 | 3.30 |
| TMOP, curved | 3 | 0.5736 | **0.8957** | 0.5736 | **0** | 0.8488 | **0.020** | 25.10 |

**Bold**: the best value in each column among the results (highest scaled Jacobian; lowest count, objective, time, change or distance); the input row is for reference.

Times: a laptop, running serial.
