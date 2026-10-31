| method | call | runs | at most |
| --- | --- | --- | --- |
| **Laplace only** | `laplace(passes=100, w=0.95)` | one Laplace stage | 100 Laplace passes |
| **Winslow only** | `winslow(sweeps=200)` | one Winslow stage at its defaults | 200 Winslow sweeps |
| **Optimizer only** | `optimize(iters=200)` | one optimizer stage at the default `delta=0.01` | 200 optimizer iterations |
| **Combined** | `suggest_schedule(quality)` | `user_schedule()` = `10 × [laplace(passes=3, w=0.97) + optimize(iters=20)] + 2 × laplace_local(passes=5, w=0.97)`; when a quad starts inverted, optimizer passes with a wide-to-narrow `delta` come first (`untangle_schedule()` = `optimize(iters=60, delta=v) for v in 1, 0.3, 0.1, 0.03, 0.01`) | 30 Laplace passes, 200 optimizer iterations, 10 local Laplace passes; plus 300 optimizer iterations when a quad starts inverted |
