| parameter | default | meaning |
| --- | --- | --- |
| `iters` | `20` | maximum conjugate-gradient iterations |
| `normals` | `None` | reference normals (default: the input's winding; pass the surface's) |
| `elements` | `None` | boolean mask of quads: only their vertices move, plus the quads they touch count (the local variant) |
| `alpha_max` | `1.0` | cap on a step, in local edge lengths |
| `restart` | `'halve'` | `'halve'` halves the Fletcher-Reeves beta on every second iteration, `'none'` is plain Fletcher-Reeves |
| `kappa` | `1000.0` | inversion-penalty weight per corner on the sum objective |
| `eps` | `0.05` | scaled-J threshold below which a corner is penalised |
| `delta` | `0.01` | barrier-removal width of the descent metric; 0 descends on the exact phi |
| `tol_step` | `0.0001` | stop when the accepted step is below this many local edge lengths |
| `tol_rel` | `1e-06` | stop when the relative drop of F in one iteration is below this |
| `weights` | `None` | per-quad weights on the shape term (0: that quad's shape is irrelevant; the penalty still applies) |
