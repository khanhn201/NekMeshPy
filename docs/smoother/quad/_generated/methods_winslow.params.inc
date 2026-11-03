| parameter | default | meaning |
| --- | --- | --- |
| `sweeps` | `200` | maximum Gauss-Seidel sweeps |
| `omega` | `1.0` | under-relaxation of every move, in (0, 1] |
| `tol` | `1e-06` | stop when the largest move of a sweep is below this many local edge lengths |
| `normals` | `None` | reference normals (default: the input's winding; pass the surface's) |
| `elements` | `None` | boolean mask of quads: only their vertices move (the local variant) |
| `max_sj_drop` | `0.05` | a sweep that adds an inverted quad or lowers the minimum scaled J by more than this below the best reached is retried with `omega` halved, then refused and the stage stops |
| `retries` | `3` | number of `omega` halvings tried on a refused sweep |
| `kappa` | `1000.0` | inversion-penalty weight, used only for the F/quad reading in the log |
| `eps` | `0.05` | inversion-penalty threshold on the scaled J, likewise only for the reading |
