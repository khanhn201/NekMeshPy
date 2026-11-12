# upipes: EVG hex mesh outside a bundle of U-tubes

`./run.sh` rebuilds `res_al1u.vtu` (16,702 hexes, valid, 9^3-lattice min scaled Jacobian 0.237, median
0.825, solver aspect 7.0) from `pipes.txt`: 35 U-tube spines (R = 6.4135) standing on the plate
y = 1938.811, in a box with 7 mm margin. The mesh fills the fluid outside the pipes. Face tags are
`pipes`, `plate`, `x_min`, `x_max`, `z_min`, `z_max` and `top`. No `.re2` is written, because boundary
codes still need choosing.

This copies the `scratch/pipes` code that made `scratch/pipes/res_al1u.vtu` on 2026-11-11. It follows
Sheffer et al. 1999 (embedded Voronoi graph), extended with cores for the focal U-bend axes and a hub
block where the inner pipes meet. It is a multi-script pipeline, not a flat example, so
`tests/test_examples.py` (top-level `examples/*.py` only) does not run it.

| stage | scripts | output |
|---|---|---|
| Voronoi diagram | `rays.py` (ray label map per pipe), `edges.py`, `graph.py` | `maps/` |
| graph fixes | `graphfix.py` | `maps/graph2.pkl` |
| decomposition + counts + columns | `build3.py` (`patches.py`, `items.py`, `structured.py`, `columns.py`, `core.py`, `hubcore.py`, `mesher.py`, ...) | `out_full.*` |
| surface smoothing | `surfsmooth.py`, `gsmooth2.py` (surface pass) | `out_full_ss_g.npz` |
| untangle, boundary fixed | `outer_untangle.py` (`psmooth.py`, `inflate.py`) | `out_full_ou.npz` |
| volume smoothing + checks | `gsmooth2.py`, `untangle2.py`, `jacscan.py`, `export.py`, `bquality.py` | `res_al1u.*` |

The `EVG_*` settings in `run.sh` are the ones that run used. `SKIP_MAPS=1 ./run.sh` reuses `maps/`.
Every output lands in this folder (`.gitignore` covers it).
