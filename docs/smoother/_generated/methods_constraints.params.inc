| call | meaning |
| --- | --- |
| `Frozen(points)` | never moves |
| `OnSurface(points, surface, slide=True)` | stays on a `Surface`; `slide=True` moves tangentially and is projected back every step, `False` projects once and holds |
| `OnCurve(points, curve, slide=True)` | the same along a `Curve` |
| `KeepLayer(seeds, rings=1)` | freezes every vertex within `rings` quad-rings of the seeds (seeds included) |
| `SlideColumns(columns, direction)` | rigid columns with one shared degree of freedom along `direction` |
| `resolve(mesh, constraints, boundary='frozen', corner_deg=30.0)` | resolves the claims once; `'frozen'` freezes every boundary vertex no constraint names, `'free'` leaves them to move, `'slide'` holds each unclaimed boundary chain on a smooth curve through the input boundary itself and freezes its corners (turning angle above `corner_deg`) |
| `on_target_report(points, corners, resolved, tol=1e-06)` | distance of constrained vertices to their targets and the largest move of a frozen one, in local edge lengths |
