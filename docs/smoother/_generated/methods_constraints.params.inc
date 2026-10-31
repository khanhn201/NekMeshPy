| call | meaning |
| --- | --- |
| `Frozen(points)` | never moves |
| `Tangent(points)` | moves along the boundary only: the normal part of each move or gradient is removed; nothing projects it back |
| `OnSurface(points, surface=None, slide=True)` | stays on a `Surface` (none: fitted to the input faces); `slide=True` moves tangentially and is projected back every step, `False` projects once and holds |
| `OnCurve(points, curve=None, slide=True)` | the same along a `Curve` (none: a spline through the input boundary) |
| `KeepLayer(seeds, rings=1)` | freezes every vertex within `rings` quad-rings of the seeds (seeds included) |
| `SlideColumns(columns, direction)` | rigid columns with one shared degree of freedom along `direction` |
| `resolve(mesh, constraints, boundary='frozen', corner_deg=30.0)` | resolves the claims once; `boundary` is a shorthand for every boundary vertex no constraint names: `'frozen'` -> `Frozen`, `'free'` -> nothing, `'tangent'` -> `Tangent`, `'slide'` -> `OnCurve` fitted to the input; corners (a turn above `corner_deg` and 3x its neighbours') frozen except under `'free'` |
| `on_target_report(points, corners, resolved, tol=1e-06)` | distance of constrained vertices to their targets and the largest move of a frozen one, in local edge lengths |
