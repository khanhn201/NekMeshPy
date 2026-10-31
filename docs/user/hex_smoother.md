# Smoothing a hex mesh (experimental)

`nekmeshpy.hexmesh.smoother` is the {doc}`quad smoother <smoother>` one rung up: the same constraint vocabulary, the
same element-shrink Laplace and shape optimizer, the same schedules, on the corners of a linear `HexMesh`. It moves
vertices only; connectivity, element tags and face tags never change.

```{warning}
Experimental. It has been tested on small blocks only (a few hundred hexes) and is pure NumPy: expect it to be slow
on a large mesh. There is no interactive session (`propose` / `accept` / `undo`) at this rung yet.
```

## Quick start

```python
import numpy as np
from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.hexmesh import smoother as hs

mesh = hexmesh.extrude(quadmesh.ogrid(linemesh.circle(1.0, 16), 4, 2), 1.0, 3)
inner = np.setdiff1d(np.arange(mesh.n_points), hexmesh.boundary_points(mesh))
P = mesh.points.copy()
P[inner] += np.random.default_rng(0).normal(scale=0.03, size=(inner.size, 3))
mesh = hs.with_points(mesh, P)                # a perturbed interior

r = hs.resolve(mesh, [], boundary="slide")    # walls slide on themselves, rims along the rims
out, logs = hs.run_schedule(mesh, r, hs.suggest_schedule(hs.quality(mesh.points, mesh.corners), outer=2))
print(hs.round_report(mesh, out, r, stages=logs))
writer.to_vtu(out, "smoothed.vtu")
```

On this 144-hex cylinder the minimum scaled Jacobian goes from 0.14 to 0.86 in a couple of seconds, every boundary vertex
staying on its cap plane, its wall or its rim.

## What exists

| piece | names |
|---|---|
| constraints | `Frozen`, `Tangent`, `OnSurface`, `OnCurve`, `KeepLayer(seeds, rings)`, `SlideColumns`, `frozen_region(mesh, tags)`; targets `plane`, `sphere`, `surface_from_trimesh`, `line`, `circle`, `spline_curve`, or any `Surface` / `Curve` |
| resolution | `resolve(mesh, constraints, boundary="frozen" \| "free" \| "tangent" \| "slide", corner_deg=30)`, `snap`, `constrain_direction`, `on_target_report` |
| Laplace | `laplace(mesh, resolved, passes=3, w=0.97)`; `laplace_local` in a schedule |
| optimizer | `optimize(mesh, resolved, iters=20, delta=0.01, kappa=1000, eps=0.05)`; `optimize_local` in a schedule |
| schedules | `run_schedule`, `user_schedule`, `untangle_schedule`, `suggest_schedule`; `register_method(name, fn)` adds a stage |
| readings | `quality`, `shape_stats`, `round_report` |

The constraint objects, `Resolved`, `snap` and `constrain_direction` are the quad smoother's own. At this rung a
string selection names a **face tag**, and `KeepLayer` counts hex rings.

**The shape metric.** At a hex corner with edge vectors `a, b, c` (right-handed), the optimizer descends
`phi = ((|a|² + |b|² + |c|²)/3)³ / det(a, b, c)²`: Knupp's mean-ratio shape measure cubed. It is 1 only at a cube
corner, does not change under scaling or rotation, and is exactly the quad metric with the dimension raised from 2
to 3. Inversion has its own penalty, `kappa * max(0, eps - s)`, on the signed scaled Jacobian
`s = det / (|a||b||c|)`. The barrier-free `delta` continuation (`untangle_schedule`) pulls folded hexes through,
which the exact metric cannot. The gradient is analytic and checked against finite differences.

## The boundary, in three layers

`boundary="slide"` holds every unclaimed boundary vertex on the boundary as it was meshed:

1. **Patches.** Boundary faces are grouped into patches: two faces sharing an edge are in one patch when their
   normals differ by at most `corner_deg`. A vertex inside a patch slides on it. A planar patch is a plane; a
   curved one is its own input faces, read as a triangle surface.
2. **Edges.** A vertex where two patches meet slides along their intersection: a line when straight, else a
   spline through the input vertices (a cylinder rim is a closed one).
3. **Corners.** A vertex where three or more patches meet, or where an edge turns by a jump (more than `corner_deg`
   and three times its neighbours' turn), is frozen.

The `boundary` shorthand, for Laplace and the optimizer alike ({doc}`../smoother/methods/constraints`):

| `boundary` | an unclaimed boundary vertex |
|---|---|
| `"frozen"` (default) | holds still |
| `"free"` | moves as proposed: Laplace shrinks a convex boundary |
| `"tangent"` | a face vertex moves in its patch's tangent plane, an edge vertex along the edge's tangent; corners frozen. No target: the volume is kept, the wall can drift ({doc}`../hexsmoother/free_boundary`) |
| `"slide"` | the three layers above: fitted `OnSurface` per patch, fitted `OnCurve` per patch edge, corners frozen |

In the Laplace pass a boundary face is never shrunk across its own normal, the 3-D form of the quad rule. Each pass
is guarded: a pass that adds an inverted hex, or lowers the minimum scaled Jacobian by more than `max_sj_drop`, is
rolled back.

## The round report

`round_report(before, after, resolved, stages=logs)` gives:

- the constraints and the stages that ran;
- every corner metric before → after;
- one row per region (`element_tags`), with counts below `THRESHOLDS = (0.1, 0.2, 0.3)`;
- the **sampled** reading: the `.re2` trilinear map read on the GLL lattice of `sample_order` (default `SCAN_ORDER`,
  the solver's), plus the curved map of a high-order `original=`;
- the distance of every target's vertices (planes that held show 0);
- whether every frozen vertex (kept layers, frozen regions, corners) is bit-identical.

A moved frozen vertex, a vertex off its target or more inverted hexes fail the round.

## Limits

- **Linear only.** A high-order mesh is refused with a message: smooth its straight-sided shadow,
  `hs.linearize(mesh)`. Its curved nodes are not rebuilt afterwards (no hex `rebuild_high_order` yet), and there is
  no native curved optimization. `round_report(..., original=mesh)` still reads the input's curved map.
- **Slow and lightly tested.** Small blocks only so far; larger meshes will be tested step by step.
- **A curved wall is approximated by its input faces**: a vertex sliding on it lands on the faceted surface, so pass
  your own `Surface` for an exact wall.
- **The metric seeks cubes** and flattens designed grading: freeze layers with `KeepLayer` or `frozen_region`.
- **An MFEM TMOP backend is planned**, as a stage registered through `register_method`.
