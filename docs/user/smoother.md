# Smoothing a quad mesh

`SmoothSession` (`nekmeshpy.quadmesh.smoother`) smooths a quad mesh, flat or a surface in 3-D, one reviewed round at a
time. It moves vertices only (connectivity never changes; high-order nodes are rebuilt afterwards), and every round
reports what it did and can be undone. The order of work, step by step, is the {doc}`workflow chart <../smoother/index>`.

## Quick start

```python
import numpy as np
from nekmeshpy import linemesh, quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import SmoothSession, OnCurve, circle, with_points

# a disk with a perturbed interior (any linear or curved QuadMesh works)
mesh = quadmesh.ogrid(linemesh.circle(1.0, 24, element_tag="wall"), 6, 4, wall_tag="wall")
jitter = np.random.default_rng(3).normal(size=mesh.points.shape) * [0.03, 0.03, 0]
mesh = with_points(mesh, mesh.points + jitter)

s = SmoothSession(mesh, [OnCurve("wall", circle([0, 0, 0], 1.0))], out_dir="smooth_rounds")
proposal = s.propose()        # runs s.suggest() on a copy; nothing is accepted yet
print(proposal.report)        # every metric before -> after, PASS / FAIL
s.accept(proposal)            # s.undo() restores the previous state bit for bit
quad_to_vtu(s.current, "smoothed.vtu")
```

| call | does |
|---|---|
| `s.propose(schedule=None)` | a round on a copy (default: `s.suggest()`) |
| `s.accept(p)` / `s.undo(n=1)` | keep a round (refused if it failed) / pop accepted rounds |
| `s.run(n, review="print")` | `n` rounds unattended, stopping at a failure; `review="auto"` also stops when nothing more is gained |
| `s.history()` | the trend over rounds |

## Constraints

| constraint | what it does | typical use |
|---|---|---|
| `Frozen(points)` | never moves | a feature placed by hand |
| `Tangent(points)` | slides along the boundary, never inward; nothing projects it back | a boundary with no geometry, under Laplace or Winslow |
| `OnSurface(points, surface=None, slide=True)` | slides tangentially, snapped back every step; no `surface`: fitted to the input faces; `slide=False`: projected once, then held | a curved wall, a planar face |
| `OnCurve(points, curve=None, slide=True)` | the same along a curve; no `curve`: a spline through the input boundary | a rim, a box crease |
| `KeepLayer(seeds, rings=1)` | freezes `rings` quad-rings off the seeds | a graded boundary layer |
| `SlideColumns(columns, direction)` | rigid columns, one shared degree of freedom each; cells between rows stay rectangles; frozen whole if any member is | rows across a wall; found by `constrain.grid_columns(mesh, seeds, direction, within=mask, outside_depth=1)` |

- `points`: ids, a boolean mask, or an **edge-tag name**. Targets: `plane`, `sphere`, `line`, `circle`, or any
  `Surface(project, normal)` / `Curve(project, tangent)`; gradients use the target's normal, never the mesh's.
- Precedence: `Frozen > OnCurve > OnSurface > Tangent > Free`; a vertex claimed by two surfaces or curves is frozen (and listed).
- `weights=w` scales each quad's shape term (`0`: shape irrelevant); the inversion penalty still applies.

`boundary=` (in `resolve(mesh, constraints, boundary=...)` and `SmoothSession`) is a shorthand for every boundary vertex
no constraint names:

| `boundary` | states | an unclaimed boundary vertex | use for |
|---|---|---|---|
| `"frozen"` (default) | `Frozen` | never moves | a boundary placed as meshed |
| `"free"` | nothing | moves as proposed: Laplace pulls a convex boundary in, the optimizer pushes it out | a boundary whose position does not matter |
| `"tangent"` | `Tangent` | slides along the boundary, never inward | Laplace, Winslow; no known geometry |
| `"slide"` | `OnCurve`, fitted | slides on a spline through the input boundary (surface quads too) | no known geometry, any method |
| — | `OnCurve` / `OnSurface` with a target | slides on the exact target | known geometry: always preferred |

Corners (a turn above `corner_deg=30` that is also a jump against its neighbours') are frozen in every mode but
`"free"`. Under `"tangent"` the optimizer can stall: when it must move the boundary, use `"slide"`.

```{warning}
The shape metric seeks squares, so an unconstrained graded boundary layer gets flattened. Freeze it with `KeepLayer`.
```

How each piece works: {doc}`../smoother/methods/constraints`.

## Schedule

`s.suggest()` proposes `10 x [laplace(3) + optimize(20)]` plus two local lifts (`user_schedule()`); when any quad starts
inverted, `optimize` stages with a wide-to-narrow `delta` come first. A schedule is a list of `(stage, knobs)`.

| stage | what it does | main knobs |
|---|---|---|
| `laplace` | moves each vertex toward its neighbours | `passes` 3, shrink `w` 0.97 |
| `optimize` | descends the shape metric plus an inversion penalty; with a `delta` continuation (`untangle_schedule()`, a preset of `optimize`, not a separate stage) it pulls folded vertices through | `iters` 20, `kappa`, `eps`, `delta` 0.01 (the preset: 1.0 → 0.01, `iters` 60 each) |
| `laplace_local` | the failing quads only, lifting the smallest edges | `passes` 5, `w` |
| `winslow` / `winslow_local` | elliptic smoothing (all quads / the failing ones): an alternative to `laplace` for concave outlines and thin curved blocks; not in the suggested schedules, add it by hand: `("winslow", {"sweeps": 100})` | `sweeps` 200, `omega` 1, `tol` 1e-6 |

- Keep graded layers with `KeepLayer` under every stage, Winslow most of all
  ({doc}`../smoother/methods/winslow`).
- `kappa` (1000) / `eps` (0.05) set the inversion penalty: `("optimize", {"iters": 20, "kappa": 1e4})` can unstall an
  inverted corner for about 10 % more time. A knob to try, not a better default.
- `adjust_schedule(schedule, "more_aggressive" | "more_conservative" | "same")` changes shrink, passes and iterations,
  never a constraint.
- Which stage when, measured side by side: {doc}`../smoother/methods/index`.

## The round report

| # | section | shows |
|---|---|---|
| 1 | constraints | vertex counts per kind, targets, tolerances |
| 2 | what ran | each stage and why it stopped |
| 3 | quality | **every** metric before → after, marked better / WORSE / same: inverted and overlapping quads first, scaled J (min, mean, median, worst 1/5/10 %), aspect ratio, corner angles, size ratio, node spacing |
| 4 | geometry | are the vertices still on their targets (`on_target_report`) |
| 5 | propagate to hex | "quad-only" for now |
| 6 | undo | what `undo(1)` returns to |

Warnings do not block acceptance; failures do (off target, more inverted or more overlapping quads). With `out_dir`,
`round_NN/before.vtu` / `after.vtu` carry `sj_min`, `phi_max`, `constraint_kind`, `dist_to_target`, `displacement`.

## Checks before smoothing

| check | catches | opt out |
|---|---|---|
| gap check (`quadmesh.check_connectivity`: gencon's rule, relative `connectivity_tol=0.2`) | points that should be one vertex, an edge in more than two quads, a quad naming a vertex twice, a collapsed edge, a pinch vertex; **refuses to start** | `allow_gaps=True` (stated in every report) |
| normal check | quads disagreeing with the surface normal (folded, reversed, wrong surface); `normal_report.n_disagree` | `check_normals=False` |
| overlap check (`quadmesh.overlapping_quads`) | overlapping input quads; recounted every round | `check_overlaps=False` |

## Curved surfaces

- **Claim the vertices with `OnSurface`**: they slide and are projected back every step, never leaving the surface; `on_target_report` gives the largest
  distance.
- **Normals come from the surface**: `normals=`, else `surface=`, else the `OnSurface` targets, and only then the
  quads' winding. `s.normal_report` says which.
- **A tessellation** becomes a surface through `surface_from_trimesh(points, tris)`: an approximation (chord error
  remains, sharp edges blur), with one sign per surface voted against the mesh winding.
- **Judge against the surface**, not only squareness:
  - *angle defect* (`quadmesh.angle_defect`): on the perturbed test surfaces its error falls from about 0.15 to
    0.01-0.02 rms, against a floor of about 0.01;
  - *sagitta*: the chord error between vertices;
  - *overlapping quads*: a branched star can have every Jacobian positive;
  - *Gauss-Bonnet residual* (`quadmesh.gauss_bonnet_residual`): a topology identity; catches bad edge counts and pinch
    vertices, not poor representation.

```python
from nekmeshpy.quadmesh.smoother import OnSurface, sphere, surface_from_trimesh

surf = sphere([0, 0, 0], 1.0)                   # analytic: plane, sphere, Surface(...)
surf = surface_from_trimesh(tri_points, tris)   # or a tessellation
s = SmoothSession(mesh, [OnSurface(np.arange(mesh.n_points), surf)], surface=surf)
print(s.normal_report)                          # normals' source; quads that disagree
```

```{warning}
Always give the surface. Normals from the quads' winding are poisoned by one folded input quad: on a perturbed
ellipsoid, smoothing then took the angle-defect error from 0.157 to 0.231 instead of 0.017. This, not the metric, was
behind an earlier finding that smoothing worsens curvature.
```

(high-order)=
## High order

- A curved input is smoothed as its straight-sided shadow (`highorder.linearize`).
- `s.rebuild_high_order(order=None)` returns the curved mesh of the accepted vertices: nodes keep their offset from the
  straight element and are projected back onto their targets; `order=N` lifts a linear mesh.
- Its `HighOrderReport` samples the scaled Jacobian at order 8 and the target distance before/after projection.
- Limits: radial projection is not exactly Gauss-Lobatto arc spacing; curved validity is reported, not guaranteed;
  quads only.

## Limitations

- **A smoothed surface is not carried into a hex mesh** (no propagation, no block-state certificates). Hexes have
  their own, experimental smoother: {doc}`hex_smoother`.
- **The shape metric seeks squares and undoes designed grading**; protect layers with `KeepLayer` (no target-matrix
  metric yet).
- **Overlapping or folded input is reported, not repaired.** A fold, mostly from projecting onto a curved surface, can
  leave a persistent overlap that, on perturbed capsules, longer schedules, a second pass or the `delta` continuation first did not
  remove (mechanism partly understood; a volume-preserving Laplacian is future work).
- **Size does not follow curvature** (the metric aims at uniform size; a size target is being evaluated). With few
  irregular vertices part of the curvature error belongs to the connectivity: angle defects sum to 2 pi times the
  Euler characteristic.
- **Repairing an inverted quad is not guaranteed**; the round says when one remains.

```{note}
Before/after viewers for flat, curved and real meshes, and what carrying a smoothed surface into hexes does, are in
the {doc}`Deep dives <../smoother/index>`. Tests: `tests/smoother2d/`.
```
