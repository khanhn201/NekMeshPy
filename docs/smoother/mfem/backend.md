# TMOP via the MFEM backend

The `tmop` stage hands the mesh to [TMOP](https://mfem.org/howto/mesh_optimizer/), MFEM's target-matrix optimizer, and
takes the vertex positions back. It runs inside the {doc}`quad <../../user/quad_smoother>` and
{doc}`hex <../../user/hex_smoother>` smoothers as one more stage, `("tmop", {...})`. The bridge,
`nekmeshpy.core.smoother.mfem` ({doc}`reference <../../reference/core>`), is written once and shared by both rungs.

## Install

```bash
pip install nekmeshpy[smooth]      # or: pip install mfem
```

The `smooth` extra pins PyMFEM 4.8, which has wheels for Python 3.9 to 3.13. MFEM is imported only when a `tmop` stage
runs. Without it that stage raises an `ImportError` naming the install, and everything else works as before.

## Running a stage

A `tmop` stage goes wherever a stage goes: in `run_schedule`, or in a `SmoothSession` round, alone or before or after
the native stages.

```python
from nekmeshpy.hexmesh import smoother as hs

# `mesh`: the perturbed cylinder of the hex quick start
r = hs.resolve(mesh, [], boundary="slide")
out, logs = hs.run_schedule(mesh, r, [("tmop", {"target": "input"})])
print(logs[0].reason)         # energy before -> after, iterations, what was held

s = hs.SmoothSession(mesh, [], boundary="slide")
p = s.propose([("tmop", {"target": "input"}), ("optimize", {"iters": 10})])
print(p.report)               # the usual round report; s.accept(p) keeps it
```

On the perturbed cylinder of the {doc}`hex quick start <../../user/hex_smoother>`, TMOP alone takes the minimum scaled
Jacobian from 0.14 to 0.75 in 5 Newton iterations. Followed by 10 `optimize` iterations, the round reaches 0.86 and
passes.

The quad rung is the same with `nekmeshpy.quadmesh.smoother`. Either way the stage returns a new linear mesh;
connectivity and tags never change.

## Constraints

The constraints are the smoothers' own. The stage maps each resolved vertex to fixed coordinate components, after
projecting every constrained vertex onto its target.

| constraint | under `tmop` |
|---|---|
| `Frozen`, `KeepLayer`, `frozen_region`, corners | fixed |
| `OnSurface` on a plane normal to a coordinate axis | slides in that plane (`boundary="slide"` on a box) |
| `OnCurve` on a line along a coordinate axis | slides along that line |
| `OnSurface` on the quad mesh's own plane | free |
| any other target: curved, or tilted off the axes | **held fixed** for now |
| `Tangent`, `SlideColumns` | **held fixed** for now |

On the cylinder above, the 82 cap vertices slide in their planes and the 64 wall and rim vertices are held. The
stage's `reason` counts each kind, so a held curved wall is stated, not silent.

## Targets and designed elements

The native optimizer asks every element to become a square or a cube. TMOP asks each element to match a **target**
element, chosen by `target`:

| `target` | each element is asked to become | use for |
|---|---|---|
| `"input"` (default) | orthogonal, at the size and aspect it already has | a graded or layered mesh |
| `"input_smoothed"` | the same, with the sizes first relaxed toward the neighbours' (`sweeps` 5, `sweep_weight` 0.5) | uneven sizes as well as poor shapes |
| `"ideal"` | a square or a cube at its own size | a uniform mesh; it thickens a thin layer unless that layer is `designed` |

`designed=` names the elements that keep **their own input shape** as target, whatever `target` says: a boundary layer,
a hand-made region. It takes an element tag, a list of tags or a boolean mask over the elements; at the hex rung
`designed="layer"` is enough when the layer's hexes carry the tag `"layer"`. A tag no element carries is an error.
The {doc}`demo <thin_layer>` shows when `designed` matters and when `target="input"` already holds the layer.

## Metric and solver

| knob | default | meaning |
|---|---|---|
| `metric` | `2` (quad), `303` (hex) | the shape metric. `80` (quad) / `333` (hex): shape + size, weighted by `metric_weight` (0.5). `313`: untangling, see below |
| `solver` | `"newton"` | or `"lbfgs"` |
| `iters`, `rtol` | 50, 1e-8 | iteration cap, relative residual |
| `quad_order` | 4 | Gauss-Lobatto quadrature order; 4 includes the corners, the geometry a `.re2` file keeps |

Metric `313` pulls an inverted hex mesh through (`target="ideal"` only, no `designed`). It uses finite differences and
is slow; the native `untangle_schedule` is usually the better first step.

## Reading the result

- `log.reason` gives the TMOP energy before and after, the iterations, the metric, the target, and how many vertices
  were fixed, slid or held.
- In a session the stage is one line of the round report, and the round passes or fails on the usual checks; `accept`
  or `undo` it as any round.
- A run that raises the TMOP energy, or ends with more inverted elements than it started with, is refused and the
  input returned (`log.stopped == "rolled_back"`).

## Limits

- **Order 1 only.** The vertices are the nodes. A `SmoothSession` given a curved mesh smooths its straight-sided
  shadow; `rebuild_high_order` then rebuilds the curved nodes at the quad rung (the hex rung cannot yet).
- **Planar quads only.** TMOP needs the element dimension to equal the space dimension: a quad mesh must lie in one
  plane (any plane, not only `z = 0`). A quad surface in 3-D is refused; use the native `optimize`.
- **An inverted input is refused** before MFEM is entered (MFEM would abort the process), except under metric `313`.
- **Curved targets are held**, not followed: a curved wall keeps its vertices where they are.
