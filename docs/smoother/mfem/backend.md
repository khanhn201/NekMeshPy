# TMOP via the MFEM backend

The `tmop` stage hands the mesh to [TMOP](https://mfem.org/howto/mesh_optimizer/), MFEM's target-matrix optimizer, and
takes the nodes back: the vertices of a linear mesh, every node of a curved one. It runs inside the {doc}`quad <../../user/quad_smoother>` and
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

The quad rung is the same with `nekmeshpy.quadmesh.smoother`. Either way the stage returns a new mesh of the input's
order; connectivity and tags never change.

## Constraints

The constraints are the smoothers' own. The stage maps each resolved vertex to fixed coordinate components. At order 1
every constrained vertex is first projected onto its target; on a curved mesh it is held at its input coordinates,
and the nodes of each edge and face follow their vertices ([High order](#high-order)).

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
| `quad_order` | 4 at order 1, `2N + 2` at order N | Gauss-Lobatto quadrature order; the rule includes the nodes, and so the corners, the geometry a `.re2` file keeps |
| `sample_order` | `max(N, 7)` | curved meshes only: the lattice the stage's log also reads the scaled Jacobian on |

Metric `313` pulls an inverted hex mesh through (`target="ideal"` only, no `designed`). It uses finite differences and
is slow; the native `untangle_schedule` is usually the better first step.

## Reading the result

- `log.reason` gives the TMOP energy before and after, the iterations, the metric, the target, and how many vertices
  were fixed, slid or held.
- In a session the stage is one line of the round report, and the round passes or fails on the usual checks; `accept`
  or `undo` it as any round.
- A run that raises the TMOP energy, or ends with more inverted elements than it started with, is refused and the
  input returned (`log.stopped == "rolled_back"`).

## High order

A curved mesh goes through TMOP as a curved mesh. Every Gauss-Lobatto node of an order-N element is an MFEM unknown,
TMOP moves all of them, and the stage returns a mesh of the same order, still conformal: a shared edge or face keeps
one set of nodes.

In code, from the {doc}`bent-pipe demo <bent_pipe>`, which walks through every step:

1. **Build curved.** Give `order=` to the factory (`linemesh.circle(..., order=3)`); the region fills and `sweep`
   carry it up. A mesh built from explicit points (`from_grid`, a plain `loft`) is high order in storage and
   straight in geometry.
2. **Constrain.** `resolve` as at order 1; the edge and face nodes take their vertices' hold (below).

   ```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
   :language: python
   :start-after: "[doc:constraints]"
   :end-before: "[/doc:constraints]"
   ```

3. **Run.** In a schedule, or in a session round, which keeps a curved mesh curved (`keep_curved=True`; `False`
   smooths the straight-sided shadow). At order N, `quad_order` defaults to `2N + 2` and `sample_order` to
   `max(N, 7)`; the session line below states both at their order-3 values.

   ```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
   :language: python
   :start-after: "[doc:tmop]"
   :end-before: "[/doc:tmop]"
   :dedent: 4
   ```

   ```python
   s = hs.SmoothSession(mesh, [], boundary="slide")          # keep_curved=True by default
   p = s.propose([("tmop", {"target": "input", "quad_order": 8, "sample_order": 7})])
   ```

4. **Read** at the solver's order and at the corners.

   ```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
   :language: python
   :start-after: "[doc:quality]"
   :end-before: "[/doc:quality]"
   :dedent: 4
   ```

5. **Export.** `writer.to_vtu` writes VTK Lagrange cells, the curved geometry; `writer.to_re2` writes the corners.

What the stage does with a curved mesh:

- **Geometry check first.** Before optimizing, the stage reads the nodes back from MFEM and evaluates MFEM's own
  element map against the input's; a mismatch beyond rounding raises instead of smoothing a different mesh.
- **Constraints per node.** A node on an edge or a face is held in a coordinate when every vertex of that edge or face
  holds it: the nodes of a frozen wall stay bit for bit, the nodes of a face in an axis-aligned plane stay in it, and
  the nodes of an edge on an axis-aligned line stay on it. Nodes on a curved target are held. An element's interior
  nodes are free unless all its corners are held.
- **Read at the solver's order.** TMOP keeps the Jacobian positive at its quadrature points; a curved element can still
  fold between them. The stage's log (`log.before`, `log.after`) gives the minimum determinant at the quadrature
  points and the scaled Jacobian at the nodes and at `sample_order`. In a session, the round report adds the curved map
  sampled at `sample_order` and the corners alone, the geometry a `.re2` file keeps, and warns when the curved map
  folds or more hexes invert at the corners.
- **Native stages move vertices only.** `laplace` and `optimize` (and `winslow` at the quad rung) run on the
  straight-sided shadow of a curved mesh, and the curved nodes follow the vertex displacement, interpolated over each element. A defect in the
  curved nodes stays where it was; the {doc}`demo <bent_pipe>` shows the difference.
- **Curved stays curved.** `run_schedule` and `SmoothSession` keep a curved input curved: `tmop` stages run on it, and
  each run of native stages runs on its shadow. `SmoothSession(..., keep_curved=False)` smooths the shadow alone and
  returns a linear mesh.

The corner reading and the curved reading need not move together. On the demo's bent pipe TMOP improves both; on other
meshes the corners can get slightly worse while the curved map improves, and the round report says so.

## Limits

- **Curved targets are held**, at every order: a node on a curved or tilted wall keeps its position, and curved nodes
  carried by a native stage are not projected back onto a curved wall.
- **Planar quads only.** TMOP needs the element dimension to equal the space dimension: a quad mesh must lie in one
  plane (any plane, not only `z = 0`), its curved nodes included. A quad surface in 3-D is refused; use the native
  `optimize`.
- **Slower at high order.** An order-N hex carries (N+1)^3 nodes and is integrated at more points: the
  {doc}`demo's table <bent_pipe>` gives TMOP's time on the curved bent pipe and on its straight-sided shadow (a laptop,
  running serial).
- **An inverted input is refused** before MFEM is entered (MFEM would abort the process), except under metric `313`.
  On a curved mesh, inverted means a non-positive Jacobian at a quadrature point.
- **No curved rebuild at the hex rung.** `rebuild_high_order` (re-projecting curved nodes onto their targets) exists
  for quads only.
