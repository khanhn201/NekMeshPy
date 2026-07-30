# Extending NekMeshPy

Extend at a few well-defined seams. In every case prefer a **free function** or
**factory classmethod** over methods on the pure-data containers (see
{doc}`architecture`).

## A new smoothing strategy

Register a `fn(qm, **opts)` that repositions one section's interior points in
place:

```python
from nekmeshpy import register_section_smoothing

@register_section_smoothing("my_method")
def _smooth(qm, **opts):
    qm.points[interior] = ...   # mutate in place
    return qm
```

It is then available via `smoothing_method="my_method"` and appears in the
`SECTION_METHODS` registry.

## A new cross-section factory

Factories are **plain free functions** (no `cls` / `self`) that live beside the
container and are bound onto the class by the package `__init__`. Adding one touches
exactly one file: write the function in `quadmesh/_open.py` (region fills, open
curves) or `quadmesh/_closed.py` (parametric closed surfaces) and add one entry to
that module's `FACTORIES` dict. The binding loop in `quadmesh/__init__.py` picks it
up, so `QuadMesh.my_factory(...)` works with no edit to `quadmesh.py`. Line factories
work the same way in `linemesh/_open.py` / `_closed.py`. Internal toolkit code calls
the free function directly (`from ._open import ogrid`) — `mypy` pins
`files=["nekmeshpy"]` and cannot see the dynamically bound names.

Return a `QuadMesh` carrying `boundaries` / `boundary_tags`. Follow the existing
factories: read wall tags from the input `LineMesh` at the lowest level, accept a
scalar override arg (upper overrides lower), and build **natively in 3-D** — never
flatten to `xy`.

### Supporting `order=N`

Take an `order: int = 1` keyword and decide which of the two node-placement
strategies your factory offers (see
[Concepts](../user/concepts.md#true-geometry-vs-straight-subdivision)):

- **Analytic shape you own** — evaluate the shape at the interior GLL parameters and
  pass the resulting nodes explicitly (`LineMesh.circle` / `LineMesh.arc` hand `loft` an
  `interior` computed on the exact arc; `QuadMesh.sphere` / `QuadMesh.hemisphere`
  project the cube's / half box's whole B-rep — `points`, `lines.interior`, `interior` —
  radially). Apply the map **entity-wise**, never to a reassembled per-element block: a
  shared edge must land in the same place seen from either incident element.
- **Region fill from given points** — build the linear mesh, then call
  `_elevate(qm, order, overlays)` from `quadmesh/_helpers.py`. It fills a straight
  tensor-subdivided interior and stamps any `Overlay` `(quad ids, local side, curve)`
  you pass onto that side, so a curved input wall survives elevation. Pass one overlay
  per bounding wall (`structured` does all four sides; `ogrid` / `half_ogrid` their
  outer ring), mapping each curve interval to the element it bounds. Elevate
  **before** smoothing, so a repositioning smoother sees the true order and raises
  cleanly.

Reject a mismatched `order` across your inputs, and remember `order == 1` must be a
byte-exact no-op — the golden regression depends on it.

## A new geometry mesher

Write a script (like those in `examples/`) that builds `QuadMesh` sections and
composes the `HexMesh` factories:

- `extrude(section, axis=…, length=…, layers=…, …)` — sweep one section along a
  straight axis,
- `loft(slices, …)` — recombine pre-positioned profiles,
- `merge([...])` — stitch blocks,
- `from_grid(P, face_tags=…)` — a structured block.

Assign the result to a `mesh` global and export; the test harness picks it up.

## External-flow domains

Name boundaries **as you build, at the lowest level**, so tags ride up the ladder
(no post-hoc detection):

- tag the `LineMesh` inputs with `element_tags=` (e.g.
  `LineMesh.circle(r, n, element_tags=["cylinder"]*n)`, or per-edge for
  `structured`); the factories read them onto the outer edges, which propagate onto
  the swept side faces via `loft` / `extrude`;
- the section-factory tag args (`structured(boundary_tags=…)`, `ogrid(wall_tag=…)`,
  `half_ogrid(wall_tag=…)`, `annulus(inner_tag=…, outer_tag=…)`) are **overrides**;
- name sweep end caps with `loft(…, first_tag=…, last_tag=…)` (hex level);
- leave a welded-away face **untagged** (`NO_BOUNDARY` / omitted side).

See the `examples/flow_past_*.py` scripts.

## Physical groups

Build with plain boundary **names**, then pass `groups=` to the exporters to map
each name to a Nek BC code / integer id: a `{name: nek_code}` dict, a
`PhysicalGroups` (e.g. the `PhysicalGroups.nek_default()` preset), or `None` to
auto-number.

## Sizing fields

Subclass `Field` and feed it to a size-field-aware mesher.
