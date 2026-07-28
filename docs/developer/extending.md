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

Add a `QuadMesh` classmethod that fills a boundary loop with quads and returns a
`QuadMesh` carrying `boundaries` / `boundary_tags`. Follow the existing factories:
read wall tags from the input `LineMesh` at the lowest level, accept a scalar
override arg (upper overrides lower), and build **natively in 3-D** — never flatten
to `xy`.

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
