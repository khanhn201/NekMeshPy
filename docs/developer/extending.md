# Extending NekMeshPy

The toolkit is designed to be extended at a few well-defined seams. In every case,
prefer adding a **free function** or a **factory classmethod** over adding methods
to the pure-data containers (see {doc}`architecture`).

## A new smoothing strategy

Register a `fn(qm, **opts)` that repositions one `QuadMesh` cross-section's
interior points in place:

```python
from nekmeshpy import register_section_smoothing

@register_section_smoothing("my_method")
def _smooth(qm, **opts):
    qm.points[interior] = ...   # mutate in place
    return qm
```

It is then available anywhere via `smoothing_method="my_method"` and appears in
the `SECTION_METHODS` registry.

## A new cross-section factory

Add a `QuadMesh` classmethod that fills a boundary loop with quads and returns a
`QuadMesh` carrying `boundaries` / `boundary_tags`. Follow the existing factories
(`structured` / `ogrid` / `half_ogrid` / `annulus`): read wall tags from the input
`LineMesh` at the lowest level, and accept a scalar override arg (upper overrides
lower). Build **natively in 3-D** — never flatten to `xy`.

## A new geometry mesher

Write a script (like those in `examples/`) that builds `QuadMesh` cross-sections
and composes the `HexMesh` factories:

- `extrude(section, axis=…, length=…, layers=…, …)` — sweep one section along a
  straight axis,
- `loft(slices, …)` — recombine pre-positioned profiles,
- `merge([...])` — stitch blocks,
- `from_grid(P, face_tags=…)` — a structured block.

Assign the result to a `mesh` global and export; the test harness picks it up.

## External-flow domains

Name the boundaries **as you build, at the lowest level**, so the tags ride up the
ladder through construction (no post-hoc boundary detection):

- tag the `LineMesh` inputs with `element_tags=` (e.g.
  `LineMesh.circle(r, n, element_tags=["cylinder"]*n)`, or a per-edge tag for
  `structured`); the section factories read them onto the section outer edges,
  which then propagate onto the swept side faces via `loft` / `extrude`;
- the section-factory tag args (`structured(boundary_tags=…)`, `ogrid(wall_tag=…)`,
  `half_ogrid(wall_tag=…)`, `annulus(inner_tag=…, outer_tag=…)`) are **overrides**
  — a non-empty value replaces the line-level tag for that side/wall;
- name the sweep end caps with `loft(…, first_tag=…, last_tag=…)` (hex level);
- leave a face welded away by `merge` **untagged** (`NO_BOUNDARY` / an omitted
  side) so merge stays a plain concatenate with no stale interior tag.

See the `examples/flow_past_*.py` scripts.

## Physical groups

Build with plain boundary **names**, then pass `groups=` to the exporters
(`to_re2` / `to_vtk` / `to_mesh` / …) to map each name to a Nek BC code / integer
id: a `{name: nek_code}` dict, a `PhysicalGroups` (use a preset such as
`PhysicalGroups.nek_default()` for byte-exact codes), or `None` to auto-number the
mesh's distinct names.

## Sizing fields

Subclass `Field` and feed it to a size-field-aware mesher.
