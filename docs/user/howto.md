# How-to recipes

Task-oriented recipes, each distilled from a runnable script under
[`examples/`](https://github.com/nekmeshpy/nekmeshpy/tree/main/examples). The
examples are flat, gmsh-style scripts (constants at the top, top-to-bottom code,
assign to a `mesh` global, export) — edit the constants and re-run:

```bash
PYTHONPATH=. python examples/circular_pipe.py    # writes .re2/.rea/.vtk in cwd
```

The test suite executes these scripts and inspects the `mesh` global, so they
double as integration tests. If a recipe here and its script ever disagree, the
**script is the source of truth**.

## Build an O-grid ("butterfly") pipe

Fill a tagged circular loop with an O-grid section and sweep it along an axis.

```python
boundary = LineMesh.circle(radius, 4 * n_side, element_tags=["wall"] * (4 * n_side))
section  = QuadMesh.ogrid(boundary, n_side=n_side, radial=uniform_spacing(4),
                          smoothing_method="bilinear")
mesh     = HexMesh.extrude(section, axis=(0, 0, 1), length=L,
                           layers=uniform_spacing(n_axial),
                           first_tag="inlet", last_tag="outlet")
```

→ `examples/circular_pipe.py`. For a branching pipe with an internal tagged
interface, see `examples/circular_pipe_tjunction.py`.

## Build a structured duct

Give `QuadMesh.structured` four open `LineMesh` edges, each tagged with its own
uniform side name; resolution comes from the edges' own points.

→ `examples/rectangular_pipe.py`, and `examples/transfinite_block.py` for the
eight-corners → trilinear → `HexMesh.from_grid` path.

## External flow past a body (2-D section, swept)

Wrap a body in an annular O-grid or structured patches, tag the body wall and each
far-field side at the line level, and sweep along the span with `loft`, naming the
two end caps.

- `examples/flow_past_cylinder.py` — `QuadMesh.annulus` between a circle and a
  tagged far-field box (the box's per-line tags split the outer ring into distinct
  sides automatically).
- `examples/flow_past_plate.py` — the same pattern around a thin ellipse.
- `examples/flow_past_half_cylinder.py` — a single `structured` section whose
  bottom edge is one composite curve (ground → semicircular bump → ground).
- `examples/backward_facing_step.py` — merged structured rectangles.

## External flow past a sphere (3-D shell)

Build a closed far-field cube surface from six `QuadMesh.from_grid` patches (each
patch carries the far-field side as its per-quad `element_tag`), derive the body
surface on the **same connectivity**, and fill the shell with `HexMesh.annulus`:

```python
cube   = HexMesh.merge([...six from_grid patches...])   # closed QuadMesh surface
sphere = QuadMesh(R * normalize(cube.points), cube.quads)   # same quads, index-paired
mesh   = HexMesh.annulus(sphere, cube, radial=geometric_spacing(n, ratio))
```

→ `examples/flow_past_sphere.py`, and `examples/flow_past_hemisphere.py` for the
half cubed-sphere shell.

## Stitch multiple blocks

Build each block independently, then weld coincident seam points with
`HexMesh.merge`. Leave a face welded away by `merge` **untagged**
(`NO_BOUNDARY` / an omitted side) so merge stays a plain concatenate with no stale
interior tag.

```python
mesh = HexMesh.merge([block_a, block_b])   # welds coincident boundary points only
```

→ used throughout the `flow_past_*` examples and the bifurcation pipeline.

## The bifurcation vessel pipeline

The original motivating case: a `TriMesh` surface (`data/car.{vtx,tri}`) is cut
into legs via seam fields, each leg is filled with `half_ogrid`, extruded/merged,
and smoothed. → `examples/bifurcation.py` (this is the golden-regression case; see
{doc}`../developer/contributing`).
