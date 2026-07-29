# Examples

Flat, **gmsh-style meshing scripts** built on the `nekmeshpy` toolkit: constants
at the top, top-to-bottom code that composes the toolkit into a `mesh` global and
exports it. No mesher classes — to change a mesh, edit the constants and re-run.

Install the package (`pip install -e .`), then run any script from the repo root
with the repo on `PYTHONPATH`:

```bash
PYTHONPATH=. python examples/bifurcation.py            # vessel surface mesher (car case)
PYTHONPATH=. python examples/circular_pipe.py          # all-hex O-grid circular pipe
PYTHONPATH=. python examples/circular_pipe_tjunction.py # analytic all-hex pipe T-junction
PYTHONPATH=. python examples/rectangular_pipe.py       # structured rectangular duct
PYTHONPATH=. python examples/transfinite_block.py      # corner-defined structured block
PYTHONPATH=. python examples/backward_facing_step.py   # backward-facing step channel
PYTHONPATH=. python examples/flow_past_cylinder.py     # external flow: circular cylinder
PYTHONPATH=. python examples/flow_past_plate.py        # external flow: thin plate
PYTHONPATH=. python examples/flow_past_half_cylinder.py # external flow: half-cylinder bump
PYTHONPATH=. python examples/flow_past_sphere.py       # external flow: sphere (cubed-sphere)
PYTHONPATH=. python examples/flow_past_hemisphere.py   # external flow: hemisphere on ground
```

Each writes native Nek5000/NekRS `.re2`/`.rea` plus a `.vtk` for ParaView.

| script | what it builds |
|---|---|
| `bifurcation.py` | vessel surface pipeline: seam fields → cut into legs → O-grid legs (`QuadMesh.spined_ogrid`) → `loft`/`merge` → smooth (uses `data/car.{vtx,tri}`) |
| `circular_pipe.py` | `QuadMesh.ogrid` disc extruded along an axis (`HexMesh.extrude`) |
| `circular_pipe_tjunction.py` | analytic three-leg junction: shared seam arcs + spine → spined-O-grid legs (`QuadMesh.spined_ogrid`) → `loft`/`merge` → smooth (no input geometry) |
| `rectangular_pipe.py` | `QuadMesh.structured` duct extruded along an axis (`HexMesh.extrude`) |
| `transfinite_block.py` | eight corners → trilinear grid → `HexMesh.from_grid` |
| `backward_facing_step.py` | three `QuadMesh.structured(boundary_tags=…)` rectangles → `merge` → span-`loft` (caps `front`/`back`) |
| `flow_past_cylinder.py` | `QuadMesh.annulus` (circle body → **named** square far-field loop) → span-`extrude` (body `cylinder`; sides `inlet`/`outlet`/`top`/`bottom` tagged on the outer loop; caps `front`/`back`) |
| `flow_past_plate.py` | `QuadMesh.annulus` around a thin ellipse → span-`extrude` (body `plate`; far-field sides tagged on the outer loop; caps `front`/`back`) |
| `flow_past_half_cylinder.py` | `QuadMesh.structured(boundary_tags=…)` with a semicircular-bump bottom edge → span-`loft` (caps `front`/`back`) |
| `flow_past_sphere.py` | `HexMesh.annulus` between a closed sphere surface and a closed cube surface (six `QuadMesh.from_grid` patches → `QuadMesh.merge`, per-patch `element_tag`) — wall faces tagged from the surfaces' per-quad `element_tags` (body → `sphere`; far field → `inlet`/`outlet`/…) |
| `flow_past_hemisphere.py` | five-patch half cubed-sphere on the ground, each `from_grid(face_tags=…)` → `merge` (body → `hemisphere`) |

The 2-D section meshers (`QuadMesh.ogrid` / `structured` / `half_ogrid` /
`spined_ogrid` / `annulus`) are toolkit primitives; the scripts supply a boundary and sweep/stack
them. Tags flow down the pipeline **`LineMesh` → `QuadMesh` edges → `HexMesh`
faces**: a boundary loop carries a tag per line element
(`LineMesh.loop([…], element_tags=[…])`), which is copied onto section edges — how
`flow_past_cylinder.py` splits its far field into `inlet`/`outlet`/`top`/`bottom`
via `LineMesh.rectangle(w, h, N, side_tags=[…])`. Sections can also tag edges directly
(`structured(boundary_tags=…)`, `annulus(inner_tag=…, outer_tag=…)`) and patches
in place (`from_grid(face_tags=…)`). All propagate onto swept side faces via
`loft`/`extrude`, which name the end caps (`first_tag=…, last_tag=…`). Faces
welded away by `merge` stay untagged.

## Tested via the toolkit

The scripts double as integration tests: `tests/conftest.py` runs them with
`runpy.run_path` and inspects the `mesh` global (plus the byte-exact golden
`.re2`/`.rea`/`.vtk` for the bifurcation). They must keep producing a valid mesh.
