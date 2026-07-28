# Examples

Flat, **gmsh-style meshing scripts** built on the `nekmeshpy` toolkit. Each is a
self-contained program: parameter constants at the top, then top-to-bottom code
that composes the toolkit (`HexMesh.extrude`/`loft`/`annulus`/`merge`/`from_grid`, `QuadMesh`,
`trisurf`, `io`, ...) to build a mesh, assign it to `mesh`, and export it. There are
**no mesher classes** — to change a mesh, edit the constants and re-run.

Install the package first (`pip install -e .`), then run any script from the repo
root with the repo on `PYTHONPATH` (so `nekmeshpy` imports):

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
| `bifurcation.py` | vessel surface pipeline: seam fields → cut into legs → O-grid legs (`QuadMesh.half_ogrid`) → `loft`/`merge` → smooth (uses `data/car.{vtx,tri}`) |
| `circular_pipe.py` | `QuadMesh.ogrid` disc extruded along an axis (`HexMesh.extrude`) |
| `circular_pipe_tjunction.py` | analytic three-leg junction: shared seam arcs + spine → half-O-grid legs (`QuadMesh.half_ogrid`) → `loft`/`merge` → smooth (no input geometry) |
| `rectangular_pipe.py` | `QuadMesh.structured` duct extruded along an axis (`HexMesh.extrude`) |
| `transfinite_block.py` | eight corners → trilinear grid → `HexMesh.from_grid` |
| `backward_facing_step.py` | three `QuadMesh.structured(boundary_tags=…)` rectangles → `merge` → span-`loft` (caps `front`/`back`) |
| `flow_past_cylinder.py` | `QuadMesh.annulus` (circle body → **named** square far-field loop) → span-`extrude` (body `cylinder`; sides `inlet`/`outlet`/`top`/`bottom` tagged on the outer loop; caps `front`/`back`) |
| `flow_past_plate.py` | `QuadMesh.annulus` around a thin ellipse → span-`extrude` (body `plate`; far-field sides tagged on the outer loop; caps `front`/`back`) |
| `flow_past_half_cylinder.py` | `QuadMesh.structured(boundary_tags=…)` with a semicircular-bump bottom edge → span-`loft` (caps `front`/`back`) |
| `flow_past_sphere.py` | `HexMesh.annulus` between a closed sphere surface and a closed cube surface (six `QuadMesh.from_grid` patches → `QuadMesh.merge`, per-patch `element_tag`) — wall faces tagged from the surfaces' per-quad `element_tags` (body → `sphere`; far field → `inlet`/`outlet`/…) |
| `flow_past_hemisphere.py` | five-patch half cubed-sphere on the ground, each `from_grid(face_tags=…)` → `merge` (body → `hemisphere`) |

The 2-D cross-section meshers (`QuadMesh.ogrid` / `structured` / `half_ogrid` /
`annulus`) are toolkit primitives; the scripts just supply a boundary and
sweep/stack them. The external-flow cases name their boundaries **as the mesh is
built**, and the tags flow one way down the pipeline **`LineMesh` →
`QuadMesh` edges → `HexMesh` faces**. A boundary loop can carry a tag per line element
(`LineMesh.loop([…], element_tags=[…])`), which survives `radial_match` and is copied
onto the section edges by `QuadMesh.annulus`/`ogrid`; that is how
`flow_past_cylinder.py` splits its square far field into `inlet`/`outlet`/`top`/`bottom`
— tagged once on the outer loop, no post-hoc detection. Sections can also tag their own
edges directly (`QuadMesh.structured(boundary_tags=…)`,
`QuadMesh.annulus(inner_tag=…, outer_tag=…)` for a whole ring), and structured patches
in place (`from_grid(face_tags=…)`). All of these propagate onto the swept side faces via
`loft`/`extrude`, which also name the end caps (`first_tag=…, last_tag=…`). Faces welded
away by `merge` are left untagged so no stale interior tag survives.

## Tested via the toolkit

The scripts double as integration tests: `tests/conftest.py` runs them with
`runpy.run_path` and inspects the resulting `mesh` global (plus the byte-exact
golden `.re2`/`.rea`/`.vtk` for the bifurcation). So they must keep producing a
valid mesh, but they read as ordinary scripts, not a library.
