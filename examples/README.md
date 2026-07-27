# Examples

Flat, **gmsh-style meshing scripts** built on the `nekmeshpy` toolkit. Each is a
self-contained program: parameter constants at the top, then top-to-bottom code
that composes the toolkit (`HexMesh.extrude`/`loft`/`merge`/`from_grid`, `QuadMesh`,
`ops`, `io`, ...) to build a mesh, assign it to `mesh`, and export it. There are
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
| `backward_facing_step.py` | three `QuadMesh.structured(boundary_names=…)` rectangles → `merge` → span-`loft` (caps `front`/`back`) |
| `flow_past_cylinder.py` | `QuadMesh.annulus(inner_name="cylinder", outer_name={…})` (circle → square ring) → span-`loft` (caps `front`/`back`) |
| `flow_past_plate.py` | `QuadMesh.annulus(inner_name="plate", outer_name={…})` around a thin ellipse → span-`loft` (caps `front`/`back`) |
| `flow_past_half_cylinder.py` | `QuadMesh.structured(boundary_names=…)` with a semicircular-bump bottom edge → span-`loft` (caps `front`/`back`) |
| `flow_past_sphere.py` | six-patch cubed-sphere shell, each `from_grid(face_tags=…)` → `HexMesh.merge` (body → `sphere`) |
| `flow_past_hemisphere.py` | five-patch half cubed-sphere on the ground, each `from_grid(face_tags=…)` → `merge` (body → `hemisphere`) |

The 2-D cross-section meshers (`QuadMesh.ogrid` / `structured` / `half_ogrid` /
`annulus`) are toolkit primitives; the scripts just supply a boundary and
sweep/stack them. The external-flow cases name their boundaries **as the mesh is
built** — the section tags its own outer edges (`QuadMesh.structured(boundary_names=…)` /
`QuadMesh.annulus(inner_name=…, outer_name=…)`), those propagate onto the swept side
faces, the sweep names its end caps (`loft(first_cap=…, last_cap=…)`), and structured
patches are tagged in place (`from_grid(face_tags=…)`). Faces welded away by `merge`
are left untagged so no stale interior tag survives.

## Tested via the toolkit

The scripts double as integration tests: `tests/conftest.py` runs them with
`runpy.run_path` and inspects the resulting `mesh` global (plus the byte-exact
golden `.re2`/`.rea`/`.vtk` for the bifurcation). So they must keep producing a
valid mesh, but they read as ordinary scripts, not a library.
