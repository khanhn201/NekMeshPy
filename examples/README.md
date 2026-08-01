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
PYTHONPATH=. python examples/quadrant_pipe_tjunction.py # welded small-branch T-junction (quadrant blocks)
PYTHONPATH=. python examples/serpentine_pipe.py        # coil pipe swept along a bent path
PYTHONPATH=. python examples/rectangular_pipe.py       # structured rectangular duct
PYTHONPATH=. python examples/transfinite_block.py      # corner-defined structured block
PYTHONPATH=. python examples/backward_facing_step.py   # backward-facing step channel
PYTHONPATH=. python examples/flow_past_cylinder.py     # external flow: circular cylinder
PYTHONPATH=. python examples/flow_past_plate.py        # external flow: thin plate
PYTHONPATH=. python examples/flow_past_half_cylinder.py # external flow: half-cylinder bump
PYTHONPATH=. python examples/flow_past_sphere.py       # external flow: sphere (cubed-sphere)
PYTHONPATH=. python examples/flow_past_hemisphere.py   # external flow: hemisphere on ground
PYTHONPATH=. python examples/high_order_curve.py       # order-2 circle: nodes on the true arc
PYTHONPATH=. python examples/high_order_quad.py        # order-2 cubed-sphere surface
PYTHONPATH=. python examples/high_order_hex.py         # order-2 spherical shell (curved volume)
```

Each writes native Nek5000/NekRS `.re2` plus a `.vtu` for ParaView; `bifurcation.py`
also writes a Nek field file (`bifurcation0.f00001`) carrying its high-order GLL nodes.

| script | what it builds |
|---|---|
| `bifurcation.py` | vessel surface pipeline: seam fields → cut into legs → O-grid legs (`QuadMesh.spined_ogrid`) → `loft`/`merge` → smooth (uses `data/car.{vtx,tri}`) |
| `circular_pipe.py` | `QuadMesh.ogrid` disc extruded along an axis (`HexMesh.extrude`) |
| `circular_pipe_tjunction.py` | analytic three-leg junction: shared seam arcs + spine → spined-O-grid legs (`QuadMesh.spined_ogrid`) → `loft`/`merge` (no input geometry; the STL wall polish is present but off at `ORDER = 4` — see `SMOOTH_ITERS`) |
| `quadrant_pipe_tjunction.py` | welded small-branch T-junction, four `QuadMesh.quadrant_ogrid` blocks per section. **One quadrant of the main pipe *is* a quadrant of the branch**: four regions (two legs, branch stub, two crotch caps) meet at the axes-crossing point `O`, and every interface between them is a quadrant face radiating from it — the branch's footprint disc contributes its `+z` and `-z` quadrants to the two legs' composite junction faces and its two lateral quadrants to the caps, while the legs share the wide `bypass` quadrant with each other. Each crotch is filled by `HexMesh.tetra`: a quadrant face is itself a three-patch triangle (core + the two halves of its ring band), so handing three of them plus a wall patch to the generic tetrahedron split yields the octant of a 3-D O-grid — core cube + three radial slabs — with the block split the faces already carry. Exact at any order: every wall curve is carried as a parametrization and meshed with `loft_curve`, each leg's transition is a `HexMesh.loft_curve` (a plain `loft` is straight along the sweep), and the caps are nested `loft_curve` blocks evaluated at every node — so all wall nodes, high-order ones included, sit on the main or branch cylinder to `2.2e-16` at `ORDER` 1–4. Single watertight, conformal component |
| `serpentine_pipe.py` | one `QuadMesh.ogrid` disc swept along an analytic serpentine centerline (`HexMesh.sweep`, `orientation="fixed"` against the coil's plane normal) — 8 passes + 7 U-bends + inlet/outlet Z-offsets, from a turtle-walk move table of straights and arcs; `LineMesh.sweep_fractions` lands a sweep station exactly on every straight↔arc junction, where the curvature jumps |
| `rectangular_pipe.py` | `QuadMesh.structured` duct extruded along an axis (`HexMesh.extrude`) |
| `transfinite_block.py` | eight corners → trilinear grid → `HexMesh.from_grid` |
| `backward_facing_step.py` | three `QuadMesh.rectangle(side_tags={…})` grids → `merge` → span-`loft` (caps `front`/`back`) |
| `flow_past_cylinder.py` | `QuadMesh.annulus` (circle body → **named** square far-field loop) → span-`extrude` (body `cylinder`; sides `inlet`/`outlet`/`top`/`bottom` tagged on the outer loop; caps `front`/`back`) |
| `flow_past_plate.py` | `QuadMesh.annulus` around a thin ellipse → span-`extrude` (body `plate`; far-field sides tagged on the outer loop; caps `front`/`back`) |
| `flow_past_half_cylinder.py` | `QuadMesh.structured` over four tagged edges, the bottom one a welded semicircular bump → span-`loft` (caps `front`/`back`) |
| `flow_past_sphere.py` | `HexMesh.annulus` between a closed sphere surface and a closed cube surface (six `QuadMesh.from_grid` patches → `QuadMesh.merge`, per-patch `element_tag`) — wall faces tagged from the surfaces' per-quad `element_tags` (body → `sphere`; far field → `inlet`/`outlet`/…) |
| `flow_past_hemisphere.py` | five-patch half cubed-sphere on the ground, each `from_grid(face_tags=…)` → `merge` (body → `hemisphere`) |
| `high_order_curve.py` | `LineMesh.circle(order=ORDER)` — every arc node placed on the true circle; `.vtu` `VTK_LAGRANGE_CURVE` |
| `high_order_quad.py` | `QuadMesh.sphere(order=ORDER)` — all `(N+1)²` surface nodes on the true sphere; `.vtu` `VTK_LAGRANGE_QUADRILATERAL` |
| `high_order_hex.py` | `HexMesh.annulus(sphere, cube)` shell over two `order=ORDER` surfaces — curved inner wall on the true sphere; `.re2` stays linear, `.vtu` `VTK_LAGRANGE_HEXAHEDRON` |

The 2-D section meshers (`QuadMesh.ogrid` / `structured` / `half_ogrid` /
`quadrant_ogrid` / `spined_ogrid` / `annulus`) are toolkit primitives; the scripts supply a boundary and sweep/stack
them. Tags flow down the pipeline **`LineMesh` → `QuadMesh` edges → `HexMesh`
faces**: a boundary loop carries a tag per line element
(`LineMesh.loft([…], element_tags=[…], loop=True)`), which is copied onto section edges — how
`flow_past_cylinder.py` splits its far field into `inlet`/`outlet`/`top`/`bottom`
via `LineMesh.rectangle(w, h, N, side_tags={"left": "inlet", …})` — the side/face tag
arguments are **Mappings** keyed by side name (`bottom`/`right`/`top`/`left`), not
positional 4-lists, so an unnamed side is simply absent rather than an empty slot.
`structured` takes its four `edges` the same
way — either that Mapping or a 4-sequence in `bottom, right, top, left` order.
Sections can also tag edges directly
(`structured(side_tags=…)`, `annulus(inner_tag=…, outer_tag=…)`) and patches
in place (`from_grid(face_tags=…)`). All propagate onto swept side faces via
`loft`/`extrude`, which name the end caps (`first_tag=…, last_tag=…`). Faces
welded away by `merge` stay untagged.

## High-order (order-N) elements

The `high_order_*.py` trio show spectral-element output: pass `order=N` to a
factory (`LineMesh.circle`, `QuadMesh.sphere`/`box`, `HexMesh.annulus`, …) and each
element carries `(N+1)` Gauss–Lobatto–Legendre nodes per parametric direction
(line `N+1`, quad `(N+1)²`, hex `(N+1)³`), placed on the **true** geometry the
factory owns — a circle's or arc's nodes lie on the exact circle, a shell's
inner-wall nodes on the exact sphere. Curvature is not automatic, though: a
factory fed a bare point array (`LineMesh.loft`, `from_grid`) has only those
points to go on and subdivides straight between them, so hand in the analytic
curve rather than samples of it. The library default is `1` (plain linear elements), but
**every example declares its own `ORDER` constant** at the top and threads it into
the factories that accept one, so any of them can be re-run linear or curved by
editing a single number. Most ship at `ORDER = 2`; the three curved-wall junction
cases go higher — `circular_pipe_tjunction.py` at **`ORDER = 4`**,
`quadrant_pipe_tjunction.py` at **`ORDER = 3`**, and `bifurcation.py` at
**`ORDER = 3`**: its walls come off a scanned STL and so have no closed form to
evaluate, but it recovers one by refitting each station's ring as a **truncated
Fourier series** (`fourier_ring`, keeping the lower half of the rFFT modes of
`x`/`y`/`z` against the uniform ring parameter) and meshing that with
`LineMesh.loft_curve`. The seam ring can't be refit per leg — all three legs share it — so
its three **arcs** are instead refit once, globally, by `_arc_curve`: a truncated
**sine** series for the arc's deviation from its own chord, whose every mode vanishes
at both ends, so the triple points `A1`/`A2` stay bit-exact and the legs still weld.
Without it that one station stayed straight-sided and showed 63° of corner at its
element joints while every other station sat within 0.2°.
Low-passing the facet-scale noise also un-pinches the worst wall
elements (min scaled Jacobian 0.0281 → 0.1207), so **both smoothers are off**
(`SMOOTHING_METHOD = "none"`, `SMOOTH_ITERS = 0`) — they move corner nodes only and
reject `order > 1` anyway. Its spine stays linear, which is all a flat half-disc seam
needs. `quadrant_pipe_tjunction.py` runs at `ORDER = 3` and is where every
straight-subdivision trap bites at once, so it dodges all three: wall curves are
carried as their **surface parametrization** and meshed with `LineMesh.loft_curve`
(sampling them into arrays and calling `loft` chords the wall); each leg's transition
is a `HexMesh.loft_curve` over the blend parameter rather than a `loft` of a section
stack, since `loft` is straight *along the sweep* and that alone put wall nodes 7.2e-4
off the cylinder at order 3; and the crotch caps are nested `loft_curve` blocks
evaluated at every node rather than `HexMesh.from_grid`, which blends straight from
corners and would both leave the wall and disagree with `quadrant_ogrid`'s bowed ring
bands along the faces they share. Corner coordinates come out bit-identical at
`ORDER` 1–4 and every wall node sits on the main or branch cylinder to 2.2e-16.
`circular_pipe_tjunction.py`
runs at `ORDER = 4` with its post-assembly smoothing switched off
(`SMOOTH_ITERS = 0`) for the same reason — set `ORDER = 1` and `SMOOTH_ITERS = 8`
to exercise it. `.re2` export **stays linear** (8 corners/hex — Nek's
re2 has no high-order format yet); `.vtu` export emits VTK Lagrange cells that a
viewer renders as curved geometry via the XML `.vtu` writer (`export.to_vtu` /
`line_to_vtu` / `quad_to_vtu`) — ParaView and VisIt render Lagrange cells reliably
from `.vtu`. To hand the curved geometry to **Nek** rather than to a viewer, use the
field-file writer `export.to_fld` (`<prefix>0.f00001`, `fields="X"`): unlike `.re2` it
stores the full `lx1³` GLL block per element. `bifurcation.py` writes one alongside its
`.re2`/`.vtu` (`EXPORT_FLD`).

## Tested via the toolkit

The scripts double as integration tests: `tests/conftest.py` runs them with
`runpy.run_path` and inspects the `mesh` global (plus the golden `.re2`/`.vtu` for
the bifurcation — coordinates to `1e-12`, topology and tags byte-exact). They must
keep producing a valid mesh.
