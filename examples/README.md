# Examples

Flat, **gmsh-style meshing scripts** built on the `nekmeshpy` toolkit: constants
at the top, top-to-bottom code that composes the toolkit into a `mesh` global and
exports it. No mesher classes — to change a mesh, edit the constants and re-run.

Install the package (`pip install -e .`), then run any script from the repo root
with the repo on `PYTHONPATH`:

```bash
PYTHONPATH=. python examples/carotid.py            # vessel surface mesher (car case)
PYTHONPATH=. python examples/circular_pipe.py          # all-hex O-grid circular pipe
PYTHONPATH=. python examples/circular_pipe_tjunction.py # analytic all-hex pipe T-junction
PYTHONPATH=. python examples/quadrant_pipe_tjunction.py # welded small-branch T-junction (quadrant blocks)
PYTHONPATH=. python examples/chimera.py                # N_COPIES two-manifold units chained, alternating connector pipe
PYTHONPATH=. python examples/chimera_full.py           # chimera's two ports fed by one serpentine coil
PYTHONPATH=. python examples/serpentine_pipe.py        # coil pipe swept along a bent path
PYTHONPATH=. python examples/rectangular_pipe.py       # structured rectangular duct
PYTHONPATH=. python examples/transfinite_block.py      # corner-defined structured block
PYTHONPATH=. python examples/backward_facing_step.py   # backward-facing step channel
PYTHONPATH=. python examples/flow_past_cylinder.py     # external flow: circular cylinder
PYTHONPATH=. python examples/flow_past_plate.py        # external flow: thin plate
PYTHONPATH=. python examples/flow_past_half_cylinder.py # external flow: half-cylinder bump
PYTHONPATH=. python examples/flow_past_sphere.py       # external flow: sphere (cubed-sphere)
PYTHONPATH=. python examples/flow_past_hemisphere.py   # external flow: hemisphere on ground
```

Each writes native Nek5000/NekRS `.re2` plus a `.vtu` for ParaView; `carotid.py`
also writes a Nek field file (`carotid0.f00001`) carrying its high-order GLL nodes.

| script | what it builds |
|---|---|
| `carotid.py` | vessel surface pipeline: seam fields → cut into legs → O-grid legs (`quadmesh.spined_ogrid`) → `loft`/`merge` → smooth (uses `data/car.{vtx,tri}`) |
| `circular_pipe.py` | `quadmesh.ogrid` disc extruded along an axis (`hexmesh.extrude`) |
| `circular_pipe_tjunction.py` | analytic three-leg junction: shared seam arcs + spine → spined-O-grid legs (`quadmesh.spined_ogrid`) → `loft`/`merge` (no input geometry; the STL wall polish is present but off at `ORDER = 4` — see `SMOOTH_ITERS`) |
| `quadrant_pipe_tjunction.py` | welded small-branch T-junction, four `quadmesh.quadrant_ogrid` blocks per section. **One quadrant of the main pipe *is* a quadrant of the branch**: four regions (two legs, branch stub, two crotch caps) meet at the axes-crossing point `O`, and every interface between them is a quadrant face radiating from it — the branch's footprint disc contributes its `+z` and `-z` quadrants to the two legs' composite junction faces and its two lateral quadrants to the caps, while the legs share the wide `bypass` quadrant with each other. Each crotch is filled by `hexmesh.tetra`: a quadrant face is itself a three-patch triangle (core + the two halves of its ring band), so handing three of them plus a wall patch to the generic tetrahedron split yields the octant of a 3-D O-grid — core cube + three radial slabs — with the block split the faces already carry. Exact at any order: every wall curve is carried as a parametrization and meshed with `loft_fn`, each leg's transition is a `hexmesh.loft_fn` (a plain `loft` is straight along the sweep), and the caps are nested `loft_fn` blocks evaluated at every node — so all wall nodes, high-order ones included, sit on the main or branch cylinder to `2.2e-16` at `ORDER` 1–4. Single watertight, conformal component. The construction itself is `tjunction_lib.build_tjunction`; this script is its reference caller — the junction core plus a straight extrude of each leg to its end plane |
| `chimera.py` | `N_COPIES` copies of a two-manifold unit chained along `z`, flush end to end (`L_HALF` translation per copy); the pipe that carries the connection between each pair of neighbours alternates A/B down the chain. The connector reaches the joint plane and welds away (`""`); the other pipe stops `GAP` short of it and is capped `wall` right there, so there is genuine empty space — no elements at all — between a dead-end and its neighbour, not just a differently-tagged flush cap. Only the chain's two true ends get `inlet`/`outlet`. The junction chain is built once with both outermost ends left bare, and pipe B still comes for free as `pipe_a.rotate(...)`; the four end stubs past the outermost junctions (one per pipe, per end, each its own length) are built independently — pipe B's in pipe A's own local frame and carried over by the same rotation — so nothing is derived twice **The unit itself:** two parallel manifolds joined by `N_BR` hairpin tube connectors: `N_BR` copies of a T-junction (`tjunction_lib.build_tjunction`, the same construction `quadrant_pipe_tjunction.py` uses) are chained along pipe A exactly as before (`hexmesh.translate` per junction, tolerance-welded leg ends), each branch stub extended with a short straight arm; pipe B is not authored at all — it is `pipe_a.rotate(np.pi, axis=(0, 0, 1), center=...)`, a rigid rotation (unlike a mirror, which inverts every element's Jacobian) offset purely along `x`, branches facing away from pipe A's; a `hexmesh.sweep` bend per junction carries the arm's end along a declarative turtle-walk path lifted by `paths.embed` (same pattern as `serpentine_pipe.py`) — a 180 degree U-turn, a straight run back toward `-x`, and a second U-turn restoring the original heading — with the straight run's length solved so the far end lands exactly on pipe B's own (rotated) arm end and welds by tolerance with no separately built return arm. |
| `chimera_full.py` | the whole manifold around `chimera.py`: each of its two ports is reached by a riser -> T1 junction -> elbow, and each T1's `-y` branch carries a **chain of `N_T2` T2 junctions** hanging off one another's `-y` legs at `T2_SPACING` intervals (only the last capped). The two chains mirror, so their k-th junctions face each other across their own copy of the serpentine coil — `N_T2` parallel coils stacked down `-y`, each planar in its own x-z plane, each an 8-pass photo-traced turtle walk imported straight from `serpentine_pipe.py` (`MOVES`, guarded there behind `if __name__ == "__main__"` so the import costs nothing beyond that name; the same table builds the coil standing alone in that script), placed but never rescaled. Negative-x T1 feeds chimera's *outlet*, positive-x its *inlet*, and the ports sit exactly 10 above the coil in `y`. Its real subject is **exact seams between differently-constructed pieces**, which `order > 1` forces (`hexmesh.merge` checks shared high-order nodes against `conform.entity_tol`, ~1e-9 x extent -- far tighter than a coordinate weld): `quadmesh.reindex` re-expresses one section through another's index labels (a permutation, exact by construction, where a coordinate rotation is only approximate); `hexmesh.adapter` blends a *small* pattern gap (~0.03, T1's leg vs chimera's) with both end slices exact; and `hexmesh.bridge` spans a *large* one (~0.94 median, T1's arc-length-stationed branch vs T2's uniform-angle leg) as a single `hexmesh.loft` -- rigid stubs off each side plus a blend across the gap, so there is no internal merge left to fail. All three began here and now live in the toolkit; what stays is the choice of which each seam needs. `hexmesh.boundary_mesh(..., template=)` reads each chimera port's own nodes straight off the built mesh, where a recipe reproducing one port need not reproduce the other. Where a seam can be *removed* rather than made exact it is: the inbound connector and the coil sweep as one turtle walk, because `_weld` fuses by rounding to a `tol`-sized bucket and a ~1 ULP disagreement across that seam failed to weld wherever a coordinate landed exactly on a bucket edge. `FAST` swaps the 350k-element real chain for two capped stubs carrying its exact port pattern |
| `serpentine_pipe.py` | one `quadmesh.ogrid` disc swept along an analytic serpentine centerline (`hexmesh.sweep`, `orientation="fixed"` against the coil's plane normal) — 8 passes + 7 U-bends + a hook at each end, from a turtle-walk move table of straights and arcs. Not top-bottom symmetric: passes 4/5 carry an extra `RAISE`, lifting the wide middle bridge (`U_R_MID`) that joins the two half-coils above the flanking hairpins (`U_R`). **This is `chimera_full.py`'s own coil** — the move table (`MOVES`, `TARGET_LEN`) is defined here and that script imports it directly (its own build guarded behind `if __name__ == "__main__"` so the import is side-effect-free), so the two really are one physical part, standalone here and swept between two T2 branches there. `paths.embed` lifts the 2-D walk onto the coil's plane and `hexmesh.sweep_path` stations it, landing a node exactly on every straight↔arc junction, where the curvature jumps; `TARGET_LEN` is floored by the tightest U-turn rather than by element aspect, since `sweep_fractions` rounds to the *nearest* station count and a target near a turn's own arc length collapses it to one degenerate 180° hex |
| `rectangular_pipe.py` | `quadmesh.structured` duct extruded along an axis (`hexmesh.extrude`) |
| `transfinite_block.py` | eight corners → trilinear grid → `hexmesh.from_grid` |
| `backward_facing_step.py` | three `quadmesh.rectangle(side_tags={…})` grids → `merge` → span-`loft` (caps `front`/`back`) |
| `flow_past_cylinder.py` | `quadmesh.annulus` (circle body → **named** square far-field loop) → span-`extrude` (body `cylinder`; sides `inlet`/`outlet`/`top`/`bottom` tagged on the outer loop; caps `front`/`back`) |
| `flow_past_plate.py` | `quadmesh.annulus` around a thin ellipse → span-`extrude` (body `plate`; far-field sides tagged on the outer loop; caps `front`/`back`) |
| `flow_past_half_cylinder.py` | `quadmesh.structured` over four tagged edges, the bottom one a welded semicircular bump → span-`loft` (caps `front`/`back`) |
| `flow_past_sphere.py` | `hexmesh.annulus` between a closed sphere surface and a closed cube surface (six `quadmesh.from_grid` patches → `quadmesh.merge`, per-patch `element_tag`) — wall faces tagged from the surfaces' per-quad `element_tags` (body → `sphere`; far field → `inlet`/`outlet`/…) |
| `flow_past_hemisphere.py` | five-patch half cubed-sphere on the ground, each `half_box(patch_tags=…)` → `merge` (body → `hemisphere`) |

The 2-D section meshers (`quadmesh.ogrid` / `structured` / `half_ogrid` /
`quadrant_ogrid` / `spined_ogrid` / `annulus`) are toolkit primitives; the scripts supply a boundary and sweep/stack
them. Tags flow down the pipeline **`LineMesh` → `QuadMesh` edges → `HexMesh`
faces**: a boundary loop carries a region tag on its lines
(`linemesh.loft([…], element_tags="wall", loop=True)`), which the section meshers copy
onto the section's wall edges — how
`flow_past_cylinder.py` splits its far field into `inlet`/`outlet`/`top`/`bottom`
via `linemesh.rectangle(w, h, N, side_tags={"left": "inlet", …})` — the side/face tag
arguments are **Mappings** keyed by side name (`bottom`/`right`/`top`/`left`), not
positional 4-lists, so an unnamed side is simply absent rather than an empty slot.
`structured` takes its four `edges` the same
way — either that Mapping or a 4-sequence in `bottom, right, top, left` order.
Sections can also tag edges directly
(`structured(side_tags=…)`, `annulus(inner_tag=…, outer_tag=…)`) and patches
in place (`from_grid(side_tags=…)`). All propagate onto swept side faces via
`loft`/`extrude`, which name the end caps (`first_tag=…, last_tag=…`). Faces
welded away by `merge` stay untagged.

## High-order (order-N) elements

Every mesher here is high order. Pass `order=N` to a
factory (`linemesh.circle`, `quadmesh.sphere`/`box`, `hexmesh.annulus`, …) and each
element carries `(N+1)` Gauss–Lobatto–Legendre nodes per parametric direction
(line `N+1`, quad `(N+1)²`, hex `(N+1)³`), placed on the **true** geometry the
factory owns — a circle's or arc's nodes lie on the exact circle, a shell's
inner-wall nodes on the exact sphere. Curvature is not automatic, though: a
factory fed a bare point array (`linemesh.loft`, `from_grid`) has only those
points to go on and subdivides straight between them, so hand in the analytic
curve rather than samples of it. The library default is `1` (plain linear elements), but
**every example declares its own `ORDER` constant** at the top and threads it into
the factories that accept one, so any of them can be re-run linear or curved by
editing a single number. Most ship at `ORDER = 2`; the three curved-wall junction
cases go higher — `circular_pipe_tjunction.py` at **`ORDER = 4`**,
`quadrant_pipe_tjunction.py` at **`ORDER = 3`**, and `carotid.py` at
**`ORDER = 3`**: its walls come off a scanned STL and so have no closed form to
evaluate, but it recovers one by refitting each station's ring as a **truncated
Fourier series** (`fourier_ring`, keeping the lower half of the rFFT modes of
`x`/`y`/`z` against the uniform ring parameter) and meshing that with
`linemesh.loft_fn`. The seam ring can't be refit per leg — all three legs share it — so
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
carried as their **surface parametrization** and meshed with `linemesh.loft_fn`
(sampling them into arrays and calling `loft` chords the wall); each leg's transition
is a `hexmesh.loft_fn` over the blend parameter rather than a `loft` of a section
stack, since `loft` is straight *along the sweep* and that alone put wall nodes 7.2e-4
off the cylinder at order 3; and the crotch caps are nested `loft_fn` blocks
evaluated at every node rather than `hexmesh.from_grid`, which blends straight from
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
stores the full `lx1³` GLL block per element. `carotid.py` writes one alongside its
`.re2`/`.vtu` (`EXPORT_FLD`).

## Library modules

One file here builds no mesh of its own and is imported by the ones that do; it
is listed in `LIBRARY_ONLY` in `tests/test_examples.py`, which is what exempts
it from the "must define a `mesh` global" check.

| file | what it holds |
|---|---|
| `tjunction_lib.py` | `build_tjunction(...)`, the quadrant T-junction generalized into a function so it can be called at different radii and positions. `auto_params(R_MAIN, R_BRANCH)` picks its three shape parameters from the radius ratio — `PHI_W = 5 * footprint_angle` (so each side quadrant spans twice the footprint quadrant), clamped to 60–170 degrees; `CAP_TIP_BIAS = 0.20`; and a hub that walks from near the branch wall back toward the axis as the branch grows. Measured on the order-2 scaled Jacobian over 16 radius ratios and validated on 7 more: the single fixed set the file used to ship leaves the junction **inverted** above ratio 0.8 and unbuildable near 1.0, where the automatic choice keeps the worst element above 0.10 everywhere and within 0.08 of the best those parameters can do at any ratio. The three ports are bit-identical whatever these are set to — only the junction interior changes — so re-tuning can never disturb a downstream seam. Returns the merged crotch/transition core plus the three plain cross-sections at the legs' outward ends, for a caller to continue building from rather than re-derive; an optional `branch_tag` caps the branch's own far end inline, since unlike the legs there is nothing left for a caller to add when it needs none. Used by `chimera_full.py` for both its T1 and T2 families, by `quadrant_pipe_tjunction.py` (`branch_tag="branch"`, then a plain extrude on each leg) as this function's reference caller, and by `chimera.py` (`element_tag="fluid"`, an extra arm extrusion on `disc_branch`, and each leg's own run-length extrusion — measured the same speed as chimera's previous hand-inlined copy of this construction, since the per-call cost is dominated by the quadrant/tetra assembly either way) |

`serpentine_pipe.py` plays a similar dual role without needing a separate file:
its coil move table (`MOVES`, `TARGET_LEN`) sits at module level, and everything
that actually builds and exports a mesh is guarded behind
`if __name__ == "__main__"`. Run directly it is a normal example; imported (as
`chimera_full.py` does, for those two names) it costs nothing beyond them — the
guard is what let the move table move out of a separate `coil_lib.py` and into
the one script that owns it.

## Tested via the toolkit

The scripts double as integration tests: `tests/conftest.py` runs them with
`runpy.run_path` and inspects the `mesh` global (plus the golden `.re2`/`.vtu` for
the carotid — coordinates to `1e-12`, topology and tags byte-exact). They must
keep producing a valid mesh.
