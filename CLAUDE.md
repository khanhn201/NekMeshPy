# CLAUDE.md

## Commands

```bash
pip install -e ".[all,dev]"

ruff check nekmeshpy tests examples docs
mypy                          # config pins files=["nekmeshpy"]; do NOT pass paths
python -m pytest              # addopts deselect `slow`
python -m pytest -m slow      # femoral, the one gmsh example
python docs/_ext/gen_viewer_assets.py   # regenerate the gallery's .vtp assets
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

The four checks and `sphinx-build` **are** the CI gates. A local pass is not the gate
— after pushing, poll `gh pr checks <n>`. `gen_viewer_assets.py` isn't a gate itself
(`sphinx-build` doesn't check the `.vtp` files exist, so a docs build passes without it)
but the deployed gallery needs it run first — `.github/workflows/docs.yml` does so
before every `sphinx-build`.

The docs build is nitpicky (`-n -W`), so every `:func:`/`:class:` role needs a fully
qualified target naming the *namespace* module, not the private container module:
``:func:`quadmesh.blend <nekmeshpy.quadmesh.morph.blend>` ``.

## Golden regression

`tests/golden/` freezes `examples/carotid.py`: **geometry to 1e-12, topology and
tags exactly** (connectivity, numbering, VTK cell types, `bc_id`, and the `.re2`
boundary block byte-for-byte). Floats are not byte-compared — interpreter builds shift
them ~7e-13.

The numerics were ported verbatim from reference MATLAB, so most refactors here are
expected to be output-preserving. After touching geometry, verify directly:

```bash
cd /tmp && PYTHONPATH=<repo> python <repo>/examples/carotid.py
python -m pytest <repo>/tests/test_regression.py
```

## Architecture

**Two layers.** `nekmeshpy/` is a toolkit with *no* geometry-specific meshers.
`examples/` holds the concrete meshers as flat, gmsh-style scripts: constants at top,
top-to-bottom code, assign to a `mesh` global, export. There are no mesher classes by
design. `tests/test_examples.py` discovers and runs every script, asserting it leaves a
valid `mesh` — so a new example is covered the moment it lands.

**Containers are pure data; everything acting on a finished mesh is a free function**
taking the container first. `<rung>/<rung>.py` holds storage, validation and derived
views only, and imports **no sibling** — that one-way edge is what keeps each package a
DAG. Operations live in sibling modules, reached through per-rung namespaces, never
bound onto the class:

```python
quadmesh.ogrid(boundary, n_side, radial)    # not QuadMesh.ogrid(...)
hexmesh.is_watertight(mesh)                 # not mesh.is_watertight()
```

No `TYPE_CHECKING` guards; the three containers have no deferred imports. Siblings do
use function-body imports in a few places (`query` → `quality`, `shape` → `linemesh`),
so a deferred import in a *sibling* is not by itself a bug; one in a container is.

Each rung re-exports its operations, so `quadmesh.ogrid` and `quadmesh.shape.ogrid` are
the same function — prefer the short form. Names collide across rungs by design (each
has its own `loft`, `merge`), which is why there is no flat namespace above this one.
Adding an operation touches one sibling module plus two `__all__`s.

Siblings split on **arity** and **rung delta** (line → quad → hex):

| module | Δ | contents |
|---|---|---|
| `assemble.py` | +1 / 0 | `loft`, `loft_fn`, `loft_spline`, `merge`, `attach` — n-ary; `select` / `remove` / `components` — the inverse |
| `lift.py` | +1 | `extrude` / `sweep` / `annulus` / `from_grid`; `adapter` / `bridge` hex-only |
| `lower.py` | −1 | `boundary_mesh` — the boundary **as** a mesh one rung down |
| `morph.py` | 0 | `blend`, `translate` / `rotate` / `scale` / `transform` / `mirror`; `reindex` quad-only |
| `query.py` | exit | read-only queries, incl. `bounds` / `centroid` and the rung's own measure (`length` / `area` / `volume`); hex also topology / `report` |
| `shape.py` | +1 | shape factories — own a *shape model*, unlike `lift` |
| `tag.py` | 0 | `retag_element`; `retag_point` / `retag_edge` / `retag_face` — rename the tag vocabulary, geometry untouched. Plus the authoring bridges: `quadmesh.tag_edges` takes `(quad, side)` rows, since factories think element-locally; `hexmesh.tag_faces` takes face ids, the natural handle after a weld |

**`loft`, `merge`, `attach`, `select`/`remove`/`components` and `boundary_mesh` are the
only operations that manufacture a global index space** — `select` and its kin are `merge`
run backwards, and sit beside it for that reason. To place a new operation: *invents a
numbering?* → `assemble` (unless it is boundary extraction → `lower`); *changes rung?* →
`lift`/`lower`; *only renames tags?* → `tag`; *neither?* → `morph`. `morph` is for the
*geometry* at delta 0, which is why a retag is not in it.

A reflection has determinant −1, so `mirror` is the coordinate map **plus** a re-winding
of the connectivity — never `transform` with a reflection matrix, which inverts every
element. The line rung is the exception: no signed measure, so nothing to re-wind.

Operations are per-rung, so a call site **names its rung**. Code meant to run at every
rung pairs each mesh with its package explicitly rather than dispatching on `type()`.

`core/` is rung-agnostic: `paths.py`, `surfaces.py`, `conform.py`, `tags.py`,
`measure.py` (one quadrature over a node block, behind every rung's measure). `TriMesh`
is the exception to the free-function rule — small queries stay on the class, the rest
is `trimesh.ops.*`.

## The B-rep ladder *is* the storage

Each container holds the rung below plus what it privately owns — `LineMesh` (a
`point_mesh` *`PointMesh` of the shared points* + `lines` + `interior (L,N-1,3)`);
`QuadMesh` (a `line_mesh` *`LineMesh` of the shared edges* + `quads`/`orient` +
`interior (Q,(N-1)²,3)`); `HexMesh` (a `quad_mesh` *`QuadMesh` of the shared faces* +
`hexes`/`orient` + `interior (E,(N-1)³,3)`).

**The slot names say which of three roles they play**, so the same word never means two
things: `<rung>_mesh` is the stored container one rung down; the **plural** `lines` /
`quads` / `hexes` is this rung's own stored incidence into it; `corners` is the derived
corner connectivity at every rung; and `points` is always the `(N,3)` coordinates. At
the line rung a point *is* its own corner, so `lines` and `corners` are one table under
both names.

`points` / `corners` are **derived read-only views**, so corner consistency is
structural and `mesh.points[:] = X` propagates for free. Conformality is likewise
structural: a shared edge or face is *one stored object* referenced by every incident
element, resolved by corner ids rather than coordinate search (`core/conform.py`).

Constructors share one argument order: `(rung below, incidence, [orientation,]
interior, element_tags)` — `LineMesh(point_mesh, lines, …)`, `QuadMesh(line_mesh, quads,
orient, …)`, `HexMesh(quad_mesh, hexes, orient, …)`. A line element has no orientation
bit, so `LineMesh` has no `orient` slot.

**No container takes or stores `order`** — it derives all the way down: `HexMesh.order`
→ `quad_mesh.order` → `line_mesh.order` → **`interior.shape[1] + 1`**. A mesh cannot disagree
with the nodes it stores. The cost: `interior`'s middle axis is unconstrained, so a
wrong-sized block yields a *different order* rather than a `ValueError`. Size blocks
through a factory or `loft`.

## The trap: high order in storage vs in geometry

`order=N` is a **factory** argument only (`linemesh.circle`, `loft`, the region fills).

Anything built from an explicit point array (`linemesh.loft`, `from_grid`)
**straight-subdivides** between the given points — high order in storage, linear in
geometry. A plain `quadmesh.loft`/`hexmesh.loft` is the same trap along its *sweep*
direction: exact profiles still give a surface straight between them — a torus lofted
from 8 exact circular profiles lands ~90% of the tube radius off.

The escapes: `loft_fn` (evaluates your parametrization on the **whole** node lattice),
`sweep` (carries one profile along a curved path by a moving frame), or handing `loft`
its intermediate profiles as `sweep_nodes=`. With no parametrization to call, `loft_spline`
fits those intermediate profiles with a cubic through the whole stack — it interpolates,
so every slice handed in comes back verbatim. Region fills carry their walls' curvature
inward, and the combinators carry it up.

When testing curved geometry, assert on the **conformal node set**, not corners —
corner-only passes on a mesh that is high-order in storage and linear in geometry.

A split runs through quality too, though no longer the corner-vs-curved one: since
`7eb73de` both rungs' `scaled_jacobian` / `quality_summary` read the **curved** block
and there is no corner-only reading in the namespace (`quality.corner_scaled_jacobian`
survives for linear meshes). What replaces it is a **sampling** split. The metric is
exact at the `(order+1)**dim` GLL nodes and silent between them, while an element's
Jacobian determinant is a polynomial of far higher degree than its map — so a positive
reading is not a certificate that the element is not folded. Measured: an order-2 quad
reading `+0.75` on its own 9 nodes reads `-0.99` on 81; a hex reading `+0.35` on 27
reads `-0.12` on 729; `femoral` reads `+0.042` and `0 inverted` at order 2 and
`-0.007` with `1 inverted` at 8, which is what Nek5000 at `lx1=8` then rejects.

So **sample at the order the solver will run at**: `quality_summary(mesh, order=8)`
reads the stored map on the finer lattice (`core.interp.resample_block` — a change of
nodal basis, inventing no geometry; below `mesh.order` it refuses). Do not assume one
extra order is enough — the fold can sit between any two lattices, and that same
`femoral` element reads *positive* again at order 5. Say which order a number came from.

`hexmesh.report` does this for you: `order_scan` re-reads the mesh at `SCAN_ORDER`
(7 — **the solver's** order; there is no universal right value, change it to match
yours, and it is read live so assigning it works), prints a `sampling` line, and raises
a `** WARNING **` plus a `logging.warning` when it inverts. Deliberately one order, not
a sweep: intermediate orders cost real time and certify nothing, since the reading is
**not monotone** — a real element reads clean at 2, folded at 3, clean at 4, clean at 7
and folded at 8. At order 7 the work is 512 points a hex (0.8 s for 8k), so
`SCAN_BUDGET` admits ~29k elements and *declines* anything larger, naming it in
`OrderScan.skipped` and reporting `not checked`. Declined is **unchecked, not clean**.
The sampling is chunked, so peak memory is flat in mesh size and order (unchunked, 500k
hexes at order 8 is 8.7 GB).

Order-N smoothing is not implemented: a repositioning smoother raises
`NotImplementedError` above order 1 rather than degrading silently.

## `examples/femoral.py`: the one mesher with a solver under it

It builds its own surface, tet-meshes the interior with **gmsh** (the only thing in the
repo that needs it: `[all]`, or `[mesh]` alone; the wheel also wants `libGLU`, which the
slow-examples CI job installs), solves P1 conduction in the volume, and cuts stations as
level sets. Caches the surfaces and the tet solve under `examples/data/` — gitignored per
file, since that directory also holds *tracked* inputs (`car.vtx` / `car.tri`).

**gmsh does not tetrahedralize the same way twice** — not across machines, not run to run
on one. Measured: 157402 / 157446 / 157483 nodes for identical input. So a result checked
only against the cached tet mesh is not checked at all; three configurations passed
locally and failed CI for exactly that. Delete `examples/data/femoral_tets.npz` and rerun
before believing any quality number. This is why femoral is in `SLOW` (316 s cold, and CI
is always cold) rather than in the default run.

That non-determinism reaches the test suite: on one unchanged commit femoral's
inverted-element check went 3 passes and 2 failures (min scaled Jacobian -0.988,
-0.989), so it is in `NONDETERMINISTIC_QUALITY` — a **non-strict** xfail, unlike
`KNOWN_INVERTED` next to it, because passing is the ordinary outcome and must not be an
error either. It is marked so a draw of the dice cannot redden an unrelated change, not
because the defect is accepted: the fix is the layer thickness below, and when the
mesher no longer depends on the draw the entry comes out.

What makes the mesher independent of that draw is **layer thickness**, not tolerance.
`snap_to_wall` moves a node a fixed distance, so the distortion is that distance
*relative to its layer* — which makes the uniform run's length the stability knob, per
leg (`NEAR_LEN` 1.5 for the main pipe, `NEAR_LEN_BRANCH` 4.5, through `LEG_NEAR_LEN`).
Tripling it globally is much worse: the junction stops being resolved. Widening
`SNAP_MAX` buys wall accuracy by *trading away* that independence — at 0.20 alone the
mesh was flawless locally and corner-inverted on CI.

## Joining: `merge` infers, `attach` is told

Two welds, and the difference is *what the caller states*.

`merge(meshes, tol=)` is the **proximity** join: it is told nothing about what meets
what and infers every seam in the assembly from coordinates, at one tolerance, over
every block's whole boundary at once. That is why `examples/chimera_full.py:444-448`
runs one seam at `tol=0.05` and the assembly at `0.005` — "loosening the tolerance for
the whole assembly welded an unrelated, closer-together pair by mistake."

`attach(meshes, seams)` is told, by a `Seam` apiece, **which** group meets which — one
rung down at each level: `face_tags` at the hex rung, `edge_tags` at the quad rung,
`point_tags` at the line rung (`tagged_faces` / `tagged_edges` are the public accessors,
and either argument also takes an explicit id array). It therefore takes **no tolerance
at all**. Inside those two groups the pairing is
nearest-neighbour, and what proves it is **bijectivity** — equal point counts plus an
injective map is a one-to-one correspondence however far apart the halves sit. A seam
with a real gap joins; a seam whose halves do not correspond is refused however close
they are.

One case no tolerance could have caught either: a seam with a rotational symmetry whose
halves are relatively rotated by a symmetry element pairs injectively, bijectively, and
at distance **zero**, onto a cyclic shift — welding the block in twisted. The point sets
are identical, so no geometry distinguishes the two readings.

**Naming the interface is the work, and it happens at the rung below.** A section's
*edge* tags become the swept block's lateral *face* tags, so a seam down a block's side
is named by tagging the section that swept it. A cap is named by `first_tag` /
`last_tag`, and those take an `ElementTags` over the slice, so a cap shared with two
different neighbours — three legs meeting about a spine, where each seam is *half* a
disc — is named per element. At the line rung a slice is one point, so `first_tag` /
`last_tag` name the chain ends. Getting a half wrong cannot pass quietly: `attach` pairs
a named group against a named group and refuses anything not one-to-one.

`attach` welds **only** what it is told, so a `merge` converts all-or-nothing — stating
one of two touching seams leaves two components, not a partial weld.

**State a seam with the lower block index first** when you care about reproducing a
`merge` byte for byte. The surviving point id is the lowest of a welded pair, but
`own="a"` writes the *a*-side's coordinates -- so a seam given as `(3, 0)` keeps block
0's id carrying block 3's numbers, and two values meant to be equal can differ in the
last ulp. Closing a ring of blocks is where this bites: the wrap-around seam is the one
that comes out backwards.

`own=` picks whose nodes the seam keeps, and it is a **byte copy**, not an average: the
shared-node re-scatter in `_stitch` checks the two sides against `conform.entity_tol`,
orders tighter than any pairing distance, and a merely-close seam fails it. The welded-shut
faces are **cleared** unless `attach_tag` names them — a named interior face makes the
exporter write one boundary row from *each* side, which callers used to strip by hand.

`attach` is **n-ary** -- `attach(meshes, seams)` -- and welds the whole assembly in one
pass. Chaining two-block joins instead rebuilds the accumulated mesh per link, which is
quadratic in the block count: 32 blocks cost 63240 hex-passes chained against 3840 in
one pass, and measured 230 ms against 17 ms.

Note this inverts the cap-argument convention below: here `attach_tag=None` means
"clear", because burying a seam is what attaching is for.

**Coincidence is a radius, and only a radius.** `conform.coincident_clusters` fuses two
points when they are **strictly closer than `tol`**, transitively. It was a lattice,
`round(x / tol)`, which missed any two points straddling a cell boundary however close
they were — 1.78e-15 was enough — and a missed weld does not raise, it silently leaves a
seam open. The lattice was then kept *alongside* the radius for one release under a
"weld more, never less" rule, so the fix could not reopen a hand-tuned seam. It is gone
now, because the two rules are not redundant in the direction that matters: a shared cell
reaches `tol*sqrt(3)`, so the lattice welded pairs up to 1.73x further apart than asked —
and *which* pairs depended on where the model sat in space, since translating everything
by `tol/2` moves the cell edges. Measured: a pair 1.697*tol apart welded. It also made
`tol` a lie exactly where it was load-bearing — chimera_full picks 0.04 to stay under a
real 0.05 feature, and `0.04*sqrt(3)` is 0.069.

**`merge`'s `tol` is a fraction, not a distance** — of `conform.bbox_scale`, the largest
of the x/y/z ranges over every point handed in, so the radius is `tol * bbox_scale` and
the default `1e-7` means the same thing at any model size. It used to be a distance when
supplied and a fraction when defaulted, which is two meanings for one name. `weld_points`
refuses anything at or above `MAX_WELD_FRACTION` (0.1), since 10% of the model is not a
coincidence tolerance under any reading and is much likelier to be a distance passed by
mistake. A caller who knows a real distance divides — `examples/chimera_full.py`'s
`merge_at(blocks, d)` does exactly that, keeping the two numbers the choice turns on (a
~0.03 residual to bridge, a 0.05 feature not to fuse) visible instead of pre-divided.

## Tags

**A rung's side tags *are* the rung below's `element_tags`.** There is one table type,
`ElementTags` from `core/tags.py`, and a mesh reads the one under it through a named
property: `HexMesh.face_tags` is `quad_mesh.element_tags`, `QuadMesh.edge_tags` is
`line_mesh.element_tags`, `LineMesh.point_tags` is `point_mesh.element_tags` on the
`PointMesh` the ladder bottoms out on. A tag is addressed by **entity id**, never by
`(element, side)`.

That is what makes tag consistency structural, the way conformality already is: a face
is one stored object both its hexes reference, so it carries one name and the two sides
cannot disagree. `merge` raises when a weld would put two different names on one
entity, and a closed sweep's two caps -- the same seam -- cannot be named differently.

An **asymmetric** boundary condition therefore cannot live on the face. It lives in the
regions either side of it: `face_tag_rows` reconstructs one `(element, face)` row per
hex carrying a named face, and `GROUPS` can key the code by that hex's own region —
`{"fluid": "W  ", "solid": None}`, where `None` writes no row from that side.

**`boundary` is reserved for the topological domain boundary** — what `boundary_faces`
/ `_edges` / `_points` compute from connectivity. A side-tag table is a *named subset*
of that, and the two differ. Row order is meaningful: `ordered()` is the only sort,
because `.re2` writes rows in stored order and `.vtu` gives a node touched by several
rows the last one's tag.

`element_tags` is sparse (`ids + tags`), so an untagged mesh stores nothing and `len()`
is the *tagged* count.

**Only the top rung's `element_tags` names a region.** One rung down, an element is a
piece of some volume's *surface* — and now literally the same object as that volume's
face — so its `element_tags` is the boundary name (`"wall"`, `"inlet"`), never
`"fluid"` / `"solid"`. Not enforced, but the mechanism punishes getting it wrong:
`first_tag`/`last_tag` default to the bounding slice's own `element_tags`, so a section
tagged `"fluid"` exports its caps as a `"fluid"` **boundary condition**.
`hexmesh.extrude(..., element_tags="fluid")` is safe precisely because a hex's own
`element_tags` never becomes a face tag — nothing is above it to read them.

**`loft`'s three tag arguments are the same shape at every rung**: `element_tags` names
the *swept* elements — one string for all of them, or an `ElementTags` over **one
slice's** elements, which tags each swept column by the slice element it came from.
`first_tag` / `last_tag` take the same two shapes and name the cap sides, defaulting to
the bounding slice's own `element_tags` (a cap side *is* that slice element) — except on
a `loop`, whose caps are the interior seam and are named only when asked. A slice at the
line rung is a single point, so all three reduce to one string there. Nothing is
inherited implicitly: an untagged argument tags nothing.

`NO_TAG` is `""`. On cap arguments (`first_tag`/`last_tag`, and `annulus`'s
`inner_tag`/`outer_tag`) **`None` means "not asked for" and `NO_TAG` means an explicit
override *to* untagged** — the difference shows where a tag would otherwise be inherited
(`annulus` inherits its input's `element_tags` only when the argument is `None`).

That default is the mechanism for naming a cap **per element** — `tjunction_lib`'s
transitions tag each quadrant of their disc (`quadrant_ogrid(..., element_tag=)`) and the
near cap picks up all four names for free. The catch is that it fires at **both** ends: a
sweep whose every station carries those names has its far cap inherit them too, and an
unconsumed seam name on an open port exports as a boundary condition. So a sweep that
names one cap this way must state the other verbatim — `last_tag=port_tag`, not
`last_tag=port_tag or None`. Same reason the port section handed back to a caller is
stripped (`unnamed()` there): it is a template to loft off, and `first_tag` would ride the
core's private seam names onto the caller's pipe.

**A tag lives on the section, so every block built from that section gets it.**
`cob_tjunction` tags the slot's boundary edges on a *copy*: the legs extrude the same
section with the band still in, where those very edges are interior, and tagging in place
named 784 interior faces that no seam consumes — one `.re2` boundary row each, in the
middle of the pipe. The geometry was identical and only the boundary block gave it away.

## Conventions

- **Typing is enforced.** Use the aliases in `_typing.py` — `FloatArray` / `IntArray` /
  `BoolArray` / `StrArray`; a bare `np.ndarray` is a `disallow_any_generics` error.
  `Point` / `Vec3` / `PointArray` document shape only.
- **No factory resamples its input** — it meshes exactly at the points given, and the
  caller proves the sampling (hence `arclength_fractions`, `path_fractions`, … which
  return a plain array, not a mesh).
- **Reports are `NamedTuple`s**, and `_fn` names the functional variant (`loft` /
  `loft_fn`).
- **Export.** `.re2` is binary-only, takes the full filename, and stays linear (corners
  only) — a mesh exports byte-identically at any order. `.vtu` emits VTK Lagrange cells,
  per-**point** `bc_id` and per-**cell** `element_tag` (region ids: 1-based positions in
  `sorted(element_tags.group_tags)`, 0 untagged, no `CellData` at all when untagged — a
  region belongs to the element, and on a conjugate mesh a per-point one would be
  ambiguous at every interface node).
  `.re2` element ids are 1-based on write; every internal index is 0-based.
- `mypy` only checks `nekmeshpy/`, so a wrong-rung call in tests or examples surfaces as
  a pytest `AttributeError`, not a type error.

`docs/user/` is the long form (`concepts.md`, `architecture.md`, `conventions.md`).

## User instructions, always confirm with user before modifying
- Git: When make a PR, always PR to main
- Spin up subagents where ever make sense. Main
objective is to maintain long sessions that retains high level understanding, minimize the
number of conversation compacting and a clear transcript.
- When attempting to web search, do it from local machine since Claude API server do not have access
to internet
- Use ./scratch/sessionname as a scratch workspace to do debugging, runs, tests
or iterations

