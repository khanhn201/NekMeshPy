# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[all,dev]"                 # dev setup (numpy, scipy, matplotlib, meshio, ruff, mypy, pytest)

ruff check nekmeshpy tests examples          # lint
mypy                                         # type-check (config pins files=["nekmeshpy"]; do NOT pass paths)
python -m pytest                             # full suite (conftest pins the Agg backend for headless viz tests)

pytest tests/test_api.py::test_to_mesh_groups   # single test
pytest -k re2                                    # by keyword

PYTHONPATH=. python examples/bifurcation.py     # run a concrete mesher (writes .re2/.rea/.vtu in cwd)

sphinx-build -b html -n -W --keep-going docs docs/_build/html   # docs (matches CI exactly)
```

Two workflows gate a PR; **all four checks must stay green** before you consider a
change done — after pushing, poll `gh pr checks <n>` until it settles, don't stop at
a local pass:

- `.github/workflows/ci.yml` — **ruff** + **mypy** on py3.12, **pytest** on py3.9–3.12.
- `.github/workflows/docs.yml` — **Build docs** via `sphinx-build -n -W` (nitpicky +
  **warnings-as-errors**). A single unresolved autodoc reference fails the build. Any
  **cross-module** `:meth:`/`:class:`/`:data:` ref in a docstring must be
  **fully-qualified** with an explicit target, e.g.
  ``:meth:`QuadMesh.blend <nekmeshpy.quadmesh.QuadMesh.blend>` `` — a bare
  ``:meth:`QuadMesh.blend` `` only resolves within the *same* class and errors from
  another module. (`Deploy to GitHub Pages` only runs post-merge.)

## The golden-regression invariant (read before editing anything numeric)

`tests/` freezes the output of `examples/bifurcation.py` in `tests/golden/`. The
tests assert it **byte-for-byte**: `.rea` and the `.re2` boundary block are byte-exact,
`.re2` coordinates match to `1e-12`, and `.vtu` is byte-identical. The numerics were
ported verbatim from a reference MATLAB/Octave implementation, so "results unchanged"
is a hard constraint — most refactors here are expected to be output-preserving.

After any change that could touch geometry/numerics, verify:

```bash
cd /tmp && PYTHONPATH=<repo> python <repo>/examples/bifurcation.py
for f in bifurcation.re2 bifurcation.rea bifurcation.vtu; do cmp -s "$f" "<repo>/tests/golden/$f" && echo "$f OK"; done
```

The pipe examples have **no** goldens (tolerance-only quality tests), so they may
change; the bifurcation must not. When a change is meant to be pure (rename,
restructure), treat a golden diff as a bug.

## Architecture

**Two layers, deliberately separated:**

- `nekmeshpy/` is a **toolkit of composable primitives** — pure data containers plus
  free-function operations. It contains *no* geometry-specific meshers.
- `examples/` holds the concrete meshers as **flat, gmsh-style scripts** (constants at
  the top, top-to-bottom code, assign to a `mesh` global, export). There are no mesher
  classes by design — a bifurcation/pipe mesher *is* its script. The test suite executes
  these scripts via `runpy.run_path` (`tests/conftest.py`) and inspects the `mesh` global,
  so examples double as integration tests and must keep producing a valid `mesh`.

**Containers are pure data; everything that acts on a finished mesh is a free function**
taking the container as its first argument — `io.export`, `io.viz`, `model.topology`, and the
per-type modules that live beside their container in the top-level `<type>/` package:
`hexmesh.quality` + `quadmesh.quality` (scaled-Jacobian metrics),
`trimesh.ops` (surface ops, reached as `nekmeshpy.trimesh.ops`), and the two smoothing
modules `hexmesh.smoothing` / `quadmesh.smoothing`. Don't add heavy
methods to the containers; add a function in the right `<type>`/`model`/`io`
module.

**Public API is re-exported from the top level** (`nekmeshpy/__init__.py`), so
`from nekmeshpy import ...` is stable regardless of internal file layout. Keep `__all__`
and the imports in sync when adding/removing public names.

### Geometry vocabulary follows gmsh

- There is **no `Point` class** — a single point is just a `(3,)` numpy array; all coordinates
  are plain numpy arrays.
- `LineMesh` — `linemesh/`. The **1-D mesh sibling** of `QuadMesh`/`HexMesh` (2 / 4 /
  8 vertices per element = line / quad / hex): a shared `(N,3)` point array on `.points` plus
  `(L,2)` `lines` connectivity that **can branch** (it is a mesh, not a single ordered path).
  Constructors accept any array-like, but input **must be 3-D** — a `(N,2)` array is rejected with
  a `ValueError`, not padded to `z=0`; all boundaries live honestly in 3-D. Open vs closed is a
  **topological property** (`is_open` / `is_closed`), not a subclass — factories set it:
  `LineMesh.open` (default consecutive chain), `LineMesh.loop` (chain that wraps),
  `LineMesh.line(start, end, fractions, *, element_tag=…)` (open straight edge placed exactly at
  `start + f*(end-start)` for each normalized-arc-length `f` — the graded-edge sibling of
  `circle`/`rectangle`; `element_tag` names every line element),
  `LineMesh.circle(radius, n, *, center=…, normal=…, start_theta=0.0, element_tags=…)` (closed;
  places `n` evenly-spaced points in the plane with the given `normal`, default `+z`, point `k` at
  angle `2πk/n + start_theta` so `start_theta` rotates the whole loop to align its index 0 with
  a `rectangle` far field's lower-left corner (`atan2(-h, -w)`) before an index-paired `annulus`;
  `element_tags` tags its line elements),
  `LineMesh.rectangle(width, height, n, *, center=…, normal=…, side_tags=…)`
  (closed far-field loop in the given plane, discretized into `n` line elements — `n` a positive
  multiple of 4: `n // 4` evenly spaced per side, running CCW from the lower-left corner with the
  corners always landing on a point, so it is a true rectangle; `side_tags` `[bottom, right, top,
  left]` names the four sides. It is the far-field outer loop for `annulus`: pass `n` equal to the
  inner loop's point count and rotate the inner `circle` with `start_theta` to the lower-left
  corner, and the two loops pair index-for-index — the radial spokes need not be straight),
  `LineMesh.from_segments` (chain unordered segments into the
  largest closed loop, or `None`). `LineMesh.merge(meshes, *, tol=…)` is the 1-D
  sibling of `QuadMesh.merge`/`HexMesh.merge`: it welds coincident **topological end points**
  (degree-1 chain ends) — never interior points — concatenating `element_tags`/`boundaries`, and
  reports the result `closed` iff no degree-1 end survives. Two shared-endpoint `A1->A2` arcs
  (reverse one so the traversal doesn't cross) weld at `A1`/`A2` into a single loop with index 0
  at `A1` and `M//2` at `A2` — the clean way to close a seam ring from two half-arcs (used by
  `bifurcation.py` and `circular_pipe_tjunction.py`). Factories check `is_closed` at runtime (`ogrid`/`annulus`
  demand closed; `structured` edges demand open), so the open/closed distinction is still
  enforced — as a value, not a type. **Every curve is meshed exactly at the points given — there
  are no `LineMesh` ordered/resampling ops** (a factory or the caller must hand in an
  exactly-sized, correctly-oriented curve, so two blocks sharing a boundary can never disagree on
  it); `.length` (index-order arc length) is the only ordered read that remains. The intrinsic
  arc-length interpolation that scanned/analytic examples still need lives in `trimesh/ops.py`
  (`trimesh.ops.resample_polyline`, `conform_ring_stack`), not on `LineMesh`.
- **Two tag systems, both propagate up the ladder** (line → quad → hex) on `extrude`/`loft`, and
  both are no-ops when untagged (this is what keeps the golden byte-identical):
  - `element_tags` — a **dense** per-element `StrArray` (`""` = untagged), one tag per line
    (region/material). Set at construction on the `LineMesh`, copied by the section
    factories onto the `QuadMesh` edges/quads and thence onto the hex faces/hexes.
    `element_group_tags` = sorted unique non-empty. **Tag at the lowest level** — every
    section-wall tag can originate on the `LineMesh` input (the circle/loop/arc/edge), which
    every section factory now reads: `ogrid`/`annulus` from the loop's per-line tags,
    `half_ogrid` from the arc's per-segment tags, `structured` from each edge's uniform tag.
    The factories' scalar/mapping args (`wall_tag` / `inner_tag` / `outer_tag` /
    `boundary_tags[side]`) are **overrides**: a non-empty arg replaces the line-level tag for
    that wall/side (**upper overrides lower**); an empty/absent arg falls through to it (and a
    present-but-empty `boundary_tags[side]` / `NO_BOUNDARY` suppresses the side). The examples
    tag at the line level and keep only the hex-level `first_tag`/`last_tag` end caps (no lower
    level exists for those).
  - `boundary_tags` — a **sparse** `StrArray` parallel with `boundaries (Nbc,2)` = `[elem id,
    side]`. A `LineMesh`'s boundary is its end **points** (`side ∈ {1,2}` → local vertex `s-1`);
    on `extrude` they become quad boundary **edges**, then hex boundary **faces**.
    `boundary_group_tags` = sorted unique. (See `examples/flow_past_cylinder.py`.)
- `TriMesh` / `QuadMesh` / `HexMesh` / `Mesh` — mesh containers; each stores coordinates as
  a **bare `(P,3)` NumPy array** on `.points` (mutate in place with `mesh.points[:] = X`).
  `QuadMesh` and `HexMesh` also carry a dense `element_tags` and sparse `boundaries`/
  `boundary_tags`, mirroring `LineMesh` one/two dimensions up.

### Factory model (also gmsh-named)

The factories are **plain free functions split across files by open-vs-closed**, then
**bound onto the container class in the package `__init__`** (`setattr(Class, name,
staticmethod(fn))`), so `LineMesh.circle(...)` / `QuadMesh.ogrid(...)` stay reachable as
class methods while the container files (`linemesh.py` / `quadmesh.py`) stay **pure data
containers with no factory code and no factory base classes** — nothing to edit there
when a shape is added. Core constructors + queries stay in `linemesh.py` / `quadmesh.py`;
parametric closed shapes live in each package's `_closed.py` (line `circle`/`rectangle`;
quad `box`/`sphere`) and region-fills / open curves in `_open.py` (line `line`; quad
`structured`/`rectangle`/`ogrid`/`half_ogrid`/`annulus`), with `quadmesh/_helpers.py`
holding the shared `_apply_smoothing`/`_check_boundary`. Each module ends with a
`FACTORIES: dict[str, Callable[..., <Class>]]` registry; the package `__init__` merges
the two dicts (`{**_CLOSED, **_OPEN}`) and binds every entry. A factory that needs a core
constructor does a lazy in-body `from .<core> import <Class>` (breaks the import cycle:
the package imports the core to bind onto it).

- **Adding a factory** touches **only** the matching `_closed.py`/`_open.py`: add the
  free function (no `cls`/`self`) and one `FACTORIES` entry. The container file and the
  `__init__` binding loop need no edit.
- **mypy** pins `files=["nekmeshpy"]`, so only toolkit code is checked and the
  dynamically-bound `LineMesh.circle` etc. are invisible to it. **Internal toolkit code
  must call the free functions directly** (e.g. `from ..linemesh._open import line`),
  never `LineMesh.line`/`QuadMesh.structured`; external callers (examples/tests/users)
  use the bound `LineMesh.circle(...)` sugar.

- **Sections** are `QuadMesh` classmethods: `QuadMesh.structured` (transfinite grid from four
  open `LineMesh` edges), `QuadMesh.rectangle(corners, nx, ny, *, x_frac=, y_frac=, side_tags=)`
  (structured-grid convenience: 4 corners + counts + optional per-axis grading + `{bottom,right,
  top,left}` side tags), `QuadMesh.ogrid` (O-grid in a closed `LineMesh`),
  `QuadMesh.half_ogrid` (half-disc O-grid bounded by a wall arc + a spine diameter),
  `QuadMesh.spined_ogrid(boundary, radial, *, spine=None, center_scale=, wall_tag=, smoothing_method=)`
  (full-disk O-grid over a closed `boundary` loop with a natural `A1..A2` seam: it splits the loop
  at index `0`/`M//2` into two `A1->A2` half-arcs, resamples the `spine` curve by arc length at the
  fractions each half needs — so a curved spine is meshed exactly and both halves share it
  point-for-point — runs two `half_ogrid`s and `merge`s them. `spine=None` (the default) uses the
  straight `A1..A2` chord `boundary.points[[0, M//2]]` — the common case for a planar disc
  (`circular_pipe_tjunction.py`); pass a curve only to bow the seam (`bifurcation.py`). The caller
  hands in only the loop (and optional spine) instead of hand-rolling the split/sample/merge),
  `QuadMesh.annulus` (ring O-grid between two loops —
  built as a radial `QuadMesh.loft` of blended rings, the sibling of `HexMesh.annulus` one
  dimension down; the periodic ring topology rides in the loops' wrapping `lines`, so there is no
  modular arithmetic, and the inner/outer rings are the loft's near/far caps).
  `QuadMesh` also has `extrude`/`loft` **one dimension down** — sweeping a `LineMesh` into a quad
  strip (mirrors `HexMesh.extrude`/`loft`), carrying the line's `element_tags` onto the quads and
  its boundary-point tags onto the side-wall edges.
  **`blend(a, b, fractions)`** is the shared morphing combinator on all three containers
  (`LineMesh`/`QuadMesh`/`HexMesh`): given two index-paired conformal profiles (equal point count,
  identical connectivity — and same open/closed topology for `LineMesh`), it returns one profile per
  fraction at `(1-t)*a.points + t*b.points`, copying `a`'s connectivity + `boundaries` +
  `boundary_tags` but **not** its `element_tags` (region/cap tags are assigned by the consuming
  `loft`/factory, which is what keeps the annulus goldens byte-identical). It is the single
  positioning step behind both `annulus` factories (`QuadMesh.annulus` = radial `loft` of
  `LineMesh.blend`ed rings; `HexMesh.annulus` = radial fill of `QuadMesh.blend`ed shells) and behind
  each leg's slice stack in `bifurcation.py`/`circular_pipe_tjunction.py`
  (`LineMesh.blend(opening, seam, …)`). Each section takes an optional
  `smoothing_method=` (see below). All build **natively in 3-D** — nothing is projected to a
  plane, so a boundary placed in any plane, or a genuinely **curvy / non-planar** boundary, is
  filled in place with its true shape (never flattened to `xy`). `ogrid`/`annulus` build a
  straight-chord initial guess and rely on `smoothing_method="conduction"` to relax the interior
  harmonically onto the curved surface spanned by the fixed boundary ring; `structured`/
  `half_ogrid` blend the 3-D edge points directly. (`LineMesh.circle` and `rectangle`
  use the private `linemesh/_plane.py` `_in_plane_axes` helper — they *construct* a
  planar loop, not project an existing boundary.) `ogrid`/`half_ogrid` are ICEM/Pointwise terms
  with no gmsh equivalent — kept deliberately.
- **Hex blocks** are `HexMesh` classmethods: `extrude` (sweep one section along a straight
  axis = gmsh Extrude+Layers+Recombine), `loft` (recombine a stack of pre-positioned
  profiles — the general case behind `extrude`), `annulus` (fill the 3-D shell between two
  **closed `QuadMesh` surfaces** — the section-to-section case, sibling of `QuadMesh.annulus`
  one dimension up), `merge` (stitch blocks, welding coincident **boundary** points only),
  `from_grid` (structured i×j×k). `HexMesh` is immutable by construction (no incremental
  building).
  - `HexMesh.annulus(inner, outer, radial)` pairs the two surfaces **by index** (identical
    `quads` + equal point count; build one from the other's points so `p`↔`p` holds, as
    `flow_past_sphere.py` does: `sphere = R*normalize(cube.points)` on `cube.quads`), blends
    `radial` shells, and tags the inner/outer wall faces from each surface's **per-quad
    `element_tags`** (a closed surface has no free boundary edges) — the inner surface's tag
    lands on face 5 of each shell column, the outer's on face 6; a non-empty scalar `inner_tag`/
    `outer_tag` **overrides** the surface tags and names a whole wall. It has no `smoothing_method` (there is no
    HexMesh smoothing registry). `loft`'s `first_tag`/`last_tag` (the end-cap tags, renamed from
    `first_cap`/`last_cap`) also accept a per-quad array (not just a scalar), which is how
    `annulus` forwards the surface tags.
  - `QuadMesh.from_grid(P, *, edge_tags=, element_tag=)` is the section sibling of
    `HexMesh.from_grid` — a structured `(ni+1,nj+1)` quad grid; `element_tag` fills the dense
    per-quad `element_tags` (e.g. tag a whole cube-face patch with the far-field side it forms).
  - **Closed 3-D surfaces** (`merge` of six `from_grid` face patches): `QuadMesh.box(half_sizes,
    n, *, face_tags=)` (watertight box surface — `half_sizes`/`n` each a scalar or per-axis
    `(sx,sy,sz)`/`(nx,ny,nz)`; `face_tags` keyed `{x_min,x_max,y_min,y_max,z_min,z_max}` tag whole
    faces) and `QuadMesh.sphere(radius, n, *, element_tag=)` (cubed-sphere: `box` connectivity with
    points projected onto the sphere, so it **pairs by index** with a `box` of the same `n` for
    `HexMesh.annulus`). See `flow_past_sphere.py`.

### Section smoothing is per-section

Cross-section interior nodes are repositioned on a single `QuadMesh` *before* extrusion,
via `quadmesh.smoothing.set_section_smoothing(qm, method)` (registry `SECTION_METHODS`;
extend with `@register_section_smoothing("name")`). Built-ins: `conduction`, `winslow`,
`bilinear`/`none`. There is no HexMesh-level smoothing registry. **The relaxers move
only corner nodes**, so a *repositioning* method (`conduction`/`winslow`) on an `order >
1` section is **rejected** (`NotImplementedError`) — high-order smoothing is not
implemented; the no-op strategies (`bilinear`/`tfi`/`none`/`""`) stay allowed at any
order because they leave every node in place (`circular_pipe.py` runs order 5 +
`bilinear`). The factories `_elevate` to order N *first*, then smooth, so the smoother
sees the true order and raises cleanly instead of silently degrading. `hexmesh.smoothing.smooth`
(the STL-constrained wall polish used by `bifurcation.py`/`circular_pipe_tjunction.py`)
rejects `order > 1` the same way.

### High-order (order-N) elements

Every factory takes an optional `order=N` (default `1`). At `order > 1` each element
carries `(N+1)` **GLL** (Gauss–Lobatto–Legendre) nodes per parametric direction —
line `N+1`, quad `(N+1)²`, hex `(N+1)³` — placed on the **true** geometry the factory
owns (a `circle`'s arc nodes on the exact circle, a `sphere`'s on the exact sphere).
GLL endpoints are exactly `0.0`/`1.0`, so corner nodes stay exact under every sweep.

**Entity-based conformal storage (corners single-sourced, HO nodes shared by topology).**
Corner connectivity (`lines`/`quads`/`hexes`) stays the authoritative topology, and
**the corners are owned solely by `points[conn]` — never duplicated into a stored
block**. The **non-corner** high-order nodes are decomposed by *topology* into shared
entities plus private interiors and stored on the private `_ho: conform.EntityTables`
(module `nekmeshpy/model/conform.py`): **edges** (unique undirected edges — canonical
min-corner-id first — with their `N−1` shared interior nodes, a per-element incidence
`elem_edges`, and an `edge_flip` bit for anti-canonical traversal); **faces** (hex only:
unique faces with their `(N−1)²` shared nodes, incidence `elem_faces`, and a **D4
orientation code** `face_orient` — one of 8 square symmetries — mapping the hex's local
face grid to the shared canonical frame); and per-element **interior** (line `N−1`, quad
`(N−1)²`, hex `(N−1)³`, never shared). Sharing is decided by **corner ids** (structural /
exact conformality): two elements meeting on an edge/face resolve to the *same* HO nodes,
and a `curved=` block whose incident copies disagree on a shared entity is **rejected at
construction** (loud error, not a silent weld). At `order == 1` every table is empty.
`.curved` is a **read-only computed property**, not a stored attribute: on each read
`conform.assemble` reassembles the full `(E, (N+1)^d, 3)` block from `points[conn]`
(corners) + the entity tables, so `mesh.curved` always has shape `(E, (order+1)^d, 3)`
regardless of order. Because corners are read fresh from `points` every time, an in-place
`mesh.points[:] = X` is **automatically reflected** in `.curved` — no staleness, and the
corner-consistency invariant is **structural**. `mesh.to_conformal()` exposes the
conformal model directly as `(nodes (M,3), conn (E,(N+1)^d))` — one global node array with
dense per-element connectivity (the HO analog of `points`+`quads`); the tables are also
readable via `.edges`/`.edge_nodes` (quad, hex) and `.faces`/`.face_nodes` (hex). Both
params ride the container `__init__` (`order: int = 1`, `curved: CurvedBlock | None = None`
— a factory may pass the *full* block or omit it); `conform.split(order, curved, points,
conn, dim, who)` validates it (exact shape + corner-consistency against `points[conn]`,
scale-relative tol) and **scatters it into the entity tables** (owner-wins + verify);
`CurvedBlock` is a `FloatArray` shape-doc alias in `_typing.py`. **Goldens stay
byte-identical because every order-1 code path branches on `order`, not on
curved-presence** — `to_re2`, quality, topology, `merge`, `FACE_POINTS`, and the `.vtu`
order-1 writer all read only `points`/`conn` (the order-1 VTK path reads `points[conn]` in
Nek/CCW order, never the reassembled `curved` block). This is why the golden
`bifurcation.*` (built with defaults) stays byte-identical: **order-1 export is a strict
no-op, and treating any golden diff at default order as a bug still holds.** Combinators
(`blend`/`annulus`/`loft`/`extrude`/`merge`) interpolate the full `curved` block at
`order > 1` (`blend_ho`) and hand it back to the constructor (validated, then split into
its entity tables); at order 1 the empty tables make that a no-op equal to the plain point
blend.

**Shared kernel.** Order-N logic is split across two modules over GLL reference nodes from
`model/fields.py` (`gll_nodes`). `model/interp.py` holds the numeric primitives
(`tensor_nodes`, `corner_indices`, `subdivide_quads`/`subdivide_hexes`, `coons_grid`,
`blend_ho`, `quad_edge_indices`/`hex_edge_indices`/`hex_face_indices`). `model/conform.py`
holds the topology + orientation + storage engine (`EntityTables`, `unique_edges`,
`unique_faces`, `split`/`assemble`/`to_conformal`, the D4 helpers
`_d4_apply`/`_perm_tables`/`_face_code`); `split` is the `curved=` validator/decomposer
and `assemble` the `.curved` reassembler used by every container. Region factories
`ogrid`/`half_ogrid`/`structured` build a linear guess and
`_elevate` it to order N (straight tensor-subdivided interior + boundary overlays
stamping the true wall curves onto the sides) — **then** smooth, so a repositioning
smoother rejects `order > 1` (see *Section smoothing*) rather than silently producing a
straight interior. Pure combinators (`blend`/`loft`/`extrude`/`merge`/`annulus`)
propagate or build `curved` directly (`loft` sweeps each column as a straight GLL blend
of the two bounding slices' in-plane blocks); `QuadMesh.annulus` is now a *single*
curved path (radial `loft` of `blend_ho`ed rings) at every order, with any repositioning
`smoothing_method` rejected at `order > 1`. All factories reject a mismatched `order`
across their inputs.

**Export.** `.re2` **stays linear** — Nek's re2 has no high-order format yet, so
`to_re2` reads only the 8 corners and a mesh exports byte-identically at any order.
The `.vtu` (XML VTK) writer becomes high-order at `order > 1`, emitting VTK Lagrange
cells (`VTK_LAGRANGE_CURVE=68` / `_QUADRILATERAL=70` / `_HEXAHEDRON=72`), `(N+1)^d`
un-welded nodes/cell, ordered via a hand-built `_lagrange_*_perm(order)` (corners →
edges → faces → interior, VTK's `PointIndexFromIJK` recursion — no `vtk`/`meshio`
dep). Face nodes inherit the face's `bc_id` via `hex_face_indices`. The writer
(`to_vtu`/`line_to_vtu`/`quad_to_vtu`) builds its node arrays via
`_hex_arrays`/`_line_arrays`/`_quad_arrays` and emits through `_write_vtu`; there is
**no legacy ASCII `.vtk` writer** — only `.re2` and `.vtu`. The order-1 path is
byte-untouched (golden `bifurcation.vtu` byte-exact).
**Order-N quality is opt-in** (defaults stay corner-based so pinned quality numbers
hold): `quadmesh.quality.scaled_jacobian_ho(curved, order)` /
`hexmesh.quality.scaled_jacobian_ho(...)` sample the scaled Jacobian at the
`(N+1)^d` GLL nodes of the curved block (tangents from
`model.fields.lagrange_derivative_matrix`; kernel `model.interp.scaled_jacobian_ho`),
reached via `mesh.scaled_jacobian(high_order=True)` / `quality_summary(high_order=True)`
— at order 1 the GLL nodes are the corners so it reduces exactly to the default corner
metric. **Order-N smoothing is not implemented** (the corner-graph Laplacian/Winslow
ignore mid/interior nodes; straight-subdivided interiors are already fine). Rather than
silently degrade, a repositioning smoother now **raises `NotImplementedError` at `order
> 1`** — `set_section_smoothing` (`conduction`/`winslow`; the no-op `bilinear`/`none`
stay allowed) and `hexmesh.smoothing.smooth` both guard on `mesh.order`. Build it only
when a real need appears. See `examples/high_order_{curve,quad,hex}.py`.

### Physical groups & export

`PhysicalGroups` maps name ↔ tag ↔ Nek BC code; pass `groups=` to the factories to control
`.re2` boundary codes without touching the exporter (`PhysicalGroups.duct()`,
`.from_tags()`, `.nek_default()` are presets). `.re2` element ids are 1-based on write;
all internal indices are 0-based.

## Conventions

- **Strong typing is enforced** (`mypy` with `disallow_untyped_defs`, `check_untyped_defs`,
  `disallow_any_generics`). Everything in `nekmeshpy/` is annotated. Geometry-object parameters
  take the concrete type (`LineMesh`, and open-vs-closed is checked at runtime, not in the type)
  with no `| np.ndarray` fallback; only genuine
  vector *literals* (axis/origin/center) use `Sequence[float] | FloatArray`. Array-valued
  numeric internals use the dtype-parametrized aliases in `nekmeshpy/_typing.py` — `FloatArray`
  (`NDArray[np.float64]`, coordinates/real data), `IntArray` (`NDArray[np.int64]`,
  connectivity/indices), `BoolArray` (masks), `StrArray` (`NDArray[np.str_]`, `element_tags`/
  `boundary_tags`) — never a bare `np.ndarray`, which
  `disallow_any_generics` rejects as an implicit `NDArray[Any]` (use an explicit `NDArray[...]`
  for any other dtype). `Point` / `Vec3` / `PointArray` are shape-documentation aliases of
  `FloatArray` marking a single `(3,)` location / a single `(3,)` direction / a `(P,3)` array of
  point coordinates (vs `(N,)` scalar data); numpy has no static shape checking, so they document
  intent only and are interchangeable with `FloatArray` to mypy.
- Full architecture, module reference, and extension points: `README.md`.
