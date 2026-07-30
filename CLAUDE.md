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
  property of the `lines` array and **is stored nowhere** — there is no `closed=` kwarg and no
  `is_open`/`is_closed`: a loop is simply a cycle of line elements, i.e. one whose `lines` carry
  the explicit wrap row `[N-1, 0]` and therefore leave no degree-1 end (`boundary_points()` is
  empty). `LineMesh` trusts its connectivity exactly as `QuadMesh`/`HexMesh` do — and it never
  *invents* it: **`lines` is a required constructor argument**, there is no "consecutive chain"
  default and therefore nothing in the container that could imply a wrap. Callers either own their
  `lines` outright (`from_segments`' chained loop, `merge`'s rewelded lines, `blend`'s copy of
  `a.lines`, the quad/hex edge `LineMesh`es built from `conform.unique_edges` or the layer-by-layer
  append) or author them one rung up with `LineMesh.loft`. Factories build the wrap where it
  belongs:
  `LineMesh.loft(points, *, loop=False, …)` (the **bottom rung of the uniform sweep primitive** —
  see *loft is the sweep primitive at every rung* below — each "profile" is a single point so the
  rungs *are* the line elements: `loop=False` gives the consecutive chain, `loop=True` appends the
  single closing rung `[N-1, 0]`. At `order > 1` an omitted `interior` is filled with the straight
  GLL blend between each line's endpoints, which is what the straight-sided factories want),
  `LineMesh.open` (consecutive chain — a thin wrapper over `loft(..., loop=False)`),
  `LineMesh.loop` (chain that wraps — a thin wrapper over `loft(..., loop=True)`, which builds the
  `[N-1, 0]` row; note the unavoidable collision between the `loop` *factory* name and the `loop=`
  *kwarg*),
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
  (degree-1 chain ends) — never interior points — concatenating `element_tags`/`boundaries`. The
  welded connectivity *is* the answer: if no degree-1 end survives the result is a loop. Two
  shared-endpoint `A1->A2` arcs
  (reverse one so the traversal doesn't cross) weld at `A1`/`A2` into a single cycle with index 0
  at `A1` and `M//2` at `A2` — the clean way to close a seam ring from two half-arcs (used by
  `bifurcation.py` and `circular_pipe_tjunction.py`). **No factory checks closedness**
  (there is no flag to check): each constrains its input through the facts it actually needs —
  `ogrid` an exact `4*n_side` point ring, `annulus`/`blend` identical `lines` on both rings,
  `structured` four edges that share corners, `spined_ogrid` an `8*Ntheta` ring. A factory that
  reads only `boundary.points` (`ogrid`, `half_ogrid`, `structured`) therefore accepts an open
  chain and silently treats it as the equivalent closed ring — a known, deliberate gap.
  **Every curve is meshed exactly at the points given — there
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
  identical connectivity — which, for `LineMesh`, is exactly what makes both open or both closed), it returns one profile per
  fraction at `(1-t)*a.points + t*b.points`, copying `a`'s connectivity + `boundaries` +
  `boundary_tags` but **not** its `element_tags` (region/cap tags are assigned by the consuming
  `loft`/factory, which is what keeps the annulus goldens byte-identical). Like every other rung
  of the ladder it is **composed downward**: `HexMesh.blend` is a `QuadMesh.blend` of the
  shared-face mesh plus a lerp of the private per-hex `interior`, and `QuadMesh.blend` in turn a
  `LineMesh.blend` of the shared-edge mesh plus the per-quad `interior` — each rung lerps only
  what it privately owns and keeps `a`'s incidence (`hex`/`face_orient`, `quad`/`flip`) verbatim.
  It is the single
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
    HexMesh smoothing registry). `loft`'s end-cap tags `first_tag`/`last_tag` accept a per-quad
    array as well as a scalar, which is how `annulus` forwards the surface tags.
  - `QuadMesh.from_grid(P, *, edge_tags=, element_tag=)` is the section sibling of
    `HexMesh.from_grid` — a structured `(ni+1,nj+1)` quad grid; `element_tag` fills the dense
    per-quad `element_tags` (e.g. tag a whole cube-face patch with the far-field side it forms).
    It is itself a **`QuadMesh.loft` of the grid's column profiles**, and each profile is in turn
    a **`LineMesh.loft`** of that column's `i` points (profile `j` = the `i`-chain lofted from
    `P[:, j, :]`, sweep along `j`), so the ladder
    `HexMesh.from_grid → QuadMesh.from_grid → LineMesh.loft` is composed at **every** rung — the
    chain connectivity and (at `order > 1`) each segment's straight-GLL interior come from the
    rung below, not hand-rolled here, and nothing is re-derived from corners with
    `conform.unique_edges`. All four `edge_tags` sides ride channels `loft` already has: the
    profile's tagged end **points** become `x_min`/`x_max` (quad sides 4/2) and the sweep's
    `first_tag`/`last_tag` caps become `y_min`/`y_max` (sides 1/3). **The loft's numbering is
    carried up unchanged** — composing the rung below means accepting its numbering, so there is
    **no relabel**: the grid comes out **sweep-major, `i` fastest** (node `(i,j)` = point
    `j*(ni+1)+i`, cell `(i,j)` = quad `j*ni+i`, i.e. `points == P.transpose(1,0,2).reshape(-1,3)`,
    *not* `P.reshape(-1,3)`). `boundaries` stays lexsorted by `(quad, side)` so its rows follow the
    quad ids; each row still names the same physical side.
  - `HexMesh.from_grid(P, *, face_tags=, element_tag=)` is the same composition one rung up: a
    **`HexMesh.loft` of the grid's `k`-sections** (section `k` = `QuadMesh.from_grid(P[:,:,k,:])`,
    sweep along `k`), so corners, shared edges *and* shared faces all come out of the
    layer-by-layer B-rep assembly instead of a `conform.unique_edges` re-derivation from a
    hand-built `(E,8)` corner table. The section's four `edge_tags` become the
    `x_min`/`x_max`/`y_min`/`y_max` swept side faces (section side `s` → hex face `s`, exactly the
    `_GRID_SIDES` Nek numbering) and the sweep's caps the `z_min`/`z_max` ones; `element_tag`
    rides the section's per-quad tags. Numbering is again the loft's, unrelabelled: `i` fastest,
    `k` slowest (`points == P.transpose(2,1,0,3).reshape(-1,3)`).
  - **Closed 3-D surfaces** (`merge` of six `from_grid` face patches): `QuadMesh.box(half_sizes,
    n, *, face_tags=)` (watertight box surface — `half_sizes`/`n` each a scalar or per-axis
    `(sx,sy,sz)`/`(nx,ny,nz)`; `face_tags` keyed `{x_min,x_max,y_min,y_max,z_min,z_max}` tag whole
    faces) and `QuadMesh.sphere(radius, n, *, element_tag=)` (cubed-sphere: `box` connectivity with
    points projected onto the sphere, so it **pairs by index** with a `box` of the same `n` for
    `HexMesh.annulus`). See `flow_past_sphere.py`.

### `loft` is the uniform sweep primitive at every rung of the ladder

`LineMesh.loft` / `QuadMesh.loft` / `HexMesh.loft` are **one primitive at three
dimensions**, each taking a stack of index-paired conformal profiles and a
`loop: bool = False` flag:

| rung | a "profile" is | a "rung" entity is | `extrude` = |
|---|---|---|---|
| `LineMesh.loft(points, *, loop=…)` | a single **point** | a **line** element | (n/a — `line`/`circle` place the points) |
| `QuadMesh.loft(slices, *, loop=…)` | a `LineMesh` | a rung **line** + the **quads** | `QuadMesh.extrude` |
| `HexMesh.loft(slices, *, loop=…)` | a `QuadMesh` | a rung **face** + the **hexes** | `HexMesh.extrude` |

The quad rung assembles **layer by layer** (append profile `i`'s own lines +
`interior` verbatim, then the rung lines joining level `i` to level `i-1`; a quad is
then just four line indices + four `flip` bits — no `unique_edges` dedupe, no
owner-wins reconciliation, because no shared edge is ever duplicated). The line rung
is that same pattern one dimension down, which is why `LineMesh.open`/`LineMesh.loop`
are **thin wrappers** over `LineMesh.loft` (`loop=False` / `loop=True`) rather than
connectivity-generating code in the container. `HexMesh.loft` instead builds the
corner table and derives its B-rep with `unique_edges` → `canonical_faces` →
`scatter_*`; `loop=True` works there because the seam faces resolve from the shared
corner ids like any other face.

**`loop=True` is a periodic sweep**: the last profile joins back to the **first**, so
`M` profiles give `M` layers instead of `M-1`. It falls out of the assembly as *one
extra iteration* — exactly one more rung block, appended once at the end, with the
first profile's lines/faces **not** re-appended — so the seam is a genuine shared
entity, the closed sweep has no free boundary in the sweep direction, and it carries
strictly fewer points than the `loop=False` stack that repeats profile 0. At every
rung `loop=True`:
- **rejects `first_tag`/`last_tag`** with an actionable `ValueError` (shared guard
  `model.fields.reject_loop_caps`) — a closed sweep has no near/far cap, so passing
  one is a caller mistake and must not be silently dropped (scalar *or* per-element
  array form);
- emits **no cap boundary rows**; side-wall boundaries derived from the profiles' own
  boundary entities are unaffected.

`loop=False` is bit-identical to the pre-`loop` behaviour at every rung (the swept-to
profile index is `i+1` unconditionally there), which is what keeps the goldens
byte-identical. Use it for a torus surface (`QuadMesh.loft` of revolved rings) or a
solid torus (`HexMesh.loft` of revolved discs) — see `tests/test_periodic_loft.py`.
Note that `QuadMesh.annulus` closes in the **ring** direction, which lives in the
loops' own connectivity, not the loft direction — it does *not* use `loop=True`.

### Section smoothing is per-section

Cross-section interior nodes are repositioned on a single `QuadMesh` *before* extrusion,
via `quadmesh.smoothing.set_section_smoothing(qm, method)` (registry `SECTION_METHODS`;
extend with `@register_section_smoothing("name")`). Built-ins: `conduction`, `winslow`,
`bilinear`/`none`. There is no HexMesh-level smoothing registry. **The relaxers move
only corner nodes**, so a *repositioning* method (`conduction`/`winslow`) on an `order >
1` section is **rejected** (`NotImplementedError`) — high-order smoothing is not
implemented; the no-op strategies (`bilinear`/`tfi`/`none`/`""`) stay allowed at any
order because they leave every node in place (`circular_pipe.py` runs order 2 +
`bilinear`). The factories `_elevate` to order N *first*, then smooth, so the smoother
sees the true order and raises cleanly instead of silently degrading. `hexmesh.smoothing.smooth`
(the STL-constrained wall polish used by `bifurcation.py`/`circular_pipe_tjunction.py`)
rejects `order > 1` the same way.

### High-order (order-N) elements

Every factory takes an optional `order=N` (default `1`). At `order > 1` each element
carries `(N+1)` **GLL** (Gauss–Lobatto–Legendre) nodes per parametric direction —
line `N+1`, quad `(N+1)²`, hex `(N+1)³`. GLL endpoint parameters are exactly
`0.0`/`1.0`, so the two extreme nodes *are* the corners and stay exact under every
sweep.

**The B-rep ladder is the storage.** There is no per-element node block anywhere, no
`.curved` attribute and no `to_conformal()` facade. Each container stores the rung
below it plus what it privately owns:

| container | stored | derived read-only views |
|---|---|---|
| `LineMesh` | `points (P,3)`, **required** `lines (L,2)`, `interior (L,N−1,3)` | — |
| `QuadMesh` | `lines`: a `LineMesh` of the shared edges (its `interior` = the edge nodes) + `quad (Q,4)` edge incidence + `flip (Q,4)` + `interior (Q,(N−1)²,3)` | `points`, `quads (Q,4)` |
| `HexMesh` | `quads`: a `QuadMesh` of the shared faces (its `interior` = the face nodes, its `lines.interior` = the edge nodes) + `hex (E,6)` face incidence + `face_orient (E,6)` D4 codes + `interior (E,(N−1)³,3)` | `points`, `hexes (E,8)` |

A hex's `points` *is* its shared-face `QuadMesh`'s `points`, which *is* its shared-edge
`LineMesh`'s `points` — one array, single-sourced — so corner consistency is
**structural**, `quads`/`hexes` are memoized derivations (`_derive_corners`), and an
in-place `mesh.points[:] = X` is picked up at every rung for free. Convenience readers:
`.edges`/`.edge_nodes` (quad, hex), `.faces`/`.face_nodes` (hex). At `order == 1` the
`interior` tables are empty `(·,0,3)` but the edge/face *topology* is still first-class
storage.

**Conformality is structural, decided by corner ids** — never a coordinate weld. A
shared edge is literally one row of the edge `LineMesh` referenced by every incident
quad; a shared face is one quad of the shared-face `QuadMesh`. `model/conform.py` owns
the topology + orientation + reconciliation engine (`unique_edges`,
`unique_faces`/`canonical_faces`/`hex_corners_from_faces`, `entity_tol`, the
`scatter_*`/`gather_*` pair, the `conformal_*` walks, D4 helpers
`_d4_apply`/`_perm_tables`/`_face_code`) and **imports no container** — everything
crosses as plain arrays. Entities: **edges** (unique undirected, canonical min-corner-id
first, `N−1` shared nodes + per-element incidence + a `flip` bit for anti-canonical
traversal); **faces** (hex only, `(N−1)²` shared nodes + incidence + a **D4 orientation
code**, one of 8 square symmetries, mapping the hex's local face grid to the canonical
frame); **interior** (private, line `N−1` / quad `(N−1)²` / hex `(N−1)³`). Where a
combinator must rebuild the shared tables against a new topology (`merge`,
`HexMesh.loft`) it reconciles with `conform.scatter_edge_nodes`/`scatter_face_nodes`:
owner-wins + verify every other incident copy within `conform.entity_tol`, loud
`ValueError` on disagreement. The **conformal walks** `conform.conformal_line`/
`conformal_quad`/`conformal_hex` flatten the ladder on demand into
`(nodes (M,3), conn_ho (E,(N+1)^d))` — the HO analog of `points`+`quads`, and the single
node numbering the `.vtu` writer and the order-N quality metrics read; `nodes[conn_ho]`
is the transient per-element block whenever one is genuinely needed.

**True geometry vs straight GLL subdivision — the thing callers get wrong.** `order=N`
is valid on every factory, but a mesh can be **high-order in storage and linear in
geometry**. Only a factory that owns an analytic shape can place a node off the straight
line between corners:
- **on the true shape**: `LineMesh.circle`/`LineMesh.arc` (interior GLL nodes evaluated
  at the exact arc angles — `_plane._arc_points`/`_arc_interior` — and handed to `loft`
  as an explicit `interior`, overriding its default chord blend),
  `QuadMesh.sphere`/`QuadMesh.hemisphere` (radial projection applied **entity-wise** to
  the cube's / half box's whole B-rep — `points`, `lines.interior`, `interior` — never to
  a reassembled block, so a shared edge lands identically from either quad).
  Straight-sided analytic shapes are exact trivially: `LineMesh.line`,
  `LineMesh.rectangle`, `QuadMesh.box`/`half_box` (planar patches).
- **straight GLL subdivision**: anything built from an explicit point array —
  `LineMesh.open`/`loop`/`loft` (each line's interior = the straight blend of its two
  endpoints), `QuadMesh.from_grid`/`HexMesh.from_grid`. Sampling a curve into points and
  calling `LineMesh.open` therefore *loses* the curve at `order > 1`; hand in the
  analytic `arc`/`circle` instead.
- **region fills — curved throughout, by two different mechanisms.** Both propagate the
  input wall's curvature into the *interior*, not just onto the boundary elements:
  - `structured` owns an exact global transfinite map, so at `order > 1` it simply
    **samples that map at the GLL-refined lattice** (`_refined_params`: interval `i`'s
    node `a` at `(i + g[a]) / n`) against each edge's own nodes in traversal order
    (`_refined_chain`), via the shared `interp.coons_grid`. The resulting
    `(Q,(order+1)²,3)` blocks are decomposed back into B-rep tables by
    `_helpers.entities_from_blocks` (the inverse of the entity→block gather:
    `scatter_edge_nodes` owner-wins + verify, plus the private interiors). Every node —
    corner, edge, interior — is the true transfinite point, so a curved input edge bows
    the whole block and no overlay is needed. At `order == 1`, `g = [0,1]` makes the
    refined lattice exactly `linspace`, so the order-1 no-op falls out by construction
    rather than by a branch.
  - `ogrid`/`half_ogrid` have no global analytic map, so they keep the linear
    construction and generalize the `Overlay` `(quad ids, local side, curve)` channel:
    **one overlay pair per O-ring**, not just the wall. Each intermediate ring's curve is
    `LineMesh.blend(block perimeter, wall, t)` — the same mechanism `annulus` uses — so a
    ring at `t` inherits its share of the wall's bow. Both incident copies of a shared
    ring must be stamped (ring `m` is block `m−1`'s outer side *and* block `m`'s inner
    side); stamping one leaves the other straight and `scatter_edge_nodes` rejects the
    mesh. Each element is then curved tangentially and straight radially — exactly
    `annulus`'s behaviour, which is right for a radial blend.
  - Underneath both, `_elevate` derives each quad's private `interior` as the
    **transfinite (Coons) patch of that element's own four edge curves**, evaluated
    *after* the overlays (`_coons_at`), instead of a bilinear fill from its corners. A
    curved side therefore bows the interior with it; with four straight edges the patch
    is algebraically that bilinear fill (it differs only in float association, ~1e-16).
- **carried through**: `extrude` translates a section's whole B-rep rigidly; `blend`
  lerps the entity tables with the same `t` the corners get; `loft` sweeps each column
  as a Coons patch curved along the profile (from the slices' own nodes, `_coons_at` /
  `_slice_block` / `_sweep_at`) and straight along the sweep; `QuadMesh.annulus` is a
  single curved path at every order (radial `loft` of `LineMesh.blend`ed rings).
All factories reject a mismatched `order` across their inputs.

**Constructors.** `order: int = 1` rides every container `__init__` alongside the native
entity fields. `Quad/HexMesh.from_corners(points, conn, ...)` is the **corner → B-rep
bridge at order 1 only** — corners are all a linear mesh has, so `order > 1` raises an
actionable `ValueError` pointing at a factory with `order=N`, or at direct construction
from the entity fields (`QuadMesh(lines, quad, flip, interior=…, order=N)` /
`HexMesh(quads, hex, face_orient, interior=…, order=N)` — what every combinator uses),
rather than silently straight-subdividing invented geometry.

**Shared kernel.** `model/fields.py` supplies the GLL reference nodes (`gll_nodes`) and
`lagrange_derivative_matrix`; `model/interp.py` the dimension-general numeric primitives
(`tensor_nodes`, `corner_indices`, `subdivide_element`, `coons_grid`, `blend_ho`,
`quad_edge_indices`/`hex_edge_indices`/`hex_face_indices`, `scaled_jacobian_ho`) over the
`(E,(N+1)^d,3)` lexicographic (`i` fastest) block, which is only ever gathered
transiently. Region factories `_elevate` **before** smoothing, so a repositioning
smoother sees the true order and raises cleanly (see *Section smoothing*) instead of
silently producing a straight interior.

**Export.** `.re2` **stays linear** — Nek's re2 has no high-order format yet, so
`to_re2` reads only the 8 corners and a mesh exports byte-identically at any order.
The `.vtu` (XML VTK) writer becomes high-order at `order > 1`, emitting VTK Lagrange
cells (`VTK_LAGRANGE_CURVE=68` / `_QUADRILATERAL=70` / `_HEXAHEDRON=72`) whose `(N+1)^d`
nodes/cell index the **conformal (welded)** node array from the `conform.conformal_*`
walk, ordered via a hand-built `_lagrange_*_perm(order)` (corners → edges → faces →
interior, VTK's `PointIndexFromIJK` recursion — no `vtk`/`meshio` dep). Face nodes
inherit the face's `bc_id` via `hex_face_indices`. The writer
(`to_vtu`/`line_to_vtu`/`quad_to_vtu`) builds its node arrays via
`_hex_arrays`/`_line_arrays`/`_quad_arrays` and emits through `_write_vtu`; there is
**no legacy ASCII `.vtk` writer** — only `.re2` and `.vtu`.

**Order-N quality is opt-in** (defaults stay corner-based so pinned quality numbers
hold): `quadmesh.quality.scaled_jacobian_ho(mesh, order)` /
`hexmesh.quality.scaled_jacobian_ho(mesh, order)` sample the scaled Jacobian at the
`(N+1)^d` GLL nodes of the block gathered transiently from the mesh's B-rep via the
conformal walk (kernel `model.interp.scaled_jacobian_ho`), reached via
`mesh.scaled_jacobian(high_order=True)` / `quality_summary(high_order=True)` — at order 1
the GLL nodes are the corners so it reduces exactly to the default corner metric.
**Order-N smoothing is not implemented** (the corner-graph Laplacian/Winslow ignore
mid/interior nodes). Rather than degrade silently, a repositioning smoother **raises
`NotImplementedError` at `order > 1`** — `set_section_smoothing`
(`conduction`/`winslow`; the no-op `bilinear`/`none` stay allowed) and
`hexmesh.smoothing.smooth` both guard on `mesh.order`. Build it only when a real need
appears.

**Order 1 is a strict no-op, which is what pins the goldens.** Every order-1 code path
branches on `order`: `to_re2`, quality, topology, `merge`, `FACE_POINTS` and the `.vtu`
order-1 writer read only `points`/`conn` (the order-1 VTK path reads `points[conn]` in
Nek/CCW order), and the combinators' entity interpolation/concatenation degenerates to
the plain point blend against empty tables. The golden `bifurcation.*` (built with
defaults) is byte-identical, and treating any golden diff at default order as a bug
still holds. See `examples/high_order_{curve,quad,hex}.py`.

### Physical groups & export

`PhysicalGroups` maps name ↔ tag ↔ Nek BC code; pass `groups=` to the factories to control
`.re2` boundary codes without touching the exporter (`PhysicalGroups.duct()`,
`.from_tags()`, `.nek_default()` are presets). `.re2` element ids are 1-based on write;
all internal indices are 0-based.

## Conventions

- **Strong typing is enforced** (`mypy` with `disallow_untyped_defs`, `check_untyped_defs`,
  `disallow_any_generics`). Everything in `nekmeshpy/` is annotated. Geometry-object parameters
  take the concrete type (`LineMesh`; open-vs-closed is neither a type nor a stored flag — it is
  read off the `lines` connectivity)
  with no `| np.ndarray` fallback; only genuine
  vector *literals* (axis/origin/center) use `Sequence[float] | FloatArray`. Array-valued
  numeric internals use the dtype-parametrized aliases in `nekmeshpy/_typing.py` — `FloatArray`
  (`NDArray[np.float64]`, coordinates/real data), `IntArray` (`NDArray[np.int64]`,
  connectivity/indices), `BoolArray` (masks), `StrArray` (`NDArray[np.str_]`, `element_tags`/
  `boundary_tags`) — never a bare `np.ndarray`, which
  `disallow_any_generics` rejects as an implicit `NDArray[Any]` (use an explicit `NDArray[...]`
  for any other dtype). `Point` / `Vec3` / `PointArray` are shape-documentation aliases of
  `FloatArray` marking a single `(3,)` location / a single `(3,)` direction / **any** array of
  point coordinates whose **trailing axis is the 3 spatial components**, with any leading shape —
  `(P,3)` `points`, `(L,order-1,3)` `LineMesh.interior`, `(Q,(order-1)**2,3)` `QuadMesh.interior`,
  `(E,6,(order-1)**2,3)` gathered hex face nodes, `(ni+1,nj+1[,nk+1],3)` `from_grid` grids. The
  concrete shape belongs in each field/parameter docstring; the alias deliberately does not encode
  it. Real data that is **not** a position keeps `FloatArray`: `fractions`/`t` blend parameters,
  `layers`/`radial` positions, `x_frac`/`y_frac` grading, GLL nodes/weights and Lagrange
  (derivative) matrices, `tensor_nodes`' `(M,dim)` *parametric* reference lattice, scaled-Jacobian
  values and quality metrics, tolerances. numpy has no static shape checking, so they document
  intent only and are interchangeable with `FloatArray` to mypy.
- Full architecture, module reference, and extension points: `README.md`.
