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

PYTHONPATH=. python examples/bifurcation.py     # run a concrete mesher (writes .re2/.rea/.vtk in cwd)
```

CI (`.github/workflows/ci.yml`) runs ruff + mypy on py3.12 and pytest on py3.9–3.12. All three must stay green.

## The golden-regression invariant (read before editing anything numeric)

`tests/` freezes the output of `examples/bifurcation.py` in `tests/golden/`. The
tests assert it **byte-for-byte**: `.rea` and the `.re2` boundary block are byte-exact,
`.re2` coordinates match to `1e-12`, and `.vtk` is byte-identical. The numerics were
ported verbatim from a reference MATLAB/Octave implementation, so "results unchanged"
is a hard constraint — most refactors here are expected to be output-preserving.

After any change that could touch geometry/numerics, verify:

```bash
cd /tmp && PYTHONPATH=<repo> python <repo>/examples/bifurcation.py
for f in bifurcation.re2 bifurcation.rea bifurcation.vtk; do cmp -s "$f" "<repo>/tests/golden/$f" && echo "$f OK"; done
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
  another loop/box before an index-paired `annulus`; `element_tags` tags its line elements),
  `LineMesh.rectangle(width, height, *, center=…, normal=…, element_tags=…)` (closed 4-corner box
  loop in the given plane — a `circle`-like sibling),
  `LineMesh.far_field_box(inner, half_width, half_height=None, *, center=…, normal=…, side_tags=…)`
  (a rectangular far-field loop **index-aligned to `inner`**: one outer point per `inner` point,
  placed where the ray from `center` through that point meets the box, so it feeds `annulus`
  directly with no matching/resampling step — the box follows the body loop the way a cubed-`sphere`
  follows a `box` one dimension up; `side_tags` `[bottom, right, top, left]` names the four sides
  by each element's midpoint direction),
  `LineMesh.from_segments` (chain unordered segments into the
  largest closed loop, or `None`). Factories check `is_closed` at runtime (`ogrid`/`annulus`
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

- **Sections** are `QuadMesh` classmethods: `QuadMesh.structured` (transfinite grid from four
  open `LineMesh` edges), `QuadMesh.rectangle(corners, nx, ny, *, x_frac=, y_frac=, side_tags=)`
  (structured-grid convenience: 4 corners + counts + optional per-axis grading + `{bottom,right,
  top,left}` side tags), `QuadMesh.ogrid` (O-grid in a closed `LineMesh`),
  `QuadMesh.half_ogrid` (half-disc O-grid), `QuadMesh.annulus` (ring O-grid between two loops —
  built as a radial `QuadMesh.loft` of blended rings, the sibling of `HexMesh.annulus` one
  dimension down; the periodic ring topology rides in the loops' wrapping `lines`, so there is no
  modular arithmetic, and the inner/outer rings are the loft's near/far caps).
  `QuadMesh` also has `extrude`/`loft` **one dimension down** — sweeping a `LineMesh` into a quad
  strip (mirrors `HexMesh.extrude`/`loft`), carrying the line's `element_tags` onto the quads and
  its boundary-point tags onto the side-wall edges. Each section takes an optional
  `smoothing_method=` (see below). All build **natively in 3-D** — nothing is projected to a
  plane, so a boundary placed in any plane, or a genuinely **curvy / non-planar** boundary, is
  filled in place with its true shape (never flattened to `xy`). `ogrid`/`annulus` build a
  straight-chord initial guess and rely on `smoothing_method="conduction"` to relax the interior
  harmonically onto the curved surface spanned by the fixed boundary ring; `structured`/
  `half_ogrid` blend the 3-D edge points directly. (`LineMesh.circle`, `rectangle`, and
  `far_field_box` use the private `linemesh/_plane.py` `_in_plane_axes` helper — they *construct* a
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
`bilinear`/`none`. There is no HexMesh-level smoothing registry.

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
