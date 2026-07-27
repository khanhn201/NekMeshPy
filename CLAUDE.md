# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[all,dev]"                 # dev setup (numpy, scipy, matplotlib, meshio, ruff, mypy, pytest)

ruff check nekmeshpy tests examples          # lint
mypy                                         # type-check (config pins files=["nekmeshpy"]; do NOT pass paths)
MPLBACKEND=Agg python -m pytest              # full suite (Agg needed: viz tests import matplotlib headless)

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
cd /tmp && PYTHONPATH=<repo> MPLBACKEND=Agg python <repo>/examples/bifurcation.py
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
taking the container as its first argument — `io.export`, `io.viz`, `model.quality`,
`model.topology`, `ops.trisurf`, `ops.smoothing`, `ops.interior`. Don't add heavy methods
to the containers; add a function in the right `ops`/`model`/`io` module.

**Public API is re-exported from the top level** (`nekmeshpy/__init__.py`), so
`from nekmeshpy import ...` is stable regardless of internal file layout. Keep `__all__`
and the imports in sync when adding/removing public names.

### Geometry vocabulary follows gmsh

- There is **no `Point` class** — a single point is just a `(3,)` numpy array; all coordinates
  are plain numpy arrays.
- `Curve` (open) / `CurveLoop` (closed) — `geometry/curve.py`. A typed wrapper over a single
  `(N,3)` coordinate array exposed as `curve.points` (constructors accept any array-like, but
  input **must be 3-D** — a `(N,2)` array is rejected with a `ValueError`, not padded to
  `z=0`; all boundaries live honestly in 3-D); they carry the curve ops (`resample`,
  `resample_spline`, `align_to`, `radial_match`, `circle`, `chain`) and return
  `Curve`/`CurveLoop`. `CurveLoop.circle(radius, n, center=…, normal=…)` places the loop in the
  plane with the given `normal` (default `+z`); `radial_match` matches loops in the target's own
  plane. `CurveLoop` is intentionally **not** a subclass of `Curve`, so a param typed `Curve`
  rejects a `CurveLoop` — the open/closed distinction is enforced by the type system. Shared
  storage/sampling lives in the private `_PointSeq` base.
- `TriMesh` / `QuadMesh` / `HexMesh` / `Mesh` — mesh containers; each stores coordinates as
  a **bare `(P,3)` NumPy array** on `.points` (mutate in place with `mesh.points[:] = X`).

### Factory model (also gmsh-named)

- **Sections** are `QuadMesh` classmethods: `QuadMesh.structured` (transfinite grid),
  `QuadMesh.ogrid` (butterfly O-grid in a `CurveLoop`), `QuadMesh.half_ogrid` (half-disc
  O-grid), `QuadMesh.annulus` (ring O-grid between two loops). Each takes an optional
  `interior_method=` (see below). All build **natively in 3-D** — nothing is projected to a
  plane, so a boundary placed in any plane, or a genuinely **curvy / non-planar** boundary, is
  filled in place with its true shape (never flattened to `xy`). `ogrid`/`annulus` build a
  straight-chord initial guess and rely on `interior_method="conduction"` to relax the interior
  harmonically onto the curved surface spanned by the fixed boundary ring; `structured`/
  `half_ogrid` blend the 3-D edge points directly. (`CurveLoop.circle` and `radial_match` still
  use the private `geometry/_plane.py` frame helpers — they *construct* / *match* a planar loop,
  not project an existing boundary.) `ogrid`/`half_ogrid` are ICEM/Pointwise terms with no gmsh
  equivalent — kept deliberately.
- **Hex blocks** are `HexMesh` classmethods: `extrude` (sweep one section along a straight
  axis = gmsh Extrude+Layers+Recombine), `loft` (recombine a stack of pre-positioned
  profiles — the general case behind `extrude`), `merge` (stitch blocks, welding coincident
  **boundary** points only), `from_grid` (structured i×j×k). `HexMesh` is immutable by
  construction (no incremental building).

### Interior repositioning is per-section

Cross-section interior nodes are repositioned on a single `QuadMesh` *before* extrusion,
via `ops.interior.set_section_interior(qm, method)` (registry `SECTION_METHODS`;
extend with `@register_section_interior("name")`). Built-ins: `conduction`, `winslow`,
`bilinear`/`none`. There is no HexMesh-level interior registry.

### Physical groups & export

`PhysicalGroups` maps name ↔ tag ↔ Nek BC code; pass `groups=` to the factories to control
`.re2` boundary codes without touching the exporter (`PhysicalGroups.duct()`,
`.from_tags()`, `.nek_default()` are presets). `.re2` element ids are 1-based on write;
all internal indices are 0-based.

## Conventions

- **Strong typing is enforced** (`mypy` with `disallow_untyped_defs`, `check_untyped_defs`,
  `disallow_any_generics`). Everything in `nekmeshpy/` is annotated. Geometry-object parameters
  take the concrete type (`Curve`, `CurveLoop`) with no `| np.ndarray` fallback; only genuine
  vector *literals* (axis/origin/center) use `Sequence[float] | FloatArray`. Array-valued
  numeric internals use the dtype-parametrized aliases in `nekmeshpy/_typing.py` — `FloatArray`
  (`NDArray[np.float64]`, coordinates/real data), `IntArray` (`NDArray[np.int64]`,
  connectivity/indices), `BoolArray` (masks) — never a bare `np.ndarray`, which
  `disallow_any_generics` rejects as an implicit `NDArray[Any]` (use an explicit `NDArray[...]`
  for any other dtype). `Point` / `Vec3` / `PointArray` are shape-documentation aliases of
  `FloatArray` marking a single `(3,)` location / a single `(3,)` direction / a `(P,3)` array of
  point coordinates (vs `(N,)` scalar data); numpy has no static shape checking, so they document
  intent only and are interchangeable with `FloatArray` to mypy.
- Full architecture, module reference, and extension points: `nekmeshpy/README.md`.
