# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[all,dev]"                  # numpy, scipy, matplotlib, meshio, ruff, mypy, pytest

ruff check nekmeshpy tests examples
mypy                                         # config pins files=["nekmeshpy"]; do NOT pass paths
python -m pytest                             # conftest pins the Agg backend for headless viz tests
sphinx-build -b html -n -W --keep-going docs docs/_build/html

pytest tests/test_pipes.py::test_quadrant_pipe_tjunction   # single test
pytest -k re2                                              # by keyword

PYTHONPATH=. python examples/bifurcation.py  # run a mesher; writes .re2/.vtu into cwd
```

**Four CI checks gate a PR** — ruff + mypy (py3.12), pytest (py3.9–3.12), and the docs
build. After pushing, poll `gh pr checks <n>` until it settles; a local pass is not the
gate. The docs build runs `-n -W` (nitpicky, warnings-as-errors), so one unresolved
autodoc reference fails it: any **cross-module** `:meth:`/`:class:`/`:data:` role must be
fully qualified with an explicit target — ``:meth:`QuadMesh.blend
<nekmeshpy.quadmesh.QuadMesh.blend>` `` — because a bare ``:meth:`QuadMesh.blend` ``
resolves only within the same class.

## The golden-regression invariant

`tests/golden/` freezes the output of `examples/bifurcation.py`. The contract is
**geometry to a tolerance, topology and tags exactly**: coordinates (`.re2` and the
`.vtu` `Points` block) to `1e-12`; connectivity, numbering, VTK cell types, `bc_id` and
the `.re2` boundary block byte-for-byte. The numerics were ported verbatim from a
reference MATLAB/Octave implementation, so "results unchanged" is a hard constraint and
most refactors here are expected to be output-preserving.

Floats are deliberately *not* byte-compared: the CI matrix reproduces the mesh
bit-for-bit across py3.9–3.12, but a differently built interpreter shifts coordinates by
up to ~7e-13 (float-association noise). A byte-exact float golden would be valid on one
build and red everywhere else.

The pipe examples have **no** goldens (tolerance-only quality tests) and may move; the
bifurcation must not. For a change meant to be pure, treat any golden diff beyond that
float noise — and *any* diff in the discrete data — as a bug. After any change that
could touch geometry or numerics, verify directly rather than trusting a green
`pytest` alone:

```bash
cd /tmp && PYTHONPATH=<repo> python <repo>/examples/bifurcation.py
python -m pytest <repo>/tests/test_regression.py    # coords to 1e-12 + exact structure
```

## Architecture

**Two layers, deliberately separated.** `nekmeshpy/` is a toolkit of composable
primitives containing *no* geometry-specific meshers. `examples/` holds the concrete
meshers as flat, gmsh-style scripts: constants at the top, top-to-bottom code, assign to
a `mesh` global, export. There are no mesher classes by design — a pipe mesher *is* its
script. `tests/conftest.py::run_example` executes them via `runpy.run_path` and inspects
the returned namespace, so examples double as integration tests and must keep producing
a valid `mesh` (tests also read other globals, e.g. `ns["R_MAIN"]`).

**Containers are pure data; everything acting on a finished mesh is a free function**
taking the container first. `<type>/<type>.py` holds storage, validation and derived
views only. Operations live in sibling modules and are bound onto the class in the
package `__init__` via registries — so `QuadMesh.ogrid(...)` and `mesh.is_watertight()`
stay reachable as methods, while adding an operation touches one sibling module (the
function plus one registry entry) and never the container.

Siblings split on two axes, **arity** and **rung delta** (how far up the line → quad →
hex ladder the op moves):

| module | arity | Δ | contents |
|---|---|---|---|
| `_assemble.py` | n-ary | +1 / 0 | `loft`, `loft_curve`, `merge` |
| `_lift.py` | fixed | +1 | `extrude` / `sweep` / `annulus` / `from_grid` → `loft` |
| `_morph.py` | fixed | 0 | `blend`; unary `translate`/`rotate`/`scale`/`transform` |
| `_query.py` | fixed | exit | read-only queries; hex also topology / `report` / `weld` |
| `_open.py`, `_closed.py` | fixed | +1 | shape factories — own a *shape model*, unlike `_lift` |

`_assemble` is load-bearing: **`loft` and `merge` are the only operations that
manufacture a global index space.** Everything fixed-arity either reuses an existing
numbering (`blend` keeps `a`'s verbatim) or delegates here. To place a new operation
ask: *does it invent a numbering?* → `_assemble`; *does it change rung?* → `_lift`;
*neither?* → `_morph`. Δ = −1 (a block's boundary **as** a `QuadMesh`) is empty at every
rung — `boundary_faces` returns `[element, face]` pairs, not a mesh.

Registries: `FACTORIES` (staticmethod-bound combinators), `METHODS` (instance-bound
queries and unary placements), and `HELPERS` in `linemesh/_open.py` and
`quadmesh/_open.py` — staticmethod-bound but returning a plain array rather than a mesh,
which is what keeps them out of `FACTORIES` (`LineMesh.arclength_fractions`,
`LineMesh.sweep_fractions`, `QuadMesh.spine_fractions`,
`QuadMesh.quadrant_seam_fractions`, `QuadMesh.quadrant_core`). These exist because **no
factory resamples its input**: a factory meshes exactly at the points it is given and
the caller proves the sampling.

Public API is re-exported from `nekmeshpy/__init__.py`; keep `__all__` in sync.

`mypy` pins `files=["nekmeshpy"]`, so it only checks toolkit code, and the
dynamically-bound sugar (`LineMesh.circle`, `mesh.weld()`, set via `setattr` in each
package `__init__`) is invisible to it. **Internal toolkit code must call the free
functions directly** (`from ..linemesh._open import line`, not `LineMesh.line`) so
mypy actually type-checks the call; external callers (examples/tests/users) use the
bound sugar.

## The B-rep ladder *is* the storage

There is no per-element node block anywhere and no `.curved` facade. Each container
holds the rung below plus what it privately owns — `LineMesh` (`points`, `lines`,
`interior (L,N-1,3)`); `QuadMesh` (a `lines` *`LineMesh` of the shared edges* +
`quad`/`flip` incidence + `interior (Q,(N-1)²,3)`); `HexMesh` (a `quads` *`QuadMesh` of
the shared faces* + `hex`/`face_orient` + `interior (E,(N-1)³,3)`). `points` / `quads` /
`hexes` are **derived read-only views**, so corner consistency is structural and
`mesh.points[:] = X` propagates for free.

Conformality is likewise structural: a shared edge or face is *one stored object*
referenced by every incident element, resolved by corner ids rather than a coordinate
search (`model/conform.py`; owner-wins reconciliation within `conform.entity_tol`).
`conform.conformal_line`/`_quad`/`_hex` flatten it on demand into `(nodes, conn_ho)` —
the high-order analog of `points` + `quads` — and that is what the `.vtu` writer and
`mesh.scaled_jacobian(high_order=True)` read.

All three containers share the same constructor argument order: `(rung below, incidence,
[orientation,] interior, boundaries, boundary_tags, element_tags, *, order)`. A line
element has no orientation bit, so `LineMesh` simply has no `flip` slot. There is no
`Point` class — a point is a `(3,)` numpy array — and input **must** be 3-D; a `(N,2)`
array is a `ValueError`, never padded to `z=0`. Open vs closed is read off the `lines`
connectivity and stored nowhere: a loop is a cycle carrying the explicit wrap row.

## The trap: high order in storage vs in geometry

Order is declared **once, at the bottom of the ladder**, and rides up: factories that
build points from nothing take `order=N`; everything taking a *mesh* in inherits it and
rejects a mismatch loudly.

Anything built from an explicit point array (`LineMesh.loft`, `from_grid`) has only
those points to go on and **straight-subdivides** between them — high order in storage,
linear in geometry. A plain `QuadMesh.loft`/`HexMesh.loft` is the same trap along its
*sweep* direction: exact profiles still give a surface that is straight between them (a
torus lofted from exact circles lands 62–83% of the tube radius off).

The escapes, at all three rungs: `loft_curve` (evaluates your parametrization on the
**whole** node lattice, corners *and* private interiors), `sweep` (carries one profile
rigidly along a curved path by a moving frame), or handing `loft` its intermediate
profiles as `sweep_nodes=`. Region fills (`ogrid` / `half_ogrid` / `quadrant_ogrid` /
`structured`) carry their input walls' curvature into the interior, and the combinators
carry it up the ladder. When writing a test for curved geometry, assert on the
**conformal node set**, not corners — corner-only passes on a mesh that is high-order in
storage and linear in geometry, which is exactly the failure mode worth guarding.

Order-N smoothing is not implemented: a repositioning smoother raises
`NotImplementedError` above order 1 rather than degrading silently.

## Conventions

- **Reports are `NamedTuple`s, not dicts** — `QualitySummary`, `TopologyReport`,
  `WeldResult`. `model/quality.py` holds the schema beside the `POOR_THRESHOLD` constant
  that names both its `n_poor` field and the formatted report's `poor (<…)` line, so the
  two cannot drift. `trimesh.surface_report` deliberately still returns a dict — that is
  what `format_report`'s `isinstance` dispatch keys off.
- **Typing is enforced** (`disallow_untyped_defs`, `check_untyped_defs`,
  `disallow_any_generics`). Use the dtype-parametrized aliases in `_typing.py` —
  `FloatArray` / `IntArray` / `BoolArray` / `StrArray`; a bare `np.ndarray` is an error.
  `Point` / `Vec3` / `PointArray` are shape-*documentation* aliases of `FloatArray`
  whose trailing axis is the 3 spatial components; concrete shapes belong in docstrings.
  Geometry parameters take the concrete container type with no `| np.ndarray` fallback;
  only vector literals (axis/origin/center) use `Sequence[float] | FloatArray`.
- **Export.** `to_re2` writes only the binary `.re2` (no `.rea` writer, no templates) and
  takes the full filename including extension; nothing is appended. `.re2` stays linear
  (corners only — Nek's format has no high-order form), so a mesh exports
  byte-identically at any order; `.vtu` emits VTK Lagrange cells, and `export.to_fld`
  writes the Nek field format that *does* carry the full `lx1³` GLL block.
- `.re2` element ids are 1-based on write; every internal index is 0-based.

## Further reading

`docs/user/` is the long form and is kept current: `concepts.md` (the ladder, tags,
`loft`/`loft_curve`/`sweep`, high order), `architecture.md`, `conventions.md`.
`examples/README.md` has a one-line description of what each script builds.
