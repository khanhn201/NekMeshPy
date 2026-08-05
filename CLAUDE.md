# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[all,dev]"                  # numpy, scipy, matplotlib, meshio, ruff, mypy, pytest

ruff check nekmeshpy tests examples
mypy                                         # config pins files=["nekmeshpy"]; do NOT pass paths
python -m pytest                             # conftest pins the Agg backend for headless viz tests
python -m pytest -m slow                     # the 2 big chimera examples, deselected by default
sphinx-build -b html -n -W --keep-going docs docs/_build/html

pytest tests/test_pipes.py::test_quadrant_pipe_tjunction   # single test
pytest -k re2                                              # by keyword

PYTHONPATH=. python examples/bifurcation.py  # run a mesher; writes .re2/.vtu into cwd
```

**Five CI checks gate a PR** — ruff + mypy (py3.12), pytest (py3.9–3.12), and the docs
build, plus a `Slow examples` job. After pushing, poll `gh pr checks <n>` until it
settles; a local pass is not the gate.

`tests/test_examples.py` runs **every** script in `examples/` and asserts it leaves a
valid `mesh` — non-empty, watertight, conforming, no inverted element. It *discovers*
the scripts rather than listing them, so a new example is covered the moment it lands;
this exists because five examples had no coverage at all and a refactor left four of
them broken with CI green. `LIBRARY_ONLY` (`tjunction_lib.py`, `coil_lib.py`) names the scripts that build no
mesh of their own and are imported by ones that do. The two large chimera meshers carry `@pytest.mark.slow`
and are deselected by `addopts`, so a bare `pytest` shows them as *deselected*, not
passed — the `Slow examples` job is what actually runs them. `KNOWN_INVERTED` records any example that ships an
inverted element as a **strict** xfail, so fixing one fails the suite until its entry
is removed; it is currently empty and worth keeping that way. The docs build runs `-n -W` (nitpicky, warnings-as-errors), so one unresolved
autodoc reference fails it. Every `:func:`/`:class:`/`:data:` role needs a fully
qualified explicit target — ``:func:`quadmesh.blend
<nekmeshpy.quadmesh.morph.blend>` `` — because the operations are module-level
functions now, so a bare ``:func:`blend` `` has no enclosing class to resolve against.
Note the target names the **namespace** module (`quadmesh.morph`), not the private
container module: Sphinx registers each object under the `__name__` of the
module that documents it.

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
views only — and imports **no** sibling. Operations live in sibling modules and are
reached through per-rung namespaces, never bound onto the class:

```python
quadmesh.ogrid(boundary, n_side, radial)    # not QuadMesh.ogrid(...)
hexmesh.is_watertight(mesh)                 # not mesh.is_watertight()
```

Each rung re-exports its own operations, so the namespace module is optional at the
call site: `quadmesh.ogrid(...)` and `quadmesh.shape.ogrid(...)` are the same function.
Prefer the short form — the *rung* is the part that carries meaning. Names are unique
within a rung, so flattening is unambiguous; across rungs they collide by design (each
rung has its own `loft` and `merge`), which is why there is no flat namespace above
this one. Keep both `__all__`s in sync when adding an operation: the namespace module's
and the package's.

That one-way import — container ← sibling, never back — is what makes each package a
strict DAG, so every sibling does a plain `from .quadmesh import QuadMesh` and there
are **no `TYPE_CHECKING` guards and no deferred function-body imports anywhere in
`nekmeshpy/`**. Adding an operation touches one sibling module plus one line of the
namespace's `__all__`.

Siblings split on two axes, **arity** and **rung delta** (how far up the line → quad →
hex ladder the op moves):

| module | arity | Δ | contents |
|---|---|---|---|
| `assemble.py` | n-ary | +1 / 0 | `loft`, `loft_fn`, `merge` |
| `lift.py` | fixed | +1 | `extrude` / `sweep` / `sweep_path` / `annulus` / `from_grid` / `adapter` / `bridge` → `loft` |
| `lower.py` | fixed | −1 | `boundary_mesh` — the boundary **as** a mesh one rung down |
| `morph.py` | fixed | 0 | `blend`, `reindex`, `place_on_path`; unary `translate`/`rotate`/`scale`/`transform` |
| `query.py` | fixed | exit | read-only queries; hex also topology / `report` / `weld` |
| `shape.py` | fixed | +1 | shape factories — own a *shape model*, unlike `lift` |

`assemble` is load-bearing: **`loft`, `merge` and `lower`'s `boundary_mesh` are the only
operations that manufacture a global index space.** Everything else fixed-arity either
reuses an existing numbering (`blend` keeps `a`'s verbatim, `reindex` relabels onto it)
or delegates to `loft`. To place a new operation ask: *does it invent a numbering?* →
`assemble`, unless it is the boundary extraction, which is `lower`; *does it change
rung?* → `lift` (up) or `lower` (down); *neither?* → `morph`.

Δ = −1 was empty for a long time on the reasoning that a caller wanting the boundary
wants to *index* it (`boundary_faces` returns `[element, face]` pairs) rather than mesh
it. Building **onto** a finished block is what overturned that: a connector swept off a
port must start from that port's own nodes, and re-deriving them from the recipe that
built the block lands close rather than exact — which `merge` rejects at order > 1.
`boundary_mesh` reads them straight out instead, and its `template=` form reuses a
caller-supplied section's numbering for when the result has to pair index-for-index
with a section already in hand (`adapter` / `bridge` / `blend` all require that).

Namespaces, one public module per group — `assemble`, `lift`, `lower`, `morph`,
`query`, plus `shape`. These modules **are** the code — there is no private
`_assemble.py` behind `assemble.py`, and no facade layer. `shape` holds both the open
and the closed shape factories: that split was storage-side, not caller-facing.

Only genuinely internal helpers stay underscored: `linemesh/_plane.py` and
`quadmesh/_helpers.py`.

`linemesh.shape` and `quadmesh.shape` also carry the samplings —
`arclength_fractions`, `sweep_fractions`, `path_fractions`, `spine_fractions`,
`quadrant_seam_fractions`, `quadrant_core` — which return a plain array rather than a
mesh. These exist because **no factory resamples its input**: a factory meshes exactly
at the points it is given and the caller proves the sampling. (`path_fractions` is the
one that resolves a `SpacePath` plus a target element length into stations; it is what
`sweep_path` calls, and is spelled out here so the three-way `target_length` / `layers`
/ `fractions` choice is validated in one place.)

Paths and surface curves are model-level, not per-rung, because neither is a mesh:
`model/paths.py` holds the 2-D turtle walk and `embed`, which lifts it onto a plane in
space; `model/surfaces.py` holds `SurfaceCurve` and its combinators. Both import **no
container**. The rung-level entry points that consume them are `linemesh.on_surface`,
`quadmesh.tri_patch`, and `sweep_path` at the quad and hex rungs.

`TriMesh` is the exception: it keeps its own small query methods in `trimesh.py` and
exposes the rest as `trimesh.ops.*`.

Public API is re-exported from `nekmeshpy/__init__.py`; keep `__all__` in sync.

`mypy` pins `files=["nekmeshpy"]`, so it only checks toolkit code — tests and examples
are unchecked, which is why a wrong-rung call there (`quadmesh.translate(hexmesh_obj, v)`)
surfaces as a pytest `AttributeError` rather than a type error. Internal toolkit code
imports the free functions directly from the private sibling
(`from ..linemesh.shape import line`) — same modules the public namespaces name.

Operations are per-rung, so a call site **names its rung**. Code meant to run at every
rung pairs each mesh with its package explicitly rather than dispatching on
`type(mesh)` — see `tests/test_morph_transforms.py::_rungs`.

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
[orientation,] interior, side_tags, element_tags, *, order)`. A line
element has no orientation bit, so `LineMesh` simply has no `flip` slot.

Both tag slots are types from `model/tags.py`, not loose arrays, and each
**validates itself at construction** — a `PointTags`/`EdgeTags`/`FaceTags` declares its
own `SIDES` (2/4/6) and rejects an out-of-range side or a negative id with no mesh in
sight, which is what makes them three types rather than one. The single check a table
cannot make for itself is the element *count*; the containers pass their own in via
`tags.check_within(n, "quads")`. The side-tag slot is
named for the entity it names — `mesh.point_tags` on a `LineMesh`, `.edge_tags` on a
`QuadMesh`, `.face_tags` on a `HexMesh` (`PointTags` / `EdgeTags` / `FaceTags`, one
shared implementation over a private `_SideTags`). **`boundary` is reserved for the
topological domain boundary** — what `boundary_faces` / `boundary_edges` /
`boundary_points` compute from connectivity; a side-tag table is the *named subset* of
that, and the two really do differ. Each holds `(element, side, tag)` rows in one
object, so the permutation / offset / filter / concat that used to be applied twice by
hand (that is what the three `_order_bnd` copies were) is now one call that cannot
desynchronize. Its **row order is meaningful**: `ordered()` is the only sort, because
`.re2` writes rows in stored order and `.vtu` gives a node touched by several rows the
last one's tag. `element_tags` is an
`ElementTags`, sparse `ids + tags` — `""` was never a real tag, so an untagged mesh now
stores nothing. Note `len()` on it is the *tagged* count. Factory keyword arguments stay
dense (`element_tags=["wall"] * n`); the containers take the sparse types. There is no
`Point` class — a point is a `(3,)` numpy array — and input **must** be 3-D; a `(N,2)`
array is a `ValueError`, never padded to `z=0`. Open vs closed is read off the `lines`
connectivity and stored nowhere: a loop is a cycle carrying the explicit wrap row.

## The trap: high order in storage vs in geometry

Order is declared **once, at the bottom of the ladder**, and rides up: factories that
build points from nothing take `order=N`; everything taking a *mesh* in inherits it and
rejects a mismatch loudly.

Anything built from an explicit point array (`linemesh.loft`, `from_grid`) has only
those points to go on and **straight-subdivides** between them — high order in storage,
linear in geometry. A plain `quadmesh.loft`/`hexmesh.loft` is the same trap along its
*sweep* direction: exact profiles still give a surface that is straight between them (a
torus lofted from exact circles lands 62–83% of the tube radius off).

The escapes, at all three rungs: `loft_fn` (evaluates your parametrization on the
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
- **`_fn` names the functional variant.** Where an operation has both a discrete form
  (it takes the sampled data) and a continuous one (it takes a parametrization and
  evaluates it), the second is the first's name plus `_fn`: `loft` / `loft_fn` at all
  three rungs, `coons_grid` / `coons_grid_fn`. An operation that only ever takes a
  callable is *not* a variant of anything and keeps its plain name
  (`linemesh.arclength_fractions`, `sweep`'s `path=`).
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
`loft`/`loft_fn`/`sweep`, high order), `architecture.md`, `conventions.md`.
`examples/README.md` has a one-line description of what each script builds.
