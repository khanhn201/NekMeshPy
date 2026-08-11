# CLAUDE.md

## Commands

```bash
pip install -e ".[all,dev]"

ruff check nekmeshpy tests examples
mypy                          # config pins files=["nekmeshpy"]; do NOT pass paths
python -m pytest              # addopts deselect `slow`
python -m pytest -m slow      # the big chimera examples
sphinx-build -b html -n -W --keep-going docs docs/_build/html
```

Those five **are** the CI gates. A local pass is not the gate — after pushing, poll
`gh pr checks <n>`.

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

There are no `TYPE_CHECKING` guards, and the three containers have no deferred imports.
Siblings do use function-body imports in a few places (`query` → `quality`,
`shape` → `linemesh`), so a deferred import in a *sibling* is not by itself a bug; one
in a container is.

Each rung re-exports its operations, so `quadmesh.ogrid` and `quadmesh.shape.ogrid` are
the same function — prefer the short form. Names collide across rungs by design (each
rung has its own `loft`, `merge`), which is why there is no flat namespace above this
one. Adding an operation touches one sibling module plus two `__all__`s.

Siblings split on **arity** and **rung delta** (line → quad → hex):

| module | Δ | contents |
|---|---|---|
| `assemble.py` | +1 / 0 | `loft`, `loft_fn`, `loft_spline`, `merge` — n-ary; `select` / `remove` / `components` — the inverse |
| `lift.py` | +1 | `extrude` / `sweep` / `annulus` / `from_grid`; `adapter` / `bridge` hex-only |
| `lower.py` | −1 | `boundary_mesh` — the boundary **as** a mesh one rung down |
| `morph.py` | 0 | `blend`, `translate` / `rotate` / `scale` / `transform` / `mirror`; `reindex` quad-only |
| `query.py` | exit | read-only queries, incl. `bounds` / `centroid` and the rung's own measure (`length` / `area` / `volume`); hex also topology / `report` / `weld` |
| `shape.py` | +1 | shape factories — own a *shape model*, unlike `lift` |

**`loft`, `merge`, `select`/`remove`/`components` and `boundary_mesh` are the only
operations that manufacture a global index space** — `select` and its kin are `merge`
run backwards, and sit beside it for that reason. To place a new operation: *invents a
numbering?* → `assemble` (unless it is boundary extraction → `lower`); *changes rung?* →
`lift`/`lower`; *neither?* → `morph`.

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

Each container holds the rung below plus what it privately owns — `LineMesh` (`points`,
`lines`, `interior (L,N-1,3)`); `QuadMesh` (a `lines` *`LineMesh` of the shared edges* +
`quad`/`flip` + `interior (Q,(N-1)²,3)`); `HexMesh` (a `quads` *`QuadMesh` of the shared
faces* + `hex`/`face_orient` + `interior (E,(N-1)³,3)`).

`points` / `quads` / `hexes` are **derived read-only views**, so corner consistency is
structural and `mesh.points[:] = X` propagates for free. Conformality is likewise
structural: a shared edge or face is *one stored object* referenced by every incident
element, resolved by corner ids rather than coordinate search (`core/conform.py`).

Constructors share one argument order: `(rung below, incidence, [orientation,] interior,
side_tags, element_tags)`. A line element has no orientation bit, so `LineMesh` has no
`flip` slot.

**No container takes or stores `order`** — it derives all the way down: `HexMesh.order`
→ `quads.order` → `lines.order` → **`interior.shape[1] + 1`**. A mesh cannot disagree
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

The same split runs through quality: `hexmesh.quality_summary` reads the **curved**
block by default, `hexmesh.scaled_jacobian` reads **corners**. They disagree, and not
slightly — a mesh reading `0 inverted` at the corners has been measured at 8 inverted
and `minSJ -0.98` once its curved nodes are read. Corner-clean is not clean; say which
reading a number came from.

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

What makes the mesher independent of that draw is **layer thickness**, not tolerance.
`snap_to_wall` moves a node a fixed distance, so the distortion is that distance
*relative to its layer* — which makes the uniform run's length the stability knob, per
leg (`NEAR_LEN` 1.5 for the main pipe, `NEAR_LEN_BRANCH` 4.5, through `LEG_NEAR_LEN`).
Tripling it globally is much worse: the junction stops being resolved. Widening
`SNAP_MAX` buys wall accuracy by *trading away* that independence — at 0.20 alone the
mesh was flawless locally and corner-inverted on CI.

## Tags

Tag slots are types from `core/tags.py` that **validate themselves at construction** —
`PointTags`/`EdgeTags`/`FaceTags` declare their own `SIDES` (2/4/6) and reject an
out-of-range side with no mesh in sight. The slot is named for the entity:
`.point_tags` / `.edge_tags` / `.face_tags`.

**`boundary` is reserved for the topological domain boundary** — what `boundary_faces`
/ `_edges` / `_points` compute from connectivity. A side-tag table is a *named subset*
of that, and the two differ. Row order is meaningful: `ordered()` is the only sort,
because `.re2` writes rows in stored order and `.vtu` gives a node touched by several
rows the last one's tag.

`element_tags` is sparse (`ids + tags`), so an untagged mesh stores nothing and `len()`
is the *tagged* count.

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
  only) — a mesh exports byte-identically at any order. `.vtu` emits VTK Lagrange cells.
  `.re2` element ids are 1-based on write; every internal index is 0-based.
- `mypy` only checks `nekmeshpy/`, so a wrong-rung call in tests or examples surfaces as
  a pytest `AttributeError`, not a type error.

`docs/user/` is the long form (`concepts.md`, `architecture.md`, `conventions.md`).
