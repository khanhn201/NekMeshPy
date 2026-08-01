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

PYTHONPATH=. python examples/bifurcation.py     # run a concrete mesher (writes .re2/.vtu in cwd)

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
contract is **geometry to a tolerance, topology and tags exactly**: coordinates
(`.re2` and the `.vtu` `Points` block) match to `1e-12`, while everything discrete —
connectivity, element/node numbering, VTK cell types, `bc_id`, the `.re2` boundary
block — is compared byte-for-byte. The numerics were ported verbatim from a reference
MATLAB/Octave implementation, so "results unchanged" is a hard constraint; most
refactors here are expected to be output-preserving.

**Floats are deliberately not byte-compared.** The CI matrix reproduces the mesh
bit-for-bit across CPython 3.9–3.12 / numpy 2.0–2.5 / scipy 1.13–1.18 (all four legs
give the same md5), but a *differently built* interpreter does not: a cp314 wheel of
the same numpy 2.5.1 + scipy 1.18.0 as the 3.12 leg shifts every coordinate by up to
7.3e-13 — float-association noise, in the same class as the `spsolve`-vs-backslash
residual `RE2_TOL` already exists for. A byte-exact float golden would therefore be
valid on exactly one interpreter build and red everywhere else.

After any change that could touch geometry/numerics, verify:

```bash
cd /tmp && PYTHONPATH=<repo> python <repo>/examples/bifurcation.py
python -m pytest <repo>/tests/test_regression.py    # coords to 1e-12 + exact structure
```

The pipe examples have **no** goldens (tolerance-only quality tests), so they may
change; the bifurcation must not. When a change is meant to be pure (rename,
restructure), treat *any* golden diff beyond that float noise — and any diff at all in
the discrete data — as a bug.

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

**The report functions return `NamedTuple`s, not dicts.** `quality_summary` hands back a
`QualitySummary` (`n_elements`/`min`/`max`/`mean`/`median`/`n_inverted`/`n_poor`) whose schema
lives container-free in `model/quality.py` beside the `POOR_THRESHOLD` constant that names both
its `n_poor` field and the `poor (<…)` line of the formatted report — one number, so the two
cannot drift apart (as a `"n_below_0.2"` dict key it was baked into the public schema four times
over, and it is not even a valid identifier, which is what kept the summary a dict). `hex_report` /
`HexMesh.topology_report()` return a `TopologyReport` whose old `"kind": "hex"` discriminator is
**gone** — it existed only so `format_report` could tell a hex report from a surface one, and the
type says that now. `hexmesh` `weld` returns `WeldResult(points, hexes, n_points)`. `trimesh`'s
`surface_report` deliberately still returns a dict, which is exactly what `format_report`'s
`isinstance` dispatch keys the surface branch off.

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
  default and therefore nothing in the container that could imply a wrap. All three containers
  take the **same constructor argument order** — `(rung below, incidence, [orientation,] interior,
  boundaries, boundary_tags, element_tags, *, order)`: `LineMesh(points, lines, interior, …)`,
  `QuadMesh(lines, quad, flip, interior, …)`, `HexMesh(quads, hex, face_orient, interior, …)`.
  A line element has no orientation bit, so it simply has no `flip`/`face_orient` slot. Callers either own their
  `lines` outright (`merge`'s rewelded lines, `blend`'s copy of
  `a.lines`, the quad/hex edge `LineMesh`es built from `conform.unique_edges` or the layer-by-layer
  append) or author them with `LineMesh.loft`, which is the **only** connectivity-authoring entry
  point — there is no `open`/`loop`/`from_segments` sugar over it. Factories build the wrap where it
  belongs:
  `LineMesh.loft(points, *, loop=False, …)` (the **bottom rung of the uniform sweep primitive** —
  see *loft is the sweep primitive at every rung* below — each "profile" is a single point so the
  rungs *are* the line elements: `loop=False` gives the consecutive chain, `loop=True` appends the
  single closing rung `[N-1, 0]`. At `order > 1` an omitted `interior` is filled with the straight
  GLL blend between each line's endpoints, which is what the straight-sided factories want),
  `LineMesh.line(start, end, fractions, *, element_tag=…)` (open straight edge placed exactly at
  `start + f*(end-start)` for each normalized-arc-length `f` — the graded-edge sibling of
  `circle`/`rectangle`; `element_tag` names every line element),
  `LineMesh.loft_curve(f, fractions, *, loop=False, order=1, element_tags=…)` (the
  **general sibling of `LineMesh.arc`**, and — being the one op besides `loft`/`merge` that
  authors a global index space — it lives in `linemesh/_assemble.py` beside `loft`, not with
  the shape factories: it *is* `loft`, with the profiles **evaluated** from a parametrization
  instead of handed in, so open-vs-closed is a `loop` flag here exactly as it is there. Meshes a
  curve on its own analytic parametrization. `f` maps a `(K,)` parameter array to `(K,3)` points
  and is called **once with the whole node lattice** — corners *and* the private high-order
  `interior` GLL nodes — so nothing is ever placed on a chord. This is what closes the
  straight-GLL-subdivision trap below: reach for it whenever the curve has a closed form that is
  not a circular arc (an ellipse, a helix, a cylinder–cylinder intersection) instead of sampling
  it into an array and calling `LineMesh.loft`. `fractions` are the **parameter values themselves**,
  passed to `f` verbatim — no normalization, no remapping: node `k` is `f(fractions[k])` and
  there are `len(fractions)-1` elements. The caller states the domain by choosing the values: for
  an `f` written on `[0,1]` they are exactly the normalized fractions the sibling `LineMesh.line`
  takes, and an `f` written on any other interval is sampled in its own units
  (`loft_curve(f, np.linspace(0.0, np.pi, n+1))`); a descending sequence is the supported way to
  reverse the traversal. The grading is honored **per element** at `order > 1`:
  element `i`'s private `interior` rides the GLL nodes of its own `fractions[i]..fractions[i+1]`
  span (`_refined_lattice`), which the old `n`-only form could not express. Even **arc-length**
  spacing is now an explicit caller step rather than a mode: the helper
  `LineMesh.arclength_fractions(f, n, *, t_range=(0.0, 1.0), samples=1001)` returns the `(n+1,)`
  **parameter values** spanning `t_range`, evenly spaced by arc length, to hand straight to
  `loft_curve` unscaled. It keeps a `t_range` where `loft_curve` has none because it genuinely
  needs a
  domain: it inverts a cumulative **chord**-length table of `samples`
  dense evaluations of `f` over that interval; only the node *positions along* the curve inherit that
  table's discretization error — every node still lies on the curve to machine precision, because
  `loft_curve` places it by evaluating `f` and never by interpolating the table, so raising
  `samples`
  improves the evenness of the spacing, not the accuracy of the curve. It is bound onto
  `LineMesh` as a `staticmethod` through a **`HELPERS` registry** in `linemesh/_open.py`, the
  twin of `quadmesh/_open.py`'s and kept out of `FACTORIES` for the same reason (it returns a
  plain array, not a mesh) — the toolkit-wide rule made consistent: a factory meshes exactly at
  the points given and the caller proves the sampling, which is why `spined_ogrid` stopped
  resampling its spine too. `arc` is **not**
  reimplemented on top of it: `arc` places its nodes without an inversion and to the last ulp.
  `loop=True` closes the curve exactly as it does on `loft`, and takes the **trailing wrap
  value**: pass `n+1` fractions whose last maps back to the first point
  (`np.linspace(0, 2*np.pi, n+1)` for a `2π`-periodic `f`) and the result is a ring of `n`
  points and `n` lines with no degree-1 end, the seam element's own private `interior`
  evaluated on `fr[n-1]..fr[n]` like every other element's. The wrap value is *why* the
  convention is `n+1` fractions rather than inferring a period — without it the seam element
  has no far parameter to evaluate at and would be straight-subdivided, which is the whole
  defect this function exists to close. `f(fr[-1])` must land on `f(fr[0])` within
  `conform.entity_tol` or it is a loud `ValueError`. Welding two ends with `LineMesh.merge`
  after an open mesh (`bifurcation.py::fourier_wall`) remains valid and is what a curve
  assembled from *several* parametrizations still needs. For a curve with **no** closed
  form (a scanned polyline) there is nothing to evaluate — but a *closed* scanned loop can be
  given one by refitting it: `bifurcation.py::fourier_ring` rFFTs `x`/`y`/`z` against the
  uniform ring parameter, keeps the lower half of the modes (dropping the STL facet noise a
  high-order wall would otherwise resolve faithfully) and hands the resulting series to
  `LineMesh.loft_curve`. An **open** scanned arc refits in the basis its endpoints demand:
  `bifurcation.py::_arc_curve` expands the arc's deviation from its own chord as a truncated
  **sine** series in the normalized arc-length parameter (a type-I DST of dense
  uniform-arc-length samples, truncated to the modes the mesh can resolve), then meshes it
  with `LineMesh.loft_curve` + `LineMesh.arclength_fractions`. Every `sin(k*pi*s)` vanishes at both
  ends, so `A1`/`A2` stay **bit-exact** for any truncation — which is what lets the three
  *shared* seam arcs be refit **once, globally** and handed to both legs that see each one, so
  the blocks still weld. (A per-leg refit could not: it would mix the two arcs that leg
  happens to see. Leaving the seam ring on `LineMesh.loft` instead was the other trap — it
  straight-subdivides, and that one station carried 63° of corner at its element joints while
  every Fourier station sat within 0.2°.) Where even that does not apply, use
  `trimesh.ops.resample_polyline` and
  accept the chord (bifurcation's *spine* still does, deliberately: a flat half-disc seam
  needs no bow). Worked example: `circular_pipe_tjunction.py::arc_collar`, where the
  T-junction collar — the intersection of two equal-radius cylinders, hence a pair of **planar
  ellipses**, `p(t) = (xside*R sin t, R cos t, R sin t)`, `t: 0 → π`, in the plane
  `x = xside*z` with semi-axes `R*sqrt(2)` and `R` — used to be a 400-point sampled polyline
  lofted straight, putting the interior GLL nodes 1.3e-2 off the true curve at order 2 (2.6% of
  `R`); it is now `LineMesh.loft_curve(f, LineMesh.arclength_fractions(f, N_HALF, t_range=(0.0, np.pi)),
  order=ORDER)`, meshed in the ellipse's own `t` units and exact to machine precision at any
  order),
  `LineMesh.circle(radius, n, *, center=…, normal=…, start_theta=0.0, element_tags=…)` (closed;
  places `n` evenly-spaced points in the plane with the given `normal`, default `+z`, point `k` at
  angle `2πk/n + start_theta` so `start_theta` rotates the whole loop to align its index 0 with
  a `rectangle` far field's lower-left corner (`atan2(-h, -w)`) before an index-paired `annulus`;
  `element_tags` tags its line elements),
  `LineMesh.rectangle(width, height, n, *, center=…, normal=…, side_tags=…)`
  (closed far-field loop in the given plane, discretized into `n` line elements — `n` a positive
  multiple of 4: `n // 4` evenly spaced per side, running CCW from the lower-left corner with the
  corners always landing on a point, so it is a true rectangle; `side_tags` is a **mapping** keyed
  `bottom`/`right`/`top`/`left` naming each side's line elements — an absent key leaves that side
  untagged, an unrecognized one is a loud `ValueError`, and the keys rather than a positional
  4-sequence are what make this spelling identical to its one-rung-up twin `QuadMesh.rectangle`
  (a transposed 4-list silently lost a wall). It is the far-field outer loop for `annulus`: pass `n` equal to the
  inner loop's point count and rotate the inner `circle` with `start_theta` to the lower-left
  corner, and the two loops pair index-for-index — the radial spokes need not be straight),
  Ordering an unordered segment soup into a ring is a *surface* op, not a container
  constructor, so it lives beside its only caller as `trimesh.ops._chain_segments`, which ends in
  `LineMesh.loft(..., loop=True)`. `LineMesh.merge(meshes, *, tol=…)` is the 1-D
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
  reads only `boundary.points` (`ogrid`, `half_ogrid`, `quadrant_ogrid`, `structured`)
  therefore accepts an open
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
    `half_ogrid`/`quadrant_ogrid` from the arc's per-segment tags (and, for the latter,
    each seam from its own), `structured` from each edge's uniform tag.
    The factories' scalar/mapping args (`wall_tag` / `inner_tag` / `outer_tag` /
    `side_tags[side]`) are **overrides**: a non-empty arg replaces the line-level tag for
    that wall/side (**upper overrides lower**); an empty/absent arg falls through to it (and a
    present-but-empty `side_tags[side]` / `NO_BOUNDARY` suppresses the side). It is spelt
    `side_tags`, not `boundary_tags`, precisely because these are *named sides*: `boundary_tags`
    everywhere else is the sparse `StrArray` running parallel with a mesh's `boundaries (Nbc,2)`
    rows, a different shape entirely. The examples
    tag at the line level and keep only the hex-level `first_tag`/`last_tag` end caps (no lower
    level exists for those).
  - `boundary_tags` — a **sparse** `StrArray` parallel with `boundaries (Nbc,2)` = `[elem id,
    side]`. A `LineMesh`'s boundary is its end **points** (`side ∈ {1,2}` → local vertex `s-1`);
    on `extrude` they become quad boundary **edges**, then hex boundary **faces**.
    `boundary_group_tags` = sorted unique. (See `examples/flow_past_cylinder.py`.)
- `TriMesh` / `QuadMesh` / `HexMesh` / `Mesh` — mesh containers; each stores coordinates as
  a **bare `(P,3)` NumPy array** on `.points` (mutate in place with `mesh.points[:] = X`).
  `QuadMesh` and `HexMesh` also carry a dense `element_tags` and sparse `boundaries`/
  `boundary_tags`, mirroring `LineMesh` one/two dimensions up. All four (with `LineMesh`) render
  the **same one-line `__repr__`** — counts, `order`, tag vocabulary — so they read as a family at
  the REPL this toolkit is mostly driven from. It is deliberately **cheap** (stored array shapes
  and the two tag properties only, no topology derived) and **total**: any failure degrades to a
  bare marker rather than raising, because a repr that throws on a half-built or degenerate mesh
  makes the debugging session it exists to serve strictly worse.

### Module layout: one axis pair, three identical packages

Each container package is split the same way. `<type>.py` is a **pure data container**
— storage, validation, `from_corners`, and the derived views, with no operation code and
no base classes. Everything that *acts* on a finished mesh is a **plain free function**
(no `cls`/`self`) in a sibling module, **bound onto the class in the package `__init__`**
(`setattr(Class, name, staticmethod(fn))`), so `LineMesh.circle(...)` /
`QuadMesh.ogrid(...)` / `mesh.is_watertight()` all stay reachable as methods.

The siblings are split by two orthogonal axes: **arity** (fixed vs variable/n-ary) and
**rung delta** (how far up or down the line → quad → hex ladder the operation moves):

| module | arity | Δ | contents |
|---|---|---|---|
| `_assemble.py` | **n-ary** | +1 / 0 | `loft`, `loft_curve` (all three rungs), `merge` |
| `_lift.py` | fixed | +1 | quad `extrude`/`sweep`/`annulus`/`from_grid`; hex `extrude`/`sweep`/`annulus`/`from_grid` |
| `_morph.py` | fixed | 0 | binary `blend`; **unary** `translate`/`rotate`/`scale`/`transform` |
| `_query.py` | fixed | exit | read-only queries; hex also topology / `report` / `weld` |
| `_open.py` / `_closed.py` | fixed | +1 | shape factories (own a *shape model*, hence separate from `_lift` — `annulus` owns none, so it is a `_lift` at both rungs). `hexmesh/_open.py` holds `tetra`, whose inputs are a rung *two* below its output |

`_assemble` is the load-bearing boundary: **`loft` (with `loft_curve`, which delegates to it)
and `merge` are the only operations
that manufacture a global point/element index space** (`loft`'s `prof_off`/`rung_off`
and `i*nn + v`; `merge`'s `remap`/`survivors`/`point_id`). Every fixed-arity operation
either reuses an existing numbering (`blend` keeps `a`'s verbatim) or delegates here
(`extrude`/`annulus`/`from_grid` all end in a `loft` call and carry its numbering up
unchanged). That is the question to ask when placing a new operation: *does it invent a
numbering?* → `_assemble`; *does it change rung?* → `_lift`; *neither?* → `_morph`.
The Δ = −1 cell (a block's boundary **as** a `QuadMesh`) is empty at every rung today.

**`LineMesh.reverse()`** is the other unary Δ0 op: the same curve traversed the other
way, `i → N-1-i`. It moves no coordinate — it relabels — and carries the high-order
`interior` with it (flipped on **both** axes), which is exactly what
`LineMesh.loft(mesh.points[::-1])` silently gets wrong: with no explicit `interior`,
`loft` refills it with straight chords and the curve is lost at `order > 1`. Tags and
`boundaries` are remapped (line `l → L-1-l`, side `s → 3-s`) so a tagged end point stays
on the same physical point. It is 1-D only: a quad/hex has no single traversal direction
(the analogous operation there is an orientation flip, a different thing).

**The unary Δ0 cell is the affine placements** — `translate(vector)` / `rotate(angle, axis=,
center=)` (radians, right-handed, Rodrigues) / `scale(factor, center=)` (scalar or per-axis)
and the general `transform(matrix, offset)` they all wrap. Each returns a **new** mesh with
only coordinates moved: incidence, `element_tags`, `boundaries` and `boundary_tags` ride
through verbatim, and the map reaches **every** coordinate table, so a curved element keeps
its shape (a rotated `LineMesh.circle` stays an exact circle). Like `blend` they are composed
downward — `HexMesh` maps its shared-face `QuadMesh` and lerps nothing but its own `interior`;
`QuadMesh` maps its shared-edge `LineMesh`. The `(matrix, offset)` pairs come from
`model/affine.py` (`translation`/`rotation`/`scaling` + `apply`), which imports no container.
A **pure translation carries `matrix=None`** so `apply` adds the offset without a matmul: it
is bit-exact, which is what lets both `extrude`s place their slices through `translate` and
keep the goldens byte-identical. Reach for these when placing a finished mesh (a revolved
profile stack, a block about to be `merge`d); a factory that can *construct* in position
(`circle(center=…, normal=…)`) still should.

Each module ends with a registry — `FACTORIES: dict[str, Callable[..., <Class>]]` for
the `staticmethod`-bound combinators, `METHODS` for the instance-method-bound queries
and the unary placements (`mesh.translate(v)`, not `LineMesh.translate(mesh, v)`), plus
a third one, `HELPERS`, in **both** `linemesh/_open.py` and `quadmesh/_open.py`: also
`staticmethod`-bound, but for functions that answer a question *about* a factory's input
contract and return a plain array rather than a mesh, which is what keeps them out of
`FACTORIES` (`LineMesh.arclength_fractions`, `LineMesh.sweep_fractions`,
`QuadMesh.spine_fractions`, `QuadMesh.quadrant_seam_fractions` — the samplings a caller must
prove now that no factory resamples; plus `QuadMesh.quadrant_core`, the core patch
`quadrant_ogrid` builds its own core with, public so that a block filling the region
*behind* a quadrant face lands on the same points instead of reproducing the formula) —
and the package `__init__` merges the dicts and binds every entry. There is no import
cycle to break: the container never imports its operation modules, so they import it
directly at module level.

- **Adding an operation** touches **only** the matching sibling module: add the free
  function (no `cls`/`self`) and one `FACTORIES`/`METHODS` entry. The container file and
  the `__init__` binding loop need no edit.
- **mypy** pins `files=["nekmeshpy"]`, so only toolkit code is checked and the
  dynamically-bound `LineMesh.circle` etc. are invisible to it. **Internal toolkit code
  must call the free functions directly** (e.g. `from ..linemesh._open import line`,
  `from ..quadmesh._morph import blend as quad_blend`), never `LineMesh.line` /
  `QuadMesh.blend` / `mesh.weld()`; external callers (examples/tests/users) use the
  bound `LineMesh.circle(...)` / `mesh.is_watertight()` sugar.

- **Sections** are `QuadMesh` classmethods: `QuadMesh.structured(edges, *, side_tags=,
  smoothing_method=)` (transfinite grid from four
  open `LineMesh` edges — `edges` either the `[bottom, right, top, left]` sequence or, preferably,
  a **mapping** keyed by those same four names, since in the positional spelling only the position
  says which edge is which and transposing two yields a plausible-looking twisted patch instead of
  an error, whereas a missing or misspelt key raises; its per-side tag argument is spelt
  `side_tags`, not `boundary_tags`, for the reason given above),
  `QuadMesh.rectangle(corners, nx, ny, *, x_frac=, y_frac=, side_tags=)`
  (structured-grid convenience: 4 corners + counts + optional per-axis grading + `{bottom,right,
  top,left}` side tags), `QuadMesh.ogrid` (O-grid in a closed `LineMesh`),
  `QuadMesh.half_ogrid` (half-disc O-grid bounded by a wall arc + a spine diameter),
  `QuadMesh.quadrant_ogrid(arc, seam1, seam2, radial, *, center_scale=, wall_tag=, side_tags=,
  smoothing_method=)` (**quarter-disk** O-grid — the 90-degree sibling of `half_ogrid`, and
  exactly the quarter of `ogrid` you get by cutting a full disk along two perpendicular
  diameters through its core-edge midpoints, so four of them `merge` back into a conforming
  disk. Two blocks: an `n x n` core plus one `2n x Nradial` ring band wrapping the core's far
  corner, from an `arc` of exactly `2n+1` points running `A1 -> A2`. **Both seams are
  arguments, not derived from a `center`** — that is the whole point: two adjacent quadrants
  hand in the *same* `LineMesh` object (the second through `LineMesh.reverse`) and therefore
  weld bit-exactly instead of to a tolerance, the `spined_ogrid` precedent one rung finer.
  Each seam is **meshed exactly at the points given** and must carry exactly
  `n+1 + Nradial` points ascending from the center `O` — the `n+1` core fan, then the
  `Nradial` ring stations — or it is a loud `ValueError`, never a silent reinterpolation;
  derive that sampling with `QuadMesh.quadrant_seam_fractions(n_side, radial, center_scale)`
  (a `HELPERS` entry, not a factory: it returns a plain array). Its one non-obvious term is
  `center_scale * cos(pi/4)`: `center_scale` places the core's *far* corner `K` on the arc
  midpoint's radius, while the seam's core end `M` is the **midpoint of the core square's
  side**, half a diagonal further in — a caller who naively uses `center_scale` gets a
  visibly skewed core. At `order > 1` a bowed seam is meshed exactly: both seams go down the
  same `Overlay` channel as the O-rings, from their own private nodes, so nothing is
  straight-subdivided between seam samples. `side_tags` is keyed `seam1`/`seam2` only.
  Its core patch is public as `QuadMesh.quadrant_core(arc, seam1, seam2, *, center_scale=)`
  → an `(n+1,n+1,3)` grid. A quadrant face is itself a **three-patch triangle** (its core
  plus the two halves of its ring band), which is exactly what `HexMesh.tetra` consumes —
  so the region *behind* three quadrant faces meeting at `O` is filled by handing them and
  a wall patch to `tetra`, and the **octant of a 3-D O-grid** (`n^3` core + three
  `n x n x Nradial` slabs) falls out of the generic tetrahedron split. Used by `examples/quadrant_pipe_tjunction.py`, which meshes a
  small-branch T-junction as a **single welded component** by making one quadrant of the
  main pipe *be* a quadrant of the branch's footprint disc: four regions (two legs, the
  branch stub, two crotch caps) meet at the axes-crossing point and every interface between
  them is a quadrant face radiating from it. That example is the worked case for the
  straight-subdivision traps below hitting at once — wall curves carried as a
  parametrization and meshed with `LineMesh.loft_curve`, each leg's transition a
  `HexMesh.loft_curve` rather than a `loft` (straight along the sweep: 7.2e-4 off the
  cylinder at order 3), and the caps nested `loft_curve` blocks rather than
  `HexMesh.from_grid`, which blends straight from corners and would both leave the wall and
  disagree with `quadrant_ogrid`'s bowed ring bands. Result: corners bit-identical at
  `ORDER` 1-4, wall nodes on their cylinder to 2.2e-16),
  `QuadMesh.spined_ogrid(boundary, radial, *, spine=None, center_scale=, wall_tag=, smoothing_method=)`
  (full-disk O-grid over a closed `boundary` loop with a natural `A1..A2` seam: it splits the loop
  at index `0`/`M//2` into two `A1->A2` half-arcs, runs two `half_ogrid`s and `merge`s them, so
  the caller hands in only the loop (and optional spine) instead of hand-rolling the
  split/merge. The **spine is meshed exactly at the points given** — this factory used to
  arc-length-resample whatever you handed it, and was the one place violating the toolkit-wide
  exact-points rule above; it no longer resamples anything. A caller-supplied spine must
  therefore carry exactly `2*Ntheta+1 + 2*Nradial` points ascending `A1 -> A2` (the `[north
  caps, center fan, south caps]` sampling `half_ogrid` consumes) or it is a loud `ValueError`
  rather than a silent reinterpolation. Derive that sampling with the new public helper
  `QuadMesh.spine_fractions(n_theta, radial, center_scale)` — it returns the normalized
  fractions, so callers evaluate their own curve there (`LineMesh.loft_curve` for an analytic spine,
  `trimesh.ops.resample_polyline` for a scanned one) instead of copy-pasting the formula. It is
  bound onto `QuadMesh` as a `staticmethod` through the **`HELPERS` registry** in
  `quadmesh/_open.py`, kept distinct from `FACTORIES` because it returns a plain array rather
  than a mesh. The second half consumes `LineMesh.reverse()` of that *same* spine mesh, so both
  halves share the seam bit-for-bit — previously two independent resamples agreed only to ~4e-16
  and the `merge` welded by tolerance. `spine=None` (the default) still gives the straight
  `A1..A2` chord `boundary.points[[0, M//2]]`, which the factory owns as a shape and so places
  itself via `linemesh._open.line` — the common case for a planar disc
  (`circular_pipe_tjunction.py`); pass a curve only to bow the seam (`bifurcation.py`).
  **A curved spine's high-order nodes are the seam geometry**: at `order > 1` the spine's point
  intervals partition the half-disk's flat side one-for-one, so `half_ogrid` overlays each onto
  the seam edge it spans — intervals `[0, Nradial)` the north radial caps (outermost first, hence
  reversed), `[Nradial, Nradial+2*Ntheta)` the inner block's `j == 0` row, the rest the south
  radial caps — through the same `Overlay` channel `_elevate` already uses for the O-rings, so a
  bowed seam is meshed exactly instead of straight-subdivided between spine points (measured at
  order 2: interior seam nodes went from 4.7e-3 off the true spine to 5.6e-17). The spine must
  therefore carry the **same `order` as the arc** — a mismatch is a loud `ValueError` — and
  `spined_ogrid`'s default straight chord is built at `boundary.order`. Order 1 is untouched
  (the bifurcation goldens stayed byte-identical)),
  `QuadMesh.annulus` (ring O-grid between two loops —
  built as a radial `QuadMesh.loft` of blended rings, the sibling of `HexMesh.annulus` one
  dimension down; the periodic ring topology rides in the loops' wrapping `lines`, so there is no
  modular arithmetic, and the inner/outer rings are the loft's near/far caps).
  `QuadMesh` also has `extrude`/`sweep`/`loft`/`loft_curve` **one dimension down** — sweeping a `LineMesh` into a quad
  strip (mirrors `HexMesh.extrude`/`sweep`/`loft`), carrying the line's `element_tags` onto the quads and
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
  `smoothing_method=` (see below), and every `layers=` / `radial=` argument at either rung takes
  **either** a plain `int` `n` — `n` uniform layers, counting *cells* exactly as
  `uniform_spacing`/`geometric_spacing` do, expanded to the `n+1` positions `linspace(0,1,n+1)` —
  or an explicit array of normalized positions for a graded sweep, through the one shared
  `model.fields.validate_layers` contract. The array branch is returned flattened and otherwise
  **untouched, bit-for-bit**, which is what keeps the graded goldens frozen; only a genuine scalar
  integer takes the count branch, so an array of ints is a position array like any other. All build **natively in 3-D** — nothing is projected to a
  plane, so a boundary placed in any plane, or a genuinely **curvy / non-planar** boundary, is
  filled in place with its true shape (never flattened to `xy`). `ogrid`/`annulus` build a
  straight-chord initial guess and rely on `smoothing_method="conduction"` to relax the interior
  harmonically onto the curved surface spanned by the fixed boundary ring; `structured`/
  `half_ogrid`/`quadrant_ogrid` blend the 3-D edge points directly. (`LineMesh.circle` and `rectangle`
  use the private `linemesh/_plane.py` `_in_plane_axes` helper — they *construct* a
  planar loop, not project an existing boundary.) `ogrid`/`half_ogrid`/`quadrant_ogrid` are ICEM/Pointwise terms
  with no gmsh equivalent — kept deliberately.
- **`HexMesh.tetra(faces, *, center=)`** (`hexmesh/_open.py`) fills the curvilinear
  **tetrahedron** enclosed by four triangular `QuadMesh` faces, each meshed as **three
  structured patches meeting at one interior node**. The structure is *recovered* from the
  connectivity — a three-patch triangle has exactly 3 nodes on one quad and 1 node on three
  — so nothing is declared and a bad face is a loud `ValueError`. The fill is the classic
  tet split, **one hex block per corner**, each inheriting whatever split the faces already
  carry along its three edges; a block's three outer sides *are* the faces' patches, taken
  verbatim (so the mesh is exact wherever the faces are), and its three inner sides are
  transfinite patches of two face spokes and two chords into `center`, computed identically
  from both blocks that share them. `center` defaults to the centroid of the four face
  centres — pass one when three faces are much smaller than the fourth, since the natural
  centroid then lands near their plane and three coplanar edges at a corner is a flat cell.
  Each face's **`element_tags`** name the boundary faces it becomes, and travel *with the
  face* through the internal reordering. Order N rides through: face nodes come off the
  conformal walk and each block is assembled from its full node lattice through the three
  rungs' `loft` (`LineMesh.loft(interior=)` → `QuadMesh.loft(sweep_nodes=)` →
  `HexMesh.loft(sweep_nodes=)`), never from corners — which is what `from_grid` cannot do.
- **Hex blocks** are `HexMesh` classmethods: `extrude` (sweep one section along a straight
  axis = gmsh Extrude+Layers+Recombine), `sweep` (carry one section rigidly along a
  **curved** path by a moving frame — the curved generalization of `extrude`; a bent pipe),
  `loft` (recombine a stack of pre-positioned
  profiles — the general case behind `extrude`), `loft_curve` (`loft` with the sections
  **evaluated** from a parametrization, exact along the sweep at `order > 1`),
  `annulus` (fill the 3-D shell between two
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
| `QuadMesh.loft(slices, *, loop=…, sweep_nodes=…)` | a `LineMesh` | a rung **line** + the **quads** | `QuadMesh.extrude` |
| `HexMesh.loft(slices, *, loop=…, sweep_nodes=…)` | a `QuadMesh` | a rung **face** + the **hexes** | `HexMesh.extrude` |

**A `loft` is straight along the sweep** — it sees only the corner-level profiles, so
at `order > 1` the rung interiors are a lerp and each quad interior a Coons patch over
lerped sweep curves. Exact input profiles therefore do **not** give an exact swept
surface: a torus lofted from exact `LineMesh.circle` rings puts its nodes 62–83% of the
tube radius off the true tube at orders 2–4. `QuadMesh.loft_curve(f, fractions, *,
loop=…, order=None, element_tags=…)` is the escape — the quad-rung twin of
`LineMesh.loft_curve`, and the same sentence: it *is* `loft`, with the profiles
**evaluated** from a parametrization instead of handed in. `f` maps **one** parameter
value to **one** `LineMesh` profile (not a vectorized `FloatArray -> …`: a callable
returning a single mesh can only take a single value) and is called once per node level
of the sweep — `_refined_lattice(fractions, order)`, the same line-rung helper, so
grading is honored per layer (level `a` of layer `i` at
`fr[i] + g[a]*(fr[i+1] - fr[i])`) and at order 1 the lattice is exactly `fractions`,
making the order-1 path a no-op by construction. `order=None` (the default) **infers** the
order from the profile `f` returns — unlike `LineMesh.loft_curve`, whose `f` hands back
bare coordinates, so there is nothing there to read an order off and the argument stays a
constructive `int`. `fractions` are the parameter values
themselves in `f`'s own units, and `loop=True` takes the **trailing wrap value** exactly
as at the line rung, verified against `conform.entity_tol`. Every profile must be
index-paired and conformal with the first (same point count, same `lines`) — the robust
idiom is to build one profile and *place* it with the `_morph` affine ops
(`ring.rotate(t, axis=…)`), which move no index; rebuilding it per parameter with a
rotating `LineMesh.circle(normal=…)` is not guaranteed to be, because `_in_plane_axes`
can flip the basis. `element_tags` here is **per sweep layer** (dense, length `nz`),
overriding the profiles' per-line tags where non-empty — upper overrides lower, as
everywhere else; it is available on plain `loft` too.

It delegates the whole assembly through **`QuadMesh.loft`'s `sweep_nodes=`**:
`sweep_nodes[i]` is the `order-1` intermediate `LineMesh` profiles lying strictly
between slice `i` and the slice it sweeps to — the quad-rung analogue of
`LineMesh.loft`'s `interior=` override, stated in rung-below vocabulary so none of
`loft`'s `used`/`rung_slot` index space leaks into the API. With it supplied the rung
line interiors and the quad interiors are gathered from those true profiles instead of
interpolated; `sweep_nodes=None` (and order 1, where it is empty) leaves the existing
code path **byte-for-byte**, which is what keeps the goldens frozen. Numbering, tags,
boundaries and the B-rep all come from `loft` unchanged — sweep-major, no `unique_edges`
re-derivation.

**`HexMesh.loft_curve(f, fractions, *, loop=…, order=None, element_tags=…)` is the same
thing one rung up**, `f` mapping one parameter value to one `QuadMesh` section (the order
again defaulting to that section's own), and it
delegates the same way through **`HexMesh.loft`'s `sweep_nodes=`** (the `order-1`
intermediate `QuadMesh` sections strictly between slice `i` and the slice it sweeps to).
The hex rung turned out to be the *easier* half, not the harder one: `HexMesh.loft`
derives its whole B-rep — including the D4 `face_orient` codes — from the corner table
via `unique_edges` → `canonical_faces`, so the codes fall out of the corner ids and the
intermediate sections only ever supply *coordinates* (edge nodes, face nodes, private
interiors) into tables whose topology is already settled.

**`QuadMesh.sweep` / `HexMesh.sweep` are the *rigid* sweep** — the same profile carried
along a curve by a moving frame rather than a stack of profiles the caller positions.
They live in `_lift.py` (fixed arity, Δ+1: they are the curved generalization of
`extrude`, which translates the section along a straight axis) and end in the same
`loft`-with-`sweep_nodes` assembly, so a swept bend is exact at every order:

```python
HexMesh.sweep(section, path, fractions, *, origin, tangent=None,
              orientation="transport", up=None, twist=0.0, close_twist=True,
              normal=None, loop=False, element_tags=None,
              first_tag="", last_tag="")
```

The load-bearing property is that the section is placed **rigidly** —
`p ↦ path(t) + R(t) @ p_local` — never offset point-by-point. Through a bend of radius
`Rb` the outboard wall traverses radius `Rb + d` and the inboard `Rb − d`, so the two
travel different distances and *neither* follows the centreline; only a frame-carried
rigid placement gets that right (a U-turn's walls come out at exactly `Rb ± Rp`). `path`
is vectorized `(K,) -> (K,3)` because the default frame generator is a sequential
integration and cannot be evaluated at one isolated parameter. `loop=True` appends the
**identical** first placement as the wrap profile, so a closed sweep (a solid torus)
welds exactly rather than to a tolerance. A bend tighter than the section is wide folds
the inboard elements and is rejected loudly by `loft`'s mixed-winding guard.

Three of those arguments are the way they are because the obvious spelling was quietly
wrong. **`origin` is required** — it is the section's reference point, the one that rides
the path; it used to default to the centroid, which is defensible and frequently wrong
(an O-grid disc's centroid misses its centre by the grid's own slight asymmetry), so the
obvious call produced an off-axis block with no error anywhere. There is no safe default,
so there is no default: pass the centre the boundary loop was built about. **There is no
`order=`** — a rigid placement cannot change the order, so the order is the section's own
and a separate argument could only disagree with it. And **`orientation` names a *mode*
and nothing else**, `Literal["transport", "fixed", "frenet"]`; the per-station up vectors
that used to be smuggled through it are now a `(K,3)` `up=` with `orientation="fixed"`,
so `up` takes either a single `(3,)` world direction or a per-station field, told apart
by rank.

`LineMesh.sweep_fractions(breaks, total_length, target)` authors the `fractions` for a
**piecewise** path: `breaks` are the cumulative arc lengths of the path's *interior*
junctions (strictly inside `(0, total_length)` — `0` and `total_length` are stations
anyway) and each interval between consecutive breaks is split into
`max(1, round(interval / target))` equal steps *on its own*, so every junction reappears
in the output bit-for-bit instead of being approached by a global `linspace`. That is the
whole point: curvature is piecewise constant along such a path and **jumps** at a
junction, so an element straddling one is fitted across two different geometries — a
visible kink in the wall of a swept bend. Like `arclength_fractions` it is a `HELPERS`
entry rather than a factory, because the sweep itself meshes exactly at the stations
given (`examples/serpentine_pipe.py`).

`model/frames.py` owns the frame machinery and, like `model/affine.py`, **imports no
container**: `tangents` (O(h²) finite difference — pass `tangent=` for the analytic
derivative when the end stations matter, ~3e-4 rad of tilt otherwise), the three
generators `fixed_up` (exact, zero-twist, right for planar paths) / `parallel_transport`
(RMF by double reflection, for genuinely non-planar paths) / `frenet` (present but wrong
for sweeps — undefined on straight runs, sign-flips through inflections), `plane_frame`
(the section's own authored frame, from an SVD best-fit plane; `normal=` overrides both
the fit *and* the planarity check, `origin=` defaults to the centroid — which an O-grid
disc's is *not* its centre, so pass it), and `sweep_placements`, which composes them and
pins the frame field's one free parameter (a constant roll about the tangent) so station
0 lands the section exactly as authored. See `examples/serpentine_pipe.py`.

The quad rung assembles **layer by layer** (append profile `i`'s own lines +
`interior` verbatim, then the rung lines joining level `i` to level `i-1`; a quad is
then just four line indices + four `flip` bits — no `unique_edges` dedupe, no
owner-wins reconciliation, because no shared edge is ever duplicated). The line rung
is that same pattern one dimension down, which is why `LineMesh.loft` is the sole
author of 1-D connectivity (`loop=False` chain / `loop=True` ring) — the container
itself generates none. `HexMesh.loft` instead builds the
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

Every factory that **authors points** takes an optional `order=N` (default `1`) —
`LineMesh.line`/`arc`/`circle`/`rectangle`/`loft`/`loft_curve`, `Quad/HexMesh.from_grid`,
`QuadMesh.box`/`half_box`/`sphere`/`hemisphere`/`rectangle`. A factory that **consumes a
finished mesh** has no `order=` at all and inherits its inputs' — `extrude`, `sweep`,
`loft`, `merge`, `blend`, `ogrid`/`half_ogrid`/`quadrant_ogrid`/`spined_ogrid`, `structured`, both
`annulus` — which is why they reject a mismatched order across their inputs rather than
elevate. The quad/hex `loft_curve` sit between the two and take `order: int | None = None`
= "the sections' own", the line rung's `order: int = 1` being genuinely constructive
(it has no mesh to read it off). At `order > 1` each element
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
  `LineMesh.loft_curve` (the same trick for an arbitrary analytic parametrization: `_refined_lattice`
  builds the `n*order+1` **parameter values** of **every** node of the chain — element `i`'s node
  `a` at `fr[i] + g[a]*(fr[i+1] - fr[i])` for the caller's `fractions` `fr` (the parameter values
  themselves, in `f`'s own units) and the GLL nodes `g`
  — and `f` is called **once** on that whole
  lattice, so the corners and each element's private `interior` are all true-curve points, and
  the grading rides *into* the lattice instead of being assumed uniform. This
  is the escape hatch from the straight-subdivision bullet below for any curve with a closed
  form; `LineMesh.arclength_fractions` perturbs only *where along* the curve the nodes sit, never
  whether they are on it),
  `QuadMesh.loft_curve` (the same trick one rung up, along the **sweep**: the profiles are
  evaluated at every node level of `_refined_lattice(fractions, order)`, not just the corner
  levels, so a swept curved surface is exact instead of straight between slices — see the
  sweep-primitive section above; `QuadMesh.loft(..., sweep_nodes=…)` is the same escape with
  the intermediate profiles handed in rather than evaluated),
  `HexMesh.loft_curve` / `HexMesh.loft(..., sweep_nodes=…)` (the identical pair at the hex
  rung, sections instead of profiles),
  `QuadMesh.sweep`/`HexMesh.sweep` (one profile carried along a curved path by a moving
  frame — every station is a rigid placement of the *authored* section, and the
  intermediate stations go down the same `sweep_nodes` channel, so a bent tube is exact at
  every order both around the section and along the bend),
  `QuadMesh.sphere`/`QuadMesh.hemisphere` (radial projection applied **entity-wise** to
  the cube's / half box's whole B-rep — `points`, `lines.interior`, `interior` — never to
  a reassembled block, so a shared edge lands identically from either quad).
  Straight-sided analytic shapes are exact trivially: `LineMesh.line`,
  `LineMesh.rectangle`, `QuadMesh.box`/`half_box` (planar patches).
- **straight GLL subdivision**: anything built from an explicit point array —
  `LineMesh.loft` (each line's interior = the straight blend of its two
  endpoints), `QuadMesh.from_grid`/`HexMesh.from_grid`. Sampling a curve into points and
  calling `LineMesh.loft` therefore *loses* the curve at `order > 1`; hand in the
  analytic `arc`/`circle` instead, or `LineMesh.loft_curve` when the closed form is neither
  (only a genuinely form-less curve — a scanned polyline — is stuck with the chord).
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
  - `ogrid`/`half_ogrid`/`quadrant_ogrid` have no global analytic map, so they keep the linear
    construction and generalize the `Overlay` `(quad ids, local side, curve)` channel:
    **one overlay pair per O-ring**, not just the wall. Each intermediate ring's curve is
    `LineMesh.blend(block perimeter, wall, t)` — the same mechanism `annulus` uses — so a
    ring at `t` inherits its share of the wall's bow. Both incident copies of a shared
    ring must be stamped (ring `m` is block `m−1`'s outer side *and* block `m`'s inner
    side); stamping one leaves the other straight and `scatter_edge_nodes` rejects the
    mesh. Each element is then curved tangentially and straight radially — exactly
    `annulus`'s behaviour, which is right for a radial blend. `quadrant_ogrid` stamps
    the same channel along **both** of its seams, from each seam's own nodes.
    `half_ogrid` stamps the
    same channel a second time along its **seam**, one overlay per spine point interval
    (see `spined_ogrid` above), so a curved spine bows the flat side too.
  - Underneath both, `_elevate` derives each quad's private `interior` as the
    **transfinite (Coons) patch of that element's own four edge curves**, evaluated
    *after* the overlays (`_coons_at`), instead of a bilinear fill from its corners. A
    curved side therefore bows the interior with it; with four straight edges the patch
    is algebraically that bilinear fill (it differs only in float association, ~1e-16).
- **carried through**: `extrude` translates a section's whole B-rep rigidly, and `sweep`
  *rotates and* translates it rigidly at each station (the affine `_morph` ops map every
  coordinate table, so the placed section keeps its exact shape); `blend`
  lerps the entity tables with the same `t` the corners get; `loft` sweeps each column
  as a Coons patch curved along the profile (from the slices' own nodes, `_coons_at` /
  `_slice_block` / `_sweep_at`) and straight along the sweep (use `loft_curve`/`sweep`, or
  `sweep_nodes=`, when the sweep path itself is curved); `QuadMesh.annulus` is a
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
**`to_fld` is the high-order Nek export**: the field-file format
(`<prefix>0.f00001`) *does* store a full `lx1*ly1*lz1` GLL block per element, so it is
the only Nek writer that preserves an `order = N` geometry. The per-element node
ordering it wants is lexicographic `i` fastest — exactly what the `conform.conformal_*`
walk produces — so the block goes out **with no permutation**; only the corners would
survive a wrong one, which is why a permutation bug there is invisible at order 1.
Layout: a 132-byte `#std` header
(`"#std %1d %2d %2d %2d %10d %10d %20.13E %9d %6d %6d %s\n"`, space-padded), the
`6.54321` float32 endian tag, an `int32` 1-based element map, then per element all `x`,
all `y`, all `z`, then the 3-D trailing metadata block (per element, per component,
`min` then `max`, **always** float32 whatever `wdsz` is). Only `fields="X"` is written —
a mesh carries geometry and nothing else — and `wdsz` picks float64/float32 coordinates.
The `.vtu` (XML VTK) writer becomes high-order at `order > 1`, emitting VTK Lagrange
cells (`VTK_LAGRANGE_CURVE=68` / `_QUADRILATERAL=70` / `_HEXAHEDRON=72`) whose `(N+1)^d`
nodes/cell index the **conformal (welded)** node array from the `conform.conformal_*`
walk, ordered via a hand-built `_lagrange_*_perm(order)` (corners → edges → faces →
interior, VTK's `PointIndexFromIJK` recursion — no `vtk`/`meshio` dep). Face nodes
inherit the face's `bc_id` via `hex_face_indices`. The writer
(`to_vtu`/`line_to_vtu`/`quad_to_vtu`) builds its node arrays via
`_hex_arrays`/`_line_arrays`/`_quad_arrays` and emits through `_write_vtu`; there is
**no legacy ASCII `.vtk` writer** — only `.re2` and `.vtu`.

**The `.vtu` writer relabels GLL → equispaced on the way out** (`_to_equispaced`, called
by all three `_*_arrays` builders). VTK's Lagrange cells are *defined* on an equispaced
node lattice — there is no GLL cell type (`VTK_BEZIER_*` takes control points, also not
GLL) — so shipping the toolkit's GLL nodes verbatim declares the wrong parametrization
and the reader reconstructs a **different polynomial**: measured on a unit cube at order
3, VTK renders the *identity* map with a 7.4e-2 excursion, one hump per element — the
visible crease at element joints. It is a **change of nodal basis, not a resample**: each
element's polynomial is one object, re-read at `order+1` different parameters per axis
(`lagrange_matrix(gll_nodes(order), uniform_spacing(order))` applied along each axis of
the `(E,(N+1)^d,3)` block), so the geometry survives to float round-off and only the
*labels* move. Shared entities stay consistent because the interpolation is a tensor
product and the two lattices agree at `0.0`/`1.0`: a node on a shared edge/face is
evaluated at the boundary parameter transversely, which selects that entity's own nodes
alone, so both incident elements compute the same value (~1e-16 apart) and the scatter is
well-defined. **At `order <= 2` the two lattices are equal and the function short-circuits**
— which is why the order-2 `high_order_*.vtu` goldens stayed byte-identical when this
landed, and why the artifact only ever appeared from order 3 on. Measured on
`LineMesh.circle(1.0, 8, order=N)` round-tripped through `vtkXMLUnstructuredGridReader`,
max `|r-1|`: order 3 `1.23e-2 → 1.97e-4`, order 4 `7.95e-4 → 1.46e-6`.

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
the plain point blend against empty tables — so a mesh built at the library default
`order=1` is bit-identical to the pre-high-order toolkit. (`examples/bifurcation.py`
itself now ships at `ORDER = 3` with both smoothers off, so its goldens are a high-order `.vtu` and the
unchanged linear `.re2`; they are still frozen — coordinates to `1e-12`, everything
discrete byte-for-byte — and a diff from a refactor is still a bug.) See `examples/high_order_{curve,quad,hex}.py`.

### Physical groups & export

`PhysicalGroups` maps name ↔ tag ↔ Nek BC code; pass `groups=` to the factories to control
`.re2` boundary codes without touching the exporter (`PhysicalGroups.duct()`,
`.from_tags()`, `.nek_default()` are presets). `.re2` element ids are 1-based on write;
all internal indices are 0-based. `to_re2` writes **only** the binary `.re2` — there is
no `.rea` writer and no `io/templates/` — and, like `to_vtu`, it takes the **full output
filename including the extension** (`to_re2(mesh, "pipe.re2")`); nothing is appended.

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
  it. `SmoothingMethod` lives here too — a `Literal` of the built-in `SECTION_METHODS` names with a
  bare `str` arm kept alongside it, so mypy and an IDE surface the built-ins without the annotation
  closing the **open** registry to a third-party `@register_section_smoothing`; it sits in
  `_typing.py` rather than beside the registry because every rung that forwards a
  `smoothing_method=` down to it (the region fills, both `annulus`) needs the same spelling. Real
  data that is **not** a position keeps `FloatArray`: `fractions`/`t` blend parameters,
  `layers`/`radial` positions, `x_frac`/`y_frac` grading, GLL nodes/weights and Lagrange
  (derivative) matrices, `tensor_nodes`' `(M,dim)` *parametric* reference lattice, scaled-Jacobian
  values and quality metrics, tolerances. numpy has no static shape checking, so they document
  intent only and are interchangeable with `FloatArray` to mypy.
- Full architecture, module reference, and extension points: `README.md`.
