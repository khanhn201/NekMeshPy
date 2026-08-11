# Conventions

## Indexing & coordinates

- Triangle/vertex indices are **0-based** internally (input `.tri` is 1-based,
  converted on load; `.re2` element ids are written 1-based).
- Coordinates are plain NumPy arrays — **no `Point` class**; a single point is a
  `(3,)` array. `LineMesh` holds a `(N,3)` `points` array + `(L,2)` `lines` (**3-D
  input required — a `(N,2)` array is rejected, not padded**; `lines` is a
  **required** constructor argument — the container never synthesizes a default
  chain, so nothing in it can imply a wrap. Connectivity is authored one rung up by
  {func}`linemesh.loft <nekmeshpy.linemesh.assemble.loft>`, whose `loop=False` / `loop=True` spell
  the chain and the ring); the mesh containers
  (`TriMesh` / `QuadMesh` / `HexMesh` / `Mesh`) store a `(P,3)` `points` array
  (mutate with `mesh.points[:] = X`).

## HexMesh immutability & storage

`HexMesh` is **immutable by construction**: build it with a factory (`extrude` /
`loft` / `annulus` / `merge` / `from_grid`) or the array constructor. `extrude` /
`loft` are shared-point (conformal slices → index arithmetic); `merge` is the one
place seam points are coordinate-welded. `loft` is the **same primitive at all three
rungs** (`LineMesh` / `QuadMesh` / `HexMesh`), each with a `loop: bool = False` flag
that closes the sweep back onto the first profile — see
[Concepts](../user/concepts.md).

Its stored state is the **B-rep**: a `quads` `QuadMesh` of the shared faces, `hex`
`(E,6)` face incidence, `face_orient` `(E,6)` D4 codes and `interior`
`(E,(order-1)**3,3)`. `points` `(P,3)` and `hexes` `(E,8)` (Nek point order) are
**derived read-only views** over it, so corner consistency is structural rather than
maintained — see [Concepts](../user/concepts.md#high-order-order-n-elements).
Alongside it is `element_tags`, an `ElementTags` naming only the hexes that carry a
region tag. `face_tags` is not a second table: it reads through to `quads.element_tags`,
so a face is named **once, by id**, as the one object both its hexes reference. It is
also *not* "the boundary" — it is the named subset of faces, while `boundary_faces()`
derives the topological domain boundary from connectivity, and `tag_report()` counts
where the two disagree. A **region** name (`"fluid"`, `"solid"`) belongs to the top
rung's `element_tags` only: one rung down an element is a piece of a *surface*, so its
`element_tags` should be the boundary name that face will carry once lifted. Nothing
enforces it, but `first_tag`/`last_tag` default to the bounding slice's own
`element_tags`, so a section tagged `"fluid"` exports its caps as a `"fluid"` boundary
condition. Renaming either vocabulary without touching the other is what
`retag_element` / `retag_face` are for.
`QuadMesh` mirrors this one dimension down (a `lines` `LineMesh` of the shared edges +
`quad`/`flip`/`interior`, its `edge_tags` reading through to `lines.element_tags`);
`LineMesh` does the same onto a `PointMesh`, which is the ladder's bottom rung and
carries nothing but coordinates and their tags. `weld()` returns a `WeldResult` NamedTuple —
`.points` (the **live** array), `.hexes`, `.n_points`; the name is historical, since
a `HexMesh` is already stored shared-point and nothing is welded or copied. Exporters
expand via `points[hexes]`. Coordinates may be repositioned in place — writing
`mesh.points[:] = X` hits the single source of truth and every rung sees it —
but topology is fixed.

## Strong typing

Enforced — `mypy` runs with `disallow_untyped_defs`, `check_untyped_defs`, and
`disallow_any_generics`. Everything in `nekmeshpy/` is annotated.

- Geometry-object parameters take the concrete type (`LineMesh`, etc.), no
  `| np.ndarray` fallback; open-vs-closed is not a type and not a stored flag —
  it is read off the `lines` connectivity. Only vector
  *literals* (axis / origin / center) use `Sequence[float] | FloatArray`.
- Numeric internals use the dtype aliases in `nekmeshpy._typing` — `FloatArray`
  (`NDArray[np.float64]`), `IntArray` (`NDArray[np.int64]`), `BoolArray` (masks),
  `StrArray` (`NDArray[np.str_]`, tags) — **never a bare `np.ndarray`**, which
  `disallow_any_generics` rejects (use an explicit `NDArray[...]` for other dtypes).
- `Point` / `Vec3` / `PointArray` are **shape-documentation aliases** of
  `FloatArray` for a single `(3,)` location / a single `(3,)` direction / **any**
  array of point coordinates whose **trailing axis is the 3 spatial components**
  (any leading shape — `(P,3)`, `(L,order-1,3)`, `(ni+1,nj+1,3)`,
  `(E,6,(order-1)**2,3)`). The concrete shape belongs in the field's or parameter's
  docstring; the alias does not encode it. Real data that is *not* a position stays
  `FloatArray` — blend fractions, layer positions, grading, GLL nodes/weights and
  interpolation matrices, quality metrics, tolerances. numpy has
  no static shape checking, so they are interchangeable with `FloatArray` to mypy.
- `SmoothingMethod` — the literal set every section factory's `smoothing_method=`
  accepts — lives there too, rather than in the region-fill module that happens to
  be its heaviest user.

The package ships a `py.typed` marker.

## Logging

Progress goes through the `nekmeshpy` logger; configure it in your script (e.g.
`logging.basicConfig(level=logging.INFO, format="%(message)s")`).
