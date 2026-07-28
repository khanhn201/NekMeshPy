# Conventions

## Indexing & coordinates

- Triangle/vertex indices are **0-based** internally (input `.tri` is 1-based,
  converted on load; `.re2` element ids are written 1-based).
- Coordinates are plain NumPy arrays — **no `Point` class**; a single point is a
  `(3,)` array. `LineMesh` holds a `(N,3)` `points` array + `(L,2)` `lines` (**3-D
  input required — a `(N,2)` array is rejected, not padded**); the mesh containers
  (`TriMesh` / `QuadMesh` / `HexMesh` / `Mesh`) store a `(P,3)` `points` array
  (mutate with `mesh.points[:] = X`).

## HexMesh immutability & storage

`HexMesh` is **immutable by construction**: build it with a factory (`extrude` /
`loft` / `annulus` / `merge` / `from_grid`) or the array constructor. `extrude` /
`loft` are shared-point (conformal slices → index arithmetic); `merge` is the one
place seam points are coordinate-welded.

It stores `points` + `hexes` `(N,8)` connectivity in Nek point order; `boundaries`
`(Nbc,2)` = `[element id, face (1–6)]` with parallel `boundary_tags`, plus a dense
`element_tags` `(N,)` inherited from the swept quad. `QuadMesh` stores boundaries
the same way one dimension down — `(Nbc,2)` = `[quad id, side (1–4)]` (side `s` =
edge `EDGE_POINTS[s-1]`). `weld()` returns `(points, hexes, n_points)`; exporters
expand via `points[hexes]`. Coordinates may be repositioned in place (smoothing);
topology is fixed.

## Strong typing

Enforced — `mypy` runs with `disallow_untyped_defs`, `check_untyped_defs`, and
`disallow_any_generics`. Everything in `nekmeshpy/` is annotated.

- Geometry-object parameters take the concrete type (`LineMesh`, etc.), no
  `| np.ndarray` fallback; open-vs-closed is checked at runtime. Only vector
  *literals* (axis / origin / center) use `Sequence[float] | FloatArray`.
- Numeric internals use the dtype aliases in `nekmeshpy._typing` — `FloatArray`
  (`NDArray[np.float64]`), `IntArray` (`NDArray[np.int64]`), `BoolArray` (masks),
  `StrArray` (`NDArray[np.str_]`, tags) — **never a bare `np.ndarray`**, which
  `disallow_any_generics` rejects (use an explicit `NDArray[...]` for other dtypes).
- `Point` / `Vec3` / `PointArray` are **shape-documentation aliases** of
  `FloatArray` for a `(3,)` location / `(3,)` direction / `(P,3)` array. numpy has
  no static shape checking, so they are interchangeable with `FloatArray` to mypy.

The package ships a `py.typed` marker.

## Logging

Progress goes through the `nekmeshpy` logger; configure it in your script (e.g.
`logging.basicConfig(level=logging.INFO, format="%(message)s")`).
