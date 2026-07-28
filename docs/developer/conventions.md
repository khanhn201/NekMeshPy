# Conventions

## Indexing & coordinates

- Triangle/vertex indices are **0-based** internally (input `.tri` is 1-based,
  converted on load; `.re2` element ids are written 1-based).
- Coordinates are plain NumPy arrays everywhere — there is **no `Point` class**. A
  single point is just a `(3,)` array (e.g. a row of `.points`). `LineMesh` is the
  1-D mesh sibling: a `(N,3)` `points` array + `(L,2)` `lines` connectivity (**3-D
  input is required — a `(N,2)` array is rejected, not padded to `z=0`**); the mesh
  containers (`TriMesh` / `QuadMesh` / `HexMesh` / `Mesh`) likewise store a `(P,3)`
  array `points` (mutate with `mesh.points[:] = X`).

## HexMesh immutability & storage

`HexMesh` is **immutable by construction** (no incremental building): build it with
a factory (`extrude` / `loft` / `annulus` / `merge` / `from_grid`) or the array
constructor. `extrude` / `loft` are shared-point by construction (conformal slices
→ index arithmetic, no weld); `merge` is the one place coincident seam points are
coordinate-welded (one explicit pass).

It stores `points` + `hexes` `(N,8)` integer connectivity in Nek point order;
`boundaries` is `(Nbc,2)` = `[element id (0-based), face (1–6)]` with a parallel
`boundary_tags` `(Nbc,)`, plus a dense `element_tags` `(N,)` inherited from the
swept quad. `QuadMesh` stores its boundaries the same way one dimension down —
`(Nbc,2)` = `[quad id (0-based), side (1–4)]` (side `s` = edge `EDGE_POINTS[s-1]`).
`weld()` returns `(points, hexes, n_points)`; exporters expand to per-element
coordinates via `points[hexes]`. Coordinates may still be repositioned in place
(smoothing); topology is fixed.

## Strong typing

Strong typing is **enforced** — `mypy` runs with `disallow_untyped_defs`,
`check_untyped_defs`, and `disallow_any_generics`. Everything in `nekmeshpy/` is
annotated.

- Geometry-object parameters take the concrete type (`LineMesh`, etc.) with no
  `| np.ndarray` fallback; open-vs-closed is checked at runtime, not in the type.
  Only genuine vector *literals* (axis / origin / center) use
  `Sequence[float] | FloatArray`.
- Array-valued numeric internals use the dtype aliases in `nekmeshpy._typing` —
  `FloatArray` (`NDArray[np.float64]`, coordinates/real data), `IntArray`
  (`NDArray[np.int64]`, connectivity/indices), `BoolArray` (masks), `StrArray`
  (`NDArray[np.str_]`, tags) — **never a bare `np.ndarray`**, which
  `disallow_any_generics` rejects as an implicit `NDArray[Any]` (use an explicit
  `NDArray[...]` for any other dtype).
- `Point` / `Vec3` / `PointArray` are **shape-documentation aliases** of
  `FloatArray` marking a single `(3,)` location / a `(3,)` direction / a `(P,3)`
  array. numpy has no static shape checking, so they document intent only and are
  interchangeable with `FloatArray` to mypy.

The package ships a `py.typed` marker.

## Logging

Progress goes through the `nekmeshpy` logger; your script configures it (e.g.
`logging.basicConfig(level=logging.INFO, format="%(message)s")`).
