# Conventions

## Indexing & coordinates

- Indices are **0-based** internally (input `.tri` is 1-based, converted on
  load; `.re2` element ids are written 1-based).
- No `Point` class — a single point is a `(3,)` array; every container stores
  a `(P,3)` `points` array, mutable in place (`mesh.points[:] = X`).
- `LineMesh` requires 3-D input (a `(N,2)` array is rejected, not padded) and a
  **required** `lines` connectivity argument — nothing implies a default
  chain or wrap. {func}`linemesh.loft <nekmeshpy.linemesh.assemble.loft>`
  authors one explicitly via `loop=False`/`loop=True`.

## HexMesh immutability & storage

`HexMesh` is immutable by construction: build it with a factory (`extrude`/
`loft`/`annulus`/`merge`/`from_grid`) or the array constructor. `extrude`/
`loft` are shared-point (index arithmetic, no weld); `merge` is the one place
seam points are coordinate-welded. `loft` is the same primitive at all three
rungs, each with `loop: bool = False` closing the sweep — see {doc}`concepts`.

Stored state is the B-rep: a `quad_mesh` (`QuadMesh` of the shared faces),
`hexes` `(E,6)` face incidence, `orient` `(E,6)` D4 codes, `interior`
`(E,(order-1)**3,3)`. `points` and `corners` `(E,8)` are **derived read-only
views** — corner consistency is structural, not maintained (see
{doc}`concepts` § high-order elements). `element_tags` names only the hexes
carrying a region tag. `face_tags` reads through to `quad_mesh.element_tags`
— a face is named once, by id, as the one object both its hexes reference —
and is **not** "the boundary": it's the named subset of faces, while
`boundary_faces()` derives the topological boundary from connectivity;
`tag_report()` counts where the two disagree. A region name (`"fluid"`,
`"solid"`) belongs only to the top rung's `element_tags` — one rung down, an
element's `element_tags` is the boundary name it becomes once lifted (not
enforced, but `first_tag`/`last_tag` default to the bounding slice's own tag,
so a `"fluid"`-tagged section exports `"fluid"` caps). `retag_element`/
`retag_face` rename one vocabulary without touching the other.

`QuadMesh` mirrors this one rung down (`line_mesh` + `quads`/`orient`/
`interior`, `edge_tags` reading through to `line_mesh.element_tags`);
`LineMesh` does the same onto a `PointMesh`, the ladder's bottom rung, holding
only coordinates and their tags. Topology is fixed; coordinates aren't.

## Strong typing

Enforced — `mypy` runs `disallow_untyped_defs`, `check_untyped_defs`,
`disallow_any_generics`. Everything in `nekmeshpy/` is annotated.

- Geometry parameters take the concrete type (`LineMesh`, etc.), no
  `| np.ndarray` fallback. Open-vs-closed isn't a type or stored flag — read
  it off `lines`. Vector *literals* (axis/origin/center) use
  `Sequence[float] | FloatArray`.
- Use the dtype aliases in `nekmeshpy._typing` — `FloatArray`, `IntArray`,
  `BoolArray`, `StrArray` — never a bare `np.ndarray` (`disallow_any_generics`
  rejects it).
- `Point`/`Vec3`/`PointArray` are shape-*documentation* aliases of
  `FloatArray` (single `(3,)` location/direction, or any array whose trailing
  axis is the 3 spatial components) — the concrete shape belongs in the
  docstring. Non-position float data (fractions, grading, GLL nodes, quality
  metrics, tolerances) stays plain `FloatArray`.
The package ships a `py.typed` marker.

## Logging

Progress goes through the `nekmeshpy` logger; configure it in your script
(e.g. `logging.basicConfig(level=logging.INFO, format="%(message)s")`).
