"""Variable-arity ``HexMesh`` operations -- the only ones that build a numbering."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .._typing import (
    FloatArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..linemesh import LineMesh
from ..linemesh.assemble import _check_fraction_count, _refined_lattice, _weld
from ..model import conform
from ..model.conform import entity_tol
from ..model.fields import gll_nodes, reject_loop_caps
from ..model.tags import (
    ElementTags,
    FaceTags,
    TagBuilder,
)
from ..quadmesh import NO_TAG, QuadMesh
from .hexmesh import HexMesh, _slice_block, _sweep_at
from .query import _boundary_points


def _face_brep(points: PointArray, canonical_conn: IntArray,
               edge_nodes: PointArray | None, face_nodes: PointArray | None,
               order: int) -> QuadMesh:
    """The shared-face ``QuadMesh`` of a hex block, from its canonical face table."""
    q_edges, q_elem_edges, q_flip = conform.unique_edges(canonical_conn, 2)
    edge_lm = LineMesh(points, q_edges, interior=edge_nodes)
    return QuadMesh(edge_lm, q_elem_edges, q_flip, face_nodes)


def _cap_tags(cap: str | Sequence[str] | StrArray | None, M: int) -> list[str]:
    """Normalize a cap tag to one tag per section quad (length ``M``): a scalar
    ``str`` tags the whole cap, an array-like is per-quad, and ``None``
    -- "no cap tag asked for" -- is ``NO_TAG`` on every quad."""
    if cap is None:
        return [NO_TAG] * M
    if isinstance(cap, str):
        return [cap] * M
    arr = np.asarray(cap, dtype=np.str_).reshape(-1)
    if arr.shape[0] != M:
        raise ValueError("cap tags length (%d) must match section quads (%d)"
                         % (arr.shape[0], M))
    return [str(x) for x in arr.tolist()]


def loft(
    slices: Sequence[QuadMesh],
    *,
    loop: bool = False,
    sweep_nodes: Sequence[Sequence[QuadMesh]] | None = None,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray | None = None,
    last_tag: str | Sequence[str] | StrArray | None = None,
) -> HexMesh:
    """Loft a stack of conformal quad profiles into a hex block (the general primitive
    behind ``extrude``, and the top rung of the uniform sweep shared with
    :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>` and
    :func:`QuadMesh.loft <nekmeshpy.quadmesh.assemble.loft>`)."""
    slices = list(slices)
    if loop:
        reject_loop_caps("HexMesh.loft", first_tag, last_tag)
    quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
    # section (quad, side) -> name; each swept side face inherits its section edge
    side_name: dict[tuple[int, int], str] = slices[0].edge_tags.as_dict()
    tag_sides = bool(slices[0].edge_tags)
    M = quads.shape[0]
    n_prof = len(slices)
    # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
    nz = n_prof if loop else n_prof - 1
    # ``nxt[i]`` is the profile layer ``i`` sweeps *to*: i+1 normally, wrapping to
    # 0 for the closing layer of a periodic sweep.
    nxt: IntArray = (np.arange(1, nz + 1, dtype=np.int64) % n_prof if nz
                     else np.zeros(0, dtype=np.int64))
    S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                  for s in slices], axis=0)             # (n_prof, nn, 3)
    nn = S.shape[1]
    points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v

    # Decide handedness once from the first layer and flip the quad template if
    # left-handed; reject a mixed-winding section rather than invert elements.
    signs = np.array([HexMesh._signed_vol(np.vstack([S[0, quads[q], :], S[1, quads[q], :]]))
                      for q in range(M)]) if nz else np.zeros(0)
    if nz and not (np.all(signs > 0) or np.all(signs < 0)):
        raise ValueError(
            "loft: layer 0 is not consistently wound (mixed hex orientation). "
            "Either the section mesher emitted mixed winding, or a sweep folded the "
            "section through its own path -- a bend tighter than the section is wide "
            "turns the inboard elements inside out.")
    flip = bool(nz and signs[0] < 0)
    qw = quads[:, [0, 3, 2, 1]] if flip else quads

    # caps stay faces 5/6 by q (the flip only reorders a quad's 4 corners); a
    # periodic sweep has no cap at all, so it emits none (and rejected the tags).
    first_caps = ([NO_TAG] * M if loop else _cap_tags(first_tag, M))
    last_caps = ([NO_TAG] * M if loop else _cap_tags(last_tag, M))

    hexes = np.empty((nz * M, 8), dtype=np.int64)
    # every layer repeats the section's per-quad tags (hex ``e = i*M + q``)
    etags: ElementTags = slices[0].element_tags.repeat_blocks(nz, M)
    if element_tags is not None:
        # the orthogonal (per-layer) tag axis: a non-empty layer tag overrides the
        # section's per-quad tag on every hex of that layer.
        layer_tags: StrArray = np.asarray(
            element_tags, dtype=np.str_).reshape(-1)
        if layer_tags.shape[0] != nz:
            raise ValueError(
                "loft: element_tags is per layer, so it needs %d entries, got %d"
                % (nz, layer_tags.shape[0]))
        etags = etags.overlay(ElementTags.blocks(layer_tags, M))
    bb = TagBuilder(FaceTags)
    e = 0
    for i in range(nz):
        j = int(nxt[i])                     # the profile this layer sweeps to
        for q in range(M):
            v = qw[q, :]
            hexes[e] = np.concatenate([i * nn + v, j * nn + v])
            if tag_sides:
                # section side s -> hex face s, or 5-s when the quad was flipped
                for s in (1, 2, 3, 4):
                    nm = side_name.get((q, s))
                    if nm is None or nm == NO_TAG:
                        continue
                    bb.add(e, (5 - s) if flip else s, nm)
            if i == 0 and first_caps[q]:
                bb.add(e, 5, first_caps[q])
            if i == nz - 1 and last_caps[q]:
                bb.add(e, 6, last_caps[q])
            e += 1

    # order-N: each hex column is a straight GLL sweep between the two bounding
    # profiles' in-plane blocks, evaluated only at the entity slots -- the two
    # z-face interiors come from the slices' own quad interiors, the four side-face
    # interiors and the cell interior are swept from the slices' edge / quad nodes,
    # the in-slice edges are the slices' shared edge nodes and the vertical edges
    # straight corner blends.  ``scatter_*`` then canonicalizes.
    order = slices[0].order
    if any(s.order != order for s in slices):
        raise ValueError("loft: all slices must share the same order")

    # the intermediate sections, if any -- one list of ``order-1`` per layer, in
    # ascending GLL-level order.  Validated here so a mis-sized stack names the
    # layer rather than failing later inside a fancy-index.
    sw: list[list[QuadMesh]] | None = None
    if sweep_nodes is not None:
        sw = [list(level) for level in sweep_nodes]
        if len(sw) != nz:
            raise ValueError(
                "loft: sweep_nodes must have one entry per layer (%d), got %d"
                % (nz, len(sw)))
        for i, level in enumerate(sw):
            if len(level) != order - 1:
                raise ValueError(
                    "loft: sweep_nodes[%d] must hold order-1 = %d intermediate "
                    "sections, got %d" % (i, order - 1, len(level)))
            for m in level:
                if (m.order != order or m.n_points != nn
                        or not np.array_equal(
                            np.asarray(m.quads, dtype=np.int64).reshape(-1, 4),
                            quads)):
                    raise ValueError(
                        "loft: sweep_nodes[%d] sections must match the slices "
                        "(order %d, %d points, %d quads), got order %d with %d "
                        "points and %d quads"
                        % (i, order, nn, M, m.order, m.n_points, m.n_quads))
        if order == 1:
            sw = None                      # order 1 has no interior level at all

    edges, elem_edges, eflip = conform.unique_edges(hexes, 3)
    canonical_conn, elem_faces, face_orient = conform.canonical_faces(hexes)
    edge_nodes: PointArray | None = None
    face_nodes: PointArray | None = None
    interior: PointArray | None = None
    if order > 1:
        g = gll_nodes(order)
        row = order + 1
        m2 = row * row
        SC = np.stack([_slice_block(s, order) for s in slices], axis=0)
        # (nz, M, m2, 3) in-plane blocks of the bottom/top slice of each layer,
        # flattened to hex order e = i*M + q.  ``nxt`` picks the top slice, so the
        # periodic closing layer sweeps back onto profile 0's own block.
        bottom = SC[np.arange(nz, dtype=np.int64)].reshape(nz * M, m2, 3)
        top = SC[nxt].reshape(nz * M, m2, 3)
        kk = np.arange(m2)
        trans = (kk // row) + row * (kk % row)        # transpose the in-plane grid
        if flip:
            bottom = bottom[:, trans, :]
            top = top[:, trans, :]
        E = nz * M

        if sw is None:
            def _at(slots: IntArray) -> PointArray:
                """The column's straight GLL sweep, evaluated at those hex slots."""
                return _sweep_at(bottom, top, g, slots, m2)
        else:
            # every sweep level is a genuine section, so the full ``(E, row, m2, 3)``
            # level stack is known outright and a node is a pure gather -- nothing is
            # interpolated along the sweep.  The intermediate levels take the *same*
            # in-plane transpose as bottom/top when the section is left-handed, or the
            # column's grid is scrambled against its own corner table.
            lev: PointArray = np.empty((E, row, m2, 3), dtype=float)
            lev[:, 0] = bottom
            lev[:, order] = top
            for k in range(1, order):
                blk: PointArray = np.stack(
                    [_slice_block(sw[i][k - 1], order) for i in range(nz)],
                    axis=0).reshape(E, m2, 3) if nz else np.zeros((0, m2, 3))
                lev[:, k] = blk[:, trans, :] if flip else blk

            def _at(slots: IntArray) -> PointArray:
                """The true node at those hex slots, read out of the level stack."""
                return lev[:, slots // m2, slots % m2, :]

        k2 = (order - 1) ** 2
        eslots = conform._edge_slots(3, order)[:, 1:-1]         # (12, order-1)
        local_e = _at(eslots.ravel()).reshape(E, 12, order - 1, 3)
        fslots = conform._face_interior_slots(order)            # (6, k2)
        local_f = _at(fslots.ravel()).reshape(E, 6, k2, 3)
        interior = _at(conform._interior_slots(3, order))
        tol = conform.entity_tol(points)
        edge_nodes = conform.scatter_edge_nodes(
            local_e, elem_edges, eflip, edges.shape[0], tol, "HexMesh.loft")
        face_nodes = conform.scatter_face_nodes(
            local_f, elem_faces, face_orient, canonical_conn.shape[0], tol,
            "HexMesh.loft")
    faces = _face_brep(points, canonical_conn, edge_nodes, face_nodes, order)
    return HexMesh(faces, elem_faces, face_orient, interior,
                   bb.build_ordered(), etags)


def _loft_evaluated(
    profs: Sequence[QuadMesh],
    t: FloatArray,
    order: int,
    *,
    loop: bool = False,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray | None = None,
    last_tag: str | Sequence[str] | StrArray | None = None,
    name: str = "loft_fn",
) -> HexMesh:
    """The shared tail of every sweep whose sections are **evaluated** on the refined
    node lattice rather than handed in: validate, close the loop, split, delegate."""
    profs = list(profs)
    if len(profs) != t.shape[0]:
        raise ValueError(
            "%s: expected one section per sweep lattice value (%d), got %d"
            % (name, t.shape[0], len(profs)))
    nz = (len(profs) - 1) // order
    ref = profs[0]
    ref_quads = np.asarray(ref.quads, dtype=np.int64).reshape(-1, 4)
    for k, m in enumerate(profs):
        if m.order != order:
            raise ValueError(
                "%s: f(%g) returned an order-%d section, but order=%d was requested"
                % (name, t[k], m.order, order))
        if (m.n_points != ref.n_points
                or not np.array_equal(
                    np.asarray(m.quads, dtype=np.int64).reshape(-1, 4), ref_quads)):
            raise ValueError(
                "%s: every section must be index-paired and conformal with the "
                "first, but f(%g) returned %d points / %d quads against f(%g)'s "
                "%d / %d.  Place one section with the affine ops rather than "
                "rebuilding it per parameter."
                % (name, t[k], m.n_points, m.n_quads, t[0], ref.n_points,
                   ref.n_quads))

    if loop:
        # the wrap level must land back on level 0; drop it, but keep the seam
        # layer's own intermediate levels -- they are what curve the seam.
        P0 = np.asarray(profs[0].points, dtype=float).reshape(-1, 3)
        Pw = np.asarray(profs[-1].points, dtype=float).reshape(-1, 3)
        gap = float(np.max(np.linalg.norm(Pw - P0, axis=1))) if P0.size else 0.0
        tol = entity_tol(P0)
        if gap > tol:
            raise ValueError(
                "%s(loop=True) needs the last fraction to map back to the first "
                "section, but f(%g) and f(%g) are %g apart (tolerance %g).  Pass "
                "the trailing wrap value as the final fraction, or use loop=False."
                % (name, t[-1], t[0], gap, tol))
        profs = profs[:-1]

    slices = profs[::order]
    sweep_nodes = [profs[i * order + 1:(i + 1) * order] for i in range(nz)]
    return loft(slices, loop=loop,
                sweep_nodes=sweep_nodes if order > 1 else None,
                element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag)


def loft_fn(
    f: Callable[[float], QuadMesh],
    fractions: FloatArray,
    *,
    loop: bool = False,
    order: int | None = None,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray | None = None,
    last_tag: str | Sequence[str] | StrArray | None = None,
) -> HexMesh:
    """Loft a block from a **parametrized family of sections** -- :func:`loft
    <nekmeshpy.hexmesh.assemble.loft>` with the slices evaluated rather than handed in,
    so **every** node (the corners *and* the sweep-direction high-order nodes) comes
    from calling ``f`` and nothing is blended along the sweep."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    _check_fraction_count(fr, loop=loop, name="loft_fn")
    if loop:
        reject_loop_caps("HexMesh.loft_fn", first_tag, last_tag)
    if order is None:
        # The node lattice the sections are sampled on is a function of the order, so
        # the order has to be settled before the sweep can start -- and ``f`` is the
        # only thing that knows it.  One throwaway evaluation at the first fraction is
        # the whole cost; the section it returns is discarded and re-evaluated with
        # the rest, so no partial state leaks out of the probe.
        probe = f(float(fr[0]))
        if not isinstance(probe, QuadMesh):
            raise TypeError(
                "loft_fn: f must return a QuadMesh section, but f(%g) returned %s. "
                "Pass order= explicitly only if you also fix f." % (fr[0], type(probe)))
        order = probe.order

    t: FloatArray = _refined_lattice(fr, order)
    profs: list[QuadMesh] = [f(float(v)) for v in t]
    return _loft_evaluated(profs, t, order, loop=loop,
                           element_tags=element_tags,
                           first_tag=first_tag, last_tag=last_tag)


def merge(
    meshes: Sequence[HexMesh],
    *,
    tol: float | None = None,
) -> HexMesh:
    """Stitch several hex blocks into one, coordinate-welding coincident seam points in
    a single pass. ``tol`` is the absolute coincidence distance (default ``1e-7`` x the
    merged bounding-box extent)."""
    meshes = list(meshes)
    pos = [m.points for m in meshes]
    counts = [p.shape[0] for p in pos]
    points, point_id = _weld(pos, [_boundary_points(m.hexes) for m in meshes], tol)

    hex_list: list[IntArray] = []
    bnd_list: list[FaceTags] = []
    etag_list: list[ElementTags] = []
    noff = eoff = 0
    for m, c in zip(meshes, counts):
        hex_list.append(point_id[m.hexes + noff])    # local -> concat -> welded id
        # ids shift by this block's offset; sides stay local to their element
        etag_list.append(m.element_tags.offset(eoff))
        bnd_list.append(m.face_tags.offset(eoff))
        noff += c
        eoff += m.hexes.shape[0]
    hexes = (np.concatenate(hex_list, axis=0) if hex_list
             else np.zeros((0, 8), np.int64))
    etags = ElementTags.concat(etag_list)
    bnd = FaceTags.concat(bnd_list).ordered()
    # order-N: the private per-hex interiors just concatenate, but the shared edge /
    # face tables must be rebuilt against the *merged* topology -- gather each
    # block's nodes into its own element-local order, concatenate in merged element
    # order, then re-scatter.  Those scatters are the conformal-weld guard: two
    # blocks that disagree on a welded shared edge / face raise instead of silently
    # welding.
    order = meshes[0].order if meshes else 1
    if any(mm.order != order for mm in meshes):
        raise ValueError("merge: all blocks must share the same order")
    edges, elem_edges, eflip = conform.unique_edges(hexes, 3)
    canonical_conn, elem_faces, face_orient = conform.canonical_faces(hexes)
    edge_nodes: PointArray | None = None
    face_nodes: PointArray | None = None
    interior: PointArray | None = None
    if order > 1:
        local_e: PointArray = np.concatenate(
            [conform.gather_edge_nodes(mm.quads.lines.interior, mm._elem_edges,
                                       mm._edge_flip)
             for mm in meshes], axis=0)                    # (E,12,order-1,3)
        local_f: PointArray = np.concatenate(
            [conform.gather_face_nodes(mm.quads.interior, mm.hex, mm.face_orient)
             for mm in meshes], axis=0)                    # (E,6,(order-1)**2,3)
        tol = conform.entity_tol(points)
        edge_nodes = conform.scatter_edge_nodes(
            local_e, elem_edges, eflip, edges.shape[0], tol, "HexMesh.merge")
        face_nodes = conform.scatter_face_nodes(
            local_f, elem_faces, face_orient, canonical_conn.shape[0], tol,
            "HexMesh.merge")
        interior = np.concatenate([mm.interior for mm in meshes], axis=0)
    faces = _face_brep(points, canonical_conn, edge_nodes, face_nodes, order)
    return HexMesh(faces, elem_faces, face_orient, interior,
                   bnd, etags)

__all__ = [
    "loft",
    "loft_fn",
    "merge",
]
