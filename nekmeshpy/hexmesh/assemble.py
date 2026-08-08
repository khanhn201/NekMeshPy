"""Variable-arity ``HexMesh`` operations -- the only ones that build a numbering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
)
from ..linemesh import LineMesh
from ..linemesh.assemble import _check_fraction_count, _refined_lattice, _weld
from ..model import conform
from ..model.conform import entity_tol
from ..model.fields import gll_nodes
from ..model.sweep import Sweep
from ..model.tags import (
    ElementTags,
    FaceTags,
    TagBuilder,
    sweep_cap_tags,
    sweep_element_tags,
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


class _SweptBrep(NamedTuple):
    """A swept hex block's B-rep, *written* from the two entity families rather than
    deduplicated out of its corners: the shared edges and faces, each element's incidence
    on them, the faces' own edge incidence one rung down, and -- since a closed form knows
    it -- the one ``(element, local entity)`` each entity's high-order nodes come from."""

    edges: IntArray
    elem_edges: IntArray
    edge_flip: BoolArray
    edge_src: IntArray
    faces: IntArray
    elem_faces: IntArray
    face_orient: IntArray
    face_edges: IntArray
    face_flip: BoolArray
    face_src: IntArray


def _swept_brep(sweep: Sweep, sec: QuadMesh, hexes: IntArray, flip: bool,
                i_idx: IntArray, q_idx: IntArray, j_idx: IntArray) -> _SweptBrep:
    """The B-rep of ``sweep`` applied to section ``sec``, in closed form.

    Every entity is *carried* (a section quad / edge sitting at one level) or *swept* (a
    section edge / point dragged across one layer), and each family is an arithmetic
    block, so nothing here deduplicates or sorts -- ids come out in family order at both
    rungs.  A face takes its canonical frame from the section's own winding (carried) or
    from the section edge dragged across the layer (swept), never from an owner element,
    which is not shift-invariant across levels."""
    nn, nlev, nz = sweep.nn, sweep.n_levels, sweep.n_layers
    quads = np.asarray(sec.quads, dtype=np.int64).reshape(-1, 4)
    M = quads.shape[0]
    # local face / in-plane edge ``k`` of a hex is section side ``fside[k]``, and its
    # local corner ``k`` section corner ``cslot[k]``; reversing the template walks the
    # section's sides backwards from side 3 and swaps its two odd corners.
    fside: IntArray = np.array([3, 2, 1, 0] if flip else [0, 1, 2, 3], dtype=np.int64)
    cslot: IntArray = np.array([0, 3, 2, 1] if flip else [0, 1, 2, 3], dtype=np.int64)
    # an entity only exists where an element carries it: an isolated section point, or a
    # section edge no quad references, would otherwise spawn an unreferenced row.  The
    # first section slot referencing each one is where its nodes will be read from.
    used_p, first_p = np.unique(quads.ravel(), return_index=True)
    used_e, first_e = np.unique(sec.quad.ravel(), return_index=True)
    pq, pc = np.divmod(first_p.astype(np.int64), 4)
    eq, es = np.divmod(first_e.astype(np.int64), 4)
    nu, ne = used_p.shape[0], used_e.shape[0]
    pslot: IntArray = np.full(nn, -1, np.int64)
    pslot[used_p] = np.arange(nu, dtype=np.int64)
    eslot: IntArray = np.full(sec.lines.n_lines, -1, np.int64)
    eslot[used_e] = np.arange(ne, dtype=np.int64)
    ab: IntArray = np.sort(sec.lines.lines[used_e], axis=1)       # (ne,2) min-first
    A, B = ab[:, 0], ab[:, 1]

    lvl: IntArray = np.arange(nlev, dtype=np.int64)
    lay: IntArray = np.arange(nz, dtype=np.int64)
    nxt = sweep.nxt

    rows: IntArray = np.concatenate([
        (ab[None] + (lvl * nn)[:, None, None]).reshape(-1, 2),
        np.stack([sweep.point(np.minimum(lay, nxt)[:, None], used_p[None, :]).ravel(),
                  sweep.point(np.maximum(lay, nxt)[:, None], used_p[None, :]).ravel()],
                 axis=1)], axis=0)

    # local edges 0-3 / 4-7 are the section sides carried to this hex's near / far level,
    # 8-11 its corners swept across the layer.  Every stored row is min-first, so a
    # traversal's flip is just the directed pair read off the corners.
    side: IntArray = eslot[sec.quad][q_idx][:, fside]             # (E,4)
    ends: IntArray = hexes[:, conform._LOCAL_EDGES[3]]            # (E,12,2) directed
    elem_edges: IntArray = np.concatenate([
        sweep.carried(i_idx[:, None], side, ne),
        sweep.carried(j_idx[:, None], side, ne),
        sweep.swept(i_idx[:, None], pslot[hexes[:, :4] % nn], ne, nu)], axis=1)

    faces: IntArray = np.concatenate([
        (quads[None] + (lvl * nn)[:, None, None]).reshape(-1, 4),
        np.stack([sweep.point(lay[:, None], A[None, :]).ravel(),
                  sweep.point(lay[:, None], B[None, :]).ravel(),
                  sweep.point(nxt[:, None], B[None, :]).ravel(),
                  sweep.point(nxt[:, None], A[None, :]).ravel()], axis=1)], axis=0)
    elem_faces: IntArray = np.concatenate([
        sweep.swept(i_idx[:, None], side, M, ne),
        sweep.carried(i_idx, q_idx, M)[:, None],
        sweep.carried(j_idx, q_idx, M)[:, None]], axis=1)

    # the faces one rung down: a carried face walks the section quad's own sides; a swept
    # face walks [carried at i, rung at B, carried at j (backwards), rung at A].
    kk: IntArray = np.arange(ne, dtype=np.int64)
    face_edges: IntArray = np.concatenate([
        sweep.carried(lvl[:, None, None], eslot[sec.quad][None], ne).reshape(-1, 4),
        np.stack([sweep.carried(lay[:, None], kk[None, :], ne).ravel(),
                  sweep.swept(lay[:, None], pslot[B][None, :], ne, nu).ravel(),
                  sweep.carried(nxt[:, None], kk[None, :], ne).ravel(),
                  sweep.swept(lay[:, None], pslot[A][None, :], ne, nu).ravel()],
                 axis=1)], axis=0)
    face_flip: BoolArray = np.concatenate([
        np.tile(quads > quads[:, [1, 2, 3, 0]], (nlev, 1)),
        np.stack([np.zeros(nz * ne, dtype=bool),
                  np.repeat(lay > nxt, ne),
                  np.ones(nz * ne, dtype=bool),
                  np.repeat(lay < nxt, ne)], axis=1)], axis=0)

    # where each entity's nodes come from.  Every level below the last is the *near*
    # level of its own layer (local face 5, in-plane edges 0-3); the open sweep's top
    # level is only ever a *far* level, so it reads off the last layer instead.
    home: IntArray = np.minimum(lvl, nz - 1)
    near: BoolArray = (lvl < nz)[:, None]
    mid: IntArray = np.arange(M, dtype=np.int64)
    edge_src: IntArray = np.stack([
        np.concatenate([(home[:, None] * M + eq[None, :]).ravel(),
                        (lay[:, None] * M + pq[None, :]).ravel()]),
        np.concatenate([np.where(near, fside[es][None, :],
                                 4 + fside[es][None, :]).ravel(),
                        np.broadcast_to(8 + cslot[pc], (nz, nu)).ravel()])],
        axis=1)
    face_src: IntArray = np.stack([
        np.concatenate([(home[:, None] * M + mid[None, :]).ravel(),
                        (lay[:, None] * M + eq[None, :]).ravel()]),
        np.concatenate([np.broadcast_to(np.where(near, 4, 5), (nlev, M)).ravel(),
                        np.broadcast_to(fside[es], (nz, ne)).ravel()])], axis=1)

    return _SweptBrep(rows, elem_edges, ends[:, :, 0] > ends[:, :, 1], edge_src,
                      faces, elem_faces,
                      conform.face_frame_code(hexes[:, conform._LOCAL_FACES],
                                              faces[elem_faces]),
                      face_edges, face_flip, face_src)


def loft(
    slices: Sequence[QuadMesh],
    *,
    loop: bool = False,
    sweep_nodes: Sequence[Sequence[QuadMesh]] | None = None,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> HexMesh:
    """Loft a stack of conformal quad profiles into a hex block (the general primitive
    behind ``extrude``, and the top rung of the uniform sweep shared with
    :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>` and
    :func:`QuadMesh.loft <nekmeshpy.quadmesh.assemble.loft>`)."""
    slices = list(slices)
    quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
    # section (quad, side) -> name; each swept side face inherits its section edge
    side_name: dict[tuple[int, int], str] = slices[0].edge_tags.as_dict()
    M = quads.shape[0]
    n_prof = len(slices)
    # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
    nz = n_prof if loop else n_prof - 1
    S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                  for s in slices], axis=0)             # (n_prof, nn, 3)
    nn = S.shape[1]
    points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v
    # a lone slice bounds no layer, so it carries no entity either -- ``n_levels`` is
    # the count of levels the sweep *uses*, and ``nxt`` is where the loop lives.
    sweep = Sweep(n_prof if nz else 0, nz, nn, loop=loop)
    nxt = sweep.nxt

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

    # caps stay faces 5/6 by q (the flip only reorders a quad's 4 corners).  A cap
    # face *is* a section quad, so with no argument it inherits that quad's own element
    # tag -- except on a closed sweep, whose "caps" are the interior seam.
    closed = ElementTags.empty()
    first_caps = sweep_cap_tags(first_tag, closed if loop else slices[0].element_tags,
                                M, "HexMesh.loft")
    last_caps = sweep_cap_tags(last_tag, closed if loop else slices[-1].element_tags,
                               M, "HexMesh.loft")

    # hex ``e = i*M + q`` is section quad ``q`` dragged across layer ``i``: its 8
    # corners are that quad's 4 at the near level then the same 4 at the far level.
    i_idx, q_idx, j_idx = sweep.elements(M)
    hexes: IntArray = np.concatenate(
        [i_idx[:, None] * nn + qw[q_idx], j_idx[:, None] * nn + qw[q_idx]], axis=1)
    brep = _swept_brep(sweep, slices[0], hexes, flip, i_idx, q_idx, j_idx)
    etags = sweep_element_tags(element_tags, nz, M, "HexMesh.loft")

    # face tags, by column rather than by element: a section side names the same hex
    # face on every layer, and a cap names one layer's worth.
    bb = TagBuilder(FaceTags)
    layer0: IntArray = np.arange(nz, dtype=np.int64) * M
    for (q, s), nm in side_name.items():
        if nm == NO_TAG:
            continue
        hf = (5 - s) if flip else s
        for e in (layer0 + q).tolist():
            bb.add(e, hf, nm)
    for q in range(M):
        if first_caps[q]:
            bb.add(q, 5, first_caps[q])
    if nz:
        for q in range(M):
            if last_caps[q]:
                bb.add((nz - 1) * M + q, 6, last_caps[q])

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

            def _pick(elems: IntArray, slots: IntArray) -> PointArray:
                """``(K,S,3)`` -- the same sweep at per-row ``(element, slot)`` pairs."""
                gg = g[slots // m2][:, :, None]
                m = slots % m2
                return ((1.0 - gg) * bottom[elems[:, None], m]
                        + gg * top[elems[:, None], m])
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

            def _pick(elems: IntArray, slots: IntArray) -> PointArray:
                """``(K,S,3)`` -- the same gather at per-row ``(element, slot)`` pairs."""
                return lev[elems[:, None], slots // m2, slots % m2, :]

        interior = _at(conform._interior_slots(3, order))

        # A shared entity has exactly one source here, named by ``*_src``, so its nodes
        # are read once and written once -- no owner election, and no ``scatter_*``
        # tolerance check, because there is no second copy to disagree with.
        ee, le = brep.edge_src[:, 0], brep.edge_src[:, 1]
        local_e = _pick(ee, conform._edge_slots(3, order)[:, 1:-1][le])
        edge_nodes = np.where(brep.edge_flip[ee, le][:, None, None],
                              local_e[:, ::-1, :], local_e)

        fe, lf = brep.face_src[:, 0], brep.face_src[:, 1]
        local_f = _pick(fe, conform._face_interior_slots(order)[lf])
        _, into_canon = conform._perm_tables(order)
        face_nodes = np.take_along_axis(
            local_f, into_canon[brep.face_orient[fe, lf]][:, :, None], axis=1)
    edge_lm = LineMesh(points, brep.edges, interior=edge_nodes)
    faces = QuadMesh(edge_lm, brep.face_edges, brep.face_flip, face_nodes)
    return HexMesh(faces, brep.elem_faces, brep.face_orient, interior,
                   bb.build_ordered(), etags)


def _loft_evaluated(
    profs: Sequence[QuadMesh],
    t: FloatArray,
    order: int,
    *,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
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
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> HexMesh:
    """Loft a block from a **parametrized family of sections** -- :func:`loft
    <nekmeshpy.hexmesh.assemble.loft>` with the slices evaluated rather than handed in,
    so **every** node (the corners *and* the sweep-direction high-order nodes) comes
    from calling ``f`` and nothing is blended along the sweep."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    _check_fraction_count(fr, loop=loop, name="loft_fn")
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
