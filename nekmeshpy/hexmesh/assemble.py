"""Variable-arity ``HexMesh`` operations -- the only ones that build a numbering."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..core import conform, stations
from ..core.fields import gll_nodes
from ..core.tags import (
    ElementTags,
    FaceTags,
    TagBuilder,
    element_mask,
    sweep_cap_tags,
    sweep_element_tags,
)
from ..linemesh import LineMesh
from ..pointmesh import PointMesh
from ..quadmesh import QuadMesh
from ..quadmesh.assemble import _subset as quad_subset
from ..quadmesh.query import element_blocks as quad_blocks
from .hexmesh import HexMesh, _sweep_at
from .query import _boundary_points


def _face_brep(points: PointArray, edges: IntArray, elem_edges: IntArray,
               edge_flip: BoolArray, elem_faces: IntArray, face_orient: IntArray,
               n_faces: int, edge_nodes: PointArray | None,
               face_nodes: PointArray | None) -> QuadMesh:
    """The shared-face ``QuadMesh`` of a hex block, read off the hex tables.

    The faces' edges *are* the hex edges -- same set, same table -- so this deduplicates
    nothing: it reads each face's incidence through one owning hex."""
    face_edges, face_flip = conform.face_edges_from_hexes(
        elem_faces, face_orient, elem_edges, edge_flip, n_faces)
    return QuadMesh(LineMesh(points, edges, interior=edge_nodes),
                    face_edges, face_flip, face_nodes)


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
    :func:`QuadMesh.loft <nekmeshpy.quadmesh.assemble.loft>`).

    The body climbs the ladder the result is stored on: the shared edges and their nodes
    (the ``LineMesh``), then the shared faces and theirs (the ``QuadMesh``), then the
    hexes and their private interiors.  Every entity of a sweep is *carried* (a section
    quad / edge sitting at one level) or *swept* (a section edge / point dragged across
    one layer), and both families are arithmetic -- ``level * n_carried + k`` and
    ``n_prof * n_carried + layer * n_swept + k``, two contiguous blocks that cannot
    collide -- so every table below is written, never deduplicated."""
    slices = list(slices)
    n_prof = len(slices)
    # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
    nz = n_prof if loop else n_prof - 1
    # A sweep with no layer builds nothing.  A closed one needs three, and the bound is
    # about identity, not size: entities here are resolved by their corner ids, so two
    # layers -- which span the same pair of levels twice -- would hand two genuinely
    # different rungs the same ids.  Three layers is already a real torus given
    # ``sweep_nodes``; the tight case is not what is being excluded.
    if nz < 1:
        raise ValueError("loft needs at least 2 slices (one layer), got %d" % n_prof)
    if loop and nz < 3:
        raise ValueError(
            "loft(loop=True) needs at least 3 slices, got %d -- one layer sweeps a slice "
            "onto itself so every element's corners repeat, and two put both layers "
            "across the same pair of levels, which gives two distinct rungs the same "
            "corner ids.  An entity here *is* its corners, so neither is representable."
            % n_prof)
    sec = slices[0]
    quads = np.asarray(sec.quads, dtype=np.int64).reshape(-1, 4)
    M = quads.shape[0]
    S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                  for s in slices], axis=0)             # (n_prof, nn, 3)
    nn = S.shape[1]
    points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v
    # ``nxt[i]`` is the level layer ``i`` sweeps *to*: i+1 normally, wrapping to 0 on the
    # closing layer of a periodic sweep -- the one place ``loop`` shows up.
    nxt: IntArray = np.arange(1, nz + 1, dtype=np.int64) % n_prof
    lvl: IntArray = np.arange(n_prof, dtype=np.int64)
    lay: IntArray = np.arange(nz, dtype=np.int64)

    order = sec.order
    if any(s.order != order for s in slices):
        raise ValueError("loft: all slices must share the same order")
    ho = order > 1
    # The corners come from the first slice alone, and every slice's high-order nodes are
    # read by its entity ids, so matching geometry is not enough -- both tables have to be
    # the same table.
    sec_lines = np.asarray(sec.lines.lines, dtype=np.int64).reshape(-1, 2)
    for k, sl in enumerate(slices):
        if not (np.array_equal(np.asarray(sl.quads, dtype=np.int64).reshape(-1, 4), quads)
                and np.array_equal(
                    np.asarray(sl.lines.lines, dtype=np.int64).reshape(-1, 2), sec_lines)):
            raise ValueError(
                "loft: every slice must be index-paired with the first, but slice %d "
                "stores a different quad or shared-edge table.  Place one section with "
                "the affine ops (translate / rotate / transform) rather than rebuilding "
                "it per level, or the sweep reads its nodes off the wrong entities." % k)

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
                            np.asarray(m.quads, dtype=np.int64).reshape(-1, 4), quads)
                        or not np.array_equal(
                            np.asarray(m.lines.lines, dtype=np.int64).reshape(-1, 2),
                            sec_lines)):
                    raise ValueError(
                        "loft: sweep_nodes[%d] sections must match the slices "
                        "(order %d, %d points, %d quads), got order %d with %d "
                        "points and %d quads"
                        % (i, order, nn, M, m.order, m.n_points, m.n_quads))
        if not ho:
            sw = None                      # order 1 has no interior level at all

    # Decide handedness once from the first layer and flip the quad template if
    # left-handed; reject a mixed-winding section rather than invert elements.  Only the
    # *sign* is read, so this is the trilinear Jacobian proxy of every layer-0 hex at
    # once -- the same three edge vectors ``HexMesh._signed_vol`` takes, vectorized.
    P: PointArray = np.concatenate([S[0, quads], S[1, quads]], axis=1)   # (M,8,3)
    r = P[:, [1, 2, 5, 6]].mean(axis=1) - P[:, [0, 3, 4, 7]].mean(axis=1)
    u = P[:, [2, 3, 6, 7]].mean(axis=1) - P[:, [0, 1, 4, 5]].mean(axis=1)
    w = P[:, [4, 5, 6, 7]].mean(axis=1) - P[:, [0, 1, 2, 3]].mean(axis=1)
    signs: FloatArray = np.einsum("ij,ij->i", np.cross(r, u), w)
    if not (np.all(signs > 0) or np.all(signs < 0)):
        raise ValueError(
            "loft: layer 0 is not consistently wound (mixed hex orientation). "
            "Either the section mesher emitted mixed winding, or a sweep folded the "
            "section through its own path -- a bend tighter than the section is wide "
            "turns the inboard elements inside out.")
    flip = bool(signs[0] < 0)
    qw = quads[:, [0, 3, 2, 1]] if flip else quads
    # local face / in-plane edge ``k`` of a hex is section side ``fside[k]``; reversing
    # the template walks the section's sides backwards from side 3.
    fside: IntArray = np.array([3, 2, 1, 0] if flip else [0, 1, 2, 3], dtype=np.int64)

    # hex ``e = i*M + q`` is section quad ``q`` dragged across layer ``i``: its 8
    # corners are that quad's 4 at the near level then the same 4 at the far level.
    i_idx: IntArray = np.repeat(lay, M)
    q_idx: IntArray = np.tile(np.arange(M, dtype=np.int64), nz)
    j_idx: IntArray = nxt[i_idx]
    hexes: IntArray = np.concatenate(
        [i_idx[:, None] * nn + qw[q_idx], j_idx[:, None] * nn + qw[q_idx]], axis=1)

    # An entity only exists where an element carries it: an isolated section point, or a
    # section edge no quad references, would otherwise spawn an unreferenced row.
    used_p: IntArray = np.unique(quads.ravel())
    used_e: IntArray = np.unique(sec.quad.ravel())
    nu, ne = used_p.shape[0], used_e.shape[0]
    pslot: IntArray = np.full(nn, -1, np.int64)
    pslot[used_p] = np.arange(nu, dtype=np.int64)
    eslot: IntArray = np.full(sec.lines.n_lines, -1, np.int64)
    eslot[used_e] = np.arange(ne, dtype=np.int64)
    ab: IntArray = np.sort(sec.lines.lines[used_e], axis=1)       # (ne,2) min-first
    A, B = ab[:, 0], ab[:, 1]
    sec_side: IntArray = eslot[sec.quad][q_idx][:, fside]         # (E,4) used-edge slot
    # a section stores each shared edge in its own direction; the rows above are
    # min-first, so this is where a section's edge nodes have to be read backwards.
    rev_e: BoolArray = sec.lines.lines[used_e][:, 0] > sec.lines.lines[used_e][:, 1]

    def _edge_int(m: QuadMesh) -> PointArray:
        """That section's own shared-edge interiors for the used edges, turned into the
        canonical (min-first) direction the rows above are stored in."""
        e: PointArray = np.asarray(m.lines.interior, dtype=float)[used_e]
        return np.where(rev_e[:, None, None], e[:, ::-1, :], e)

    # -- the column block: the private per-hex interior's only source ------------
    # Order-N only.  Each hex column is a straight GLL sweep between its two bounding
    # slices' in-plane blocks -- or, given ``sweep_nodes``, a pure gather out of the true
    # intermediate sections, with nothing interpolated along the sweep at all.  Only
    # step 6 needs it: every *shared* node below is written from the sections directly.
    if ho:
        g = gll_nodes(order)
        row = order + 1
        m2 = row * row
        E = nz * M
        SC = np.stack([quad_blocks(s) for s in slices], axis=0)
        bottom = SC[lay].reshape(E, m2, 3)
        top = SC[nxt].reshape(E, m2, 3)
        kk = np.arange(m2)
        trans = (kk // row) + row * (kk % row)        # transpose the in-plane grid
        if flip:
            # the reversed corner template transposes the in-plane grid with it, or the
            # column's grid is scrambled against its own corner table
            bottom = bottom[:, trans, :]
            top = top[:, trans, :]

        if sw is None:
            def _at(slots: IntArray) -> PointArray:
                """The column's straight GLL sweep, evaluated at those hex slots."""
                return _sweep_at(bottom, top, g, slots, m2)

        else:
            lev: PointArray = np.empty((E, row, m2, 3), dtype=float)
            lev[:, 0] = bottom
            lev[:, order] = top
            for k in range(1, order):
                blk: PointArray = np.stack(
                    [quad_blocks(sw[i][k - 1]) for i in range(nz)],
                    axis=0).reshape(E, m2, 3)
                lev[:, k] = blk[:, trans, :] if flip else blk

            def _at(slots: IntArray) -> PointArray:
                """The true node at those hex slots, read out of the level stack."""
                return lev[:, slots // m2, slots % m2, :]

    # -- 1. the shared edges: the global LineMesh's topology ---------------------
    # Rows are min-first, so a traversal's flip is the directed pair read off the corners.
    carried_rows: IntArray = stations.at_levels(ab, lvl, nn)         # a section edge at a level
    swept_rows: IntArray = np.stack([                        # a section point, dragged
        stations.at_levels(used_p, np.minimum(lay, nxt), nn),
        stations.at_levels(used_p, np.maximum(lay, nxt), nn)], axis=1)
    edges: IntArray = np.concatenate([carried_rows, swept_rows], axis=0)

    # -- 2. that LineMesh's interior: one write per shared edge ------------------
    # A carried edge *is* that slice's own shared-edge interior, verbatim.  A rung is the
    # straight GLL blend of its two corner points, or -- given ``sweep_nodes`` -- the
    # intermediate sections' own points.  Nothing is read through an element.
    edge_nodes: PointArray | None = None
    carried_e: PointArray = np.zeros((0, ne, max(order - 1, 0), 3))
    if ho:
        carried_e = np.stack([_edge_int(s) for s in slices], axis=0)
        if sw is not None:
            rungs: PointArray = np.stack(
                [np.stack([np.asarray(sw[i][k - 1].points, dtype=float)
                           .reshape(-1, 3)[used_p] for k in range(1, order)], axis=1)
                 for i in range(nz)], axis=0)
        else:
            Sp: PointArray = S[:, used_p, :]            # (n_prof, nu, 3)
            gg = g[1:order][None, None, :, None]
            rungs = ((1.0 - gg) * Sp[lay][:, :, None, :] + gg * Sp[nxt][:, :, None, :])
        # the rung rows above are min-first, so the wrap layer stores its own backwards
        rungs = np.where((lay > nxt)[:, None, None, None], rungs[:, :, ::-1, :], rungs)
        edge_nodes = np.concatenate(
            [carried_e.reshape(-1, order - 1, 3), rungs.reshape(-1, order - 1, 3)], axis=0)
    edge_lm = LineMesh(points, edges, interior=edge_nodes)

    # -- 3. the shared faces: their corners, and their edges in that LineMesh ----
    # Each family fixes its own canonical frame, from the section -- never from an owner
    # element, whose frame is not shift-invariant across levels.
    # carried: the section quad at a level.  Its row is that quad's own CCW corners, so
    # its four sides are the section's own sides, walked in the section's own direction.
    carried_conn: IntArray = stations.at_levels(quads, lvl, nn)
    carried_sides: IntArray = stations.at_levels(eslot[sec.quad], lvl, ne)
    carried_flip: BoolArray = np.tile(quads > quads[:, [1, 2, 3, 0]], (n_prof, 1))
    # swept: the section edge ``(A, B)`` dragged across a layer.  Its row is
    # ``[A_i, B_i, B_j, A_j]``, so its sides walk [carried at i, rung at B, carried at j
    # (backwards), rung at A] -- the two rung sides run with the layer, and the stored
    # rung row is min-first, which is what the wrap layer's flips say.
    k_e: IntArray = np.arange(ne, dtype=np.int64)
    rung0 = n_prof * ne                                   # the first swept edge id
    swept_conn: IntArray = np.stack([
        stations.at_levels(A, lay, nn), stations.at_levels(B, lay, nn),
        stations.at_levels(B, nxt, nn), stations.at_levels(A, nxt, nn)], axis=1)
    swept_sides: IntArray = np.stack([
        stations.at_levels(k_e, lay, ne),
        rung0 + stations.at_levels(pslot[B], lay, nu),
        stations.at_levels(k_e, nxt, ne),
        rung0 + stations.at_levels(pslot[A], lay, nu)], axis=1)
    swept_flip: BoolArray = np.stack([
        np.zeros(nz * ne, dtype=bool), np.repeat(lay > nxt, ne),
        np.ones(nz * ne, dtype=bool), np.repeat(lay < nxt, ne)], axis=1)

    face_conn: IntArray = np.concatenate([carried_conn, swept_conn], axis=0)
    face_edges: IntArray = np.concatenate([carried_sides, swept_sides], axis=0)
    face_flip: BoolArray = np.concatenate([carried_flip, swept_flip], axis=0)

    # -- 4. that QuadMesh's interior: one write per shared face ------------------
    # Written straight into the canonical frame, so no D4 code enters here.  A carried
    # face *is* that slice's own private quad interior -- same frame, because the
    # canonical row is that quad's own CCW corner order.  A swept face is its section
    # edge's own nodes along ``u``, carried across the layer along ``v``.
    face_nodes: PointArray | None = None
    if ho:
        k2 = (order - 1) ** 2
        carried_f: PointArray = np.stack(
            [np.asarray(s.interior, dtype=float) for s in slices], axis=0)
        if sw is not None:
            swept_f: PointArray = np.stack(
                [np.stack([_edge_int(sw[i][v - 1]) for v in range(1, order)], axis=0)
                 for i in range(nz)], axis=0).transpose(0, 2, 1, 3, 4)
        else:
            gv = g[1:order][None, None, :, None, None]
            swept_f = ((1.0 - gv) * carried_e[lay][:, :, None, :, :]
                       + gv * carried_e[nxt][:, :, None, :, :])
        face_nodes = np.concatenate(
            [carried_f.reshape(-1, k2, 3), swept_f.reshape(-1, k2, 3)], axis=0)
    faces = QuadMesh(edge_lm, face_edges, face_flip, face_nodes)

    # -- 5. the hexes, as indices into that QuadMesh ----------------------------
    # Local faces 0-3 are the section's sides swept across the layer, 4 / 5 the section
    # quad carried to the near / far level.  ``face_frame_code`` then fits each hex's
    # local frame onto the canonical row chosen above.
    elem_faces: IntArray = np.concatenate([
        n_prof * M + i_idx[:, None] * ne + sec_side,
        (i_idx * M + q_idx)[:, None],
        (j_idx * M + q_idx)[:, None]], axis=1)
    face_orient = conform.face_frame_code(hexes[:, conform._LOCAL_FACES],
                                          face_conn[elem_faces])

    # -- 6. the private per-hex interior ----------------------------------------
    interior: PointArray | None = None
    if ho:
        interior = _at(conform._interior_slots(3, order))

    # -- tags: by column rather than by element ---------------------------------
    # a section side names the same hex face on every layer, and a cap names one layer's
    # worth.  Caps stay faces 5/6 by q (the flip only reorders a quad's 4 corners); a cap
    # face *is* a section quad, so with no argument it inherits that quad's own element
    # tag -- except on a closed sweep, whose "caps" are the interior seam.
    closed = ElementTags.empty()
    first_caps = sweep_cap_tags(first_tag, closed if loop else sec.element_tags,
                                M, "HexMesh.loft")
    last_caps = sweep_cap_tags(last_tag, closed if loop else slices[-1].element_tags,
                               M, "HexMesh.loft")
    bb = TagBuilder(FaceTags)
    # the section's edge tags now name shared *edges*, so each quad's four sides are
    # read back through its own edge indices -- a section edge between two quads is
    # therefore named from both, which is the point: one edge, one name, seen twice
    side_names: StrArray = sec.edge_tags.dense(sec.lines.n_lines)[
        np.asarray(sec.quad, dtype=np.int64)]
    for q, side0 in zip(*np.nonzero(side_names != "")):
        side = int(side0) + 1
        bb.add_if_tagged(lay * M + int(q), (5 - side) if flip else side,
                         str(side_names[q, side0]))
    cap: IntArray = np.arange(M, dtype=np.int64)
    bb.add_if_tagged(cap, 5, first_caps)
    bb.add_if_tagged((nz - 1) * M + cap, 6, last_caps)
    etags = sweep_element_tags(element_tags, nz, M, "HexMesh.loft")
    return HexMesh(faces, elem_faces, face_orient, interior, bb.build_ordered(), etags)


def _loft_evaluated(
    profs: Sequence[QuadMesh],
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
    slices, sweep_nodes = stations.split_evaluated(
        profs, order, loop=loop, conn=lambda m: np.asarray(m.quads, dtype=np.int64).reshape(-1, 4),
        noun="section", elems="quads", name=name)
    return loft(slices, loop=loop,
                sweep_nodes=sweep_nodes if order > 1 else None,
                element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag)


def loft_spline(
    slices: Sequence[QuadMesh],
    *,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> HexMesh:
    """:func:`loft <nekmeshpy.hexmesh.assemble.loft>` with the sweep-direction nodes read
    off a **cubic spline through the whole stack** of sections, rather than blended
    straight between the two bounding a layer.

    Same arguments, same numbering, same tags, and the sections given come back verbatim
    as the levels -- the spline interpolates them, so this adds curvature between sections
    without moving any.  It is the automatic form of ``loft(..., sweep_nodes=...)``: where
    that asks the caller for the intermediate sections, this fits them.  All three node
    blocks a section stores are fitted -- its shared corners, its shared edges' interiors
    and its faces' own -- so the block is curved along the sweep at every node, not only
    on the skeleton.

    Reach for it when a sweep has a feature that turns sharply across a handful of
    sections: ``loft`` cuts the corner with a chord however high the order, and refining
    the order alone will not fix that -- the nodes it adds land on the same chord."""
    prof = list(slices)
    order = prof[0].order if prof else 1
    nz = len(prof) if loop else len(prof) - 1
    if order < 2 or nz < 1:
        return loft(prof, loop=loop, element_tags=element_tags,
                    first_tag=first_tag, last_tag=last_tag)
    fr: FloatArray = np.arange(nz + 1, dtype=float)
    t: FloatArray = stations.refined_lattice(fr, order)

    ref = prof[0]
    edges: IntArray = np.asarray(ref.lines.lines, dtype=np.int64).reshape(-1, 2)
    quads: IntArray = np.asarray(ref.quads, dtype=np.int64).reshape(-1, 4)
    # checked before the stack, so a mismatch names the section rather than failing
    # inside numpy with a shape it cannot explain
    for k, m in enumerate(prof):
        if (m.order != order or m.n_points != ref.n_points
                or not np.array_equal(
                    np.asarray(m.quads, dtype=np.int64).reshape(-1, 4), quads)
                or not np.array_equal(
                    np.asarray(m.lines.lines, dtype=np.int64).reshape(-1, 2), edges)):
            raise ValueError(
                "loft_spline: every section must be index-paired with the first, but "
                "section %d stores a different order / point count / quad or shared-edge "
                "table.  Place one section with the affine ops (translate / rotate / "
                "transform) rather than rebuilding it per level." % k)
    P: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.lines.points, dtype=float).reshape(-1, 3)
                  for s in prof]), t, loop=loop)
    E: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.lines.interior, dtype=float) for s in prof]),
        t, loop=loop)
    F: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.interior, dtype=float) for s in prof]), t, loop=loop)
    fitted = [QuadMesh(LineMesh(PointMesh(P[k], ref.lines.point_tags), edges, E[k],
                                ref.lines.element_tags),
                       ref.quad, ref.flip, F[k], ref.element_tags)
              for k in range(t.shape[0])]
    return _loft_evaluated(fitted, order, loop=loop, element_tags=element_tags,
                           first_tag=first_tag, last_tag=last_tag, name="loft_spline")


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
    stations.check_fraction_count(fr, loop=loop, name="loft_fn")
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

    t: FloatArray = stations.refined_lattice(fr, order)
    profs: list[QuadMesh] = [f(float(v)) for v in t]
    return _loft_evaluated(profs, order, loop=loop,
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
    points, point_id = conform.weld_points(pos, [_boundary_points(m.hexes) for m in meshes], tol)

    # Each block's own B-rep is already correct and already unique; the weld can only
    # ever join entities whose corners are *all* welded, so the tables are concatenated
    # and only that subset is fused.  Nothing is re-derived from the corners.
    hex_list: list[IntArray] = []
    erow_list: list[IntArray] = []
    frow_list: list[IntArray] = []
    ee_list: list[IntArray] = []
    eflip_list: list[BoolArray] = []
    ef_list: list[IntArray] = []
    forient_list: list[IntArray] = []
    bnd_list: list[FaceTags] = []
    etag_list: list[ElementTags] = []
    noff = eoff = foff = elem_off = 0
    for m, c in zip(meshes, counts):
        hex_list.append(point_id[m.hexes + noff])    # local -> concat -> welded id
        erow_list.append(point_id[m.edges + noff])
        frow_list.append(point_id[np.asarray(m.quads.quads, dtype=np.int64) + noff])
        ee_list.append(m._elem_edges + eoff)
        eflip_list.append(m._edge_flip)
        ef_list.append(m.hex + foff)
        forient_list.append(m.face_orient)
        # ids shift by this block's offset; sides stay local to their element
        etag_list.append(m.element_tags.offset(elem_off))
        bnd_list.append(m.face_tags.offset(elem_off))
        noff += c
        eoff += m.edges.shape[0]
        foff += m.quads.n_quads
        elem_off += m.hexes.shape[0]
    hexes = (np.concatenate(hex_list, axis=0) if hex_list
             else np.zeros((0, 8), np.int64))
    etags = ElementTags.concat(etag_list)
    bnd = FaceTags.concat(bnd_list).ordered()

    order = meshes[0].order if meshes else 1
    if any(mm.order != order for mm in meshes):
        raise ValueError("merge: all blocks must share the same order")

    e_rows = (np.concatenate(erow_list, axis=0) if erow_list
              else np.zeros((0, 2), np.int64))
    f_rows = (np.concatenate(frow_list, axis=0) if frow_list
              else np.zeros((0, 4), np.int64))
    elem_edges = (np.concatenate(ee_list, axis=0) if ee_list
                  else np.zeros((0, 12), np.int64))
    eflip = (np.concatenate(eflip_list, axis=0) if eflip_list
             else np.zeros((0, 12), bool))
    elem_faces = (np.concatenate(ef_list, axis=0) if ef_list
                  else np.zeros((0, 6), np.int64))

    # welding renumbers points, so a stored edge row can come out the wrong way round;
    # put it back min-first and toggle the traversals that referenced it, *before*
    # fusing, so a fused pair is then two identical rows and no direction survives it.
    swap: BoolArray = e_rows[:, 0] > e_rows[:, 1]
    e_rows = np.where(swap[:, None], e_rows[:, ::-1], e_rows)
    eflip = eflip ^ swap[elem_edges]

    welded: BoolArray = (np.bincount(point_id, minlength=points.shape[0]) > 1
                         if point_id.size else np.zeros(points.shape[0], bool))
    e_new, e_keep = conform.fuse_entities(e_rows, welded)
    f_new, f_keep = conform.fuse_entities(f_rows, welded)
    edges, canonical_conn = e_rows[e_keep], f_rows[f_keep]
    elem_edges = e_new[elem_edges]
    old_elem_faces, elem_faces = elem_faces, f_new[elem_faces]
    # A fused face keeps its survivor's row, which may be a different frame from the one
    # the losing block stored -- but only there.  Every other block's codes still describe
    # their own row, so the refit is the seam, not the volume.
    face_orient = (np.concatenate(forient_list, axis=0) if forient_list
                   else np.zeros((0, 6), np.int64))
    reframed: BoolArray = np.any(
        ~np.all(f_rows == canonical_conn[f_new], axis=1)[old_elem_faces], axis=1)
    if reframed.any():
        e_i = np.flatnonzero(reframed)
        face_orient[e_i] = conform.face_frame_code(
            hexes[e_i][:, conform._LOCAL_FACES], canonical_conn[elem_faces[e_i]])

    # order-N: the private per-hex interiors just concatenate, but the shared edge /
    # face nodes must be re-pinned against the *merged* tables -- gather each block's
    # into element-local order, concatenate in merged element order, then re-scatter.
    # Those scatters are the conformal-weld guard: two blocks that disagree on a welded
    # shared edge / face raise instead of silently welding.
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
    faces = _face_brep(points, edges, elem_edges, eflip, elem_faces, face_orient,
                       canonical_conn.shape[0], edge_nodes, face_nodes)
    return HexMesh(faces, elem_faces, face_orient, interior, bnd, etags)

def _subset(mesh: HexMesh, keep: BoolArray) -> tuple[HexMesh, IntArray]:
    """``(the kept hexes as a HexMesh, new_hex_of)`` -- the top rung of
    :func:`quadmesh._subset <nekmeshpy.quadmesh.assemble._subset>`, which it calls to
    carry the shared faces down (and which calls the line rung under that).

    Shared faces no kept hex references are dropped, with the edges and points under
    them.  The ``face_orient`` codes survive untouched: they describe each hex's frame
    against its own canonical face row, and dropping a *neighbour* changes neither."""
    kept, new_hex_of = conform.renumber_map(keep)
    hexes: IntArray = mesh.hex[kept]
    face_keep: BoolArray = np.zeros(mesh.quads.n_quads, dtype=bool)
    if hexes.size:
        face_keep[np.unique(hexes)] = True
    sub_quads, new_face_of = quad_subset(mesh.quads, face_keep)
    return (HexMesh(sub_quads, new_face_of[hexes], mesh.face_orient[kept],
                    mesh.interior[kept],
                    mesh.face_tags.renumber(new_hex_of),
                    mesh.element_tags.gather(kept)),
            new_hex_of)


def select(mesh: HexMesh, which: str | BoolArray | IntArray | Sequence[int]
           ) -> HexMesh:
    """The named hexes as a block of their own, renumbered from zero.

    ``which`` is a tag string (every hex carrying it), an ``(E,)`` boolean mask, or an
    array of hex ids.  Kept hexes hold their relative order, their ``element_tags`` and
    whichever ``face_tags`` rows name them; faces, edges and points nothing kept touches
    are dropped.  The inverse of :func:`merge` -- ``merge(components(mesh))`` reproduces
    a mesh, and ``select`` by a region tag undoes what ``merge`` joined.

    The faces a removal **exposes** are new topological boundary and carry no tag:
    ``face_tags`` names what it named before, so name the hole yourself from
    :func:`boundary_faces <nekmeshpy.hexmesh.query.boundary_faces>` if the export needs
    it.  For the same reason the result is not guaranteed watertight -- that is the
    point of it."""
    return _subset(mesh, element_mask(which, mesh.element_tags, mesh.n_hexes,
                                      "hexmesh.select"))[0]


def remove(mesh: HexMesh, which: str | BoolArray | IntArray | Sequence[int]
           ) -> HexMesh:
    """The complement of :func:`select`: everything ``which`` does **not** name -- the
    "drop this block and re-fill it" half of the pair."""
    return _subset(mesh, ~element_mask(which, mesh.element_tags, mesh.n_hexes,
                                       "hexmesh.remove"))[0]


def components(mesh: HexMesh) -> list[HexMesh]:
    """The mesh split into its connected pieces -- one ``HexMesh`` per group of hexes
    reachable through shared corner points, in the order their first hex appears.

    What :func:`weld <nekmeshpy.hexmesh.query.weld>` and
    :func:`topology_report <nekmeshpy.hexmesh.query.topology_report>` tell you *about*
    (a mesh that turned out to be two bodies), this hands you as meshes."""
    n, labels = conform.element_components(mesh.hexes, mesh.n_points)
    return [_subset(mesh, labels == c)[0] for c in range(n)]


__all__ = [
    "components",
    "loft",
    "loft_fn",
    "loft_spline",
    "merge",
    "remove",
    "select",
]
