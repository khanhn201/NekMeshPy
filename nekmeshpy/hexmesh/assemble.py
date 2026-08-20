"""Variable-arity ``HexMesh`` operations -- the only ones that build a numbering."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

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
    element_mask,
    sweep_cap_tags,
    sweep_element_tags,
    welded_element_tags,
)
from ..linemesh import LineMesh
from ..pointmesh import PointMesh
from ..quadmesh import QuadMesh
from ..quadmesh.assemble import _subset as quad_subset
from ..quadmesh.query import element_blocks as quad_blocks
from .hexmesh import HexMesh, _sweep_at
from .query import _boundary_points, boundary_face_ids, tagged_faces

_log = logging.getLogger(__name__)


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
    quads = np.asarray(sec.corners, dtype=np.int64).reshape(-1, 4)
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
    sec_lines = np.asarray(sec.line_mesh.lines, dtype=np.int64).reshape(-1, 2)
    for k, sl in enumerate(slices):
        if not (np.array_equal(np.asarray(sl.corners, dtype=np.int64).reshape(-1, 4), quads)
                and np.array_equal(
                    np.asarray(sl.line_mesh.lines, dtype=np.int64).reshape(-1, 2), sec_lines)):
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
                            np.asarray(m.corners, dtype=np.int64).reshape(-1, 4), quads)
                        or not np.array_equal(
                            np.asarray(m.line_mesh.lines, dtype=np.int64).reshape(-1, 2),
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
    used_e: IntArray = np.unique(sec.quads.ravel())
    nu, ne = used_p.shape[0], used_e.shape[0]
    pslot: IntArray = np.full(nn, -1, np.int64)
    pslot[used_p] = np.arange(nu, dtype=np.int64)
    eslot: IntArray = np.full(sec.line_mesh.n_lines, -1, np.int64)
    eslot[used_e] = np.arange(ne, dtype=np.int64)
    ab: IntArray = np.sort(sec.line_mesh.lines[used_e], axis=1)       # (ne,2) min-first
    A, B = ab[:, 0], ab[:, 1]
    sec_side: IntArray = eslot[sec.quads][q_idx][:, fside]         # (E,4) used-edge slot
    # a section stores each shared edge in its own direction; the rows above are
    # min-first, so this is where a section's edge nodes have to be read backwards.
    rev_e: BoolArray = sec.line_mesh.lines[used_e][:, 0] > sec.line_mesh.lines[used_e][:, 1]

    def _edge_int(m: QuadMesh) -> PointArray:
        """That section's own shared-edge interiors for the used edges, turned into the
        canonical (min-first) direction the rows above are stored in."""
        e: PointArray = np.asarray(m.line_mesh.interior, dtype=float)[used_e]
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
    carried_sides: IntArray = stations.at_levels(eslot[sec.quads], lvl, ne)
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
    # -- 4b. the face tags, written onto the shared faces themselves -------------
    # Both families are closed forms, so a tag lands on a face id rather than on some
    # hex's view of one.  A tagged section *edge* names the face swept from it, one per
    # layer -- which is where the old ``(5 - side) if flip`` remap went: a swept face is
    # one object, and which way the hex on either side reads it is not its business.  A
    # cap face **is** a section quad, so with no argument it inherits that quad's own
    # element tag; on a closed sweep the two caps are the same faces and naming them
    # differently is refused rather than resolved by whichever is written second.
    fnamed = np.full(face_edges.shape[0], "", dtype=object)

    def _name(ids: IntArray, names: StrArray) -> None:
        hit = names != ""
        fnamed[np.asarray(ids, dtype=np.int64)[hit]] = names[hit]

    enames: StrArray = sec.edge_tags.dense(sec.line_mesh.n_lines)
    for e0 in sec.edge_tags.ids:
        if eslot[e0] >= 0:
            fnamed[n_prof * M + lay * ne + eslot[e0]] = enames[e0]
    closed = ElementTags.empty()
    cap: IntArray = np.arange(M, dtype=np.int64)
    first_caps = sweep_cap_tags(first_tag, closed if loop else sec.element_tags,
                                M, "HexMesh.loft")
    last_caps = sweep_cap_tags(last_tag, closed if loop else slices[-1].element_tags,
                               M, "HexMesh.loft")
    if loop:
        clash = np.flatnonzero((first_caps != "") & (last_caps != "")
                               & (first_caps != last_caps))
        if clash.size:
            raise ValueError(
                "HexMesh.loft: on a loop the first and last caps are the same seam "
                "faces, so they cannot be named differently -- got %r and %r on "
                "section quad %d. Name the seam once, or leave one side untagged."
                % (str(first_caps[clash[0]]), str(last_caps[clash[0]]),
                   int(clash[0])))
    _name(cap, first_caps)
    _name(nxt[nz - 1] * M + cap, last_caps)

    faces = QuadMesh(edge_lm, face_edges, face_flip, face_nodes,
                     ElementTags.from_dense(np.asarray(fnamed, dtype=np.str_)))

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

    etags = sweep_element_tags(element_tags, nz, M, "HexMesh.loft")
    return HexMesh(faces, elem_faces, face_orient, interior, etags)


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
        profs, order, loop=loop, conn=lambda m: np.asarray(m.corners, dtype=np.int64).reshape(-1, 4),
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
    edges: IntArray = np.asarray(ref.line_mesh.lines, dtype=np.int64).reshape(-1, 2)
    quads: IntArray = np.asarray(ref.corners, dtype=np.int64).reshape(-1, 4)
    # checked before the stack, so a mismatch names the section rather than failing
    # inside numpy with a shape it cannot explain
    for k, m in enumerate(prof):
        if (m.order != order or m.n_points != ref.n_points
                or not np.array_equal(
                    np.asarray(m.corners, dtype=np.int64).reshape(-1, 4), quads)
                or not np.array_equal(
                    np.asarray(m.line_mesh.lines, dtype=np.int64).reshape(-1, 2), edges)):
            raise ValueError(
                "loft_spline: every section must be index-paired with the first, but "
                "section %d stores a different order / point count / quad or shared-edge "
                "table.  Place one section with the affine ops (translate / rotate / "
                "transform) rather than rebuilding it per level." % k)
    P: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.line_mesh.points, dtype=float).reshape(-1, 3)
                  for s in prof]), t, loop=loop)
    E: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.line_mesh.interior, dtype=float) for s in prof]),
        t, loop=loop)
    F: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.interior, dtype=float) for s in prof]), t, loop=loop)
    fitted = [QuadMesh(LineMesh(PointMesh(P[k], ref.line_mesh.point_tags), edges, E[k],
                                ref.line_mesh.element_tags),
                       ref.quads, ref.orient, F[k], ref.element_tags)
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
    merged bounding-box extent).

    This is the **proximity** join: it is told nothing about what meets what and infers
    every seam in the assembly from coordinates, at one tolerance, over every block's
    whole boundary at once. When you know which face group meets which, :func:`attach`
    states it and confines the search to those two groups."""
    meshes = list(meshes)
    pos = [m.points for m in meshes]
    points, point_id = conform.weld_points(
        pos, [_boundary_points(m.corners) for m in meshes], tol)
    return _stitch(meshes, points, point_id, who="HexMesh.merge")


def _stitch(meshes: Sequence[HexMesh], points: PointArray, point_id: IntArray, *,
            who: str, seam_faces: Mapping[int, IntArray] | None = None,
            named_seams: Sequence[tuple[int, IntArray, str]] = ()) -> HexMesh:
    """Everything a weld does *after* the point remap is decided, shared by
    :func:`merge` and :func:`attach`: concatenate the blocks' B-rep tables, fuse the
    entities whose corners all welded, refit the seam's D4 codes, re-pin the shared
    high-order nodes, and combine the face tags.

    ``seam_faces`` names, per block index, the **local** face ids the caller welded shut
    -- those lose their names, since a buried face that keeps one makes the exporter
    write a boundary row from each side.  ``named_seams`` re-names a subset of them,
    ``(block, local faces, name)`` apiece, which is how one interface of an n-ary join
    can be named while the rest vanish.  :func:`merge`, never told what met what, passes
    neither."""
    meshes = list(meshes)
    counts = [m.points.shape[0] for m in meshes]

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
    etag_list: list[ElementTags] = []
    noff = eoff = foff = elem_off = 0
    for m, c in zip(meshes, counts):
        hex_list.append(point_id[m.corners + noff])    # local -> concat -> welded id
        erow_list.append(point_id[m.edges + noff])
        frow_list.append(point_id[np.asarray(m.quad_mesh.corners, dtype=np.int64) + noff])
        ee_list.append(m._elem_edges + eoff)
        eflip_list.append(m._edge_flip)
        ef_list.append(m.hexes + foff)
        forient_list.append(m.orient)
        etag_list.append(m.element_tags.offset(elem_off))
        noff += c
        eoff += m.edges.shape[0]
        foff += m.quad_mesh.n_quads
        elem_off += m.corners.shape[0]
    hexes = (np.concatenate(hex_list, axis=0) if hex_list
             else np.zeros((0, 8), np.int64))
    etags = ElementTags.concat(etag_list)

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
        ent_tol = conform.entity_tol(points)
        if seam_faces is not None:
            # A *stated* join knows which entities fuse, so the shared node tables are a
            # renumbering of blocks that already agree -- no gather into element-local
            # order, no scatter back, and no verification of the ~26k entities that did
            # not move.  Only the seam can disagree, and it is checked below.
            edge_nodes, face_nodes = _stated_shared_nodes(
                meshes, e_new, f_new, swap, edges.shape[0], canonical_conn.shape[0],
                f_rows, canonical_conn, ent_tol, who)
        else:
            local_e: PointArray = np.concatenate(
                [conform.gather_edge_nodes(mm.quad_mesh.line_mesh.interior,
                                           mm._elem_edges, mm._edge_flip)
                 for mm in meshes], axis=0)                # (E,12,order-1,3)
            local_f: PointArray = np.concatenate(
                [conform.gather_face_nodes(mm.quad_mesh.interior, mm.hexes, mm.orient)
                 for mm in meshes], axis=0)                # (E,6,(order-1)**2,3)
            edge_nodes = conform.scatter_edge_nodes(
                local_e, elem_edges, eflip, edges.shape[0], ent_tol, who)
            face_nodes = conform.scatter_face_nodes(
                local_f, elem_faces, face_orient, canonical_conn.shape[0], ent_tol,
                who)
        interior = np.concatenate([mm.interior for mm in meshes], axis=0)
    faces = _face_brep(points, edges, elem_edges, eflip, elem_faces, face_orient,
                       canonical_conn.shape[0], edge_nodes, face_nodes)
    # A face tag rides the face, so it waits for the merged face table: block ``m``'s
    # local face ``m.hexes[e, f]`` is merged face ``elem_faces[elem_off + e, f]``.  Two
    # blocks welding onto one shared face can each name it, so the combine is the
    # weld's own conflict rule rather than a concatenation -- which is the point of
    # storing the tag on the face: the disagreement is now visible instead of being
    # two rows nobody reconciles.
    off = 0
    ftag_list: list[ElementTags] = []
    seam_merged: list[IntArray] = []
    local_to_merged: list[IntArray] = []
    for bi, m in enumerate(meshes):
        mine: IntArray = np.full(m.quad_mesh.n_quads, -1, dtype=np.int64)
        mine[np.asarray(m.hexes, dtype=np.int64).ravel()] = np.asarray(
            elem_faces[off:off + m.corners.shape[0]], dtype=np.int64).ravel()
        ftag_list.append(m.face_tags.renumber(mine))
        local_to_merged.append(mine)
        if seam_faces is not None and bi in seam_faces:
            # the seam in this block's local numbering, carried onto the merged one
            seam_merged.append(mine[np.asarray(seam_faces[bi], dtype=np.int64)])
        off += m.corners.shape[0]
    named = [(local_to_merged[bi][np.asarray(f, dtype=np.int64)], tag)
             for bi, f, tag in named_seams]
    faces = QuadMesh(faces.line_mesh, faces.quads, faces.orient, faces.interior,
                     _seam_named(ftag_list, seam_merged, named, faces.n_quads, who))
    return HexMesh(faces, elem_faces, face_orient, interior, etags)


def _first_wins(dst: PointArray, idx: IntArray, src: PointArray) -> None:
    """``dst[idx] = src`` with the **lowest** source index winning each collision.

    Assigning in reverse leaves the first write last, which is the same owner
    ``scatter_edge_nodes`` picks (``np.unique``'s first occurrence).  With the blocks
    concatenated in order, that owner is always the earlier block."""
    dst[idx[::-1]] = src[::-1]


def _stated_shared_nodes(
    meshes: Sequence[HexMesh], e_new: IntArray, f_new: IntArray, swap: BoolArray,
    n_edges: int, n_faces: int, f_rows: IntArray, canonical_conn: IntArray,
    ent_tol: float, who: str,
) -> tuple[PointArray, PointArray]:
    """The shared edge- and face-interior tables for a join whose seam was **stated**.

    Every block's own tables are already conformal, so a weld cannot change any node
    that is not on the seam -- it only renumbers the entity that holds it.  This is
    therefore a concatenate plus two fixups:

    * an edge whose row came out reversed by the renumber reads its nodes backwards;
    * a *fused* entity keeps the survivor's block, and the loser's copy is checked
      against it rather than the whole mesh being re-verified.

    ``merge`` cannot take this path: it is not told what fused, so it has to gather every
    element's nodes and scatter them back to find out."""
    edge_src: PointArray = np.concatenate(
        [np.asarray(m.quad_mesh.line_mesh.interior, dtype=float) for m in meshes], axis=0)
    if edge_src.size:
        edge_src = np.where(swap[:, None, None], edge_src[:, ::-1, :], edge_src)
    face_src: PointArray = np.concatenate(
        [np.asarray(m.quad_mesh.interior, dtype=float) for m in meshes], axis=0)
    # a kept face keeps its own stored row, so its nodes stay in their own frame; a fused
    # one adopts the survivor's row and has to be turned into that frame to compare
    face_cmp: PointArray = conform.face_nodes_in_frame(
        face_src, canonical_conn[f_new], f_rows) if face_src.size else face_src

    edge_nodes: PointArray = np.empty((n_edges,) + edge_src.shape[1:], dtype=float)
    face_nodes: PointArray = np.empty((n_faces,) + face_cmp.shape[1:], dtype=float)
    _first_wins(edge_nodes, e_new, edge_src)
    _first_wins(face_nodes, f_new, face_cmp)

    # the conformal guard, on the seam alone: every entity two blocks now share must
    # agree, exactly as the full scatter would have demanded of all of them
    for name, new, src, table in (("edge", e_new, edge_src, edge_nodes),
                                  ("face", f_new, face_cmp, face_nodes)):
        if not src.size:
            continue
        dup: BoolArray = np.bincount(new, minlength=table.shape[0])[new] > 1
        if dup.any() and not np.allclose(src[dup], table[new[dup]],
                                         rtol=0.0, atol=ent_tol):
            raise ValueError(
                "%s: non-conforming high-order %s -- the two sides disagree on a welded "
                "shared %s's interior nodes beyond tolerance (%.3e). Pass own= so the "
                "seam takes one side's nodes outright." % (who, name, name, ent_tol))
    return edge_nodes, face_nodes


def _seam_named(ftag_list: Sequence[ElementTags], seam_merged: Sequence[IntArray],
                named: Sequence[tuple[IntArray, str]], n_faces: int,
                who: str) -> ElementTags:
    """The merged face tags, with the welded-shut seam renamed to what the caller asked.

    The seam's rows are dropped from **both** sides *before* the combine rather than
    overwritten after it: the caller has said what that face is, so the two sides stop
    being asked about it and cannot conflict. Every face off the seam still goes through
    :func:`welded_element_tags <nekmeshpy.core.tags.welded_element_tags>` and its
    refuse-on-disagreement rule."""
    if not seam_merged:
        return welded_element_tags(list(ftag_list), who)
    seam_ids: IntArray = np.unique(np.concatenate(list(seam_merged)))
    kept = [t.select(~np.isin(t.ids, seam_ids)) for t in ftag_list]
    merged = welded_element_tags(kept, who)
    if not named:
        return merged
    # object dtype, as ``tag_faces`` does: the merged table's own dtype is only as wide
    # as the longest name already in it, and a new one may be longer.
    dense = np.asarray(merged.dense(n_faces), dtype=object)
    for ids, tag in named:
        if tag:
            dense[ids] = tag
    return ElementTags.from_dense(np.asarray(dense, dtype=np.str_))


def _face_group(mesh: HexMesh, which: str | IntArray | Sequence[int],
                side: str) -> IntArray:
    """The seam's face ids on one side, from a tag name or given outright."""
    if isinstance(which, str):
        try:
            ids = tagged_faces(mesh, which)
        except ValueError as exc:
            # with several seams in one call, "no face carries the tag" is unactionable
            # unless it says which seam asked for it
            raise ValueError("attach: %s: %s" % (side, exc)) from None
    else:
        ids = np.asarray(which, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= mesh.quad_mesh.n_quads):
            raise ValueError(
                "attach: %s names face %d, outside this mesh's %d shared faces"
                % (side, int(ids.max()), mesh.quad_mesh.n_quads))
    buried = ids[~boundary_face_ids(mesh)[ids]]
    if buried.size:
        raise ValueError(
            "attach: %s names %d face(s) that already carry a hex on both sides (first "
            "is face %d). Joining onto a buried face would make the seam non-manifold; "
            "name a group that is still on its block's boundary."
            % (side, buried.size, int(buried[0])))
    return ids


def _pair_seam(a: HexMesh, fa: IntArray, b: HexMesh,
               fb: IntArray) -> tuple[IntArray, float]:
    """``((M,2) point pairs in each mesh's own numbering, the worst pairing distance)``
    -- the one place :func:`attach` reads a coordinate.

    There is **no tolerance**.  The pairing is each of ``a``'s seam points to its nearest
    on ``b``, and what proves it is not a distance but **bijectivity**: equal point counts
    plus an injective nearest-neighbour map is a one-to-one correspondence, whatever the
    two sides' separation.  So a seam whose halves sit far apart still joins, and a seam
    whose halves do not correspond is refused however close they are -- which is the
    right way round, and not a trade a radius can make.

    The search is confined to the two named groups, so nothing outside them can ever be
    paired. The worst distance comes back for the caller to log, not to test.

    One case neither this nor any tolerance can catch: a seam with a rotational symmetry
    whose two halves are relatively rotated by a symmetry element pairs up injectively,
    bijectively, and at distance zero -- onto a cyclic shift of the intended
    correspondence, welding a block in twisted. The point sets are identical, so there is
    nothing in the geometry left to distinguish the two readings."""
    pa: IntArray = np.unique(np.asarray(a.quad_mesh.corners, dtype=np.int64)[fa])
    pb: IntArray = np.unique(np.asarray(b.quad_mesh.corners, dtype=np.int64)[fb])
    if pa.size != pb.size:
        raise ValueError(
            "attach: the two groups are not the same surface -- %d faces / %d points on "
            "a, %d faces / %d points on b. Equal face counts with unequal point counts "
            "usually means the two sides are refined differently, which has no "
            "conformal weld." % (fa.size, pa.size, fb.size, pb.size))
    dist, loc = cKDTree(b.points[pb]).query(a.points[pa])
    dup = loc.size - np.unique(loc).size
    if dup:
        raise ValueError(
            "attach: the pairing is not one-to-one -- %d of a's %d seam points share a "
            "nearest point on b, so the two patterns do not correspond one for one. "
            "Either the groups are the same surface meshed differently, or one of them "
            "is the wrong group." % (dup, loc.size))
    return np.stack([pa, pb[loc]], axis=1), float(np.max(dist)) if dist.size else 0.0


def _adopt_seam(m: HexMesh, faces: IntArray, pts: IntArray, owner: HexMesh,
                owner_faces: IntArray, owner_pts: IntArray) -> HexMesh:
    """``m`` with its seam nodes replaced by ``owner``'s own, node for node.

    Until this runs the two sides agree only to within the pairing distance. Afterwards
    they agree bit for bit -- which matters because the shared-node re-scatter in
    :func:`_stitch` checks them against ``conform.entity_tol``, some four orders tighter
    than any pairing distance, and a merely-close seam fails it. Corners, the shared
    edges' interiors and the faces' own interiors are copied; the hexes' private
    interiors are not, so a seam that moved leaves the layer behind it distorted."""
    pmap: IntArray = np.arange(m.n_points, dtype=np.int64)
    pmap[pts] = owner_pts                                  # m's point id -> owner's

    P: PointArray = np.array(m.points, dtype=float, copy=True)
    P[pts] = owner.points[owner_pts]

    qm = m.quad_mesh
    lm = qm.line_mesh
    ei: PointArray = np.array(lm.interior, dtype=float, copy=True)
    fi: PointArray = np.array(qm.interior, dtype=float, copy=True)
    if m.order > 1:
        # Search the owner's *seam* entities, not its whole B-rep.  ``locate_rows`` sorts
        # whatever haystack it is handed, so passing the entire edge and face tables cost
        # 31 ms of a 71 ms join on a 7.7k-hex mesh -- to find 44 edges and 20 faces whose
        # owner-side ids are already known from ``owner_faces``.
        o_edges: IntArray = np.unique(
            np.asarray(owner.quad_mesh.quads, dtype=np.int64)[owner_faces])
        seam_edges: IntArray = np.unique(np.asarray(qm.quads, dtype=np.int64)[faces])
        rows: IntArray = pmap[np.asarray(lm.lines, dtype=np.int64)[seam_edges]]
        oidx = o_edges[conform.locate_rows(owner.edges[o_edges], rows,
                                           who="attach", what="edge")]
        en: PointArray = np.asarray(owner.edge_nodes, dtype=float)[oidx].copy()
        # the owner stores an edge min->max corner; flip the ones we traverse the other
        # way, so they read along this mesh's own direction
        rev = owner.edges[oidx, 0] != rows[:, 0]
        if en.size:
            en[rev] = en[rev][:, ::-1]
        ei[seam_edges] = en

        frows: IntArray = pmap[np.asarray(qm.corners, dtype=np.int64)[faces]]
        fidx = owner_faces[conform.locate_rows(owner.faces[owner_faces], frows,
                                               who="attach", what="face")]
        # turned out of the owner's stored frame into this mesh's, the same read
        # ``boundary_mesh`` does when it lifts a face out of its parent
        fi[faces] = conform.face_nodes_in_frame(
            np.asarray(owner.face_nodes, dtype=float)[fidx], frows,
            np.asarray(owner.quad_mesh.corners, dtype=np.int64)[fidx])

    lines = LineMesh(PointMesh(P, lm.point_tags), lm.lines, ei, lm.element_tags)
    quads = QuadMesh(lines, qm.quads, qm.orient, fi, qm.element_tags)
    return HexMesh(quads, m.hexes, m.orient, m.interior, m.element_tags)


class Seam(NamedTuple):
    """One stated interface: which face group of which block meets which.

    ``a`` and ``b`` name a block either by its position in ``attach``'s ``meshes`` or by
    the mesh object itself -- resolved by identity, so a block appearing twice in the
    list must be named by index.  ``tag_a`` / ``tag_b`` are face-tag names, or explicit
    arrays of face ids (:func:`tagged_faces <nekmeshpy.hexmesh.query.tagged_faces>`).

    ``own`` says which side's node coordinates the seam keeps.  ``attach_tag`` names the
    welded-shut faces; ``None`` -- the default -- clears them, because a buried face that
    keeps its name makes the exporter write one boundary row from *each* side of it."""
    a: int | HexMesh
    tag_a: str | IntArray
    b: int | HexMesh
    tag_b: str | IntArray
    own: str = "a"
    attach_tag: str | None = None


def _block_index(ref: int | HexMesh, meshes: Sequence[HexMesh], who: str) -> int:
    """A ``Seam`` endpoint resolved to a position in ``meshes``."""
    if isinstance(ref, HexMesh):
        for i, m in enumerate(meshes):
            if m is ref:
                return i
        raise ValueError(
            "attach: %s names a mesh that is not in the meshes list. Pass the block "
            "itself, or its index." % who)
    i = int(ref)
    if not 0 <= i < len(meshes):
        raise ValueError("attach: %s names block %d of %d" % (who, i, len(meshes)))
    return i


def attach(meshes: Sequence[HexMesh], seams: Sequence[Seam]) -> HexMesh:
    """Join blocks along the interfaces each :class:`Seam` names, in **one pass**.

    Every seam states which face group of which block meets which, so no tolerance is
    needed anywhere: inside a named pair of groups the pairing is nearest-neighbour, and
    what proves it is bijectivity -- equal point counts plus an injective map is a
    one-to-one correspondence however far apart the two halves sit.  A seam with a real
    gap joins; a seam whose halves do not correspond is refused however close they are.

    Contrast :func:`merge`, which is told nothing and infers every seam in the assembly
    from coordinates at one global tolerance.

    One pass is the point of the n-ary form.  Chaining binary joins would concatenate and
    rebuild the whole accumulated mesh once per link -- work quadratic in the block count,
    since 32 blocks of 120 hexes cost 63240 hex-passes chained against 3840 in a single
    pass.  Here every seam is paired first (each cheap, and confined to its own groups),
    then the blocks are welded and stitched exactly once::

        mesh = hexmesh.attach([core, leg_p, leg_m],
                              [Seam(core, "port_p", leg_p, "join"),
                               Seam(core, "port_m", leg_m, "join")])

    Blocks are concatenated in list order and a welded point keeps the lowest id, so the
    first block's own numbering comes through untouched."""
    meshes = list(meshes)
    seams = list(seams)
    if not meshes:
        raise ValueError("attach: no meshes to join")
    if len(meshes) == 1 and not seams:
        return meshes[0]
    order = meshes[0].order
    if any(m.order != order for m in meshes):
        raise ValueError("attach: every block must share the same order, got %s"
                         % sorted({m.order for m in meshes}))

    # 1. resolve every seam to (block, faces) on each side, and pair its points.  The
    #    pairing reads only the two named groups, so it is independent of mesh size.
    resolved: list[tuple[int, IntArray, int, IntArray, str, str | None]] = []
    for k, sm in enumerate(seams):
        who = "seams[%d]" % k
        ia = _block_index(sm.a, meshes, who + ".a")
        ib = _block_index(sm.b, meshes, who + ".b")
        if sm.own not in ("a", "b"):
            raise ValueError("attach: %s.own must be 'a' or 'b', got %r" % (who, sm.own))
        fa = _face_group(meshes[ia], sm.tag_a, who + ".tag_a")
        fb = _face_group(meshes[ib], sm.tag_b, who + ".tag_b")
        if fa.size != fb.size:
            raise ValueError(
                "attach: %s joins groups of different face counts (%d and %d), so they "
                "cannot be the same interface." % (who, fa.size, fb.size))
        if fa.size == 0:
            raise ValueError("attach: %s names empty groups; there is nothing to join"
                             % who)
        resolved.append((ia, fa, ib, fb, sm.own, sm.attach_tag))

    # 2. make each seam's two sides agree bit for bit before anything is welded.  Done
    #    in seam order and written back into ``meshes``, so a block carrying several
    #    seams accumulates them rather than losing all but the last.
    pair_list: list[IntArray] = []
    for ia, fa, ib, fb, own, _tag in resolved:
        pairs, worst = _pair_seam(meshes[ia], fa, meshes[ib], fb)
        _log.debug("attach: %d faces, %d points, worst pairing distance %.3e",
                   fa.size, pairs.shape[0], worst)
        if own == "a":
            meshes[ib] = _adopt_seam(meshes[ib], fb, pairs[:, 1], meshes[ia], fa,
                                     pairs[:, 0])
        else:
            meshes[ia] = _adopt_seam(meshes[ia], fa, pairs[:, 0], meshes[ib], fb,
                                     pairs[:, 1])
        pair_list.append(np.stack([pairs[:, 0], pairs[:, 1]], axis=1))

    # 3. one weld and one stitch over the whole assembly
    offs: IntArray = np.concatenate(
        [[0], np.cumsum([m.n_points for m in meshes])]).astype(np.int64)
    cat = [np.stack([pr[:, 0] + offs[ia], pr[:, 1] + offs[ib]], axis=1)
           for (ia, _fa, ib, _fb, _o, _t), pr in zip(resolved, pair_list)]
    stated: IntArray = (np.concatenate(cat, axis=0) if cat
                        else np.zeros((0, 2), dtype=np.int64))
    points, point_id = conform.weld_pairs([m.points for m in meshes], stated)

    seam_faces: dict[int, list[IntArray]] = {}
    for ia, fa, ib, fb, _o, _t in resolved:
        seam_faces.setdefault(ia, []).append(fa)
        seam_faces.setdefault(ib, []).append(fb)
    named = [(ia, fa, tag) for ia, fa, _ib, _fb, _o, tag in resolved if tag]
    return _stitch(meshes, points, point_id, who="hexmesh.attach",
                   seam_faces={b: np.unique(np.concatenate(v))
                               for b, v in seam_faces.items()},
                   named_seams=named)


def _subset(mesh: HexMesh, keep: BoolArray) -> tuple[HexMesh, IntArray]:
    """``(the kept hexes as a HexMesh, new_hex_of)`` -- the top rung of
    :func:`quadmesh._subset <nekmeshpy.quadmesh.assemble._subset>`, which it calls to
    carry the shared faces down (and which calls the line rung under that).

    Shared faces no kept hex references are dropped, with the edges and points under
    them.  The ``face_orient`` codes survive untouched: they describe each hex's frame
    against its own canonical face row, and dropping a *neighbour* changes neither."""
    kept, new_hex_of = conform.renumber_map(keep)
    hexes: IntArray = mesh.hexes[kept]
    face_keep: BoolArray = np.zeros(mesh.quad_mesh.n_quads, dtype=bool)
    if hexes.size:
        face_keep[np.unique(hexes)] = True
    sub_quads, new_face_of = quad_subset(mesh.quad_mesh, face_keep)
    # the face tags ride ``sub_quads`` -- ``quad_subset`` already carried them onto
    # the compacted face numbering
    return (HexMesh(sub_quads, new_face_of[hexes], mesh.orient[kept],
                    mesh.interior[kept],
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

    What :func:`topology_report <nekmeshpy.hexmesh.query.topology_report>` tells you
    *about*
    (a mesh that turned out to be two bodies), this hands you as meshes."""
    n, labels = conform.element_components(mesh.corners, mesh.n_points)
    return [_subset(mesh, labels == c)[0] for c in range(n)]


__all__ = [
    "Seam",
    "attach",
    "components",
    "loft",
    "loft_fn",
    "loft_spline",
    "merge",
    "remove",
    "select",
]
