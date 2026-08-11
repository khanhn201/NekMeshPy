"""Variable-arity ``QuadMesh`` operations -- the only ones that build a numbering."""

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
    element_mask,
    sweep_cap_tags,
    sweep_element_tags,
    welded_element_tags,
)
from ..linemesh import LineMesh
from ..linemesh.assemble import _subset as line_subset
from ..linemesh.query import element_blocks as line_blocks
from ..pointmesh import PointMesh
from .quadmesh import (
    QuadMesh,
    _quad_interior_slots,
)
from .query import _boundary_mask


def loft(
    slices: Sequence[LineMesh],
    *,
    loop: bool = False,
    sweep_nodes: Sequence[Sequence[LineMesh]] | None = None,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> QuadMesh:
    """Loft a stack of conformal ``LineMesh`` profiles into a quad section (the general
    primitive behind :func:`extrude <nekmeshpy.quadmesh.lift.extrude>`, and the middle
    rung of the uniform sweep shared with :func:`linemesh.assemble.loft
    <nekmeshpy.linemesh.assemble.loft>` and :func:`HexMesh.loft
    <nekmeshpy.hexmesh.assemble.loft>`)."""
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
        raise ValueError(
            "loft needs at least 2 profiles (one layer), got %d" % n_prof)
    if loop and nz < 3:
        raise ValueError(
            "loft(loop=True) needs at least 3 profiles, got %d -- one layer sweeps a profile "
            "onto itself so every element's corners repeat, and two put both layers "
            "across the same pair of levels, which gives two distinct rungs the same "
            "corner ids.  An entity here *is* its corners, so neither is representable."
            % n_prof)
    lines = np.asarray(slices[0].lines, dtype=np.int64).reshape(-1, 2)
    L = lines.shape[0]
    S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                  for s in slices], axis=0)              # (n_prof, nn, 3)
    nn = S.shape[1]
    points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v

    order = slices[0].order
    if any(s.order != order for s in slices):
        raise ValueError("loft: all slices must share the same order")
    ho = order > 1
    # Every profile's high-order nodes are read by the *first* profile's entity ids, so
    # matching corners is not enough -- the line tables have to be the same table.
    for k, prof in enumerate(slices):
        if not np.array_equal(np.asarray(prof.lines, dtype=np.int64).reshape(-1, 2),
                              lines):
            raise ValueError(
                "loft: every profile must be index-paired with the first, but profile "
                "%d stores a different line table.  Place one profile with the affine "
                "ops (translate / rotate / transform) rather than rebuilding it per "
                "level, or the sweep reads its nodes off the wrong lines." % k)

    # the intermediate profiles, if any -- one list of ``order-1`` per layer, in
    # ascending GLL-level order.  Validated here so a mis-sized stack names the
    # layer rather than failing later inside a fancy-index.
    sw: list[list[LineMesh]] | None = None
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
                    "profiles, got %d" % (i, order - 1, len(level)))
            for m in level:
                if (m.order != order or m.n_points != nn
                        or not np.array_equal(
                            np.asarray(m.lines, dtype=np.int64).reshape(-1, 2), lines)):
                    raise ValueError(
                        "loft: sweep_nodes[%d] profiles must match the slices "
                        "(order %d, %d points, and the same line table), got order %d "
                        "with %d points" % (i, order, nn, m.order, m.n_points))
        if not ho:
            sw = None                      # order 1 has no interior level at all

    a = lines[:, 0]
    b = lines[:, 1]
    # ``nxt[i]`` is the level layer ``i`` sweeps *to*: i+1 normally, wrapping to 0 on the
    # closing layer of a periodic sweep -- the one place ``loop`` shows up.
    nxt: IntArray = np.arange(1, nz + 1, dtype=np.int64) % n_prof
    lay: IntArray = np.arange(nz, dtype=np.int64)
    lvl: IntArray = np.arange(n_prof, dtype=np.int64)
    # quad ``i*L + l`` is profile line ``l`` dragged across layer ``i``
    i_idx: IntArray = np.repeat(lay, L)
    l_idx: IntArray = np.tile(np.arange(L, dtype=np.int64), nz)
    j_idx: IntArray = nxt[i_idx]
    av, bv = a[l_idx], b[l_idx]

    # Only points a line actually carries get a rung -- an isolated point borders no
    # quad, so a rung there would be a dangling edge.
    used: IntArray = (np.unique(lines.ravel()) if L else np.zeros(0, np.int64))
    nu = used.shape[0]
    slot: IntArray = np.full(nn, -1, np.int64)
    slot[used] = np.arange(nu, dtype=np.int64)

    g = gll_nodes(order)
    row = order + 1
    # the profiles' own curves -- the single source both node steps read from
    curves: PointArray = np.empty((n_prof if ho else 0, L, row, 3), dtype=float)
    for k, prof in enumerate(slices if ho else ()):
        curves[k] = line_blocks(prof)

    # -- 1. the shared edges: the global LineMesh's topology --------------------
    # The two entity families of any sweep, at this rung.  Both are closed forms -- id
    # ``level*L + line`` carried, ``nlev*L + layer*nu + point`` swept, two contiguous
    # blocks that cannot collide -- so the table is written, never deduplicated.
    carried_rows: IntArray = stations.at_levels(lines, lvl, nn)      # a profile line at a level
    swept_rows: IntArray = np.stack([                        # a profile point, dragged
        stations.at_levels(used, lay, nn), stations.at_levels(used, nxt, nn)], axis=1)
    edges: IntArray = np.concatenate([carried_rows, swept_rows], axis=0)

    # -- 2. that LineMesh's interior: one write per shared edge -----------------
    # Each edge has exactly one source, so there is no owner election.  A carried edge is
    # the profile's own curve verbatim; a rung is the straight GLL blend of its two
    # corners, or -- given ``sweep_nodes`` -- the intermediate profiles' own points.
    edge_nodes: PointArray | None = None
    if ho:
        carried: PointArray = curves[lvl][:, :, 1:order, :]
        if sw is not None:
            rungs: PointArray = np.stack(
                [np.stack([np.asarray(sw[i][k - 1].points, dtype=float)
                           .reshape(-1, 3)[used] for k in range(1, order)], axis=1)
                 for i in range(nz)], axis=0)
        else:
            Sp: PointArray = S[:, used, :]              # (n_prof, nu, 3)
            lo, hi = Sp[lay], Sp[nxt]                   # (nz,nu,3) the rung's two ends
            rungs = (lo[:, :, None, :]
                     + g[1:order][None, None, :, None] * (hi - lo)[:, :, None, :])
        edge_nodes = np.concatenate(
            [carried.reshape(-1, order - 1, 3), rungs.reshape(-1, order - 1, 3)], axis=0)
    # -- 2b. the edge tags, written onto the shared edges themselves ------------
    # Both families are closed forms, so a tag lands on an id rather than on some
    # quad's view of an id.  A tagged profile point names the edge *swept* from it, one
    # per layer; a cap edge **is** a profile line carried to the bounding level, so with
    # no argument it inherits that line's own element tag -- except on a closed sweep,
    # where the "caps" are the interior seam and only an explicit tag names them.  On a
    # loop the two caps are the same edges and the later write wins: one edge, one name.
    enamed = np.full(edges.shape[0], "", dtype=object)

    def _name(ids: IntArray, names: StrArray) -> None:
        hit = names != ""
        enamed[np.asarray(ids, dtype=np.int64)[hit]] = names[hit]

    pnames: StrArray = slices[0].point_tags.dense(nn)
    for p0 in slices[0].point_tags.ids:
        if slot[p0] >= 0:
            enamed[n_prof * L + lay * nu + slot[p0]] = pnames[p0]
    closed = ElementTags.empty()
    cap: IntArray = np.arange(L, dtype=np.int64)
    first_caps = sweep_cap_tags(first_tag, closed if loop else slices[0].element_tags,
                                L, "QuadMesh.loft")
    last_caps = sweep_cap_tags(last_tag, closed if loop else slices[-1].element_tags,
                               L, "QuadMesh.loft")
    if loop:
        # a closed sweep's two "caps" are the *same* seam edges, approached from either
        # side.  Naming each differently used to give two (quad, side) rows on one
        # edge; there is one name per edge now, so the disagreement has to be refused
        # rather than resolved by whichever happens to be written second.
        clash = np.flatnonzero((first_caps != "") & (last_caps != "")
                               & (first_caps != last_caps))
        if clash.size:
            raise ValueError(
                "QuadMesh.loft: on a loop the first and last caps are the same seam "
                "edges, so they cannot be named differently -- got %r and %r on "
                "section line %d. Name the seam once, or leave one side untagged."
                % (str(first_caps[clash[0]]), str(last_caps[clash[0]]),
                   int(clash[0])))
    _name(cap, first_caps)
    _name(nxt[nz - 1] * L + cap, last_caps)

    lm = LineMesh(points, edges, interior=edge_nodes,
                  element_tags=ElementTags.from_dense(
                      np.asarray(enamed, dtype=np.str_)))

    # -- 3. the quads, as indices into that LineMesh ----------------------------
    # corners [a_i, b_i, b_j, a_j]; local edges [carried at i, rung at b, carried at j,
    # rung at a].  Sides 3 / 4 traverse their stored edge backwards, hence the flip row.
    quad: IntArray = np.stack([
        i_idx * L + l_idx,
        n_prof * L + i_idx * nu + slot[bv],
        j_idx * L + l_idx,
        n_prof * L + i_idx * nu + slot[av]], axis=1)
    flip: BoolArray = np.tile(np.array([False, False, True, True]), (nz * L, 1))

    # -- 4. the private per-quad interior ---------------------------------------
    # Without ``sweep_nodes`` a column quad is *ruled*: curved along the profile line,
    # straight along the sweep.  A transfinite (Coons) patch would give the same surface
    # -- with rungs this straight its left/right terms cancel the bilinear exactly -- so
    # the two level curves alone are the whole answer.  With ``sweep_nodes`` every level
    # is a genuine profile and the block is a pure gather.  Either way the boundary
    # already lives on the shared edges, so only the private slots are evaluated.
    interior: PointArray | None = None
    if ho:
        islots = _quad_interior_slots(order)
        if sw is not None:
            lev: PointArray = np.empty((nz, row, L, row, 3), dtype=float)
            lev[:, 0] = curves[lay]
            lev[:, order] = curves[nxt]
            for k in range(1, order):
                lev[:, k] = np.stack(
                    [line_blocks(sw[i][k - 1]) for i in range(nz)], axis=0)
            interior = lev[i_idx[:, None], (islots // row)[None, :],
                           l_idx[:, None], (islots % row)[None, :], :]
        else:
            iu = islots % row                           # along the profile line
            gv = g[islots // row][None, :, None]        # along the sweep
            interior = ((1.0 - gv) * curves[i_idx, l_idx][:, iu, :]
                        + gv * curves[j_idx, l_idx][:, iu, :])

    etags = sweep_element_tags(element_tags, nz, L, "QuadMesh.loft")
    return QuadMesh(lm, quad, flip, interior, etags)


def _loft_evaluated(
    profs: Sequence[LineMesh],
    order: int,
    *,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
    name: str = "loft_fn",
) -> QuadMesh:
    """The shared tail of every sweep whose profiles are **evaluated** on the refined
    node lattice rather than handed in: validate, close the loop, split, delegate."""
    slices, sweep_nodes = stations.split_evaluated(
        profs, order, loop=loop, conn=lambda m: np.asarray(m.lines, dtype=np.int64).reshape(-1, 2),
        noun="profile", elems="lines", name=name)
    return loft(slices, loop=loop,
                sweep_nodes=sweep_nodes if order > 1 else None,
                element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag)


def loft_spline(
    slices: Sequence[LineMesh],
    *,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> QuadMesh:
    """:func:`loft <nekmeshpy.quadmesh.assemble.loft>` with the sweep-direction nodes read
    off a **cubic spline through the whole stack** of profiles, rather than blended
    straight between the two bounding a layer.

    Same arguments, same numbering, same tags, and the profiles given come back verbatim
    as the levels -- the spline interpolates them, so this adds curvature between profiles
    without moving any.  It is the automatic form of ``loft(..., sweep_nodes=...)``: where
    that asks the caller for the intermediate profiles, this fits them.  Every node block
    a profile stores is fitted, its ``interior`` as well as its points, so the result is
    curved along the sweep at every node rather than only at the corners.

    Reach for it when a sweep has a feature that turns sharply across a handful of
    profiles: ``loft`` cuts the corner with a chord however high the order, and refining
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
    lines: IntArray = np.asarray(ref.lines, dtype=np.int64).reshape(-1, 2)
    # checked before the stack, so a mismatch names the profile rather than failing
    # inside numpy with a shape it cannot explain
    for k, m in enumerate(prof):
        if (m.order != order or m.n_points != ref.n_points
                or not np.array_equal(
                    np.asarray(m.lines, dtype=np.int64).reshape(-1, 2), lines)):
            raise ValueError(
                "loft_spline: every profile must be index-paired with the first, but "
                "profile %d stores a different order / point count / line table.  Place "
                "one profile with the affine ops (translate / rotate / transform) rather "
                "than rebuilding it per level." % k)
    P: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3) for s in prof]),
        t, loop=loop)
    I: PointArray = stations.spline_levels(
        np.stack([np.asarray(s.interior, dtype=float) for s in prof]), t, loop=loop)
    fitted = [LineMesh(PointMesh(P[k], ref.point_tags), lines, I[k],
                       ref.element_tags)
              for k in range(t.shape[0])]
    return _loft_evaluated(fitted, order, loop=loop, element_tags=element_tags,
                           first_tag=first_tag, last_tag=last_tag, name="loft_spline")


def loft_fn(
    f: Callable[[float], LineMesh],
    fractions: FloatArray,
    *,
    loop: bool = False,
    order: int | None = None,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> QuadMesh:
    """Loft a section from a **parametrized family of profiles** -- :func:`loft
    <nekmeshpy.quadmesh.assemble.loft>` with the slices evaluated rather than handed in,
    so **every** node (the corners *and* the sweep-direction high-order nodes) comes
    from calling ``f`` and nothing is blended along the sweep."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    stations.check_fraction_count(fr, loop=loop, name="loft_fn")
    if order is None:
        # The node lattice the profiles are sampled on is a function of the order, so
        # the order has to be settled before the sweep can start -- and ``f`` is the
        # only thing that knows it.  One throwaway evaluation at the first fraction is
        # the whole cost; the profile it returns is discarded and re-evaluated with
        # the rest, so no partial state leaks out of the probe.
        probe = f(float(fr[0]))
        if not isinstance(probe, LineMesh):
            raise TypeError(
                "loft_fn: f must return a LineMesh profile, but f(%g) returned %s. "
                "Pass order= explicitly only if you also fix f." % (fr[0], type(probe)))
        order = probe.order

    t: FloatArray = stations.refined_lattice(fr, order)
    profs: list[LineMesh] = [f(float(v)) for v in t]
    return _loft_evaluated(profs, order, loop=loop,
                           element_tags=element_tags,
                           first_tag=first_tag, last_tag=last_tag)


def merge(meshes: Sequence[QuadMesh], *, tol: float | None = None) -> QuadMesh:
    """Merge quad sections into one, welding coincident boundary points. ``tol`` is the
    absolute coincidence distance (default ``1e-7`` x the extent)."""
    meshes = list(meshes)
    pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
    counts = [p.shape[0] for p in pos]
    # the weldable points here are the vertices of the boundary edges -- the 2-D
    # analogue of the line rung's chain ends, which is the same weld with a different
    # notion of "boundary".
    seams: list[IntArray] = []
    for m in meshes:
        edges, mask = _boundary_mask(m.quads)
        seams.append(np.unique(edges[mask]))
    points, point_id = conform.weld_points(pos, seams, tol)

    quad_list: list[IntArray] = []
    etag_list: list[ElementTags] = []
    noff = qoff = 0
    for m, c in zip(meshes, counts):
        quad_list.append(point_id[m.quads + noff])   # local -> welded id
        etag_list.append(m.element_tags.offset(qoff))
        noff += c
        qoff += m.n_quads
    quads = np.concatenate(quad_list, axis=0) if quad_list else np.zeros((0, 4), np.int64)
    etags = ElementTags.concat(etag_list)

    # order-N: the private per-quad interiors just concatenate, but the shared edge
    # tables must be rebuilt against the *merged* topology -- gather each block's
    # nodes into its own element traversal order, concatenate in merged element
    # order, then re-scatter.  That scatter is the conformal-weld guard: two blocks
    # that disagree on a welded shared edge raise instead of silently welding.
    order = meshes[0].order if meshes else 1
    if any(m.order != order for m in meshes):
        raise ValueError("merge: all sections must share the same order")
    edges, elem_edges, flip = conform.unique_edges(quads, 2)
    edge_nodes: PointArray | None = None
    interior: PointArray | None = None
    if order > 1:
        local: PointArray = np.concatenate(
            [conform.gather_edge_nodes(m.lines.interior, m.quad, m.flip)
             for m in meshes], axis=0)                     # (Q,4,order-1,3)
        edge_nodes = conform.scatter_edge_nodes(
            local, elem_edges, flip, edges.shape[0],
            conform.entity_tol(points), "QuadMesh.merge")
        interior = np.concatenate([m.interior for m in meshes], axis=0)
    # An edge tag rides the edge, so it has to wait for the merged edge table: block
    # ``m``'s local edge ``m.quad[q, s]`` is merged edge ``elem_edges[qoff + q, s]``,
    # which is the whole map.  Two blocks welding onto one shared edge can each name
    # it, so the combine is the weld's own conflict rule rather than a concatenation.
    etag_off = 0
    edge_tag_list: list[ElementTags] = []
    for m in meshes:
        mine: IntArray = np.full(m.lines.n_lines, -1, dtype=np.int64)
        mine[np.asarray(m.quad, dtype=np.int64).ravel()] = np.asarray(
            elem_edges[etag_off:etag_off + m.n_quads], dtype=np.int64).ravel()
        edge_tag_list.append(m.edge_tags.renumber(mine))
        etag_off += m.n_quads
    lm = LineMesh(points, edges, interior=edge_nodes,
                  element_tags=welded_element_tags(edge_tag_list, "QuadMesh.merge"))
    return QuadMesh(lm, elem_edges, flip, interior, etags)

def _subset(mesh: QuadMesh, keep: BoolArray) -> tuple[QuadMesh, IntArray]:
    """``(the kept quads as a QuadMesh, new_quad_of)`` -- the quad rung of
    :func:`linemesh._subset <nekmeshpy.linemesh.assemble._subset>`, which it calls to
    carry the shared edges down.

    Shared edges no kept quad references are dropped with the points under them, so the
    B-rep comes back complete rather than trailing orphaned entities.  Order is
    preserved: the shared edge-interior and private per-quad nodes ride along."""
    kept, new_quad_of = conform.renumber_map(keep)
    quad: IntArray = mesh.quad[kept]
    edge_keep: BoolArray = np.zeros(mesh.lines.n_lines, dtype=bool)
    if quad.size:
        edge_keep[np.unique(quad)] = True
    sub_lines, new_edge_of = line_subset(mesh.lines, edge_keep)
    # the edge tags ride ``sub_lines`` -- ``line_subset`` already carried them onto
    # the compacted edge numbering, which is the whole of it
    return (QuadMesh(sub_lines, new_edge_of[quad], mesh.flip[kept],
                     mesh.interior[kept],
                     mesh.element_tags.gather(kept)),
            new_quad_of)


def select(mesh: QuadMesh, which: str | BoolArray | IntArray | Sequence[int]
           ) -> QuadMesh:
    """The named quads as a section of their own, renumbered from zero.

    ``which`` is a tag string (every quad carrying it), a ``(Q,)`` boolean mask, or an
    array of quad ids.  Kept quads hold their relative order, their ``element_tags`` and
    whichever ``edge_tags`` rows name them; edges and points nothing kept touches are
    dropped.  The inverse of :func:`merge`.

    Removing elements can open the section up, so the result is **not** guaranteed to be
    simply connected -- or connected at all.  Ask :func:`components` if that matters."""
    return _subset(mesh, element_mask(which, mesh.element_tags, mesh.n_quads,
                                      "quadmesh.select"))[0]


def remove(mesh: QuadMesh, which: str | BoolArray | IntArray | Sequence[int]
           ) -> QuadMesh:
    """The complement of :func:`select`: everything ``which`` does **not** name."""
    return _subset(mesh, ~element_mask(which, mesh.element_tags, mesh.n_quads,
                                       "quadmesh.remove"))[0]


def components(mesh: QuadMesh) -> list[QuadMesh]:
    """The section split into its connected pieces -- one ``QuadMesh`` per group of
    quads reachable through shared corner points, in the order their first quad
    appears."""
    n, labels = conform.element_components(mesh.quads, mesh.n_points)
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
