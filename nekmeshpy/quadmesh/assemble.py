"""Variable-arity ``QuadMesh`` operations -- the only ones that build a numbering."""

from __future__ import annotations

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
from ..core.interp import _CORNER_IJK, resample_block_at
from ..core.tags import (
    ElementTags,
    element_mask,
    sweep_cap_tags,
    sweep_element_tags,
    welded_element_tags,
)
from ..linemesh import LineMesh
from ..linemesh.assemble import _subset as line_subset
from ..linemesh.assemble import refine as line_refine
from ..linemesh.query import element_blocks as line_blocks
from ..pointmesh import PointMesh
from ._helpers import entities_from_blocks
from .quadmesh import (
    QuadMesh,
    _quad_interior_slots,
)
from .query import _boundary_mask, element_blocks, tagged_edges


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


def merge(meshes: Sequence[QuadMesh], *, tol: float = 1e-7) -> QuadMesh:
    """Merge quad sections into one, welding coincident boundary points.

    ``tol`` is a **fraction** of ``conform.bbox_scale`` -- the largest of the x/y/z
    ranges over every point handed in -- so the coincidence radius is
    ``tol * bbox_scale``.  Not a distance: if you know one, divide by
    ``conform.bbox_scale(...)``.

    The **proximity** join: told nothing about what meets what, it infers every seam
    from coordinates at one tolerance. :func:`attach` is told which edge group meets
    which and confines the search to those two."""
    meshes = list(meshes)
    pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
    # the weldable points here are the vertices of the boundary edges -- the 2-D
    # analogue of the line rung's chain ends, which is the same weld with a different
    # notion of "boundary".
    seams: list[IntArray] = []
    for m in meshes:
        edges, mask = _boundary_mask(m.corners)
        seams.append(np.unique(edges[mask]))
    points, point_id = conform.weld_points(pos, seams, tol)
    return _stitch(meshes, points, point_id, who="QuadMesh.merge")


def _stitch(meshes: Sequence[QuadMesh], points: PointArray, point_id: IntArray, *,
            who: str, seam_edges: Mapping[int, IntArray] | None = None,
            own_edges: Mapping[int, IntArray] | None = None, check: bool = True,
            named_seams: Sequence[tuple[int, IntArray, str]] = ()) -> QuadMesh:
    """Everything a weld does *after* the point remap is decided, shared by
    :func:`merge` and :func:`attach` -- the quad rung's counterpart of
    :func:`hexmesh._stitch <nekmeshpy.hexmesh.assemble._stitch>`.

    ``seam_edges`` names, per block index, the **local** edge ids the caller welded shut
    -- those lose their names.  ``named_seams`` re-names a subset, ``(block, local edges,
    name)`` apiece, so one interface of an n-ary join can be named while the rest
    vanish."""
    meshes = list(meshes)
    counts = [m.points.shape[0] for m in meshes]

    # Each block's own B-rep is already correct and already unique, so the tables are
    # concatenated and only the entities whose corners *all* welded are fused.  Nothing
    # is re-derived from the corners -- the same climb the hex rung makes, and it keeps
    # the merged numbering a function of the order the blocks were handed in.
    erow_list: list[IntArray] = []
    ee_list: list[IntArray] = []
    eflip_list: list[BoolArray] = []
    etag_list: list[ElementTags] = []
    edge_offs: list[int] = []
    noff = eoff = qoff = 0
    for m, c in zip(meshes, counts):
        erow_list.append(point_id[np.asarray(m.line_mesh.lines, dtype=np.int64) + noff])
        ee_list.append(np.asarray(m.quads, dtype=np.int64) + eoff)
        eflip_list.append(np.asarray(m.orient, dtype=bool))
        etag_list.append(m.element_tags.offset(qoff))
        edge_offs.append(eoff)
        noff += c
        eoff += m.line_mesh.n_lines
        qoff += m.n_quads
    etags = ElementTags.concat(etag_list)

    order = meshes[0].order if meshes else 1
    if any(m.order != order for m in meshes):
        raise ValueError("merge: all sections must share the same order")

    e_rows = (np.concatenate(erow_list, axis=0) if erow_list
              else np.zeros((0, 2), np.int64))
    elem_edges = (np.concatenate(ee_list, axis=0) if ee_list
                  else np.zeros((0, 4), np.int64))
    flip = (np.concatenate(eflip_list, axis=0) if eflip_list
            else np.zeros((0, 4), bool))

    # welding renumbers points, so a stored edge row can come out the wrong way round;
    # put it back min-first and toggle the traversals that referenced it, *before*
    # fusing, so a fused pair is then two identical rows and no direction survives it.
    swap: BoolArray = e_rows[:, 0] > e_rows[:, 1]
    e_rows = np.where(swap[:, None], e_rows[:, ::-1], e_rows)
    flip = flip ^ swap[elem_edges]

    welded: BoolArray = (np.bincount(point_id, minlength=points.shape[0]) > 1
                         if point_id.size else np.zeros(points.shape[0], bool))
    e_new, e_keep = conform.fuse_entities(e_rows, welded)
    edges = e_rows[e_keep]
    elem_edges = e_new[elem_edges]

    edge_nodes: PointArray | None = None
    interior: PointArray | None = None
    if order > 1:
        # A weld can only fuse edges whose two corners both welded; every other edge
        # keeps its one stored row, so this is a renumbering plus a guard on the fused
        # subset alone -- the only edges that can disagree.
        prefer_e = None
        if own_edges:
            prefer_e = np.zeros(e_rows.shape[0], dtype=bool)
            for bi3, off3 in enumerate(edge_offs):
                loc = own_edges.get(bi3)
                if loc is not None and len(loc):
                    prefer_e[off3 + np.asarray(loc, dtype=np.int64)] = True
        edge_nodes = _shared_edge_nodes(meshes, e_new, swap, edges.shape[0],
                                        conform.entity_tol(points), who,
                                        prefer_e, check)
        interior = np.concatenate([m.interior for m in meshes], axis=0)
    # An edge tag rides the edge, so it has to wait for the merged edge table: block
    # ``m``'s local edge ``m.quads[q, s]`` is merged edge ``elem_edges[qoff + q, s]``,
    # which is the whole map.  Two blocks welding onto one shared edge can each name
    # it, so the combine is the weld's own conflict rule rather than a concatenation.
    edge_tag_list: list[ElementTags] = []
    seam_merged: list[IntArray] = []
    loc2mrg: dict[int, IntArray] = {}
    for bi2, (m, off3) in enumerate(zip(meshes, edge_offs)):
        # a block's local edge j *is* concatenated row off3 + j, so the map onto the
        # merged table is a slice of ``e_new`` -- no detour through the elements
        mine: IntArray = e_new[off3:off3 + m.line_mesh.n_lines]
        loc2mrg[bi2] = mine
        # drop this block's seam names before the renumber -- see the hex rung: a
        # self-join collapses two tagged edges onto one id and ``renumber`` refuses it
        et = m.edge_tags
        if seam_edges is not None and bi2 in seam_edges:
            et = et.select(~np.isin(et.ids,
                                    np.asarray(seam_edges[bi2], dtype=np.int64)))
            seam_merged.append(mine[np.asarray(seam_edges[bi2], dtype=np.int64)])
        edge_tag_list.append(et.renumber(mine))
    named = [(loc2mrg[bi][np.asarray(e, dtype=np.int64)], tag)
             for bi, e, tag in named_seams]
    lm = LineMesh(points, edges, interior=edge_nodes,
                  element_tags=_seam_named(edge_tag_list, seam_merged, named,
                                           edges.shape[0], who))
    return QuadMesh(lm, elem_edges, flip, interior, etags)



def _first_wins(dst: PointArray, idx: IntArray, src: PointArray,
                prefer: BoolArray | None = None) -> None:
    """``dst[idx] = src`` with the **lowest** source index winning each collision.

    The quad rung's copy of :func:`hexmesh._first_wins
    <nekmeshpy.hexmesh.assemble._first_wins>`, and the same ``own=`` rule: a marked row
    beats every unmarked one, and among marked rows the lowest still wins."""
    dst[idx[::-1]] = src[::-1]
    if prefer is not None and prefer.any():
        p: IntArray = np.flatnonzero(prefer)
        dst[idx[p][::-1]] = src[p][::-1]


def _shared_edge_nodes(meshes: Sequence[QuadMesh], e_new: IntArray, swap: BoolArray,
                       n_edges: int, ent_tol: float, who: str,
                       prefer: BoolArray | None = None,
                       check: bool = True) -> PointArray:
    """The shared edge-interior table after a weld -- the quad rung's counterpart of
    :func:`hexmesh._shared_nodes <nekmeshpy.hexmesh.assemble._shared_nodes>`.

    Every block's own table is already conformal, so a weld cannot change any node that
    is not on the seam; it only renumbers the edge that holds it.  This is therefore a
    concatenate plus two fixups: an edge whose row came out reversed reads its nodes
    backwards, and a *fused* edge keeps the survivor's block with the loser's copy
    checked against it rather than the whole mesh being re-verified."""
    src: PointArray = np.concatenate(
        [np.asarray(m.line_mesh.interior, dtype=float) for m in meshes], axis=0)
    if src.size:
        src = np.where(swap[:, None, None], src[:, ::-1, :], src)
    out: PointArray = np.empty((n_edges,) + src.shape[1:], dtype=float)
    _first_wins(out, e_new, src, prefer)
    if check and src.size:
        dup: BoolArray = np.bincount(e_new, minlength=n_edges)[e_new] > 1
        if dup.any() and not np.allclose(src[dup], out[e_new[dup]],
                                         rtol=0.0, atol=ent_tol):
            raise ValueError(
                "%s: non-conforming high-order edge -- the two sides disagree on a "
                "welded shared edge's interior nodes beyond tolerance (%.3e). If they "
                "really are the same interface, state it with attach()."
                % (who, ent_tol))
    return out


def _seam_named(etag_list: Sequence[ElementTags], seam_merged: Sequence[IntArray],
                named: Sequence[tuple[IntArray, str]], n_edges: int,
                who: str) -> ElementTags:
    """The merged edge tags, with the welded-shut seam renamed -- the quad rung's
    counterpart of :func:`hexmesh._seam_named
    <nekmeshpy.hexmesh.assemble._seam_named>`, and the same rule: the seam's rows leave
    both sides before the combine, so the two cannot conflict over a name the caller has
    already given."""
    if not seam_merged:
        return welded_element_tags(list(etag_list), who)
    seam_ids: IntArray = np.unique(np.concatenate(list(seam_merged)))
    kept = [t.select(~np.isin(t.ids, seam_ids)) for t in etag_list]
    merged = welded_element_tags(kept, who)
    if not named:
        return merged
    dense = np.asarray(merged.dense(n_edges), dtype=object)
    for ids, tag in named:
        if tag:
            dense[ids] = tag
    return ElementTags.from_dense(np.asarray(dense, dtype=np.str_))


def _edge_group(mesh: QuadMesh, which: str | IntArray | Sequence[int],
                side: str) -> IntArray:
    """The seam's edge ids on one side, from a tag name or given outright."""
    if isinstance(which, str):
        ids = tagged_edges(mesh, which)
    else:
        ids = np.asarray(which, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= mesh.line_mesh.n_lines):
            raise ValueError(
                "attach: %s names edge %d, outside this section's %d shared edges"
                % (side, int(ids.max()), mesh.line_mesh.n_lines))
    borne = np.bincount(np.asarray(mesh.quads, dtype=np.int64).ravel(),
                        minlength=mesh.line_mesh.n_lines)
    buried = ids[borne[ids] != 1]
    if buried.size:
        raise ValueError(
            "attach: %s names %d edge(s) that are not on this section's boundary (first "
            "is edge %d); joining onto one would make the seam non-manifold."
            % (side, buried.size, int(buried[0])))
    return ids


class Seam(NamedTuple):
    """One stated interface between two sections -- the quad rung's counterpart of
    :class:`hexmesh.Seam <nekmeshpy.hexmesh.assemble.Seam>`, naming **edge** groups.

    ``a`` and ``b`` name a section by its position in ``attach``'s ``meshes`` or by the
    section itself.  ``tag_a`` / ``tag_b`` are edge-tag names or explicit edge-id arrays
    (:func:`tagged_edges <nekmeshpy.quadmesh.query.tagged_edges>`)."""
    a: int | QuadMesh
    tag_a: str | IntArray
    b: int | QuadMesh
    tag_b: str | IntArray
    own: str = "a"
    attach_tag: str | None = None


def _section_index(ref: int | QuadMesh, meshes: Sequence[QuadMesh], who: str) -> int:
    if isinstance(ref, QuadMesh):
        for i, m in enumerate(meshes):
            if m is ref:
                return i
        raise ValueError(
            "attach: %s names a section that is not in the meshes list. Pass the "
            "section itself, or its index." % who)
    i = int(ref)
    if not 0 <= i < len(meshes):
        raise ValueError("attach: %s names section %d of %d" % (who, i, len(meshes)))
    return i


def _pair_edge_seam(a: QuadMesh, ea: IntArray, b: QuadMesh, eb: IntArray,
                    who: str) -> IntArray:
    """``(M,2)`` point pairs across one stated edge seam, proved by bijectivity."""
    la = np.asarray(a.line_mesh.lines, dtype=np.int64)[ea]
    lb = np.asarray(b.line_mesh.lines, dtype=np.int64)[eb]
    pa: IntArray = np.unique(la)
    pb: IntArray = np.unique(lb)
    if pa.size != pb.size:
        raise ValueError(
            "attach: %s joins groups that are not the same curve -- %d edges / %d "
            "points on a, %d / %d on b." % (who, ea.size, pa.size, eb.size, pb.size))
    _dist, loc = cKDTree(b.points[pb]).query(a.points[pa])
    dup = loc.size - np.unique(loc).size
    if dup:
        raise ValueError(
            "attach: %s: the pairing is not one-to-one -- %d of a's %d seam points "
            "share a nearest point on b." % (who, dup, loc.size))
    return np.stack([pa, pb[loc]], axis=1)


def attach(meshes: Sequence[QuadMesh], seams: Sequence[Seam]) -> QuadMesh:
    """Join sections along the edge groups each :class:`Seam` names, in one pass -- the
    quad rung's :func:`hexmesh.attach <nekmeshpy.hexmesh.assemble.attach>`, with the same
    contract one rung down.

    No tolerance: within a named pair of groups the pairing is nearest-neighbour, proved
    by bijectivity rather than by a distance.  The joined edges are cleared of their names
    unless the seam's ``attach_tag`` names them, and ``own`` says whose coordinates the
    seam keeps."""
    meshes = list(meshes)
    seams = list(seams)
    if not meshes:
        raise ValueError("attach: no sections to join")
    if len(meshes) == 1 and not seams:
        return meshes[0]
    order = meshes[0].order
    if any(m.order != order for m in meshes):
        raise ValueError("attach: every section must share the same order, got %s"
                         % sorted({m.order for m in meshes}))

    resolved: list[tuple[int, IntArray, int, IntArray, str, str | None]] = []
    for k, sm in enumerate(seams):
        who = "seams[%d]" % k
        ia = _section_index(sm.a, meshes, who + ".a")
        ib = _section_index(sm.b, meshes, who + ".b")
        if sm.own not in ("a", "b"):
            raise ValueError("attach: %s.own must be 'a' or 'b', got %r" % (who, sm.own))
        ea = _edge_group(meshes[ia], sm.tag_a, who + ".tag_a")
        eb = _edge_group(meshes[ib], sm.tag_b, who + ".tag_b")
        if ea.size != eb.size:
            raise ValueError(
                "attach: %s joins groups of different edge counts (%d and %d), so they "
                "cannot be the same interface." % (who, ea.size, eb.size))
        if ea.size == 0:
            raise ValueError("attach: %s names empty groups; there is nothing to join"
                             % who)
        resolved.append((ia, ea, ib, eb, sm.own, sm.attach_tag))

    # Nothing is copied: the weld keeps exactly one point per fused pair anyway, so
    # making the two sides agree beforehand only decided *whose* coordinate that was --
    # which ``own=`` now says directly, as ``keep`` for the points and ``own_edges``
    # for the shared high-order nodes.
    offs: IntArray = np.concatenate(
        [[0], np.cumsum([m.n_points for m in meshes])]).astype(np.int64)
    cat: list[IntArray] = []
    keep: list[IntArray] = []
    own_edges: dict[int, list[IntArray]] = {}
    for k, (ia, ea, ib, eb, own, _tag) in enumerate(resolved):
        pairs = _pair_edge_seam(meshes[ia], ea, meshes[ib], eb, "seams[%d]" % k)
        cat.append(np.stack([pairs[:, 0] + offs[ia], pairs[:, 1] + offs[ib]], axis=1))
        side, blk, loc = ((0, ia, ea) if own == "a" else (1, ib, eb))
        keep.append(pairs[:, side] + offs[blk])
        own_edges.setdefault(blk, []).append(loc)

    stated: IntArray = (np.concatenate(cat, axis=0) if cat
                        else np.zeros((0, 2), dtype=np.int64))
    kept: IntArray = (np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64))
    points, point_id = conform.weld_pairs([m.points for m in meshes], stated, kept)

    seam_edges: dict[int, list[IntArray]] = {}
    for ia, ea, ib, eb, _o, _t in resolved:
        seam_edges.setdefault(ia, []).append(ea)
        seam_edges.setdefault(ib, []).append(eb)
    named = [(ia, ea, tag) for ia, ea, _ib, _eb, _o, tag in resolved if tag]
    return _stitch(meshes, points, point_id, who="quadmesh.attach",
                   seam_edges={b: np.unique(np.concatenate(v))
                               for b, v in seam_edges.items()},
                   named_seams=named,
                   own_edges={b: np.unique(np.concatenate(v))
                              for b, v in own_edges.items()},
                   check=False)


def _subset(mesh: QuadMesh, keep: BoolArray) -> tuple[QuadMesh, IntArray]:
    """``(the kept quads as a QuadMesh, new_quad_of)`` -- the quad rung of
    :func:`linemesh._subset <nekmeshpy.linemesh.assemble._subset>`, which it calls to
    carry the shared edges down.

    Shared edges no kept quad references are dropped with the points under them, so the
    B-rep comes back complete rather than trailing orphaned entities.  Order is
    preserved: the shared edge-interior and private per-quad nodes ride along."""
    kept, new_quad_of = conform.renumber_map(keep)
    quad: IntArray = mesh.quads[kept]
    edge_keep: BoolArray = np.zeros(mesh.line_mesh.n_lines, dtype=bool)
    if quad.size:
        edge_keep[np.unique(quad)] = True
    sub_lines, new_edge_of = line_subset(mesh.line_mesh, edge_keep)
    # the edge tags ride ``sub_lines`` -- ``line_subset`` already carried them onto
    # the compacted edge numbering, which is the whole of it
    return (QuadMesh(sub_lines, new_edge_of[quad], mesh.orient[kept],
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
    n, labels = conform.element_components(mesh.corners, mesh.n_points)
    return [_subset(mesh, labels == c)[0] for c in range(n)]


def _refine_parts(mesh: QuadMesh) -> tuple[LineMesh, PointArray, IntArray, PointArray]:
    """The pieces of :func:`refine` from before its own :func:`entities_from_blocks
    <nekmeshpy.quadmesh._helpers.entities_from_blocks>` call: ``(refined_line, points,
    flat_corners, flat_blocks)`` -- ``points`` already includes every quad's new
    center, and ``flat_corners``/``flat_blocks`` are the ``4*n_quads`` sub-quads' own
    corner ids and curved blocks, in ``4*q+k`` order.

    Split out so :func:`hexmesh.refine <nekmeshpy.hexmesh.assemble.refine>` can fold
    its own new (hex-interior) quads into the *same* ``entities_from_blocks`` call
    this makes -- an edge on the boundary between an original quad's own refinement
    and a hex's new interior split (a face-center-to-edge-midpoint spoke, which is
    both at once) must be deduplicated in one pass, not stitched after the fact."""
    order = mesh.order
    refined_line = line_refine(mesh.line_mesh)
    n0_line = mesh.line_mesh.n_points
    blocks = element_blocks(mesh)                          # (Q, (order+1)**2, 3)
    q_count = blocks.shape[0]

    centers = resample_block_at(
        blocks, order, [np.array([0.5]), np.array([0.5])], 2)[:, 0, :]
    points = np.concatenate([refined_line.points, centers])
    center_id = np.arange(refined_line.n_points, refined_line.n_points + q_count,
                          dtype=np.int64)

    c = mesh.corners                                        # (Q,4) point ids, CCW
    mid = n0_line + mesh.quads                              # (Q,4) side-midpoint ids

    # quadrant k's 4 corners, CCW, matching _CORNER_IJK[2] = [(0,0),(1,0),(1,1),(0,1)]
    subcorners = np.stack([
        np.stack([c[:, 0], mid[:, 0], center_id, mid[:, 3]], axis=1),
        np.stack([mid[:, 0], c[:, 1], mid[:, 1], center_id], axis=1),
        np.stack([center_id, mid[:, 1], c[:, 2], mid[:, 2]], axis=1),
        np.stack([mid[:, 3], center_id, mid[:, 2], c[:, 3]], axis=1),
    ], axis=0)                                              # (4,Q,4)

    g = gll_nodes(order)
    row = (order + 1) ** 2
    subblocks = np.empty((4, q_count, row, 3), dtype=float)
    for k, (bu, bv) in enumerate(_CORNER_IJK[2]):
        subblocks[k] = resample_block_at(
            blocks, order, [0.5 * g + 0.5 * bu, 0.5 * g + 0.5 * bv], 2)

    flat_corners = subcorners.transpose(1, 0, 2).reshape(4 * q_count, 4)
    flat_blocks = subblocks.transpose(1, 0, 2, 3).reshape(4 * q_count, row, 3)
    return refined_line, points, flat_corners, flat_blocks


def refine(mesh: QuadMesh) -> QuadMesh:
    """Uniform H-refinement: split every quad into 4 -- its own true center point plus
    the edge midpoints :func:`linemesh.refine <nekmeshpy.linemesh.assemble.refine>`
    already put on its shared ``line_mesh``.

    Exact at any order, the same way ``linemesh.refine`` is: the new center, and every
    child's own curved interior, are read off the parent's *stored* polynomial map via
    the internal ``core.interp`` order-N kernel, not a bilinear guess through the
    corners. An edge shared between two quads is refined
    through the one shared ``line_mesh`` exactly once, so both neighbours land on the
    identical midpoint automatically -- no coincidence weld needed for that; the new
    per-quad center points are then welded/deduped implicitly too, since each is
    private to its own quad and never shared.

    Quad ``q``'s new center is point id ``linemesh.refine(mesh.line_mesh).n_points +
    q``. Child ``4*q + k`` (``k`` = 0..3, the ``_CORNER_IJK[2][k]`` corner) is the
    quadrant nearer corner ``k``. One call is one level."""
    order = mesh.order
    refined_line, points, flat_corners, flat_blocks = _refine_parts(mesh)
    q_count = mesh.n_quads

    lm, elem_edges, flip, interior = entities_from_blocks(
        flat_blocks, flat_corners, points, order, "quadmesh.refine")

    if len(refined_line.element_tags):
        match = conform.locate_rows(lm.lines, refined_line.lines,
                                    who="quadmesh.refine", what="refined edge")
        lm = LineMesh(lm.points, lm.lines, lm.interior,
                     refined_line.element_tags.renumber(match))

    # each parent's tag propagates to all 4 children, consecutively -- see
    # linemesh.refine's own note on why this is ``gather``, not ``repeat_blocks``.
    element_tags = mesh.element_tags.gather(np.repeat(np.arange(q_count), 4))
    return QuadMesh(lm, elem_edges, flip, interior, element_tags)


__all__ = [
    "Seam",
    "attach",
    "components",
    "loft",
    "loft_fn",
    "loft_spline",
    "merge",
    "refine",
    "remove",
    "select",
]
