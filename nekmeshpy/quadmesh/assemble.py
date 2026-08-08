"""Variable-arity ``QuadMesh`` operations -- the only ones that build a numbering."""

from __future__ import annotations

from collections.abc import Callable, Sequence

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
    EdgeTags,
    ElementTags,
    TagBuilder,
    sweep_cap_tags,
    sweep_element_tags,
)
from .quadmesh import (
    NO_TAG,
    QuadMesh,
    _coons_at,
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
    lines = np.asarray(slices[0].lines, dtype=np.int64).reshape(-1, 2)
    L = lines.shape[0]
    n_prof = len(slices)
    # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
    nz = n_prof if loop else n_prof - 1
    S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                  for s in slices], axis=0)              # (n_prof, nn, 3)
    nn = S.shape[1]
    points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v

    order = slices[0].order
    if any(s.order != order for s in slices):
        raise ValueError("loft: all slices must share the same order")
    ho = order > 1

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
                if m.order != order or m.n_points != nn:
                    raise ValueError(
                        "loft: sweep_nodes[%d] profiles must match the slices "
                        "(order %d, %d points), got order %d with %d points"
                        % (i, order, nn, m.order, m.n_points))
        if not ho:
            sw = None                      # order 1 has no interior level at all

    a = lines[:, 0]
    b = lines[:, 1]
    # a lone profile bounds no layer, so it carries no entity either
    sweep = Sweep(n_prof if nz else 0, nz, nn, loop=loop)
    nxt = sweep.nxt
    i_idx, l_idx, j_idx = sweep.elements(L)
    av = a[l_idx]
    bv = b[l_idx]

    # Only points a line actually carries get a rung -- an isolated point borders no
    # quad, so a rung there would be a dangling edge.
    used: IntArray = (np.unique(lines.ravel()) if L else np.zeros(0, np.int64))
    nu = used.shape[0]
    slot: IntArray = np.full(nn, -1, np.int64)
    slot[used] = np.arange(nu, dtype=np.int64)

    # The two entity families of any sweep, at this rung: each profile line *carried*
    # to a level, and each profile point *swept* across a layer.  Both are closed
    # forms, so the edge table is written rather than deduplicated.
    lay: IntArray = np.arange(nz, dtype=np.int64)
    lvl: IntArray = np.arange(sweep.n_levels, dtype=np.int64)
    edges: IntArray = np.concatenate([
        (lines[None] + (lvl * nn)[:, None, None]).reshape(-1, 2),
        np.stack([sweep.point(lay[:, None], used[None, :]).ravel(),
                  sweep.point(nxt[:, None], used[None, :]).ravel()], axis=1)], axis=0)
    # quad ``i*L + l`` is profile line ``l`` dragged across layer ``i``: corners
    # [a_i, b_i, b_j, a_j], local edges [carried at i, rung at b, carried at j, rung
    # at a].  Sides 3 / 4 traverse their stored edge backwards, hence the flip row.
    quad: IntArray = np.stack([
        sweep.carried(i_idx, l_idx, L),
        sweep.swept(i_idx, slot[bv], L, nu),
        sweep.carried(j_idx, l_idx, L),
        sweep.swept(i_idx, slot[av], L, nu)], axis=1)
    flip: BoolArray = np.tile(np.array([False, False, True, True]), (nz * L, 1))

    etags = sweep_element_tags(element_tags, nz, L, "QuadMesh.loft")

    # order-N: without ``sweep_nodes`` each column quad is a transfinite (Coons)
    # patch -- curved along the profile line (from the slices' own points + private
    # interior nodes), straight along the sweep between consecutive slices.  Its
    # boundary already lives on the merged/rung lines, so the patch is evaluated only
    # at the private interior slots.
    interior: PointArray | None = None
    edge_nodes: PointArray | None = None
    if ho:
        g = gll_nodes(order)
        row = order + 1

        def _line_nodes(m: LineMesh) -> PointArray:
            """That profile's ``(L, order+1, 3)`` per-line node curve, assembled
            natively from the shared corner points and its private interiors."""
            P: PointArray = np.asarray(m.points, dtype=float).reshape(-1, 3)
            out: PointArray = np.empty((L, row, 3), dtype=float)
            out[:, 0, :] = P[a]
            out[:, order, :] = P[b]
            out[:, 1:order, :] = np.asarray(m.interior, dtype=float)
            return out

        # each profile's own high-order curve, assembled natively from the shared
        # corner points and that profile's private interior nodes
        Scur: PointArray = np.empty((n_prof, L, row, 3), dtype=float)
        Scur[:, :, 0, :] = S[:, a, :]
        Scur[:, :, order, :] = S[:, b, :]
        Scur[:, :, 1:order, :] = np.stack(
            [np.asarray(s.interior, dtype=float) for s in slices], axis=0)
        islots = _quad_interior_slots(order)
        # a shared edge has exactly one source in a closed-form sweep, so both families
        # are *written* -- no element block to read them back out of, and no owner
        # election.  A carried edge is the profile's own curve verbatim; a rung is the
        # straight GLL blend of its two corners, or the true intermediate profiles'
        # own points when there are any.
        Sp: PointArray = S[:, used, :]                  # (n_prof, nu, 3)
        carried: PointArray = Scur[lvl][:, :, 1:order, :]
        if sw is not None:
            # every node of every sweep level is a genuine profile point, so the
            # element block is a straight gather out of them -- no patch, nothing
            # interpolated in either direction.
            lev: PointArray = np.empty((nz, row, L, row, 3), dtype=float)
            if nz:
                lev[:, 0] = Scur[np.arange(nz, dtype=np.int64)]
                lev[:, order] = Scur[nxt]
                for k in range(1, order):
                    lev[:, k] = np.stack(
                        [_line_nodes(sw[i][k - 1]) for i in range(nz)], axis=0)
            interior = lev[i_idx[:, None], (islots // row)[None, :],
                           l_idx[:, None], (islots % row)[None, :], :]
            rungs: PointArray = np.stack(
                [np.stack([np.asarray(sw[i][k - 1].points, dtype=float)
                           .reshape(-1, 3)[used] for k in range(1, order)], axis=1)
                 for i in range(nz)], axis=0) if nz else np.zeros((0, nu, order - 1, 3))
        else:
            bottom = Scur[i_idx, l_idx]                 # (Q,row,3) a->b at i
            top = Scur[j_idx, l_idx]                    # (Q,row,3) a->b at next(i)
            a_lo, a_hi = S[i_idx, av], S[j_idx, av]     # (Q,3) sweep at a
            b_lo, b_hi = S[i_idx, bv], S[j_idx, bv]     # (Q,3) sweep at b
            gg = g[None, :, None]
            left = a_lo[:, None, :] + gg * (a_hi - a_lo)[:, None, :]   # (Q,row,3)
            right = b_lo[:, None, :] + gg * (b_hi - b_lo)[:, None, :]
            # only the private interior is blended; the profiles' and rungs' own nodes
            # bound the patch verbatim, so nothing is resampled.
            interior = _coons_at(bottom, top, left, right, g,
                                 islots % row, islots // row)
            lo, hi = Sp[lay], Sp[nxt]                   # (nz,nu,3) the rung's two ends
            rungs = (lo[:, :, None, :]
                     + g[1:order][None, None, :, None] * (hi - lo)[:, :, None, :])
        edge_nodes = np.concatenate(
            [carried.reshape(-1, order - 1, 3), rungs.reshape(-1, order - 1, 3)], axis=0)

    # tagged boundary point -> swept wall edge: vertex 0 -> side 4, vertex 1 -> 2
    bb = TagBuilder(EdgeTags)
    for l0, side, tag in slices[0].point_tags:
        if tag == NO_TAG:
            continue
        qside = 4 if side == 1 else 2
        for ii in range(nz):
            bb.add(ii * L + l0, qside, tag)
    # a cap edge *is* a profile line, so with no argument it inherits that line's own
    # element tag -- except on a closed sweep, where the "caps" are the interior seam
    # and only an explicit tag names them.
    closed = ElementTags.empty()
    first_caps = sweep_cap_tags(first_tag, closed if loop else slices[0].element_tags,
                                L, "QuadMesh.loft")
    last_caps = sweep_cap_tags(last_tag, closed if loop else slices[-1].element_tags,
                               L, "QuadMesh.loft")
    for l0 in range(L):
        bb.add_if_tagged(l0, 1, first_caps[l0])
    if nz:
        for l0 in range(L):
            bb.add_if_tagged((nz - 1) * L + l0, 3, last_caps[l0])
    lm = LineMesh(points, edges, interior=edge_nodes)
    return QuadMesh(lm, quad, flip, interior, bb.build_ordered(), etags)


def _loft_evaluated(
    profs: Sequence[LineMesh],
    t: FloatArray,
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
    profs = list(profs)
    if len(profs) != t.shape[0]:
        raise ValueError(
            "%s: expected one profile per sweep lattice value (%d), got %d"
            % (name, t.shape[0], len(profs)))
    nz = (len(profs) - 1) // order
    ref = profs[0]
    ref_lines = np.asarray(ref.lines, dtype=np.int64).reshape(-1, 2)
    for k, m in enumerate(profs):
        if m.order != order:
            raise ValueError(
                "%s: f(%g) returned an order-%d profile, but order=%d was "
                "requested" % (name, t[k], m.order, order))
        if (m.n_points != ref.n_points
                or not np.array_equal(
                    np.asarray(m.lines, dtype=np.int64).reshape(-1, 2), ref_lines)):
            raise ValueError(
                "%s: every profile must be index-paired and conformal with "
                "the first, but f(%g) returned %d points / %d lines against f(%g)'s "
                "%d / %d.  Place one profile with the affine ops rather than "
                "rebuilding it per parameter."
                % (name, t[k], m.n_points, m.n_lines, t[0], ref.n_points,
                   ref.n_lines))

    if loop:
        # the wrap level must land back on level 0; drop it, but keep the seam
        # layer's own intermediate levels -- they are what curve the seam.
        P0 = np.asarray(profs[0].points, dtype=float).reshape(-1, 3)
        Pw = np.asarray(profs[-1].points, dtype=float).reshape(-1, 3)
        gap = float(np.max(np.linalg.norm(Pw - P0, axis=1))) if P0.size else 0.0
        tol = entity_tol(P0)
        if gap > tol:
            raise ValueError(
                "%s(loop=True) needs the last fraction to map back to the "
                "first profile, but f(%g) and f(%g) are %g apart (tolerance %g).  "
                "Pass the trailing wrap value as the final fraction, or use "
                "loop=False." % (name, t[-1], t[0], gap, tol))
        profs = profs[:-1]

    slices = profs[::order]
    sweep_nodes = [profs[i * order + 1:(i + 1) * order] for i in range(nz)]
    return loft(slices, loop=loop,
                sweep_nodes=sweep_nodes if order > 1 else None,
                element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag)


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
    _check_fraction_count(fr, loop=loop, name="loft_fn")
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

    t: FloatArray = _refined_lattice(fr, order)
    profs: list[LineMesh] = [f(float(v)) for v in t]
    return _loft_evaluated(profs, t, order, loop=loop,
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
    points, point_id = _weld(pos, seams, tol)

    quad_list: list[IntArray] = []
    bnd_list: list[EdgeTags] = []
    etag_list: list[ElementTags] = []
    noff = qoff = 0
    for m, c in zip(meshes, counts):
        quad_list.append(point_id[m.quads + noff])   # local -> welded id
        # ids shift by this block's offset; sides stay local to their element
        etag_list.append(m.element_tags.offset(qoff))
        bnd_list.append(m.edge_tags.offset(qoff))
        noff += c
        qoff += m.n_quads
    quads = np.concatenate(quad_list, axis=0) if quad_list else np.zeros((0, 4), np.int64)
    etags = ElementTags.concat(etag_list)
    bnd = EdgeTags.concat(bnd_list).ordered()

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
    lm = LineMesh(points, edges, interior=edge_nodes)
    return QuadMesh(lm, elem_edges, flip, interior, bnd, etags)

__all__ = [
    "loft",
    "loft_fn",
    "merge",
]
