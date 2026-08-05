"""Variable-arity ``QuadMesh`` operations -- the only ones that build a numbering.

``loft`` (``n`` line profiles -> a section, rung delta +1), ``loft_fn`` (the same,
with the profiles evaluated from a parametrization rather than handed in) and ``merge``
(``n`` sections -> one, rung delta 0) are the n-ary operations at this rung, and the
only code here
that manufactures a global point/element index space from scratch: ``loft`` numbers the
layer-by-layer B-rep it assembles (``prof_off`` / ``rung_off``, global id
``i*nn + v``), ``merge`` builds the ``remap`` / ``survivors`` / ``point_id`` tables of
the weld.  Every fixed-arity operation either reuses an existing numbering (``blend``)
or delegates here (``extrude``, ``from_grid``, the region factories).

``loft_fn`` lives here rather than with the region fills for the same reason its
line-rung twin does -- it *is* ``loft``, and it delegates the whole assembly to it
through ``sweep_nodes``, contributing only the evaluation.

They also differ in what they owe the B-rep: ``loft`` *generates* topology, so no
shared entity is ever duplicated and nothing needs reconciling; ``merge`` *rewrites* it
against a new corner numbering, so it must re-scatter the shared edge nodes owner-wins
and verify every other incident copy.

Free functions bound onto :class:`~nekmeshpy.QuadMesh` by ``quadmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``QuadMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..linemesh import LineMesh
from ..linemesh._assemble import _check_fraction_count, _refined_lattice, _weld
from ..model import conform
from ..model.conform import entity_tol
from ..model.fields import gll_nodes, reject_loop_caps
from ..model.tags import (
    EdgeTags,
    ElementTags,
    TagBuilder,
)
from ._query import _boundary_mask
from .quadmesh import (
    NO_TAG,
    QuadMesh,
    _coons_at,
    _quad_interior_slots,
)


def loft(
    slices: Sequence[LineMesh],
    *,
    loop: bool = False,
    sweep_nodes: Sequence[Sequence[LineMesh]] | None = None,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> QuadMesh:
    """Loft a stack of conformal ``LineMesh`` profiles into a quad section
    (the general primitive behind :meth:`extrude`, and the middle rung of the
    uniform sweep shared with
    :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>` and
    :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`).

    ``slices`` is ``nz+1`` line profiles sharing the same ``lines``,
    ``element_tags``, and ``edge_tags``; consecutive profiles form ``nz`` quad
    layers.  For line ``(a, b)`` at layer ``i`` the column quad is
    ``[a_i, b_i, b_{i+1}, a_{i+1}]``.  The line's ``element_tags`` ride onto every
    quad in its column and tagged boundary points onto the swept wall edges;
    ``first_tag`` / ``last_tag`` name the near / far cap edges (scalar or per-line
    array).  ``element_tags`` is the orthogonal, **per-layer** dense tag array
    (length ``nz``, ``""`` = untagged): where a layer's tag is non-empty it
    *overrides* the profile line tag on every quad of that layer, following the
    toolkit's upper-overrides-lower rule.  Left ``None`` the profile tags stand
    alone, which is the historical behaviour.

    **The sweep is straight unless you say otherwise.**  With only the corner-level
    profiles to go on, each column quad is a transfinite patch curved along the
    profile (from the slices' own nodes) and *straight along the sweep*, so at
    ``order > 1`` a swept curved surface is high-order in storage and linear in
    geometry between consecutive slices -- exact input rings do not save it.
    ``sweep_nodes`` is the escape hatch, and the exact analogue of
    :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>`'s ``interior``
    override one rung down: ``sweep_nodes[i]`` is the ``order-1`` profiles lying
    strictly *between* slice ``i`` and the slice it sweeps to, at that layer's
    interior GLL levels.  Supplied, they replace both interpolations -- the rung
    lines' interiors and the quads' private interiors are then read straight out of
    them, so every node is a genuine profile point and nothing is blended.
    :meth:`loft_fn` builds them by evaluating a parametrization on the refined
    sweep lattice; that is the intended way in.

    ``loop=True`` makes the sweep **periodic**: the last profile is joined back to
    the *first*, so ``M`` profiles give ``M`` layers instead of ``M-1``.  It falls
    out of the layer-by-layer assembly as one extra iteration -- exactly one more
    rung block, appended once at the end, with the first profile's lines *not*
    duplicated -- so the seam is a genuine shared entity and the closed tube has
    no free boundary edge in the sweep direction.  A closed sweep has no near/far
    cap, so ``first_tag`` / ``last_tag`` with ``loop=True`` raise ``ValueError``
    rather than being silently dropped, and no cap boundary row is emitted.
    (Side-wall edge tags derived from the profiles' own tagged end points are
    unaffected.)

    The section's B-rep is assembled **layer by layer**, never re-derived from
    corner connectivity: each profile's own ``LineMesh`` is merged into the growing
    global edge mesh (its ``lines`` and their private ``interior`` nodes carry
    through untouched), then the *rung* lines joining that level's points to the
    previous level's are appended (their ``interior`` is the straight GLL blend
    between the two corner points).  A quad is then just its four line indices plus
    four ``flip`` bits -- no ``unique_edges`` dedupe and no owner-wins
    ``scatter_edge_nodes`` reconciliation, because no shared edge is ever
    duplicated in the first place."""
    slices = list(slices)
    if loop:
        reject_loop_caps("QuadMesh.loft", first_tag, last_tag)
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
    # only points actually carried by a profile line get a rung (an isolated point
    # borders no quad, so a rung there would be a dangling edge).
    used: IntArray = (np.unique(lines.ravel()) if L
                      else np.zeros(0, dtype=np.int64))
    nu = used.shape[0]
    rung_slot: IntArray = np.full(nn, -1, dtype=np.int64)
    rung_slot[used] = np.arange(nu, dtype=np.int64)

    # -- grow the global edge LineMesh layer by layer -------------------
    # level i's profile lines occupy [prof_off[i], prof_off[i]+L); the rungs of
    # transition i->next(i) occupy [rung_off[i], rung_off[i]+nu).  ``nxt`` is the
    # profile each layer sweeps *to*: i+1 normally, wrapping to 0 for the closing
    # layer of a periodic sweep -- the single extra iteration below.
    nxt: IntArray = (np.arange(1, nz + 1, dtype=np.int64) % n_prof if nz
                     else np.zeros(0, dtype=np.int64))
    gi: FloatArray = gll_nodes(order)[1:order] if ho else np.zeros(0)
    prof_off: IntArray = np.empty(n_prof, dtype=np.int64)
    rung_off: IntArray = np.empty(max(nz, 0), dtype=np.int64)
    line_rows: list[IntArray] = []
    inter_rows: list[PointArray] = []
    cur = 0

    def _append_rung(layer: int) -> None:
        """Append the rung block of layer ``layer`` (profile ``layer`` ->
        ``nxt[layer]``) to the growing edge mesh -- appended exactly once per
        layer, so the seam rung of a periodic sweep is never duplicated."""
        nonlocal cur
        j = int(nxt[layer])
        rung_off[layer] = cur
        line_rows.append(np.column_stack([layer * nn + used, j * nn + used]))
        if ho:
            if sw is not None:
                # the true profile points at this layer's interior GLL levels --
                # no blend, so a curved sweep survives.
                inter_rows.append(np.stack(
                    [np.asarray(m.points, dtype=float).reshape(-1, 3)[used]
                     for m in sw[layer]], axis=1))          # (nu, order-1, 3)
            else:
                lo, hi = S[layer, used, :], S[j, used, :]
                inter_rows.append(lo[:, None, :]
                                  + gi[None, :, None] * (hi - lo)[:, None, :])
        cur += nu

    for i in range(n_prof):
        prof_off[i] = cur                                # merge level i's LineMesh
        line_rows.append(lines + i * nn)
        if ho:
            inter_rows.append(np.asarray(slices[i].interior, dtype=float))
        cur += L
        if i:                                            # rungs (i-1) -> i
            _append_rung(i - 1)
    if loop and nz:
        # the one extra iteration: the closing rung (M-1) -> 0.  No profile is
        # re-appended, so the seam shares profile 0's lines and nodes outright.
        _append_rung(nz - 1)
    all_lines: IntArray = (np.concatenate(line_rows, axis=0) if line_rows
                           else np.zeros((0, 2), dtype=np.int64))
    edge_nodes: PointArray | None = (
        np.concatenate(inter_rows, axis=0) if ho else None)

    # -- quads purely by line index -------------------------------------
    # quad row for (layer i, line l) = i*L + l; corners [a_i, b_i, b_{i+1}, a_{i+1}].
    # local edge 1 = level-i profile line l (forward), 2 = rung at b (forward),
    # 3 = level-(i+1) profile line l (reversed), 4 = rung at a (reversed).
    i_idx: IntArray = np.repeat(np.arange(nz, dtype=np.int64), L)
    l_idx = np.tile(np.arange(L, dtype=np.int64), nz)
    j_idx: IntArray = nxt[i_idx] if nz else i_idx        # the swept-to profile
    av = a[l_idx]
    bv = b[l_idx]
    quad: IntArray = np.stack([
        prof_off[i_idx] + l_idx,
        rung_off[i_idx] + rung_slot[bv],
        prof_off[j_idx] + l_idx,
        rung_off[i_idx] + rung_slot[av]], axis=1)
    flip: BoolArray = np.tile(
        np.array([False, False, True, True]), (nz * L, 1))
    # quad ``i*L + l`` inherits profile line ``l``'s tag
    etags: ElementTags = slices[0].element_tags.gather(l_idx)
    if element_tags is not None:
        # the orthogonal (per-layer) tag axis: a non-empty layer tag overrides the
        # profile's line tag on every quad of that layer.
        layer_tags: StrArray = np.asarray(
            element_tags, dtype=np.str_).reshape(-1)
        if layer_tags.shape[0] != nz:
            raise ValueError(
                "loft: element_tags is per layer, so it needs %d entries, got %d"
                % (nz, layer_tags.shape[0]))
        etags = etags.overlay(ElementTags.blocks(layer_tags, L))

    # order-N: without ``sweep_nodes`` each column quad is a transfinite (Coons)
    # patch -- curved along the profile line (from the slices' own points + private
    # interior nodes), straight along the sweep between consecutive slices.  Its
    # boundary already lives on the merged/rung lines, so the patch is evaluated only
    # at the private interior slots.
    interior: PointArray | None = None
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
        else:
            bottom = Scur[i_idx, l_idx]                 # (Q,row,3) a->b at i
            top = Scur[j_idx, l_idx]                    # (Q,row,3) a->b at next(i)
            a_lo, a_hi = S[i_idx, av], S[j_idx, av]     # (Q,3) sweep at a
            b_lo, b_hi = S[i_idx, bv], S[j_idx, bv]     # (Q,3) sweep at b
            gg = g[None, :, None]
            left = a_lo[:, None, :] + gg * (a_hi - a_lo)[:, None, :]   # (Q,row,3)
            right = b_lo[:, None, :] + gg * (b_hi - b_lo)[:, None, :]
            interior = _coons_at(bottom, top, left, right, g,
                                 islots % row, islots // row)

    # tagged boundary point -> swept wall edge: vertex 0 -> side 4, vertex 1 -> 2
    bb = TagBuilder(EdgeTags)
    for l0, side, tag in slices[0].point_tags:
        if tag == NO_TAG:
            continue
        qside = 4 if side == 1 else 2
        for ii in range(nz):
            bb.add(ii * L + l0, qside, tag)
    # caps: scalar tags the whole cap, an array tags per section line.  A periodic
    # sweep has no near/far cap edge at all, so it emits no cap row (and rejected
    # the tags above).
    if not loop:
        first_caps = QuadMesh._cap_tags(first_tag, L)
        last_caps = QuadMesh._cap_tags(last_tag, L)
        for l0 in range(L):
            bb.add_if_tagged(l0, 1, first_caps[l0])
        if nz:
            for l0 in range(L):
                bb.add_if_tagged((nz - 1) * L + l0, 3, last_caps[l0])
    lm = LineMesh(points, all_lines, order=order, interior=edge_nodes)
    return QuadMesh(lm, quad, flip, interior, bb.build_ordered(), etags,
                    order=order)


def _loft_evaluated(
    profs: Sequence[LineMesh],
    t: FloatArray,
    order: int,
    *,
    loop: bool = False,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
    name: str = "loft_fn",
) -> QuadMesh:
    """The shared tail of every sweep whose profiles are **evaluated** on the refined
    node lattice rather than handed in: validate, close the loop, split, delegate.

    The quad-rung twin of ``hexmesh._assemble._loft_evaluated``.  ``profs`` is one
    profile per entry of the sweep lattice ``t`` (``nz*order + 1`` of them; ``t`` is in
    the caller's own parameter units and is used only in error messages);
    ``profs[i*order]`` are the corner-level slices and the ``order-1`` profiles between
    consecutive ones are that layer's :func:`loft` ``sweep_nodes``.  Every profile must
    be index-paired and conformal with the first, and with ``loop=True`` the trailing
    wrap profile must reproduce the first point-for-point
    (``model.conform.entity_tol``) before it is dropped -- its layer's intermediates
    stay, since they are what curve the seam.  ``name`` is the caller's name for the
    error messages.

    Factored out of :func:`loft_fn` so a sweep that already *has* the profile list
    (rather than a callable to evaluate) reuses the same contract instead of restating
    it -- which is exactly what ``QuadMesh.sweep`` needs, because a
    rotation-minimizing frame is integrated along the whole path and cannot be
    evaluated at one isolated parameter value."""
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
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> QuadMesh:
    """Loft a section from a **parametrized family of profiles** -- :func:`loft` with
    the slices evaluated rather than handed in, so **every** node (the corners *and*
    the sweep-direction high-order nodes) comes from calling ``f`` and nothing is
    blended along the sweep.

    This is the quad rung of
    :func:`linemesh.assemble.loft_fn <nekmeshpy.linemesh.assemble.loft_fn>`, and it
    exists for the same reason: a plain :func:`loft` has only the corner-level
    profiles to go on, so at ``order > 1`` it subdivides the sweep straight and a
    swept curved surface ends up high-order in storage and linear in geometry (a
    torus lofted from *exact* rings puts its interior nodes tens of percent of the
    tube radius off the true surface).  Evaluating the profiles at the intermediate
    GLL levels too is what closes that.

    ``f`` maps a **single parameter value** to that profile as a ``LineMesh`` and is
    called once per node level -- ``n*order + 1`` times, or ``n*order`` when looping.
    Every profile it returns must be index-paired and conformal with the first: same
    ``lines``, same point count, same ``order``.  The robust idiom is to build one
    profile and *place* it with the affine ops (``ring.rotate(t, axis=...)
    .translate(...)``), which move no index and are exact; re-deriving it from a
    factory whose orientation depends on ``t`` can silently renumber it.

    **Why ``f`` is scalar here and ``sweep``'s ``path`` is vectorized.**  ``f`` returns
    a *mesh*, and a callable that hands back one mesh can only be given one parameter
    value -- there is no array-of-meshes for a vectorized form to return (which is also
    why :func:`linemesh.assemble.loft_fn <nekmeshpy.linemesh.assemble.loft_fn>`, whose
    ``f`` returns plain coordinates, *is* vectorized).  :func:`sweep`'s ``path`` is
    vectorized for a different reason again: it returns coordinates *and* the default
    frame generator is a sequential integration along the whole curve, so it cannot be
    evaluated at one isolated parameter at all.

    ``order`` defaults to ``None`` = **inferred**: ``f`` is called once at
    ``fractions[0]`` purely to read ``.order`` off the profile it returns, and the real
    sweep then proceeds as usual -- so a ``None`` order costs exactly one extra
    evaluation of ``f`` (the lattice the profiles are sampled on depends on the order,
    so it cannot be built before the order is known).  Pass an explicit ``int`` to
    assert it instead: a profile of a different order is then a ``ValueError`` naming
    the mismatch, and no probe call is made.  There is no inference at the line rung --
    ``LineMesh.loft_fn``'s ``f`` returns points, not a mesh, so its ``order`` is
    constructive rather than inherited.

    ``fractions`` are the **parameter values themselves**, passed to ``f`` with no
    normalization and no remapping -- the same contract as at the line rung -- so
    ``len(fractions) - 1`` is the layer count and the grading is honored *per layer*:
    layer ``i``'s sweep-direction nodes ride the GLL nodes of its own
    ``fractions[i] .. fractions[i+1]`` span.

    ``loop=True`` makes the sweep periodic and takes the **trailing wrap value**, as
    at the line rung: pass ``n+1`` fractions whose last maps back to the first
    profile and the result has ``n`` layers with the seam a genuine shared entity.
    The wrap value stays in because it is what gives the seam layer a far parameter
    to evaluate its interior levels at.  That ``f(fractions[-1])`` really does
    reproduce ``f(fractions[0])`` is checked, not assumed (``ValueError`` otherwise,
    at the scale-relative coincidence tolerance ``model.conform.entity_tol``).  A
    closed sweep has no near/far cap, so ``first_tag`` / ``last_tag`` are rejected.

    ``element_tags`` is the per-layer tag array and the caps the per-line ones,
    exactly as on :func:`loft`, which does all the assembly and whose numbering is
    carried up unchanged."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    _check_fraction_count(fr, loop=loop, name="loft_fn")
    if loop:
        reject_loop_caps("QuadMesh.loft_fn", first_tag, last_tag)
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
    """Merge quad sections into one, welding coincident boundary points.
    ``tol`` is the absolute coincidence distance (default ``1e-7`` x the extent).
    ``edge_tags`` and ``element_tags`` concatenate with each
    block's quad ids offset; an interior seam is not auto-dropped."""
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
    lm = LineMesh(points, edges, order=order, interior=edge_nodes)
    return QuadMesh(lm, elem_edges, flip, interior, bnd, etags, order=order)


#: Variable-arity combinators bound onto ``QuadMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "loft": loft,
    "loft_fn": loft_fn,
    "merge": merge,
}
