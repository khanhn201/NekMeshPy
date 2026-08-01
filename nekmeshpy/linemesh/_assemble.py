"""Variable-arity ``LineMesh`` operations -- the only ones that build a numbering.

``loft`` (``n`` points -> a line mesh, rung delta +1), ``loft_curve`` (a callable ->
the same, sampled rather than given) and ``merge`` (``n`` line meshes -> one, rung
delta 0) are the n-ary operations at this rung, and they are the only code here that
manufactures a global point/element index space from scratch: ``loft`` numbers the
chain it authors, ``merge`` builds the ``remap`` / ``survivors`` / ``point_id`` tables
of the weld.  Every fixed-arity operation either reuses an existing numbering
(``blend``) or delegates to one of these two.

``loft_curve`` lives here rather than with the shape factories because it authors
connectivity exactly as ``loft`` does -- it *is* ``loft``, with the profiles evaluated
from a parametrization instead of handed in -- and so takes the same ``loop`` flag.
Being open or closed is therefore not a property of the function: the same ``f`` meshes
either way.

Free functions bound onto :class:`~nekmeshpy.LineMesh` by ``linemesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``LineMesh.<name>`` sugar.
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
from ..model.conform import entity_tol
from ..model.fields import gll_nodes, reject_loop_caps
from ._query import boundary_points
from .linemesh import LineMesh, _as_points


def loft(
    points: PointArray,
    *,
    loop: bool = False,
    interior: PointArray | None = None,
    boundaries: IntArray | None = None,
    boundary_tags: StrArray | Sequence[str] | None = None,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
    order: int = 1,
) -> LineMesh:
    """Loft a stack of point "profiles" into a 1-D mesh -- the bottom rung of the
    uniform sweep primitive shared with
    :meth:`QuadMesh.loft <nekmeshpy.quadmesh.QuadMesh.loft>` and
    :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`.

    One dimension below a quad loft each profile is a **single point**, so the
    rungs joining consecutive profiles *are* the line elements: ``points``
    ``(N,3)`` lofts into the consecutive chain ``[[0,1], ..., [N-2,N-1]]``, and
    ``loop=True`` appends one more rung -- from the last point back to the first
    (``[N-1,0]``) -- closing the curve.  The seam rung is appended exactly once
    and no profile is duplicated, so a lofted loop carries ``N`` points and ``N``
    lines with no degree-1 end.  This is the only method that authors ``lines``:
    a chain is ``loft(points)``, a ring ``loft(points, loop=True)``.

    ``element_tags`` is the dense per-line tag array (line ``m`` = point ``m`` ->
    ``m+1``, and for ``loop=True`` line ``N-1`` = point ``N-1`` -> ``0``);
    ``boundaries`` / ``boundary_tags`` are passed through verbatim.
    ``first_tag`` / ``last_tag`` name the near / far **end points** of the chain
    (the 1-D end caps: line ``0`` side ``1`` and line ``L-1`` side ``2``).  A cap
    here is a single node, so each takes a scalar ``str`` or -- for shape parity
    with :meth:`QuadMesh.loft <nekmeshpy.quadmesh.QuadMesh.loft>` /
    :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`, whose caps carry one tag
    per section line / quad -- a one-element array-like.  A closed sweep has no
    near/far cap, so passing either with ``loop=True`` raises ``ValueError``
    rather than silently dropping it.

    At ``order > 1`` an explicit ``interior`` ``(L, order-1, 3)`` is used as-is
    (that is how ``circle`` stamps true-arc nodes); when it is omitted each line's
    private interior is built here as the **straight GLL blend** between its two
    endpoints, which is exactly what a straight-sided curve wants."""
    pts = _as_points(points)
    n = pts.shape[0]
    if loop:
        reject_loop_caps("LineMesh.loft", first_tag, last_tag)
    idx = np.arange(n, dtype=np.int64)
    if n < 2:
        lines: IntArray = np.zeros((0, 2), dtype=np.int64)
    elif loop:
        lines = np.column_stack([idx, np.roll(idx, -1)])
    else:
        lines = np.column_stack([idx[:-1], idx[1:]])

    bnd = boundaries
    names = boundary_tags
    # a chain's cap is a single end node, so ``_cap_tags`` normalizes to one tag --
    # the rung-1 form of the same scalar-or-per-element argument ``QuadMesh.loft`` /
    # ``HexMesh.loft`` take, so all three rungs accept the same shapes.
    first, last = LineMesh._cap_tags(first_tag)[0], LineMesh._cap_tags(last_tag)[0]
    if first or last:
        rows = [[int(r[0]), int(r[1])]
                for r in np.asarray(bnd if bnd is not None else
                                    np.zeros((0, 2), np.int64),
                                    dtype=np.int64).reshape(-1, 2)]
        tags = [str(t) for t in np.asarray(
            names if names is not None else np.empty(0, dtype=np.str_),
            dtype=np.str_).reshape(-1).tolist()]
        L = lines.shape[0]
        if first and L:
            rows.append([0, 1])
            tags.append(first)
        if last and L:
            rows.append([L - 1, 2])
            tags.append(last)
        bnd, names = LineMesh._order_bnd(rows, tags)

    if order > 1 and interior is None:
        # straight GLL blend between each line's two endpoints -- the same
        # expression the straight-sided factories (``line`` / ``rectangle``) use.
        a: PointArray = pts[lines[:, 0]]
        b: PointArray = pts[lines[:, 1]]
        g = gll_nodes(order)[1:order]              # interior GLL nodes only
        interior = a[:, None, :] + g[None, :, None] * (b - a)[:, None, :]
    return LineMesh(pts, lines, interior, bnd, names, element_tags, order=order)


def _refined_lattice(fractions: FloatArray, order: int) -> FloatArray:
    """The ``n*order + 1`` parameter positions of **every** node of the order-``order``
    chain graded by ``fractions`` (``n = len(fractions) - 1`` elements): element ``i``'s
    node ``a`` sits at ``fr[i] + g[a]*(fr[i+1] - fr[i])`` for the GLL nodes ``g`` on
    ``[0, 1]``, and the chain ends at ``fr[-1]``.

    This is the 1-D twin of :func:`~nekmeshpy.quadmesh._open._refined_params`; the
    grading rides in ``fractions`` rather than being assumed uniform, so each element's
    private interior lands inside that element's own span.  At ``order == 1``
    (``g = [0, 1]``) it is exactly ``fractions``, so the order-1 placement falls out by
    construction rather than by a branch."""
    g: FloatArray = gll_nodes(order)
    fr = fractions
    u: FloatArray = (fr[:-1, None]
                     + g[None, :order] * np.diff(fr)[:, None]).ravel()
    return np.concatenate([u, fr[-1:]])


def _sweep_lattice(fractions: FloatArray, order: int, *, loop: bool,
                   name: str) -> tuple[FloatArray, FloatArray]:
    """Validate a sweep's ``fractions`` and return ``(fr, node lattice)``.

    The shared front half of every evaluated sweep -- :func:`loft_curve` at each rung
    and ``sweep`` at the quad and hex ones -- so the two guards that decide how many
    layers there are read identically wherever a caller meets them.  ``loop`` costs one
    extra fraction because the last is the wrap back onto the first profile rather than
    a level of its own."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    if fr.shape[0] - 1 < 1:
        raise ValueError("%s needs at least 2 fractions (one layer), got %d"
                         % (name, fr.shape[0]))
    if loop and fr.shape[0] - 1 < 2:
        raise ValueError(
            "%s(loop=True) needs at least 3 fractions (two layers), got %d -- the "
            "last one is the wrap back to the first profile, so it is not a level of "
            "its own" % (name, fr.shape[0]))
    return fr, _refined_lattice(fr, order)


def _eval_curve(f: Callable[[FloatArray], PointArray], t: FloatArray) -> PointArray:
    """``f(t)`` as a validated ``(len(t), 3)`` array."""
    P: PointArray = np.asarray(f(t), dtype=float)
    if P.shape != (t.shape[0], 3):
        raise ValueError(
            "curve callable must return (len(t), 3) points; got shape %r for %d "
            "parameters" % (P.shape, t.shape[0]))
    return P


def loft_curve(f: Callable[[FloatArray], PointArray], fractions: float | FloatArray, *,
               loop: bool = False, order: int = 1,
               element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
    """Loft a curve given as its own analytic parametrization -- :func:`loft` with the
    profiles **evaluated** rather than handed in, so **every** node (corners *and* the
    private high-order ``interior``) comes from calling ``f`` and nothing is ever
    placed on a chord.

    ``f`` maps a ``(K,)`` parameter array to ``(K, 3)`` points and is called **once**
    with the whole node lattice (vectorize it; ``np.column_stack`` of the three
    component expressions is the usual shape).

    **Why vectorized here and scalar one rung up.**  ``f`` returns *coordinates*, and a
    whole lattice of them is just a ``(K,3)`` array -- so the natural call is one, on
    the whole lattice, which is also the cheapest.  ``QuadMesh.loft_curve`` /
    ``HexMesh.loft_curve`` instead take a **scalar** ``Callable[[float], LineMesh]`` /
    ``Callable[[float], QuadMesh]``, because a callable returning a *single mesh* can
    only be handed a single parameter value: there is no array-of-meshes to return.
    ``sweep``'s ``path`` is vectorized again, for a third reason -- the default frame
    generator is a sequential integration along the curve and cannot be evaluated at
    one isolated parameter.

    ``order`` (default 1) is **constructive** at this rung, not inherited: ``f`` hands
    back points, not a mesh, so there is nothing to read an order off -- the argument
    is what *decides* the node lattice ``f`` is sampled on.  That is why it stays an
    ``int`` here while ``QuadMesh.loft_curve`` / ``HexMesh.loft_curve`` default it to
    ``None`` and infer it from the profile ``f`` returns.

    ``fractions`` are the **parameter values themselves**, passed to ``f`` with no
    normalization and no remapping: node ``k`` is ``f(fractions[k])``, and there are
    ``len(fractions) - 1`` line elements.  The caller states the domain by choosing the
    values -- for an ``f`` written on ``[0, 1]`` they are exactly the normalized
    fractions the sibling
    :meth:`LineMesh.line <nekmeshpy.linemesh.LineMesh.line>` takes, and an ``f`` written
    on any other interval is sampled in its own units
    (``np.linspace(0.0, np.pi, n + 1)`` for a uniform chain over ``[0, pi]``).  A
    descending sequence runs the curve backwards; nothing here requires ascending order.

    ``loop=True`` closes the curve, exactly as it does for :func:`loft`: the last
    element joins back to the first point, so ``n+1`` fractions give ``n`` points
    *and* ``n`` lines, with no duplicated point and no degree-1 end.  The seam element
    is a real element with its own high-order nodes, and they can only be evaluated if
    its parameter span is known -- so the **trailing wrap value stays in**
    ``fractions``: pass ``np.linspace(0.0, 2*np.pi, n + 1)`` for a ``2*pi``-periodic
    ``f``, and the seam rides ``fractions[-2] .. fractions[-1]`` before its far end is
    welded onto point 0.  That the two ends do coincide is checked, not assumed
    (``ValueError`` otherwise, at the scale-relative coincidence tolerance
    ``model.conform.entity_tol``), because a curve that does not close would otherwise
    be silently kinked shut.

    The values grade the nodes in **parameter** space; for nodes spaced evenly by **arc
    length** pass :meth:`LineMesh.arclength_fractions
    <nekmeshpy.linemesh.LineMesh.arclength_fractions>`, whose chord-length table
    perturbs only *where along* the curve the nodes sit -- every node still lies on the
    curve to machine precision, because it is placed by evaluating ``f`` and never by
    interpolating the table.  At ``order > 1`` the grading is honored **per element**:
    element ``i``'s private ``interior`` rides the GLL nodes of its own
    ``fractions[i] .. fractions[i+1]`` span.

    This is the general sibling of
    :meth:`LineMesh.arc <nekmeshpy.linemesh.LineMesh.arc>`, which is the special case
    ``f = circle`` (kept separate because it can place its nodes without an inversion
    and to the last ulp).  Reach for ``loft_curve`` whenever a curve has a closed form
    that is not a circular arc -- an ellipse, a helix, a cylinder-cylinder intersection
    -- instead of sampling it into an array and calling :func:`loft`, which can only
    subdivide straight between the samples and therefore loses the curve at
    ``order > 1``.  For a curve with **no** closed form (a scanned polyline) there is
    nothing to evaluate; resample it with ``trimesh.ops.resample_polyline`` and accept
    the chord.

    ``element_tags`` (length ``len(fractions) - 1``) tags the line elements at
    construction; ``order`` (default 1 = linear) sets the polynomial order."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    ni = fr.shape[0] - 1
    if ni < 1:
        raise ValueError(
            "loft_curve needs at least 2 fractions (one element), got %d" % fr.shape[0])
    if loop and ni < 2:
        raise ValueError(
            "loft_curve(loop=True) needs at least 3 fractions (two elements), got %d -- "
            "the last one is the wrap back to the first point, so it is not a point of "
            "its own" % fr.shape[0])

    # every node of the chain, corners and interiors alike, as one parameter array --
    # so the interiors ride the true curve instead of ``loft``'s straight chord blend.
    t: FloatArray = _refined_lattice(fr, order)
    P: PointArray = _eval_curve(f, t)
    corners: PointArray = P[::order]
    if loop:
        # the wrap node is the seam element's far end; ``loft(loop=True)`` supplies it
        # from point 0, so it must not also be numbered as a point of its own.
        gap = float(np.linalg.norm(corners[-1] - corners[0]))
        tol = entity_tol(corners)
        if gap > tol:
            raise ValueError(
                "loft_curve(loop=True) needs the last fraction to map back to the "
                "first point, but f(%g) and f(%g) are %g apart (tolerance %g).  Pass "
                "the trailing wrap value -- np.linspace(0, 2*np.pi, n+1) for a "
                "2*pi-periodic f -- so the seam element's own nodes can be evaluated."
                % (fr[-1], fr[0], gap, tol))
        corners = corners[:-1]

    if order == 1:
        return loft(corners, loop=loop, element_tags=element_tags)
    slot: IntArray = (np.arange(ni)[:, None] * order
                      + np.arange(1, order)[None, :])        # (n, order-1)
    return loft(corners, loop=loop, interior=P[slot],
                element_tags=element_tags, order=order)


def merge(meshes: Sequence[LineMesh], *,
          tol: float | None = None) -> LineMesh:
    """Merge line meshes into one, welding coincident **topological end
    points** (the degree-1 chain ends -- the 1-D analogue of the boundary
    vertices ``QuadMesh.merge``/``HexMesh.merge`` weld).  ``tol`` is the
    absolute coincidence distance (default ``1e-7`` x the extent).  Dense
    ``element_tags`` and tagged ``boundaries`` concatenate with each block's
    line ids offset; interior points are never welded.  Closedness is not
    tracked anywhere -- it simply falls out of the welded connectivity: if no
    degree-1 end survives the result *is* a loop, so two shared-endpoint
    ``A1 -> A2`` arcs (reverse one so the traversal doesn't cross) weld at
    ``A1`` and ``A2`` into a single cycle, the clean way to close a ring from
    two half-arcs."""
    meshes = list(meshes)
    pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
    counts = [p.shape[0] for p in pos]
    P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
    total = P.shape[0]

    remap = np.arange(total, dtype=np.int64)
    is_bnd: BoolArray = np.zeros(total, dtype=bool)
    noff = 0
    for m, c in zip(meshes, counts):
        is_bnd[noff + boundary_points(m)] = True
        noff += c
    bidx = np.flatnonzero(is_bnd)
    if bidx.size:
        scl = float(np.max(P.max(axis=0) - P.min(axis=0))) if total else 0.0
        t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
        keys = np.round(P[bidx, :] / t).astype(np.int64)
        _, first_local, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True)
        remap[bidx] = bidx[first_local][inverse.ravel()]

    survivors = np.unique(remap)
    new_id: IntArray = np.empty(total, dtype=np.int64)
    new_id[survivors] = np.arange(survivors.size)
    point_id = new_id[remap]
    points = P[survivors, :]

    line_list: list[IntArray] = []
    bnd_list: list[IntArray] = []
    name_list: list[StrArray] = []
    etag_list: list[StrArray] = []
    noff = loff = 0
    for m, c in zip(meshes, counts):
        line_list.append(point_id[m.lines + noff])   # local -> welded id
        etag_list.append(m.element_tags)
        if m.boundaries.shape[0]:
            b: IntArray = m.boundaries.copy()
            b[:, 0] += loff                          # shift line ids; sides local
            bnd_list.append(b)
            name_list.append(m.boundary_tags)
        noff += c
        loff += m.n_lines
    lines = (np.concatenate(line_list, axis=0) if line_list
             else np.zeros((0, 2), np.int64))
    etags = (np.concatenate(etag_list) if etag_list
             else np.empty(0, dtype=np.str_))
    bnd = (np.concatenate(bnd_list, axis=0) if bnd_list
           else np.zeros((0, 2), np.int64))
    names = (np.concatenate(name_list) if name_list
             else np.empty(0, dtype=np.str_))

    # order-N: welding only touches endpoints (corners, which are re-numbered into
    # the merged points), and every high-order node of a line is *private*, so the
    # merged interior is just the blocks concatenated in the same order the lines
    # were -- nothing to reconcile, nothing to re-pin.
    order = meshes[0].order if meshes else 1
    if any(m.order != order for m in meshes):
        raise ValueError("LineMesh.merge: all meshes must share the same order")
    interior: PointArray | None = None
    if meshes:
        interior = np.concatenate([m.interior for m in meshes], axis=0)

    return LineMesh(points, lines, interior, bnd, names, etags, order=order)


#: Variable-arity combinators bound onto ``LineMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "loft": loft,
    "loft_curve": loft_curve,
    "merge": merge,
}
