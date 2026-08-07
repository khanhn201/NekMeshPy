"""Variable-arity ``LineMesh`` operations -- the only ones that build a numbering."""

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
from ..model.conform import entity_tol
from ..model.fields import gll_nodes, reject_loop_caps
from ..model.tags import ElementTags, PointTags, TagBuilder
from .linemesh import LineMesh, _as_points
from .query import boundary_points


def _cap_tags(cap: str | Sequence[str] | StrArray | None, N: int = 1) -> list[str]:
    """Normalize a cap tag to one tag per cap **node** (length ``N``): a scalar ``str``
    tags the whole cap, an array-like is one tag per node, and ``None`` -- "no cap tag
    asked for" -- is untagged on every node."""
    if cap is None:
        return [""] * N
    if isinstance(cap, str):
        return [cap] * N
    arr = np.asarray(cap, dtype=np.str_).reshape(-1)
    if arr.shape[0] != N:
        raise ValueError("cap tags length (%d) must match cap nodes (%d)"
                         % (arr.shape[0], N))
    return [str(x) for x in arr.tolist()]


def loft(
    points: PointArray,
    *,
    loop: bool = False,
    interior: PointArray | None = None,
    point_tags: PointTags | None = None,
    element_tags: Sequence[str] | StrArray | None = None,
    first_tag: str | Sequence[str] | StrArray | None = None,
    last_tag: str | Sequence[str] | StrArray | None = None,
    order: int = 1,
) -> LineMesh:
    """Loft a stack of point "profiles" into a 1-D mesh -- the bottom rung of the
    uniform sweep primitive shared with :func:`QuadMesh.loft
    <nekmeshpy.quadmesh.assemble.loft>` and :func:`HexMesh.loft
    <nekmeshpy.hexmesh.assemble.loft>`."""
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

    bnd = point_tags if point_tags is not None else PointTags.empty()
    # a chain's cap is a single end **node**, so each cap normalizes to one tag --
    # the rung-1 form of the same scalar-or-per-element argument ``QuadMesh.loft`` /
    # ``HexMesh.loft`` take, so all three rungs accept the same shapes.
    first, last = _cap_tags(first_tag)[0], _cap_tags(last_tag)[0]
    if first or last:
        bb = TagBuilder(PointTags)
        bb.extend(bnd)
        L = lines.shape[0]
        if first and L:
            bb.add(0, 1, first)
        if last and L:
            bb.add(L - 1, 2, last)
        bnd = bb.build_ordered()

    if order > 1 and interior is None:
        # straight GLL blend between each line's two endpoints -- the same
        # expression the straight-sided factories (``line`` / ``rectangle``) use.
        a: PointArray = pts[lines[:, 0]]
        b: PointArray = pts[lines[:, 1]]
        g = gll_nodes(order)[1:order]              # interior GLL nodes only
        interior = a[:, None, :] + g[None, :, None] * (b - a)[:, None, :]
    return LineMesh(pts, lines, interior, bnd,
                    ElementTags.from_dense(element_tags, lines.shape[0], "lines")
                    if element_tags is not None else None)


def _refined_lattice(fractions: FloatArray, order: int) -> FloatArray:
    """The ``n*order + 1`` parameter positions of **every** node of the order-``order``
    chain graded by ``fractions`` (``n = len(fractions) - 1`` elements): element ``i``'s
    node ``a`` sits at ``fr[i] + g[a]*(fr[i+1] - fr[i])`` for the GLL nodes ``g`` on
    ``[0, 1]``, and the chain ends at ``fr[-1]``."""
    g: FloatArray = gll_nodes(order)
    fr = fractions
    u: FloatArray = (fr[:-1, None]
                     + g[None, :order] * np.diff(fr)[:, None]).ravel()
    return np.concatenate([u, fr[-1:]])


def _check_fraction_count(fr: FloatArray, *, loop: bool, name: str) -> None:
    """Raise unless ``fr`` carries enough fractions for at least one sweep layer (two,
    with ``loop=True`` -- the last fraction is the wrap back onto the first profile
    rather than a level of its own)."""
    if fr.shape[0] - 1 < 1:
        raise ValueError("%s needs at least 2 fractions (one layer), got %d"
                         % (name, fr.shape[0]))
    if loop and fr.shape[0] - 1 < 2:
        raise ValueError(
            "%s(loop=True) needs at least 3 fractions (two layers), got %d -- the "
            "last one is the wrap back to the first profile, so it is not a level of "
            "its own" % (name, fr.shape[0]))


def _sweep_lattice(fractions: FloatArray, order: int, *, loop: bool,
                   name: str) -> tuple[FloatArray, FloatArray]:
    """Validate a sweep's ``fractions`` and return ``(fr, node lattice)``."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    _check_fraction_count(fr, loop=loop, name=name)
    return fr, _refined_lattice(fr, order)


def _sweep_path(path: Callable[[FloatArray], PointArray],
                tangent: Callable[[FloatArray], PointArray] | None,
                tv: FloatArray) -> tuple[PointArray, PointArray | None]:
    """Sample a sweep's centreline (and, if given, its analytic derivative) on the
    station parameters ``tv``, as ``(K,3)`` arrays."""
    P: PointArray = np.asarray(path(tv), dtype=float)
    if P.shape != (tv.shape[0], 3):
        raise ValueError("sweep: path must map the (%d,) sweep lattice to a (%d,3) "
                         "array of centreline points, got %s"
                         % (tv.shape[0], tv.shape[0], (P.shape,)))
    if tangent is None:
        return P, None
    T: PointArray = np.asarray(tangent(tv), dtype=float)
    if T.shape != (tv.shape[0], 3):
        raise ValueError("sweep: tangent must map the (%d,) sweep lattice to a "
                         "(%d,3) array of unit tangents, got %s"
                         % (tv.shape[0], tv.shape[0], (T.shape,)))
    return P, T / np.linalg.norm(T, axis=1)[:, None]


def _eval_curve(f: Callable[[FloatArray], PointArray], t: FloatArray) -> PointArray:
    """``f(t)`` as a validated ``(len(t), 3)`` array."""
    P: PointArray = np.asarray(f(t), dtype=float)
    if P.shape != (t.shape[0], 3):
        raise ValueError(
            "curve callable must return (len(t), 3) points; got shape %r for %d "
            "parameters" % (P.shape, t.shape[0]))
    return P


def loft_fn(f: Callable[[FloatArray], PointArray], fractions: float | FloatArray, *,
            loop: bool = False, order: int = 1,
            element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
    """Loft a curve given as its own analytic parametrization -- :func:`loft
    <nekmeshpy.linemesh.assemble.loft>` with the profiles **evaluated** rather than
    handed in, so **every** node (corners *and* the private high-order ``interior``)
    comes from calling ``f`` and nothing is ever placed on a chord."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    ni = fr.shape[0] - 1
    if ni < 1:
        raise ValueError(
            "loft_fn needs at least 2 fractions (one element), got %d" % fr.shape[0])
    if loop and ni < 2:
        raise ValueError(
            "loft_fn(loop=True) needs at least 3 fractions (two elements), got %d -- "
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
                "loft_fn(loop=True) needs the last fraction to map back to the "
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


def _weld(pos: Sequence[PointArray], seams: Sequence[IntArray],
          tol: float | None) -> tuple[PointArray, IntArray]:
    """The corner half of every ``merge``: concatenate the blocks' points, fuse the
    coincident *weldable* ones, and renumber."""
    P: PointArray = np.concatenate(list(pos), axis=0) if pos else np.zeros((0, 3))
    total = P.shape[0]

    remap = np.arange(total, dtype=np.int64)
    is_bnd: BoolArray = np.zeros(total, dtype=bool)
    noff = 0
    for p, seam in zip(pos, seams):
        is_bnd[noff + seam] = True
        noff += p.shape[0]
    bidx = np.flatnonzero(is_bnd)
    if bidx.size:
        scl = float(np.max(P.max(axis=0) - P.min(axis=0)))
        t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
        keys = np.round(P[bidx, :] / t).astype(np.int64)
        _, first_local, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True)
        remap[bidx] = bidx[first_local][inverse.ravel()]

    survivors = np.unique(remap)
    new_id: IntArray = np.empty(total, dtype=np.int64)
    new_id[survivors] = np.arange(survivors.size)
    return P[survivors, :], new_id[remap]


def merge(meshes: Sequence[LineMesh], *,
          tol: float | None = None) -> LineMesh:
    """Merge line meshes into one, welding coincident **topological end points** (the
    degree-1 chain ends -- the 1-D analogue of the boundary vertices
    ``QuadMesh.merge``/``HexMesh.merge`` weld)."""
    meshes = list(meshes)
    pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
    counts = [p.shape[0] for p in pos]
    # the weldable points here are the chain **ends** -- the 1-D boundary
    points, point_id = _weld(pos, [boundary_points(m) for m in meshes], tol)

    line_list: list[IntArray] = []
    bnd_list: list[PointTags] = []
    etag_list: list[ElementTags] = []
    noff = loff = 0
    for m, c in zip(meshes, counts):
        line_list.append(point_id[m.lines + noff])   # local -> welded id
        # ids shift by this block's offset; sides stay local to their element
        etag_list.append(m.element_tags.offset(loff))
        bnd_list.append(m.point_tags.offset(loff))
        noff += c
        loff += m.n_lines
    lines = (np.concatenate(line_list, axis=0) if line_list
             else np.zeros((0, 2), np.int64))
    etags = ElementTags.concat(etag_list)
    bnd = PointTags.concat(bnd_list)

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

    return LineMesh(points, lines, interior, bnd, etags)

__all__ = [
    "loft",
    "loft_fn",
    "merge",
]
