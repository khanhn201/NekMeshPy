"""Shared internals for :func:`hexmesh.assemble.refine <nekmeshpy.hexmesh.assemble.refine>`."""

from __future__ import annotations

import numpy as np

from .._typing import BoolArray, IntArray
from ..core.conform import _LOCAL_EDGES, _face_axes
from ..core.interp import _CORNER_IJK

#: For every (octant ``k``, octant-local corner ``m``) pair, the role that corner plays
#: on the parent's own 27-point auxiliary lattice (8 corners + 12 edge-midpoints + 6
#: face-centers + 1 cell-center), and which of the parent's own corners/edges/faces it
#: is the midpoint/center *of*. Purely combinatorial -- independent of any mesh -- so
#: it is built once at import time rather than per call.
#:
#: The classification: octant ``k`` occupies grid range ``[bit, bit+1]`` on each axis
#: (``bit`` from ``_CORNER_IJK[3][k]``), so its own local corner ``m`` (bits ``mu,mv,mw``
#: from ``_CORNER_IJK[3][m]``) sits at absolute grid coordinate ``bit + m_bit`` per axis,
#: each in ``{0,1,2}``. All-even -> one of the parent's own 8 corners; exactly one ``1``
#: -> the midpoint of one of the parent's 12 edges (the free axis identifies which pair
#: of corners, hence which :data:`_LOCAL_EDGES` row); exactly two ``1``s -> the center
#: of one of the parent's 6 faces (the one fixed axis identifies which, via
#: :func:`_face_axes`); all ``1``s -> the parent's own cell-center.
_HEX_OCTANT_ROLE: dict[tuple[int, int], tuple[str, int]] = {}


def _build_octant_role_table() -> dict[tuple[int, int], tuple[str, int]]:
    ck = _CORNER_IJK[3]
    edges = _LOCAL_EDGES[3].tolist()
    faces = _face_axes()
    role: dict[tuple[int, int], tuple[str, int]] = {}
    for k in range(8):
        bu, bv, bw = ck[k]
        for m in range(8):
            mu, mv, mw = ck[m]
            g = (bu + mu, bv + mv, bw + mw)
            ones = sum(1 for x in g if x == 1)
            if ones == 0:
                m0 = ck.index(tuple(x // 2 for x in g))
                role[k, m] = ("corner", m0)
            elif ones == 1:
                free = g.index(1)
                ca = list(g)
                ca[free] = 0
                cb = list(g)
                cb[free] = 2
                ma = ck.index(tuple(x // 2 for x in ca))
                mb = ck.index(tuple(x // 2 for x in cb))
                e_idx = next(i for i, (a, b) in enumerate(edges) if {a, b} == {ma, mb})
                role[k, m] = ("edge", e_idx)
            elif ones == 2:
                pinned = next(a for a in range(3) if g[a] != 1)
                pin = g[pinned] // 2
                f_idx = next(f for f, (p, pv, _, _, _) in enumerate(faces)
                            if p == pinned and pv == pin)
                role[k, m] = ("face", f_idx)
            else:
                role[k, m] = ("cell", -1)
    return role


_HEX_OCTANT_ROLE = _build_octant_role_table()

#: For every (octant ``k``, octant-local face ``f`` 0-5), whether that face is
#: **outer** -- aligned with the parent's own local face ``f`` (same index: the local
#: ``(pinned, pin)`` frame is a mesh-independent hex property, shared by every octant)
#: -- or **inner** (genuinely new, cutting through the parent's interior). Outer iff the
#: octant's own bit on the face's pinned axis equals the face's pin value.
_HEX_OCTANT_OUTER: BoolArray = np.zeros((8, 6), dtype=bool)


def _build_octant_outer_table() -> BoolArray:
    ck = _CORNER_IJK[3]
    faces = _face_axes()
    out: BoolArray = np.zeros((8, 6), dtype=bool)
    for k in range(8):
        bits = ck[k]
        for f, (pinned, pin, _, _, _) in enumerate(faces):
            out[k, f] = bits[pinned] == pin
    return out


_HEX_OCTANT_OUTER = _build_octant_outer_table()


#: Per hex local face, the ``(swap, reflect_u, reflect_v)`` transform carrying the
#: standard CCW ``_CORNER_IJK[2]`` frame onto :func:`_face_axes`'s own ``corner_uv``
#: frame -- i.e. how a block naturally extracted "``u_ax`` fastest, raw grid" (the
#: natural output of sampling the two in-face axes directly) must be re-lexed before
#: :func:`corner_indices <nekmeshpy.core.interp.corner_indices>` on it reproduces
#: ``_LOCAL_FACES[f]``'s own corner order. Computed once (mesh- and order-independent):
#: faces 0/1/5 need no correction, 2/3 reflect ``u``, and 4 swaps ``u``/``v`` -- because
#: ``_face_axes``'s ascending-axis convention for the in-face frame does not always
#: agree with the winding ``_LOCAL_FACES`` itself uses.
_FACE_TRANSFORM: list[tuple[bool, bool, bool]] = [
    (False, False, False), (False, False, False), (False, True, False),
    (False, True, False), (True, False, False), (False, False, False)]


def hex_face_full_slots(order: int) -> IntArray:
    """``(6, (order+1)**2)`` block slots of each hex face's FULL boundary+interior
    nodes, in the face's own ``(u_ax, v_ax)`` frame with ``u_ax`` fastest (raw, before
    :func:`face_lex_perm`'s correction) -- the ``(order+1)**2``-sized generalization of
    :func:`conform._face_interior_slots <nekmeshpy.core.conform._face_interior_slots>`,
    which only covers the strictly-interior ``(order-1)**2``."""
    n = order
    strides = (1, n + 1, (n + 1) ** 2)
    out: IntArray = np.empty((6, (n + 1) ** 2), dtype=np.int64)
    for f, (pinned, pin, u_ax, v_ax, _) in enumerate(_face_axes()):
        m = 0
        for vv in range(n + 1):
            for uu in range(n + 1):
                coord = [0, 0, 0]
                coord[pinned] = pin * n
                coord[u_ax] = uu
                coord[v_ax] = vv
                out[f, m] = sum(coord[a] * strides[a] for a in range(3))
                m += 1
    return out


def face_lex_perm(f: int, order: int) -> IntArray:
    """``(order+1)**2`` permutation: a face block sampled "``u_ax`` fastest, raw grid"
    (:func:`_face_axes`'s own in-face axes, no correction), reindexed
    ``block[face_lex_perm(f, order)]``, comes out in :data:`_LOCAL_FACES` ``f``'s own
    corner winding -- so :func:`corner_indices <nekmeshpy.core.interp.corner_indices>`
    on the result matches ``_LOCAL_FACES[f]`` directly."""
    n = order
    row = n + 1
    swap, ru, rv = _FACE_TRANSFORM[f]
    perm: IntArray = np.empty(row * row, dtype=np.int64)
    for tv in range(row):
        for tu in range(row):
            t = tu + row * tv
            su, sv = (tv, tu) if swap else (tu, tv)
            su = n - su if ru else su
            sv = n - sv if rv else sv
            perm[t] = su + row * sv
    return perm


def octant_corner_ids(k: int, corners: IntArray, elem_edges: IntArray,
                      hexes: IntArray, n0_line: int, n0_face_center: int,
                      cell_center_id: IntArray) -> IntArray:
    """``(E,8)`` point ids of octant ``k``'s 8 corners, Nek order.

    ``corners`` is the parent's own ``(E,8)`` corner-point-id table; ``elem_edges`` is
    ``mesh._elem_edges`` ``(E,12)``; ``hexes`` is the parent's own ``(E,6)`` face-id
    table. A shared edge/face's own new midpoint/center point id is a fixed offset
    from its row: :func:`linemesh.refine <nekmeshpy.linemesh.assemble.refine>` puts
    edge ``r``'s midpoint at ``n0_line + r``, and :func:`quadmesh.refine
    <nekmeshpy.quadmesh.assemble.refine>` puts face ``r``'s center at
    ``n0_face_center + r`` -- so no lookup table is needed, just those two offsets.
    ``cell_center_id`` is ``(E,)``, one new point per parent hex."""
    e_count = corners.shape[0]
    out: IntArray = np.empty((e_count, 8), dtype=np.int64)
    for m in range(8):
        kind, idx = _HEX_OCTANT_ROLE[k, m]
        if kind == "corner":
            out[:, m] = corners[:, idx]
        elif kind == "edge":
            out[:, m] = n0_line + elem_edges[:, idx]
        elif kind == "face":
            out[:, m] = n0_face_center + hexes[:, idx]
        else:
            out[:, m] = cell_center_id
    return out
