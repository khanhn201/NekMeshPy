"""Cross-section interior-repositioning strategies (free functions + registry).

Interior repositioning is done **per cross-section** on a single O-grid
:class:`~nekmeshpy.geometry.quadmesh.QuadMesh` before it is swept/extruded, so the
strategies here take a ``QuadMesh`` (not an assembled ``HexMesh``).  A slice's
point indices are already global, so no weld is needed; the fixed points are found
generically as those on a quad edge belonging to a single quad (for a half-disk
this is exactly the wall arc + the flat spine, so the two halves stay conformal).

Each strategy is a callable ``fn(qm, **opts) -> qm`` registered under one or more
names.  :func:`set_section_interior` looks the name up here, so new strategies can
be added from user code without touching the meshers::

    from nekmeshpy.ops.interior import register_section_interior

    @register_section_interior("laplacian")
    def _laplacian(qm, **opts):
        ...
        return qm

The built-in elliptic strategies (``conduction`` / ``winslow``) are numerically
identical to the original verbatim implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._typing import BoolArray, IntArray

if TYPE_CHECKING:
    from ..geometry.quadmesh import QuadMesh

F = TypeVar("F", bound=Callable[..., "QuadMesh"])

# quad edge ring (CCW)
_QE_RING = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)


def _section_edges(quads: IntArray) -> tuple[IntArray, IntArray]:
    """``(edges (nE,2), counts (nE,))``: unique undirected quad edges of a
    cross-section and how many quads each borders."""
    Ei = quads[:, _QE_RING[:, 0]].ravel()
    Ej = quads[:, _QE_RING[:, 1]].ravel()
    pairs = np.sort(np.column_stack([Ei, Ej]), axis=1)
    return np.unique(pairs, axis=0, return_counts=True)


def _section_boundary(quads: IntArray, nu: int) -> BoolArray:
    """Boolean mask (nu,) of points on a boundary edge (borne by one quad)."""
    edges, counts = _section_edges(quads)
    bnd: BoolArray = np.zeros(nu, dtype=bool)
    bnd[edges[counts == 1].ravel()] = True
    return bnd


def conduction_section(qm: QuadMesh) -> QuadMesh:
    """Discrete-conduction map of one cross-section's interior, boundary
    (wall + flat edge) held fixed.  Repositions ``qm.points`` in place."""
    X = qm.points
    nu = X.shape[0]
    edges, _ = _section_edges(qm.quads)
    A = sp.coo_matrix((np.ones(edges.shape[0] * 2),
                       (np.concatenate([edges[:, 0], edges[:, 1]]),
                        np.concatenate([edges[:, 1], edges[:, 0]]))),
                      shape=(nu, nu)).tocsr()
    d = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(d) - A
    bc = _section_boundary(qm.quads, nu)
    I = np.flatnonzero(~bc)
    if I.size == 0:
        return qm
    B = np.flatnonzero(bc)
    Lc = L.tocsc()
    rhs = -(Lc[I, :][:, B] @ X[B, :])
    X[I, :] = spla.spsolve(Lc[I, :][:, I].tocsc(), rhs)
    return qm


def winslow_section(qm: QuadMesh, iters: int = 30, omega: float = 0.5) -> QuadMesh:
    """Winslow-type elliptic smoothing of one cross-section's interior
    (floored, under-relaxed), boundary held fixed.  In place."""
    X = qm.points
    nu = X.shape[0]
    edges, _ = _section_edges(qm.quads)
    adj: list[Any] = [[] for _ in range(nu)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    adj = [np.asarray(a, dtype=np.int64) for a in adj]
    interior = np.flatnonzero(~_section_boundary(qm.quads, nu))
    if interior.size == 0:
        return qm
    dfloor = np.zeros(nu)
    for v in interior:
        nb = adj[v]
        dfloor[v] = 0.1 * np.mean(np.sqrt(np.sum((X[nb, :] - X[v, :]) ** 2, axis=1)))
    for _ in range(iters):
        for v in interior:
            nb = adj[v]
            dist = np.sqrt(np.sum((X[nb, :] - X[v, :]) ** 2, axis=1))
            w = 1.0 / np.maximum(dist, dfloor[v])
            xn = (w @ X[nb, :]) / np.sum(w)
            X[v, :] = (1 - omega) * X[v, :] + omega * xn
    return qm


# -- registry -----------------------------------------------------------
SECTION_METHODS: dict[str, Callable[..., QuadMesh]] = {}


def register_section_interior(*names: str) -> Callable[[F], F]:
    """Decorator: register a section strategy ``fn(qm, **opts) -> qm`` under each
    of ``names`` (case-insensitive)."""
    def deco(fn: F) -> F:
        for n in names:
            SECTION_METHODS[n.lower()] = fn
        return fn
    return deco


def available() -> list[str]:
    """Sorted list of registered (non-empty) strategy names."""
    return sorted(n for n in SECTION_METHODS if n)


@register_section_interior("bilinear", "tfi", "none", "")
def _section_noop(qm: QuadMesh, **opts: Any) -> QuadMesh:
    return qm                         # keep the algebraic transfinite fill


@register_section_interior("conduction")
def _conduction(qm: QuadMesh, **opts: Any) -> QuadMesh:
    return conduction_section(qm)


@register_section_interior("winslow")
def _winslow(qm: QuadMesh, **opts: Any) -> QuadMesh:
    return winslow_section(qm, iters=opts.get("iters", 30))


def set_section_interior(qm: QuadMesh, method: str | None, **opts: Any) -> QuadMesh:
    """Reposition one O-grid cross-section's interior points in place via a
    per-slice strategy registered here (built-ins: bilinear, conduction,
    winslow).  Extra keywords are forwarded to the strategy."""
    m = (method or "conduction").lower()
    fn = SECTION_METHODS.get(m)
    if fn is None:
        raise ValueError('set_section_interior: unknown method "%s" (available: %s)'
                         % (method, ", ".join(available())))
    return fn(qm, **opts)
