"""Cross-section interior-repositioning strategies (free functions + registry).

Each strategy is a callable ``fn(hexmesh, twall, **opts) -> hexmesh`` registered
under one or more names.  :func:`set_interior` looks the name up here, so new
strategies can be added from user code without touching :class:`HexMesh`::

    from nekmeshpy.ops.interior import register_interior

    @register_interior("laplacian")
    def _laplacian(mesh, twall, **opts):
        ...
        return mesh

The built-in elliptic strategies operate on the welded shared-node view of the
mesh (:meth:`HexMesh.weld`) and are numerically identical to the original
verbatim implementations.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..model import quality

INTERIOR_METHODS: dict = {}


def register_interior(*names):
    """Decorator: register ``fn`` under each of ``names`` (case-insensitive)."""
    def deco(fn):
        for n in names:
            INTERIOR_METHODS[n.lower()] = fn
        return fn
    return deco


def available():
    """Sorted list of registered (non-empty) strategy names."""
    return sorted(n for n in INTERIOR_METHODS if n)


def set_interior(mesh, method, twall, **opts):
    """Reposition the O-grid cross-section interior nodes of ``mesh`` via a
    strategy registered here (built-ins: bilinear, harmonic, harmonic3d,
    winslow).  Extra keywords are forwarded to the strategy."""
    mesh.finalize()
    m = (method or "harmonic").lower()
    fn = INTERIOR_METHODS.get(m)
    if fn is None:
        raise ValueError('set_interior: unknown method "%s" (available: %s)'
                         % (method, ", ".join(available())))
    return fn(mesh, twall, **opts)


# -- elliptic interior implementations ----------------------------------
def harmonic_interior(mesh, twall, full3d=False):
    """Discrete-harmonic map of each cross-section interior (in-section
    coupling only, unless ``full3d``)."""
    mesh.finalize()
    X, HC, nu = mesh.weld()
    he = mesh._HE_SECTION if not full3d else np.vstack([mesh._HE_SECTION, mesh._HE_AXIAL])
    E = mesh._unique_edges(HC, he)
    A = sp.coo_matrix((np.ones(E.shape[0] * 2),
                       (np.concatenate([E[:, 0], E[:, 1]]),
                        np.concatenate([E[:, 1], E[:, 0]]))), shape=(nu, nu)).tocsr()
    d = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(d) - A
    is_wall, is_fixed = mesh.classify_nodes(twall)
    bc = is_wall | is_fixed
    I = np.flatnonzero(~bc)
    B = np.flatnonzero(bc)
    Lc = L.tocsc()
    rhs = -(Lc[I, :][:, B] @ X[B, :])
    X[I, :] = spla.spsolve(Lc[I, :][:, I].tocsc(), rhs)
    mesh._write_back(X, HC)
    return mesh


def winslow_interior(mesh, twall, iters=30):
    """Winslow-type elliptic interior (floored, under-relaxed, keep-best)."""
    mesh.finalize()
    omega = 0.5
    X, HC, nu = mesh.weld()
    E = mesh._unique_edges(HC, mesh._HE_SECTION)
    adj = [[] for _ in range(nu)]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    adj = [np.asarray(a, dtype=np.int64) for a in adj]
    is_wall, is_fixed = mesh.classify_nodes(twall)
    interior = np.flatnonzero(~(is_wall | is_fixed))
    dfloor = np.zeros(nu)
    for v in interior:
        nb = adj[v]
        dfloor[v] = 0.1 * np.mean(np.sqrt(np.sum((X[nb, :] - X[v, :]) ** 2, axis=1)))
    cap = max(5, int(np.ceil(0.01 * HC.shape[0])))
    sj0 = quality.scaled_jacobian(X, HC)
    bestX = X.copy()
    bestMean = float(np.mean(sj0))
    stall = 0
    for _ in range(iters):
        for v in interior:
            nb = adj[v]
            d = np.sqrt(np.sum((X[nb, :] - X[v, :]) ** 2, axis=1))
            w = 1.0 / np.maximum(d, dfloor[v])
            xn = (w @ X[nb, :]) / np.sum(w)
            X[v, :] = (1 - omega) * X[v, :] + omega * xn
        sj = quality.scaled_jacobian(X, HC)
        if np.sum(sj <= 0) > cap:
            break
        if np.mean(sj) > bestMean + 1e-6:
            bestMean = float(np.mean(sj))
            bestX = X.copy()
            stall = 0
        else:
            stall += 1
            if stall >= 3:
                break
    mesh._write_back(bestX, HC)
    return mesh


# -- registered strategies ----------------------------------------------
@register_interior("bilinear", "tfi", "none", "")
def _bilinear(mesh, twall, **opts):
    return mesh                      # keep the algebraic transfinite fill


@register_interior("harmonic")
def _harmonic(mesh, twall, **opts):
    return harmonic_interior(mesh, twall, False)


@register_interior("harmonic3d")
def _harmonic3d(mesh, twall, **opts):
    return harmonic_interior(mesh, twall, True)


@register_interior("winslow")
def _winslow(mesh, twall, **opts):
    return winslow_interior(mesh, twall, iters=opts.get("iters", 30))
