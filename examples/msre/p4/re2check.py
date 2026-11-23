"""parRSB's own connectivity checks (nekRS ``parCon``), run on a written .re2.
usage: python re2check.py mesh.re2

parCon merges the element corners geometrically (two corners closer than ``connectivityTol`` times
their local shortest edge become one vertex), then refuses the mesh in ``element_check`` (a hex with
a repeated vertex) or ``face_check`` (an element holding exactly 3 of the 4 vertices of another
element's face -- two faces that agree on three corners and not the fourth, i.e. a non-conformal
seam). Reports both, plus the largest connectivityTol that still keeps every distinct corner apart."""
import sys
import numpy as np
from multiprocessing import Pool
from scipy.spatial import cKDTree

FACES = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
EDGES = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]])


def read_re2(fn):
    with open(fn, "rb") as f:
        ne = int(f.read(80).decode()[5:21])
        f.read(4)
        rec = np.frombuffer(f.read(ne * 25 * 8), "<f8").reshape(ne, 25)
    return rec[:, 1:].reshape(ne, 3, 8).transpose(0, 2, 1).copy()


_V = _CNT = _OFF = _ELEM = None


def _init(V, cnt, off, elem):
    global _V, _CNT, _OFF, _ELEM
    _V, _CNT, _OFF, _ELEM = V, cnt, off, elem


def _chunk(rng):
    s, e = rng
    E = len(_V)
    fv = _V[s:e][:, FACES].reshape(-1, 4)
    fids = np.arange(s * 6, e * 6, dtype=np.int64)
    keys = []
    for k in range(4):
        v = fv[:, k]
        c = _CNT[v]
        first = np.repeat(_OFF[v], c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
        keys.append(np.repeat(fids, c) * E + _ELEM[first])
    u, n = np.unique(np.concatenate(keys), return_counts=True)
    hit = u[n == 3]
    return np.stack([hit // E // 6, hit // E % 6, hit % E], 1)


def check(X, nproc=8):
    """X (E,8,3) corners in .re2 order -> (bad (K,3) [element, face, other element], safe tol)."""
    E = len(X)
    U, inv = np.unique(X.reshape(-1, 3), axis=0, return_inverse=True)
    V = inv.reshape(E, 8)
    dmin = np.full(len(U), np.inf)
    el = np.linalg.norm(X[:, EDGES[:, 0]] - X[:, EDGES[:, 1]], axis=2)
    for k, (a, b) in enumerate(EDGES):
        np.minimum.at(dmin, V[:, a], el[:, k])
        np.minimum.at(dmin, V[:, b], el[:, k])
    d, j = cKDTree(U).query(U, k=2)
    safe = float(np.min(d[:, 1] / np.minimum(dmin, dmin[j[:, 1]])))
    srt = np.sort(V, axis=1)
    if (srt[:, 1:] == srt[:, :-1]).any():
        raise RuntimeError("element_check: a hex has a repeated corner")
    order = np.argsort(V.ravel(), kind="stable")
    cnt = np.bincount(V.ravel(), minlength=len(U))
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    step = 50000
    with Pool(nproc, initializer=_init, initargs=(V, cnt, off, order // 8)) as pool:
        bad = np.concatenate(pool.map(_chunk, [(s, min(E, s + step)) for s in range(0, E, step)]))
    return bad, safe


if __name__ == "__main__":
    X = read_re2(sys.argv[1])
    bad, safe = check(X)
    print("parCon checks: %d hexes, connectivityTol must stay below %.3f" % (len(X), safe))
    cen = X.mean(1)
    for a, f, b in bad[:20]:
        c = cen[a]
        print("  face_check: hex %d face %d vs hex %d at (%.2f, %.2f, %.2f)" % (a + 1, f + 1, b + 1, *c))
    print("face_check: %s" % ("FAILED, %d faces" % len(bad) if len(bad) else "ok"))
    sys.exit(1 if len(bad) else 0)
