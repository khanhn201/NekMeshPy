"""Volumetric conduction machinery for the femoral mesher.

The tet mesh itself, its P1 Laplacian, the three seam fields and the sign-based
partition into legs now live in :mod:`nekmeshpy.tetmesh` (a ``TetMesh`` and
``tetmesh.ops``) -- this module is what is genuinely femoral's own: point location and
a Newton walk on the solved field (``FieldWalker``), tet clipping by a field, and
marching-tets isosurface extraction, projection and disc parametrization for building
one leg's O-grid stations from it.

The point of going volumetric in the first place is that a harmonic field with a
no-flux wall has grad(u) tangent to the wall, so every level set cuts the wall at a
right angle -- which a field solved on the wall alone cannot tell you anything about.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from nekmeshpy import tetmesh
from nekmeshpy.tetmesh import TetMesh


class FieldWalker:
    """Point location and a Newton walk on a P1 field over tets.

    Extracting each station as its own triangle soup and projecting nodes onto it treats
    the stations as unrelated surfaces, so nothing stops two of them from interleaving --
    and a marching-tets isosurface is only C0, with kinks the size of a tet.  Once the
    station spacing drops below the tet size, as it does wherever the layers are graded
    hard toward a seam, the kinks are larger than the gap and the sweep inverts.

    Walking fixes that by construction: a node's position at the next level is reached by
    following the gradient from its position at the previous one, so consecutive stations
    are points on a common streamline and cannot cross however thin the layer gets.  The
    walk lands on the level set exactly (``u`` is continuous even though ``grad u`` is
    not), and because the wall carries a no-flux condition the gradient is tangent to it,
    so wall nodes stay on the wall as they travel."""

    def __init__(self, P, TET, u):
        self.P, self.TET, self.u = P, TET, u
        self.grad, _ = tetmesh.ops.tet_gradients(TetMesh(P, TET))
        self.gu = np.einsum("eij,ei->ej", self.grad, u[TET])
        self.tree = cKDTree(P[TET].mean(axis=1))
        self.un = u[TET]
        # A tet whose four nodes are all pinned to the same Dirichlet value has grad u
        # exactly zero, and a walk starting inside one has no direction to move in.  The
        # seam disc sits squarely on such a plateau, so keep a tree of the tets that do
        # carry a gradient and borrow the nearest one's direction to step off.
        n2 = np.einsum("ej,ej->e", self.gu, self.gu)
        self.live = np.flatnonzero(n2 > 1e-12 * np.median(n2[n2 > 0.0]))
        self.live_tree = cKDTree(P[TET[self.live]].mean(axis=1))
        # grad u of a P1 field is constant per tet and jumps across every face, so two
        # neighbouring nodes walking on the raw gradient set off in visibly different
        # directions and the section picks up facet noise -- the same C0 defect that
        # projecting onto isosurfaces suffers, just moved into the step direction.
        # Recover a continuous gradient by volume-averaging onto the nodes instead.
        _, vol = tetmesh.ops.tet_gradients(TetMesh(P, TET))
        acc = np.zeros_like(P)
        wgt = np.zeros(P.shape[0])
        for c in range(4):
            np.add.at(acc, TET[:, c], vol[:, None] * self.gu)
            np.add.at(wgt, TET[:, c], vol)
        self.gn = acc / np.maximum(wgt, 1e-30)[:, None]

    def isosurface(self, level, smooth=6):
        """``(pts, tris)`` of one level set -- the walker's own tets, so callers do not
        have to keep the subset around beside it.

        Marching tets places every vertex on a tet edge, so the raw surface carries the
        tet mesh's own irregularity: vertices scattered along edges, sliver triangles,
        and kinks the size of an element.  ``smooth`` passes of Taubin relaxation take
        that out tangentially, and a Newton step after each pass puts the surface back on
        the level set, so the smoothing cannot drift it off the field or shrink it.  The
        boundary ring is held fixed -- it lies on the wall, which is not ours to move."""
        pts, tris = isosurface(self.P, self.TET, self.u, level)
        if tris.shape[0] == 0:
            return pts, tris
        tris = orient_soup(tris)
        if not smooth:
            return pts, tris
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        rim = np.zeros(pts.shape[0], bool)
        key = np.sort(e, axis=1)
        uk, cnt = np.unique(key, axis=0, return_counts=True)
        rim[np.unique(uk[cnt == 1])] = True
        e = np.concatenate([e, e[:, ::-1]])
        free = ~rim

        def relax(X, w):
            acc = np.zeros_like(X)
            n = np.zeros(X.shape[0])
            np.add.at(acc, e[:, 0], X[e[:, 1]])
            np.add.at(n, e[:, 0], 1.0)
            ok = n > 0
            out = X.copy()
            m = ok & free
            out[m] += w * (acc[m] / n[m, None] - X[m])
            return out

        for _ in range(int(smooth)):
            pts = relax(relax(pts, 0.5), -0.53)     # Taubin: shrink, then unshrink
            moved = self.advance(pts[free], level)
            pts[free] = moved
        return pts, tris

    def _bary(self, tets, Q):
        """Barycentric coordinates of ``Q`` in ``tets`` -- affine, so straight off the
        basis gradients: ``lam_i(x) = lam_i(p0) + grad_i . (x - p0)``."""
        d = Q - self.P[self.TET[tets, 0]]
        lam = np.einsum("...ij,...j->...i", self.grad[tets], d)
        lam[..., 0] += 1.0
        return lam

    def locate(self, Q, k=24):
        """The tet containing each point, or the nearest one when it lies outside."""
        cand = self.tree.query(Q, k=min(k, self.TET.shape[0]))[1]
        lam = self._bary(cand, Q[:, None, :])
        best = np.argmax(lam.min(axis=2), axis=1)
        rows = np.arange(Q.shape[0])
        return cand[rows, best], np.clip(lam[rows, best], 0.0, None)

    def value(self, Q):
        """``(u, containing tet, grad u)`` at each point, from the gradient-enhanced
        reconstruction ``sum_i lam_i (u_i + g_i . (x - p_i))``.

        Straight linear interpolation is what makes a level set kink: it is only as good
        as the nodal values, so the surface it defines is piecewise planar per tet.
        Blending each node's own first-order Taylor expansion instead costs one extra dot
        product, still agrees with the nodal values, and is still continuous across faces
        (only the face's own nodes have a non-zero ``lam`` there) -- but its level sets
        follow the field rather than the tets."""
        t, lam = self.locate(Q)
        lam = lam / np.maximum(lam.sum(axis=1, keepdims=True), 1e-30)
        nodes = self.TET[t]
        gi = self.gn[nodes]
        d = Q[:, None, :] - self.P[nodes]
        vi = self.un[t] + np.einsum("eij,eij->ei", gi, d)
        return (np.einsum("ei,ei->e", lam, vi), t,
                np.einsum("ei,eij->ej", lam, gi))

    def advance(self, Q, target, iters=24, tol=1e-10):
        """Move ``Q`` onto the ``u = target`` level set along the gradient."""
        Q = np.array(Q, dtype=float)
        for _ in range(iters):
            val, t, g = self.value(Q)
            r = target - val
            if np.all(np.abs(r) < tol):
                break
            n2 = np.einsum("ej,ej->e", g, g)
            dead = n2 <= 1e-12 * max(np.median(n2[n2 > 0.0]), 1e-30)
            if dead.any():
                near = self.live[self.live_tree.query(Q[dead])[1]]
                g = g.copy()
                g[dead] = self.gu[near]
                n2 = np.einsum("ej,ej->e", g, g)
            step = (r / np.maximum(n2, 1e-30))[:, None] * g
            # a full Newton step can leave the mesh where the gradient is small; cap it
            # at a tet's worth of travel and let the next iteration carry the rest
            h = np.linalg.norm(step, axis=1)
            scale = np.minimum(1.0, 2.0 * self._h / np.maximum(h, 1e-30))
            Q = Q + scale[:, None] * step
        return Q

    @property
    def _h(self):
        if not hasattr(self, "_hcache"):
            p = self.P[self.TET]
            self._hcache = float(np.median(np.linalg.norm(p[:, 1] - p[:, 0], axis=1)))
        return self._hcache


def _prism_tets(v):
    """Split a prism ``(v0,v1,v2 / v3,v4,v5)`` into three tets, **conformingly**.

    A prism has three quadrilateral faces, and a quad can be cut along either diagonal.
    Two tet meshes that pick different diagonals on a shared face do not match along it --
    the face is two triangles on one side and two *different* triangles on the other -- so
    the choice cannot be made locally.  Dompierre's rule makes it global: every quad's
    diagonal must contain that quad's lowest-numbered vertex.  Rotating the prism to put
    its own lowest vertex at ``v0`` satisfies the rule on the two faces through ``v0`` for
    free, and the third is decided by comparing the two candidate diagonals' lowest ends.
    Both neighbours see the same global numbers, so both make the same choice."""
    k = int(np.argmin(v))
    r = [k % 3, (k + 1) % 3, (k + 2) % 3]
    lo, hi = (0, 3) if k < 3 else (3, 0)
    w = [v[r[0] + lo], v[r[1] + lo], v[r[2] + lo],
         v[r[0] + hi], v[r[1] + hi], v[r[2] + hi]]
    a, b, c, d, e, f = w
    if min(b, f) < min(c, e):
        return [[a, b, c, f], [a, b, f, e], [a, e, f, d]]
    return [[a, b, c, e], [a, e, c, f], [a, e, f, d]]


_CLIP_EP = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_CLIP_EI = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 2): 3, (1, 3): 4, (2, 3): 5}


def clip_tets(P, F, TET, s, tol=1e-9):
    """Cut a tet mesh along the level set ``s = 0`` and keep ``s > 0``.

    This is what makes a leg's seam a real face set rather than a quantized one.  Every
    node the cut creates sits *exactly* on ``s = 0`` -- it is placed by linear
    interpolation along a straddling edge, which for a P1 field is the exact zero set, the
    same construction ``interface`` uses -- so the boundary of the clipped mesh is the
    smooth cut, and a Dirichlet condition imposed there is imposed on the cut itself.

    Conformity comes from the numbering: a crossing node is keyed on the *edge* it splits,
    so the two tets sharing that edge get the same node rather than two coincident ones,
    and the prism decomposition follows a global diagonal rule (``_prism_tets``).

    ``F`` rides along, interpolated onto the new nodes.  Returns
    ``(P2, F2, TET2, orig)``, where ``orig`` is each new node's index in ``P`` or ``-1``
    if the cut created it -- which is exactly the "this node is on the cut" flag."""
    s = np.array(s, dtype=float)
    sc = float(np.abs(s).max()) or 1.0
    s[np.abs(s) < tol * sc] = tol * sc     # a node *on* the cut joins the kept side
    pos = s > 0.0
    npos = pos[TET].sum(axis=1)
    whole = TET[npos == 4]
    cutT = TET[(npos > 0) & (npos < 4)]

    n0 = P.shape[0]
    if cutT.shape[0]:
        E = cutT[:, _CLIP_EP]
        cr = pos[E[..., 0]] != pos[E[..., 1]]
        uk, inv = np.unique(np.sort(E, axis=2)[cr], axis=0, return_inverse=True)
        a, b = uk[:, 0], uk[:, 1]
        t = (s[a] / (s[a] - s[b]))[:, None]
        Pall = np.vstack([P, P[a] + t * (P[b] - P[a])])
        Fall = np.vstack([F, F[a] + t * (F[b] - F[a])])
        X = np.full((cutT.shape[0], 6), -1, np.int64)
        X[cr] = inv + n0
    else:
        Pall, Fall = P.copy(), F.copy()
        X = np.zeros((0, 6), np.int64)
    orig = np.concatenate([np.arange(n0), np.full(Pall.shape[0] - n0, -1)])

    built = []
    for i in range(cutT.shape[0]):
        loc = cutT[i]
        pl = [k for k in range(4) if pos[loc[k]]]
        nl = [k for k in range(4) if not pos[loc[k]]]

        def cx(u, w, _i=i):
            return int(X[_i, _CLIP_EI[(min(u, w), max(u, w))]])

        if len(pl) == 1:
            p = pl[0]
            built.append([int(loc[p])] + [cx(p, q) for q in nl])
        elif len(pl) == 3:
            q = nl[0]
            built += _prism_tets([int(loc[pl[0]]), int(loc[pl[1]]), int(loc[pl[2]]),
                                  cx(pl[0], q), cx(pl[1], q), cx(pl[2], q)])
        else:
            p0, p1 = pl
            built += _prism_tets([int(loc[p0]), cx(p0, nl[0]), cx(p0, nl[1]),
                                  int(loc[p1]), cx(p1, nl[0]), cx(p1, nl[1])])

    TET2 = (np.vstack([whole, np.array(built, np.int64).reshape(-1, 4)])
            if built else whole.copy())
    p = Pall[TET2]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0])
    TET2[vol < 0.0] = TET2[vol < 0.0][:, [0, 2, 1, 3]]
    TET2 = TET2[np.abs(vol) > 1e-12 * np.median(np.abs(vol))]

    used = np.zeros(Pall.shape[0], bool)
    used[TET2.ravel()] = True
    idx = np.flatnonzero(used)
    newid = np.full(Pall.shape[0], -1, np.int64)
    newid[idx] = np.arange(idx.size)
    return Pall[idx], Fall[idx], newid[TET2], orig[idx]


# -- marching tets -----------------------------------------------------------
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)


def isosurface(P, TET, u, level, carry=None):
    """``{u == level}`` as a triangle soup ``(pts (N,3), tris (M,3))``.

    ``carry`` is an optional ``(n_nodes, k)`` of further nodal fields; they are
    interpolated along the *same* edge parameter as the geometry, so their values on the
    surface are the exact P1 trace rather than a nearby node's.

    One triangle where a single vertex is cut off, two where the tet splits two against
    two.  Cut points are indexed by ``(tet, local edge)`` and welded afterwards, so the
    result comes out as a connected surface rather than loose facets."""
    d = u[TET] - level
    pos = d > 0.0
    npos = pos.sum(axis=1)
    live = np.flatnonzero((npos > 0) & (npos < 4))
    if live.size == 0:
        empty = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return empty if carry is None else empty + (np.zeros((0, np.shape(carry)[1])),)

    # cut point on every edge whose ends straddle the level
    ea, eb = _TET_EDGES[:, 0], _TET_EDGES[:, 1]
    da, db = d[live][:, ea], d[live][:, eb]
    cut = (da > 0.0) != (db > 0.0)
    t = np.where(cut, da / np.where(da == db, 1.0, da - db), 0.0)
    pa = P[TET[live][:, ea]]
    pb = P[TET[live][:, eb]]
    X = pa + t[..., None] * (pb - pa)                      # (L,6,3)
    if carry is not None:
        C = np.asarray(carry, dtype=float)
        ca, cb = C[TET[live][:, ea]], C[TET[live][:, eb]]
        XC = ca + t[..., None] * (cb - ca)                 # (L,6,k)

    tris = []
    for k in range(live.size):
        e = np.flatnonzero(cut[k])
        if e.size == 3:
            tris.append([(k, e[0]), (k, e[1]), (k, e[2])])
        elif e.size == 4:
            # order the four cut points round the quad: two of them share a tet vertex
            q = _TET_EDGES[e]
            order_ = [0]
            rest = [1, 2, 3]
            while rest:
                last = q[order_[-1]]
                nxt = next(i for i in rest if len(set(q[i]) & set(last)) == 1)
                order_.append(nxt)
                rest.remove(nxt)
            o = [e[i] for i in order_]
            tris.append([(k, o[0]), (k, o[1]), (k, o[2])])
            tris.append([(k, o[0]), (k, o[2]), (k, o[3])])
    if not tris:
        empty = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return empty if carry is None else empty + (np.zeros((0, np.shape(carry)[1])),)
    flat = np.array([[a * 6 + b for a, b in tri] for tri in tris], dtype=np.int64)
    pts = X.reshape(-1, 3)
    used, inv = np.unique(flat.ravel(), return_inverse=True)
    if carry is None:
        return _weld(pts[used], inv.reshape(-1, 3))
    vals = XC.reshape(-1, XC.shape[-1])[used]
    wp, wt, keep = _weld(pts[used], inv.reshape(-1, 3), want_map=True)
    return wp, wt, vals[keep]


def _weld(pts, tris, rel=1e-9, want_map=False):
    """Fuse coincident points so the soup becomes a surface with real connectivity."""
    if pts.shape[0] == 0:
        return (pts, tris, np.zeros(0, np.int64)) if want_map else (pts, tris)
    scale = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    tol = max(rel * scale, 1e-12)
    key = np.round(pts / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    t = inv.ravel()[tris]
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return (pts[first], t[keep], first) if want_map else (pts[first], t[keep])


# -- closest point on a triangle soup ----------------------------------------
def _closest_on_tris(q, A, B, C):
    """Exact closest point to ``q`` on each triangle ``(A,B,C)``, all ``(K,3)``."""
    ab, ac, aq = B - A, C - A, q - A
    d1 = np.einsum("ij,ij->i", ab, aq)
    d2 = np.einsum("ij,ij->i", ac, aq)
    bq = q - B
    d3 = np.einsum("ij,ij->i", ab, bq)
    d4 = np.einsum("ij,ij->i", ac, bq)
    cq = q - C
    d5 = np.einsum("ij,ij->i", ab, cq)
    d6 = np.einsum("ij,ij->i", ac, cq)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    # barycentric interior solution, then clamp onto the edges / vertices it falls off
    v = np.divide(vb, den, out=np.zeros_like(den), where=den != 0)
    w = np.divide(vc, den, out=np.zeros_like(den), where=den != 0)
    P = A + v[:, None] * ab + w[:, None] * ac
    def seg(P0, P1):
        d = P1 - P0
        t = np.clip(np.einsum("ij,ij->i", q - P0, d)
                    / np.maximum(np.einsum("ij,ij->i", d, d), 1e-30), 0.0, 1.0)
        return P0 + t[:, None] * d
    P = np.where(((d1 <= 0) & (d2 <= 0))[:, None], A, P)
    P = np.where(((d3 >= 0) & (d4 <= d3))[:, None], B, P)
    P = np.where(((d6 >= 0) & (d5 <= d6))[:, None], C, P)
    P = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], seg(A, B), P)
    P = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], seg(A, C), P)
    P = np.where((((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)))[:, None],
                 seg(B, C), P)
    return P


def project_to_soup(pts, tris, Q, k=12):
    """Closest point on the triangle soup for each of ``Q``.

    ``trimesh.ops.project_points`` snaps to the nearest *vertex's* fan, which assumes the
    query already lies near the surface.  Here it does not -- the O-grid's algebraic
    interior is precisely the thing that is in the wrong place -- so search the fans of
    the ``k`` nearest vertices and take the true closest point on each candidate
    triangle."""
    from scipy.spatial import cKDTree
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    nv = pts.shape[0]
    fan = [[] for _ in range(nv)]
    for t, tri in enumerate(tris):
        for v in tri:
            fan[v].append(t)
    tree = cKDTree(pts)
    _, near = tree.query(Q, k=min(k, nv))
    near = np.atleast_2d(near)
    out = np.array(Q, dtype=float)
    for i in range(Q.shape[0]):
        cand = set()
        for v in near[i]:
            cand.update(fan[v])
        c = np.fromiter(cand, dtype=np.int64, count=len(cand))
        T = tris[c]
        P = _closest_on_tris(np.repeat(Q[i][None, :], c.size, axis=0),
                             pts[T[:, 0]], pts[T[:, 1]], pts[T[:, 2]])
        out[i] = P[int(np.argmin(np.einsum("ij,ij->i", P - Q[i], P - Q[i])))]
    return out


def interface(P, TET, U, i, j, want_cut=False):
    """The interface between legs ``i`` and ``j`` as a genuine level set.

    Two legs differ in the sign of exactly one field, so their common boundary lies on
    that field's zero set; the other two fields' signs say which *part* of it.  Marching
    tets then places every vertex by linear interpolation along a straddling edge, which
    for a P1 field is the exact zero set.

    Taking the tet **faces** that happen to separate two labels instead gives a staircase
    along the existing faces -- jagged at tet scale, and refining only makes the spikes
    finer.  That is the difference between reading a level set and quantizing one."""
    which = {frozenset((1, 2)): 2, frozenset((1, 3)): 1, frozenset((2, 3)): 0}
    trim = {frozenset((1, 2)): ((0, -1), (1, +1)),
            frozenset((1, 3)): ((0, +1), (2, -1)),
            frozenset((2, 3)): ((1, -1), (2, +1))}
    key = frozenset((i, j))
    k = which[key]
    pts, tris, vals = isosurface(P, TET, U[:, k], 0.0, carry=U)
    if tris.shape[0] == 0:
        return (pts, tris, np.zeros(0, dtype=bool)) if want_cut else (pts, tris)
    # keep only the part of the sheet that lies between these two legs; the rest trails
    # off down the third leg, where nothing reads it.  The other fields are carried onto
    # the surface by the same interpolation the geometry uses, so the test is exact --
    # a nearest-node lookup is wrong by up to a cell exactly where the three interfaces
    # meet, which is the spine, which is the one place it must not be.
    return _clip_soup(pts, tris, vals, trim[key])[:2 if not want_cut else 3]


def _clip_soup(pts, tris, vals, constraints):
    """Cut a triangle soup down to where every constraint field is positive.

    Keeping or dropping whole triangles on a sign test quantizes the **boundary** to
    triangle scale -- the same staircase this module refuses to accept for a surface,
    moved to its rim, and it shows up as a row of teeth wherever the sheet ends.
    Refining only makes the teeth finer.

    The constraint fields are carried onto the surface by the same linear interpolation
    that placed its vertices, so within one triangle a constraint vanishes along a
    straight line and the cut is exact.  Clip each triangle against that line
    (Sutherland-Hodgman), interpolating the carried fields onto the new vertices so the
    next constraint can be applied to them in turn, and fan what survives."""
    P, V, T, index, CUT = [], [], [], {}, []

    def vid(p, v, is_cut):
        k = (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))
        got = index.get(k)
        if got is None:
            got = index[k] = len(P)
            P.append(p)
            V.append(v)
            CUT.append(is_cut)
        elif is_cut:
            CUT[got] = True
        return got

    for tri in tris:
        poly = [(pts[i], vals[i], False) for i in tri]
        for f, sign in constraints:
            if len(poly) < 3:
                break
            out = []
            m = len(poly)
            for a in range(m):
                pa, va, ca = poly[a]
                pb, vb, _cb = poly[(a + 1) % m]
                fa, fb = sign * va[f], sign * vb[f]
                if fa >= 0.0:
                    out.append((pa, va, ca))
                if (fa >= 0.0) != (fb >= 0.0) and fa != fb:
                    s = fa / (fa - fb)
                    # born on the line where the constraint vanishes -- which *is* the
                    # curve where this sheet meets the next one
                    out.append((pa + s * (pb - pa), va + s * (vb - va), True))
            poly = out
        if len(poly) < 3:
            continue
        ids = [vid(p, v, c) for p, v, c in poly]
        for a in range(1, len(ids) - 1):
            if ids[0] != ids[a] and ids[a] != ids[a + 1] and ids[0] != ids[a + 1]:
                T.append((ids[0], ids[a], ids[a + 1]))
    if not T:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                np.zeros(0, dtype=bool))
    T = np.array(T, dtype=np.int64)
    # welding by position already dropped the vertices the cut left behind
    used, inv = np.unique(T.ravel(), return_inverse=True)
    return np.array(P)[used], inv.reshape(-1, 3), np.array(CUT, dtype=bool)[used]


# -- laying a half O-grid onto an interface ----------------------------------
def orient_soup(tris):
    """Give a triangle soup one consistent winding.

    Marching tets emits every triangle from its own tet's local vertex numbering, so the
    soup has no coherent orientation at all -- adjacent triangles disagree about which
    side is out, and about half of any signed quantity comes out with the wrong sign.
    Walk the adjacency and flip as needed, so that a shared edge is traversed in opposite
    directions by the two triangles holding it, which is what consistent means."""
    tris = np.array(tris, dtype=np.int64, copy=True)
    m = tris.shape[0]
    share = {}
    for t in range(m):
        for i in range(3):
            a, b = int(tris[t, i]), int(tris[t, (i + 1) % 3])
            share.setdefault((a, b) if a < b else (b, a), []).append(t)

    seen = np.zeros(m, bool)
    for start in range(m):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        while stack:
            t = stack.pop()
            for i in range(3):
                a, b = int(tris[t, i]), int(tris[t, (i + 1) % 3])
                for o in share.get((a, b) if a < b else (b, a), ()):
                    if o == t or seen[o]:
                        continue
                    seen[o] = True
                    # same direction on the shared edge means the neighbour is reversed
                    if any(int(tris[o, j]) == a and int(tris[o, (j + 1) % 3]) == b
                           for j in range(3)):
                        tris[o, [1, 2]] = tris[o, [2, 1]]
                    stack.append(o)
    return tris


def tutte_disc(pts, tris, ring, uv_ring=None):
    """Embed a triangulated disc in the unit circle: boundary by chord length, interior
    by Tutte's barycentric condition (each interior vertex at the mean of its neighbours).

    Uniform weights plus a convex boundary is the one construction here that is
    *provably* fold-free -- Tutte's theorem -- which is exactly what nearest-point
    projection can never promise.  Cotangent weights would follow the geometry more
    faithfully but lose the guarantee the moment a triangle is obtuse, and the whole
    point of coming here is the guarantee.

    ``uv_ring`` overrides where the boundary goes.  The default circle is right for a disc
    being filled with a full O-grid, but a *half* O-grid needs its spine running through
    the interior -- and if the spine is part of the boundary, as it is on an interface,
    the circle puts it on the rim and the O-grid's core collapses onto it.  Passing a
    half-disc instead (arc to the semicircle, spine to the diameter) keeps the boundary
    convex, so Tutte still applies, and puts the spine where a half O-grid expects it.

    Returns ``uv (N,2)``; the rows for ``ring`` land on the given boundary in order."""
    n = pts.shape[0]
    # Mean-value coordinates (Floater).  Uniform weights are what Tutte's theorem is
    # stated for, but they ignore the geometry entirely and distort area exponentially --
    # on a disc this size the parameter coordinates collide outright and the embedding
    # stops being a valid triangulation.  Cotangent weights follow the geometry but go
    # negative on obtuse triangles, and clamping them is a fudge.  Mean-value weights are
    # positive by construction, so every row is still a convex combination and Tutte's
    # guarantee survives intact, while the map stays close to conformal.
    rows, cols, wts = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = pts[tris[:, b]] - pts[tris[:, a]]
        v = pts[tris[:, c]] - pts[tris[:, a]]
        ru = np.linalg.norm(u, axis=1)
        rv = np.linalg.norm(v, axis=1)
        cos = np.einsum("ij,ij->i", u, v) / np.maximum(ru * rv, 1e-30)
        half = np.tan(0.5 * np.arccos(np.clip(cos, -1.0, 1.0)))
        # the angle at `a` is shared by both edges leaving it
        rows.extend([tris[:, a], tris[:, a]])
        cols.extend([tris[:, b], tris[:, c]])
        wts.extend([half / np.maximum(ru, 1e-30), half / np.maximum(rv, 1e-30)])
    W = sp.coo_matrix((np.concatenate(wts),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()

    # Tutte's theorem is about a disc.  A second boundary loop is not a disc, and pinning
    # only the outer one lets the hole fold silently, so refuse rather than embed it.
    e_all = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
                    axis=1)
    _, cnt = np.unique(e_all, axis=0, return_counts=True)
    if int((cnt == 1).sum()) != ring.size:
        raise RuntimeError(
            "tutte_disc: %d boundary edges but the ring has %d points -- the surface is "
            "not a disc" % (int((cnt == 1).sum()), ring.size))

    uv = np.zeros((n, 2))
    if uv_ring is None:
        d = np.linalg.norm(np.diff(pts[np.append(ring, ring[0])], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])[:-1]
        a = 2.0 * np.pi * s / max(s[-1] + d[-1], 1e-30)
        uv[ring] = np.column_stack([np.cos(a), np.sin(a)])
    else:
        uv[ring] = np.asarray(uv_ring, dtype=float).reshape(-1, 2)

    fixed = np.zeros(n, bool)
    fixed[ring] = True
    free = np.flatnonzero(~fixed)
    if free.size == 0:
        return uv
    # one row per free vertex: its position is the weighted mean of its neighbours, with
    # the boundary neighbours already known and so moved to the right-hand side
    W = W.tocoo()
    r, c, wv = W.row, W.col, W.data
    keep = ~fixed[r]
    r, c, wv = r[keep], c[keep], wv[keep]
    tot = np.bincount(r, weights=wv, minlength=n)
    idx = -np.ones(n, dtype=np.int64)
    idx[free] = np.arange(free.size)
    inner, bnd = ~fixed[c], fixed[c]
    rows = np.concatenate([np.arange(free.size), idx[r[inner]]])
    cols = np.concatenate([np.arange(free.size), idx[c[inner]]])
    vals = np.concatenate([np.ones(free.size), -wv[inner] / tot[r[inner]]])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(free.size, free.size)).tocsc()
    rhs = np.zeros((free.size, 2))
    np.add.at(rhs, idx[r[bnd]], uv[c[bnd]] * (wv[bnd] / tot[r[bnd]])[:, None])
    uv[free] = spla.spsolve(A, rhs)
    return uv


class DiscMap:
    """A triangulated disc, its Tutte embedding, and the lift back.

    Placing grid nodes by locating them in the *parameter* triangulation and lifting is
    a homeomorphism, so a fold-free grid in the circle comes back fold-free on the
    surface.  Projecting them onto the surface instead has no such property: two
    neighbouring nodes can land on the same fold, which is what turns O-grid quads
    inside out wherever the section creases."""

    def __init__(self, pts, tris, ring, uv_ring=None):
        self.pts, self.tris, self.ring = pts, tris, ring
        self.uv = tutte_disc(pts, tris, ring, uv_ring)
        a = self.uv[tris[:, 0]]
        self._a = a
        self._v0 = self.uv[tris[:, 1]] - a
        self._v1 = self.uv[tris[:, 2]] - a
        den = self._v0[:, 0] * self._v1[:, 1] - self._v1[:, 0] * self._v0[:, 1]
        self._den = np.where(np.abs(den) < 1e-300, 1e-300, den)

    def ring_uv(self, Q):
        """Parameter coordinates of points known to lie on the boundary polyline."""
        R = self.pts[np.append(self.ring, self.ring[0])]
        U = self.uv[np.append(self.ring, self.ring[0])]
        seg = R[1:] - R[:-1]
        L2 = np.einsum("ij,ij->i", seg, seg)
        t = np.clip(np.einsum("qij,ij->qi", Q[:, None, :] - R[None, :-1, :], seg)
                    / np.maximum(L2, 1e-30)[None, :], 0.0, 1.0)
        foot = R[None, :-1, :] + t[..., None] * seg[None, :, :]
        k = np.argmin(np.linalg.norm(foot - Q[:, None, :], axis=2), axis=1)
        rows = np.arange(Q.shape[0])
        return U[k] + t[rows, k][:, None] * (U[k + 1] - U[k])

    def lift(self, Q2, chunk=4096):
        """3D positions of parameter-plane points, by barycentric interpolation.

        The containing triangle is found by testing *every* triangle, which sounds
        wasteful and is not: a section is a couple of hundred nodes against a couple of
        thousand triangles.  The obvious accelerations are all wrong here.  A Tutte
        embedding distorts area enormously, so "search the nearest few centroids" misses
        the containing triangle routinely -- and answering with the nearest of the wrong
        ones puts the node an arbitrary distance away across the surface, which is
        precisely the failure this class exists to prevent.  An exact test has no
        preconditions on the triangulation and cannot be fooled by the distortion."""
        Q2 = np.asarray(Q2, dtype=float).reshape(-1, 2)
        out = np.empty((Q2.shape[0], 3))
        for s in range(0, Q2.shape[0], chunk):
            q = Q2[s:s + chunk]
            v2 = q[:, None, :] - self._a[None, :, :]
            w1 = (v2[..., 0] * self._v1[None, :, 1]
                  - self._v1[None, :, 0] * v2[..., 1]) / self._den
            w2 = (self._v0[None, :, 0] * v2[..., 1]
                  - v2[..., 0] * self._v0[None, :, 1]) / self._den
            lam = np.stack([1.0 - w1 - w2, w1, w2], axis=2)
            best = np.argmax(lam.min(axis=2), axis=1)
            rows = np.arange(q.shape[0])
            L = np.clip(lam[rows, best], 0.0, None)
            L /= np.maximum(L.sum(axis=1, keepdims=True), 1e-30)
            out[s:s + chunk] = np.einsum("qi,qij->qj", L,
                                         self.pts[self.tris[best]])
        return out


def split_boundary_by_cut(ring, cut):
    """Split an interface's boundary into ``(wall arc, spine)`` by reading the *clip*.

    No tolerance, no projection, no averaging.  ``_clip_soup`` trims each sheet against
    the other two and creates a vertex wherever the constraint vanishes -- so the vertices
    it flags **are** the curve where the sheets meet.  The spine is exactly the flagged
    run of the boundary, and the arc is the rest, which is the part marching tets ended
    against the wall.

    Estimating this instead -- by distance to the wall, or by distance to the other
    sheets, then averaging three such estimates -- gives three curves that disagree by up
    to 0.78 and end in different places, because each is a subset of its own
    triangulation's vertices chosen by a threshold rather than the intersection itself."""
    on = np.asarray(cut, dtype=bool)[ring]
    if not on.any() or on.all():
        raise RuntimeError("interface boundary is entirely on or off the cut")
    start = int(np.flatnonzero(~on)[0])
    o = np.roll(on, -start)
    r = np.roll(ring, -start)
    idx = np.flatnonzero(o)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    best = max(groups, key=len)
    i0, i1 = int(best[0]), int(best[-1])
    return np.concatenate([r[i1 + 1:], r[:i0]])[::-1], r[i0:i1 + 1]


def split_boundary_by_others(disc_pts, ring, others, tol):
    """Split an interface's boundary into ``(wall arc, spine)`` using the *other two
    interfaces* rather than the wall.

    The spine is the triple curve -- where all three interfaces meet -- so a boundary
    vertex is on it exactly when it also lies on the other two.  That is a sharp test: a
    spine vertex is ~0.003 from them, an arc vertex is most of a radius away.

    Deciding by distance to the **wall** instead cannot work, however the cut-off is
    chosen, because the triple points are *on* the wall: the spine approaches it at both
    ends, its last nodes measure as wall nodes, and the spine comes back short by about
    the tolerance -- which is what left the seam disc notched at each corner."""
    from scipy.spatial import cKDTree
    R = disc_pts[ring]
    d = np.max([cKDTree(p).query(R)[0] for p in others], axis=0)
    on = d < tol
    if not on.any() or on.all():
        raise RuntimeError("interface boundary is entirely on or off the triple curve")
    start = int(np.flatnonzero(~on)[0])
    o = np.roll(on, -start)
    r = np.roll(ring, -start)
    idx = np.flatnonzero(o)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    best = max(groups, key=len)
    i0, i1 = int(best[0]), int(best[-1])
    spine = r[i0:i1 + 1]
    arc = np.concatenate([r[i1 + 1:], r[:i0]])
    return arc[::-1], spine


def split_boundary(disc_pts, ring, wall_pts, tol):
    """Split an interface's boundary ring into ``(wall arc, spine)``, both running from
    one triple point to the other.

    An interface is a half-disc bounded by one arc lying *on the wall* and one curve
    through the interior -- the spine -- meeting at the two triple points where all three
    interfaces come together.  Which is which is decided by distance to the wall, but not
    by thresholding it directly: the spine dives toward the wall as it approaches each
    triple point, so any single cut-off flickers there and reports four or six crossings
    instead of two.  The **longest run** off the wall is stable where the crossing count
    is not, so the spine is taken as that run, extended one node each way to land on the
    triple points, and the arc is everything else."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(wall_pts).query(disc_pts[ring])
    off = d >= tol
    if not off.any() or off.all():
        raise RuntimeError("interface boundary is entirely on or off the wall")
    # rotate so the ring starts on the wall, which un-wraps the runs
    start = int(np.flatnonzero(~off)[0])
    o = np.roll(off, -start)
    r = np.roll(ring, -start)
    idx = np.flatnonzero(o)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    best = max(groups, key=len)
    i0, i1 = int(best[0]), int(best[-1])
    spine = r[i0 - 1:i1 + 2]                       # triple point -> triple point
    arc = np.concatenate([r[i1 + 1:], r[:i0]])     # the other way round, on the wall
    return arc[::-1], spine                        # both now run triple point 1 -> 2


def map_to_surface(section, pts, tris):
    """Pull a section onto the surface it is meant to lie on -- **every** node, not just
    the corners.

    A ``QuadMesh`` at order N stores three separate node tables: the shared corners, the
    shared edge interiors, and the private per-quad interiors.  ``spined_ogrid`` places
    the last two to match the corners it was given, so moving only the corners leaves the
    curved nodes behind and every face bulges away from the surface -- visibly, on all
    elements, not only the bad ones.

    Nodes on the boundary are left alone: the rim already lies on the surface's own edge,
    and it carries the wall's analytic curve, which snapping to a faceted triangulation
    would only coarsen."""
    keep = np.zeros(section.n_points, dtype=bool)
    keep[_boundary_points(section)] = True
    free = np.flatnonzero(~keep)
    if free.size:
        section.points[free] = project_to_soup(pts, tris, section.points[free])

    order = section.order
    if order > 1:
        # shared edge interiors, skipping the rim's own edges
        cnt = np.bincount(section.quad.ravel(), minlength=section.line_mesh.n_lines)
        inner = np.flatnonzero(cnt > 1)
        ei = section.line_mesh.interior
        if inner.size and ei.shape[1]:
            flat = ei[inner].reshape(-1, 3)
            ei[inner] = project_to_soup(pts, tris, flat).reshape(inner.size, -1, 3)
        # private per-quad interiors: never on the rim, so all of them
        fi = section.interior
        if fi.shape[1]:
            flat = fi.reshape(-1, 3)
            section.interior[...] = project_to_soup(pts, tris, flat).reshape(fi.shape)
    return section


def _boundary_points(section):
    from nekmeshpy import quadmesh
    return quadmesh.boundary_points(section)


# kept under its old name for the half-interface case, which is the same operation
def map_half_ogrid(section, disc_pts, disc_tris, ring_ids=None):
    """A half O-grid pulled onto its interface -- see :func:`map_to_surface`."""
    return map_to_surface(section, disc_pts, disc_tris)


def shared_spine(spines, n):
    """One spine from the three interfaces' independent estimates of it.

    The three level sets meet along a single curve in the continuum, but each is resolved
    to about a third of an element, so extracting them separately gives three curves a
    fraction of a cell apart -- far too much to weld along.  Averaging three estimates of
    the same curve is a noise reduction, and it is what ``carotid.py`` does one dimension
    down when it takes the mean of its three seam arcs.

    They arrive with whatever orientation their own boundary walk produced, so flip each
    onto the first before resampling all of them to a common arc-length parametrization.
    """
    ref = np.asarray(spines[0], dtype=float)
    out = []
    for s in spines:
        s = np.asarray(s, dtype=float)
        if (np.linalg.norm(s[0] - ref[0]) + np.linalg.norm(s[-1] - ref[-1])
                > np.linalg.norm(s[0] - ref[-1]) + np.linalg.norm(s[-1] - ref[0])):
            s = s[::-1]
        out.append(_resample(s, n))
    return np.mean(out, axis=0)


def _resample(P, n):
    """Polyline resampled to ``n`` points, evenly by arc length."""
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    d /= d[-1]
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(t, d, P[:, k]) for k in range(3)])
