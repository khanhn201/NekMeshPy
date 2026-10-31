"""Analytic surfaces (target geometry + exact curvature) and quad meshes on them, for the curvature checks.

Every surface is a :class:`nekmeshpy.quadmesh.smoother.constrain.Surface` (retraction + unit normal) plus the
exact Gaussian curvature ``K(P)`` at a point on it.  Projection onto the implicit ones is Newton along
the gradient, ``x <- x - f grad f / |grad f|^2``: idempotent and landing on the surface, not exactly
the closest point -- all the smoother needs is a consistent retraction with the surface's own normal.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn


class Analytic(NamedTuple):
    name: str
    surface: cn.Surface
    K: Callable[[np.ndarray], np.ndarray]


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def _implicit(f, grad, iters=20):
    def project(P):
        X = np.array(P, float)
        for _ in range(iters):
            g = grad(X)
            X = X - (f(X) / np.maximum((g * g).sum(1), 1e-300))[:, None] * g
        return X
    return project, lambda P: _unit(grad(P))


def sphere(R=1.0) -> Analytic:
    return Analytic("sphere", cn.sphere([0, 0, 0], R), lambda P: np.full(len(P), 1.0 / R ** 2))


def ellipsoid(a=1.0, b=1.5, c=2.0) -> Analytic:
    def f(X):
        return (X[:, 0] / a) ** 2 + (X[:, 1] / b) ** 2 + (X[:, 2] / c) ** 2 - 1.0

    def g(X):
        return 2.0 * np.stack([X[:, 0] / a ** 2, X[:, 1] / b ** 2, X[:, 2] / c ** 2], 1)

    def K(X):
        s = (X[:, 0] / a ** 2) ** 2 + (X[:, 1] / b ** 2) ** 2 + (X[:, 2] / c ** 2) ** 2
        return 1.0 / ((a * b * c) ** 2 * s ** 2)

    pr, nm = _implicit(f, g)
    return Analytic("ellipsoid", cn.Surface(pr, nm, "ellipsoid"), K)


def torus(R=1.0, r=0.35) -> Analytic:
    def project(P):
        rho = np.maximum(np.linalg.norm(P[:, :2], axis=1), 1e-300)
        q = np.c_[R * P[:, 0] / rho, R * P[:, 1] / rho, np.zeros(len(P))]
        return q + r * _unit(P - q)

    def normal(P):
        rho = np.maximum(np.linalg.norm(P[:, :2], axis=1), 1e-300)
        q = np.c_[R * P[:, 0] / rho, R * P[:, 1] / rho, np.zeros(len(P))]
        return _unit(P - q)

    def K(P):
        rho = np.linalg.norm(P[:, :2], axis=1)
        return (rho - R) / (r ** 2 * rho)

    return Analytic("torus", cn.Surface(project, normal, "torus"), K)


def capsule(r=1.0, L=2.0) -> Analytic:
    """Cylinder of radius ``r`` and length ``L`` along z with a hemispherical cap at each end."""
    h = L / 2.0

    def parts(P):
        z = np.clip(P[:, 2], -h, h)
        c = np.c_[np.zeros(len(P)), np.zeros(len(P)), z]
        return c, _unit(P - c)

    def project(P):
        c, d = parts(P)
        return c + r * d

    def normal(P):
        return parts(P)[1]

    def K(P):
        return np.where(np.abs(P[:, 2]) <= h + 1e-12, 0.0, 1.0 / r ** 2)

    return Analytic("capsule", cn.Surface(project, normal, "capsule"), K)


def saddle() -> Analytic:
    def f(X):
        return X[:, 2] - X[:, 0] * X[:, 1]

    def g(X):
        return np.stack([-X[:, 1], -X[:, 0], np.ones(len(X))], 1)

    def K(P):
        return -1.0 / (1.0 + P[:, 0] ** 2 + P[:, 1] ** 2) ** 2

    pr, nm = _implicit(f, g)
    return Analytic("saddle", cn.Surface(pr, nm, "saddle"), K)


# ----------------------------------------------------------------------- meshes
def tri_to_quads(points, tris):
    """Split every triangle (CCW) into three quads (edge midpoints + centroid), edges shared by
    adjacent triangles share their midpoint.  Gives valence-3 centroids and the original valence
    on the old vertices -- the surface a tet-to-hex conversion leaves."""
    pts = [p for p in points]
    mid, quads = {}, []

    def m(i, j):
        k = (min(i, j), max(i, j))
        if k not in mid:
            mid[k] = len(pts)
            pts.append((points[i] + points[j]) / 2)
        return mid[k]

    for a, b, c in tris:
        g = len(pts)
        pts.append((points[a] + points[b] + points[c]) / 3)
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        quads += [[a, ab, g, ca], [b, bc, g, ab], [c, ca, g, bc]]
    return np.array(pts), np.array(quads)


def random_quad_surface(an: Analytic, n_points=60, seed=0, stretch=None):
    """Random points on the unit sphere -> convex hull -> triangles -> three quads each, then
    carried onto ``an`` by ``stretch`` (a map of the unit sphere) and its projection."""
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_points, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    tris = ConvexHull(X).simplices.copy()
    for t in tris:                                         # outward CCW
        if np.linalg.det(X[t]) < 0:
            t[[1, 2]] = t[[2, 1]]
    P, q = tri_to_quads(X, tris)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    if stretch is not None:
        P = stretch(P)
    return an.surface.project(P), q


def structured_torus(nu=24, nv=12, R=1.0, r=0.35):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.stack([(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)], -1).reshape(-1, 3)
    idx = np.arange(nu * nv).reshape(nu, nv)
    q = np.stack([idx, np.roll(idx, -1, 0), np.roll(np.roll(idx, -1, 0), -1, 1), np.roll(idx, -1, 1)], -1).reshape(-1, 4)
    return P, q


def saddle_grid(n=12):
    xs = np.linspace(-1, 1, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), (X * Y).ravel()], 1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    return P, q


def orient_outward(P, q, center=(0.0, 0.0, 0.0)):
    """Flip the winding of every quad so its normal points away from ``center`` (closed convex-ish
    surfaces); a torus needs ``orient_with`` instead."""
    n = np.cross(P[q[:, 1]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 0]])
    out = np.einsum("ij,ij->i", n, P[q].mean(1) - np.asarray(center)) < 0
    q = q.copy()
    q[out] = q[out][:, ::-1]
    return q


def orient_with(P, q, normal_fn):
    n = np.cross(P[q[:, 1]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 0]])
    flip = np.einsum("ij,ij->i", n, normal_fn(P[q].mean(1))) < 0
    q = q.copy()
    q[flip] = q[flip][:, ::-1]
    return q


# ------------------------------------------------------------- the measurements
def dual_cell_curvature(mesh, an: Analytic, n=6) -> np.ndarray:
    """``(P,)`` exact integral of the Gaussian curvature over each vertex's dual cell: the union of the
    four-point sectors ``[vertex, edge midpoint, quad centre, edge midpoint]`` of its quads.  Each sector is
    split ``n x n`` bilinearly, every sub-corner carried onto the surface, and every sub-cell contributes
    ``K`` at its foot point times its area (midpoint rule, error ``O(h^2)`` of the sub-cell)."""
    P = np.asarray(mesh.points, float)
    q = mesh.corners
    Q = P[q]
    g = Q.mean(1)
    out = np.zeros(len(P))
    t = np.linspace(0.0, 1.0, n + 1)
    for k in range(4):
        c = [Q[:, k], (Q[:, k] + Q[:, (k + 1) % 4]) / 2, g, (Q[:, k] + Q[:, (k - 1) % 4]) / 2]
        grid = np.empty((len(q), n + 1, n + 1, 3))
        for i, a in enumerate(t):
            for j, b in enumerate(t):
                grid[:, i, j] = (1 - a) * (1 - b) * c[0] + a * (1 - b) * c[1] + a * b * c[2] + (1 - a) * b * c[3]
        S = an.surface.project(grid.reshape(-1, 3)).reshape(len(q), n + 1, n + 1, 3)
        p00, p10, p11, p01 = S[:, :-1, :-1], S[:, 1:, :-1], S[:, 1:, 1:], S[:, :-1, 1:]
        area = 0.5 * (np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
                      + np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1))
        foot = an.surface.project(((p00 + p10 + p11 + p01) / 4).reshape(-1, 3))
        out += np.bincount(q[:, k], weights=(area.reshape(len(q), -1) * an.K(foot).reshape(len(q), -1)).sum(1),
                           minlength=len(P))
    return out


def sagitta(mesh, an: Analytic, n=3) -> np.ndarray:
    """Distances to the exact surface of ``n x n`` bilinear sample points inside every quad (the
    between-vertex fidelity a vertex-on-surface constraint cannot see)."""
    Q = np.asarray(mesh.points, float)[mesh.corners]
    t = (np.arange(n) + 0.5) / n
    S = [(1 - a) * (1 - b) * Q[:, 0] + a * (1 - b) * Q[:, 1] + a * b * Q[:, 2] + (1 - a) * b * Q[:, 3] for a in t for b in t]
    X = np.concatenate(S)
    return np.linalg.norm(an.surface.project(X) - X, axis=1)


def area(mesh) -> float:
    P = np.asarray(mesh.points, float)
    t = quadmesh.query._shorter_diagonal_triangles(P, mesh.corners)
    return float(0.5 * np.linalg.norm(np.cross(P[t[:, 1]] - P[t[:, 0]], P[t[:, 2]] - P[t[:, 0]]), axis=1).sum())


def volume(mesh) -> float:
    """Enclosed volume of a closed outward-oriented mesh (divergence theorem on the split triangles)."""
    P = np.asarray(mesh.points, float)
    t = quadmesh.query._shorter_diagonal_triangles(P, mesh.corners)
    return float(np.einsum("ij,ij->i", P[t[:, 0]], np.cross(P[t[:, 1]], P[t[:, 2]])).sum() / 6.0)


def jittered_case(name, seed=0, noise=0.25):
    """``(analytic, jittered mesh, reference mesh)``: a quad mesh on ``name`` with every vertex perturbed
    tangentially by ``noise`` of its local edge and carried back onto the surface."""
    from nekmeshpy.quadmesh.smoother import smooth

    if name == "sphere":
        an = sphere()
        P, q = random_quad_surface(an, 60, seed=seed)
        q = orient_outward(P, q)
    elif name == "ellipsoid":
        an = ellipsoid()
        P, q = random_quad_surface(an, 60, seed=seed, stretch=lambda X: X * [1, 1.5, 2])
        q = orient_outward(P, q)
    elif name == "capsule":
        an = capsule()
        P, q = random_quad_surface(
            an, 80, seed=seed,
            stretch=lambda X: X + np.c_[0 * X[:, 0], 0 * X[:, 0], np.sign(X[:, 2])])
        q = orient_outward(P, q)
    elif name == "torus":
        an = torus()
        P, q = structured_torus(24, 12)
        q = orient_with(P, q, an.surface.normal)
    elif name == "saddle":
        an = saddle()
        P, q = saddle_grid(12)
    else:
        raise ValueError(name)
    ref = quadmesh.QuadMesh.from_corners(P, q)
    ell = cn.vertex_length(ref.points, ref.corners)[:, None]
    Pn = ref.points + np.random.default_rng(seed + 100).normal(size=ref.points.shape) * noise * ell
    if name == "saddle":                                           # the boundary stays where it is
        b = quadmesh.query.boundary_points(ref)
        Pn[b] = ref.points[b]
    return an, smooth.with_points(ref, an.surface.project(Pn)), ref


def curvature_metrics(m, an, interior_only=False):
    """Gauss-Bonnet residual, local curvature error (angle defect vs the exact dual-cell integral),
    sagitta statistics, area."""
    dk = dual_cell_curvature(m, an)
    d = quadmesh.angle_defect(m)
    keep = np.ones(len(d), bool)
    if interior_only:
        keep[quadmesh.query.boundary_points(m)] = False
    sg = sagitta(m, an)
    return {"gb": quadmesh.gauss_bonnet_residual(m),
            "rms": float(np.sqrt(((d - dk)[keep] ** 2).mean())), "max": float(np.abs(d - dk)[keep].max()),
            "sag_max": float(sg.max()), "sag_mean": float(sg.mean()), "area": area(m)}
