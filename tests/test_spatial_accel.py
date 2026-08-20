"""The spatial-acceleration rewrites: each replaced a Python scan with a scipy index, and
each must be **bit-identical** to the scan it replaced.

That is a stronger claim than "close", and it is the whole reason these are safe to make:
``examples/carotid.py`` is frozen as a golden regression to 1e-12 and it exercises
``project_points`` directly and ``project_to_surface`` through the smoother.  So the
reference implementations are kept here, in the tests, and compared against.
"""

import numpy as np
import pytest

from nekmeshpy import TriMesh
from nekmeshpy.core.fields import DistanceField
from nekmeshpy.hexmesh import smoothing as hsmooth
from nekmeshpy.trimesh import ops


def _surface(rng, nv=40, nt=80):
    V = rng.normal(size=(nv, 3))
    t = rng.integers(0, nv, size=(nt, 3))
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return TriMesh(V, t)


def _scan_project_to_surface(surface, P, faces=None):
    """The O(T*P) full scan ``project_to_surface`` used to be."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Vx = surface.points
    T = surface.tris if faces is None else np.asarray(faces, np.int64).reshape(-1, 3)
    Q = P.copy()
    best = np.full(P.shape[0], np.inf)
    for e in range(T.shape[0]):
        q, d2 = ops._closest_on_tri_vec(P, Vx[T[e, 0]], Vx[T[e, 1]], Vx[T[e, 2]])
        upd = d2 < best
        if np.any(upd):
            best[upd] = d2[upd]
            Q[upd, :] = q[upd, :]
    return Q


def _loop_lerp_along(P, arclen, targets):
    """The per-target Python loop ``_lerp_along`` used to be."""
    K = P.shape[0]
    out = np.zeros((targets.shape[0], 3))
    for k in range(targets.shape[0]):
        s = targets[k]
        idx = min(int(np.flatnonzero(arclen <= s)[-1]), K - 2)
        span = arclen[idx + 1] - arclen[idx]
        t = (s - arclen[idx]) / span if span > 0 else 0.0
        out[k, :] = P[idx, :] + t * (P[idx + 1, :] - P[idx, :])
    return out


# -- _lerp_along --------------------------------------------------------------
def test_lerp_along_matches_the_loop_it_replaced():
    """``searchsorted(side="right")-1`` is the last index with ``arclen[idx] <= s``.
    Targets deliberately include the knots themselves, which is where a ``<`` / ``<=``
    convention would diverge."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        K = int(rng.integers(2, 40))
        P = rng.normal(size=(K, 3))
        arclen = np.concatenate([[0.0], np.cumsum(np.abs(rng.normal(size=K - 1)) + 1e-9)])
        targets = np.concatenate([rng.uniform(0, arclen[-1], int(rng.integers(1, 40))),
                                  arclen])                       # exact knot hits
        assert np.array_equal(ops._lerp_along(P, arclen, targets),
                              _loop_lerp_along(P, arclen, targets))


# -- project_to_surface -------------------------------------------------------
@pytest.mark.parametrize("subset", [False, True])
def test_project_to_surface_matches_the_full_scan(subset):
    """The broad phase must be conservative, not merely plausible: a dropped triangle is
    a *wrong* answer, not a slightly different one.  ``subset=True`` covers the ``faces=``
    path the smoother uses -- where the bound has to be taken over that subset's own
    vertices, since the nearest vertex of the whole surface can be nearer than anything
    in the subset."""
    rng = np.random.default_rng(7)
    for _ in range(40):
        S = _surface(rng)
        if S.n_tris < 4:
            continue
        faces = None
        if subset:
            k = max(1, S.n_tris // 4)
            faces = S.tris[rng.choice(S.n_tris, size=k, replace=False)]
        Q = rng.normal(size=(int(rng.integers(1, 40)), 3)) * rng.choice([0.5, 2.0])
        assert np.array_equal(ops.project_to_surface(S, Q, faces),
                              _scan_project_to_surface(S, Q, faces))


def test_projection_candidates_never_drops_the_winner():
    """Stated as the property rather than the output: every triangle achieving the
    minimum distance must survive the broad phase."""
    rng = np.random.default_rng(3)
    S = _surface(rng, nv=30, nt=60)
    Q = rng.normal(size=(25, 3))
    kept = {p: set() for p in range(Q.shape[0])}
    for e, pts in ops._projection_candidates(S.points, S.tris, Q):
        for p in pts.tolist():
            kept[p].add(e)
    Vx, T = S.points, S.tris
    for p in range(Q.shape[0]):
        d2 = np.array([ops._closest_on_tri_vec(Q[p:p + 1], Vx[T[e, 0]], Vx[T[e, 1]],
                                               Vx[T[e, 2]])[1][0]
                       for e in range(T.shape[0])])
        winners = set(np.flatnonzero(d2 == d2.min()).tolist())
        assert winners <= kept[p], f"point {p}: dropped winning triangles"


def test_projection_candidates_yields_triangles_in_ascending_order():
    """The caller reduces with a strict ``d2 < best``, so ascending order is what keeps
    the lowest-index triangle winning an exact tie, as the full scan did."""
    rng = np.random.default_rng(5)
    S = _surface(rng)
    seen = [e for e, _ in ops._projection_candidates(S.points, S.tris,
                                                     rng.normal(size=(10, 3)))]
    assert seen == sorted(seen) and len(seen) == len(set(seen))


# -- project_points -----------------------------------------------------------
def test_project_points_lands_on_the_surface():
    """The KD-tree replaced only the nearest-*vertex* search; the fan arithmetic is
    untouched, so every result must still sit on some triangle of the surface."""
    rng = np.random.default_rng(11)
    S = _surface(rng, nv=30, nt=60)
    Q = ops.project_points(S, S.points + 1e-3 * rng.normal(size=S.points.shape))
    ref = _scan_project_to_surface(S, Q)          # projecting a surface point is a no-op
    assert np.max(np.linalg.norm(Q - ref, axis=1)) < 1e-9


# -- DistanceField ------------------------------------------------------------
def test_distance_field_matches_the_broadcast_form():
    rng = np.random.default_rng(0)
    src, P = rng.normal(size=(150, 3)), rng.normal(size=(400, 3))
    f = DistanceField(src, 0.1, 2.0, 1.0)
    d = np.sqrt(((P[:, None, :] - src[None, :, :]) ** 2).sum(-1)).min(1)
    want = 0.1 + np.clip(d / 2.0, 0.0, 1.0) * (1.0 - 0.1)
    assert np.array_equal(f(P), want)


# -- smoother adjacency / incidence ------------------------------------------
def test_csr_groups_match_the_list_of_lists():
    rng = np.random.default_rng(0)
    nu = 200
    HC = rng.integers(0, nu, size=(400, 8))
    E = np.unique(np.sort(rng.integers(0, nu, size=(1500, 2)), axis=1), axis=0)
    E = E[E[:, 0] != E[:, 1]]

    adj = [[] for _ in range(nu)]
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)
    nh = [[] for _ in range(nu)]
    for e in range(HC.shape[0]):
        for k in range(8):
            nh[HC[e, k]].append(e)

    got_adj, got_nh = hsmooth._adjacency_lists(E, nu), hsmooth._incidence_lists(HC, nu)
    for v in range(nu):
        assert np.array_equal(np.unique(adj[v]), np.unique(got_adj[v]))
        assert np.array_equal(np.unique(nh[v]), np.unique(got_nh[v]))
