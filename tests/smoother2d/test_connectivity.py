"""The gencon vertex rule and the preflight gap check."""

import numpy as np
import pytest
from grids import grid

from nekmeshpy import quadmesh
from nekmeshpy.core import conform


def _soup(xs, ys, noise=0.0, seed=0):
    """Structured quads as an unwelded corner soup, with the true vertex id of every corner."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nx, ny = len(xs) - 1, len(ys) - 1
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i, j = i.ravel(), j.ravel()
    cid = np.stack([np.stack([i, j], -1), np.stack([i + 1, j], -1),
                    np.stack([i + 1, j + 1], -1), np.stack([i, j + 1], -1)], axis=1)
    P = np.zeros((len(i), 4, 3))
    P[..., 0], P[..., 1] = X[cid[..., 0], cid[..., 1]], Y[cid[..., 0], cid[..., 1]]
    truth = (cid[..., 0] * (ny + 1) + cid[..., 1]).reshape(-1)
    return P, truth


def _same_partition(a, b):
    return len(np.unique(a)) == len(np.unique(b)) == len(np.unique(np.stack([a, b], 1), axis=0))


def test_gencon_recovers_the_exact_vertex_partition_on_a_noisy_uniform_soup():
    xs = np.linspace(0, 1, 13)
    P, truth = _soup(xs, xs)
    P = P + np.random.default_rng(0).uniform(-1e-12, 1e-12, P.shape)
    ell = conform.corner_length(P).reshape(-1)
    assert _same_partition(conform.gencon_clusters(P.reshape(-1, 3), ell), truth)


def test_the_tolerance_is_local_graded_mesh_welds_exactly_where_a_global_radius_cannot():
    n1 = 40
    xs = np.cumsum(np.r_[0, 1e-5 * 1.3 ** np.arange(n1)])
    xs = xs / xs[-1]
    P, truth = _soup(xs, np.linspace(0, 1, 20))
    ell = conform.corner_length(P)
    P = P + np.random.default_rng(1).uniform(-1, 1, P.shape) * (1e-3 * ell)[..., None]   # seam error scales with the cell
    X, L = P.reshape(-1, 3), conform.corner_length(P).reshape(-1)
    assert _same_partition(conform.gencon_clusters(X, L), truth)
    for tol in (1e-7 * 1.0, 0.2 * L.min()):                      # a tiny and a "safe" global radius both miss seams
        assert not _same_partition(conform.coincident_clusters(X, tol), truth)


def test_a_pair_closer_than_the_local_tolerance_fuses_and_one_beyond_it_does_not():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.]])
    near = np.stack([P, P + [1.0, 0, 0] - [0.1, 0, 0]])            # shares an edge to within 0.1 of a unit edge
    c2 = conform.gencon_clusters(near.reshape(-1, 3), conform.corner_length(near).reshape(-1), 0.2)
    assert c2[1] == c2[4] and c2[2] == c2[7]                       # 0.1 < 0.2 x 1: fused
    far = np.stack([P, P + [1.0, 0, 0] + [0.3, 0, 0]])
    c3 = conform.gencon_clusters(far.reshape(-1, 3), conform.corner_length(far).reshape(-1), 0.2)
    assert len(np.unique(c3)) == 8                                 # 0.3 > 0.2: stay apart


def test_result_does_not_depend_on_the_order_of_the_soup():
    xs = np.linspace(0, 1, 9)
    P, truth = _soup(xs, xs)
    X = P.reshape(-1, 3) + np.random.default_rng(2).uniform(-1e-12, 1e-12, (P.size // 3, 3))
    L = conform.corner_length(P).reshape(-1)
    perm = np.random.default_rng(3).permutation(len(X))
    a = conform.gencon_clusters(X, L)
    b = conform.gencon_clusters(X[perm], L[perm])
    assert _same_partition(a[perm], b)


def test_bad_arguments():
    with pytest.raises(ValueError, match="connectivity_tol"):
        conform.gencon_clusters(np.zeros((3, 3)), np.ones(3), 0.0)
    with pytest.raises(ValueError, match="length"):
        conform.gencon_clusters(np.zeros((3, 3)), np.ones(2))
    with pytest.raises(ValueError, match="2, 4 or 8"):
        conform.corner_length(np.zeros((1, 3, 3)))


def test_a_conformal_mesh_passes_the_gap_check():
    m = quadmesh.QuadMesh.from_corners(*grid(6))
    rep = quadmesh.check_connectivity(m)
    assert rep.passed and rep.n_gap_clusters == 0 and rep.n_components == 1
    assert "PASS" in str(rep)


def test_an_unwelded_seam_is_reported_with_the_points_and_their_location():
    """Two 4x4 blocks side by side whose shared edge is duplicated 1e-3 of a cell apart."""
    A, qa = grid(4)
    B, qb = grid(4)
    B = B + [1.0 + 1e-3 / 4, 0, 0]                                   # block B's left edge is 0.25e-3 off block A's right edge
    P = np.vstack([A, B])
    q = np.vstack([qa, qb + len(A)])
    m = quadmesh.QuadMesh.from_corners(P, q)
    rep = quadmesh.check_connectivity(m)
    assert not rep.passed
    assert rep.n_gap_clusters == 5 and rep.gap_points.size == 10      # the 5 seam vertices, each stored twice
    assert rep.n_gap_on_boundary == 10                                 # an open seam: all on the boundary
    assert 0 < rep.worst_gap_in_lengths < 0.2
    assert rep.n_components == 2 and "gencon" in str(rep)
    assert (np.abs(P[rep.gap_points][:, 0] - 1.0) < 1e-2).all()      # all of them on the seam


def test_a_seam_gap_wider_than_the_tolerance_is_a_real_gap_not_a_defect():
    A, qa = grid(4)
    B, qb = grid(4)
    B = B + [1.0 + 0.5 / 4, 0, 0]                                      # half a cell apart: a deliberate gap
    m = quadmesh.QuadMesh.from_corners(np.vstack([A, B]), np.vstack([qa, qb + len(A)]))
    rep = quadmesh.check_connectivity(m)
    assert rep.passed and rep.n_components == 2 and any("disconnected" in n for n in rep.notes)


def test_degenerate_and_nonmanifold_are_counted():
    P, q = grid(3)
    qd = q.copy()
    qd[0] = [q[0, 0], q[0, 1], q[0, 1], q[0, 3]]        # names one point twice
    assert quadmesh.check_connectivity(quadmesh.QuadMesh.from_corners(P, qd)).n_degenerate_quads == 1
    # a third quad glued onto the interior edge (5, 6) makes that edge borne by three quads
    P2 = np.vstack([P, [[0.33, 0.33, 1.0], [0.66, 0.33, 1.0]]])
    q3 = np.vstack([q, [[5, 6, 17, 16]]])
    rep = quadmesh.check_connectivity(quadmesh.QuadMesh.from_corners(P2, q3))
    assert rep.n_nonmanifold_edges == 1 and not rep.passed


def test_a_tangle_is_not_mistaken_for_a_seam():
    """Two non-adjacent vertices pushed close together (a fold) are chained by gencon's box rule but
    are not a seam: the report must say 'tangle' and must not ask for a weld."""
    P, q = grid(10)
    c = 5 * 11 + 5
    P[c, :2] = P[c + 12, :2] + [0.004, 0.002]          # one vertex dragged to sit beside its diagonal neighbour
    rep = quadmesh.check_connectivity(quadmesh.QuadMesh.from_corners(P, q))
    assert rep.n_gap_clusters == 0 and rep.passed
    assert rep.n_chained_clusters >= 1 and any("tangle" in n for n in rep.notes)


def test_real_duplicates_are_still_reported_after_the_direct_check():
    A, qa = grid(4)
    B, qb = grid(4)
    m = quadmesh.QuadMesh.from_corners(np.vstack([A, B + [1.0 + 2e-3 / 4, 0, 0]]), np.vstack([qa, qb + len(A)]))
    rep = quadmesh.check_connectivity(m)
    assert not rep.passed and rep.n_gap_clusters == 5 and rep.n_chained_clusters == 0


def test_a_thin_layer_between_coarse_cells_is_not_flagged_whatever_the_element_order():
    """Found on the real case_bz slices: a 1 mm layer between coarse cells gave '15 chained groups' on a valid
    mesh, because the local length was taken per corner *copy* (a coarse neighbour's copy is lenient).  The
    local length of a vertex is the minimum over every quad that shares it."""
    xs = np.array([0, .04, .08, .12, .16])
    ys = np.array([0, .0075, .0085, .0485, .0885])               # rows 7.5 mm | 1 mm | 40 mm | 40 mm
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), 0 * X.ravel()], 1)
    idx = np.arange(len(xs) * len(ys)).reshape(len(xs), len(ys))
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    misfired = 0
    for seed in range(12):
        m = quadmesh.QuadMesh.from_corners(P, q[np.random.default_rng(seed).permutation(len(q))])
        rep = quadmesh.check_connectivity(m)
        assert rep.passed and rep.n_gap_clusters == 0 and rep.n_chained_clusters == 0
        soup = P[m.corners]
        cl = conform.gencon_clusters(soup.reshape(-1, 3), conform.corner_length(soup).reshape(-1), 0.2)
        key = np.unique(np.stack([cl, m.corners.reshape(-1)], 1), axis=0)
        misfired += int((np.bincount(key[:, 0]) > 1).any())
    assert misfired > 0          # the per-copy rule really would have flagged some orderings


def _weld(P, q, a, b):
    """Merge point ``b`` into ``a``; ids above ``b`` shift down."""
    q = np.where(q == b, a, q)
    return np.delete(P, b, 0), np.where(q > b, q - 1, q)


def test_pinch_vertex_is_found_and_fails_the_check():
    """Two patches touching at one vertex: every edge is fine, the gap check used to pass; GB fires."""
    A, qa = grid(3)
    B, qb = grid(3)
    B = B + [3.0, 3.0, 0.0]
    P, q = _weld(np.vstack([A, B]), np.vstack([qa, qb + len(A)]), len(A) - 1, len(A))
    m = quadmesh.QuadMesh.from_corners(P, q)
    assert list(quadmesh.pinch_vertices(m)) == [len(A) - 1]
    rep = quadmesh.check_connectivity(m)
    assert rep.n_pinch_vertices == 1 and not rep.passed and any("pinch" in n for n in rep.notes)
    assert rep.n_nonmanifold_edges == 0 and rep.n_degenerate_quads == 0 and rep.n_gap_clusters == 0
    assert abs(quadmesh.gauss_bonnet_residual(m)) > 1.0


def test_pinched_ring_is_found():
    """A ring around a 2x2 hole with the hole closed on its diagonal."""
    G, Q = grid(4)
    keep = [i for i in range(16) if i not in (5, 6, 9, 10)]
    P, q = _weld(G, Q[keep], 1 * 5 + 1, 3 * 5 + 3)
    m = quadmesh.QuadMesh.from_corners(P, q)
    assert list(quadmesh.pinch_vertices(m)) == [6]


def test_no_pinch_on_manifold_meshes():
    P, q = grid(5)
    assert quadmesh.pinch_vertices(quadmesh.QuadMesh.from_corners(P, q)).size == 0
    P, q = grid(4)
    keep = [i for i in range(16) if i != 5]                            # a hole is not a pinch
    assert quadmesh.pinch_vertices(quadmesh.QuadMesh.from_corners(P, q[keep])).size == 0
    an_sphere = quadmesh.QuadMesh.from_corners(*_cube_sphere())
    assert quadmesh.pinch_vertices(an_sphere).size == 0


def _cube_sphere():
    """A closed quad surface: the six faces of a cube grid, welded."""
    n = 3
    pts, quads, index = [], [], {}
    def vid(p):
        k = tuple(np.round(p, 9))
        if k not in index:
            index[k] = len(pts)
            pts.append(p)
        return index[k]
    t = np.linspace(-1, 1, n + 1)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            for i in range(n):
                for j in range(n):
                    c = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.empty(3)
                        p[axis] = sign
                        p[(axis + 1) % 3], p[(axis + 2) % 3] = t[i + di], t[j + dj]
                        c.append(vid(p))
                    quads.append(c if sign > 0 else c[::-1])
    return np.array(pts), np.array(quads)


def _radial_cube_sphere(n):
    """Closed, welded, outward quad sphere: 6 n x n cube faces projected radially (Q = 6 n^2)."""
    t = np.linspace(-1, 1, n + 1)
    A, B = np.meshgrid(t, t, indexing="ij")
    one = np.ones_like(A)
    faces = [np.stack(v, -1) for v in [(one, A, B), (-one, B, A), (B, one, A), (A, -one, B), (A, B, one), (B, A, -one)]]
    pts = np.concatenate([f.reshape(-1, 3) for f in faces])
    _, first, inv = np.unique(np.round(pts * n).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    base = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    q = np.concatenate([inv[base + f * (n + 1) ** 2] for f in range(6)])
    P = pts[first] / np.linalg.norm(pts[first], axis=1, keepdims=True)
    out = np.einsum("ij,ij->i", np.cross(P[q[:, 1]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 0]]), P[q].mean(1)) < 0
    q[out] = q[out][:, ::-1]
    return P, q


def test_gencon_alone_overmerges_a_symmetric_lattice_and_the_refinement_does_not():
    """The segment sort chains x, y, z through different points; the exact box rule does not (and is what the
    gap check reports).  Measured on a valid cube-sphere: 8 false clusters of up to 919 points at n=40."""
    P, q = _radial_cube_sphere(40)
    m = quadmesh.QuadMesh.from_corners(P, q)
    soup = P[m.corners]
    X, stored = soup.reshape(-1, 3), m.corners.reshape(-1)
    ev = np.full(m.n_points, np.inf)
    np.minimum.at(ev, stored, conform.corner_length(soup).reshape(-1))
    ell = ev[stored]

    def multi(cl):
        key = np.unique(np.stack([cl, stored], axis=1), axis=0)
        return int((np.bincount(key[:, 0]) > 1).sum())

    raw = conform.gencon_clusters(X, ell)
    assert multi(raw) > 0                                                   # the known over-merge
    assert multi(conform.box_refine_clusters(X, ell, 0.2, raw, stored)) == 0
    rep = quadmesh.check_connectivity(m)
    assert rep.passed and rep.n_chained_clusters == 0 and rep.n_gap_clusters == 0


def test_a_real_gap_survives_the_refinement():
    P, q = _radial_cube_sphere(12)
    m0 = quadmesh.QuadMesh.from_corners(P, q)
    # split vertex 0 into two stored points a tiny step apart: quads 0.. keep id 0, the rest take a copy
    P2 = np.vstack([P, P[0] + 1e-4])
    q2 = q.copy()
    users = np.flatnonzero((q == 0).any(axis=1))
    q2[users[: len(users) // 2]] = np.where(q2[users[: len(users) // 2]] == 0, len(P), q2[users[: len(users) // 2]])
    rep = quadmesh.check_connectivity(quadmesh.QuadMesh.from_corners(P2, q2))
    assert rep.n_gap_clusters == 1 and not rep.passed and m0.n_points == len(P)


def test_check_connectivity_stays_fast_on_a_big_symmetric_mesh():
    import time
    P, q = _radial_cube_sphere(60)                                         # 21,600 quads; was 7 s with the false clusters
    m = quadmesh.QuadMesh.from_corners(P, q)
    t = time.perf_counter()
    quadmesh.check_connectivity(m)
    assert time.perf_counter() - t < 5.0


def test_collapsed_edge_fails_the_check_and_the_session_refuses_it():
    from nekmeshpy.quadmesh.smoother.session import ConnectivityError, SmoothSession
    P, q = grid(4)
    P2 = np.vstack([P, P[6]])                                              # a second stored point exactly on vertex 6
    q2 = q.copy()
    q2[5] = np.where(q2[5] == 7, len(P), q2[5])                            # quad 5 now uses the copy in place of 7
    P2[len(P)] = P[6]                                                      # ... which sits on 6: the 6-copy edge is collapsed
    m = quadmesh.QuadMesh.from_corners(P2, q2)
    rep = quadmesh.check_connectivity(m)
    assert rep.n_collapsed_edges >= 1 and not rep.passed and any("collapsed edge" in n for n in rep.notes)
    with pytest.raises(ConnectivityError):
        SmoothSession(m, [])
    SmoothSession(m, [], allow_gaps=True)                                  # the explicit override still works
