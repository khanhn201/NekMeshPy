"""The MFEM / TMOP bridge: arrays in and out, constraints as essential DOFs, the run itself, the registered stage.

The whole module is skipped without PyMFEM (``pip install mfem``); the lazy-import contract, which must hold
*without* it, is in ``test_mfem_lazy.py``.  Meshes are tiny: every case here runs in well under a second.
"""

import os
import sys

import numpy as np
import pytest

mfem = pytest.importorskip("mfem.ser")

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("quad", "hex"):
    path = os.path.join(_HERE, os.pardir, sub)
    if path not in sys.path:
        sys.path.append(path)

from grids import grid  # noqa: E402
from hexblocks import block, block_arrays  # noqa: E402

from nekmeshpy import hexmesh, quadmesh  # noqa: E402
from nekmeshpy.core.smoother import mfem as M  # noqa: E402
from nekmeshpy.core.smoother.constraints import ON_SURFACE, OnSurface, sphere  # noqa: E402
from nekmeshpy.hexmesh import smoother as hs  # noqa: E402
from nekmeshpy.quadmesh import smoother as qs  # noqa: E402

SJ = hexmesh.quality.corner_scaled_jacobian


def _quad_mesh(n=6, jitter=0.4, seed=1):
    P, Q = grid(n, jitter, seed)
    return quadmesh.QuadMesh.from_corners(P, Q), P, Q


def _graded_block(n=5, growth=1.5, jitter=0.0, seed=0):
    """An ``n^3`` box whose z spacing grows geometrically: a designed, graded block."""
    P, H = block_arrays(n, 0.0)
    z = np.cumsum(np.r_[0.0, growth ** np.arange(n)])
    z = z / z[-1]
    k = np.rint(P[:, 2] * n).astype(int)
    P[:, 2] = z[k]
    if jitter:
        rng = np.random.default_rng(seed)
        inner = ((P > 1e-9) & (P < 1 - 1e-9)).all(axis=1)
        P[inner] += rng.uniform(-jitter, jitter, (int(inner.sum()), 3)) / n
    return P, H


def _center_det(X, cells):
    """det of the multilinear map's Jacobian at the element centre (exact: mean edge vector per axis)."""
    dim = cells.shape[1] // 4 + 1
    pairs, axes = M._corner_edges(dim)
    Xc = X[cells]
    R = M._REF[dim]
    cols = [np.mean([(R[j, d] - R[i, d]) * (Xc[:, j] - Xc[:, i]) for (i, j), a in zip(pairs, axes) if a == d],
                    axis=0) for d in range(dim)]
    return np.linalg.det(np.stack(cols, axis=2))


# -- arrays <-> MFEM ---------------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_round_trip_preserves_orientation_attributes_and_jacobians(dim):
    if dim == 2:
        P, C = grid(5, 0.3, 3)
        X = P[:, :2]
    else:
        X, C = block_arrays(3, 0.25, 4)
    attr = (np.arange(len(C)) % 3 + 1).astype(np.int64)
    mesh = M.to_mfem_mesh(X, C, attr)
    X2, C2, attr2 = M.from_mfem_mesh(mesh)
    np.testing.assert_array_equal(X2, X)
    np.testing.assert_array_equal(C2, C)
    np.testing.assert_array_equal(attr2, attr)
    ref = _center_det(X, C)
    assert (ref > 0).all()
    got = []
    for e in range(len(C)):
        T = mesh.GetElementTransformation(e)
        ip = mfem.IntegrationPoint()
        ip.Set3(0.5, 0.5, 0.5) if dim == 3 else ip.Set2(0.5, 0.5)
        T.SetIntPoint(ip)
        got.append(T.Jacobian().Det())
    np.testing.assert_allclose(got, ref, rtol=1e-12)
    assert M.min_det(X, C) > 0


def test_a_mirrored_mesh_reads_negative_and_is_not_silently_repaired():
    X, C = block_arrays(2, 0.0)
    X = X.copy()
    X[:, 0] = 1.0 - X[:, 0]                              # a reflection: every Jacobian flips
    mesh = M.to_mfem_mesh(X, C)
    X2, C2, _ = M.from_mfem_mesh(mesh)
    np.testing.assert_array_equal(C2, C)
    assert M.min_det(X, C) < 0
    with pytest.raises(ValueError, match="inverted"):
        M.tmop(X, C, hs.resolve(hexmesh.HexMesh.from_corners(X, C), []))


def test_corner_table_is_a_permutation_and_arrays_are_validated():
    for dim, k in ((2, 4), (3, 8)):
        assert sorted(M.NEK_TO_MFEM[dim].tolist()) == list(range(k))
    with pytest.raises(ValueError, match="1-based"):
        M.to_mfem_mesh(np.zeros((4, 2)) + np.eye(4)[:, :2], np.array([[0, 1, 2, 3]]), np.array([0]))
    with pytest.raises(ValueError, match="quads"):
        M.tmop(np.zeros((6, 3)), np.zeros((1, 6), dtype=np.int64), hs.resolve(block(1), []))


# -- the planar frame --------------------------------------------------------------------------------
def test_plane_frame_axis_plane_keeps_its_axes_and_a_tilted_plane_round_trips():
    P, Q = grid(4, 0.0)
    f = M.plane_frame(P, Q)
    np.testing.assert_array_equal(f.basis, np.eye(3)[:, :2])
    # the same mesh turned into the xz plane (flipped so the quads keep area) and into a tilted plane
    Pxz = P[:, [0, 2, 1]]
    fx = M.plane_frame(Pxz, Q)
    assert fx.basis.shape == (3, 2) and np.abs(fx.basis[1]).max() == 0.0
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    Pt = P @ R.T + np.array([0.3, -0.2, 0.7])
    ft = M.plane_frame(Pt, Q)
    U = (Pt - ft.origin) @ ft.basis
    np.testing.assert_allclose(ft.origin + U @ ft.basis.T, Pt, atol=1e-12)
    a, b = U[Q[:, 1]] - U[Q[:, 0]], U[Q[:, 3]] - U[Q[:, 0]]
    assert (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0).all()


def test_a_non_planar_quad_mesh_is_refused_with_the_reason():
    P, Q = grid(4, 0.0)
    P = P.copy()
    P[:, 2] = 0.2 * np.sin(3 * P[:, 0])
    m = quadmesh.QuadMesh.from_corners(P, Q)
    with pytest.raises(ValueError, match="not planar.*optimize"):
        M.tmop(P, Q, qs.resolve(m, []))


def test_a_planar_quad_mesh_off_the_xy_plane_is_smoothed_in_its_own_plane():
    P, Q = grid(6, 0.4, 1)
    R = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0]
    Pt = P @ R.T + 0.5
    m = quadmesh.QuadMesh.from_corners(Pt, Q)
    Pn, log = M.tmop(Pt, Q, qs.resolve(m, [], boundary="frozen"))
    assert log.stopped in ("converged", "completed")
    off = Pn - Pt
    assert np.abs(off @ (R @ [0, 0, 1])).max() < 1e-12          # never left the plane
    assert np.linalg.norm(off, axis=1).max() > 1e-3             # and did move


# -- constraints -> essential components -------------------------------------------------------------
def test_dof_mapping_frozen_planes_lines_and_held_targets():
    P, H = block_arrays(3, 0.0)
    m = hexmesh.HexMesh.from_corners(P, H)
    r = hs.resolve(m, [], boundary="slide")
    fm = M.fixed_components(P, r)
    assert fm.fixed.shape == (len(P), 3)
    on_plane = (np.abs(P[:, :, None] - np.array([0.0, 1.0])[None, None, :]) < 1e-12).any(axis=2).sum(axis=1)
    np.testing.assert_array_equal(fm.fixed.sum(axis=1), on_plane)      # face -> 1, edge -> 2, corner -> 3, interior 0
    assert fm.counts["frozen"] == 8 and fm.counts["held_curved"] == 0
    # a corner holds every component; a face centre exactly the one normal to its face
    face = np.flatnonzero(np.isclose(P[:, 0], 0.0) & (P[:, 1] > 0) & (P[:, 1] < 1) & (P[:, 2] > 0) & (P[:, 2] < 1))
    assert (fm.fixed[face] == [True, False, False]).all()
    # a sphere is not an axis-aligned plane, even where its normal happens to be an axis: held fixed
    ids = np.flatnonzero((np.abs(P - 0.5).max(axis=1) < 0.4) & (np.abs(P - 0.5).max(axis=1) > 0))
    r2 = hs.resolve(m, [OnSurface(ids, sphere((0.5, 0.5, 0.5), 0.6))], boundary="frozen")
    fm2 = M.fixed_components(P, r2)
    assert (r2.kind[ids] == ON_SURFACE).all()
    assert fm2.fixed[ids].all() and fm2.counts["held_curved"] == len(ids)
    assert "not supported yet" in " ".join(fm2.notes)


def test_dof_mapping_snaps_to_the_target_first():
    P, H = block_arrays(3, 0.0)
    m = hexmesh.HexMesh.from_corners(P, H)
    r = hs.resolve(m, [], boundary="slide")
    Pb = P.copy()
    face = np.flatnonzero(np.isclose(P[:, 0], 0.0) & (P[:, 1] > 0) & (P[:, 1] < 1) & (P[:, 2] > 0) & (P[:, 2] < 1))
    Pb[face, 0] = 1e-4                                         # a hair off its plane
    fm = M.fixed_components(Pb, r)
    np.testing.assert_allclose(fm.points[face, 0], 0.0, atol=1e-15)


# -- the runs ----------------------------------------------------------------------------------------
def test_perturbed_quad_patch_improves_and_frozen_and_sliding_vertices_hold_exactly():
    m, P, Q = _quad_mesh()
    r = qs.resolve(m, [], boundary="slide")
    nrm = qs.reference_normals(P, Q)
    Pn, log = M.tmop(P, Q, r, quality=lambda X: qs.quality(X, Q, nrm))
    assert log.stopped == "converged" and log.after.n_inverted == 0
    assert log.after.sj_min > log.before.sj_min + 0.3
    assert "energy" in log.reason
    frozen = r.kind == 3
    np.testing.assert_array_equal(Pn[frozen], P[frozen])               # bit-identical
    bnd = (P[:, 0] == 0) | (P[:, 0] == 1)
    np.testing.assert_array_equal(Pn[bnd, 0], P[bnd, 0])               # sliding along the line: x exactly kept
    bnd = (P[:, 1] == 0) | (P[:, 1] == 1)
    np.testing.assert_array_equal(Pn[bnd, 1], P[bnd, 1])
    assert np.abs(Pn[:, 2]).max() == 0.0


def test_hex_block_improves_with_every_target_and_the_planes_hold_exactly():
    P, H = block_arrays(4, 0.3, 2)
    m = hexmesh.HexMesh.from_corners(P, H)
    r = hs.resolve(m, [], boundary="slide")
    s0 = SJ(P, H).min()
    for target in ("input", "input_smoothed", "ideal"):
        Pn, log = M.tmop(P, H, r, target=target, quality=lambda X: hs.quality(X, H))
        assert log.after.n_inverted == 0 and SJ(Pn, H).min() > s0 + 0.2, target
        fro = r.kind == 3
        np.testing.assert_array_equal(Pn[fro], P[fro])
        for ax in range(3):
            for val in (0.0, 1.0):
                on = P[:, ax] == val
                np.testing.assert_array_equal(Pn[on, ax], P[on, ax])
    Pl, logl = M.tmop(P, H, r, solver="lbfgs", iters=200, quality=lambda X: hs.quality(X, H))
    assert logl.after.n_inverted == 0 and SJ(Pl, H).min() > s0 + 0.1


def test_target_input_keeps_a_designed_graded_block():
    P, H = _graded_block(5, 1.5)
    m = hexmesh.HexMesh.from_corners(P, H)
    r = hs.resolve(m, [], boundary="slide")
    Pn, log = M.tmop(P, H, r)
    assert np.abs(Pn - P).max() < 1e-9                                 # near no-op: the grading is the target
    P2, H2 = _graded_block(5, 1.5, jitter=0.15, seed=3)
    r2 = hs.resolve(hexmesh.HexMesh.from_corners(P2, H2), [], boundary="slide")
    Pm, _ = M.tmop(P2, H2, r2)
    assert SJ(Pm, H2).min() > SJ(P2, H2).min()
    # the same on a quad graded sheet
    xs = np.cumsum(np.r_[0.0, 1.5 ** np.arange(5)])
    xs /= xs[-1]
    X, Y = np.meshgrid(np.linspace(0, 1, 6), xs, indexing="ij")
    Pq = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    idx = np.arange(36).reshape(6, 6)
    Q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    rq = qs.resolve(quadmesh.QuadMesh.from_corners(Pq, Q), [], boundary="slide")
    Pqn, _ = M.tmop(Pq, Q, rq)
    assert np.abs(Pqn - Pq).max() < 1e-9


def test_designed_elements_keep_their_own_shape_while_the_bulk_is_smoothed():
    P, H = block_arrays(4, 0.3, 1)
    tags = np.where(P[H, 2].min(axis=1) < 0.2, "layer", "bulk")
    m = block(4, 0.3, seed=1, tags=tags)
    r = hs.resolve(m, [], boundary="slide")
    out, log = hs.tmop(m, r, designed="layer")
    assert log.stopped == "converged" and "20 designed" in log.reason
    layer = tags == "layer"
    assert SJ(out.points, H)[~layer].min() > SJ(P, H)[~layer].min()
    with pytest.raises(ValueError, match="no element carries"):
        hs.tmop(m, r, designed="nope")


def test_inverted_input_is_refused_before_mfem_and_313_untangles_it():
    P, H = block_arrays(3, 0.0)
    m = hexmesh.HexMesh.from_corners(P, H)
    r = hs.resolve(m, [], boundary="frozen")
    Pt = P.copy()
    v = np.flatnonzero(np.isclose(P, 1 / 3).all(axis=1))[0]
    Pt[v] = 0.75                                                       # pulled through the opposite corner's cell
    assert SJ(Pt, H).min() <= 0
    with pytest.raises(ValueError, match="313"):
        M.tmop(Pt, H, r)
    with pytest.raises(ValueError, match="ideal"):
        M.tmop(Pt, H, r, metric=313)
    Pn, log = M.tmop(Pt, H, r, metric=313, target="ideal", iters=100)
    assert M.min_det(Pn, H) > 0 and SJ(Pn, H).min() > 0.0 and log.stopped == "converged"


def test_bad_knobs_and_a_curved_order_here_are_refused_clearly():
    P, H = block_arrays(2, 0.2)
    r = hs.resolve(hexmesh.HexMesh.from_corners(P, H), [], boundary="slide")
    with pytest.raises(ValueError, match="tmop_blocks"):
        M.tmop(P, H, r, order=3)
    with pytest.raises(ValueError, match="target"):
        M.tmop(P, H, r, target="cube")
    with pytest.raises(ValueError, match="solver"):
        M.tmop(P, H, r, solver="cg")
    with pytest.raises(ValueError, match="3-D metric|2-D metric"):
        M.tmop(P, H, r, metric=2)


# -- the registered stage and the session ------------------------------------------------------------
def test_tmop_is_a_registered_stage_in_both_rungs_and_runs_in_a_schedule():
    assert "tmop" in qs.methods() and "tmop" in hs.methods()
    m, P, Q = _quad_mesh()
    r = qs.resolve(m, [], boundary="slide")
    out, logs = qs.run_schedule(m, r, [("tmop", {}), ("optimize", {"iters": 3})])
    assert [lg.method for lg in logs] == ["tmop", "optimize"] and logs[0].after.sj_min > logs[0].before.sj_min
    np.testing.assert_array_equal(m.points, P)                         # the input mesh is untouched
    mh = block(4, 0.3, seed=2)
    rh = hs.resolve(mh, [], boundary="slide")
    outh, logh = hs.run_schedule(mh, rh, [("tmop", {"target": "ideal"})])
    assert logh[0].method == "tmop" and logh[0].after.n_inverted == 0
    assert outh.order == 1 and outh.corners.shape == mh.corners.shape


def test_session_runs_a_tmop_stage_for_quad_and_hex():
    m, P, Q = _quad_mesh()
    s = qs.SmoothSession(m, [], boundary="slide", check_overlaps=False)
    rep = s.run(1, schedule=[("tmop", {})])[0]
    assert rep.passed and rep.after.sj_min > rep.before.sj_min
    mh = block(3, 0.3, seed=2)
    sh = hs.SmoothSession(mh, [], boundary="slide", sample_order=2)
    reph = sh.run(1, schedule=[("tmop", {"iters": 20})])[0]
    assert reph.passed and reph.after.sj_min > reph.before.sj_min


def test_a_stage_that_would_add_inverted_elements_is_rolled_back_not_returned():
    P, H = block_arrays(3, 0.2, 1)
    m = hexmesh.HexMesh.from_corners(P, H)
    r = hs.resolve(m, [], boundary="slide")
    calls = []

    def fake_quality(X):                  # a reading that claims the run made things worse
        calls.append(1)
        return hs.quality(X, H)._replace(n_inverted=len(calls) - 1)

    Pn, log = M.tmop(P, H, r, quality=fake_quality)
    assert log.stopped == "rolled_back" and "refused" in log.reason
    np.testing.assert_array_equal(Pn, P)
