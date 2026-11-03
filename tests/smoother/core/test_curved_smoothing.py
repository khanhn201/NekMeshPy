"""Smoothing a curved (order N) mesh as a curved mesh: node blocks in and out of the storage ladder, the nodes that
follow a vertex move, and the MFEM / TMOP stage that optimises every node.

The block, carry and session tests need no PyMFEM; the ``tmop`` ones skip without it (``mfem.ser`` is the name
to import: ``tests/smoother/mfem/`` shadows ``mfem`` under pytest).  Meshes are tiny: a 12 to 36 hex pipe segment, a
24 quad annulus, order 2 and 3.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core.smoother import mfem as M
from nekmeshpy.core.smoother.blocks import carry_blocks, min_det_blocks, node_fixed
from nekmeshpy.hexmesh.quality import scaled_jacobian as hex_sj
from nekmeshpy.hexmesh.query import boundary_face_ids
from nekmeshpy.hexmesh.query import element_blocks as hex_blocks
from nekmeshpy.hexmesh.smoother import SmoothSession as HexSession
from nekmeshpy.hexmesh.smoother import highorder as hho
from nekmeshpy.hexmesh.smoother import resolve as hex_resolve
from nekmeshpy.hexmesh.smoother import smooth as hs
from nekmeshpy.quadmesh.query import element_blocks as quad_blocks
from nekmeshpy.quadmesh.query import scaled_jacobian as quad_sj
from nekmeshpy.quadmesh.smoother import SmoothSession as QuadSession
from nekmeshpy.quadmesh.smoother import highorder as qho
from nekmeshpy.quadmesh.smoother import resolve as quad_resolve
from nekmeshpy.quadmesh.smoother import smooth as qs


@pytest.fixture
def mfem_ser():
    return pytest.importorskip("mfem.ser")


# -- meshes ---------------------------------------------------------------------------------------------
def pipe(order, layers=1, jitter=0.0, seed=1):
    """A straight curved pipe segment (O-grid section of a circle, 12 hexes a layer) with named caps; ``jitter``
    perturbs the nodes of interior faces and the hex interiors only (stored once each: conformal)."""
    sec = quadmesh.ogrid(linemesh.circle(1.0, 8, order=order), 2, 1)
    m = hexmesh.extrude(sec, 1.0, layers, element_tags="fluid", first_tag="inlet", last_tag="outlet")
    if jitter:
        rng = np.random.default_rng(seed)
        qm = m.quad_mesh
        face = qm.interior.copy()
        inner = ~boundary_face_ids(m)
        face[inner] += rng.normal(size=face[inner].shape) * jitter
        qm = quadmesh.QuadMesh(qm.line_mesh, qm.quads, qm.orient, face, qm.element_tags)
        m = hexmesh.HexMesh(qm, m.hexes, m.orient, m.interior + rng.normal(size=m.interior.shape) * jitter,
                            m.element_tags)
    return m


def annulus(order, jitter=0.0, seed=1):
    """A planar curved annulus, 24 quads, interior nodes perturbed in the plane."""
    a = quadmesh.annulus(linemesh.circle(0.5, 8, order=order), linemesh.circle(1.0, 8, order=order), 3)
    if jitter:
        d = np.random.default_rng(seed).normal(size=a.interior.shape) * jitter * np.array([1.0, 1.0, 0.0])
        a = quadmesh.QuadMesh(a.line_mesh, a.quads, a.orient, a.interior + d, a.element_tags)
    return a


def node_table(mesh):
    """Every distinct stored node with its value: ``(ids (E, M), flat (n, 3))`` of the mesh's own storage."""
    pk = hho if isinstance(mesh, hexmesh.HexMesh) else qho
    return pk.node_ids(mesh), pk._flat(mesh)


def assert_conformal(mesh):
    """Every element that references a stored node reads the same value there (to rounding)."""
    ids = (hho if isinstance(mesh, hexmesh.HexMesh) else qho).node_ids(mesh)
    blocks = (hex_blocks if isinstance(mesh, hexmesh.HexMesh) else quad_blocks)(mesh)
    first = {}
    worst = 0.0
    for i, b in zip(ids.reshape(-1), blocks.reshape(-1, 3)):
        worst = max(worst, float(np.abs(first.setdefault(int(i), b) - b).max()))
    assert worst < 1e-13


# -- node blocks <-> the ladder -------------------------------------------------------------------------------
@pytest.mark.parametrize("order", [2, 3])
def test_hex_blocks_round_trip_is_exact(order):
    m = pipe(order, layers=2, jitter=0.01)
    f, o = m.hexes.reshape(-1), m.orient.reshape(-1)               # a shared face read in two different frames
    assert any(len(np.unique(o[f == k])) == 2 for k in np.unique(f[np.bincount(f)[f] == 2]))
    out = hho.with_blocks(m, hex_blocks(m))
    assert out.order == order and out is not m
    np.testing.assert_array_equal(hex_blocks(out), hex_blocks(m))
    for a, b in ((out.points, m.points), (out.edge_nodes, m.edge_nodes), (out.face_nodes, m.face_nodes),
                 (out.interior, m.interior)):
        np.testing.assert_array_equal(a, b)
    for a, b in ((out.corners, m.corners), (out.hexes, m.hexes), (out.orient, m.orient)):
        np.testing.assert_array_equal(a, b)
    assert list(out.element_tags.to_dense()) == list(m.element_tags.to_dense())
    assert list(out.face_tags.to_dense()) == list(m.face_tags.to_dense())
    assert out.face_tags.unique() == m.face_tags.unique() != []
    assert_conformal(m)


@pytest.mark.parametrize("order", [2, 3])
def test_quad_blocks_round_trip_is_exact(order):
    a = annulus(order, jitter=0.01)
    out = qho.with_blocks(a, quad_blocks(a))
    np.testing.assert_array_equal(quad_blocks(out), quad_blocks(a))
    for x, y in ((out.points, a.points), (out.edge_nodes, a.edge_nodes), (out.interior, a.interior),
                 (out.corners, a.corners), (out.orient, a.orient)):
        np.testing.assert_array_equal(x, y)
    assert_conformal(a)


def test_blocks_written_back_land_in_the_shared_storage_once():
    """A moved shared node is one stored value: both hexes then read it, with no coordinate search."""
    m = pipe(2, layers=2)
    ids, flat = node_table(m)
    B = hex_blocks(m)
    shared = np.flatnonzero(np.bincount(ids.reshape(-1)) > 1)
    pick = int(shared[len(shared) // 2])
    B2 = B.copy()
    B2[ids == pick] += 0.01
    out = hho.with_blocks(m, B2)
    assert_conformal(out)
    assert int((np.abs(hex_blocks(out) - B).max(axis=2) > 0).sum()) == int((ids == pick).sum())


def test_node_fixed_holds_a_node_when_every_vertex_of_its_entity_does():
    # one hex of order 3: hold the four bottom vertices only
    cells = np.arange(8)[None, :]
    vf = np.zeros((8, 3), bool)
    vf[:4] = True
    held = node_fixed(vf, cells, 3, 3)[0]
    k = np.arange(64) // 16
    assert (held.all(axis=1) == (k == 0)).all() and (held.any(axis=1) == (k == 0)).all()    # the bottom face, edges too
    idx = np.arange(64)
    inner = ((idx % 4 % 3 != 0) & (idx // 4 % 4 % 3 != 0) & (idx // 16 % 3 != 0))
    assert inner.sum() == 8 and not held[inner].any()                  # half the corners held: the interior is free
    vf[:] = True                                    # every vertex held (a frozen element): the interior is held too
    assert node_fixed(vf, cells, 3, 3)[0].all()
    # a plane (z only) holds the z of the face nodes whose vertices all hold it
    vz = np.zeros((8, 3), bool)
    vz[:4, 2] = True
    h = node_fixed(vz, cells, 2, 3)[0]
    assert (h[:, 2] == (np.arange(27) // 9 == 0)).all() and not h[:, :2].any()


@pytest.mark.parametrize("order", [2, 3])
def test_min_det_blocks_of_a_straight_block_is_its_volume(order):
    sec = quadmesh.rectangle(np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], float), 1, 1, order=order)
    h = hexmesh.extrude(sec, 3.0, 1)
    d = min_det_blocks(hex_blocks(h), order, 3, 4)
    np.testing.assert_allclose(d, 6.0, rtol=1e-12)
    d2 = min_det_blocks(quad_blocks(sec)[:, :, :2], order, 2, 4)
    np.testing.assert_allclose(d2, 2.0, rtol=1e-12)


# -- carrying the curved nodes through a linear stage ---------------------------------------------------------
def test_carry_moves_every_node_by_the_trilinear_vertex_displacement_and_stays_conformal():
    m = pipe(3, layers=2, jitter=0.004)
    P = np.array(m.points)
    rng = np.random.default_rng(5)
    free = np.flatnonzero(~np.isclose(np.hypot(P[:, 0], P[:, 1]), 1.0, atol=1e-9))   # off the wall: the centre column
    P2 = P.copy()
    P2[free[:1]] += rng.normal(size=(1, 3)) * 0.01                         # one vertex: most hexes are untouched
    moved = hs.with_points(hho.linearize(m), P2)
    out = hho.carry(m, moved)
    assert out.order == 3 and np.array_equal(out.corners, m.corners)
    np.testing.assert_array_equal(out.points, P2)                          # the vertices exactly
    assert_conformal(out)
    d = P2 - P
    expect = carry_blocks(hex_blocks(m), d[m.corners], 3, 3)
    np.testing.assert_allclose(hex_blocks(out), expect, atol=1e-13)
    still = ~d[m.corners].any(axis=(1, 2))
    assert still.any() and (hex_blocks(out)[still] == hex_blocks(m)[still]).all()      # untouched hexes, bit for bit
    wall = np.isclose(np.hypot(*hex_blocks(m)[..., :2].transpose(2, 0, 1)), 1.0, atol=1e-9)
    assert (hex_blocks(out)[wall] == hex_blocks(m)[wall]).all()                        # held wall nodes: bit for bit


def test_a_native_stage_on_a_curved_input_keeps_it_curved_and_conformal():
    m = pipe(3, layers=2, jitter=0.004)
    r = hex_resolve(m, [], boundary="frozen")
    out, logs = hs.run_schedule(m, r, [("laplace", {"passes": 3}), ("optimize", {"iters": 5})])
    assert out.order == 3 and [lg.method for lg in logs] == ["laplace", "optimize"]
    assert np.abs(out.points - m.points).max() > 0
    assert_conformal(out)
    assert list(out.element_tags.to_dense()) == list(m.element_tags.to_dense())
    assert list(out.face_tags.to_dense()) == list(m.face_tags.to_dense())
    a = annulus(3, jitter=0.01)
    ra = quad_resolve(a, [], boundary="frozen")
    qout, qlogs = qs.run_schedule(a, ra, [("laplace", {"passes": 2}), ("optimize", {"iters": 3})])
    assert qout.order == 3 and len(qlogs) == 2 and np.abs(qout.points - a.points).max() > 0
    assert_conformal(qout)


def test_a_session_on_a_curved_hex_mesh_keeps_its_order_and_reads_the_curved_nodes():
    m = pipe(2, layers=2, jitter=0.004)
    s = HexSession(m, [], boundary="frozen", sample_order=5)
    assert s.current is m and s.original is m
    rep = s.run(1, schedule=[("laplace", {"passes": 2}), ("optimize", {"iters": 3})])[0]
    assert rep.passed and s.current.order == 2
    np.testing.assert_allclose(rep.before.sj_min, hex_sj(m, 2).min())              # the curved reading, not the corners
    np.testing.assert_allclose(rep.after.sj_min, hex_sj(s.current, 2).min())
    assert rep.curved_after is not None and rep.curved_after.orders == (5,) and "curved map" in str(rep)
    assert rep.corners_before[0] == 0 and not np.isnan(rep.corners_after[1])
    assert s.undo(1) is m


def test_keep_curved_false_smooths_the_straight_shadow_instead():
    m = pipe(2, layers=1, jitter=0.004)
    s = HexSession(m, [], keep_curved=False, sample_order=4)
    assert s.original is m and s.current.order == 1
    rep = s.run(1, schedule=[("laplace", {"passes": 1})])[0]
    assert s.current.order == 1 and rep.curved_before is not None and "result is linear" in str(rep)


def test_a_session_on_a_curved_quad_mesh_keeps_its_order():
    a = annulus(3, jitter=0.01)
    s = QuadSession(a, [], boundary="frozen", check_overlaps=False, sample_order=5)
    assert s.current is a and s.original is a
    rep = s.run(1, schedule=[("laplace", {"passes": 2})])[0]
    assert rep.passed and s.current.order == 3
    np.testing.assert_allclose(rep.before.sj_min, quad_sj(a).min())
    assert any("curved map sampled at order 5" in ln for ln in str(rep).splitlines())
    out, _ = s.rebuild_high_order()                                                  # the old rebuild still works
    assert out.order == 3


# -- the TMOP stage on curved meshes --------------------------------------------------------------------------
def moved_mask(a, b):
    return (a != b).any(axis=-1)


def wall_nodes(blocks):
    return np.isclose(np.hypot(blocks[..., 0], blocks[..., 1]), 1.0, atol=1e-9)


def test_tmop_blocks_validates_before_running(mfem_ser):
    m = pipe(2)
    r = hex_resolve(m, [], boundary="frozen")
    B = hex_blocks(m)
    with pytest.raises(ValueError, match="order must be >= 2"):
        M.tmop_blocks(B, m.corners, r, order=1)
    with pytest.raises(ValueError, match="blocks must be"):
        M.tmop_blocks(B[:, :-1], m.corners, r, order=2)
    with pytest.raises(ValueError, match="target"):
        M.tmop_blocks(B, m.corners, r, order=2, target="cube")
    with pytest.raises(ValueError, match="order-1 mesh"):
        M.tmop(m.points, m.corners, r, order=2)
    bad = B.copy()
    bad[:, :, 2] *= -1.0                                     # every hex inside out
    with pytest.raises(ValueError, match="inverted element"):
        M.tmop_blocks(bad, m.corners, r, order=2)


def test_a_node_order_mismatch_is_caught_by_the_geometry_reproduction_check(mfem_ser, monkeypatch):
    m = pipe(2, layers=3, jitter=0.004)
    r = hex_resolve(m, [], boundary="frozen")
    monkeypatch.setattr(M, "_native_to_lex", lambda mfem, fes, order, dim: np.arange((order + 1) ** dim))
    with pytest.raises(RuntimeError, match="node orders disagree|disagree"):
        hs.tmop(m, r, iters=2)


def test_tmop_on_a_curved_hex_pipe_improves_it_and_holds_the_wall(mfem_ser):
    m = pipe(2, layers=3, jitter=0.006)
    r = hex_resolve(m, [], boundary="frozen")
    out, log = hs.tmop(m, r, iters=30)
    assert log.method == "tmop" and log.stopped == "converged" and log.passes_done > 0
    assert out.order == 2 and out is not m
    assert isinstance(log.before, M.BlockReading) and log.after.energy < log.before.energy
    assert log.before.sample_order == 7 and "order 2" in log.reason and "frozen" in log.reason
    sj0, sj1 = hex_sj(m, 2).min(), hex_sj(out, 2).min()
    s70, s71 = hex_sj(m, 7).min(), hex_sj(out, 7).min()
    assert sj1 > sj0 and s71 > s70 > 0
    np.testing.assert_allclose([log.before.sj_min, log.after.sj_min, log.after.sj_sampled], [sj0, sj1, s71])
    assert log.after.n_inverted == 0 and log.after.det_min > 0
    # connectivity, tags and conformality are untouched
    for a, b in ((out.corners, m.corners), (out.hexes, m.hexes), (out.orient, m.orient)):
        np.testing.assert_array_equal(a, b)
    assert list(out.face_tags.to_dense()) == list(m.face_tags.to_dense())
    assert list(out.element_tags.to_dense()) == list(m.element_tags.to_dense())
    assert_conformal(out)
    # the wall and the caps: every node of a boundary face, and every boundary-edge node, is bit for bit
    bf = boundary_face_ids(m)
    np.testing.assert_array_equal(out.face_nodes[bf], m.face_nodes[bf])
    B0, B1 = hex_blocks(m), hex_blocks(out)
    held = node_fixed(np.repeat((r.kind == hs.FROZEN)[:, None], 3, axis=1), m.corners, 2, 3)
    assert held.any() and (B1[held] == B0[held]).all()
    free = ~held.any(axis=2)
    assert moved_mask(B0, B1)[free].any()                    # the interior did move


def test_tmop_on_a_curved_hex_pipe_slides_the_caps_in_their_planes_and_holds_the_curved_wall(mfem_ser):
    m = pipe(3, layers=1, jitter=0.006)
    r = hex_resolve(m, [], boundary="slide")
    assert r.counts()["on_surface"] > 0
    out, log = hs.tmop(m, r, iters=30)
    assert log.stopped == "converged" and out.order == 3
    assert hex_sj(out, 3).min() > hex_sj(m, 3).min() and hex_sj(out, 7).min() > hex_sj(m, 7).min()
    assert log.after.energy < log.before.energy
    B0, B1 = hex_blocks(m), hex_blocks(out)
    for z in (0.0, 1.0):                                     # the two caps: z is held, x and y slide
        cap = np.isclose(B0[..., 2], z, atol=1e-12)
        assert cap.any() and (B1[..., 2][cap] == B0[..., 2][cap]).all()
        inside = cap & ~wall_nodes(B0)
        assert (B1[..., :2][inside] != B0[..., :2][inside]).any()
    wall = wall_nodes(B0)
    assert wall.any() and (B1[wall] == B0[wall]).all()       # the curved wall: held, bit for bit
    assert_conformal(out)


def test_tmop_on_a_curved_quad_annulus_improves_it_and_holds_its_rims(mfem_ser):
    a = annulus(3, jitter=0.006)
    r = quad_resolve(a, [], boundary="frozen")
    out, log = qs.tmop(a, r, iters=30)
    assert out.order == 3 and log.stopped == "converged" and log.after.energy < log.before.energy
    d0, d1 = quad_sj(a).min(), quad_sj(out).min()
    s0, s1 = quad_sj(a, order=7).min(), quad_sj(out, order=7).min()
    assert d1 > d0 and s1 > s0 > 0
    assert log.before.det_min < log.after.det_min and log.after.n_inverted == 0
    B0, B1 = quad_blocks(a), quad_blocks(out)
    rim = np.isclose(np.hypot(B0[..., 0], B0[..., 1]), 0.5, atol=1e-9) | np.isclose(np.hypot(B0[..., 0], B0[..., 1]), 1.0, atol=1e-9)
    assert rim.any() and (B1[rim] == B0[rim]).all()
    np.testing.assert_array_equal(B1[..., 2], B0[..., 2])    # stays in its plane
    assert moved_mask(B0, B1)[~rim].any()
    np.testing.assert_array_equal(out.edge_nodes[quadmesh.boundary_edge_ids(a)], a.edge_nodes[quadmesh.boundary_edge_ids(a)])
    assert_conformal(out)


def test_a_curved_quad_mesh_off_its_plane_is_refused(mfem_ser):
    a = annulus(2)
    d = np.zeros_like(a.interior)
    d[..., 2] = 0.05
    bent = quadmesh.QuadMesh(a.line_mesh, a.quads, a.orient, a.interior + d, a.element_tags)
    with pytest.raises(ValueError, match="not planar at its curved nodes"):
        qs.tmop(bent, quad_resolve(bent, [], boundary="frozen"))


def test_a_schedule_mixes_curved_tmop_and_linear_stages_on_a_curved_mesh(mfem_ser):
    m = pipe(2, layers=2, jitter=0.004)
    r = hex_resolve(m, [], boundary="frozen")
    out, logs = hs.run_schedule(m, r, [("laplace", {"passes": 1}), ("tmop", {"iters": 10}),
                                       ("optimize", {"iters": 2})])
    assert [lg.method for lg in logs] == ["laplace", "tmop", "optimize"] and out.order == 2
    assert isinstance(logs[1].after, M.BlockReading) and not isinstance(logs[0].after, M.BlockReading)
    assert_conformal(out)


def test_a_session_runs_a_tmop_round_on_a_curved_mesh_and_reports_the_curved_nodes(mfem_ser):
    m = pipe(2, layers=2, jitter=0.006)
    s = HexSession(m, [], boundary="frozen", sample_order=5)
    rep = s.run(1, schedule=[("tmop", {"iters": 30})])[0]
    assert rep.passed and s.current.order == 2 and rep.after.sj_min > rep.before.sj_min
    assert rep.curved_after.orders == (5,) and rep.curved_after.min_sj[0] > rep.curved_before.min_sj[0]
    assert rep.n_frozen_moved == 0
    a = annulus(3, jitter=0.006)
    qsess = QuadSession(a, [], boundary="frozen", check_overlaps=False, sample_order=5)
    qrep = qsess.run(1, schedule=[("tmop", {"iters": 30})])[0]
    assert qrep.passed and qsess.current.order == 3 and qrep.after.sj_min > qrep.before.sj_min
