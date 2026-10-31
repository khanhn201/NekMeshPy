"""The hex optimizer: improvement, protected layers and regions, untangling, local runs."""

import numpy as np
from hexblocks import block, block_arrays

from nekmeshpy import hexmesh
from nekmeshpy.hexmesh.smoother import constrain as cn
from nekmeshpy.hexmesh.smoother import smooth


def test_a_perturbed_block_recovers_with_a_frozen_boundary():
    m = block(6, jitter=0.3, seed=1)
    r = cn.resolve(m, [])
    out, log = smooth.optimize(m, r, iters=25)
    assert log.after.n_inverted == 0 and log.after.sj_min > 0.95 and log.after.objective_mean < 1e-2
    fro = r.kind == cn.FROZEN
    np.testing.assert_array_equal(out.points[fro], m.points[fro])
    assert [h[0] for h in log.history] == list(range(1, log.passes_done + 1))


def test_keep_layer_and_frozen_region_stay_bit_identical():
    P, H = block_arrays(4)
    m = block(4, jitter=0.3, seed=2, tags=list(np.where(P[H].mean(axis=1)[:, 2] > 0.75, "solid", "fluid")))
    floor = np.flatnonzero(hexmesh.boundary_face_ids(m) & (m.points[m.quad_mesh.corners][:, :, 2].max(1) < 1e-12))
    m = hexmesh.tag_faces(m, floor, "wall")
    cons = [cn.KeepLayer("wall", rings=1), cn.frozen_region(m, "solid")]
    r = cn.resolve(m, cons, boundary="slide")
    layer = cn.layer_rings(m, "wall", 1)
    region = cn.frozen_region(m, "solid").points
    out, logs = smooth.run_schedule(m, r, smooth.user_schedule(outer=2, n_opt=10))
    np.testing.assert_array_equal(out.points[layer], m.points[layer])
    np.testing.assert_array_equal(out.points[region], m.points[region])
    assert logs[-1].after.objective_mean < logs[0].before.objective_mean


def test_the_delta_continuation_untangles_what_the_exact_metric_cannot():
    m = block(4, jitter=0.45, seed=0)
    r = cn.resolve(m, [])
    assert smooth.quality(m.points, m.corners).n_inverted > 0
    exact, _ = smooth.optimize(m, r, iters=60, delta=0.0)
    assert smooth.quality(exact.points, exact.corners).n_inverted > 0     # a wall at zero Jacobian
    out, logs = smooth.run_schedule(m, r, smooth.untangle_schedule(iters=40))
    q = smooth.quality(out.points, out.corners)
    assert q.n_inverted == 0 and q.sj_min > 0.5
    assert all(lg.after.n_inverted <= lg.before.n_inverted for lg in logs)


def test_optimize_local_and_zero_weights():
    m = block(5, jitter=0.3, seed=6)
    r = cn.resolve(m, [])
    sel = smooth.failing_elements(m.points, m.corners, sj_target=0.3)
    assert sel.any() and not sel.all()
    out, log = smooth.optimize(m, r, iters=10, elements=sel)
    assert log.method == "optimize_local"
    moved = np.flatnonzero((out.points != m.points).any(axis=1))
    assert set(moved) <= set(np.unique(m.corners[sel]))
    same, log0 = smooth.optimize(m, r, iters=5, weights=np.zeros(m.n_hexes))
    assert log0.after.n_inverted <= log0.before.n_inverted     # only the inversion penalty acts
