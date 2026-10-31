"""M2 of the quad smoother: constraint resolution, snap, direction projection, on-target check."""

import numpy as np
import pytest
from grids import grid

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.quadmesh.query import boundary_points
from nekmeshpy.quadmesh.smoother import constrain as cn


def _square(n=6, jitter=0.0):
    P, q = grid(n, jitter=jitter, seed=1)
    return quadmesh.QuadMesh.from_corners(P, q)


def test_targets_project_and_report_unit_normals():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3)) * 2
    s = cn.sphere([0.5, -1, 2], 1.5)
    on = s.project(X)
    np.testing.assert_allclose(np.linalg.norm(on - [0.5, -1, 2], axis=1), 1.5, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(s.normal(on), axis=1), 1.0, atol=1e-14)
    p = cn.plane([0, 0, 1], [0, 0, 2])
    np.testing.assert_allclose(p.project(X)[:, 2], 1.0, atol=1e-14)
    ci = cn.circle([0, 0, 0], 2.0)
    q = ci.project(X)
    np.testing.assert_allclose(np.linalg.norm(q[:, :2], axis=1), 2.0, atol=1e-14)
    np.testing.assert_allclose(q[:, 2], 0.0, atol=1e-14)
    np.testing.assert_allclose(np.einsum("ij,ij->i", ci.tangent(q), q), 0.0, atol=1e-14)
    ln = cn.line([0, 0, 0], [1, 1, 0])
    np.testing.assert_allclose(np.cross(ln.project(X), [1, 1, 0])[:, :2], 0.0, atol=1e-14)


def test_resolve_precedence_conflicts_and_default_boundary_freeze():
    m = _square(4)
    n = m.n_points
    bd = boundary_points(m)
    inside = np.setdiff1d(np.arange(n), bd)
    s1, s2 = cn.plane([0, 0, 0], [0, 0, 1]), cn.plane([0, 0, 0], [0, 1, 0])
    r = cn.resolve(m, [cn.OnSurface(inside[:3], s1), cn.OnSurface(inside[2:5], s2),
                       cn.Frozen(inside[6:7]), cn.OnCurve(inside[6:8], cn.line([0, 0, 0], [1, 0, 0]))])
    assert r.kind[inside[0]] == cn.ON_SURFACE and r.kind[inside[1]] == cn.ON_SURFACE
    assert r.kind[inside[2]] == cn.FROZEN                    # two surfaces -> frozen, never implicit order
    assert r.kind[inside[6]] == cn.FROZEN                    # Frozen beats OnCurve
    assert r.kind[inside[7]] == cn.ON_CURVE
    assert (r.kind[bd] == cn.FROZEN).all()                   # unclaimed boundary frozen by default
    assert any("more than one surface" in s for s in r.notes) and any("boundary" in s for s in r.notes)
    assert (cn.resolve(m, [], boundary="free").kind == cn.FREE).all()


def test_snap_projects_fully_or_partly_and_restores_frozen():
    m = _square(4)
    P = np.asarray(m.points, float).copy()
    bd = boundary_points(m)
    inside = np.setdiff1d(np.arange(m.n_points), bd)
    sph = cn.sphere([0, 0, 0], 1.0)
    r = cn.resolve(m, [cn.OnSurface(inside, sph)])
    P[inside] += 0.3
    full = cn.snap(P, r)
    np.testing.assert_allclose(np.linalg.norm(full[inside], axis=1), 1.0, atol=1e-14)
    half = cn.snap(P, r, weight=0.5)
    d0 = np.abs(np.linalg.norm(P[inside], axis=1) - 1.0)
    d1 = np.abs(np.linalg.norm(half[inside], axis=1) - 1.0)
    np.testing.assert_allclose(d1, 0.5 * d0, rtol=1e-6)
    P[bd] += 0.1                                              # frozen boundary was moved
    np.testing.assert_allclose(cn.snap(P, r)[bd], m.points[bd], atol=0)
    assert P is not cn.snap(P, r)                             # a new array, input untouched


def test_direction_projection_uses_the_targets_normal_tangent_and_zeroes_frozen():
    m = _square(4)
    rng = np.random.default_rng(5)
    P = np.asarray(m.points, float).copy()
    bd = boundary_points(m)
    inside = np.setdiff1d(np.arange(m.n_points), bd)
    sph = cn.sphere([0.5, 0.5, -3.0], 3.0)                     # curved target; mesh normal is the plane's
    P[inside] = sph.project(P[inside])
    P[bd[:2]] = cn.circle([0.5, 0.5, 0], 0.7).project(P[bd[:2]])
    r = cn.resolve(m, [cn.OnSurface(inside, sph), cn.OnCurve(bd[:2], cn.circle([0.5, 0.5, 0], 0.7))])
    g = rng.normal(size=P.shape)
    out = cn.constrain_direction(g, P, r)
    nrm = sph.normal(P[inside])
    np.testing.assert_allclose(np.einsum("ij,ij->i", out[inside], nrm), 0.0, atol=1e-13)
    tan = cn.circle([0.5, 0.5, 0], 0.7).tangent(P[bd[:2]])
    np.testing.assert_allclose(out[bd[:2]] - np.einsum("ij,ij->i", out[bd[:2]], tan)[:, None] * tan, 0, atol=1e-13)
    assert (out[r.kind == cn.FROZEN] == 0).all()
    assert not np.array_equal(g, out)                          # a new array


def test_slide_false_projects_once_then_holds():
    m = _square(4)
    bd = boundary_points(m)
    inside = np.setdiff1d(np.arange(m.n_points), bd)
    P = np.asarray(m.points, float).copy()
    sph = cn.sphere([0.5, 0.5, -3.0], 3.0)
    r = cn.resolve(m, [cn.OnSurface(inside, sph, slide=False)])
    assert (r.kind[inside] == cn.FROZEN).all()
    np.testing.assert_allclose(r.anchor[inside], sph.project(P[inside]), atol=1e-14)
    np.testing.assert_allclose(cn.snap(P + 1.0, r)[inside], r.anchor[inside], atol=1e-14)


def test_on_target_report_passes_clean_and_fails_off_target_or_moved_frozen():
    m = _square(4)
    bd = boundary_points(m)
    inside = np.setdiff1d(np.arange(m.n_points), bd)
    P = np.asarray(m.points, float).copy()
    sph = cn.sphere([0.5, 0.5, -3.0], 3.0)
    r = cn.resolve(m, [cn.OnSurface(inside, sph)])
    P = cn.snap(P, r)
    assert cn.on_target_report(P, m.corners, r).passed
    bad = P.copy()
    bad[inside[0]] += [0, 0, 1e-3]
    rep = cn.on_target_report(bad, m.corners, r)
    assert not rep.passed and rep.max_distance_in_edges > 1e-6 and "FAIL" in str(rep)
    moved = P.copy()
    moved[bd[0]] += 1e-3
    assert not cn.on_target_report(moved, m.corners, r).passed


def test_tag_name_selects_the_points_of_the_tagged_edges():
    ring = linemesh.circle(1.0, 16, element_tag="wall")
    sec = quadmesh.ogrid(ring, 4, 3, wall_tag="wall")
    wall = cn.resolve(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))], boundary="free")
    pts = np.flatnonzero(wall.kind == cn.ON_CURVE)
    assert len(pts) == 16
    np.testing.assert_allclose(np.linalg.norm(sec.points[pts], axis=1), 1.0, atol=1e-12)


def test_disk_boundary_slides_on_the_circle_and_stays_on_it():
    ring = linemesh.circle(1.0, 16, element_tag="wall")
    sec = quadmesh.ogrid(ring, 4, 3, wall_tag="wall")
    r = cn.resolve(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))])
    P = np.asarray(sec.points, float).copy()
    rng = np.random.default_rng(2)
    P += rng.normal(size=P.shape) * [0.02, 0.02, 0.0]          # perturb everything, boundary included
    pts = r.kind == cn.ON_CURVE
    g = constrain = cn.constrain_direction(rng.normal(size=P.shape), P, r)
    stepped = cn.snap(P + 0.1 * g, r)                          # step, then strict snap
    rep = cn.on_target_report(stepped, sec.corners, r)
    assert rep.passed and rep.max_distance < 1e-14
    # the step slid points along the circle (angles changed) without leaving it
    ang0 = np.arctan2(P[pts, 1], P[pts, 0])
    ang1 = np.arctan2(stepped[pts, 1], stepped[pts, 0])
    assert np.abs(ang1 - ang0).max() > 1e-4
    assert (constrain[~pts & (r.kind == cn.FROZEN)] == 0).all()


def test_vertex_length_is_the_shortest_incident_edge():
    P, q = grid(3)
    ell = cn.vertex_length(P, q)
    np.testing.assert_allclose(ell, 1.0 / 3.0, atol=1e-14)


def test_keep_layer_freezes_the_rings_off_the_seeds_and_nothing_else():
    m = _square(8)
    seeds = np.flatnonzero(np.isclose(m.points[:, 1], 0.0))               # the y=0 row
    for rings in (0, 1, 2):
        ids = cn.layer_rings(m, seeds, rings)
        want = np.flatnonzero(m.points[:, 1] <= rings / 8 + 1e-12)        # rows 0..rings
        np.testing.assert_array_equal(ids, want)
    r = cn.resolve(m, [cn.KeepLayer(seeds, rings=2)], boundary="free")
    assert (r.kind[m.points[:, 1] <= 2 / 8 + 1e-12] == cn.FROZEN).all()
    assert (r.kind[m.points[:, 1] > 2 / 8 + 1e-12] == cn.FREE).all()
    with pytest.raises(ValueError, match="rings"):
        cn.layer_rings(m, seeds, -1)


def test_keep_layer_preserves_the_layer_exactly_through_a_full_smoothing_run():
    from nekmeshpy.quadmesh.smoother import smooth
    P, q = grid(10, jitter=0.4, seed=3)
    P[:, 1] = np.where(P[:, 1] < 1e-12, 0.0, P[:, 1])
    m = quadmesh.QuadMesh.from_corners(P, q)
    seeds = np.flatnonzero(np.isclose(m.points[:, 1], 0.0))
    r = cn.resolve(m, [cn.KeepLayer(seeds, rings=2)])
    out, logs = smooth.run_schedule(m, r, smooth.user_schedule(outer=3))
    keep = cn.layer_rings(m, seeds, 2)
    np.testing.assert_array_equal(out.points[keep], m.points[keep])         # thickness untouched, bit for bit
    assert not np.allclose(out.points, m.points) and logs[-1].after.n_inverted == 0


def _strip(nx=12, rows=(0.0, 0.0075, 0.0085, 0.0485), seed=2):
    """A strip of rectangles: len(rows) horizontal rows of vertices, uneven spacing along x."""
    rng = np.random.default_rng(seed)
    xs = np.sort(np.r_[0.0, rng.uniform(0.05, 0.95, nx - 1), 1.0])
    X, Y = np.meshgrid(xs, np.asarray(rows), indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), 0 * X.ravel()], 1)
    idx = np.arange(P.shape[0]).reshape(len(xs), len(rows))
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    return quadmesh.QuadMesh.from_corners(P, q), idx


def test_grid_columns_walk_straight_across_the_layers_and_stop_where_they_should():
    m, idx = _strip()
    mid = idx[3, 1]                                              # an interior vertex on row 1 (the interface)
    col = cn.grid_columns(m, [mid], [1, 0, 0])[0]
    np.testing.assert_array_equal(col, np.sort(idx[3, :]))        # the whole column across the rows, nothing else
    col1 = cn.grid_columns(m, [mid], [1, 0, 0], depth_in=1, depth_out=1)[0]
    assert len(col1) == 3 and mid in col1                         # depth 1 each side
    cols = cn.grid_columns(m, idx[2:6, 1], [1, 0, 0])
    assert len(cols) == 4 and all(len(c) == 4 for c in cols)
    assert cn.grid_columns(m, [mid], [0, 1, 0])[0].size >= 2    # along the other axis it walks the row instead


def test_slide_columns_keep_every_cell_a_rectangle_and_even_out_the_spacing():
    from nekmeshpy.quadmesh.smoother import smooth
    m, idx = _strip()
    n_x = idx.shape[0]
    cols = [idx[i, :] for i in range(1, n_x - 1)]                 # every interior column slides along x
    r = cn.resolve(m, [cn.SlideColumns(cols, [1, 0, 0])], boundary="frozen")
    assert (r.kind[idx[1:-1, :]] == cn.TIED).all() and (r.tie[idx[1:-1, :]] >= 0).all()
    out, logs = smooth.run_schedule(m, r, smooth.user_schedule(outer=4))
    P = out.points
    for j in range(idx.shape[1]):                                  # rows stay rows (same y)
        np.testing.assert_allclose(P[idx[:, j], 1], m.points[idx[:, j], 1], atol=1e-13)
    for i in range(1, n_x - 1):                                    # columns stay vertical: one shared x per column
        np.testing.assert_allclose(P[idx[i, :], 0], P[idx[i, 0], 0], atol=1e-13)
    assert logs[-1].after.sj_min > 1 - 1e-9 and logs[-1].after.n_inverted == 0
    gaps0, gaps1 = np.diff(m.points[idx[:, 0], 0]), np.diff(P[idx[:, 0], 0])
    assert gaps1.std() < 0.2 * gaps0.std()                         # the spacing moved toward uniform
    assert np.abs(P[idx[0, :], 0]).max() == 0 and np.abs(P[idx[-1, :], 0] - 1).max() == 0   # the two end columns are frozen


def test_a_column_with_a_frozen_member_is_frozen_as_a_whole():
    m, idx = _strip()
    col = idx[5, :]
    r = cn.resolve(m, [cn.SlideColumns([col], [1, 0, 0]), cn.Frozen([int(col[1])])], boundary="frozen")
    assert (r.kind[col] == cn.FROZEN).all() and (r.tie[col] == -1).all()
    assert any("rigid column" in n for n in r.notes)


def test_grid_columns_within_a_region_ride_through_it_and_take_only_a_row_outside():
    m, idx = _strip(rows=(0.0, 0.0075, 0.0150, 0.0225, 0.0600, 0.1000))
    inside = np.zeros(m.n_points, bool)
    inside[idx[:, :3]] = True            # rows 0..2 are the "region"
    seed = idx[4, 2]                                                           # on the region's top row
    col = cn.grid_columns(m, [seed], [1, 0, 0], within=inside, outside_depth=1)[0]
    np.testing.assert_array_equal(col, np.sort(idx[4, :4]))                    # rows 0,1,2 (region) + row 3 (one outside)
    col2 = cn.grid_columns(m, [seed], [1, 0, 0], within=inside, outside_depth=2)[0]
    np.testing.assert_array_equal(col2, np.sort(idx[4, :5]))
