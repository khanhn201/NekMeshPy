"""Unit tests for the QuadMesh section factories (structured / ogrid /
half_ogrid / annulus).  All boundary inputs are 3-D ``(N,3)`` coordinates."""

import numpy as np
import pytest

from nekmeshpy import Curve, CurveLoop, QuadMesh
from nekmeshpy.model.fields import geometric_spacing, uniform_spacing


def _rect_edges(x0, x1, y0, y1, nx=1, ny=1):
    # structured takes the edges' own nodes (no resampling): sample each straight
    # edge to the wanted division count (uniform).
    c0, c1, c2, c3 = ((x0, y0, 0.0), (x1, y0, 0.0),
                      (x1, y1, 0.0), (x0, y1, 0.0))
    return [Curve([c0, c1]).resample(uniform_spacing(nx)),
            Curve([c1, c2]).resample(uniform_spacing(ny)),
            Curve([c2, c3]).resample(uniform_spacing(nx)),
            Curve([c3, c0]).resample(uniform_spacing(ny))]


def test_structured_grid():
    qm = QuadMesh.structured(_rect_edges(-1, 1, -0.5, 0.5, 3, 2))
    assert qm.n_quads == 3 * 2
    assert qm.n_points == (3 + 1) * (2 + 1)
    # every edge on the outline is a wall edge (2*(nx+ny) perimeter edges)
    assert len(qm.boundaries) == 2 * (3 + 2)
    assert np.asarray(qm.points).shape == (12, 3)     # (P,3) coordinates


def test_structured_straight_edges_equal_bilinear():
    # straight edges must reduce the Coons patch to the exact bilinear grid
    qm = QuadMesh.structured(_rect_edges(-1, 1, -0.5, 0.5, 3, 2))
    P = np.asarray(qm.points)
    us = np.linspace(0, 1, 4)
    vs = np.linspace(0, 1, 3)
    C = np.array([[-1, -0.5, 0], [1, -0.5, 0], [1, 0.5, 0], [-1, 0.5, 0]], float)
    expect = np.array([(1 - u) * (1 - v) * C[0] + u * (1 - v) * C[1]
                       + u * v * C[2] + (1 - u) * v * C[3]
                       for u in us for v in vs])
    assert np.allclose(P, expect, atol=1e-12)


def test_structured_uses_edge_nodes_verbatim():
    # structured does not resample: a curved bottom edge's own nodes are the
    # section's bottom row exactly (honoured on the boundary, blended inward).
    nx, ny = 8, 4
    xb = np.linspace(-1, 1, nx + 1)
    bottom = Curve(np.column_stack(                                # nx+1 pts
        [xb, 0.3 * np.sin(np.pi * (xb + 1) / 2), np.zeros(nx + 1)]))
    top = Curve([(1, 1, 0), (-1, 1, 0)]).resample(uniform_spacing(nx))
    right = Curve([(1, bottom.points[-1, 1], 0), (1, 1, 0)]).resample(uniform_spacing(ny))
    left = Curve([(-1, 1, 0), (-1, bottom.points[0, 1], 0)]).resample(uniform_spacing(ny))
    qm = QuadMesh.structured([bottom, right, top, left])
    assert qm.n_quads == nx * ny
    P = np.asarray(qm.points).reshape(nx + 1, ny + 1, 3)
    row = P[:, 0, :]
    assert np.max(np.abs(row[:, 1])) > 0.1
    assert np.allclose(row, bottom.points, atol=1e-12)   # edge nodes used verbatim
    assert not np.any(np.isnan(P))


def test_structured_graded_edges_cluster_toward_walls():
    # non-uniform (symmetric) edge sampling grades the section: cells thin toward
    # both walls, and the exact tensor product is preserved (no skew for a rect).
    g = geometric_spacing(4, 1.4)                    # cells grow away from 0
    xf = np.concatenate([0.5 * g[:-1], 1.0 - 0.5 * g[::-1]])   # symmetric, 9 fracs
    c0, c1, c2, c3 = ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0),
                      (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
    edges = [Curve([c0, c1]).resample(xf), Curve([c1, c2]).resample(xf),
             Curve([c2, c3]).resample(xf), Curve([c3, c0]).resample(xf)]
    qm = QuadMesh.structured(edges)
    P = np.asarray(qm.points).reshape(9, 9, 3)
    # first cell (near wall) is thinner than the middle cell in both directions
    dx = np.diff(P[:, 0, 0])
    assert dx[0] < dx[len(dx) // 2]
    # exact tensor product: column x-coords are constant down each i-line
    assert np.allclose(P[:, :, 0], (-1.0 + 2.0 * xf)[:, None], atol=1e-12)


def _circle(radius, n):
    return CurveLoop.circle(radius, n)


def test_ogrid_counts_and_boundary():
    qm = QuadMesh.ogrid(_circle(0.5, 16), n_side=4, radial=uniform_spacing(3))
    # central n_side^2 + n_radial rings of 4*n_side quads
    assert qm.n_quads == 4 * 4 + 3 * (4 * 4)
    assert qm.n_points == (4 + 1) ** 2 + 3 * (4 * 4)
    assert len(qm.boundaries) == 4 * 4               # outer ring = the wall loop


def test_ogrid_interior_method_repositions():
    boundary = _circle(0.5, 16)
    raw = QuadMesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3))
    smoothed = QuadMesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3),
                              interior_method="conduction")
    r = np.asarray(raw.points)
    s = np.asarray(smoothed.points)
    # same topology, interior points moved, boundary (wall) held fixed
    assert r.shape == s.shape
    assert np.max(np.abs(r - s)) > 1e-9
    bn = raw.boundary_points()
    assert np.allclose(r[bn], s[bn])


def _square_loop(half):
    return CurveLoop([(-half, -half, 0.0), (half, -half, 0.0),
                      (half, half, 0.0), (-half, half, 0.0)])


def test_annulus_counts_and_boundary():
    inner = _circle(0.5, 16)
    qm = QuadMesh.annulus(inner, _square_loop(2.0).radial_match(inner),
                          radial=uniform_spacing(3))
    # N azimuthal x n_radial rings of quads; (n_radial+1) rings of N points
    assert qm.n_quads == 16 * 3
    assert qm.n_points == 16 * (3 + 1)
    assert len(qm.boundaries) == 2 * 16              # inner + outer rings
    assert not np.any(np.isnan(np.asarray(qm.points)))


def test_annulus_extrudes_to_watertight_block():
    from nekmeshpy import HexMesh
    inner = _circle(0.5, 24)
    qm = QuadMesh.annulus(inner, _square_loop(3.0).radial_match(inner),
                          radial=uniform_spacing(4))
    block = HexMesh.extrude(qm, length=1.0, layers=uniform_spacing(2))
    assert block.is_watertight() and block.is_conforming()
    assert float(np.min(block.scaled_jacobian())) > 0.0     # no inverted hexes


def test_extrude_explicit_initial_offsets_block():
    from nekmeshpy import HexMesh
    # an explicit initial layer position > 0 places the near cap partway along the
    # axis: layers=[0.5, 0.75, 1.0] extrudes only the far half of length
    inner = _circle(0.5, 16)
    qm = QuadMesh.annulus(inner, _square_loop(2.0).radial_match(inner),
                          radial=uniform_spacing(2))
    block = HexMesh.extrude(qm, length=2.0, layers=np.array([0.5, 0.75, 1.0]))
    z = block.points[:, 2]
    assert np.isclose(z.min(), 1.0)                 # 0.5 * length, near cap
    assert np.isclose(z.max(), 2.0)                 # 1.0 * length, far cap
    assert block.is_watertight() and block.is_conforming()


def test_extrude_rejects_single_layer_position():
    from nekmeshpy import HexMesh
    # the explicit-initial form needs >= 2 positions (>= 1 layer)
    inner = _circle(0.5, 16)
    qm = QuadMesh.annulus(inner, _square_loop(2.0).radial_match(inner),
                          radial=uniform_spacing(2))
    with pytest.raises(ValueError, match="at least 2 layer positions"):
        HexMesh.extrude(qm, length=1.0, layers=np.array([1.0]))


def test_annulus_interior_method_repositions():
    inner = _circle(0.5, 24)
    outer = _square_loop(3.0).radial_match(inner)
    raw = QuadMesh.annulus(inner, outer, radial=uniform_spacing(4))
    smoothed = QuadMesh.annulus(inner, outer, radial=uniform_spacing(4), interior_method="winslow")
    r, s = np.asarray(raw.points), np.asarray(smoothed.points)
    assert r.shape == s.shape
    assert np.max(np.abs(r - s)) > 1e-9             # interior rings moved
    bn = raw.boundary_points()                       # inner + outer rings held
    assert np.allclose(r[bn], s[bn])


def test_annulus_grading_clusters_toward_inner():
    # a graded radial array (geometric ratio > 1) puts the first ring gap smaller
    # than the last -- clustering rings toward the inner body
    inner, outer = _circle(1.0, 8), _circle(4.0, 8)   # equal counts, index-aligned
    qm = QuadMesh.annulus(inner, outer, geometric_spacing(6, 1.5))
    P = np.asarray(qm.points)
    rad = np.linalg.norm(P[:, :2], axis=1).reshape(7, 8)   # (ring, theta)
    gaps = np.diff(rad.mean(axis=1))
    assert gaps[0] < gaps[-1]


def test_annulus_rejects_mismatched_point_counts():
    # inner/outer are paired by index, so unequal counts are rejected (the 4-point
    # square must be radial_match'd to the inner first)
    with pytest.raises(ValueError, match="equal point counts"):
        QuadMesh.annulus(_circle(0.5, 16), _square_loop(2.0), radial=uniform_spacing(3))


def test_radial_match_aligns_outer_to_inner():
    inner = _circle(0.5, 20)
    outer = _square_loop(2.0).radial_match(inner)
    assert len(outer) == len(inner)                  # one outer point per inner point
    # every matched point lands on the square's boundary (max(|x|,|y|) == half)
    assert np.allclose(np.max(np.abs(outer.points[:, :2]), axis=1), 2.0)
    # and each is radially aligned with its inner point (same direction, no tangle)
    din = inner.points[:, :2]                         # inner dirs about the centroid (0,0)
    dot = np.sum(din * outer.points[:, :2], axis=1)
    assert np.all(dot > 0.0)


def test_annulus_rejects_empty_radial():
    with pytest.raises(ValueError, match="at least 2 layer positions"):
        QuadMesh.annulus(_circle(0.5, 16), _square_loop(2.0), np.array([]))


def test_annulus_rejects_radial_not_reaching_wall():
    with pytest.raises(ValueError, match="last layer position must be 1.0"):
        QuadMesh.annulus(_circle(0.5, 16), _square_loop(2.0), np.array([0.3, 0.6]))


def test_annulus_rejects_non_loop():
    with pytest.raises(TypeError, match="must be a CurveLoop"):
        QuadMesh.annulus(Curve([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), _square_loop(2.0),
                         radial=uniform_spacing(3))


def test_half_ogrid_valid():
    Nt, Nr = 2, 2
    na = 4 * Nt + 1
    ang = np.linspace(np.pi, 0.0, na)                # semicircle A1(-1,0)..A2(1,0)
    arc = Curve(np.column_stack([np.cos(ang), np.sin(ang), np.zeros(na)]))
    spine = Curve([[-1.0, 0, 0], [1.0, 0, 0]])    # the diameter A1..A2
    qm = QuadMesh.half_ogrid(arc, spine, uniform_spacing(2), center_scale=0.5)
    assert qm.n_quads == 2 * Nt * Nt + 4 * Nt * Nr
    assert len(qm.boundaries) == 4 * Nt              # the wall arc
    assert not np.any(np.isnan(np.asarray(qm.points)))


# -- plane-awareness (sections built in any 3-D plane) -----------------------

def _plane_normal(pts):
    """Unit normal of an (assumed) coplanar point set via SVD."""
    d = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(d, full_matrices=False)
    return vh[-1]


def test_circle_normal_places_loop_in_plane():
    # circle(normal=n) lays the loop in the plane through center with normal n,
    # radius preserved, points coplanar
    n = np.array([1.0, 1.0, 1.0])
    n = n / np.linalg.norm(n)
    center = np.array([0.3, -0.2, 0.7])
    loop = CurveLoop.circle(2.0, 32, center=center, normal=n)
    P = loop.points
    assert P.shape == (32, 3)
    # coplanar with the requested plane and correct radius about the center
    assert np.max(np.abs((P - center) @ n)) < 1e-12
    assert np.allclose(np.linalg.norm(P - center, axis=1), 2.0)


def test_circle_default_is_xy_plane():
    # default normal +z reproduces the classic xy circle exactly
    loop = CurveLoop.circle(1.5, 16)
    P = loop.points
    assert np.allclose(P[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(P[:, :2], axis=1), 1.5)


def test_ogrid_on_tilted_plane_is_coplanar_and_extrudes():
    from nekmeshpy import HexMesh
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    boundary = CurveLoop.circle(0.5, 16, normal=n)
    qm = QuadMesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3))
    P = np.asarray(qm.points)
    # every section point lies in the boundary's plane (through its centroid)
    c = P.mean(axis=0)
    assert np.max(np.abs((P - c) @ n)) < 1e-9
    # sweeping along the plane normal yields a valid block
    block = HexMesh.extrude(qm, axis=n, length=1.0, layers=uniform_spacing(2))
    assert block.is_watertight() and block.is_conforming()
    assert float(np.min(block.scaled_jacobian())) > 0.0


def test_annulus_on_tilted_plane_is_coplanar():
    n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    inner = CurveLoop.circle(1.0, 24, normal=n)
    outer = CurveLoop.circle(3.0, 24, normal=n)
    qm = QuadMesh.annulus(inner, outer, radial=uniform_spacing(4))
    P = np.asarray(qm.points)
    c = P.mean(axis=0)
    assert np.max(np.abs((P - c) @ n)) < 1e-9
    assert not np.any(np.isnan(P))


def _saddle_loop(n, amp=0.4):
    """A genuinely non-planar closed loop: the unit circle lifted by
    ``z = amp*cos(2 theta)`` (a saddle / Pringle), sampled densely."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return CurveLoop(np.column_stack([np.cos(th), np.sin(th), amp * np.cos(2 * th)]))


def test_ogrid_on_curvy_boundary_stays_nonplanar():
    # a curvy (non-planar) boundary must NOT be flattened to a plane: the wall ring
    # sits on the true curved surface and conduction lifts the interior onto it.
    amp = 0.4
    boundary = _saddle_loop(400, amp)
    qm = QuadMesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3),
                        interior_method="conduction")
    X = np.asarray(qm.points)
    # the whole section is genuinely non-planar (not snapped to a best-fit plane)
    dev = np.abs((X - X.mean(axis=0)) @ _plane_normal(X))
    assert dev.max() > 0.3                       # ~ the saddle's own z-amplitude
    # the wall ring lies exactly on the analytic saddle surface z = amp*cos(2*theta)
    wall_ids = sorted({i for e in qm.boundaries for i in e})
    wall = X[wall_ids]
    ang = np.arctan2(wall[:, 1], wall[:, 0])
    assert np.max(np.abs(wall[:, 2] - amp * np.cos(2 * ang))) < 1e-4
    assert np.max(np.abs(np.hypot(wall[:, 0], wall[:, 1]) - 1.0)) < 1e-4
    # conduction lifts the interior onto the curved surface (not flat at z=0),
    # holding the boundary ring fixed
    interior = np.setdiff1d(np.arange(len(X)), wall_ids)
    assert np.abs(X[interior, 2]).max() > 0.1
    raw = np.asarray(QuadMesh.ogrid(boundary, n_side=4,
                                    radial=uniform_spacing(3)).points)
    assert np.max(np.abs(raw[interior] - X[interior])) > 1e-3   # interior moved
    assert np.allclose(raw[wall_ids], X[wall_ids])              # wall held fixed


def test_annulus_on_curvy_boundaries_stays_nonplanar():
    # a curvy inner/outer pair blends in 3-D (no projection): the result follows
    # the curved surface rather than collapsing onto a plane.
    inner = _saddle_loop(24, amp=0.4)
    outer = _saddle_loop(24, amp=0.4)
    outer = CurveLoop(2.0 * outer.points)        # scaled-out saddle, same 24 points
    qm = QuadMesh.annulus(inner, outer, radial=uniform_spacing(4),
                          interior_method="conduction")
    X = np.asarray(qm.points)
    dev = np.abs((X - X.mean(axis=0)) @ _plane_normal(X))
    assert dev.max() > 0.3
    assert not np.any(np.isnan(X))


def test_radial_match_in_tilted_plane():
    # radial_match works in other's plane, not just xy
    n = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    inner = CurveLoop.circle(0.5, 20, normal=n)
    outer = CurveLoop.circle(2.0, 4, normal=n).radial_match(inner)
    assert len(outer) == len(inner)
    # matched loop stays coplanar with inner's plane
    c = inner.points.mean(axis=0)
    assert np.max(np.abs((outer.points - c) @ n)) < 1e-9


# -- input validation --------------------------------------------------------

def test_curve_rejects_2d_input():
    # boundaries must be 3-D (N,3); a 2-D (N,2) array is rejected
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        Curve([(0, 0), (1, 0)])
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        CurveLoop([(0, 0), (1, 0), (1, 1)])


def test_structured_rejects_wrong_edge_count():
    with pytest.raises(ValueError, match="exactly 4 edge"):
        QuadMesh.structured(_rect_edges(-1, 1, -1, 1)[:3])


def test_structured_rejects_mismatched_edge_counts():
    # bottom (4 pts) and top (3 pts) disagree on nx -> rejected (no resampling)
    edges = [Curve([(-1, -1, 0), (0, -1, 0), (1, -1, 0)]).resample(uniform_spacing(3)),
             Curve([(1, -1, 0), (1, 1, 0)]),
             Curve([(1, 1, 0), (-1, 1, 0)]).resample(uniform_spacing(2)),
             Curve([(-1, 1, 0), (-1, -1, 0)])]
    with pytest.raises(ValueError, match="bottom and top .* equal point counts"):
        QuadMesh.structured(edges)


def test_structured_rejects_non_curve_edge():
    edges = _rect_edges(-1, 1, -1, 1)
    edges[0] = np.array([[-1, -1, 0], [1, -1, 0]])   # bare array, not a Curve
    with pytest.raises(TypeError, match="must be a Curve"):
        QuadMesh.structured(edges)


def test_structured_rejects_open_loop():
    # four edges that do not share corners -> not a closed loop
    edges = [Curve([(0, 0, 0), (1, 0, 0)]), Curve([(1, 0, 0), (1, 1, 0)]),
             Curve([(1, 1, 0), (0, 1, 0)]), Curve([(0, 1, 0), (0.5, 0.5, 0)])]
    with pytest.raises(ValueError, match="closed loop"):
        QuadMesh.structured(edges)


def test_ogrid_rejects_non_loop_boundary():
    with pytest.raises(TypeError, match="must be a CurveLoop"):
        QuadMesh.ogrid(Curve([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), n_side=4,
                       radial=uniform_spacing(3))


def test_ogrid_rejects_bad_center_scale():
    with pytest.raises(ValueError, match="center_scale in"):
        QuadMesh.ogrid(_circle(0.5, 16), n_side=4, radial=uniform_spacing(3), center_scale=1.5)


def test_ogrid_rejects_bad_radial():
    with pytest.raises(ValueError, match="strictly increasing"):
        QuadMesh.ogrid(_circle(0.5, 16), n_side=4, radial=np.array([1.0, 0.5]))


def _semicircle_arc(Nt):
    na = 4 * Nt + 1
    ang = np.linspace(np.pi, 0.0, na)
    return Curve(np.column_stack([np.cos(ang), np.sin(ang), np.zeros(na)]))


def test_half_ogrid_rejects_bad_arc_count():
    arc = Curve(np.column_stack([np.linspace(-1, 1, 6), np.zeros(6), np.zeros(6)]))
    spine = Curve([[-1.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="4.Ntheta"):
        QuadMesh.half_ogrid(arc, spine, uniform_spacing(2), center_scale=0.5)


def test_half_ogrid_rejects_non_increasing_radial():
    arc = _semicircle_arc(2)
    spine = Curve([[-1.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        QuadMesh.half_ogrid(arc, spine, np.array([1.0, 0.5]), center_scale=0.5)


def test_half_ogrid_rejects_radial_not_reaching_wall():
    arc = _semicircle_arc(2)
    spine = Curve([[-1.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="last layer position must be 1.0"):
        QuadMesh.half_ogrid(arc, spine, np.array([0.3, 0.6]), center_scale=0.5)
