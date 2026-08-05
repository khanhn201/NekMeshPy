"""Unit tests for the QuadMesh section factories (structured / ogrid /
half_ogrid / spined_ogrid / annulus).  All boundary inputs are 3-D ``(N,3)``
coordinates."""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.model.fields import geometric_spacing, uniform_spacing


def _rect_edges(x0, x1, y0, y1, nx=1, ny=1):
    # structured takes the edges' own nodes (no resampling): sample each straight
    # edge to the wanted division count (uniform).
    c0, c1, c2, c3 = ((x0, y0, 0.0), (x1, y0, 0.0),
                      (x1, y1, 0.0), (x0, y1, 0.0))
    return [linemesh.line(c0, c1, uniform_spacing(nx)),
            linemesh.line(c1, c2, uniform_spacing(ny)),
            linemesh.line(c2, c3, uniform_spacing(nx)),
            linemesh.line(c3, c0, uniform_spacing(ny))]


def test_structured_grid():
    qm = quadmesh.structured(_rect_edges(-1, 1, -0.5, 0.5, 3, 2))
    assert qm.n_quads == 3 * 2
    assert qm.n_points == (3 + 1) * (2 + 1)
    # no names passed -> no tagged edges; the outline is a topological query
    assert qm.n_edge_tags == 0
    assert quadmesh.boundary_edges(qm).shape[0] == 2 * (3 + 2)   # 2*(nx+ny) perimeter edges
    assert np.asarray(qm.points).shape == (12, 3)     # (P,3) coordinates


def test_structured_straight_edges_equal_bilinear():
    # straight edges must reduce the Coons patch to the exact bilinear grid
    qm = quadmesh.structured(_rect_edges(-1, 1, -0.5, 0.5, 3, 2))
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
    bottom = linemesh.loft(np.column_stack(                                # nx+1 pts
        [xb, 0.3 * np.sin(np.pi * (xb + 1) / 2), np.zeros(nx + 1)]))
    top = linemesh.line((1, 1, 0), (-1, 1, 0), uniform_spacing(nx))
    right = linemesh.line((1, bottom.points[-1, 1], 0), (1, 1, 0), uniform_spacing(ny))
    left = linemesh.line((-1, 1, 0), (-1, bottom.points[0, 1], 0), uniform_spacing(ny))
    qm = quadmesh.structured([bottom, right, top, left])
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
    edges = [linemesh.line(c0, c1, xf), linemesh.line(c1, c2, xf),
             linemesh.line(c2, c3, xf), linemesh.line(c3, c0, xf)]
    qm = quadmesh.structured(edges)
    P = np.asarray(qm.points).reshape(9, 9, 3)
    # first cell (near wall) is thinner than the middle cell in both directions
    dx = np.diff(P[:, 0, 0])
    assert dx[0] < dx[len(dx) // 2]
    # exact tensor product: column x-coords are constant down each i-line
    assert np.allclose(P[:, :, 0], (-1.0 + 2.0 * xf)[:, None], atol=1e-12)


# -- rectangle (structured convenience factory) ------------------------------

def _rect_corners(x0, x1, y0, y1):
    return [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]


def test_rectangle_counts_and_side_tags():
    qm = quadmesh.rectangle(_rect_corners(-1, 1, -0.5, 0.5), 3, 2,
                            side_tags={"bottom": "wall", "left": "inlet"})
    assert qm.n_quads == 3 * 2 and qm.n_points == 4 * 3
    from collections import Counter
    counts = Counter(qm.edge_tags.tags.tolist())
    assert counts == {"wall": 3, "inlet": 2}          # only the named sides tagged


def test_rectangle_matches_manual_structured():
    # rectangle is a thin wrapper over structured: identical points for equal input
    corners = _rect_corners(-1, 1, -0.5, 0.5)
    qm = quadmesh.rectangle(corners, 3, 2)
    manual = quadmesh.structured(_rect_edges(-1, 1, -0.5, 0.5, 3, 2))
    assert np.allclose(qm.points, manual.points)
    assert np.array_equal(qm.quads, manual.quads)


def test_rectangle_grading_via_fracs():
    from nekmeshpy.model.fields import symmetric_spacing
    xf = symmetric_spacing(4, 1.4)
    qm = quadmesh.rectangle(_rect_corners(-1, 1, -1, 1), 4, 4, x_frac=xf, y_frac=xf)
    P = np.asarray(qm.points).reshape(5, 5, 3)
    dx = np.diff(P[:, 0, 0])
    assert dx[0] < dx[len(dx) // 2]                   # clustered toward the walls


# -- box / sphere (closed 3-D surfaces) --------------------------------------

def test_box_is_closed_surface_with_face_tags():
    qm = quadmesh.box(2.0, 4, patch_tags={
        "x_min": "inlet", "x_max": "outlet", "y_min": "bottom",
        "y_max": "top", "z_min": "front", "z_max": "back"})
    assert qm.n_quads == 6 * 4 * 4
    assert quadmesh.boundary_edges(qm).shape[0] == 0          # watertight: no free edges
    assert np.max(np.abs(qm.points), axis=0) == pytest.approx([2.0, 2.0, 2.0])
    from collections import Counter
    counts = Counter(qm.element_tags.dense(qm.n_quads).tolist())
    for side in ("inlet", "outlet", "bottom", "top", "front", "back"):
        assert counts[side] == 4 * 4                  # one face-worth of quads each


def test_box_nonuniform_half_sizes_and_counts():
    qm = quadmesh.box((1.0, 2.0, 3.0), (2, 3, 4))
    # per-axis counts: 2*(nx*ny + ny*nz + nz*nx)
    assert qm.n_quads == 2 * (2 * 3 + 3 * 4 + 4 * 2)
    assert quadmesh.boundary_edges(qm).shape[0] == 0          # still closed and conforming
    assert np.max(np.abs(qm.points), axis=0) == pytest.approx([1.0, 2.0, 3.0])


def test_sphere_pairs_with_box_by_index():
    R, S, N = 0.5, 3.0, 4
    box = quadmesh.box(S, N, patch_tags={"x_max": "outlet"})
    sph = quadmesh.sphere(R, N)
    assert np.array_equal(sph.quads, box.quads)       # identical connectivity
    assert sph.n_points == box.n_points
    assert np.allclose(np.linalg.norm(sph.points, axis=1), R)   # all on the sphere
    assert set(sph.element_tags.dense(sph.n_quads).tolist()) == {"sphere"}


def test_sphere_box_annulus_is_watertight():
    from nekmeshpy.model.fields import uniform_spacing as us
    box = quadmesh.box(3.0, 4)
    sph = quadmesh.sphere(0.5, 4)
    block = hexmesh.annulus(sph, box, radial=us(3))
    assert hexmesh.is_watertight(block) and hexmesh.is_conforming(block)
    assert float(np.min(hexmesh.scaled_jacobian(block))) > 0.0


def test_half_box_is_open_at_the_ground_with_face_tags():
    qm = quadmesh.half_box(2.0, 4, n_vertical=3, rim_tag="ground", patch_tags={
        "x_min": "inlet", "x_max": "outlet",
        "y_min": "front", "y_max": "back", "z_max": "top"})
    # four upright sides (4 x n_vertical each) + the flat lid
    assert qm.n_quads == 4 * (4 * 3) + 4 * 4
    assert np.min(qm.points[:, 2]) == pytest.approx(0.0)      # sits on z = 0
    assert np.max(np.abs(qm.points), axis=0) == pytest.approx([2.0, 2.0, 2.0])
    from collections import Counter
    counts = Counter(qm.element_tags.dense(qm.n_quads).tolist())
    for side in ("inlet", "outlet", "front", "back"):
        assert counts[side] == 4 * 3
    assert counts["top"] == 4 * 4
    # open at the rim, and every free edge is named
    rim = quadmesh.boundary_edges(qm)
    assert rim.shape[0] == 4 * 4
    assert set(qm.edge_tags.tags.tolist()) == {"ground"}


def test_hemisphere_pairs_with_half_box_by_index():
    R, S, N, NV = 0.5, 3.0, 4, 3
    hb = quadmesh.half_box(S, N, n_vertical=NV, patch_tags={"x_max": "outlet"})
    hs = quadmesh.hemisphere(R, N, n_vertical=NV, rim_tag="ground")
    assert np.array_equal(hs.quads, hb.quads)          # identical connectivity
    assert hs.n_points == hb.n_points
    assert np.allclose(np.linalg.norm(hs.points, axis=1), R)   # all on the sphere
    assert np.min(hs.points[:, 2]) > -1e-15                    # upper half only
    assert set(hs.element_tags.dense(hs.n_quads).tolist()) == {"hemisphere"}
    assert set(hs.edge_tags.tags.tolist()) == {"ground"}


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_hemisphere_high_order_nodes_lie_on_the_exact_sphere(order):
    R = 1.25
    hs = quadmesh.hemisphere(R, 3, n_vertical=2, order=order)
    for block in (hs.points, hs.lines.interior.reshape(-1, 3),
                  hs.interior.reshape(-1, 3)):
        if block.size:
            assert np.max(np.abs(np.linalg.norm(block, axis=1) - R)) < 1e-13
    # the rim stays exactly on the ground plane at every order
    assert np.min(hs.points[:, 2]) > -1e-15


def test_hemisphere_half_box_annulus_is_watertight():
    from nekmeshpy.model.fields import uniform_spacing as us
    hb = quadmesh.half_box(3.0, 4, n_vertical=4)
    hs = quadmesh.hemisphere(0.5, 4, n_vertical=4, rim_tag="ground")
    block = hexmesh.annulus(hs, hb, radial=us(3))
    assert hexmesh.is_watertight(block) and hexmesh.is_conforming(block)
    assert float(np.min(hexmesh.scaled_jacobian(block))) > 0.0
    # the open rim sweeps into the ground annulus at z = 0
    assert "ground" in block.face_group_tags


def _circle(radius, n):
    return linemesh.circle(radius, n)


def test_ogrid_counts_and_boundary():
    qm = quadmesh.ogrid(_circle(0.5, 16), n_side=4, radial=uniform_spacing(3))
    # central n_side^2 + n_radial rings of 4*n_side quads
    assert qm.n_quads == 4 * 4 + 3 * (4 * 4)
    assert qm.n_points == (4 + 1) ** 2 + 3 * (4 * 4)
    assert qm.n_edge_tags == 0                       # no wall_name -> untagged
    assert quadmesh.boundary_edges(qm).shape[0] == 4 * 4      # outer ring = the wall loop


def test_ogrid_smoothing_method_repositions():
    boundary = _circle(0.5, 16)
    raw = quadmesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3))
    smoothed = quadmesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3),
                              smoothing_method="conduction")
    r = np.asarray(raw.points)
    s = np.asarray(smoothed.points)
    # same topology, interior points moved, boundary (wall) held fixed
    assert r.shape == s.shape
    assert np.max(np.abs(r - s)) > 1e-9
    bn = quadmesh.boundary_points(raw)
    assert np.allclose(r[bn], s[bn])


def _square_loop(half):
    return linemesh.loft([(-half, -half, 0.0), (half, -half, 0.0),
                      (half, half, 0.0), (-half, half, 0.0)], loop=True)


def _far_box(half, n, side_tags=None):
    # square far-field loop, per-side discretized into n line elements, index-paired
    # by count with an n-point inner loop
    return linemesh.rectangle(2 * half, 2 * half, n, side_tags=side_tags)


def _aligned_circle(radius, n, **kw):
    # circle rotated so its index 0 meets the far-field box's lower-left corner, so
    # the two loops pair index-for-index (radial spokes are not straight)
    return linemesh.circle(radius, n, start_theta=np.arctan2(-1.0, -1.0), **kw)


def test_annulus_counts_and_boundary():
    inner = _circle(0.5, 16)
    qm = quadmesh.annulus(inner, _far_box(2.0, inner.n_points),
                          radial=uniform_spacing(3))
    # N azimuthal x n_radial rings of quads; (n_radial+1) rings of N points
    assert qm.n_quads == 16 * 3
    assert qm.n_points == 16 * (3 + 1)
    assert qm.n_edge_tags == 0                       # no names -> untagged
    assert quadmesh.boundary_edges(qm).shape[0] == 2 * 16     # inner + outer rings
    assert not np.any(np.isnan(np.asarray(qm.points)))


def test_annulus_extrudes_to_watertight_block():
    inner = _aligned_circle(0.5, 24)
    qm = quadmesh.annulus(inner, _far_box(3.0, inner.n_points),
                          radial=uniform_spacing(4))
    block = hexmesh.extrude(qm, length=1.0, layers=uniform_spacing(2))
    assert hexmesh.is_watertight(block) and hexmesh.is_conforming(block)
    assert float(np.min(hexmesh.scaled_jacobian(block))) > 0.0     # no inverted hexes


def test_extrude_explicit_initial_offsets_block():
    # an explicit initial layer position > 0 places the near cap partway along the
    # axis: layers=[0.5, 0.75, 1.0] extrudes only the far half of length
    inner = _circle(0.5, 16)
    qm = quadmesh.annulus(inner, _far_box(2.0, inner.n_points),
                          radial=uniform_spacing(2))
    block = hexmesh.extrude(qm, length=2.0, layers=np.array([0.5, 0.75, 1.0]))
    z = block.points[:, 2]
    assert np.isclose(z.min(), 1.0)                 # 0.5 * length, near cap
    assert np.isclose(z.max(), 2.0)                 # 1.0 * length, far cap
    assert hexmesh.is_watertight(block) and hexmesh.is_conforming(block)


def test_extrude_rejects_single_layer_position():
    # the explicit-initial form needs >= 2 positions (>= 1 layer)
    inner = _circle(0.5, 16)
    qm = quadmesh.annulus(inner, _far_box(2.0, inner.n_points),
                          radial=uniform_spacing(2))
    with pytest.raises(ValueError, match="at least 2 layer positions"):
        hexmesh.extrude(qm, length=1.0, layers=np.array([1.0]))


def test_annulus_smoothing_method_repositions():
    inner = _circle(0.5, 24)
    outer = _far_box(3.0, inner.n_points)
    raw = quadmesh.annulus(inner, outer, radial=uniform_spacing(4))
    smoothed = quadmesh.annulus(inner, outer, radial=uniform_spacing(4), smoothing_method="winslow")
    r, s = np.asarray(raw.points), np.asarray(smoothed.points)
    assert r.shape == s.shape
    assert np.max(np.abs(r - s)) > 1e-9             # interior rings moved
    bn = quadmesh.boundary_points(raw)                       # inner + outer rings held
    assert np.allclose(r[bn], s[bn])


def test_annulus_grading_clusters_toward_inner():
    # a graded radial array (geometric ratio > 1) puts the first ring gap smaller
    # than the last -- clustering rings toward the inner body
    inner, outer = _circle(1.0, 8), _circle(4.0, 8)   # equal counts, index-aligned
    qm = quadmesh.annulus(inner, outer, geometric_spacing(6, 1.5))
    P = np.asarray(qm.points)
    rad = np.linalg.norm(P[:, :2], axis=1).reshape(7, 8)   # (ring, theta)
    gaps = np.diff(rad.mean(axis=1))
    assert gaps[0] < gaps[-1]


def test_annulus_rejects_mismatched_point_counts():
    # inner/outer are paired by index, so unequal counts are rejected (build the
    # outer index-aligned to the inner with linemesh.rectangle(w, h, N) first)
    with pytest.raises(ValueError, match="equal point counts"):
        quadmesh.annulus(_circle(0.5, 16), _square_loop(2.0), radial=uniform_spacing(3))


def test_rectangle_far_field_pairs_with_rotated_circle():
    # a circle rotated to the box's lower-left corner pairs index-for-index with a
    # per-side rectangle far field (equal counts); every outer point on the box
    inner = _aligned_circle(0.5, 20)
    outer = _far_box(2.0, inner.n_points)
    assert len(outer) == len(inner)                  # one outer point per inner point
    # every outer point lands on the square's boundary (max(|x|,|y|) == half)
    assert np.allclose(np.max(np.abs(outer.points[:, :2]), axis=1), 2.0)
    # rough radial alignment: each outer point shares its inner point's half-plane
    # (spokes are not straight, but they never tangle)
    din = inner.points[:, :2]
    dot = np.sum(din * outer.points[:, :2], axis=1)
    assert np.all(dot > 0.0)


def test_annulus_rejects_empty_radial():
    with pytest.raises(ValueError, match="at least 2 layer positions"):
        quadmesh.annulus(_circle(0.5, 16), _square_loop(2.0), np.array([]))


def test_annulus_rejects_radial_not_reaching_wall():
    with pytest.raises(ValueError, match="last layer position must be 1.0"):
        quadmesh.annulus(_circle(0.5, 16), _square_loop(2.0), np.array([0.3, 0.6]))


def test_annulus_rejects_non_loop():
    # closedness is not a stored flag: what annulus needs is that the two rings pair
    # index-for-index, i.e. share identical `lines`.  An open chain over the outer
    # loop's own points has one line element fewer (no wrap row), so the radial
    # blend rejects it structurally.
    outer = _square_loop(2.0)
    chain = linemesh.loft(outer.points * 0.25)
    assert chain.n_lines == outer.n_lines - 1        # the missing wrap row
    with pytest.raises(ValueError, match="identical connectivity"):
        quadmesh.annulus(chain, outer, radial=uniform_spacing(3))


def _diameter_spine(arc, center_scale, radial):
    # the exact straight A1..A2 spine half_ogrid meshes: the caller pre-samples the
    # diameter monotonically A1 -> A2 as [north caps, center fan, south caps] (no
    # resampling inside half_ogrid)
    nt = (arc.n_points - 1) // 4
    s_n, s_s = (1.0 - center_scale) / 2, (1.0 + center_scale) / 2
    fr = np.concatenate([((1.0 - radial[1:]) * s_n)[::-1],
                         np.linspace(s_n, s_s, 2 * nt + 1),
                         s_s + radial[1:] * (1.0 - s_s)])
    e1, e2 = arc.points[0], arc.points[-1]
    return linemesh.loft(e1 + fr[:, None] * (e2 - e1))


def test_half_ogrid_valid():
    Nt, Nr = 2, 2
    na = 4 * Nt + 1
    ang = np.linspace(np.pi, 0.0, na)                # semicircle A1(-1,0)..A2(1,0)
    arc = linemesh.loft(np.column_stack([np.cos(ang), np.sin(ang), np.zeros(na)]))
    spine = _diameter_spine(arc, 0.5, uniform_spacing(2))   # the diameter A1..A2
    qm = quadmesh.half_ogrid(arc, spine, uniform_spacing(2), center_scale=0.5)
    assert qm.n_quads == 2 * Nt * Nt + 4 * Nt * Nr
    assert qm.n_edge_tags == 0                       # no wall_name -> untagged
    # topological perimeter = wall arc + straight diameter, both 4*Nt edges
    assert quadmesh.boundary_edges(qm).shape[0] == 2 * (4 * Nt)
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
    loop = linemesh.circle(2.0, 32, center=center, normal=n)
    P = loop.points
    assert P.shape == (32, 3)
    # coplanar with the requested plane and correct radius about the center
    assert np.max(np.abs((P - center) @ n)) < 1e-12
    assert np.allclose(np.linalg.norm(P - center, axis=1), 2.0)


def test_circle_default_is_xy_plane():
    # default normal +z reproduces the classic xy circle exactly
    loop = linemesh.circle(1.5, 16)
    P = loop.points
    assert np.allclose(P[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(P[:, :2], axis=1), 1.5)


def test_ogrid_on_tilted_plane_is_coplanar_and_extrudes():
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    boundary = linemesh.circle(0.5, 16, normal=n)
    qm = quadmesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3))
    P = np.asarray(qm.points)
    # every section point lies in the boundary's plane (through its centroid)
    c = P.mean(axis=0)
    assert np.max(np.abs((P - c) @ n)) < 1e-9
    # sweeping along the plane normal yields a valid block
    block = hexmesh.extrude(qm, axis=n, length=1.0, layers=uniform_spacing(2))
    assert hexmesh.is_watertight(block) and hexmesh.is_conforming(block)
    assert float(np.min(hexmesh.scaled_jacobian(block))) > 0.0


def test_annulus_on_tilted_plane_is_coplanar():
    n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    inner = linemesh.circle(1.0, 24, normal=n)
    outer = linemesh.circle(3.0, 24, normal=n)
    qm = quadmesh.annulus(inner, outer, radial=uniform_spacing(4))
    P = np.asarray(qm.points)
    c = P.mean(axis=0)
    assert np.max(np.abs((P - c) @ n)) < 1e-9
    assert not np.any(np.isnan(P))


def _saddle_loop(n, amp=0.4):
    """A genuinely non-planar closed loop: the unit circle lifted by
    ``z = amp*cos(2 theta)`` (a saddle / Pringle), sampled densely."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return linemesh.loft(np.column_stack([np.cos(th), np.sin(th), amp * np.cos(2 * th)]), loop=True)


def test_ogrid_on_curvy_boundary_stays_nonplanar():
    # a curvy (non-planar) boundary must NOT be flattened to a plane: the wall ring
    # sits on the true curved surface and conduction lifts the interior onto it.
    amp = 0.4
    boundary = _saddle_loop(4 * 4, amp)              # 4*n_side points, meshed exactly
    qm = quadmesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3),
                        smoothing_method="conduction")
    X = np.asarray(qm.points)
    # the whole section is genuinely non-planar (not snapped to a best-fit plane)
    dev = np.abs((X - X.mean(axis=0)) @ _plane_normal(X))
    assert dev.max() > 0.3                       # ~ the saddle's own z-amplitude
    # the wall ring lies exactly on the analytic saddle surface z = amp*cos(2*theta)
    wall_ids = quadmesh.boundary_points(qm).tolist()      # topological outline (untagged)
    wall = X[wall_ids]
    ang = np.arctan2(wall[:, 1], wall[:, 0])
    assert np.max(np.abs(wall[:, 2] - amp * np.cos(2 * ang))) < 1e-4
    assert np.max(np.abs(np.hypot(wall[:, 0], wall[:, 1]) - 1.0)) < 1e-4
    # conduction lifts the interior onto the curved surface (not flat at z=0),
    # holding the boundary ring fixed
    interior = np.setdiff1d(np.arange(len(X)), wall_ids)
    assert np.abs(X[interior, 2]).max() > 0.1
    raw = np.asarray(quadmesh.ogrid(boundary, n_side=4,
                                    radial=uniform_spacing(3)).points)
    assert np.max(np.abs(raw[interior] - X[interior])) > 1e-3   # interior moved
    assert np.allclose(raw[wall_ids], X[wall_ids])              # wall held fixed


def test_annulus_on_curvy_boundaries_stays_nonplanar():
    # a curvy inner/outer pair blends in 3-D (no projection): the result follows
    # the curved surface rather than collapsing onto a plane.
    inner = _saddle_loop(24, amp=0.4)
    outer = _saddle_loop(24, amp=0.4)
    outer = linemesh.loft(2.0 * outer.points, loop=True)        # scaled-out saddle, same 24 points
    qm = quadmesh.annulus(inner, outer, radial=uniform_spacing(4),
                          smoothing_method="conduction")
    X = np.asarray(qm.points)
    dev = np.abs((X - X.mean(axis=0)) @ _plane_normal(X))
    assert dev.max() > 0.3
    assert not np.any(np.isnan(X))


def test_rectangle_far_field_in_tilted_plane():
    # rectangle builds the outer loop in the requested plane, not just xy
    n = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    inner = linemesh.circle(0.5, 20, normal=n)
    outer = linemesh.rectangle(4.0, 4.0, inner.n_points, normal=n)
    assert len(outer) == len(inner)
    # the box loop stays coplanar with inner's plane (both centered at the origin)
    assert np.max(np.abs(outer.points @ n)) < 1e-9


# -- element tags: tagged on the loop's line elements, carried onto the section
# boundary edges ------------------------------------------------------------

def test_loop_element_tags_length_validated():
    # a closed 3-point loop has 3 line elements; element_tags must match
    with pytest.raises(ValueError, match="element_tags length .* must match lines"):
        linemesh.loft([(0, 0, 0), (1, 0, 0), (1, 1, 0)], element_tags=["a", "b"], loop=True)
    # a matching count (one per line element) is accepted
    assert linemesh.loft([(0, 0, 0), (1, 0, 0), (1, 1, 0)],
                     element_tags=["a", "b", "c"], loop=True).element_tags.dense(3).tolist() == ["a", "b", "c"]


def test_unnamed_rectangle_far_field_stays_untagged():
    # no side_tags -> the box loop carries no element tags (the common path)
    assert linemesh.rectangle(4.0, 4.0, 16).element_group_tags == []


def test_rectangle_far_field_carries_element_tags_by_side():
    outer = linemesh.rectangle(
        12.0, 12.0, 64, side_tags={"bottom": "bottom", "right": "outlet", "top": "top", "left": "inlet"})
    assert outer.n_lines == 64
    assert len(outer.element_tags) == 64      # every line is tagged
    from collections import Counter
    counts = Counter(outer.element_tags.dense(outer.n_lines).tolist())
    # 64 line elements split evenly across the four symmetric box sides
    assert counts == {"bottom": 16, "outlet": 16, "top": 16, "inlet": 16}
    # each output line element's midpoint direction matches the side it was tagged with
    pts = outer.points
    mids = (pts + np.roll(pts, -1, axis=0)) / 2.0
    ang = np.degrees(np.arctan2(mids[:, 1], mids[:, 0]))
    for target, side in ((0, "outlet"), (90, "top"), (180, "inlet"), (-90, "bottom")):
        k = int(np.argmin(np.abs(((ang - target + 180) % 360) - 180)))
        assert outer.element_tags.dense(outer.n_lines)[k] == side


def test_annulus_consumes_outer_loop_element_tags():
    # a tagged outer loop splits the outer ring into distinct sides automatically;
    # the scalar inner_tag still tags the whole inner ring.
    inner = _circle(0.5, 64)
    outer = linemesh.rectangle(
        12.0, 12.0, 64, side_tags={"bottom": "bottom", "right": "outlet", "top": "top", "left": "inlet"})
    qm = quadmesh.annulus(inner, outer, geometric_spacing(6, 1.12),
                          inner_tag="cylinder")
    from collections import Counter
    counts = Counter(qm.edge_tags.tags.tolist())
    # inner ring (64 edges) all "cylinder"; outer ring split 16 per tagged side
    assert counts["cylinder"] == 64
    assert counts["outlet"] == counts["top"] == counts["inlet"] == counts["bottom"] == 16


def test_element_tags_propagate_line_to_hex_faces():
    # the full chain: LineMesh element tags -> QuadMesh boundary edges -> HexMesh faces
    inner = _circle(0.5, 32)
    outer = linemesh.rectangle(
        12.0, 12.0, 32, side_tags={"bottom": "bottom", "right": "outlet", "top": "top", "left": "inlet"})
    section = quadmesh.annulus(inner, outer, uniform_spacing(4), inner_tag="cylinder")
    block = hexmesh.extrude(section, axis=(0.0, 0.0, 1.0), length=1.0,
                            layers=uniform_spacing(2), first_tag="front",
                            last_tag="back")
    assert set(block.face_group_tags) == {
        "cylinder", "inlet", "outlet", "top", "bottom", "front", "back"}
    # the four far-field sides carry equal face counts (symmetric split)
    n = {s: block.face_tags.count(s)
         for s in ("inlet", "outlet", "top", "bottom")}
    assert n["inlet"] == n["outlet"] == n["top"] == n["bottom"] > 0


# -- line-level tagging + upper-overrides-lower precedence -------------------

def test_ogrid_reads_boundary_element_tags():
    # the wall is named at the lowest level -- the boundary loop's element_tags --
    # with no scalar wall_tag.  The wall ring (4*n_side edges) all inherit the tag.
    boundary = linemesh.circle(0.5, 16, element_tags=["wall"] * 16)
    qm = quadmesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3))
    assert qm.n_edge_tags == 4 * 4
    assert set(qm.edge_tags.tags.tolist()) == {"wall"}


def test_ogrid_wall_tag_overrides_boundary_element_tags():
    # a non-empty scalar wall_tag OVERRIDES the loop's element_tags for the whole wall
    boundary = linemesh.circle(0.5, 16, element_tags=["ring"] * 16)
    qm = quadmesh.ogrid(boundary, n_side=4, radial=uniform_spacing(3),
                        wall_tag="override")
    assert set(qm.edge_tags.tags.tolist()) == {"override"}
    assert qm.n_edge_tags == 4 * 4


def _tagged_arc(Nt, tags):
    na = 4 * Nt + 1
    ang = np.linspace(np.pi, 0.0, na)
    return linemesh.loft(np.column_stack([np.cos(ang), np.sin(ang), np.zeros(na)]),
                         element_tags=tags)


def test_half_ogrid_reads_arc_element_tags():
    # the arc wall is named per-segment at the line level (no scalar wall_tag); wall
    # edge k tracks arc segment k, so each side of the arc keeps its own tag.
    Nt = 4
    tags = ["a"] * (2 * Nt) + ["b"] * (2 * Nt)      # first half "a", second half "b"
    arc = _tagged_arc(Nt, tags)
    spine = _diameter_spine(arc, 0.5, uniform_spacing(2))
    qm = quadmesh.half_ogrid(arc, spine, uniform_spacing(2), center_scale=0.5)
    from collections import Counter
    assert Counter(qm.edge_tags.tags.tolist()) == {"a": 2 * Nt, "b": 2 * Nt}


def test_half_ogrid_wall_tag_overrides_arc_element_tags():
    Nt = 2
    arc = _tagged_arc(Nt, ["arc"] * (4 * Nt))
    spine = _diameter_spine(arc, 0.5, uniform_spacing(2))
    qm = quadmesh.half_ogrid(arc, spine, uniform_spacing(2), center_scale=0.5,
                             wall_tag="override")
    assert set(qm.edge_tags.tags.tolist()) == {"override"}
    assert qm.n_edge_tags == 4 * Nt


def _tagged_rect_edges(nx, ny, tags):
    # a unit rectangle whose four edges each carry a single uniform element tag
    c0, c1, c2, c3 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                      (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    return [linemesh.line(c0, c1, uniform_spacing(nx), element_tag=tags[0]),
            linemesh.line(c1, c2, uniform_spacing(ny), element_tag=tags[1]),
            linemesh.line(c2, c3, uniform_spacing(nx), element_tag=tags[2]),
            linemesh.line(c3, c0, uniform_spacing(ny), element_tag=tags[3])]


def test_structured_reads_edge_element_tags():
    # each side is named from its own edge's uniform element tag (no edge_tags)
    qm = quadmesh.structured(_tagged_rect_edges(3, 2, ["wall", "outlet", "top", "inlet"]))
    from collections import Counter
    counts = Counter(qm.edge_tags.tags.tolist())
    assert counts == {"wall": 3, "top": 3, "outlet": 2, "inlet": 2}


def test_structured_side_tags_override_edge_tags():
    # a non-empty side_tags entry OVERRIDES that side's edge tag
    qm = quadmesh.structured(
        _tagged_rect_edges(3, 2, ["wall", "wall", "wall", "wall"]),
        side_tags={"bottom": "floor"})
    from collections import Counter
    counts = Counter(qm.edge_tags.tags.tolist())
    assert counts["floor"] == 3            # bottom overridden
    assert counts["wall"] == 3 + 2 + 2     # the other three sides keep their edge tag


def test_structured_side_tags_empty_suppresses_edge_tag():
    # a present-but-empty override (NO_TAG / "") suppresses a tagged edge -- e.g.
    # a shared edge that merge will weld away
    from nekmeshpy import NO_TAG
    qm = quadmesh.structured(
        _tagged_rect_edges(3, 2, ["wall", "wall", "wall", "wall"]),
        side_tags={"left": NO_TAG})
    names = qm.edge_tags.tags.tolist()
    assert "wall" in names
    assert qm.n_edge_tags == 3 + 2 + 3    # left (ny=2 edges) suppressed


def test_annulus_inner_tag_overrides_loop_element_tags():
    # a tagged inner loop names the inner ring at the line level; a non-empty
    # inner_tag OVERRIDES it for the whole inner ring
    inner = linemesh.circle(0.5, 16, element_tags=["body"] * 16)
    outer = linemesh.rectangle(
        12.0, 12.0, 16, side_tags={"bottom": "bottom", "right": "outlet", "top": "top", "left": "inlet"})
    qm = quadmesh.annulus(inner, outer, geometric_spacing(4, 1.12),
                          inner_tag="override")
    from collections import Counter
    counts = Counter(qm.edge_tags.tags.tolist())
    assert counts["override"] == 16        # inner ring overridden
    assert "body" not in counts
    # outer sides still come from the outer loop's element_tags
    assert counts["outlet"] == counts["top"] == counts["inlet"] == counts["bottom"] == 4


# -- input validation --------------------------------------------------------

def test_linemesh_rejects_2d_input():
    # points must be 3-D (N,3); a 2-D (N,2) array is rejected
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        linemesh.loft([(0, 0), (1, 0)])
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        linemesh.loft([(0, 0), (1, 0), (1, 1)], loop=True)


def test_structured_rejects_wrong_edge_count():
    with pytest.raises(ValueError, match="exactly 4 edge"):
        quadmesh.structured(_rect_edges(-1, 1, -1, 1)[:3])


def test_structured_rejects_mismatched_edge_counts():
    # bottom (4 pts) and top (3 pts) disagree on nx -> rejected (no resampling)
    edges = [linemesh.line((-1, -1, 0), (1, -1, 0), uniform_spacing(3)),   # 4 pts
             linemesh.loft([(1, -1, 0), (1, 1, 0)]),
             linemesh.line((1, 1, 0), (-1, 1, 0), uniform_spacing(2)),     # 3 pts
             linemesh.loft([(-1, 1, 0), (-1, -1, 0)])]
    with pytest.raises(ValueError, match="bottom and top .* equal point counts"):
        quadmesh.structured(edges)


def test_structured_rejects_non_linemesh_edge():
    edges = _rect_edges(-1, 1, -1, 1)
    edges[0] = np.array([[-1, -1, 0], [1, -1, 0]])   # bare array, not a LineMesh
    with pytest.raises(TypeError, match="must be a LineMesh"):
        quadmesh.structured(edges)


def test_structured_rejects_open_loop():
    # four edges that do not share corners -> not a closed loop
    edges = [linemesh.loft([(0, 0, 0), (1, 0, 0)]), linemesh.loft([(1, 0, 0), (1, 1, 0)]),
             linemesh.loft([(1, 1, 0), (0, 1, 0)]), linemesh.loft([(0, 1, 0), (0.5, 0.5, 0)])]
    with pytest.raises(ValueError, match="closed loop"):
        quadmesh.structured(edges)


def test_ogrid_rejects_non_loop_boundary():
    # ogrid needs a ring of exactly 4*n_side points; a 3-point chain cannot be one.
    # (Closedness itself is no longer a stored flag -- see
    # test_ogrid_reads_the_loop_wrap_from_connectivity below.)
    with pytest.raises(ValueError, match=r"exactly 4\*n_side"):
        quadmesh.ogrid(linemesh.loft([(0, 0, 0), (1, 0, 0), (1, 1, 0)]), n_side=4,
                       radial=uniform_spacing(3))


def test_ogrid_reads_the_loop_wrap_from_connectivity():
    # the structural fact behind "must be a closed loop": the boundary ring the
    # ogrid fills is the loop's wrapping `lines`, so its perimeter is 4*n_side
    # edges with no degree-1 end anywhere.
    loop = _circle(0.5, 16)
    assert loop.lines[-1].tolist() == [15, 0] and linemesh.boundary_points(loop).size == 0
    qm = quadmesh.ogrid(loop, n_side=4, radial=uniform_spacing(3))
    assert quadmesh.boundary_edges(qm).shape[0] == 16


def test_ogrid_rejects_bad_center_scale():
    with pytest.raises(ValueError, match="center_scale in"):
        quadmesh.ogrid(_circle(0.5, 16), n_side=4, radial=uniform_spacing(3), center_scale=1.5)


def test_ogrid_rejects_bad_radial():
    with pytest.raises(ValueError, match="strictly increasing"):
        quadmesh.ogrid(_circle(0.5, 16), n_side=4, radial=np.array([1.0, 0.5]))


def _semicircle_arc(Nt):
    na = 4 * Nt + 1
    ang = np.linspace(np.pi, 0.0, na)
    return linemesh.loft(np.column_stack([np.cos(ang), np.sin(ang), np.zeros(na)]))


def test_half_ogrid_rejects_bad_arc_count():
    arc = linemesh.loft(np.column_stack([np.linspace(-1, 1, 6), np.zeros(6), np.zeros(6)]))
    spine = linemesh.loft([[-1.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="4.Ntheta"):
        quadmesh.half_ogrid(arc, spine, uniform_spacing(2), center_scale=0.5)


def test_half_ogrid_rejects_non_increasing_radial():
    arc = _semicircle_arc(2)
    spine = linemesh.loft([[-1.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        quadmesh.half_ogrid(arc, spine, np.array([1.0, 0.5]), center_scale=0.5)


def test_half_ogrid_rejects_radial_not_reaching_wall():
    arc = _semicircle_arc(2)
    spine = linemesh.loft([[-1.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="last layer position must be 1.0"):
        quadmesh.half_ogrid(arc, spine, np.array([0.3, 0.6]), center_scale=0.5)


# -- spined_ogrid (closed loop + spine -> two half_ogrids welded) ------------

def _circle_loop(Nt, tag="wall"):
    # closed disc boundary of M = 8*Nt points, index 0 at A1 and index M//2 at A2
    M = 8 * Nt
    th = np.linspace(0.0, 2 * np.pi, M, endpoint=False)
    pts = np.column_stack([np.cos(th), np.sin(th), np.zeros(M)])
    return linemesh.loft(pts, element_tags=[tag] * M, loop=True)


def test_spined_ogrid_valid_and_tagged():
    Nt, Nr = 2, 2
    loop = _circle_loop(Nt)
    qm = quadmesh.spined_ogrid(loop, uniform_spacing(Nr), center_scale=0.5)
    # two half-discs merged (no quads dropped, points welded along the spine)
    assert qm.n_quads == 2 * (2 * Nt * Nt + 4 * Nt * Nr)
    assert set(qm.edge_tags.tags.tolist()) == {"wall"}        # loop tag on the wall
    assert not np.any(np.isnan(np.asarray(qm.points)))


def test_spined_ogrid_default_spine_equals_explicit_chord():
    # omitting spine must use the straight A1..A2 chord (boundary's two split points).
    # The spine is meshed exactly at the points given -- never resampled -- so the
    # "equivalent explicit chord" is that chord sampled at spine_fractions, which is
    # exactly what the factory itself places for spine=None.
    Nt = 2
    loop = _circle_loop(Nt)
    radial = uniform_spacing(2)
    fr = quadmesh.spine_fractions(Nt, radial, 0.5)
    chord = linemesh.line(loop.points[0], loop.points[4 * Nt], fr)
    auto = quadmesh.spined_ogrid(loop, radial, center_scale=0.5)
    explicit = quadmesh.spined_ogrid(loop, radial, spine=chord, center_scale=0.5)
    assert np.array_equal(np.asarray(auto.points), np.asarray(explicit.points))
    assert np.array_equal(np.asarray(auto.quads), np.asarray(explicit.quads))


# -- the spine is meshed exactly, never resampled ----------------------------

def test_spined_ogrid_rejects_wrong_length_spine():
    # a 2-point chord is no longer silently resampled onto the required sampling;
    # the error names the helper that derives it
    Nt = 2
    loop = _circle_loop(Nt)
    chord = linemesh.loft(loop.points[[0, 4 * Nt], :])
    with pytest.raises(ValueError, match="spine_fractions"):
        quadmesh.spined_ogrid(loop, uniform_spacing(2), spine=chord,
                              center_scale=0.5)


def test_spine_fractions_shape_and_ordering():
    Nt, radial, cs = 3, uniform_spacing(4), 0.4
    fr = np.asarray(quadmesh.spine_fractions(Nt, radial, cs))
    Nr = len(radial) - 1
    assert fr.shape == (2 * Nt + 1 + 2 * Nr,)
    assert np.all(np.diff(fr) > 0.0)                  # strictly ascending A1 -> A2
    assert fr[0] == 0.0 and fr[-1] == 1.0             # spans the full chord exactly


def test_spine_fractions_rejects_bad_n_theta():
    with pytest.raises(ValueError, match="n_theta >= 1"):
        quadmesh.spine_fractions(0, uniform_spacing(2), 0.5)


@pytest.mark.parametrize("cs", [0.0, 1.0, -0.2, 1.5])
def test_spine_fractions_rejects_bad_center_scale(cs):
    with pytest.raises(ValueError, match=r"center_scale in \(0, 1\)"):
        quadmesh.spine_fractions(2, uniform_spacing(2), cs)


def test_spined_ogrid_curved_spine_is_meshed_exactly():
    # the "meshed exactly, not resampled" property: a bowed spine sampled at
    # spine_fractions appears verbatim in the merged disc's points, to the bit.
    Nt, radial, cs = 3, uniform_spacing(3), 0.5
    loop = _circle_loop(Nt)
    A1, A2 = loop.points[0], loop.points[4 * Nt]
    fr = np.asarray(quadmesh.spine_fractions(Nt, radial, cs))
    pts = A1 + fr[:, None] * (A2 - A1)
    pts[:, 2] = 0.35 * np.sin(np.pi * fr)             # +z bow, pinned at both ends
    spine = linemesh.loft(pts)
    qm = quadmesh.spined_ogrid(loop, radial, spine=spine, center_scale=cs)

    assert not np.any(np.isnan(np.asarray(qm.points)))
    assert qm.n_quads == 2 * (2 * Nt * Nt + 4 * Nt * radial[1:].size)
    P = np.asarray(qm.points)
    # every spine point is a point of the mesh: the given coordinates are used, not
    # a resampling of them.  Most land bit-for-bit; a few center-fan nodes are
    # rebuilt by the fan blend and land within a couple of ulp -- nowhere near the
    # ~1e-2 a resampling of a bowed spine would cost.
    dev = np.array([np.min(np.linalg.norm(P - p, axis=1)) for p in pts])
    assert dev.max() < 1e-15
    assert dev[0] == 0.0 and dev[-1] == 0.0           # both spine ends exactly
    assert np.count_nonzero(dev == 0.0) > len(dev) // 2
    # and the bow really left the plane
    assert np.max(np.abs(P[:, 2])) == pytest.approx(np.max(np.abs(pts[:, 2])))


def test_spined_ogrid_matches_two_half_ogrids():
    # spined_ogrid must equal the hand-rolled split -> two half_ogrids -> merge
    Nt = 2
    loop = _circle_loop(Nt)
    P = loop.points
    M, nh = 8 * Nt, 4 * Nt
    radial = uniform_spacing(2)
    combined = quadmesh.spined_ogrid(loop, radial, center_scale=0.5)

    arc1 = linemesh.loft(P[0:nh + 1, :], element_tags=["wall"] * nh)
    arc2 = linemesh.loft(np.vstack([P[nh:M, :], P[0:1, :]]), element_tags=["wall"] * nh)
    h1 = quadmesh.half_ogrid(arc1, _diameter_spine(arc1, 0.5, radial), radial,
                             center_scale=0.5, wall_tag="")
    h2 = quadmesh.half_ogrid(arc2, _diameter_spine(arc2, 0.5, radial), radial,
                             center_scale=0.5, wall_tag="")
    manual = quadmesh.merge([h1, h2])

    assert manual.n_quads == combined.n_quads
    assert np.array_equal(np.asarray(manual.quads), np.asarray(combined.quads))
    assert np.allclose(np.asarray(manual.points), np.asarray(combined.points))


def test_spined_ogrid_wall_tag_overrides_loop_tags():
    loop = _circle_loop(2, tag="skin")
    qm = quadmesh.spined_ogrid(loop, uniform_spacing(2), center_scale=0.5,
                               wall_tag="override")
    assert set(qm.edge_tags.tags.tolist()) == {"override"}


def test_spined_ogrid_curved_spine_drives_interior_geometry():
    # a spine that bulges out of the xy-plane is meshed exactly, so the merged disc
    # leaves the plane (the seam and its neighbourhood follow the curve)
    Nt = 2
    loop = _circle_loop(Nt)
    e1, e2 = loop.points[0], loop.points[4 * Nt]
    t = np.linspace(0.0, 1.0, 9)
    curved = e1 + t[:, None] * (e2 - e1)
    curved[:, 2] = 0.3 * np.sin(np.pi * t)                   # +z bulge, 0 at both ends
    qm = quadmesh.spined_ogrid(loop, uniform_spacing(2), spine=linemesh.loft(curved),
                               center_scale=0.5)
    assert np.max(np.abs(np.asarray(qm.points)[:, 2])) > 1e-3


def test_spined_ogrid_rejects_bad_boundary_count():
    # 12 points is not a multiple of 8 (cannot split into two 4*Nt+1 arcs)
    th = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    loop = linemesh.loft(np.column_stack([np.cos(th), np.sin(th), np.zeros(12)]), loop=True)
    with pytest.raises(ValueError, match="8.Ntheta"):
        quadmesh.spined_ogrid(loop, uniform_spacing(2), center_scale=0.5)


def test_spined_ogrid_boundary_wrap_is_structural():
    # spined_ogrid splits its boundary at index 0 / M//2 into two A1->A2 half-arcs
    # and welds them back; the seam ring it produces is closed because the input
    # ring's `lines` wrap -- nothing is read from a stored flag.
    M = 16
    th = np.linspace(0.0, 2 * np.pi, M, endpoint=False)
    loop = linemesh.loft(np.column_stack([np.cos(th), np.sin(th), np.zeros(M)]), loop=True)
    assert loop.lines[-1].tolist() == [M - 1, 0]
    assert linemesh.boundary_points(loop).size == 0
    qm = quadmesh.spined_ogrid(loop, uniform_spacing(2), center_scale=0.5)
    # the filled disc's only free perimeter is the wall ring itself
    assert quadmesh.boundary_edges(qm).shape[0] == M


# -- structured: edges as a keyed Mapping ------------------------------------

def test_structured_accepts_edges_as_a_keyed_mapping():
    # the same four edges named rather than positional -- byte-identical result,
    # so the Sequence form stays the canonical one and this is pure sugar
    seq = _tagged_rect_edges(3, 2, ["wall", "outlet", "top", "inlet"])
    a = quadmesh.structured(seq)
    b = quadmesh.structured(dict(zip(("bottom", "right", "top", "left"), seq)))
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.quads, b.quads)
    assert a.edge_tags.tags.tolist() == b.edge_tags.tags.tolist()


def test_structured_edge_mapping_is_order_insensitive():
    seq = _tagged_rect_edges(3, 2, ["wall", "outlet", "top", "inlet"])
    keys = ("bottom", "right", "top", "left")
    shuffled = {k: e for k, e in sorted(zip(keys, seq), key=lambda kv: kv[0])}
    assert np.array_equal(quadmesh.structured(seq).points,
                          quadmesh.structured(shuffled).points)


def test_structured_edge_mapping_rejects_missing_and_unknown_keys():
    seq = _tagged_rect_edges(3, 2, ["a", "b", "c", "d"])
    keys = ("bottom", "right", "top", "left")
    full = dict(zip(keys, seq))
    with pytest.raises(ValueError):                      # all four are required
        quadmesh.structured({k: full[k] for k in keys[:3]})
    with pytest.raises(ValueError):                      # typo must not pass silently
        quadmesh.structured({**full, "lft": full["left"]})
