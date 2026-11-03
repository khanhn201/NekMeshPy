"""Hex smoother constraints: selections, rings, frozen regions, resolve and the boundary patches."""

import numpy as np
import pytest
from hexblocks import block

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.hexmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import constrain as quad_cn


def _box_classes(m, tol=1e-12):
    """Per vertex, how many box faces it lies on (0 interior, 1 face, 2 edge, 3 corner)."""
    P = m.points
    lo, hi = P.min(axis=0), P.max(axis=0)
    return ((np.abs(P - lo) < tol) | (np.abs(P - hi) < tol)).sum(axis=1)


def test_the_vocabulary_is_the_quad_smoothers_own():
    assert cn.Frozen is quad_cn.Frozen and cn.OnSurface is quad_cn.OnSurface and cn.Resolved is quad_cn.Resolved
    assert cn.snap is quad_cn.snap and cn.constrain_direction is quad_cn.constrain_direction


def test_hex_local_faces_are_the_face_points_the_shrink_relies_on():
    m = block(3, jitter=0.2)
    for f in range(6):
        want = np.sort(m.corners[:, m.FACE_POINTS[f]], axis=1)
        got = np.sort(m.quad_mesh.corners[m.hexes[:, f]], axis=1)
        np.testing.assert_array_equal(got, want)


def test_a_face_tag_selects_its_points_and_rings_grow_by_hexes():
    m = block(4)
    bottom = np.flatnonzero(hexmesh.boundary_face_ids(m) & (m.points[m.quad_mesh.corners][:, :, 2].max(1) < 1e-12))
    m = hexmesh.tag_faces(m, bottom, "floor")
    r = cn.resolve(m, [cn.Frozen("floor")], boundary="free")
    assert set(np.flatnonzero(r.kind == cn.FROZEN)) == set(np.flatnonzero(m.points[:, 2] < 1e-12))
    for k in range(3):
        ring = cn.layer_rings(m, "floor", k)
        np.testing.assert_array_equal(ring, np.flatnonzero(m.points[:, 2] < k / 4 + 1e-9))
    with pytest.raises(ValueError):
        cn.resolve(m, [cn.Frozen("no_such_tag")])


def test_frozen_region_holds_every_vertex_of_the_tagged_hexes():
    tags = ["solid" if i % 4 == 0 else "fluid" for i in range(27)]
    m = block(3, tags=tags)
    fr = cn.frozen_region(m, "solid")
    sel = m.element_tags.to_dense() == "solid"
    np.testing.assert_array_equal(fr.points, np.unique(m.corners[sel]))
    with pytest.raises(ValueError, match="element tag"):
        cn.frozen_region(m, "nope")


def test_resolve_frozen_free_and_two_surfaces_conflict():
    m = block(3, jitter=0.2)
    cls = _box_classes(m)
    r = cn.resolve(m, [])
    np.testing.assert_array_equal(r.kind == cn.FROZEN, cls > 0)
    assert any("boundary='frozen'" in n for n in r.notes)
    assert (cn.resolve(m, [], boundary="free").kind == cn.FREE).all()
    x0 = np.flatnonzero(m.points[:, 0] < 1e-12)
    y0 = np.flatnonzero(m.points[:, 1] < 1e-12)
    r = cn.resolve(m, [cn.OnSurface(x0, cn.plane([0, 0, 0], [1, 0, 0])),
                       cn.OnSurface(y0, cn.plane([0, 0, 0], [0, 1, 0]))])
    both = np.intersect1d(x0, y0)
    assert (r.kind[both] == cn.FROZEN).all() and any("more than one surface" in n for n in r.notes)
    with pytest.raises(ValueError, match="boundary"):
        cn.resolve(m, [], boundary="loose")


def test_slide_on_a_box_faces_slide_edges_follow_lines_corners_frozen():
    m = block(3, jitter=0.25, seed=4)
    cls = _box_classes(m)
    r = cn.resolve(m, [], boundary="slide")
    np.testing.assert_array_equal(r.kind == cn.ON_SURFACE, cls == 1)
    np.testing.assert_array_equal(r.kind == cn.ON_CURVE, cls == 2)
    np.testing.assert_array_equal(r.kind == cn.FROZEN, cls == 3)
    assert {s.name for s in r.surfaces} == {"plane"} and len(r.surfaces) == 6
    assert {c.name for c in r.curves} == {"line"} and len(r.curves) == 12
    bp = cn.boundary_patches(m)
    assert bp.planar.all() and bp.planar.size == 6 and bp.faces.shape == (6 * 9, 4)


def test_boundary_faces_are_wound_outward():
    m = block(2, jitter=0.1)
    F = cn.boundary_faces_outward(m)
    P = m.points
    n = np.cross(P[F[:, 2]] - P[F[:, 0]], P[F[:, 3]] - P[F[:, 1]])
    out = P[F].mean(axis=1) - 0.5
    assert (np.einsum("ij,ij->i", n, out) > 0).all()


def test_slide_on_a_cylinder_one_curved_wall_two_planar_caps_two_closed_rims():
    sec = quadmesh.ogrid(linemesh.circle(1.0, 16), 4, 1)
    cyl = hexmesh.extrude(sec, 1.0, 2)
    r = cn.resolve(cyl, [], boundary="slide")
    bp = cn.boundary_patches(cyl)
    assert bp.planar.size == 3 and int(bp.planar.sum()) == 2
    assert len(r.curves) == 2 and int((r.kind == cn.ON_CURVE).sum()) == 32
    assert int((r.kind == cn.FROZEN).sum()) == 0
    names = sorted(s.name for s in r.surfaces)
    assert names.count("plane") == 2 and len(names) == 3


def test_vertex_length_is_the_shortest_hex_edge():
    m = block(2, size=(2.0, 1.0, 4.0))
    np.testing.assert_allclose(cn.vertex_length(m.points, m.corners), 0.5)
    rep = cn.on_target_report(m.points, m.corners, cn.resolve(m, [], boundary="slide"))
    assert rep.passed and rep.max_distance < 1e-14
