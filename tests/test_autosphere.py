"""prenek's Autosphere: quadmesh.quadrant_autosphere / autosphere, hexmesh.octant_autosphere /
autosphere, and the shape model in nekmeshpy.core.autosphere."""

import numpy as np
import pytest

from nekmeshpy import hexmesh, quadmesh
from nekmeshpy.core import autosphere as model

R = 0.5


def _normals(qm):
    P, Q = qm.points, qm.corners
    return np.cross(P[Q[:, 2]] - P[Q[:, 0]], P[Q[:, 3]] - P[Q[:, 1]])


def test_geometric_fractions_is_prenek_get_ratio():
    fr = model.geometric_fractions(6, 0.7)
    d = np.diff(fr)
    assert fr[0] == 0.0 and fr[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(d[1:] / d[:-1], 0.7)
    np.testing.assert_allclose(model.shell_fractions([0.0, 0.25, 1.0]), [0.0, 0.25, 1.0])


def test_quadrant_maps_conform():
    core, sx, sy = model.quadrant_maps(R)
    t = np.linspace(0, 1, 11)
    np.testing.assert_allclose(sx(0 * t, t), core(1 + 0 * t, t), atol=1e-14)        # core x-side
    np.testing.assert_allclose(sy(0 * t, t), core(1 - t, 1 + 0 * t), atol=1e-14)    # core y-side
    np.testing.assert_allclose(sx(t, 1 + 0 * t), sy(t, 0 * t), atol=1e-14)          # 45 deg seam
    for s in (sx, sy):
        np.testing.assert_allclose(np.linalg.norm(s(1 + 0 * t, t), axis=-1), R)


def test_octant_maps_conform():
    core, sz, sx, sy = model.octant_maps(R)
    g = np.linspace(0, 1, 7)
    U, V = np.meshgrid(g, g, indexing="ij")
    one, zero = np.ones_like(U), np.zeros_like(U)
    np.testing.assert_allclose(sz(U, V, zero), core(U, V, one), atol=1e-14)
    np.testing.assert_allclose(sx(U, V, zero), core(one, U, V), atol=1e-14)
    np.testing.assert_allclose(sy(U, V, zero), core(V, one, U), atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(sz(U, V, one), axis=-1), R)
    # shell_z's x = z side meets shell_x's v = 1 side node for node
    np.testing.assert_allclose(sz(one, U, V), sx(U, one, V), atol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_disk(order):
    qm = quadmesh.autosphere(R, 4, 3, ratio=0.7, order=order, wall_tag="wall",
                             normal=(1.0, 0.0, 0.0), center=(2.0, 0.0, 0.0))
    assert qm.n_quads == 4 * (4 * 4 + 2 * 4 * 3)
    assert quadmesh.is_conforming(qm)
    n = _normals(qm)
    assert np.all(n[:, 0] > 0)                      # counter-clockwise about +x, as ogrid
    rows, names = quadmesh.edge_tag_rows(qm)
    assert len(rows) == 8 * 4 and set(names) == {"wall"}
    assert quadmesh.quality_summary(qm).min > 0.3
    if order > 1:
        assert quadmesh.area(qm, high_order=True) == pytest.approx(np.pi * R * R, rel=1e-5)


def test_quadrant_custom_shell_and_start_theta():
    fr = np.array([0.0, 0.5, 0.8, 1.0])
    qm = quadmesh.quadrant_autosphere(R, 3, fr, start_theta=np.pi / 2, order=2)
    assert qm.n_quads == 9 + 2 * 3 * 3
    x = qm.points
    assert np.all(x[:, 0] <= 1e-12) and np.all(x[:, 1] >= -1e-12)   # second quadrant


@pytest.mark.parametrize("order", [1, 2])
def test_ball(order):
    hm = hexmesh.autosphere(R, 3, 2, order=order, wall_tag="wall")
    assert hm.n_hexes == 8 * (27 + 3 * 9 * 2)
    assert hexmesh.boundary_mesh(hm).n_quads == 8 * 3 * 9        # only the sphere is left
    assert hexmesh.quality_summary(hm).min > 0
    wall = hexmesh.boundary_mesh(hm, "wall")
    assert wall.n_quads == 8 * 3 * 9
    if order > 1:
        assert hexmesh.volume(hm, high_order=True) == pytest.approx(4 / 3 * np.pi * R ** 3, rel=1e-4)


def test_octant_tags_and_symmetry_planes():
    hm = hexmesh.octant_autosphere(R, 2, 2, order=2, wall_tag="sphere", element_tags="ball")
    assert hm.face_group_tags == ["sphere"]
    # sphere 3 n_core^2, each symmetry plane n_core^2 + 2 n_core n_shell
    assert hexmesh.boundary_mesh(hm).n_quads == 3 * 4 + 3 * (4 + 2 * 2 * 2)
    assert np.all(hm.points >= -1e-12)
