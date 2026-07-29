"""Tests for the ``blend`` profile-morph classmethod on LineMesh / QuadMesh /
HexMesh: ``(1-t)*a + t*b`` per fraction, carrying ``a``'s connectivity and
boundary tags."""

import numpy as np
import pytest

from nekmeshpy import HexMesh, LineMesh, QuadMesh


def _loop(radius):
    th = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    return np.column_stack([radius * np.cos(th), radius * np.sin(th), np.zeros(8)])


def test_linemesh_blend_endpoints_and_midpoint():
    a = LineMesh.loop(_loop(1.0))
    b = LineMesh.loop(_loop(3.0))
    lo, mid, hi = LineMesh.blend(a, b, [0.0, 0.5, 1.0])
    assert np.allclose(lo.points, a.points)                  # t=0 reproduces a
    assert np.allclose(hi.points, b.points)                  # t=1 reproduces b
    assert np.allclose(mid.points, 0.5 * (a.points + b.points))
    # connectivity + topology carried from a
    assert mid.is_closed and np.array_equal(mid.lines, a.lines)


def test_linemesh_blend_carries_boundary_tags_not_element_tags():
    pts_a = np.column_stack([np.linspace(0, 1, 5), np.zeros(5), np.zeros(5)])
    pts_b = np.column_stack([np.linspace(0, 1, 5), np.ones(5), np.zeros(5)])
    a = LineMesh.open(pts_a, element_tags=["wall"] * 4,
                      boundaries=[[0, 1]], boundary_tags=["inlet"])
    b = LineMesh.open(pts_b)
    mid = LineMesh.blend(a, b, [0.5])[0]
    # positional BC markers follow the morph; per-element region tags do not
    assert mid.boundary_group_tags == ["inlet"]
    assert mid.element_group_tags == []


def test_linemesh_blend_rejects_count_and_topology_mismatch():
    a = LineMesh.loop(_loop(1.0))
    with pytest.raises(ValueError, match="equal point counts"):
        LineMesh.blend(a, LineMesh.loop(_loop(2.0)[:6]), [0.5])
    with pytest.raises(ValueError, match="open/closed"):
        LineMesh.blend(a, LineMesh.open(_loop(2.0)), [0.5])


def _quad_grid(z):
    xs = np.linspace(0, 1, 4)
    ys = np.linspace(0, 1, 3)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return QuadMesh.from_grid(np.stack([X, Y, np.full_like(X, z)], axis=-1))


def test_quadmesh_blend_endpoints_and_connectivity():
    a, b = _quad_grid(0.0), _quad_grid(2.0)
    lo, mid, hi = QuadMesh.blend(a, b, [0.0, 0.5, 1.0])
    assert np.allclose(lo.points, a.points)
    assert np.allclose(hi.points, b.points)
    assert np.allclose(mid.points, 0.5 * (a.points + b.points))
    assert np.array_equal(mid.quads, a.quads)


def test_quadmesh_blend_rejects_connectivity_mismatch():
    a = _quad_grid(0.0)
    xs = np.linspace(0, 1, 5)                                 # different ni -> different quads
    ys = np.linspace(0, 1, 3)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    b = QuadMesh.from_grid(np.stack([X, Y, np.ones_like(X)], axis=-1))
    with pytest.raises(ValueError, match="equal point counts|identical connectivity"):
        QuadMesh.blend(a, b, [0.5])


def _hex_block(z0):
    P = np.zeros((3, 2, 2, 3))
    for i in range(3):
        for j in range(2):
            for k in range(2):
                P[i, j, k] = [i / 2.0, j, z0 + k]
    return HexMesh.from_grid(P)


def test_hexmesh_blend_endpoints_and_connectivity():
    a, b = _hex_block(0.0), _hex_block(5.0)
    lo, mid, hi = HexMesh.blend(a, b, [0.0, 0.5, 1.0])
    assert np.allclose(lo.points, a.points)
    assert np.allclose(hi.points, b.points)
    assert np.allclose(mid.points, 0.5 * (a.points + b.points))
    assert np.array_equal(mid.hexes, a.hexes)
