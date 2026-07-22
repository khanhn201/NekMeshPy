"""Tests for the generic HexAlgorithm layer: the transfinite block primitive,
the algorithm registry, and sizing fields."""

import numpy as np
import pytest

from nekmeshpy import (
    ALGORITHMS,
    AxisLinearField,
    ConstantField,
    HexAlgorithm,
    TransfiniteBlock,
    export,
    fields,
    make,
    quality,
)


def _scaled_jac(mesh):
    X, HC, _ = mesh.weld()
    return quality.scaled_jacobian(X, HC)


def test_algorithms_registered():
    assert "bifurcation" in ALGORITHMS
    assert "transfinite_block" in ALGORITHMS


def test_unit_cube_block():
    corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
               [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    mesh = TransfiniteBlock(corners, divisions=(2, 3, 4)).run()
    assert mesh.n_elements == 2 * 3 * 4
    # a perfect axis-aligned grid -> scaled Jacobian 1 everywhere
    sj = _scaled_jac(mesh)
    assert float(np.min(sj)) == pytest.approx(1.0, abs=1e-12)
    # welded node count = (nx+1)(ny+1)(nz+1)
    _, _, nu = mesh.weld()
    assert nu == 3 * 4 * 5


def test_block_satisfies_protocol():
    blk = TransfiniteBlock([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    assert isinstance(blk, HexAlgorithm)


def test_block_boundary_groups():
    mesh = make("transfinite_block",
                corners=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                divisions=(2, 2, 2)).run()
    m = export.to_mesh(mesh)
    # each of the 6 faces of a 2x2x2 block has 4 quads
    for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        assert m.cell_sets[name]["quad"].size == 4


def test_block_grading_changes_spacing():
    corners = [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],
               [0, 0, 1], [2, 0, 1], [2, 1, 1], [0, 1, 1]]
    mesh = TransfiniteBlock(corners, divisions=(4, 1, 1),
                            grading=(2.0, 1.0, 1.0)).run()
    X, _, _ = mesh.weld()
    xs = np.unique(np.round(X[:, 0], 8))
    d = np.diff(xs)
    assert d[-1] > d[0]                 # cells grow along x


def test_geometric_spacing():
    p = fields.geometric_spacing(4, 1.0)
    assert np.allclose(p, [0, 0.25, 0.5, 0.75, 1.0])
    g = fields.geometric_spacing(3, 2.0)
    assert g[0] == 0.0 and g[-1] == pytest.approx(1.0)
    assert np.all(np.diff(np.diff(g)) > 0)   # widths increase


def test_constant_and_linear_fields():
    cf = ConstantField(0.1)
    assert np.allclose(cf(np.zeros((5, 3))), 0.1)
    lf = AxisLinearField(0, 0.0, 0.1, 1.0, 0.5)
    got = lf(np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], float))
    assert got[0] == pytest.approx(0.1)
    assert got[1] == pytest.approx(0.5)
    assert got[2] == pytest.approx(0.3)


def test_size_field_drives_divisions():
    corners = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
               [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    mesh = TransfiniteBlock(corners, size_field=ConstantField(0.25)).run()
    X, _, _ = mesh.weld()
    xs = np.unique(np.round(X[:, 0], 6))
    # ~4 cells of size 0.25 along a unit edge
    assert 4 <= len(xs) <= 6
    assert float(np.min(_scaled_jac(mesh))) == pytest.approx(1.0, abs=1e-9)
