"""Run the transfinite-block example script, plus toolkit sizing-field /
``HexMesh.from_grid`` unit tests (the grading + size-field coverage that used to
live on the block class, now tested against the toolkit directly)."""

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import AxisLinearField, ConstantField, HexMesh, export, fields, quality
from nekmeshpy.model.fields import distribution_from_field


def _scaled_jac(mesh):
    X, HC, _ = mesh.weld()
    return quality.scaled_jacobian(X, HC)


def test_transfinite_block_example(tmp_path):
    mesh = run_example("transfinite_block.py", tmp_path)["mesh"]
    assert mesh.n_hexes == 4 * 4 * 4                  # unit cube, DIVISIONS=(4,4,4)
    assert float(np.min(_scaled_jac(mesh))) == pytest.approx(1.0, abs=1e-12)
    m = export.to_mesh(mesh)
    for name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        assert m.cell_sets[name]["quad"].size == 16  # 4x4 quads per face


def test_from_grid_grading():
    """Graded 1-D spacing builds a graded structured block via from_grid."""
    xs = fields.geometric_spacing(4, 2.0)            # cells grow along x
    P = np.zeros((len(xs), 2, 2, 3))
    for i, x in enumerate(xs):
        for j, y in enumerate((0.0, 1.0)):
            for k, z in enumerate((0.0, 1.0)):
                P[i, j, k] = (x, y, z)
    mesh = HexMesh.from_grid(P)
    xu = np.unique(np.round(mesh.points[:, 0], 8))
    d = np.diff(xu)
    assert d[-1] > d[0]
    assert float(np.min(_scaled_jac(mesh))) == pytest.approx(1.0, abs=1e-12)


def test_geometric_spacing():
    p = fields.geometric_spacing(4, 1.0)
    assert np.allclose(p, [0, 0.25, 0.5, 0.75, 1.0])
    g = fields.geometric_spacing(3, 2.0)
    assert g[0] == 0.0 and g[-1] == pytest.approx(1.0)
    assert np.all(np.diff(np.diff(g)) > 0)           # widths increase


def test_constant_and_linear_fields():
    cf = ConstantField(0.1)
    assert np.allclose(cf(np.zeros((5, 3))), 0.1)
    lf = AxisLinearField(0, 0.0, 0.1, 1.0, 0.5)
    got = lf(np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], float))
    assert got[0] == pytest.approx(0.1)
    assert got[1] == pytest.approx(0.5)
    assert got[2] == pytest.approx(0.3)


def test_distribution_from_field():
    s = distribution_from_field(ConstantField(0.25), np.zeros(3), np.array([1.0, 0, 0]))
    assert s[0] == 0.0 and s[-1] == pytest.approx(1.0)
    assert 4 <= len(s) - 1 <= 6                        # ~4 cells of size 0.25
