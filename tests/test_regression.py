"""Golden regression tests: the default surface pipeline must keep reproducing
the validated MATLAB/Octave reference to machine precision, and its ASCII
exports must stay byte-identical.

These pin the numerics so every generalization refactor is safe.
"""

import os

import numpy as np
import pytest
from conftest import GOLDEN, read_re2_coords

from nekmeshpy import quality

# Residual is scipy.spsolve vs MATLAB backslash; ~2.7e-13 observed.
RE2_TOL = 1e-12


def test_element_and_boundary_counts(built_mesh):
    mesh = built_mesh["mesh"]
    assert mesh.hexes.shape == (3840, 8)         # (N,8) shared-point connectivity
    assert mesh.points.shape == (4484, 3)
    assert mesh.boundaries.shape[0] == 1280


def test_tag_face_counts(built_mesh):
    mesh = built_mesh["mesh"]
    names = mesh.boundary_names
    counts = {n: int(np.sum(names == n)) for n in
              ("wall", "trunk_outlet", "top_outlet_1", "top_outlet_2")}
    assert counts["wall"] == 960
    assert counts["trunk_outlet"] == 64
    assert counts["top_outlet_1"] == 64
    assert counts["top_outlet_2"] == 64


def test_scaled_jacobian_quality(built_mesh):
    X, HC, _ = built_mesh["mesh"].weld()
    sj = quality.scaled_jacobian(X, HC)
    assert float(np.min(sj)) == pytest.approx(0.2267, abs=1e-3)
    assert float(np.mean(sj)) == pytest.approx(0.8177, abs=1e-3)
    assert float(np.min(sj)) > 0.0   # no inverted elements


def test_re2_coords_match_golden(built_mesh):
    ne, coords, _ = read_re2_coords(built_mesh["re2"])
    gne, gcoords, _ = read_re2_coords(os.path.join(GOLDEN, "bifurcation.re2"))
    assert ne == gne == 3840
    assert coords.shape == gcoords.shape
    assert np.max(np.abs(coords - gcoords)) < RE2_TOL


def test_re2_boundary_block_identical(built_mesh):
    _, _, bnd = read_re2_coords(built_mesh["re2"])
    _, _, gbnd = read_re2_coords(os.path.join(GOLDEN, "bifurcation.re2"))
    assert bnd == gbnd           # BC block is exact integers/codes


def test_rea_byte_identical(built_mesh):
    with open(built_mesh["rea"], "rb") as f:
        got = f.read()
    with open(os.path.join(GOLDEN, "bifurcation.rea"), "rb") as f:
        ref = f.read()
    assert got == ref


def test_vtk_byte_identical(built_mesh):
    with open(built_mesh["vtk"], "rb") as f:
        got = f.read()
    with open(os.path.join(GOLDEN, "bifurcation.vtk"), "rb") as f:
        ref = f.read()
    assert got == ref
