"""Tests for the straight-pipe algorithms (CircularPipe / RectangularPipe)."""

import numpy as np
import pytest

from nekmeshpy import (
    CircularPipe,
    HexAlgorithm,
    RectangularPipe,
    available,
    make,
    quality,
)


def _tag_count(mesh, tag):
    return int(np.sum(mesh.boundaries[:, 2] == tag))


def _scaled_jac(mesh):
    X, HC, _ = mesh.weld()
    return quality.scaled_jacobian(X, HC)


# -- circular -----------------------------------------------------------
def test_circular_counts_and_quality():
    n_side, n_radial, n_axial = 4, 3, 8
    mesh = CircularPipe(radius=0.5, length=2.0, n_axial=n_axial,
                        n_side=n_side, n_radial=n_radial).run()
    quads_per_section = n_side * n_side + n_radial * 4 * n_side
    assert mesh.n_elements == quads_per_section * n_axial

    sj = _scaled_jac(mesh)
    assert sj.min() > 0.0                      # no inverted/degenerate cells
    assert sj.min() > 0.5                       # a well-formed O-grid


def test_circular_boundary_tags():
    n_side, n_radial, n_axial = 4, 3, 8
    mesh = CircularPipe(radius=0.5, length=2.0, n_axial=n_axial,
                        n_side=n_side, n_radial=n_radial).run()
    quads_per_section = n_side * n_side + n_radial * 4 * n_side
    assert _tag_count(mesh, 2) == quads_per_section       # inlet cap
    assert _tag_count(mesh, 3) == quads_per_section       # outlet cap
    assert _tag_count(mesh, 1) == 4 * n_side * n_axial    # wall ring x layers


def test_circular_radius_and_center():
    R = 0.75
    mesh = CircularPipe(radius=R, length=1.0, n_axial=3,
                        center=(2.0, -1.0, 0.5)).run()
    X, _, _ = mesh.weld()
    # radius measured in the cross-section plane about the (shifted) axis
    rel = X[:, 0:2] - np.array([2.0, -1.0])
    r = np.sqrt((rel ** 2).sum(axis=1))
    assert r.max() == pytest.approx(R, abs=1e-9)


def test_arbitrary_axis_and_length():
    R, L = 0.5, 3.0
    mesh = CircularPipe(radius=R, length=L, n_axial=5, axis=(1.0, 1.0, 0.0)).run()
    assert _scaled_jac(mesh).min() > 0.0
    X, _, _ = mesh.weld()
    # extent along the (normalized) axis equals the length
    ez = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    proj = X @ ez
    assert (proj.max() - proj.min()) == pytest.approx(L, abs=1e-9)


def test_axial_grading_changes_spacing():
    uniform = CircularPipe(length=1.0, n_axial=10, axial_grading=1.0).run()
    graded = CircularPipe(length=1.0, n_axial=10, axial_grading=1.3).run()
    # z-coordinates present differ once graded
    zu = np.unique(np.round(uniform.weld()[0][:, 2], 9))
    zg = np.unique(np.round(graded.weld()[0][:, 2], 9))
    assert not np.array_equal(zu, zg)


# -- rectangular --------------------------------------------------------
def test_rectangular_counts_and_perfect_quality():
    nx, ny, nz = 6, 4, 8
    mesh = RectangularPipe(width=1.0, height=0.5, length=2.0,
                           nx=nx, ny=ny, n_axial=nz).run()
    assert mesh.n_elements == nx * ny * nz
    sj = _scaled_jac(mesh)
    assert sj.min() == pytest.approx(1.0, abs=1e-9)      # axis-aligned boxes
    assert _tag_count(mesh, 2) == nx * ny                # inlet
    assert _tag_count(mesh, 3) == nx * ny                # outlet
    assert _tag_count(mesh, 1) == 2 * (nx + ny) * nz     # 4 walls x layers


# -- registry integration ----------------------------------------------
def test_registered_as_algorithms():
    assert "circular_pipe" in available()
    assert "rectangular_pipe" in available()
    mesh = make("circular_pipe", radius=0.5, length=1.0, n_axial=4).run()
    assert isinstance(make("rectangular_pipe"), HexAlgorithm)
    assert mesh.n_elements > 0
