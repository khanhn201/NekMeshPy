"""The automatic choice of the quadrant T-junction's three shape parameters.

``examples/tjunction_lib.py`` has one construction that has to work across a 10:1 range
of branch-to-main radius, and a single fixed set of ``PHI_W`` / ``CAP_TIP_BIAS`` /
``ORIGIN`` -- what it shipped before -- is only good near the middle of that range: it
leaves the junction **inverted** above ratio 0.8 and fails outright near 1.0.
``auto_params`` picks them from the ratio instead.

The properties pinned here are the ones the tuning rests on:

1. **The three ports are unaffected.**  Whatever these parameters are set to,
   ``disc_minus`` / ``disc_plus`` / ``disc_branch`` come out bit-identical, so
   re-tuning can never disturb a seam something downstream is already bonded to.
   This is what makes the parameters safe to change at all.
2. **The junction is valid across the range**, where the old fixed values were not.
3. **The rule is the stated geometry**: ``PHI_W = 5 * footprint_angle`` makes each side
   quadrant span exactly twice the footprint quadrant.
"""

import os
import sys

import numpy as np
import pytest

from nekmeshpy import hexmesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples"))

from tjunction_lib import (  # noqa: E402
    _PHI_W_CEIL,
    _PHI_W_FLOOR,
    auto_params,
    build_tjunction,
    footprint_angle,
)

CFG = dict(order=2, N_QUAD=2, RADIAL=np.array([0.0, 0.6, 1.0]), CENTER_SCALE=0.5,
           QUADRANT_SCALE=0.4, N_TRANS=3, N_BRANCH=3, Z_NEAR=1.2)
RATIOS = [0.15, 0.30, 0.4167, 0.60, 0.80, 0.95, 0.999]


def _core(ratio, **kw):
    return build_tjunction(1.0, ratio, 3.0, **CFG, **kw).core


# -- the safety property ------------------------------------------------------
@pytest.mark.parametrize("ratio", [0.25, 0.4167, 0.80])
def test_the_three_ports_do_not_depend_on_these_parameters(ratio):
    """The reason re-tuning is safe: only the junction's interior moves."""
    a = build_tjunction(1.0, ratio, 3.0, **CFG,
                        PHI_W=np.deg2rad(100.0), CAP_TIP_BIAS=1 / 3, ORIGIN=(0, 0, 0))
    b = build_tjunction(1.0, ratio, 3.0, **CFG,
                        PHI_W=np.deg2rad(150.0), CAP_TIP_BIAS=0.2, ORIGIN=(0.5, 0, 0))
    for name in ("disc_minus", "disc_plus", "disc_branch"):
        x, y = getattr(a, name), getattr(b, name)
        assert np.array_equal(x.points, y.points), name
        assert np.array_equal(x.corners, y.corners), name
        assert np.array_equal(x.interior, y.interior), name
    assert not np.array_equal(a.core.points, b.core.points)   # the core really moved


# -- the rule -----------------------------------------------------------------
def test_footprint_angle_is_the_documented_closed_form():
    for r in (0.1, 0.5, 1.0):
        assert footprint_angle(r) == pytest.approx(np.arcsin(r / np.sqrt(2.0)))


@pytest.mark.parametrize("ratio", RATIOS)
def test_phi_w_makes_the_side_quadrant_twice_the_footprint(ratio):
    """The rule's geometric reading, wherever it is not clamped."""
    phi_w, _, _ = auto_params(1.0, ratio)
    pf = footprint_angle(ratio)
    if _PHI_W_FLOOR < phi_w < _PHI_W_CEIL:      # the rule, where unclamped
        assert (phi_w - pf) == pytest.approx(2.0 * (2.0 * pf))


@pytest.mark.parametrize("ratio", RATIOS)
def test_phi_w_stays_inside_its_bounds_and_clears_the_footprint(ratio):
    phi_w, bias, origin = auto_params(1.0, ratio)
    assert _PHI_W_FLOOR <= phi_w <= _PHI_W_CEIL
    assert phi_w > footprint_angle(ratio)      # below it the side quadrants invert
    assert 0.0 < bias < 1.0
    assert origin.shape == (3,) and origin[1] == 0.0 and origin[2] == 0.0


def test_the_hub_walks_back_to_the_axis_as_the_branch_grows():
    xs = [auto_params(1.0, r)[2][0] for r in (0.15, 0.35, 0.55, 0.75, 0.95)]
    assert all(a > b for a, b in zip(xs, xs[1:]))


def test_origin_scales_with_the_main_radius():
    """It is a length, so it must follow R_MAIN rather than being absolute."""
    for scale in (0.5, 3.0):
        assert auto_params(scale, 0.4 * scale)[2][0] == pytest.approx(
            scale * auto_params(1.0, 0.4)[2][0])


# -- what it buys -------------------------------------------------------------
@pytest.mark.parametrize("ratio", RATIOS)
def test_the_junction_is_valid_across_the_whole_range(ratio):
    core = _core(ratio)
    assert hexmesh.is_conforming(core)
    assert hexmesh.is_watertight(core)
    # 0.10 is the measured floor of the rule over ratios 0.10 to 0.99; it is a
    # guarantee about the *worst* ratio, not a typical value (most are above 0.25)
    assert hexmesh.scaled_jacobian(core, high_order=True).min() > 0.10


@pytest.mark.parametrize("ratio", [0.90, 0.95, 0.999])
def test_it_fixes_ratios_the_old_fixed_values_could_not_mesh(ratio):
    """Above ~0.8 the shipped constants inverted the crotch caps outright, and near
    1.0 the build failed altogether."""
    try:
        old = hexmesh.scaled_jacobian(
            _core(ratio, PHI_W=np.deg2rad(100.0), CAP_TIP_BIAS=1 / 3,
                  ORIGIN=(0, 0, 0)), high_order=True).min()
    except ValueError:
        old = None                      # would not even build
    assert old is None or old < 0.0
    assert hexmesh.scaled_jacobian(_core(ratio), high_order=True).min() > 0.10


@pytest.mark.parametrize("ratio", RATIOS)
def test_it_never_does_worse_than_the_old_fixed_values(ratio):
    try:
        old = hexmesh.scaled_jacobian(
            _core(ratio, PHI_W=np.deg2rad(100.0), CAP_TIP_BIAS=1 / 3,
                  ORIGIN=(0, 0, 0)), high_order=True).min()
    except ValueError:
        return                          # the old values could not build this at all
    assert hexmesh.scaled_jacobian(_core(ratio), high_order=True).min() > old


def test_an_explicit_value_still_overrides_just_that_one():
    phi = np.deg2rad(123.0)
    a = build_tjunction(1.0, 0.5, 3.0, **CFG, PHI_W=phi)
    b = build_tjunction(1.0, 0.5, 3.0, **CFG, PHI_W=phi,
                        CAP_TIP_BIAS=auto_params(1.0, 0.5)[1],
                        ORIGIN=auto_params(1.0, 0.5)[2])
    assert np.array_equal(a.core.points, b.core.points)
