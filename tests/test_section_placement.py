"""Unit tests for the two section-level queries a builder needs before continuing from
a piece it has already placed: ``quadmesh.plane_normal`` ("which way does this disc
face?") and ``quadmesh.place_on_path`` ("where would a sweep put it?").

The contract, in order of how badly getting it wrong would hurt:

1. **``place_on_path`` goes through the sweep's own placement machinery**, so a piece
   built to continue from a swept tube's terminal section lands on it *exactly*.
   Re-deriving the placement lands close, and at ``order > 1``
   ``HexMesh.merge`` verifies shared high-order edge nodes against
   ``conform.entity_tol`` (~1e-9 of the model extent), which close does not meet.
2. **It takes the whole fraction array, not one station.**  Under
   ``orientation="transport"`` a frame is a sequential integration along everything
   before it, so a placement computed from a different sampling is a different
   placement -- silently, and only on curved paths.
3. **``plane_normal`` reads the section's own fitted plane**, not the direction between
   two centroids: a disc not perfectly centred on its nominal position tilts the latter
   by a small angle, and ``"fixed"`` orientation turns that tilt into a first station
   that misses the very disc it was meant to reproduce.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.model import paths
from nekmeshpy.model.fields import uniform_spacing

CENTER = (0.4, -1.0, 2.5)


def _section(order=2, normal=(0.0, 0.0, 1.0)):
    ring = linemesh.circle(0.5, 8, center=CENTER, normal=normal,
                           element_tag="wall", order=order)
    return quadmesh.ogrid(ring, 2, uniform_spacing(2), wall_tag="wall")


# -- plane_normal -------------------------------------------------------------
@pytest.mark.parametrize("normal", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0),
                                    (1.0, -1.0, 1.0)])
def test_plane_normal_recovers_the_plane_the_section_was_authored_in(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    got = quadmesh.plane_normal(_section(normal=normal))
    assert np.allclose(np.abs(got @ n), 1.0)


def test_plane_normal_is_a_unit_vector():
    """Not merely near-unit: it scales a stub's length at every call site."""
    assert np.linalg.norm(quadmesh.plane_normal(_section())) == pytest.approx(1.0,
                                                                              abs=0.0)


def test_hint_picks_the_sign():
    sec = _section()
    assert quadmesh.plane_normal(sec, hint=(0.0, 0.0, 1.0))[2] > 0.0
    assert quadmesh.plane_normal(sec, hint=(0.0, 0.0, -1.0))[2] < 0.0


def test_plane_normal_rides_a_rotation():
    """The normal of a rotated section is its own normal rotated -- so a caller can
    place a disc and then ask which way it faces, rather than tracking the rotation."""
    angle, axis = np.pi / 3.0, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    sec = _section()
    spun = quadmesh.rotate(sec, angle, axis=axis)
    n0 = quadmesh.plane_normal(sec, hint=(0.0, 0.0, 1.0))
    n1 = quadmesh.plane_normal(spun, hint=None)
    # Rodrigues, applied to n0
    expect = (n0 * np.cos(angle) + np.cross(axis, n0) * np.sin(angle)
              + axis * (axis @ n0) * (1.0 - np.cos(angle)))
    assert np.allclose(np.abs(n1 @ expect), 1.0)
    assert not np.allclose(n0, n1)          # it really moved


def test_a_non_planar_section_is_refused_unless_check_is_off():
    sec = _section()
    pts = np.array(sec.points, dtype=float)
    pts[0, 2] += 0.4                      # lift one node well off the plane
    sec.points[:] = pts
    with pytest.raises(ValueError, match="not planar"):
        quadmesh.plane_normal(sec)
    assert np.isfinite(quadmesh.plane_normal(sec, check=False)).all()


# -- place_on_path ------------------------------------------------------------
@pytest.fixture
def path():
    walk = paths.turtle_path([("line", 2.0, 0.0), ("arc", 1.2, 90.0),
                              ("line", 1.5, 0.0)], start=(CENTER[0], CENTER[2]))
    return paths.embed(walk, u=(1.0, 0.0, 0.0), v=(0.0, 0.0, 1.0),
                       origin=(0.0, CENTER[1], 0.0))


def test_placement_at_station_zero_reproduces_the_section(path):
    """``"fixed"`` makes the section perpendicular to the tangent it is handed, and
    the path starts at the section's own centre heading along its own plane."""
    sec = _section(normal=(1.0, 0.0, 0.0))
    first = quadmesh.place_on_path(sec, path, [0.0, 1.0], orientation="fixed",
                                   up=(0.0, 1.0, 0.0), origin=np.asarray(CENTER))[0]
    assert np.allclose(first.points, sec.points, atol=1e-12)


def test_placement_matches_what_the_sweep_actually_builds(path):
    """The property the whole helper exists for: the terminal section of a sweep,
    without building the sweep."""
    sec = _section(normal=(1.0, 0.0, 0.0))
    fr = linemesh.path_fractions(path, layers=6)
    block = hexmesh.sweep_path(sec, path, fractions=fr, orientation="fixed",
                               up=(0.0, 1.0, 0.0), origin=np.asarray(CENTER))
    placed = quadmesh.place_on_path(sec, path, fr, orientation="fixed",
                                    up=(0.0, 1.0, 0.0), origin=np.asarray(CENTER))
    assert len(placed) == len(fr)
    # every node of the terminal placement is a node the block actually has, exactly
    for station, section in ((0, placed[0]), (-1, placed[-1])):
        d = np.linalg.norm(block.points[:, None, :] - section.points[None, :, :],
                           axis=2).min(axis=0)
        assert d.max() == 0.0, "station %d is not bit-exact on the swept block" % station


def test_one_placement_per_fraction(path):
    sec = _section(normal=(1.0, 0.0, 0.0))
    fr = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = quadmesh.place_on_path(sec, path, fr, orientation="fixed",
                                 up=(0.0, 1.0, 0.0), origin=np.asarray(CENTER))
    assert len(out) == len(fr)
    assert all(np.array_equal(q.quads, sec.quads) for q in out)


def test_placements_are_rigid(path):
    """A placement moves the section; it must not resize or shear it."""
    sec = _section(normal=(1.0, 0.0, 0.0))
    out = quadmesh.place_on_path(sec, path, [0.0, 0.5, 1.0], orientation="fixed",
                                 up=(0.0, 1.0, 0.0), origin=np.asarray(CENTER))

    def gram(m):
        d = m.points - m.points.mean(axis=0)
        return d @ d.T

    for q in out:
        assert np.allclose(gram(q), gram(sec), atol=1e-10)


def test_high_order_interiors_are_carried(path):
    sec = _section(order=3, normal=(1.0, 0.0, 0.0))
    out = quadmesh.place_on_path(sec, path, [0.0, 1.0], orientation="fixed",
                                 up=(0.0, 1.0, 0.0), origin=np.asarray(CENTER))
    assert out[-1].interior.shape == sec.interior.shape
    assert not np.allclose(out[-1].interior, sec.interior)   # they really moved
    assert out[-1].order == sec.order
