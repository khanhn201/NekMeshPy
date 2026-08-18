"""Unit tests for ``paths.walk`` -- the declarative 3-D turtle -- and for the
``sweep_path`` that consumes its result at the quad and hex rungs.

What these pin down, in order of how badly getting it wrong would hurt:

1. **The frame the walk carries is the frame the sweep uses.**  A path that bends out
   of plane, or twists as it runs, has no other way to say so: with no ``orientation``
   asked for, ``sweep_path`` holds ``path.up`` per station.  Getting this wrong is a
   silently *rotated* mesh, not an exception.
2. **The samplers are exact, not merely close.**  Every move is a closed-form screw in
   arc length, so ``total_length`` is the true arc length, ``tangent`` is the analytic
   derivative of ``centerline``, and a straight/arc junction lands on an exact ``s``.
   Differenced tangents are only O(h**2) and worst exactly at a junction, which tilts
   the sections there.
3. **The turn conventions.**  A positive angle turns toward the walk's left, ``tilt``
   rolls the bend plane about the heading (90 pitches up), and a positive ``roll`` spins
   the section right-handed about the heading.  These are the whole vocabulary; a sign
   error in any of them mirrors a part.
4. **``sweep_path`` is exactly ``sweep``** with its stations resolved from the path,
   and resolves them so that every junction still carries one.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core import paths
from nekmeshpy.core.fields import uniform_spacing

MOVES = [paths.line(2.0), paths.arc(1.5, 90.0), paths.line(3.0),
         paths.arc(1.5, -90.0), paths.line(1.0)]
ORIGIN = (0.5, -2.0, 7.0)
HEADING = (0.0, 0.0, 1.0)
UP = (0.0, 1.0, 0.0)


@pytest.fixture
def walk():
    """The reference walk: planar, five moves, both turn signs."""
    return paths.walk(MOVES, start=ORIGIN, heading=HEADING, up=UP)


def _dense(n=4001):
    return np.linspace(0.0, 1.0, n)


# -- the samplers -------------------------------------------------------------
def test_the_walk_starts_where_it_was_told_to(walk):
    assert np.allclose(walk.centerline(np.array([0.0]))[0], ORIGIN, atol=1e-15)
    assert np.allclose(walk.tangent(np.array([0.0]))[0], HEADING, atol=1e-15)
    assert np.allclose(walk.up(np.array([0.0]))[0], UP, atol=1e-15)


def test_total_length_is_the_curve_s_own_arc_length(walk):
    """Exact, not quadratured: a straight contributes its length and an arc
    ``radius * angle``, so the polyline of a dense sampling converges up onto it."""
    P = walk.centerline(_dense(200001))
    assert np.linalg.norm(np.diff(P, axis=0), axis=1).sum() == pytest.approx(
        walk.total_length, abs=1e-8)


def test_the_walk_is_parametrized_by_arc_length(walk):
    """Equal steps in ``s`` are equal steps along the curve -- what lets
    ``path_fractions`` turn an element length into stations."""
    P = walk.centerline(np.linspace(0.0, 1.0, 20001))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    assert seg.max() / seg.min() == pytest.approx(1.0, abs=1e-6)


def test_the_tangent_is_unit_and_is_the_centerline_derivative(walk):
    s = _dense()
    T = walk.tangent(s)
    assert np.allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-14)
    h = 1e-6
    fd = (walk.centerline(s[1:-1] + h) - walk.centerline(s[1:-1] - h)) / (2.0 * h)
    fd = fd / np.linalg.norm(fd, axis=1)[:, None]
    assert np.abs(fd - T[1:-1]).max() < 1e-8


def test_the_up_is_a_unit_cross_section_axis_everywhere(walk):
    s = _dense()
    U, T = walk.up(s), walk.tangent(s)
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-14)
    assert np.abs(np.einsum("kj,kj->k", U, T)).max() < 1e-14


def test_break_fractions_are_the_junctions_and_nothing_else(walk):
    """One per interior junction, at the exact cumulative arc length -- this is what
    lets a sweep land a station where the curvature jumps."""
    lengths = np.array([2.0, 1.5 * np.pi / 2.0, 3.0, 1.5 * np.pi / 2.0, 1.0])
    want = np.cumsum(lengths)[:-1] / lengths.sum()
    assert np.allclose(walk.break_fractions, want, atol=1e-15)


def test_a_planar_walk_stays_in_its_plane_with_a_constant_up(walk):
    s = _dense()
    P = walk.centerline(s) - np.asarray(ORIGIN, dtype=float)
    assert np.abs(P @ np.asarray(UP, dtype=float)).max() < 1e-13
    assert np.abs(walk.up(s) - np.asarray(UP, dtype=float)).max() < 1e-14


# -- the turn conventions -----------------------------------------------------
ONE = np.array([1.0])


@pytest.mark.parametrize("move, end, heading", [
    (paths.arc(1.0, 90.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),           # left
    (paths.arc(1.0, -90.0), (1.0, -1.0, 0.0), (0.0, -1.0, 0.0)),        # right
    (paths.arc(1.0, 90.0, tilt=90.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),   # up
    (paths.arc(1.0, 90.0, tilt=270.0), (1.0, 0.0, -1.0), (0.0, 0.0, -1.0)),  # down
])
def test_a_positive_turn_goes_left_and_tilt_rolls_the_bend_plane(move, end, heading):
    """``tilt`` is what makes the turtle three-dimensional: the same arc, rolled about
    the heading, reaches every direction out of the starting frame."""
    p = paths.walk([move], start=(0.0, 0.0, 0.0), heading=(1.0, 0.0, 0.0),
                   up=(0.0, 0.0, 1.0))
    assert np.allclose(p.centerline(ONE)[0], end, atol=1e-14)
    assert np.allclose(p.tangent(ONE)[0], heading, atol=1e-14)


def test_a_roll_spins_the_section_right_handed_over_its_own_move():
    """The move's *own* length carries the roll, so the frame is continuous -- a
    zero-length spin is rejected rather than jumping the frame at one station."""
    p = paths.walk([paths.line(4.0, roll=90.0)], heading=(1.0, 0.0, 0.0),
                   up=(0.0, 0.0, 1.0))
    assert np.allclose(p.up(ONE)[0], (0.0, -1.0, 0.0), atol=1e-14)
    half = p.up(np.array([0.5]))[0]
    assert np.allclose(half, (0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4)), atol=1e-14)


def test_a_roll_moves_the_frame_and_not_its_own_move():
    """Within the move it is written on, a roll is pure frame -- which is what makes it
    safe to hang one on a run whose shape is already settled."""
    s = _dense()
    a = paths.walk([paths.arc(1.5, 60.0, tilt=35.0)])
    b = paths.walk([paths.arc(1.5, 60.0, tilt=35.0, roll=-25.0)])
    assert np.abs(a.centerline(s) - b.centerline(s)).max() < 1e-14
    assert np.abs(a.tangent(s) - b.tangent(s)).max() < 1e-14
    turn = np.rad2deg(np.arccos(float(a.up(ONE)[0] @ b.up(ONE)[0])))
    assert turn == pytest.approx(25.0, abs=1e-10)


def test_a_roll_re_aims_the_bend_that_follows_it():
    """And *between* moves it is a steering verb: rolling a quarter turn before a
    left bend is the same part as pitching up instead -- the second way the turtle
    leaves its plane."""
    rolled = paths.walk([paths.line(1.0, roll=90.0), paths.arc(2.0, 45.0)],
                        heading=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    tilted = paths.walk([paths.line(1.0), paths.arc(2.0, 45.0, tilt=90.0)],
                        heading=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    assert np.allclose(rolled.centerline(ONE)[0], tilted.centerline(ONE)[0], atol=1e-14)
    assert np.allclose(rolled.tangent(ONE)[0], tilted.tangent(ONE)[0], atol=1e-14)


def test_a_bend_carries_the_frame_round_with_it():
    """With no roll the frame is transported by the bend itself, so a full circle in
    the frame's own bend plane brings the section back exactly as it set out."""
    p = paths.walk([paths.arc(2.0, 360.0)], heading=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    assert np.allclose(p.centerline(ONE)[0], (0.0, 0.0, 0.0), atol=1e-14)
    assert np.allclose(p.up(ONE)[0], (0.0, 0.0, 1.0), atol=1e-14)


# -- the helix ----------------------------------------------------------------
def test_a_helix_is_exactly_a_screw_about_its_own_axis():
    """The one shape a planar turtle cannot reach at all: constant curvature *and*
    constant torsion, in closed form rather than sampled."""
    p = paths.walk([paths.helix(1.0, 720.0, rise=2.0)], heading=(1.0, 0.0, 0.0),
                   up=(0.0, 0.0, 1.0))
    s = _dense()
    P = p.centerline(s)
    axis_pt = np.array([0.0, 1.0, 0.0])          # a left turn: centre one radius to +y
    radial = P - axis_pt - np.outer(P[:, 2], (0.0, 0.0, 1.0))
    assert np.allclose(np.linalg.norm(radial, axis=1), 1.0, atol=1e-13)
    assert p.centerline(ONE)[0][2] == pytest.approx(4.0, abs=1e-13)    # 2 turns * rise
    assert p.total_length == pytest.approx(
        np.deg2rad(720.0) * np.hypot(1.0, 2.0 / (2.0 * np.pi)), abs=1e-13)


def test_a_helix_climbs_along_the_axis_it_turns_about_either_way_round():
    """``rise`` names a climb along the axis ``tilt`` picked, so reversing the turn
    reverses the winding without flipping the pipe end for end."""
    up = paths.walk([paths.helix(1.0, 360.0, rise=2.0)])
    down = paths.walk([paths.helix(1.0, -360.0, rise=2.0)])
    assert up.centerline(ONE)[0][2] == pytest.approx(2.0, abs=1e-13)
    assert down.centerline(ONE)[0][2] == pytest.approx(2.0, abs=1e-13)


def test_a_zero_rise_helix_is_an_arc():
    s = _dense()
    a = paths.walk([paths.helix(1.7, 130.0, tilt=40.0)])
    b = paths.walk([paths.arc(1.7, 130.0, tilt=40.0)])
    assert np.abs(a.centerline(s) - b.centerline(s)).max() < 1e-14
    assert np.abs(a.up(s) - b.up(s)).max() < 1e-14


# -- what the walk refuses ----------------------------------------------------
def test_rejects_an_unknown_move():
    with pytest.raises(ValueError, match="must be a paths.line"):
        paths.walk([("line", 2.0, 0.0)])


def test_rejects_a_move_that_goes_nowhere():
    with pytest.raises(ValueError, match="nothing to spin over"):
        paths.walk([paths.line(2.0), paths.line(0.0, roll=90.0)])


def test_rejects_a_zero_angle_bend():
    with pytest.raises(ValueError, match="not positive"):
        paths.walk([paths.arc(2.0, 0.0)])


def test_rejects_a_non_positive_radius():
    with pytest.raises(ValueError, match="needs a positive"):
        paths.walk([paths.arc(-2.0, 90.0)])


def test_rejects_an_up_along_the_heading():
    with pytest.raises(ValueError, match="parallel to heading"):
        paths.walk([paths.line(1.0)], heading=(0.0, 0.0, 1.0), up=(0.0, 0.0, -1.0))


def test_rejects_an_empty_walk():
    with pytest.raises(ValueError, match="at least one move"):
        paths.walk([])


# -- path_fractions -----------------------------------------------------------
def test_path_fractions_puts_a_station_on_every_junction(walk):
    fr = linemesh.path_fractions(walk, target_length=0.4)
    assert np.all(np.diff(fr) > 0.0)
    assert np.isin(walk.break_fractions, fr).all()


def test_layers_is_the_average_element_length(walk):
    fr = linemesh.path_fractions(walk, layers=12)
    assert fr.shape[0] >= 13          # junctions can only add stations
    assert fr[0] == 0.0 and fr[-1] == 1.0


def test_fractions_are_handed_through_verbatim(walk):
    given = np.array([0.0, 0.3, 0.55, 1.0])
    assert np.array_equal(linemesh.path_fractions(walk, fractions=given), given)


@pytest.mark.parametrize("kwargs", [{}, {"layers": 4, "target_length": 0.5}])
def test_path_fractions_demands_exactly_one_spacing_argument(walk, kwargs):
    with pytest.raises(ValueError, match="exactly one of"):
        linemesh.path_fractions(walk, **kwargs)


def test_path_fractions_rejects_a_zero_layer_count(walk):
    with pytest.raises(ValueError, match="layers must be"):
        linemesh.path_fractions(walk, layers=0)


# -- sweep_path, at both rungs ------------------------------------------------
def _profile(order):
    return linemesh.circle(0.3, 8, center=ORIGIN, normal=HEADING,
                           element_tag="wall", order=order)


def _section(order):
    return quadmesh.ogrid(_profile(order), 2, uniform_spacing(2), wall_tag="wall")


@pytest.mark.parametrize("order", [1, 2])
def test_hex_sweep_path_reproduces_sweep(walk, order):
    fr = linemesh.path_fractions(walk, target_length=0.5)
    direct = hexmesh.sweep(_section(order), walk.centerline, fr, tangent=walk.tangent,
                           orientation="fixed", up=walk.up, origin=ORIGIN,
                           first_tag="inlet", last_tag="outlet")
    viapath = hexmesh.sweep_path(_section(order), walk, target_length=0.5,
                                 origin=ORIGIN, first_tag="inlet", last_tag="outlet")
    assert np.array_equal(viapath.points, direct.points)
    assert np.array_equal(viapath.corners, direct.corners)
    assert np.array_equal(viapath.interior, direct.interior)


@pytest.mark.parametrize("order", [1, 2])
def test_quad_sweep_path_reproduces_sweep(walk, order):
    fr = linemesh.path_fractions(walk, layers=9)
    direct = quadmesh.sweep(_profile(order), walk.centerline, fr, tangent=walk.tangent,
                            orientation="fixed", up=walk.up, origin=ORIGIN)
    viapath = quadmesh.sweep_path(_profile(order), walk, layers=9, origin=ORIGIN)
    assert np.array_equal(viapath.points, direct.points)
    assert np.array_equal(viapath.corners, direct.corners)


def test_swept_block_is_valid_and_not_inverted(walk):
    """A planar walk with ``up`` normal to its plane is the exact, zero-twist case."""
    mesh = hexmesh.sweep_path(_section(2), walk, target_length=0.4, origin=ORIGIN,
                              first_tag="inlet", last_tag="outlet")
    assert hexmesh.is_watertight(mesh)
    assert hexmesh.is_conforming(mesh)
    assert hexmesh.scaled_jacobian(mesh).min() > 0.0
    assert set(mesh.face_group_tags) == {"wall", "inlet", "outlet"}


# -- the path's own frame reaches the sweep -----------------------------------
#: A roll a square section is *not* invariant under -- a quarter turn maps a square
#: onto itself, so it would read as no roll at all.
ROLL = 45.0


def _twisted():
    """A straight run with a square section, so the section's own roll is visible in
    the far cap's corners."""
    path = paths.walk([paths.line(4.0, roll=ROLL)], heading=(0.0, 0.0, 1.0),
                      up=(1.0, 0.0, 0.0))
    section = quadmesh.ogrid(
        linemesh.circle(0.5, 8, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)),
        2, uniform_spacing(2))
    return path, hexmesh.sweep_path(section, path, layers=6,
                                    origin=(0.0, 0.0, 0.0)), section


def _far_layer(block, section):
    """The block's last sweep level, index-paired with ``section`` -- a loft numbers its
    points level-major, so the final level is the section's own table shifted up."""
    n = section.points.shape[0]
    assert block.points.shape[0] % n == 0
    return block.points[-n:]


def test_the_paths_own_roll_arrives_at_the_swept_block():
    """The property the ``up`` field exists for: no ``twist=``, no ``orientation=`` --
    the roll is a property of the path and the sweep honours it."""
    _, block, section = _twisted()
    near = section.points
    c, s = np.cos(np.deg2rad(ROLL)), np.sin(np.deg2rad(ROLL))
    turned = np.stack([c * near[:, 0] - s * near[:, 1],
                       s * near[:, 0] + c * near[:, 1],
                       np.full(near.shape[0], 4.0)], axis=1)
    assert np.abs(_far_layer(block, section) - turned).max() < 1e-12
    assert hexmesh.scaled_jacobian(block).min() > 0.0


def test_a_named_orientation_overrides_the_frame_the_path_carries():
    """An explicit generator is a request for *that* frame -- here ``"transport"``,
    which is rotation-minimizing and so carries no roll at all along a straight."""
    path, _, section = _twisted()
    straight = hexmesh.sweep_path(section, path, layers=6, origin=(0.0, 0.0, 0.0),
                                  orientation="transport")
    flat = section.points + np.array([0.0, 0.0, 4.0])
    assert np.abs(_far_layer(straight, section) - flat).max() < 1e-12


def test_an_out_of_plane_walk_sweeps_to_a_valid_block():
    """The whole point of the 3-D turtle: a path no plane contains, swept in one call."""
    path = paths.walk([paths.line(1.0), paths.arc(1.2, 90.0),
                       paths.arc(1.2, 90.0, tilt=90.0), paths.helix(1.5, 180.0, rise=2.0),
                       paths.line(1.0, roll=30.0)],
                      start=ORIGIN, heading=HEADING, up=UP)
    s = _dense()
    P = path.centerline(s)
    normal = np.linalg.svd(P - P.mean(axis=0))[2][2]
    assert np.abs((P - P.mean(axis=0)) @ normal).max() > 0.5, "not actually out of plane"
    mesh = hexmesh.sweep_path(_section(2), path, target_length=0.25, origin=ORIGIN)
    assert hexmesh.is_watertight(mesh)
    assert hexmesh.is_conforming(mesh)
    assert hexmesh.scaled_jacobian(mesh).min() > 0.0
