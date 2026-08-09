"""Unit tests for ``paths.embed`` -- lifting a 2-D turtle walk onto a plane in space --
and for the ``sweep_path`` that consumes the result at the quad and hex rungs.

What these pin down, in order of how badly getting it wrong would hurt:

1. **The origin enters the centerline and not the tangent.**  A tangent is a
   direction; translating it tilts it, every frame along the sweep inherits the tilt,
   and the section stops being perpendicular to the path.  That is a wrong *mesh*, not
   an exception, which is why the lift is one shared function rather than six
   hand-written copies.
2. **The lift is exact, not merely close.**  It is the composition of the walk with a
   linear map, so it must reproduce the hand-written ``origin + x*u + y*v`` bit for
   bit -- the examples' frozen outputs depend on it.
3. **The axes are used verbatim.**  A non-unit or non-orthogonal pair would rescale or
   shear the walk, silently putting ``total_length`` and ``break_fractions`` out of
   step with the curve they describe, so both are rejected up front.
4. **``sweep_path`` is exactly ``sweep``** with its stations resolved from the path,
   and resolves them so that every straight/arc junction still carries one.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core import paths
from nekmeshpy.core.fields import uniform_spacing

MOVES = [("line", 2.0, 0.0), ("arc", 1.5, 90.0), ("line", 3.0, 0.0),
         ("arc", 1.5, -90.0), ("line", 1.0, 0.0)]
U, V = (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)
ORIGIN = (0.5, -2.0, 7.0)
S = np.linspace(0.0, 1.0, 11)


@pytest.fixture
def walk():
    return paths.turtle_path(MOVES, start=(0.0, 0.0), heading=0.0)


def _reference(walk, s, *, origin):
    """The hand-written lift every example used to carry, verbatim."""
    u, v, o = np.asarray(U), np.asarray(V), np.asarray(origin, dtype=float)
    xy = walk.centerline(s)
    tx = walk.tangent(s)
    return (o + (xy[:, 0, None] * u + xy[:, 1, None] * v),
            tx[:, 0, None] * u + tx[:, 1, None] * v)


def test_embed_matches_the_hand_written_lift_bit_for_bit(walk):
    path = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    ref_c, ref_t = _reference(walk, S, origin=ORIGIN)
    assert np.array_equal(path.centerline(S), ref_c)
    assert np.array_equal(path.tangent(S), ref_t)


def test_embed_carries_length_and_breaks_through(walk):
    path = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    assert path.total_length == walk.total_length
    assert np.array_equal(path.break_fractions, walk.break_fractions)


def test_origin_translates_the_centerline_and_leaves_the_tangent_alone(walk):
    """The property the whole helper exists for."""
    at_zero = paths.embed(walk, u=U, v=V)
    shifted = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    delta = shifted.centerline(S) - at_zero.centerline(S)
    assert np.allclose(delta, np.asarray(ORIGIN, dtype=float), atol=0.0)
    assert np.array_equal(shifted.tangent(S), at_zero.tangent(S))


def test_embedded_tangent_is_unit_and_is_the_centerline_derivative(walk):
    path = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    T = path.tangent(S)
    assert np.allclose(np.linalg.norm(T, axis=1), 1.0)
    # finite-difference the centerline away from the junctions, where curvature jumps
    h, mid = 1e-7, np.array([0.12, 0.37, 0.62, 0.88])
    fd = (path.centerline(mid + h) - path.centerline(mid - h)) / (2.0 * h)
    fd = fd / np.linalg.norm(fd, axis=1)[:, None]
    assert np.allclose(fd, path.tangent(mid), atol=1e-6)


def test_embedded_path_stays_in_its_own_plane(walk):
    path = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    normal = np.cross(U, V)
    off = (path.centerline(S) - np.asarray(ORIGIN, dtype=float)) @ normal
    assert np.allclose(off, 0.0, atol=0.0)


@pytest.mark.parametrize("u, v, message", [
    ((0.0, 0.0, 2.0), (-1.0, 0.0, 0.0), "not a unit vector"),
    ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), "not orthogonal"),
    # unit, so it clears the length check and is caught by the orthogonality one
    ((0.0, 0.0, 1.0), (np.sqrt(0.5), 0.0, np.sqrt(0.5)), "not orthogonal"),
])
def test_embed_rejects_axes_that_would_rescale_or_shear(walk, u, v, message):
    with pytest.raises(ValueError, match=message):
        paths.embed(walk, u=u, v=v)


def test_embed_rejects_a_2d_origin(walk):
    with pytest.raises(ValueError, match="origin must be a"):
        paths.embed(walk, u=U, v=V, origin=(0.0, 1.0))


# -- station resolution -------------------------------------------------------
def test_path_fractions_puts_a_station_on_every_junction(walk):
    path = paths.embed(walk, u=U, v=V)
    fr = linemesh.path_fractions(path, target_length=0.4)
    assert np.all(np.diff(fr) > 0.0)
    assert fr[0] == 0.0 and fr[-1] == 1.0
    assert np.isin(path.break_fractions, fr).all()


def test_layers_is_the_average_element_length(walk):
    path = paths.embed(walk, u=U, v=V)
    n = 7
    assert np.array_equal(linemesh.path_fractions(path, layers=n),
                          linemesh.path_fractions(path,
                                                  target_length=path.total_length / n))


def test_fractions_are_handed_through_verbatim(walk):
    path = paths.embed(walk, u=U, v=V)
    mine = np.array([0.0, 0.3, 0.9, 1.0])
    assert np.array_equal(linemesh.path_fractions(path, fractions=mine), mine)


@pytest.mark.parametrize("kwargs", [
    {},
    {"layers": 3, "target_length": 1.0},
    {"layers": 3, "fractions": [0.0, 1.0]},
])
def test_path_fractions_demands_exactly_one_spacing_argument(walk, kwargs):
    path = paths.embed(walk, u=U, v=V)
    with pytest.raises(ValueError, match="exactly one of"):
        linemesh.path_fractions(path, **kwargs)


def test_path_fractions_rejects_a_zero_layer_count(walk):
    path = paths.embed(walk, u=U, v=V)
    with pytest.raises(ValueError, match="layers must be"):
        linemesh.path_fractions(path, layers=0)


# -- sweep_path, at both rungs ------------------------------------------------
def _profile(order):
    return linemesh.circle(0.3, 8, center=(0.5, -2.0, 7.0), normal=(0.0, 0.0, 1.0),
                           element_tag="wall", order=order)


def _section(order):
    return quadmesh.ogrid(_profile(order), 2, uniform_spacing(2), wall_tag="wall")


@pytest.mark.parametrize("order", [1, 2])
def test_hex_sweep_path_reproduces_sweep(walk, order):
    path = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    fr = linemesh.path_fractions(path, target_length=0.5)
    direct = hexmesh.sweep(_section(order), path.centerline, fr, tangent=path.tangent,
                           orientation="fixed", up=(0.0, 1.0, 0.0), origin=ORIGIN,
                           first_tag="inlet", last_tag="outlet")
    viapath = hexmesh.sweep_path(_section(order), path, target_length=0.5,
                                 orientation="fixed", up=(0.0, 1.0, 0.0), origin=ORIGIN,
                                 first_tag="inlet", last_tag="outlet")
    assert np.array_equal(viapath.points, direct.points)
    assert np.array_equal(viapath.hexes, direct.hexes)
    assert np.array_equal(viapath.interior, direct.interior)


@pytest.mark.parametrize("order", [1, 2])
def test_quad_sweep_path_reproduces_sweep(walk, order):
    path = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    fr = linemesh.path_fractions(path, layers=9)
    direct = quadmesh.sweep(_profile(order), path.centerline, fr, tangent=path.tangent,
                            orientation="fixed", up=(0.0, 1.0, 0.0), origin=ORIGIN)
    viapath = quadmesh.sweep_path(_profile(order), path, layers=9,
                                  orientation="fixed", up=(0.0, 1.0, 0.0),
                                  origin=ORIGIN)
    assert np.array_equal(viapath.points, direct.points)
    assert np.array_equal(viapath.quads, direct.quads)


def test_swept_block_is_valid_and_not_inverted(walk):
    """A planar walk with ``up`` normal to its plane is the exact, zero-twist case."""
    path = paths.embed(walk, u=U, v=V, origin=ORIGIN)
    mesh = hexmesh.sweep_path(_section(2), path, target_length=0.4,
                              orientation="fixed", up=(0.0, 1.0, 0.0), origin=ORIGIN,
                              first_tag="inlet", last_tag="outlet")
    assert hexmesh.is_watertight(mesh)
    assert hexmesh.is_conforming(mesh)
    assert hexmesh.scaled_jacobian(mesh).min() > 0.0
    assert set(mesh.face_group_tags) == {"wall", "inlet", "outlet"}
