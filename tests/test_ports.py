"""Unit tests for ``quadmesh.Port`` -- a section plus the two facts it cannot state
about itself -- and for the checks it lets ``bridge`` / ``adapter`` make.

The point of the type is that those two joins currently *guess*: they take each
section's outward direction to be the one pointing at the other section's centroid.
That is right whenever the two really do face each other, and silently wrong when they
do not -- it flips one of them and folds the connector, with nothing to catch it.

So the tests here come in two halves:

1. **The guess is preserved exactly.** A bare ``QuadMesh`` must still produce the
   identical mesh it did before ``Port`` existed, bit for bit -- otherwise every
   frozen example output moves for no reason.
2. **A stated direction is checked.** Given two ``Port``s, the joins verify what the
   guess cannot: that they face each other, and that they are the same size.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh

RADIAL = np.array([0.0, 0.5, 1.0])


def _disc(radius=1.0, center=(0.0, 0.0, 0.0), order=2, start_theta=0.0):
    ring = linemesh.circle(radius, 8, center=center, start_theta=start_theta,
                           element_tag="wall", order=order)
    return quadmesh.ogrid(ring, 2, RADIAL, wall_tag="wall")


# -- the type -----------------------------------------------------------------
def test_port_derives_normal_center_and_radius():
    sec = _disc(radius=1.5, center=(2.0, 0.0, 3.0))
    p = quadmesh.port(sec, outward=(0.0, 0.0, 1.0))
    assert np.allclose(p.normal, [0.0, 0.0, 1.0])
    assert np.allclose(p.center, [2.0, 0.0, 3.0], atol=1e-12)
    assert p.radius == pytest.approx(1.5, rel=1e-9)


def test_outward_is_only_a_sign_hint():
    """The precision comes from the section's own fitted plane, so a rough hint --
    a path tangent, the axis a component was built along -- is enough."""
    sec = _disc()
    exact = quadmesh.port(sec, outward=(0.0, 0.0, 1.0)).normal
    rough = quadmesh.port(sec, outward=(0.1, -0.2, 0.9)).normal
    assert np.array_equal(exact, rough)
    assert np.array_equal(quadmesh.port(sec, outward=(0.0, 0.0, -1.0)).normal, -exact)


def test_center_can_be_named_when_the_centroid_is_not_the_axis():
    sec = _disc()
    named = quadmesh.port(sec, outward=(0.0, 0.0, 1.0), center=(0.25, 0.0, 0.0))
    assert np.allclose(named.center, [0.25, 0.0, 0.0])
    assert named.radius > quadmesh.port(sec, outward=(0.0, 0.0, 1.0)).radius


def test_reversed_flips_only_the_normal():
    p = quadmesh.port(_disc(), outward=(0.0, 0.0, 1.0))
    r = p.reversed()
    assert np.array_equal(r.normal, -p.normal)
    assert r.section is p.section and np.array_equal(r.center, p.center)
    assert p.faces(r) and not p.faces(p)


@pytest.mark.parametrize("kwargs, match", [
    ({"normal": (0.0, 0.0, 2.0)}, "not a unit vector"),
    ({"normal": (0.0, 1.0)}, "must be a"),
    ({"center": (0.0, 1.0)}, "must be a"),
    ({"radius": 0.0}, "must be positive"),
    ({"radius": -1.0}, "must be positive"),
])
def test_port_validates_itself_at_construction(kwargs, match):
    sec = _disc()
    fields = {"section": sec, "normal": np.array([0.0, 0.0, 1.0]),
              "center": np.zeros(3), "radius": 1.0}
    fields.update(kwargs)
    with pytest.raises(ValueError, match=match):
        quadmesh.Port(**fields)


def test_port_has_no_equality_trap():
    """``eq=False``: the generated ``__eq__`` would compare ndarray fields and raise
    on the ambiguous truth value, exactly as the tag tables document."""
    p = quadmesh.port(_disc(), outward=(0.0, 0.0, 1.0))
    assert p == p                       # identity, not elementwise
    assert p != quadmesh.port(_disc(), outward=(0.0, 0.0, 1.0))


def test_repr_summarises():
    r = repr(quadmesh.port(_disc(), outward=(0.0, 0.0, 1.0)))
    assert r.startswith("<Port ") and "facing" in r


# -- the guess is preserved ---------------------------------------------------
def _far_pair(order=2):
    return (_disc(order=order),
            _disc(order=order, center=(0.0, 0.0, 6.0), start_theta=0.37))


@pytest.mark.parametrize("op", ["bridge", "adapter"])
def test_bare_sections_reproduce_the_old_guess_bit_for_bit(op):
    """A bare QuadMesh must give the identical mesh a stated pair does, when the
    stated directions agree with what the guess would have picked."""
    a, b = _far_pair()
    if op == "bridge":
        bare = hexmesh.bridge(a, b)
        stated = hexmesh.bridge(quadmesh.port(a, outward=(0.0, 0.0, 1.0)),
                                quadmesh.port(b, outward=(0.0, 0.0, -1.0)))
    else:
        b = _disc(radius=1.04, center=(0.0, 0.0, 1.0))
        bare = hexmesh.adapter(a, b, axis=(0.0, 0.0, 1.0))
        stated = hexmesh.adapter(quadmesh.port(a, outward=(0.0, 0.0, 1.0)),
                                 quadmesh.port(b, outward=(0.0, 0.0, -1.0)))
    assert np.array_equal(bare.points, stated.points)
    assert np.array_equal(bare.corners, stated.corners)
    assert np.array_equal(bare.interior, stated.interior)


# -- what a stated direction buys ---------------------------------------------
@pytest.mark.parametrize("op", ["bridge", "adapter"])
def test_ports_facing_the_same_way_are_refused(op):
    """The bug the guess cannot see: it would flip one of them and fold the
    connector."""
    a, b = _far_pair()
    pa = quadmesh.port(a, outward=(0.0, 0.0, 1.0))
    pb = quadmesh.port(b, outward=(0.0, 0.0, 1.0))     # same way, not facing
    with pytest.raises(ValueError, match="do not face each other"):
        getattr(hexmesh, op)(pa, pb)


@pytest.mark.parametrize("op", ["bridge", "adapter"])
def test_ports_of_different_size_are_refused(op):
    a = _disc(radius=1.0)
    b = _disc(radius=2.0, center=(0.0, 0.0, 6.0))
    with pytest.raises(ValueError, match="different sizes"):
        getattr(hexmesh, op)(quadmesh.port(a, outward=(0.0, 0.0, 1.0)),
                             quadmesh.port(b, outward=(0.0, 0.0, -1.0)))


def test_a_small_radius_difference_is_allowed():
    """Real components differ slightly -- chimera_full joins a 1.201 port to a 1.200
    one -- so the check is proportional, not exact."""
    a = _disc(radius=1.0)
    # start_theta avoids pi/8 (and odd multiples): ogrid's octagon core is 8-fold
    # symmetric, and a quarter-symmetric rotation there is the one angle where its
    # own point pattern nearly maps onto itself, which starves bridge's nearest-
    # neighbour pairing of a clean bijection between the two discs.
    b = _disc(radius=1.02, center=(0.0, 0.0, 6.0), start_theta=0.3)
    assert hexmesh.is_conforming(
        hexmesh.bridge(quadmesh.port(a, outward=(0.0, 0.0, 1.0)),
                       quadmesh.port(b, outward=(0.0, 0.0, -1.0))))


def test_checks_are_skipped_when_either_side_only_guessed():
    """A guessed direction is derived from the very geometry being checked, so
    checking it would only ever confirm itself."""
    a, b = _far_pair()
    # b guessed: no facing check, so a deliberately wrong stated `a` still builds
    hexmesh.bridge(quadmesh.port(a, outward=(0.0, 0.0, -1.0)), b)


def test_adapter_takes_its_roll_axis_from_the_port():
    a = _disc()
    b = _disc(radius=1.04, center=(0.0, 0.0, 1.0))
    explicit = hexmesh.adapter(a, b, axis=(0.0, 0.0, 1.0))
    inferred = hexmesh.adapter(quadmesh.port(a, outward=(0.0, 0.0, 1.0)),
                               quadmesh.port(b, outward=(0.0, 0.0, -1.0)))
    assert np.array_equal(explicit.points, inferred.points)


def test_adapter_still_demands_an_axis_for_bare_sections():
    a = _disc()
    b = _disc(radius=1.04, center=(0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="give axis="):
        hexmesh.adapter(a, b)


def test_port_is_reachable_from_the_namespace_module():
    assert quadmesh.ports.Port is quadmesh.Port
    assert quadmesh.ports.port is quadmesh.port
