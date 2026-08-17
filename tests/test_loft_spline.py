"""Unit tests for ``loft_spline`` at all three rungs.

``loft`` blends its sweep-direction nodes straight between the two profiles bounding a
layer, so a swept curved surface is high-order in *storage* and linear in *geometry* --
refining the order does not help, because the nodes it adds land on the same chord.
``loft_spline`` fits a cubic through the whole stack instead, so a station's nodes bend
the way the stack bends.

Two contracts, and the second is the one that keeps the goldens frozen:

* it **interpolates** -- every profile handed in comes back verbatim as a level, so this
  adds curvature between profiles without moving any; and
* at ``order 1`` it *is* ``loft``, node for node, because there are no interior nodes for
  a spline to place differently.

Errors are measured on the **conformal node set** (``element_blocks``), never on corners:
corner-only agreement is exactly what a mesh linear in geometry already has.
"""

import numpy as np
import pytest

from nekmeshpy import ElementTags, hexmesh, linemesh, quadmesh

R, RT, NU = 2.0, 0.6, 8
ORDER = 3


def _ring(order, n=NU):
    """A tube cross-section of the torus, sitting at ``theta = 0``."""
    return linemesh.circle(RT, n, center=(R, 0.0, 0.0), normal=(0, 1, 0), order=order)


def _rings(order, n_prof, n=NU):
    """``n_prof`` exact tube rings placed around the torus by rotation."""
    base = _ring(order, n)
    th = np.linspace(0.0, 2.0 * np.pi, n_prof + 1)[:-1]
    return [linemesh.rotate(base, float(t), axis=(0, 0, 1)) for t in th]


def _torus_err(pts):
    """Distance of each node from the exact torus surface."""
    rho = np.hypot(pts[:, 0], pts[:, 1])
    return np.abs(np.hypot(rho - R, pts[:, 2]) - RT)


# -- the bottom rung ---------------------------------------------------------
def test_line_spline_beats_the_chord_on_a_circle():
    """A chain of points around a bend comes out bent, not faceted."""
    th = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    pts = np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])

    straight = linemesh.loft(pts, loop=True, order=ORDER)
    fitted = linemesh.loft_spline(pts, loop=True, order=ORDER)

    def radial(m):
        b = linemesh.element_blocks(m).reshape(-1, 3)
        return np.abs(np.linalg.norm(b[:, :2], axis=1) - 1.0).max()

    assert radial(fitted) < radial(straight) / 20.0


def test_line_spline_interpolates_its_points():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0],
                    [3.0, 1.5, 0.0], [4.0, 0.0, 0.0]])
    m = linemesh.loft_spline(pts, order=ORDER)
    assert np.allclose(m.points, pts, atol=0.0)


def test_line_spline_is_loft_at_order_one():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0],
                    [3.0, 1.5, 0.0]])
    a = linemesh.loft(pts, element_tags="w", first_tag="lo", last_tag="hi")
    b = linemesh.loft_spline(pts, element_tags="w", first_tag="lo", last_tag="hi")
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(np.asarray(a.lines), np.asarray(b.lines))
    assert list(a.point_tags) == list(b.point_tags)
    assert np.array_equal(a.element_tags.dense(a.n_lines),
                          b.element_tags.dense(b.n_lines))


# -- the middle rung ---------------------------------------------------------
def test_quad_spline_beats_loft_on_a_torus():
    """The trap the module docs name: exact profiles, chord between them."""
    profs = _rings(ORDER, 8)
    straight = quadmesh.loft(profs, loop=True)
    fitted = quadmesh.loft_spline(profs, loop=True)

    e_straight = _torus_err(quadmesh.element_blocks(straight).reshape(-1, 3)).max()
    e_fitted = _torus_err(quadmesh.element_blocks(fitted).reshape(-1, 3)).max()
    # the chord is tens of percent of the tube radius out; the spline is a small
    # fraction of a percent, so the bar is deliberately far from either
    assert e_straight > 0.1 * RT
    assert e_fitted < e_straight / 50.0


def test_quad_spline_interpolates_its_profiles():
    """Level ``k`` is profile ``k``, node for node -- corners and curved alike."""
    profs = _rings(ORDER, 8)
    fitted = quadmesh.loft_spline(profs, loop=True)
    nn = profs[0].n_points
    for k, p in enumerate(profs):
        assert np.allclose(fitted.points[k * nn:(k + 1) * nn], p.points, atol=1e-12)


def test_quad_spline_matches_loft_at_order_one():
    profs = _rings(1, 8)
    a = quadmesh.loft(profs, loop=True, element_tags="w")
    b = quadmesh.loft_spline(profs, loop=True, element_tags="w")
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(np.asarray(a.corners), np.asarray(b.corners))
    assert np.array_equal(a.element_tags.dense(a.n_quads),
                          b.element_tags.dense(b.n_quads))


def test_quad_spline_carries_the_tag_arguments():
    profs = _rings(ORDER, 6)
    m = quadmesh.loft_spline(profs[:4], element_tags="skin",
                             first_tag="lo", last_tag="hi")
    names = set(np.asarray(m.edge_tags.tags))
    assert {"lo", "hi"} <= names
    assert set(m.element_tags.dense(m.n_quads)) == {"skin"}


def test_quad_spline_rejects_a_mismatched_profile():
    profs = _rings(ORDER, 6)
    profs[2] = _ring(ORDER, NU + 2)
    with pytest.raises(ValueError, match="index-paired"):
        quadmesh.loft_spline(profs)


# -- the top rung ------------------------------------------------------------
def _section(order):
    """A small section in the x-z plane at ``x = R``, to sweep about the z axis."""
    d = 0.2
    corners = [(R - d, 0.0, -d), (R + d, 0.0, -d), (R + d, 0.0, d), (R - d, 0.0, d)]
    return quadmesh.rectangle(corners, 2, 2, order=order)


def _sections(order, n_prof):
    base = _section(order)
    th = np.linspace(0.0, 2.0 * np.pi, n_prof + 1)[:-1]
    return [quadmesh.rotate(base, float(t), axis=(0, 0, 1)) for t in th]


def test_hex_spline_beats_loft_on_a_swept_ring():
    """Measured on the **inner** wall, whose exact radius is known.

    ``|rho - R|`` over every node would mostly report the section's own extent rather
    than the sweep error, and the rung's ``volume`` cannot see this at all -- it reads
    corners only, so a curved block and its straight-sided twin measure the same.  The
    innermost node of a solid of revolution, though, sits at exactly ``R - d``, and a
    chord sweep pulls it inward by the sagitta of the layer it cuts across."""
    secs = _sections(ORDER, 8)
    inner = R - 0.2

    def wall_err(m):
        b = hexmesh.element_blocks(m).reshape(-1, 3)
        return abs(float(np.hypot(b[:, 0], b[:, 1]).min()) - inner)

    e_straight = wall_err(hexmesh.loft(secs, loop=True))
    e_fitted = wall_err(hexmesh.loft_spline(secs, loop=True))
    assert e_straight > 0.05 * inner
    assert e_fitted < e_straight / 20.0


def test_hex_spline_interpolates_its_sections():
    secs = _sections(ORDER, 8)
    fitted = hexmesh.loft_spline(secs, loop=True)
    nn = secs[0].n_points
    for k, s in enumerate(secs):
        assert np.allclose(fitted.points[k * nn:(k + 1) * nn], s.points, atol=1e-12)


def test_hex_spline_builds_a_valid_block():
    secs = _sections(ORDER, 8)
    m = hexmesh.loft_spline(secs, loop=True)
    assert hexmesh.is_watertight(m)
    assert hexmesh.is_conforming(m)
    assert hexmesh.scaled_jacobian(m).min() > 0.0


def test_hex_spline_matches_loft_at_order_one():
    secs = _sections(1, 8)
    a = hexmesh.loft(secs, loop=True, element_tags="core")
    b = hexmesh.loft_spline(secs, loop=True, element_tags="core")
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(np.asarray(a.corners), np.asarray(b.corners))
    assert np.array_equal(a.element_tags.dense(a.n_hexes),
                          b.element_tags.dense(b.n_hexes))


def test_hex_spline_takes_element_tags_over_a_slice():
    """The same tag shapes ``loft`` takes -- an ``ElementTags`` over one slice."""
    secs = _sections(ORDER, 6)
    per = ElementTags.uniform(secs[0].n_quads, "col")
    m = hexmesh.loft_spline(secs[:4], element_tags=per)
    assert set(m.element_tags.dense(m.n_hexes)) == {"col"}
