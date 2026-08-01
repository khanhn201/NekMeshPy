"""``loft`` as the uniform sweep primitive at every rung of the B-rep ladder, and
its periodic ``loop=True`` mode.

``LineMesh.loft`` / ``QuadMesh.loft`` / ``HexMesh.loft`` all assemble the same way:
each profile is appended once, then the *rung* entities joining it to the previous
profile.  ``loop=True`` adds exactly one more rung block -- from the last profile
back to the **first** -- and appends no extra profile, so the seam is a genuine
shared entity rather than a duplicated layer.  These tests pin that: a lofted loop
matches the ``loop`` factory one dimension down, a revolved surface closes into a
torus with no free edge and no duplicate line, a revolved disc closes into a
watertight solid torus whose only free faces are the outer wall, and an end-cap tag
on a closed sweep is rejected at all three levels."""

import numpy as np
import pytest

from nekmeshpy import HexMesh, LineMesh, QuadMesh
from nekmeshpy.model import topology

R0, RSEC = 3.0, 1.0          # torus major / minor radius
NSEC, NRING = 8, 12          # sections around the axis / points around a section


def _ring_profiles(order=1, nsec=NSEC, nring=NRING):
    """``nsec`` closed section rings of a torus about the +y axis, one per angle
    ``2*pi*k/nsec`` -- index-paired, so they loft directly.

    Placing them is exactly the rung-preserving ``rotate``: it maps the ring's
    high-order ``interior`` by the same rigid map as its corners, so each profile
    stays an exact circle."""
    ring = LineMesh.circle(RSEC, nring, center=(R0, 0.0, 0.0),
                           element_tags=["wall"] * nring, order=order)
    return [ring.rotate(2.0 * np.pi * k / nsec, axis=(0.0, 1.0, 0.0))
            for k in range(nsec)]


def _disc_profiles(order=1, nsec=NSEC, nring=8):
    """The same sections filled with an O-grid disc, ready for a hex loft."""
    return [QuadMesh.ogrid(r, 2, np.linspace(0.5, 1.0, 2))
            for r in _ring_profiles(order=order, nsec=nsec, nring=nring)]


# -- rung 1: LineMesh.loft ----------------------------------------------------
@pytest.mark.parametrize("order", [1, 3])
def test_line_loft_loop_is_the_loop_factory(order):
    """One dimension down each profile is a single point and the rungs *are* the
    lines, so ``loft(loop=True)`` is what makes the curve closed at all."""
    P = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.5]])
    lofted = LineMesh.loft(P, loop=True, order=order)
    factory = LineMesh.loft(P, order=order, loop=True)
    assert np.array_equal(lofted.points, factory.points)
    assert np.array_equal(lofted.lines, factory.lines)
    assert np.array_equal(lofted.interior, factory.interior)
    # a closed sweep has no degree-1 end
    assert lofted.boundary_points().size == 0
    assert lofted.lines.tolist() == [[0, 1], [1, 2], [2, 3], [3, 0]]


@pytest.mark.parametrize("order", [1, 3])
def test_line_loft_open_is_the_open_factory(order):
    P = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.5]])
    lofted = LineMesh.loft(P, loop=False, order=order)
    factory = LineMesh.loft(P, order=order)
    assert np.array_equal(lofted.points, factory.points)
    assert np.array_equal(lofted.lines, factory.lines)
    assert np.array_equal(lofted.interior, factory.interior)
    # the closing rung is the only difference between the two modes
    assert lofted.n_lines == P.shape[0] - 1
    assert LineMesh.loft(P, loop=True).n_lines == P.shape[0]
    assert lofted.boundary_points().tolist() == [0, P.shape[0] - 1]


def test_line_loft_high_order_interior_is_the_straight_gll_blend():
    """With no explicit ``interior`` each line's private nodes are the straight GLL
    blend between its endpoints -- the same nodes ``LineMesh.line`` places."""
    from nekmeshpy.model.fields import gll_nodes
    P = np.array([[0.0, 0, 0], [2, 0, 0], [2, 3, 0]])
    lm = LineMesh.loft(P, loop=True, order=4)
    g = gll_nodes(4)[1:4]
    a, b = lm.points[lm.lines[:, 0]], lm.points[lm.lines[:, 1]]
    assert np.allclose(lm.interior,
                       a[:, None, :] + g[None, :, None] * (b - a)[:, None, :])


def test_line_loft_end_point_tags():
    """``first_tag``/``last_tag`` name the 1-D end caps: the chain's two end points."""
    P = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    lm = LineMesh.loft(P, first_tag="inlet", last_tag="outlet")
    assert lm.boundaries.tolist() == [[0, 1], [1, 2]]
    assert lm.boundary_group_tags == ["inlet", "outlet"]


# -- rung 2: QuadMesh.loft ----------------------------------------------------
@pytest.mark.parametrize("order", [1, 3])
def test_quad_loft_loop_builds_a_closed_torus_surface(order):
    """Revolving a closed ring with ``loop=True`` gives a torus surface: no free
    boundary edge, no duplicate line, and exactly the periodic element count."""
    profiles = _ring_profiles(order=order)
    torus = QuadMesh.loft(profiles, loop=True)

    # periodic count: NSEC layers, not NSEC-1 (a duplicated seam would inflate it)
    assert torus.n_quads == NSEC * NRING
    assert torus.n_points == NSEC * NRING

    # every stored line is referenced by exactly two quads -> zero free edges
    counts = np.bincount(torus.quad.ravel(), minlength=torus.lines.n_lines)
    assert int(np.sum(counts == 1)) == 0        # free boundary edges
    assert int(np.sum(counts == 0)) == 0        # unreferenced lines
    assert np.all(counts == 2)

    # zero duplicate lines: the seam rung was appended once
    key = np.sort(torus.lines.lines, axis=1)
    assert np.unique(key, axis=0).shape[0] == torus.lines.n_lines

    # every corner pair shared by two quads resolves to the SAME line index
    seen = {}
    quads = torus.quads
    for e in range(quads.shape[0]):
        for k in range(4):
            u, v = int(quads[e, k]), int(quads[e, (k + 1) % 4])
            pair = (min(u, v), max(u, v))
            lid = int(torus.quad[e, k])
            assert seen.setdefault(pair, lid) == lid
    assert len(seen) == torus.lines.n_lines

    # the surface really is a torus: every point at distance RSEC from the ring
    axis_r = np.hypot(torus.points[:, 0], torus.points[:, 2])
    d = np.hypot(axis_r - R0, torus.points[:, 1])
    assert np.allclose(d, RSEC)


def test_quad_loft_loop_beats_repeating_the_first_profile():
    """The ``loop=False`` stack that repeats profile 0 covers the same geometry with
    a duplicated seam layer: same quads, strictly more points and lines."""
    profiles = _ring_profiles()
    closed = QuadMesh.loft(profiles, loop=True)
    repeated = QuadMesh.loft([*profiles, profiles[0]], loop=False)
    assert closed.n_quads == repeated.n_quads
    assert closed.n_points < repeated.n_points
    assert closed.lines.n_lines < repeated.lines.n_lines
    # ... and the repeated stack is *not* closed: it has free cap edges
    counts = np.bincount(repeated.quad.ravel(),
                         minlength=repeated.lines.n_lines)
    assert int(np.sum(counts == 1)) == 2 * NRING


def test_quad_loft_loop_emits_no_cap_rows_but_keeps_side_walls():
    """No cap boundary row on a periodic sweep; side walls from the profiles' own
    tagged boundary points are unaffected."""
    chain = LineMesh.loft(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                          first_tag="left", last_tag="right")
    profiles = [chain.translate((0.0, 0.0, z)) for z in (0.0, 1.0, 2.0, 3.0)]
    closed = QuadMesh.loft(profiles, loop=True)
    # 4 layers x 2 lines, one side-wall edge per layer per tagged end point
    assert closed.n_quads == 4 * 2
    assert sorted(set(closed.boundary_tags.tolist())) == ["left", "right"]
    assert closed.boundaries.shape[0] == 2 * 4
    assert set(closed.boundaries[:, 1].tolist()) == {2, 4}   # never sides 1/3


# -- rung 3: HexMesh.loft -----------------------------------------------------
@pytest.mark.parametrize("order", [1, 3])
def test_hex_loft_loop_builds_a_watertight_solid_torus(order):
    """Revolving an O-grid disc with ``loop=True`` gives a solid torus: watertight,
    conformal, one component, and the only free faces are the outer wall."""
    profiles = _disc_profiles(order=order)
    solid = HexMesh.loft(profiles, loop=True)
    report = topology.hex_report(solid.points, solid.hexes)
    assert report.n_components == 1
    assert report.n_nonmanifold_faces == 0
    assert report.n_hanging_points == 0
    assert report.n_open_edges == 0
    assert report.watertight and report.conformal

    # the boundary is the wall only -- no cap faces (8 wall quads per section)
    n_wall = NSEC * 8
    assert report.n_boundary_faces == n_wall
    assert solid.boundaries.shape[0] == n_wall
    assert sorted(set(solid.boundary_tags.tolist())) == ["wall"]
    assert set(solid.boundaries[:, 1].tolist()).isdisjoint({5, 6})


def test_hex_loft_loop_beats_repeating_the_first_profile():
    profiles = _disc_profiles()
    closed = HexMesh.loft(profiles, loop=True)
    repeated = HexMesh.loft([*profiles, profiles[0]], loop=False)
    assert closed.n_hexes == repeated.n_hexes
    assert closed.n_points < repeated.n_points
    # the open stack still has its two cap face layers
    open_report = topology.hex_report(repeated.points, repeated.hexes)
    closed_report = topology.hex_report(closed.points, closed.hexes)
    assert (open_report.n_boundary_faces
            > closed_report.n_boundary_faces)


# -- cap tags are rejected on a closed sweep, at every rung -------------------
@pytest.mark.parametrize("cap", ["first_tag", "last_tag"])
def test_line_loft_loop_rejects_cap_tags(cap):
    with pytest.raises(ValueError, match="no near/far cap"):
        LineMesh.loft(np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]]),
                      loop=True, **{cap: "cap"})


@pytest.mark.parametrize("cap", ["first_tag", "last_tag"])
def test_quad_loft_loop_rejects_cap_tags(cap):
    with pytest.raises(ValueError, match="no near/far cap"):
        QuadMesh.loft(_ring_profiles(nsec=4), loop=True, **{cap: "cap"})


@pytest.mark.parametrize("cap", ["first_tag", "last_tag"])
def test_hex_loft_loop_rejects_cap_tags(cap):
    with pytest.raises(ValueError, match="no near/far cap"):
        HexMesh.loft(_disc_profiles(nsec=4), loop=True, **{cap: "cap"})


def test_loft_loop_rejects_per_element_cap_arrays():
    """An array cap tag is rejected too -- not just the scalar form."""
    profiles = _ring_profiles(nsec=4)
    with pytest.raises(ValueError, match="no near/far cap"):
        QuadMesh.loft(profiles, loop=True, first_tag=["cap"] * NRING)
