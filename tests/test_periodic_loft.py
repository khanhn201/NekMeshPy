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
on a closed sweep lands on the seam -- from whichever side names it."""

import numpy as np
import pytest

from nekmeshpy import ElementTags, hexmesh, linemesh, quadmesh
from nekmeshpy.core import topology

R0, RSEC = 3.0, 1.0          # torus major / minor radius
NSEC, NRING = 8, 12          # sections around the axis / points around a section


def _ring_profiles(order=1, nsec=NSEC, nring=NRING):
    """``nsec`` closed section rings of a torus about the +y axis, one per angle
    ``2*pi*k/nsec`` -- index-paired, so they loft directly.

    Placing them is exactly the rung-preserving ``rotate``: it maps the ring's
    high-order ``interior`` by the same rigid map as its corners, so each profile
    stays an exact circle."""
    ring = linemesh.circle(RSEC, nring, center=(R0, 0.0, 0.0),
                           element_tag="wall", order=order)
    return [linemesh.rotate(ring, 2.0 * np.pi * k / nsec, axis=(0.0, 1.0, 0.0))
            for k in range(nsec)]


def _disc_profiles(order=1, nsec=NSEC, nring=8):
    """The same sections filled with an O-grid disc, ready for a hex loft."""
    return [quadmesh.ogrid(r, 2, np.linspace(0.5, 1.0, 2))
            for r in _ring_profiles(order=order, nsec=nsec, nring=nring)]


# -- rung 1: LineMesh.loft ----------------------------------------------------
@pytest.mark.parametrize("order", [1, 3])
def test_line_loft_loop_is_the_loop_factory(order):
    """One dimension down each profile is a single point and the rungs *are* the
    lines, so ``loft(loop=True)`` is what makes the curve closed at all."""
    P = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.5]])
    lofted = linemesh.loft(P, loop=True, order=order)
    factory = linemesh.loft(P, order=order, loop=True)
    assert np.array_equal(lofted.points, factory.points)
    assert np.array_equal(lofted.lines, factory.lines)
    assert np.array_equal(lofted.interior, factory.interior)
    # a closed sweep has no degree-1 end
    assert linemesh.boundary_points(lofted).size == 0
    assert lofted.lines.tolist() == [[0, 1], [1, 2], [2, 3], [3, 0]]


@pytest.mark.parametrize("order", [1, 3])
def test_line_loft_open_is_the_open_factory(order):
    P = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.5]])
    lofted = linemesh.loft(P, loop=False, order=order)
    factory = linemesh.loft(P, order=order)
    assert np.array_equal(lofted.points, factory.points)
    assert np.array_equal(lofted.lines, factory.lines)
    assert np.array_equal(lofted.interior, factory.interior)
    # the closing rung is the only difference between the two modes
    assert lofted.n_lines == P.shape[0] - 1
    assert linemesh.loft(P, loop=True).n_lines == P.shape[0]
    assert linemesh.boundary_points(lofted).tolist() == [0, P.shape[0] - 1]


def test_line_loft_high_order_interior_is_the_straight_gll_blend():
    """With no explicit ``interior`` each line's private nodes are the straight GLL
    blend between its endpoints -- the same nodes ``LineMesh.line`` places."""
    from nekmeshpy.core.fields import gll_nodes
    P = np.array([[0.0, 0, 0], [2, 0, 0], [2, 3, 0]])
    lm = linemesh.loft(P, loop=True, order=4)
    g = gll_nodes(4)[1:4]
    a, b = lm.points[lm.lines[:, 0]], lm.points[lm.lines[:, 1]]
    assert np.allclose(lm.interior,
                       a[:, None, :] + g[None, :, None] * (b - a)[:, None, :])


def test_line_loft_end_point_tags():
    """``first_tag``/``last_tag`` name the 1-D end caps: the chain's two end points."""
    P = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    lm = linemesh.loft(P, first_tag="inlet", last_tag="outlet")
    # the two end points themselves, not one line's view of each
    assert list(lm.point_tags) == [(0, "inlet"), (2, "outlet")]
    assert lm.point_group_tags == ["inlet", "outlet"]


# -- rung 2: QuadMesh.loft ----------------------------------------------------
@pytest.mark.parametrize("order", [1, 3])
def test_quad_loft_loop_builds_a_closed_torus_surface(order):
    """Revolving a closed ring with ``loop=True`` gives a torus surface: no free
    boundary edge, no duplicate line, and exactly the periodic element count."""
    profiles = _ring_profiles(order=order)
    torus = quadmesh.loft(profiles, loop=True)

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
    closed = quadmesh.loft(profiles, loop=True)
    repeated = quadmesh.loft([*profiles, profiles[0]], loop=False)
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
    chain = linemesh.loft(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                          first_tag="left", last_tag="right")
    profiles = [linemesh.translate(chain, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0, 3.0)]
    closed = quadmesh.loft(profiles, loop=True)
    # 4 layers x 2 lines, one side-wall edge per layer per tagged end point
    assert closed.n_quads == 4 * 2
    assert sorted(set(closed.edge_tags.tags.tolist())) == ["left", "right"]
    assert len(closed.edge_tags) == 2 * 4
    assert set(closed.edge_tags.sides[:].tolist()) == {2, 4}   # never sides 1/3


# -- rung 3: HexMesh.loft -----------------------------------------------------
@pytest.mark.parametrize("order", [1, 3])
def test_hex_loft_loop_builds_a_watertight_solid_torus(order):
    """Revolving an O-grid disc with ``loop=True`` gives a solid torus: watertight,
    conformal, one component, and the only free faces are the outer wall."""
    profiles = _disc_profiles(order=order)
    solid = hexmesh.loft(profiles, loop=True)
    report = topology.hex_report(solid.points, solid.hexes)
    assert report.n_components == 1
    assert report.n_nonmanifold_faces == 0
    assert report.n_hanging_points == 0
    assert report.n_open_edges == 0
    assert report.watertight and report.conformal

    # the boundary is the wall only -- no cap faces (8 wall quads per section)
    n_wall = NSEC * 8
    assert report.n_boundary_faces == n_wall
    assert len(solid.face_tags) == n_wall
    assert sorted(set(solid.face_tags.tags.tolist())) == ["wall"]
    assert set(solid.face_tags.sides[:].tolist()).isdisjoint({5, 6})


def test_hex_loft_loop_beats_repeating_the_first_profile():
    profiles = _disc_profiles()
    closed = hexmesh.loft(profiles, loop=True)
    repeated = hexmesh.loft([*profiles, profiles[0]], loop=False)
    assert closed.n_hexes == repeated.n_hexes
    assert closed.n_points < repeated.n_points
    # the open stack still has its two cap face layers
    open_report = topology.hex_report(repeated.points, repeated.hexes)
    closed_report = topology.hex_report(closed.points, closed.hexes)
    assert (open_report.n_boundary_faces
            > closed_report.n_boundary_faces)


# -- cap tags on a closed sweep name the seam, at every rung ------------------
@pytest.mark.parametrize("cap", ["first_tag", "last_tag"])
def test_line_loft_loop_places_cap_tags(cap):
    """A closed sweep has no free end, but its seam is still a real side -- so a cap
    tag is placed rather than refused: the caller may well mean to name it."""
    lm = linemesh.loft(np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0]]),
                       loop=True, **{cap: "seam"})
    assert lm.point_tags.count("seam") == 1


@pytest.mark.parametrize("cap", ["first_tag", "last_tag"])
def test_quad_loft_loop_places_cap_tags(cap):
    qm = quadmesh.loft(_ring_profiles(nsec=4), loop=True, **{cap: "seam"})
    # one tagged edge per section line, on the seam layer's own side
    assert qm.edge_tags.count("seam") == NRING
    tagged = qm.edge_tags.select(qm.edge_tags.mask_for("seam"))
    assert set(tagged.sides.tolist()) == {{"first_tag": 1, "last_tag": 3}[cap]}


@pytest.mark.parametrize("cap", ["first_tag", "last_tag"])
def test_hex_loft_loop_places_cap_tags(cap):
    profiles = _disc_profiles(nsec=4)
    hm = hexmesh.loft(profiles, loop=True, **{cap: "seam"})
    tagged = hm.face_tags.select(hm.face_tags.mask_for("seam"))
    assert len(tagged) == profiles[0].n_quads       # one face per section quad
    assert set(tagged.sides.tolist()) == {{"first_tag": 5, "last_tag": 6}[cap]}


def test_loft_loop_places_a_per_line_cap_table():
    """The per-slice-element form lands on the seam too -- not just the scalar."""
    profiles = _ring_profiles(nsec=4)
    caps = ElementTags.from_dense(["seam"] + [""] * (NRING - 1))
    qm = quadmesh.loft(profiles, loop=True, first_tag=caps)
    assert qm.edge_tags.count("seam") == 1


def test_loft_loop_can_tag_one_side_of_the_seam_only():
    """The two caps are the same seam seen from either side, so naming one names one
    side: the whole point of allowing cap tags on a closed sweep."""
    qm = quadmesh.loft(_ring_profiles(nsec=4), loop=True, first_tag="in")
    both = quadmesh.loft(_ring_profiles(nsec=4), loop=True,
                         first_tag="in", last_tag="out")
    assert qm.edge_tags.count("in") == NRING and qm.edge_tags.count("out") == 0
    assert both.edge_tags.count("in") == both.edge_tags.count("out") == NRING
