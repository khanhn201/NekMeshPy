"""Unit tests for ``QuadMesh.sweep`` / ``HexMesh.sweep`` -- one cross-section carried
along a curved path by a moving frame -- and for the ``model.frames`` placement
primitives they are built on.

The contract these pin down, in order of how badly getting it wrong would hurt:

1. **The section moves rigidly.**  A sweep is a placement problem, not an offset
   problem.  Through a bend of radius ``Rb`` a node ``d`` outboard of the centreline
   traverses radius ``Rb + d`` and one ``d`` inboard traverses ``Rb - d`` -- they cover
   different distances and neither follows the path.  Offsetting every section point
   along its own copy of the curve would keep them equal, which shears the section and,
   on a tight bend, inverts the inboard elements.
2. **The sweep direction is exact at every order**, because ``sweep`` evaluates the
   placement at the intermediate GLL levels too and delegates through ``sweep_nodes``.
   A plain ``loft`` of the same corner-level placements is straight between them.
3. **A straight path reproduces ``extrude``**, and the section lands at station 0 in
   the orientation it was authored in -- so ``sweep`` is a strict generalization rather
   than a second, subtly different way to place a block.
"""

import numpy as np
import pytest
from conftest import assert_same_side_tags

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.model import conform, frames
from nekmeshpy.model.fields import uniform_spacing

RP, RB, NU, NS, NR = 0.3, 1.0, 16, 4, 2


def elbow(s):
    """A quarter turn of radius ``RB`` in the ``xz`` plane, about the ``y`` axis."""
    a = np.asarray(s, dtype=float) * (np.pi / 2)
    return np.stack([RB * np.cos(a), np.zeros_like(a), RB * np.sin(a)], axis=-1)


def elbow_t(s):
    a = np.asarray(s, dtype=float) * (np.pi / 2)
    return np.stack([-np.sin(a), np.zeros_like(a), np.cos(a)], axis=-1)


def uturn(s):
    """A 180-degree turn -- the case the user asked for, and the one where the inner
    and outer walls travel most differently."""
    a = np.asarray(s, dtype=float) * np.pi
    return np.stack([RB * np.cos(a), np.zeros_like(a), RB * np.sin(a)], axis=-1)


def straight(s):
    s = np.asarray(s, dtype=float)
    return np.stack([np.zeros_like(s), np.zeros_like(s), s], axis=-1)


def disc(order=1, normal=(0, 0, 1), center=(RB, 0.0, 0.0), tag="wall"):
    """An O-grid disc of radius ``RP``, wall-tagged on the loop (the lowest rung)."""
    ring = linemesh.circle(RP, NU, center=center, normal=normal, order=order,
                           element_tags=[tag] * NU)
    return quadmesh.ogrid(ring, NS, uniform_spacing(NR))


def hex_nodes(b):
    """Every conformal node of the block, corners and high-order alike."""
    nodes, _ = conform.conformal_hex(b.points, b.hexes, b._elem_edges, b._edge_flip,
                                     b.quads.lines.interior, b.hex, b.face_orient,
                                     b.quads.interior, b.interior, b.order)
    return nodes


def tube_radius(nodes):
    """Distance of each node from the elbow's own centreline circle."""
    x, y, z = np.asarray(nodes).T
    return np.hypot(np.hypot(x, z) - RB, y)


# -- the sweep direction is exact ---------------------------------------------

@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_node_lies_on_the_true_bent_tube(order):
    blk = hexmesh.sweep(disc(order), elbow, np.linspace(0.0, 1.0, 5),
                        orientation="fixed", up=(0, 1, 0), origin=(RB, 0.0, 0.0))
    assert np.max(tube_radius(hex_nodes(blk))) == pytest.approx(RP, abs=1e-13)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_a_plain_loft_of_the_same_placements_is_straight_along_the_sweep(order):
    """The baseline: exact placements at the corner levels are not enough."""
    sec = disc(order)
    fr = np.linspace(0.0, 1.0, 5)
    places = frames.sweep_placements(sec.points, elbow(fr), orientation="fixed",
                                     up=(0, 1, 0), origin=(RB, 0.0, 0.0))
    plain = hexmesh.loft([quadmesh.transform(sec, m, o) for m, o in places])
    # the wall bulges off the true tube -- small in absolute terms, but far above the
    # 1e-13 the evaluated sweep achieves, and it does not shrink with the order
    assert np.max(tube_radius(hex_nodes(plain))) > RP + 1e-6


# -- the placement is rigid, which is the whole design constraint --------------

@pytest.mark.parametrize("path", [elbow, uturn])
def test_the_section_is_moved_rigidly_at_every_station(path):
    """Every pairwise distance inside the section survives every placement exactly.

    This is the property that distinguishes a swept solid from a point-by-point offset
    of the profile, and it is what keeps element quality identical to the section's.
    """
    sec = disc(2)
    fr = np.linspace(0.0, 1.0, 7)
    P = path(fr)
    places = frames.sweep_placements(sec.points, P, orientation="fixed", up=(0, 1, 0),
                                     origin=(RB, 0.0, 0.0))
    ref = np.linalg.norm(sec.points[:, None, :] - sec.points[None, :, :], axis=-1)
    for m, o in places:
        Q = sec.points @ np.asarray(m).T + o
        got = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=-1)
        assert np.allclose(got, ref, atol=1e-13)


def test_inboard_and_outboard_walls_travel_different_distances():
    """The user's constraint stated as a measurement: through a U-turn of radius ``RB``
    the outer wall sweeps ``pi*(RB+RP)`` and the inner ``pi*(RB-RP)``.  A point-by-point
    offset would give both of them ``pi*RB``.
    """
    blk = hexmesh.sweep(disc(1), uturn, np.linspace(0.0, 1.0, 33),
                        orientation="fixed", up=(0, 1, 0), origin=(RB, 0.0, 0.0))
    r = np.hypot(blk.points[:, 0], blk.points[:, 2])
    assert r.max() == pytest.approx(RB + RP, abs=1e-12)
    assert r.min() == pytest.approx(RB - RP, abs=1e-12)
    # ... so the two walls differ in swept length by 2*pi*RP, not by zero
    assert (r.max() - r.min()) * np.pi == pytest.approx(2.0 * np.pi * RP, abs=1e-12)


def test_a_bend_tighter_than_the_section_folds_and_is_caught_not_hidden():
    """Nothing *prevents* a fold -- a bend radius below the section's own in-plane
    extent turns the inboard elements inside out -- but it is rejected loudly rather
    than silently meshed: ``loft`` sees layer 0 come out mixed-winding."""
    def tight(s):
        a = np.asarray(s, dtype=float) * np.pi
        return np.stack([0.2 * np.cos(a), np.zeros_like(a), 0.2 * np.sin(a)], axis=-1)

    with pytest.raises(ValueError, match="a sweep folded the section"):
        hexmesh.sweep(disc(1, center=(0.2, 0.0, 0.0)), tight,
                      np.linspace(0.0, 1.0, 17), orientation="fixed", up=(0, 1, 0),
                      origin=(0.2, 0.0, 0.0))


# -- a straight path is exactly ``extrude`` ------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_a_straight_path_reproduces_extrude(order):
    sec = disc(order, center=(0.0, 0.0, 0.0))
    sw = hexmesh.sweep(sec, straight, np.linspace(0.0, 2.0, 5),
                       orientation="fixed", up=(1, 0, 0), origin=(0.0, 0.0, 0.0),
                       first_tag="in", last_tag="out")
    ex = hexmesh.extrude(sec, axis=(0, 0, 1), length=2.0, layers=uniform_spacing(4),
                         first_tag="in", last_tag="out")
    assert np.allclose(sw.points, ex.points, atol=1e-14)
    assert np.array_equal(sw.hexes, ex.hexes)
    assert_same_side_tags(sw.face_tags, ex.face_tags)


def test_the_section_lands_at_station_zero_as_authored():
    """Every frame generator fixes the field only up to a constant roll; ``sweep``
    pins that phase to the section's own orientation, so the three agree."""
    sec = disc(2)
    fr = np.linspace(0.0, 1.0, 5)
    kw = dict(origin=(RB, 0.0, 0.0), tangent=elbow_t)
    a = hexmesh.sweep(sec, elbow, fr, orientation="fixed", up=(0, 1, 0), **kw)
    b = hexmesh.sweep(sec, elbow, fr, orientation="transport", **kw)
    # a planar path has zero torsion, so the transport frame *is* the fixed-up frame
    assert np.allclose(a.points, b.points, atol=1e-13)
    assert np.allclose(a.points[:sec.n_points], sec.points, atol=1e-13)


def test_differenced_tangents_tilt_the_frames_and_an_analytic_tangent_fixes_it():
    """The accuracy limit ``tangent=`` exists to remove: without it the *centreline* is
    placed exactly but the frames are finite differences, so the section is tilted."""
    sec, fr = disc(1), np.linspace(0.0, 1.0, 5)
    kw = dict(orientation="fixed", up=(0, 1, 0), origin=(RB, 0.0, 0.0))
    diffed = hexmesh.sweep(sec, elbow, fr, **kw)
    exact = hexmesh.sweep(sec, elbow, fr, tangent=elbow_t, **kw)
    # the inlet cap should lie exactly in the z = 0 plane; differencing tilts it
    n = sec.n_points
    assert np.max(np.abs(diffed.points[:n, 2])) > 1e-5
    assert np.max(np.abs(exact.points[:n, 2])) < 1e-15
    assert np.allclose(exact.points[:n], sec.points, atol=1e-14)


# -- the closed sweep ----------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2])
def test_loop_gives_a_closed_torus_with_no_duplicated_layer(order):
    def ring_path(t):
        t = np.asarray(t, dtype=float)
        return np.stack([RB * np.cos(t), RB * np.sin(t), np.zeros_like(t)], axis=-1)

    sec = disc(order, normal=(0, 1, 0))
    nz = 12
    tor = hexmesh.sweep(sec, ring_path, np.linspace(0.0, 2.0 * np.pi, nz + 1),
                        loop=True, origin=(RB, 0.0, 0.0))
    assert tor.n_hexes == sec.n_quads * nz          # nz layers, not nz-1
    assert tor.n_points == sec.n_points * nz        # no seam profile duplicated
    # closed in the sweep direction: the only boundary left is the tube wall, and it
    # is a single unbroken sleeve of one face per wall line per layer -- no caps
    assert list(np.unique(tor.face_tags.tags)) == ["wall"]
    assert len(tor.face_tags) == NU * nz
    assert hexmesh.is_watertight(tor) and hexmesh.is_conforming(tor)
    x, y, z = hex_nodes(tor).T
    assert np.max(np.hypot(np.hypot(x, y) - RB, z)) == pytest.approx(RP, abs=1e-13)


def test_loop_rejects_end_caps():
    with pytest.raises(ValueError):
        hexmesh.sweep(disc(1), elbow, np.linspace(0.0, 1.0, 5), loop=True,
                      origin=(RB, 0.0, 0.0), first_tag="in")


def test_loop_needs_three_fractions():
    with pytest.raises(ValueError, match="at least 3 fractions"):
        hexmesh.sweep(disc(1), elbow, np.array([0.0, 1.0]), loop=True,
                      origin=(RB, 0.0, 0.0))


def test_needs_two_fractions():
    with pytest.raises(ValueError, match="at least 2 fractions"):
        hexmesh.sweep(disc(1), elbow, np.array([0.0]), origin=(RB, 0.0, 0.0))


# -- tags ----------------------------------------------------------------------

def test_the_wall_tag_rides_up_from_the_loop_and_the_caps_are_named():
    blk = hexmesh.sweep(disc(1), elbow, np.linspace(0.0, 1.0, 5),
                        orientation="fixed", up=(0, 1, 0), origin=(RB, 0.0, 0.0),
                        first_tag="inlet", last_tag="outlet")
    assert sorted(blk.face_group_tags) == ["inlet", "outlet", "wall"]


def test_per_layer_element_tags_override_the_section_tags():
    sec = disc(1)
    blk = hexmesh.sweep(sec, elbow, np.linspace(0.0, 1.0, 4),
                        orientation="fixed", up=(0, 1, 0), origin=(RB, 0.0, 0.0),
                        element_tags=["", "hot", ""])
    tags = blk.element_tags.dense(blk.n_hexes).reshape(3, sec.n_quads)      # hex e = layer*M + q
    assert list(np.unique(tags[1])) == ["hot"]
    assert list(np.unique(tags[0])) == list(np.unique(tags[2])) == [""]


# -- validation ------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_order_is_the_sections_own(order):
    """``sweep`` takes no ``order``: a rigid placement cannot change the section's
    order, so asking the caller to restate it could only ever disagree with it."""
    sec = disc(order)
    blk = hexmesh.sweep(sec, elbow, np.linspace(0.0, 1.0, 5), origin=(RB, 0.0, 0.0),
                        orientation="fixed", up=(0, 1, 0))
    assert blk.order == sec.order == order


def test_rejects_a_path_that_does_not_return_k_by_3():
    with pytest.raises(ValueError, match=r"\(5,3\) array of centreline points"):
        hexmesh.sweep(disc(1), lambda s: np.zeros((len(s), 2)),
                      np.linspace(0.0, 1.0, 5), origin=(RB, 0.0, 0.0))


def test_fixed_without_up_is_an_actionable_error():
    with pytest.raises(ValueError, match="orientation='fixed' needs up="):
        hexmesh.sweep(disc(1), elbow, np.linspace(0.0, 1.0, 5),
                      origin=(RB, 0.0, 0.0), orientation="fixed")


def test_rejects_an_unknown_orientation():
    with pytest.raises(ValueError, match="'transport', 'fixed' or 'frenet'"):
        hexmesh.sweep(disc(1), elbow, np.linspace(0.0, 1.0, 5),
                      origin=(RB, 0.0, 0.0), orientation="rmf")


def test_a_per_station_up_must_match_the_lattice():
    with pytest.raises(ValueError, match=r"must be \(5,3\)"):
        hexmesh.sweep(disc(1), elbow, np.linspace(0.0, 1.0, 5),
                      origin=(RB, 0.0, 0.0), orientation="fixed",
                      up=np.tile([0.0, 1.0, 0.0], (4, 1)))


def test_a_per_station_up_is_accepted():
    fr = np.linspace(0.0, 1.0, 5)
    up = np.tile([0.0, 1.0, 0.0], (fr.shape[0], 1))
    got = hexmesh.sweep(disc(1), elbow, fr, orientation="fixed", up=up,
                        origin=(RB, 0.0, 0.0))
    want = hexmesh.sweep(disc(1), elbow, fr, orientation="fixed", up=(0, 1, 0),
                         origin=(RB, 0.0, 0.0))
    assert np.allclose(got.points, want.points, atol=1e-13)


def test_a_per_station_up_needs_the_fixed_frame():
    """A transported frame is pinned by its seed alone, so K-1 of the rows would be
    silently discarded -- the mode and the data are separate arguments precisely so
    that this combination can be named and rejected."""
    fr = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="needs orientation='fixed'"):
        hexmesh.sweep(disc(1), elbow, fr, origin=(RB, 0.0, 0.0),
                      orientation="transport",
                      up=np.tile([0.0, 1.0, 0.0], (fr.shape[0], 1)))


def test_frenet_refuses_the_straight_run_it_cannot_frame():
    with pytest.raises(ValueError):
        hexmesh.sweep(disc(1, center=(0.0, 0.0, 0.0)), straight,
                      np.linspace(0.0, 2.0, 5), origin=(0.0, 0.0, 0.0),
                      orientation="frenet")


# -- twist -------------------------------------------------------------------------

def test_twist_rolls_the_section_about_the_tangent_without_moving_the_axis():
    fr = np.linspace(0.0, 2.0, 9)
    sec = disc(1, center=(0.0, 0.0, 0.0))
    plain = hexmesh.sweep(sec, straight, fr, orientation="fixed", up=(1, 0, 0),
                          origin=(0.0, 0.0, 0.0))
    holed = hexmesh.sweep(sec, straight, fr, orientation="fixed", up=(1, 0, 0),
                          origin=(0.0, 0.0, 0.0), twist=np.pi / 2)
    # same z levels, same radii -- a pure roll
    assert np.allclose(np.sort(holed.points[:, 2]), np.sort(plain.points[:, 2]))
    assert np.allclose(np.sort(np.hypot(holed.points[:, 0], holed.points[:, 1])),
                       np.sort(np.hypot(plain.points[:, 0], plain.points[:, 1])),
                       atol=1e-13)
    assert not np.allclose(holed.points, plain.points)   # it really did roll
    # the far cap has turned by exactly a quarter turn
    n = sec.n_points
    a, b = plain.points[-n:], holed.points[-n:]
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(b, a @ rot.T, atol=1e-13)


# -- the quad rung -----------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 3])
def test_quad_rung_sweeps_a_segment_into_an_exact_flat_annulus(order):
    seg = linemesh.line((RB - RP, 0, 0), (RB + RP, 0, 0), np.linspace(0.0, 1.0, 5),
                        element_tag="fin", order=order)
    rib = quadmesh.sweep(seg, elbow, np.linspace(0.0, 1.0, 6),
                         origin=(RB, 0.0, 0.0), normal=(0, 0, 1),
                         orientation="fixed", up=(0, 1, 0),
                         first_tag="a", last_tag="b")
    nodes, _ = conform.conformal_quad(rib.points, rib.quads, rib.quad, rib.flip,
                                      rib.lines.interior, rib.interior, rib.order)
    x, y, z = nodes.T
    assert np.max(np.abs(y)) < 1e-14                      # stayed in the xz plane
    r = np.hypot(x, z)
    assert r.min() == pytest.approx(RB - RP, abs=1e-13)
    assert r.max() == pytest.approx(RB + RP, abs=1e-13)
    assert sorted(rib.edge_group_tags) == ["a", "b"]
    assert rib.element_tags.group_tags == ["fin"]


def test_quad_rung_needs_a_normal_for_a_collinear_profile():
    seg = linemesh.line((0, 0, 0), (1, 0, 0), np.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match="collapses onto the normal"):
        quadmesh.sweep(seg, elbow, np.linspace(0.0, 1.0, 4), normal=(1, 0, 0),
                       origin=(0.0, 0.0, 0.0))


# -- the placement primitives ------------------------------------------------------

def test_plane_frame_is_right_handed_and_spans_the_section_plane():
    P = disc(1, normal=(0, 1, 0)).points
    R, o = frames.plane_frame(P)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)
    assert abs(abs(float(R[:, 2] @ [0.0, 1.0, 0.0])) - 1.0) < 1e-12
    assert np.allclose(o, P.mean(axis=0))


def test_plane_frame_is_deterministic():
    P = disc(1).points
    assert np.array_equal(frames.plane_frame(P)[0], frames.plane_frame(P)[0])


def test_plane_frame_hint_flips_the_normal():
    P = disc(1, normal=(0, 0, 1)).points
    a, _ = frames.plane_frame(P, hint=(0, 0, 1))
    b, _ = frames.plane_frame(P, hint=(0, 0, -1))
    assert float(a[:, 2] @ [0.0, 0.0, 1.0]) > 0.0
    assert float(b[:, 2] @ [0.0, 0.0, 1.0]) < 0.0


def test_plane_frame_rejects_a_non_planar_section():
    P = disc(1).points.copy()
    P[3, 2] += 0.1
    with pytest.raises(ValueError, match="not planar"):
        frames.plane_frame(P)
    frames.plane_frame(P, normal=(0, 0, 1))      # ... unless the caller names the plane


def test_plane_frame_needs_three_points_to_fit_a_plane():
    with pytest.raises(ValueError, match="do not determine a plane"):
        frames.plane_frame(np.array([[0.0, 0, 0], [1.0, 0, 0]]))
