"""Unit tests for curves carried as their **surface parametrization** --
``model.surfaces``, ``linemesh.on_surface`` and ``quadmesh.tri_patch``.

The property everything here exists for: a curve derived from others by these
combinators stays *exactly on* the surface.  Interpolating two curves' **points**
instead cuts a chord, and a chord between two points of a cylinder dips inside it -- so
every intermediate station of a transition would sit slightly proud of the wall by an
amount no refinement removes.  Parameter space has no chord.

That is why the assertions below are on the **conformal node set** rather than on
corners: a corner-only check passes on a mesh that is high-order in storage and linear
in geometry, which is exactly the failure mode worth guarding.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.model import conform, surfaces

R = 1.3


def cyl(u):
    """The test surface: a cylinder about ``z``, parametrized by ``(phi, z)``."""
    u = np.asarray(u, dtype=float)
    return np.stack([R * np.cos(u[:, 0]), R * np.sin(u[:, 0]), u[:, 1]], axis=1)


def _line_nodes(m):
    return conform.conformal_line(m.points, m.lines, m.interior, m.order)[0]


def _quad_nodes(m):
    return conform.conformal_quad(m.points, m.quads, m.quad, m.flip,
                                  m.lines.interior, m.interior, m.order)[0]


def _off_cylinder(nodes):
    """Worst radial deviation of any node from the true cylinder."""
    p = np.asarray(nodes, dtype=float).reshape(-1, 3)
    return float(np.abs(np.hypot(p[:, 0], p[:, 1]) - R).max())


A = np.array([0.2, -0.5])
B = np.array([1.4, 0.9])
C = np.array([2.6, -0.1])


# -- combinators --------------------------------------------------------------
def test_ruled_is_straight_in_parameters_and_curved_in_space():
    c = surfaces.ruled(A, B, 4)
    assert c.fr.size == 5
    mid = c.g(np.array([0.5]))[0]
    assert np.allclose(mid, 0.5 * (A + B))          # straight in parameters
    lm = linemesh.on_surface(c, cyl, order=3)
    assert _off_cylinder(_line_nodes(lm)) < 1e-14   # curved in space, still on the wall


def test_blend_stays_on_the_surface_where_a_point_lerp_would_not():
    """The whole reason curves are carried as parametrizations."""
    # the two curves must differ in *phi*: offsetting only in z would put both nodes on
    # the same generator, whose chord happens to lie on the cylinder already
    a = surfaces.ruled(A, B, 4)
    b = surfaces.ruled(A + np.array([0.8, 0.3]), B + np.array([0.8, 0.3]), 4)
    for lam in (0.0, 0.25, 0.5, 1.0):
        mid = linemesh.on_surface(surfaces.blend(a, b, lam), cyl, order=3)
        assert _off_cylinder(_line_nodes(mid)) < 1e-14
    # and a point-space lerp of the same two really does leave the wall
    chord = 0.5 * (cyl(a.g(a.fr)) + cyl(b.g(b.fr)))
    assert _off_cylinder(chord) > 1e-2


def test_blend_endpoints_reproduce_their_inputs():
    a = surfaces.ruled(A, B, 4)
    b = surfaces.ruled(B, C, 4)
    x = np.linspace(0.0, 1.0, 7)
    assert np.allclose(surfaces.blend(a, b, 0.0).g(x), a.g(x))
    assert np.allclose(surfaces.blend(a, b, 1.0).g(x), b.g(x))


def test_reverse_walks_the_same_curve_backwards():
    c = surfaces.ruled(A, B, 4)
    r = surfaces.reverse(c)
    assert np.array_equal(r.fr, c.fr[::-1])
    fwd = linemesh.on_surface(c, cyl, order=2)
    back = linemesh.on_surface(r, cyl, order=2)
    assert np.allclose(fwd.points, back.points[::-1])


def test_shift_rebranches_a_periodic_coordinate_onto_the_same_points():
    """A whole turn in ``phi`` names the identical points on another branch."""
    c = surfaces.ruled(A, B, 4)
    turned = surfaces.shift(c, (2.0 * np.pi, 0.0))
    assert np.allclose(linemesh.on_surface(c, cyl, order=2).points,
                       linemesh.on_surface(turned, cyl, order=2).points)


def test_reparam_rewrites_a_segment_onto_another_curves_domain():
    """So a shaped edge and a plain one can be blended station by station."""
    shaped = surfaces.curve(lambda t: np.stack([t, 0.3 * np.sin(t)], axis=1),
                            np.linspace(0.5, 2.5, 5))
    plain = surfaces.reparam(shaped, (0.5, 0.0), (2.5, 0.0))
    assert np.array_equal(plain.fr, shaped.fr)
    assert np.allclose(plain.g(shaped.fr[:1])[0], [0.5, 0.0])
    assert np.allclose(plain.g(shaped.fr[-1:])[0], [2.5, 0.0])
    # the morph between them is on the surface at every station
    for lam in (0.0, 0.5, 1.0):
        lm = linemesh.on_surface(surfaces.blend(shaped, plain, lam), cyl, order=3)
        assert _off_cylinder(_line_nodes(lm)) < 1e-14


def test_node_reads_the_curves_own_sampling_not_the_parameter_midpoint():
    """On a graded curve those are different points, and a patch split at the wrong
    one stops meeting its neighbour."""
    graded = surfaces.curve(lambda t: np.stack([t, np.zeros_like(t)], axis=1),
                            np.array([0.0, 0.1, 0.9, 1.0]))
    assert np.allclose(surfaces.node(graded, 1), [0.1, 0.0])
    assert not np.allclose(surfaces.node(graded, 1), [0.5, 0.0])


def test_segment_reproduces_the_parents_grading():
    graded = surfaces.curve(lambda t: np.stack([t, np.zeros_like(t)], axis=1),
                            np.array([0.0, 0.1, 0.9, 1.0]))
    half = surfaces.segment(graded, 0, 2)
    assert np.allclose(half(np.array([0.0]))[0], [0.0, 0.0])
    assert np.allclose(half(np.array([0.5]))[0], [0.1, 0.0])    # the node, not 0.45
    assert np.allclose(half(np.array([1.0]))[0], [0.9, 0.0])


def test_segment_can_walk_backwards():
    c = surfaces.ruled(A, B, 4)
    back = surfaces.segment(c, 4, 2)
    assert np.allclose(back(np.array([0.0]))[0], c.g(c.fr[4:5])[0])
    assert np.allclose(back(np.array([1.0]))[0], c.g(c.fr[2:3])[0])


@pytest.mark.parametrize("bad, match", [
    (lambda: surfaces.curve(lambda t: t, [0.0]), "at least 2"),
    (lambda: surfaces.segment(surfaces.ruled(A, B, 4), 2, 2), "same node"),
])
def test_degenerate_inputs_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        bad()


# -- on_surface ---------------------------------------------------------------
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_on_surface_places_every_node_on_the_surface(order):
    """Corners *and* the private GLL interiors -- a ``loft`` of sampled points would
    straight-subdivide between them."""
    lm = linemesh.on_surface(surfaces.ruled(A, C, 3), cyl, order=order)
    assert lm.order == order
    assert _off_cylinder(_line_nodes(lm)) < 1e-14


def test_on_surface_carries_element_tags():
    lm = linemesh.on_surface(surfaces.ruled(A, B, 3), cyl, order=2,
                             element_tag="wall")
    assert lm.element_group_tags == ["wall"]


# -- tri_patch ----------------------------------------------------------------
def _tri(n=2):
    return (surfaces.ruled(A, B, 2 * n), surfaces.ruled(B, C, 2 * n),
            surfaces.ruled(C, A, 2 * n))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_tri_patch_lands_entirely_on_the_surface(order):
    patch = quadmesh.tri_patch(cyl, *_tri(), order=order, element_tag="wall")
    assert patch.order == order
    assert _off_cylinder(_quad_nodes(patch)) < 1e-14


def test_tri_patch_is_three_quad_patches_meeting_at_a_tip():
    n = 3
    patch = quadmesh.tri_patch(cyl, *_tri(n), order=2)
    assert patch.n_quads == 3 * n * n


def test_tri_patch_is_a_single_conforming_sheet():
    patch = quadmesh.tri_patch(cyl, *_tri(), order=2)
    assert len(quadmesh.boundary_edges(patch)) > 0        # it is an open sheet
    assert patch.n_points < 3 * (2 + 1) ** 2              # the three patches shared


def test_tri_patch_boundary_is_the_three_curves():
    """Every boundary node of the patch lies on one of the curves it was built from --
    which is what lets it cap a tetra whose other faces are built on those curves."""
    ab, bc, ca = _tri()
    patch = quadmesh.tri_patch(cyl, ab, bc, ca, order=2)
    rim = quadmesh.boundary_mesh(patch)
    on_curves = np.vstack([_line_nodes(linemesh.on_surface(c, cyl, order=2))
                           .reshape(-1, 3) for c in (ab, bc, ca)])
    from scipy.spatial import cKDTree
    assert float(cKDTree(on_curves).query(rim.points)[0].max()) < 1e-12


def test_tip_bias_moves_the_tip_toward_the_first_curve():
    ab, bc, ca = _tri()
    mids = [surfaces.node(c, 2) for c in (ab, bc, ca)]
    centroid = quadmesh.tri_patch_tip(*mids, tip_bias=1.0 / 3.0)
    pulled = quadmesh.tri_patch_tip(*mids, tip_bias=0.8)
    assert np.linalg.norm(pulled - mids[0]) < np.linalg.norm(centroid - mids[0])


def test_tri_patch_tip_is_what_the_patch_actually_uses():
    """A caller placing a solid apex against the patch must get the same point."""
    ab, bc, ca = _tri()
    mids = [surfaces.node(c, 2) for c in (ab, bc, ca)]
    tip = cyl(quadmesh.tri_patch_tip(*mids)[None, :])[0]
    patch = quadmesh.tri_patch(cyl, ab, bc, ca, order=2)
    assert float(np.linalg.norm(patch.points - tip, axis=1).min()) < 1e-12


@pytest.mark.parametrize("curves, match", [
    ((surfaces.ruled(A, B, 4), surfaces.ruled(B, C, 6), surfaces.ruled(C, A, 4)),
     "share a node count"),
    ((surfaces.ruled(A, B, 3), surfaces.ruled(B, C, 3), surfaces.ruled(C, A, 3)),
     "odd node count"),
])
def test_tri_patch_refuses_curves_it_cannot_split(curves, match):
    with pytest.raises(ValueError, match=match):
        quadmesh.tri_patch(cyl, *curves)


def test_tri_patch_caps_a_tetra():
    """Its real call site: the curved side of a curvilinear tetrahedron."""
    ab, bc, ca = _tri()                       # 5-node curves, so quadrant n = 2
    radial, center_scale = np.array([0.0, 0.6, 1.0]), 0.5
    # the seams are meshed exactly at the points given, never resampled, so the
    # sampling quadrant_ogrid wants has to be asked for by name
    fr = quadmesh.quadrant_seam_fractions(2, radial, center_scale)
    o = np.zeros(3)
    sa, sb, sc = (linemesh.line(o, cyl(np.asarray(p)[None, :])[0], fr, order=2)
                  for p in (A, B, C))
    faces = [quadmesh.quadrant_ogrid(linemesh.on_surface(c, cyl, order=2), s1, s2,
                                     radial, center_scale=center_scale)
             for c, s1, s2 in ((ab, sa, sb), (bc, sb, sc), (ca, sc, sa))]
    block = hexmesh.tetra([*faces, quadmesh.tri_patch(cyl, ab, bc, ca, order=2)])
    assert hexmesh.is_conforming(block)
    assert hexmesh.is_watertight(block)
