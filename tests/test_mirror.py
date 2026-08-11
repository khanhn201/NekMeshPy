"""``mirror`` -- the reflection that puts the winding back.

A reflection has determinant ``-1``, so the coordinate map alone inverts every element
that has a signed Jacobian.  ``mirror`` pairs it with a re-winding, and that pairing is
what the tests below pin: the geometry really is reflected (every node, not just the
corners), the quality is *unchanged* rather than negated, and a bare
``transform`` by the same matrix demonstrably is not.

The line rung is deliberately different -- a line element has no signed measure, so
there is nothing to re-wind and ``mirror`` there is only the coordinate map.  The
payoff case is the symmetric domain: mesh the half, mirror it, ``merge`` the two, and
get a conformal whole welded along the plane.
"""

import numpy as np
import pytest
from conftest import face_rows

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core import affine

RADIAL = np.linspace(0.5, 1.0, 3)
NORMAL = (1.0, 0.0, 0.0)


def _rungs(order):
    ring = linemesh.circle(1.0, 8, element_tag="wall", order=order)
    section = quadmesh.ogrid(ring, 2, RADIAL, wall_tag="wall")
    block = hexmesh.extrude(section, length=2.0, layers=2,
                            first_tag="inlet", last_tag="outlet")
    return ((ring, linemesh), (section, quadmesh), (block, hexmesh))


# -- the coordinate map reaches every node ------------------------------
def test_mirror_reflects_every_node_not_just_the_corners():
    """The high-order interior tables have to move too, so the mirrored node set is the
    original's reflected node set -- compared as multisets, since ids are re-wound."""
    for order in (1, 3):
        for mesh, pkg in _rungs(order):
            got = pkg.element_blocks(pkg.mirror(mesh, NORMAL))
            want = pkg.element_blocks(mesh) * np.array([-1.0, 1.0, 1.0])
            for axis in range(3):
                assert np.allclose(np.sort(got[..., axis].ravel()),
                                   np.sort(want[..., axis].ravel()))


def test_mirror_about_an_offset_plane_keeps_that_plane_fixed():
    block = _rungs(1)[2][0]
    got = hexmesh.mirror(block, NORMAL, point=(5.0, 0.0, 0.0))
    b = hexmesh.bounds(got)
    assert b.min[0] == pytest.approx(9.0, abs=1e-12)     # 10 - 1
    assert b.max[0] == pytest.approx(11.0, abs=1e-12)


# -- the winding really is put back --------------------------------------
def test_mirror_preserves_quality_where_a_bare_transform_negates_it():
    """The whole reason ``mirror`` exists, stated as an assertion -- at the hex rung,
    where the Jacobian carries the sign."""
    for order in (1, 3):
        block = _rungs(order)[2][0]
        good = hexmesh.mirror(block, NORMAL)
        M, off = affine.reflection(NORMAL)
        bad = hexmesh.transform(block, M, off)
        assert np.allclose(np.sort(hexmesh.scaled_jacobian(good)),
                           np.sort(hexmesh.scaled_jacobian(block)))
        assert (hexmesh.scaled_jacobian(bad) < 0).all()
        assert hexmesh.quality_summary(good).n_inverted == 0
        assert hexmesh.quality_summary(bad).n_inverted == block.n_hexes
        assert hexmesh.volume(bad) == pytest.approx(-hexmesh.volume(block), rel=1e-12)


def test_mirror_rewinds_the_quad_corner_order():
    """Structurally: ``(c0,c1,c2,c3)`` becomes ``(c0,c3,c2,c1)``.  Mirroring renumbers
    no point, so the two connectivities are directly comparable."""
    section = _rungs(3)[1][0]
    assert np.array_equal(quadmesh.mirror(section, NORMAL).quads,
                          section.quads[:, [0, 3, 2, 1]])


def test_the_quad_rung_mirror_halves_merge_into_a_consistent_section():
    """The section factories and ``extrude`` both take their orientation from the input
    rather than its winding, so a bare reflection is not *visibly* wrong one rung down
    -- the sign only surfaces at the hex rung.  What the quad ``mirror`` buys is a half
    that merges with its original into one consistently wound section."""
    corners = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float)
    half = quadmesh.rectangle(corners, 4, 2)
    full = quadmesh.merge([half, quadmesh.mirror(half, NORMAL)])
    assert full.n_quads == 2 * half.n_quads
    assert full.n_points < 2 * half.n_points               # the symmetry line welded
    assert quadmesh.quality_summary(full).n_inverted == 0
    assert quadmesh.area(full) == pytest.approx(2.0 * quadmesh.area(half), rel=1e-12)
    sj = quadmesh.scaled_jacobian(full)
    assert np.sign(sj).min() == np.sign(sj).max()          # one consistent winding


def test_a_mirrored_block_has_positive_volume():
    for order in (1, 3):
        block = _rungs(order)[2][0]
        assert hexmesh.volume(hexmesh.mirror(block, NORMAL), high_order=True) == (
            pytest.approx(hexmesh.volume(block, high_order=True), rel=1e-12))


def test_mirror_preserves_per_element_size():
    """Element for element, not merely in total -- a re-winding that scrambled the
    lattice would keep the sum and lose this."""
    for order in (1, 3):
        for mesh, pkg, per in ((_rungs(order)[0][0], linemesh, "element_lengths"),
                               (_rungs(order)[1][0], quadmesh, "element_areas"),
                               (_rungs(order)[2][0], hexmesh, "element_volumes")):
            a = getattr(pkg, per)(mesh, high_order=True)
            b = getattr(pkg, per)(pkg.mirror(mesh, NORMAL), high_order=True)
            assert np.allclose(a, b)


def test_mirror_twice_is_the_identity():
    """Geometry *and* connectivity: the second re-winding has to undo the first."""
    for order in (1, 3):
        for mesh, pkg in _rungs(order):
            back = pkg.mirror(pkg.mirror(mesh, NORMAL), NORMAL)
            assert np.allclose(back.points, mesh.points)
            assert np.allclose(pkg.element_blocks(back), pkg.element_blocks(mesh))


def test_a_mirrored_block_is_still_a_valid_mesh():
    for order in (1, 3):
        got = hexmesh.mirror(_rungs(order)[2][0], NORMAL)
        assert hexmesh.is_watertight(got) and hexmesh.is_conforming(got)


# -- tags ride the re-winding --------------------------------------------
def test_face_tags_follow_their_faces_through_the_rewind():
    """Faces 5 / 6 trade places under the re-winding, so a cap tag has to move with
    them -- pinned by where the tagged faces actually *are* in space."""
    block = _rungs(1)[2][0]
    got = hexmesh.mirror(block, NORMAL)
    assert sorted(got.face_group_tags) == sorted(block.face_group_tags)
    assert len(got.face_tags) == len(block.face_tags)
    for tag, z in (("inlet", 0.0), ("outlet", 2.0)):
        rows = np.array([(e, f) for e, f, t in face_rows(got) if t == tag])
        corners = got.hexes[rows[:, 0][:, None],
                            hexmesh.HexMesh.FACE_POINTS[rows[:, 1] - 1]]
        assert np.allclose(got.points[corners][..., 2], z)
    wall = np.array([(e, f) for e, f, t in face_rows(got) if t == "wall"])
    corners = got.hexes[wall[:, 0][:, None],
                        hexmesh.HexMesh.FACE_POINTS[wall[:, 1] - 1]]
    assert np.allclose(np.linalg.norm(got.points[corners][..., :2], axis=-1), 1.0)


def test_edge_tags_follow_their_edges_through_the_rewind():
    section = _rungs(1)[1][0]
    got = quadmesh.mirror(section, NORMAL)
    assert len(got.edge_tags) == len(section.edge_tags)
    rows = got.edge_tags.select(got.edge_tags.mask_for("wall"))
    corners = got.line_mesh.lines[rows.ids]
    assert np.allclose(np.linalg.norm(got.points[corners][..., :2], axis=-1), 1.0)


# -- the line rung is the exception --------------------------------------
def test_the_line_rung_mirror_is_only_the_coordinate_map():
    """No signed measure, so nothing to re-wind -- and the region fills take their
    orientation from the loop rather than its traversal, so a mirrored loop still
    fills into a correctly wound section."""
    for order in (1, 3):
        ring = _rungs(order)[0][0]
        got = linemesh.mirror(ring, NORMAL)
        assert np.array_equal(got.lines, ring.lines)
        assert np.allclose(got.points, ring.points * np.array([-1.0, 1.0, 1.0]))
        filled = quadmesh.ogrid(got, 2, RADIAL)
        assert quadmesh.quality_summary(filled).n_inverted == 0
        assert quadmesh.area(filled, high_order=True) == pytest.approx(
            quadmesh.area(quadmesh.ogrid(ring, 2, RADIAL), high_order=True), rel=1e-12)


# -- the payoff: a symmetric domain ---------------------------------------
def test_mirror_and_merge_rebuild_a_symmetric_domain():
    """Mesh the half, mirror it, weld the two -- the idiom the operation exists for."""
    corners = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float)
    half = hexmesh.extrude(quadmesh.rectangle(corners, 4, 2), length=1.0, layers=2,
                           first_tag="inlet", last_tag="outlet")
    full = hexmesh.merge([half, hexmesh.mirror(half, NORMAL)])

    assert full.n_hexes == 2 * half.n_hexes
    assert full.n_points < 2 * half.n_points              # the plane welded
    assert hexmesh.is_watertight(full) and hexmesh.is_conforming(full)
    assert hexmesh.quality_summary(full).n_inverted == 0
    assert hexmesh.volume(full) == pytest.approx(2.0 * hexmesh.volume(half), rel=1e-12)
    b = hexmesh.bounds(full)
    assert b.min == pytest.approx([-2.0, 0.0, 0.0], abs=1e-12)
    assert b.max == pytest.approx([2.0, 1.0, 1.0], abs=1e-12)


def test_reflection_is_its_own_inverse_and_rejects_a_degenerate_plane():
    M, off = affine.reflection((0.0, 0.0, 3.0), point=(0.0, 0.0, 1.0))
    assert np.linalg.det(M) == pytest.approx(-1.0, abs=1e-14)
    assert np.allclose(M @ M, np.eye(3))
    on_plane = affine.apply(np.array([[0.0, 0.0, 1.0]]), M, off)
    assert np.allclose(on_plane, [[0.0, 0.0, 1.0]])          # the plane is fixed
    assert np.allclose(affine.apply(np.array([[0.0, 0.0, 3.0]]), M, off),
                       [[0.0, 0.0, -1.0]])
    with pytest.raises(ValueError, match="non-zero"):
        affine.reflection((0.0, 0.0, 0.0))
