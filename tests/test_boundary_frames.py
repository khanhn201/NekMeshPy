"""``boundary_mesh`` at order > 1: the face-interior nodes must arrive in the frame of
the quad they are being read *into*.

A shared face's interior is stored once, in the frame of the canonical row the builder
chose for it.  A hex reads it back through its own ``face_orient`` codes; a *quad* being
extracted has no such code and must fit one, row against row
(:func:`~nekmeshpy.core.conform.quad_frame_code`).  Skipping that fit leaves the corners
and the shared edges correct and permutes only the interior, so nothing about the mesh
*looks* wrong -- which is why these tests assert on measured geometry rather than on
structure.

The bug this pins: the extracted wall of an order-3 O-grid pipe read 20.26 against an
exact 18.85, with 3 of its 24 faces folded.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core import conform
from nekmeshpy.core.interp import corner_indices

RADIAL = np.linspace(0.5, 1.0, 3)
N_SEG, RADIUS, HEIGHT = 8, 1.0, 3.0


def _pipe(order):
    """An O-grid pipe whose wall and caps both have closed forms: the wall is the
    ``N_SEG``-gon prism at order 1 and the true cylinder above it."""
    ring = linemesh.circle(RADIUS, N_SEG, element_tag="wall", order=order)
    section = quadmesh.ogrid(ring, 2, RADIAL, wall_tag="wall")
    block = hexmesh.extrude(section, length=HEIGHT, layers=3,
                            first_tag="inlet", last_tag="outlet")
    return ring, section, block


# -- the geometry the extraction has to reproduce ------------------------
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_the_extracted_wall_measures_the_cylinder_it_came_from(order):
    """The sweep is straight, so the wall *is* the ring swept ``HEIGHT`` -- an identity
    between two rungs that holds exactly at every order, where a comparison against the
    true circle would only hold to the interpolation error the mesh still carries."""
    ring, _, block = _pipe(order)
    wall = hexmesh.boundary_mesh(block, "wall")
    assert quadmesh.area(wall, high_order=True) == pytest.approx(
        linemesh.length(ring, high_order=True) * HEIGHT, rel=1e-12)


@pytest.mark.parametrize("order,rel", [(1, 1e-12), (2, 1e-3), (3, 1e-5), (4, 1e-7)])
def test_the_extracted_wall_converges_on_the_true_cylinder(order, rel):
    """And the same area approaches the analytic cylinder as the order rises -- the
    check that the identity above is pinned to the right geometry."""
    wall = hexmesh.boundary_mesh(_pipe(order)[2], "wall")
    exact = (N_SEG * 2 * np.sin(np.pi / N_SEG) if order == 1
             else 2 * np.pi * RADIUS) * HEIGHT
    assert quadmesh.area(wall, high_order=True) == pytest.approx(exact, rel=rel)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_wall_face_measures_the_same(order):
    """A uniform pipe's wall faces are congruent, so one folded face stands out where a
    total might average it away."""
    wall = hexmesh.boundary_mesh(_pipe(order)[2], "wall")
    a = quadmesh.element_areas(wall, high_order=True)
    assert a.max() - a.min() < 1e-12 * a.mean()


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_the_extracted_caps_measure_the_section_they_came_from(order):
    """The end caps are the swept section itself -- the same disc, so the same area, to
    the last digit.  These are hex faces 5 / 6, which the wall never exercises."""
    section, block = _pipe(order)[1:]
    want = quadmesh.area(section, high_order=True)
    for tag in ("inlet", "outlet"):
        cap = hexmesh.boundary_mesh(block, tag)
        assert quadmesh.area(cap, high_order=True) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_every_extracted_node_still_lies_on_the_wall(order):
    """Necessary but not sufficient -- a permuted face keeps its node *set*.  Asserted
    anyway, because it is what makes the area failure a frame bug rather than a lost
    node."""
    wall = hexmesh.boundary_mesh(_pipe(order)[2], "wall")
    B = quadmesh.element_blocks(wall)
    assert np.allclose(np.hypot(B[..., 0], B[..., 1]), RADIUS, atol=1e-13)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_the_extracted_wall_advances_monotonically_around_the_pipe(order):
    """The structural form of the same statement: within a face, the angle must run one
    way along the block's ``i`` axis.  A transposed or reflected interior breaks the
    monotonicity even though every node is on the cylinder."""
    wall = hexmesh.boundary_mesh(_pipe(order)[2], "wall")
    W = quadmesh.element_blocks(wall).reshape(-1, order + 1, order + 1, 3)
    d = np.diff(np.unwrap(np.arctan2(W[..., 1], W[..., 0]), axis=2), axis=2)
    assert (np.all(d > 0, axis=2) | np.all(d < 0, axis=2)).all()


@pytest.mark.parametrize("order", [1, 3])
def test_a_templated_extraction_agrees_with_the_plain_one(order):
    """The template path reads the same faces through a caller's own pattern, so it
    needs the same fit -- and the section that was swept *is* the cap's pattern."""
    section, block = _pipe(order)[1:]
    plain = hexmesh.boundary_mesh(block, "inlet")
    templated = hexmesh.boundary_mesh(block, "inlet", template=section)
    assert quadmesh.area(templated, high_order=True) == pytest.approx(
        quadmesh.area(plain, high_order=True), rel=1e-12)
    assert np.allclose(quadmesh.element_blocks(templated),
                       quadmesh.element_blocks(section))


# -- the fit itself ------------------------------------------------------
def test_a_row_against_itself_is_the_identity_fit():
    rows = np.array([[3, 7, 5, 1], [0, 1, 2, 3], [9, 4, 6, 8]], dtype=np.int64)
    assert conform.quad_frame_code(rows, rows).tolist() == [0, 0, 0]


def test_every_reordering_of_a_quad_is_some_d4_code():
    """A CCW row has exactly 8 re-windings, and the fit must name a distinct code for
    each -- that is what makes the turn invertible."""
    row = np.array([4, 9, 2, 7], dtype=np.int64)
    turns = [np.roll(row, k) for k in range(4)]
    turns += [t[::-1] for t in turns]
    codes = conform.quad_frame_code(np.stack(turns),
                                    np.broadcast_to(row, (8, 4)))
    assert sorted(codes.tolist()) == list(range(8))


def test_a_row_of_different_corners_is_rejected():
    with pytest.raises(ValueError, match="same quadrilateral"):
        conform.quad_frame_code(np.array([[0, 1, 2, 3]]), np.array([[0, 1, 2, 9]]))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_the_row_fit_reproduces_the_hex_read_where_the_frames_coincide(order):
    """Hex faces 1 / 2 / 5 / 6 carry the CCW frame themselves, so there the row fit must
    agree with :func:`gather_face_nodes` node for node -- the cross-check that the new
    path is the old one, not merely a different one."""
    block = _pipe(order)[2]
    local = conform.gather_face_nodes(block.face_nodes, block.hex, block.face_orient)
    canonical = np.asarray(block.quads.quads, dtype=np.int64)
    for f in range(6):
        if not np.array_equal(conform._FACE_CORNER_UV[f], conform._CCW_UV):
            continue
        poly = block.hexes[:, hexmesh.HexMesh.FACE_POINTS[f]]
        idx = conform.locate_rows(block.faces, poly, who="test", what="face")
        got = conform.face_nodes_in_frame(
            np.asarray(block.face_nodes, dtype=float)[idx], poly, canonical[idx])
        assert np.allclose(got, local[:, f])


@pytest.mark.parametrize("order", [1, 3])
def test_the_extraction_keeps_the_parent_corners_exactly(order):
    """The corners were never the broken part; pinned so a fix to the interior cannot
    quietly disturb them."""
    block = _pipe(order)[2]
    wall = hexmesh.boundary_mesh(block, "wall")
    B = quadmesh.element_blocks(wall)
    assert np.allclose(B[:, corner_indices(order, 2), :], wall.points[wall.quads])
