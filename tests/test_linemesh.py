"""Unit tests for the 1-D mesh sibling :class:`~nekmeshpy.LineMesh`: construction
(open vs closed as a topological property), the two tag systems (dense per-line
``element_tags`` + sparse tagged ``boundaries``/``boundary_tags``), the
``line`` / ``circle`` / ``rectangle`` / ``far_field_box`` / ``from_segments``
factories (every curve meshed exactly at the points given -- no resampling), and
the line -> quad -> hex tag ladder (``QuadMesh.extrude(LineMesh)`` and
``HexMesh.extrude`` carrying the tags up)."""

from collections import Counter

import numpy as np
import pytest

from nekmeshpy import HexMesh, LineMesh, QuadMesh
from nekmeshpy.model.fields import uniform_spacing

# -- construction ------------------------------------------------------------

def test_open_default_chain_connectivity():
    lm = LineMesh.open([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    assert lm.is_open and not lm.is_closed
    assert lm.n_points == 3
    assert lm.n_lines == 2                      # open chain: N-1 line elements
    assert lm.lines.tolist() == [[0, 1], [1, 2]]
    assert lm.element_group_tags == []          # untagged by default


def test_loop_default_chain_wraps():
    lm = LineMesh.loop([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    assert lm.is_closed and not lm.is_open
    assert lm.n_points == 3
    assert lm.n_lines == 3                       # closed loop: N line elements
    assert lm.lines.tolist() == [[0, 1], [1, 2], [2, 0]]


def test_rejects_2d_points():
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        LineMesh.open([(0, 0), (1, 0)])


def test_element_tags_length_must_match_lines():
    # open 3-point chain has 2 line elements
    with pytest.raises(ValueError, match="element_tags length .* must match lines"):
        LineMesh.open([(0, 0, 0), (1, 0, 0), (2, 0, 0)], element_tags=["a", "b", "c"])
    lm = LineMesh.open([(0, 0, 0), (1, 0, 0), (2, 0, 0)], element_tags=["a", "b"])
    assert lm.element_tags.tolist() == ["a", "b"]
    assert lm.element_group_tags == ["a", "b"]


def test_boundary_tags_length_must_match_boundaries():
    with pytest.raises(ValueError, match="boundary_tags length .* must match boundaries"):
        LineMesh.open([(0, 0, 0), (1, 0, 0)],
                      boundaries=[[0, 1]], boundary_tags=["a", "b"])


# -- topological queries -----------------------------------------------------

def test_boundary_points_are_open_ends():
    lm = LineMesh.open([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    assert lm.boundary_points().tolist() == [0, 2]     # degree-1 ends
    # a closed loop has no degree-1 ends
    assert LineMesh.loop([(0, 0, 0), (1, 0, 0), (1, 1, 0)]).boundary_points().size == 0


def test_tagged_boundary_points_via_boundaries():
    # side 1 -> local vertex 0, side 2 -> local vertex 1 of the referenced line
    lm = LineMesh.open([(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                       boundaries=[[0, 1], [1, 2]], boundary_tags=["start", "end"])
    assert lm.n_boundaries == 2
    assert lm.boundary_group_tags == ["end", "start"]


# -- length + exact-mesh factories -------------------------------------------

def test_open_length():
    lm = LineMesh.open([(0, 0, 0), (2, 0, 0)])
    assert np.isclose(lm.length, 2.0)


def test_line_grades_directly_no_resample():
    frac = np.array([0.0, 0.25, 0.5, 1.0])            # non-uniform: graded edge
    lm = LineMesh.line((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), frac)
    assert lm.is_open and lm.n_points == 4
    # the graded fractions land at the exact lerped points -- meshed as given
    assert np.allclose(lm.points[:, 0], [0.0, 1.0, 2.0, 4.0])


def test_line_element_tag_names_every_segment():
    lm = LineMesh.line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       uniform_spacing(3), element_tag="wall")
    assert lm.element_tags.tolist() == ["wall", "wall", "wall"]
    # untagged by default (empty tag falls through to no tags)
    assert LineMesh.line((0, 0, 0), (1, 0, 0), uniform_spacing(3)).element_group_tags == []


def test_circle_is_closed_loop_on_radius():
    lm = LineMesh.circle(2.0, 32)
    assert lm.is_closed
    assert lm.n_points == 32
    assert np.allclose(lm.points[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(lm.points[:, :2], axis=1), 2.0)


def test_circle_normal_places_loop_in_plane():
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    lm = LineMesh.circle(1.5, 16, center=(0.3, -0.2, 0.7), normal=n)
    c = np.array([0.3, -0.2, 0.7])
    assert np.max(np.abs((lm.points - c) @ n)) < 1e-12
    assert np.allclose(np.linalg.norm(lm.points - c, axis=1), 1.5)


def test_rectangle_corners_and_tags():
    # default +z plane: 4-corner loop CCW from lower-left, sides tagged per element
    lm = LineMesh.rectangle(4.0, 6.0, element_tags=["bottom", "right", "top", "left"])
    assert lm.is_closed and lm.n_points == 4 and lm.n_lines == 4
    assert np.allclose(lm.points, [[-2, -3, 0], [2, -3, 0], [2, 3, 0], [-2, 3, 0]])
    assert lm.element_tags.tolist() == ["bottom", "right", "top", "left"]


def test_rectangle_matches_manual_loop_farfield_box():
    # LineMesh.rectangle reproduces the far-field box built by hand in the examples
    hb = 6.0
    tags = ["bottom", "outlet", "top", "inlet"]
    manual = LineMesh.loop([(-hb, -hb, 0.0), (hb, -hb, 0.0),
                            (hb, hb, 0.0), (-hb, hb, 0.0)], element_tags=tags)
    rect = LineMesh.rectangle(2 * hb, 2 * hb, element_tags=tags)
    assert np.allclose(rect.points, manual.points)
    assert rect.element_tags.tolist() == manual.element_tags.tolist()


def test_rectangle_in_tilted_plane_is_planar():
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    c = np.array([0.2, -0.1, 0.4])
    lm = LineMesh.rectangle(3.0, 1.0, center=c, normal=n)
    assert np.max(np.abs((lm.points - c) @ n)) < 1e-12   # coplanar
    assert lm.is_closed and lm.n_points == 4


def test_far_field_box_is_index_aligned_and_on_box():
    # one box-perimeter point per inner-point ray -> index-aligned by construction
    inner = LineMesh.circle(0.5, 32)
    outer = LineMesh.far_field_box(inner, 2.0)
    assert outer.is_closed and outer.n_points == inner.n_points == 32
    # every outer point lies exactly on the square box |x| = 2 or |y| = 2
    on_box = np.isclose(np.abs(outer.points[:, :2]).max(axis=1), 2.0)
    assert on_box.all()
    # each outer point is radially outward of its paired inner point (same ray)
    assert (np.sum(outer.points[:, :2] * inner.points[:, :2], axis=1) > 0).all()


def test_far_field_box_carries_element_tags_by_sector():
    inner = LineMesh.circle(0.5, 32)
    outer = LineMesh.far_field_box(
        inner, 2.0, side_tags=["bottom", "right", "top", "left"])
    assert outer.n_lines == inner.n_lines == 32
    # 32 sectors split evenly across the four symmetric box sides
    assert Counter(outer.element_tags.tolist()) == {
        "bottom": 8, "right": 8, "top": 8, "left": 8}


def test_from_segments_chains_a_loop():
    # four segments of a unit square, given unordered, chain into a closed loop
    segs = np.array([[0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0],
                     [1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0]], float)
    lm = LineMesh.from_segments(segs)
    assert lm is not None and lm.is_closed
    # the four square corners are recovered (the walk repeats the start to close)
    assert len(np.unique(np.round(lm.points, 9), axis=0)) == 4
    assert LineMesh.from_segments(None) is None


# -- line -> quad tag ladder -------------------------------------------------

def _quad_edge_mid(qm, row):
    q, s = int(qm.boundaries[row, 0]), int(qm.boundaries[row, 1])
    return qm.points[qm.quads[q, QuadMesh.EDGE_POINTS[s - 1]]].mean(axis=0)


def test_extrude_line_to_quad_carries_element_and_boundary_tags():
    # an open line along +x, tagged per element, with tagged end points; sweep
    # along +y into a quad strip and check both tag chains land correctly.
    line = LineMesh.open([(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                         element_tags=["seg0", "seg1"],
                         boundaries=[[0, 1], [1, 2]], boundary_tags=["start", "end"])
    qm = QuadMesh.extrude(line, axis=(0.0, 1.0, 0.0), length=1.0,
                          layers=uniform_spacing(1),
                          first_tag="near", last_tag="far")
    # element tag rides onto the swept quads (one quad per line, nz=1)
    assert qm.n_quads == 2
    assert qm.element_tags.tolist() == ["seg0", "seg1"]

    # boundary-point tags land on the correct side-wall edges: "start" at x=0,
    # "end" at x=2; caps "near" at y=0, "far" at y=1.  Assert by edge geometry.
    tag_of = {}
    for r in range(qm.n_boundaries):
        tag_of.setdefault(str(qm.boundary_tags[r]), []).append(_quad_edge_mid(qm, r))
    assert np.isclose(np.array(tag_of["start"])[:, 0], 0.0).all()
    assert np.isclose(np.array(tag_of["end"])[:, 0], 2.0).all()
    assert np.isclose(np.array(tag_of["near"])[:, 1], 0.0).all()
    assert np.isclose(np.array(tag_of["far"])[:, 1], 1.0).all()


# -- quad -> hex element-tag carry-through ------------------------------------

def test_hex_extrude_carries_quad_element_tags():
    # two-quad section carrying element_tags -> extrude 2 layers -> 4 hexes, each
    # inheriting its column quad's element tag.
    section = QuadMesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]],
        [[0, 1, 2, 3], [1, 4, 5, 2]], element_tags=["A", "B"])
    block = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=1.0,
                            layers=uniform_spacing(2))
    assert block.n_hexes == 4
    assert Counter(block.element_tags.tolist()) == {"A": 2, "B": 2}
    assert block.element_group_tags == ["A", "B"]
