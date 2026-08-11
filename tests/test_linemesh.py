"""Unit tests for the 1-D mesh sibling :class:`~nekmeshpy.LineMesh`: construction
(open vs closed read off the ``lines`` connectivity -- stored nowhere), the two
tag systems (dense per-line
``element_tags`` + a sparse ``point_tags`` table), the
``loft`` / ``line`` / ``arc`` / ``circle`` / ``rectangle``
factories (every curve meshed exactly at the points given -- no resampling), and
the line -> quad -> hex tag ladder (``quadmesh.extrude(LineMesh)`` and
``HexMesh.extrude`` carrying the tags up)."""

from collections import Counter

import numpy as np
import pytest
from conftest import conformal

from nekmeshpy import (
    ElementTags,
    LineMesh,
    PointMesh,
    PointTags,
    QuadMesh,
    hexmesh,
    linemesh,
    quadmesh,
)
from nekmeshpy.core.fields import uniform_spacing

# -- construction ------------------------------------------------------------

def test_open_default_chain_connectivity():
    lm = linemesh.loft([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    assert lm.n_points == 3
    assert lm.n_lines == 2                      # open chain: N-1 line elements
    assert lm.lines.tolist() == [[0, 1], [1, 2]]
    # openness is the connectivity: the chain does not wrap, so both ends survive
    assert linemesh.boundary_points(lm).tolist() == [0, 2]
    assert lm.element_group_tags == []          # untagged by default


def test_loop_default_chain_wraps():
    lm = linemesh.loft([(0, 0, 0), (1, 0, 0), (1, 1, 0)], loop=True)
    assert lm.n_points == 3
    assert lm.n_lines == 3                       # closed loop: N line elements
    # closedness lives here and nowhere else: the wrap row [2, 0] is explicit
    assert lm.lines.tolist() == [[0, 1], [1, 2], [2, 0]]
    assert linemesh.boundary_points(lm).size == 0        # a cycle has no degree-1 end


def test_rejects_2d_points():
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        linemesh.loft([(0, 0), (1, 0)])
    # ``loop`` reports the same actionable error (it no longer relies on falling
    # through to a default-connectivity branch in the container -- there is none)
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        linemesh.loft([(0, 0), (1, 0)], loop=True)
    with pytest.raises(ValueError, match=r"must be \(N,3\)"):
        linemesh.loft(np.array([[0.0, 0], [1.0, 0]]), loop=True)


def test_lines_is_a_required_constructor_argument():
    """The container never invents connectivity: ``lines`` must be handed in, and
    ``loft`` (or a factory) is what authors it one rung up."""
    pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    with pytest.raises(TypeError, match="lines"):
        LineMesh(pts)
    explicit = LineMesh(pts, [[0, 1], [1, 2]])
    assert explicit.lines.tolist() == [[0, 1], [1, 2]]
    assert np.array_equal(explicit.lines, linemesh.loft(pts).lines)


def test_lines_must_index_points_that_exist():
    """Connectivity is checked against the point array it indexes, so a stray id is a
    ``ValueError`` here rather than an ``IndexError`` deep in whatever dereferences
    ``points[lines]`` later."""
    pts = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    with pytest.raises(ValueError, match="lines must index the 3 points"):
        LineMesh(pts, [[0, 1], [1, 7]])
    with pytest.raises(ValueError, match="lines must index the 3 points"):
        LineMesh(pts, [[-1, 0]])
    # a mesh with no lines has nothing to index, and must still build
    assert LineMesh(pts, np.zeros((0, 2), dtype=np.int64)).n_lines == 0


def test_element_tags_is_one_name_for_every_lofted_line():
    """A slice at this rung is a single point, so there is nothing to vary a tag
    over: one string names every line, and a per-line sequence is a TypeError."""
    lm = linemesh.loft([(0, 0, 0), (1, 0, 0), (2, 0, 0)], element_tags="a")
    assert lm.element_tags.dense(lm.n_lines).tolist() == ["a", "a"]
    assert lm.element_group_tags == ["a"]
    with pytest.raises(TypeError, match="single tag string"):
        linemesh.loft([(0, 0, 0), (1, 0, 0), (2, 0, 0)], element_tags=["a", "b"])


def test_boundary_table_columns_must_match_in_length():
    """The pairing is now structural: the table itself refuses a ragged build,
    so a LineMesh can no longer be given desynchronized rows and names."""
    with pytest.raises(ValueError, match="same length"):
        PointTags.from_pairs([[0, 1]], ["a", "b"])


# -- topological queries -----------------------------------------------------

def test_boundary_points_are_open_ends():
    lm = linemesh.loft([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    assert linemesh.boundary_points(lm).tolist() == [0, 2]     # degree-1 ends
    # a closed loop has no degree-1 ends
    assert linemesh.boundary_points(linemesh.loft([(0, 0, 0), (1, 0, 0), (1, 1, 0)], loop=True)).size == 0


def test_tagged_boundary_points_via_boundaries():
    # a point tag names the point itself, not one line's view of it
    lm = LineMesh(PointMesh([(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                            ElementTags([0, 2], ["start", "end"])),
                  [[0, 1], [1, 2]])
    assert lm.n_point_tags == 2
    assert lm.point_group_tags == ["end", "start"]


# -- exact-mesh factories -----------------------------------------------------

def test_line_grades_directly_no_resample():
    frac = np.array([0.0, 0.25, 0.5, 1.0])            # non-uniform: graded edge
    lm = linemesh.line((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), frac)
    assert lm.n_points == 4
    assert lm.lines.tolist() == [[0, 1], [1, 2], [2, 3]]      # open: no wrap row
    # the graded fractions land at the exact lerped points -- meshed as given
    assert np.allclose(lm.points[:, 0], [0.0, 1.0, 2.0, 4.0])


def test_line_element_tag_names_every_segment():
    lm = linemesh.line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       uniform_spacing(3), element_tag="wall")
    assert lm.element_tags.dense(lm.n_lines).tolist() == ["wall", "wall", "wall"]
    # untagged by default (empty tag falls through to no tags)
    assert linemesh.line((0, 0, 0), (1, 0, 0), uniform_spacing(3)).element_group_tags == []


def test_circle_is_closed_loop_on_radius():
    lm = linemesh.circle(2.0, 32)
    # the loop actually wraps: 32 points -> 32 line elements ending [31, 0]
    assert lm.n_lines == 32
    assert lm.lines[-1].tolist() == [31, 0]
    assert linemesh.boundary_points(lm).size == 0
    assert lm.n_points == 32
    assert np.allclose(lm.points[:, 2], 0.0)
    assert np.allclose(np.linalg.norm(lm.points[:, :2], axis=1), 2.0)


def test_circle_normal_places_loop_in_plane():
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    lm = linemesh.circle(1.5, 16, center=(0.3, -0.2, 0.7), normal=n)
    c = np.array([0.3, -0.2, 0.7])
    assert np.max(np.abs((lm.points - c) @ n)) < 1e-12
    assert np.allclose(np.linalg.norm(lm.points - c, axis=1), 1.5)


def test_arc_is_open_chain_on_radius():
    # n elements -> n+1 points, both ends free (this is circle's *open* sibling)
    lm = linemesh.arc(2.0, 8, start_theta=0.0, end_theta=np.pi / 2.0)
    assert lm.n_lines == 8 and lm.n_points == 9
    assert linemesh.boundary_points(lm).tolist() == [0, 8]      # open: both ends free
    assert np.allclose(np.linalg.norm(lm.points[:, :2], axis=1), 2.0)
    assert np.allclose(lm.points[:, 2], 0.0)
    # evenly spaced in angle, endpoints exactly at start/end
    th = np.arctan2(lm.points[:, 1], lm.points[:, 0])
    assert np.allclose(th, np.linspace(0.0, np.pi / 2.0, 9))
    assert np.allclose(lm.points[0], [2.0, 0.0, 0.0])
    assert np.allclose(lm.points[-1], [0.0, 2.0, 0.0])


def test_arc_runs_clockwise_when_end_theta_is_smaller():
    lm = linemesh.arc(1.0, 6, start_theta=np.pi, end_theta=0.0)
    assert lm.points[0][0] == pytest.approx(-1.0)
    assert lm.points[-1][0] == pytest.approx(1.0)
    assert np.all(lm.points[:, 1] >= -1e-15)          # the upper half, left to right


def test_arc_in_tilted_plane_is_planar_and_on_radius():
    n = np.array([1.0, -2.0, 0.5])
    n = n / np.linalg.norm(n)
    c = np.array([0.3, -0.2, 0.7])
    lm = linemesh.arc(1.5, 7, center=c, normal=n, start_theta=0.4, end_theta=2.1)
    assert np.max(np.abs((lm.points - c) @ n)) < 1e-14
    assert np.allclose(np.linalg.norm(lm.points - c, axis=1), 1.5)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_arc_high_order_nodes_lie_on_the_exact_circle(order):
    # every node the conformal walk yields -- corners and interior alike -- is on
    # the true arc, not on the chord (which would miss by the sagitta ~1e-2 here)
    lm = linemesh.arc(3.0, 5, start_theta=-0.7, end_theta=1.9, order=order)
    nodes, conn = conformal(lm)
    assert conn.shape == (5, order + 1)
    assert np.max(np.abs(np.linalg.norm(nodes, axis=1) - 3.0)) < 1e-13
    assert lm.interior.shape == (5, order - 1, 3)
    if order > 1:
        assert np.max(np.abs(np.linalg.norm(
            lm.interior.reshape(-1, 3), axis=1) - 3.0)) < 1e-13


def test_arc_element_tags_name_every_segment():
    lm = linemesh.arc(1.0, 4, element_tag="wall")
    assert lm.element_tags.dense(lm.n_lines).tolist() == ["wall"] * 4
    assert lm.element_group_tags == ["wall"]


def test_arc_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="n >= 1"):
        linemesh.arc(1.0, 0)
    with pytest.raises(ValueError, match="start_theta != end_theta"):
        linemesh.arc(1.0, 4, start_theta=0.5, end_theta=0.5)


def test_circle_output_is_not_perturbed_by_the_shared_arc_placement():
    # circle deliberately does not delegate to arc (its step is exactly 2*pi/n);
    # both must agree to round-off on the sub-arc they share
    n, order = 12, 3
    circ = linemesh.circle(1.0, n, order=order)
    a = linemesh.arc(1.0, 3, start_theta=0.0, end_theta=3 * 2.0 * np.pi / n,
                     order=order)
    assert np.allclose(circ.points[:4], a.points, atol=1e-15)
    assert np.allclose(circ.interior[:3], a.interior, atol=1e-15)


def test_rectangle_corners_and_per_side_tags():
    # n=4 -> the 4 corners, CCW from lower-left, one line element per side
    lm = linemesh.rectangle(4.0, 6.0, 4, side_tags={"bottom": "bottom", "right": "right", "top": "top", "left": "left"})
    assert lm.n_points == 4 and lm.n_lines == 4
    assert lm.lines.tolist() == [[0, 1], [1, 2], [2, 3], [3, 0]]   # wraps
    assert np.allclose(lm.points, [[-2, -3, 0], [2, -3, 0], [2, 3, 0], [-2, 3, 0]])
    assert lm.element_tags.dense(lm.n_lines).tolist() == ["bottom", "right", "top", "left"]


def test_rectangle_discretizes_per_side_on_box():
    # n=32 -> 8 evenly spaced points per side, every point on the box perimeter,
    # corners always landing on a point (so the loop is a true rectangle)
    hb = 6.0
    lm = linemesh.rectangle(2 * hb, 2 * hb, 32)
    assert lm.n_points == 32 and lm.n_lines == 32
    assert lm.lines[-1].tolist() == [31, 0] and linemesh.boundary_points(lm).size == 0
    assert np.isclose(np.abs(lm.points[:, :2]).max(axis=1), hb).all()
    for cxy in ([-hb, -hb], [hb, -hb], [hb, hb], [-hb, hb]):
        assert np.isclose(lm.points[:, :2] - cxy, 0.0).all(axis=1).any()


def test_rectangle_side_tag_counts_are_even():
    lm = linemesh.rectangle(2.0, 2.0, 32,
                            side_tags={"bottom": "bottom", "right": "right", "top": "top", "left": "left"})
    assert Counter(lm.element_tags.dense(lm.n_lines).tolist()) == {
        "bottom": 8, "right": 8, "top": 8, "left": 8}


def test_rectangle_requires_multiple_of_four():
    with pytest.raises(ValueError, match="multiple of 4"):
        linemesh.rectangle(2.0, 2.0, 6)


def test_rectangle_in_tilted_plane_is_planar():
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    c = np.array([0.2, -0.1, 0.4])
    lm = linemesh.rectangle(3.0, 1.0, 8, center=c, normal=n)
    assert np.max(np.abs((lm.points - c) @ n)) < 1e-12   # coplanar
    assert lm.n_points == 8
    assert lm.lines[-1].tolist() == [7, 0] and linemesh.boundary_points(lm).size == 0


# -- merge (weld coincident end points) --------------------------------------

def test_merge_two_arcs_close_into_a_loop():
    # two half-circle arcs sharing both endpoints A1=(1,0,0), A2=(-1,0,0):
    # upper (z>=0) and lower (z<=0). Reverse the second so the traversal runs
    # A1->A2 down the upper arc then A2->A1 back up the lower.
    tu = np.linspace(0.0, np.pi, 5)                     # A1 -> A2, z >= 0
    upper = np.column_stack([np.cos(tu), np.zeros(5), np.sin(tu)])
    tl = np.linspace(0.0, np.pi, 5)                     # A1 -> A2, z <= 0
    lower = np.column_stack([np.cos(tl), np.zeros(5), -np.sin(tl)])
    ring = linemesh.merge([linemesh.loft(upper),
                           linemesh.loft(lower[::-1])])
    # welded at A1 and A2 -> one closed loop of 8 unique points (2*(5-1))
    assert linemesh.boundary_points(ring).size == 0     # no degree-1 end survived
    assert ring.n_points == 8
    assert ring.n_lines == 8
    # index 0 stays A1, index 4 (M//2) is A2 -- the split points spined_ogrid needs
    assert np.allclose(ring.points[0], [1.0, 0.0, 0.0])
    assert np.allclose(ring.points[4], [-1.0, 0.0, 0.0])
    # connectivity is a single wrapping cycle
    assert np.array_equal(ring.lines,
                          np.array([[i, (i + 1) % 8] for i in range(8)]))


def test_merge_open_chains_stay_open_and_carry_tags():
    # two collinear open chains meeting at (1,0,0); the shared point welds but the
    # far ends stay degree-1, so the result is still open.
    a = LineMesh(PointMesh([(0, 0, 0), (1, 0, 0)], ElementTags([0], ["start"])),
                 [[0, 1]], element_tags=ElementTags.uniform(1, "a"))
    b = linemesh.loft([(1, 0, 0), (2, 0, 0)], element_tags="b")
    m = linemesh.merge([a, b])
    assert linemesh.boundary_points(m).tolist() == [0, 2]       # the two far ends survive
    assert m.n_points == 3                              # the shared point welded
    assert m.element_tags.dense(m.n_lines).tolist() == ["a", "b"]        # dense tags concatenate
    assert m.point_group_tags == ["start"]           # sparse BC markers carried


def test_merge_does_not_weld_interior_points():
    # an interior point coincident with another chain's interior is NOT welded
    # (only degree-1 ends weld), mirroring QuadMesh/HexMesh.merge.
    a = linemesh.loft([(0, 0, 0), (1, 0, 0), (2, 0, 0)])   # (1,0,0) is interior
    b = linemesh.loft([(1, 0, 0), (1, 1, 0)])              # end at (1,0,0)
    m = linemesh.merge([a, b])
    # b's end welds to... nothing on a's interior; only ends weld -> 5 points
    assert m.n_points == 5


# -- line -> quad tag ladder -------------------------------------------------

def _quad_edge_mid(qm, row):
    q, s = int(qm.edge_tags.elements[row]), int(qm.edge_tags.sides[row])
    return qm.points[qm.quads[q, QuadMesh.EDGE_POINTS[s - 1]]].mean(axis=0)


def test_extrude_line_to_quad_carries_element_and_edge_tags():
    # an open line along +x, tagged per element, with tagged end points; sweep
    # along +y into a quad strip and check both tag chains land correctly.
    line = LineMesh(PointMesh([(0, 0, 0), (1, 0, 0), (2, 0, 0)],
                              ElementTags([0, 2], ["start", "end"])),
                    [[0, 1], [1, 2]],
                    element_tags=ElementTags.from_dense(["seg0", "seg1"]))
    # the swept quads are new elements, so the profile's tags reach them only by
    # being asked for -- one ElementTags over the profile's lines
    qm = quadmesh.extrude(line, axis=(0.0, 1.0, 0.0), length=1.0,
                          layers=uniform_spacing(1),
                          element_tags=line.element_tags,
                          first_tag="near", last_tag="far")
    # element tag rides onto the swept quads (one quad per line, nz=1)
    assert qm.n_quads == 2
    assert qm.element_tags.dense(qm.n_quads).tolist() == ["seg0", "seg1"]

    # boundary-point tags land on the correct side-wall edges: "start" at x=0,
    # "end" at x=2; caps "near" at y=0, "far" at y=1.  Assert by edge geometry.
    tag_of = {}
    for r in range(qm.n_edge_tags):
        tag_of.setdefault(str(qm.edge_tags.tags[r]), []).append(_quad_edge_mid(qm, r))
    assert np.isclose(np.array(tag_of["start"])[:, 0], 0.0).all()
    assert np.isclose(np.array(tag_of["end"])[:, 0], 2.0).all()
    assert np.isclose(np.array(tag_of["near"])[:, 1], 0.0).all()
    assert np.isclose(np.array(tag_of["far"])[:, 1], 1.0).all()


# -- quad -> hex element-tag carry-through ------------------------------------

def test_hex_extrude_carries_quad_element_tags():
    # two-quad section carrying element_tags -> extrude 2 layers -> 4 hexes, each
    # inheriting its column quad's element tag.
    section = QuadMesh.from_corners(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]],
        [[0, 1, 2, 3], [1, 4, 5, 2]],
        element_tags=ElementTags.from_dense(["A", "B"]))
    block = hexmesh.extrude(section, axis=(0.0, 0.0, 1.0), length=1.0,
                            layers=uniform_spacing(2),
                            element_tags=section.element_tags)
    assert block.n_hexes == 4
    assert Counter(block.element_tags.dense(block.n_hexes).tolist()) == {"A": 2, "B": 2}
    assert block.element_group_tags == ["A", "B"]


# -- sweep_fractions (HELPERS: a plain array, not a mesh) ---------------------

def test_sweep_fractions_lands_a_node_on_every_junction():
    # a 10-long path with junctions at 3 and 7, elements ~0.5 long
    fr = linemesh.sweep_fractions([3.0, 7.0], 10.0, 0.5)
    assert fr[0] == 0.0 and fr[-1] == 1.0
    assert (np.diff(fr) > 0).all()                 # strictly ascending
    # the junctions are *in* the output bit-for-bit, not merely approached
    assert 0.3 in fr.tolist() and 0.7 in fr.tolist()
    # each piece subdivided on its own: 3/0.5=6, 4/0.5=8, 3/0.5=6 -> 20 elements
    assert fr.size == 21


def test_sweep_fractions_never_drops_a_piece_below_one_element():
    # a piece far shorter than target still gets its own single element, so the
    # junction survives rather than being swallowed
    fr = linemesh.sweep_fractions([0.01], 1.0, 0.5)
    assert 0.01 in fr.tolist()
    assert fr.tolist() == [0.0, 0.01, 0.505, 1.0]


def test_sweep_fractions_rejects_breaks_on_the_endpoints():
    # 0 and total_length are stations unconditionally; admitting them as breaks
    # would duplicate a station and break the ascending contract
    with pytest.raises(ValueError):
        linemesh.sweep_fractions([0.0, 5.0], 10.0, 1.0)
    with pytest.raises(ValueError):
        linemesh.sweep_fractions([5.0, 10.0], 10.0, 1.0)


def test_sweep_fractions_with_no_breaks_is_a_plain_subdivision():
    assert np.allclose(linemesh.sweep_fractions([], 2.0, 0.5),
                       np.linspace(0.0, 1.0, 5))


# -- repr ---------------------------------------------------------------------

def test_repr_reports_counts_order_and_tag_groups():
    lm = linemesh.circle(1.0, 8, element_tag="wall")
    assert repr(lm) == ("<LineMesh 8 points, 8 lines, order 1, "
                        "element_tags={wall}, point_tags={}>")


def test_repr_never_raises_on_a_half_built_container():
    # a repr that can throw is worse than useless in a debugger
    assert repr(LineMesh.__new__(LineMesh)) == "<LineMesh (unprintable)>"
