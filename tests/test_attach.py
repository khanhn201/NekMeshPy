"""``hexmesh.attach`` / ``quadmesh.attach`` -- joining two meshes along a **named**
interface, and the binning fix underneath every weld.

The distinction from ``merge``: ``merge`` is told nothing and infers every seam in an
assembly from coordinates at one global tolerance, so the tolerance that welds a slack
seam also reaches unrelated points elsewhere -- ``examples/chimera_full.py:444-448``
records exactly that ("loosening the tolerance for the whole assembly welded an
unrelated, closer-together pair by mistake").  ``attach`` is told *which* face group
meets which, so coordinates only ever settle the pairing **inside** those two groups.
That is what lets a seam be joined across a gap no global tolerance could take safely.

Orders 1 and 3, not 2: order 2 stores a single node per edge and face interior, so it
cannot catch a reversed edge or a permuted face frame.  Order 3 gives 2 and 4.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core import conform
from nekmeshpy.core.fields import uniform_spacing

ORDERS = [1, 3]


def _block(order, last="outlet"):
    ring = linemesh.circle(0.5, 8, element_tag="wall", order=order)
    sec = quadmesh.ogrid(ring, 2, uniform_spacing(2), wall_tag="wall")
    return hexmesh.extrude(sec, 2.0, 3, first_tag="inlet", last_tag=last)


def _stub(block, order, shift=0.0):
    """A stub grown off ``block``'s outlet, so its start face is that block's own."""
    cap = hexmesh.boundary_mesh(block, "outlet")
    s = hexmesh.extrude(cap, 0.5, 2, axis=(0.0, 0.0, 1.0),
                        first_tag="join", last_tag="outlet")
    return hexmesh.translate(s, (0.0, 0.0, shift)) if shift else s


def _rect(x0, x1, order, tags):
    return quadmesh.rectangle([[x0, 0, 0], [x1, 0, 0], [x1, 1, 0], [x0, 1, 0]],
                              3, 3, side_tags=tags, order=order)


# -- the binning fix ----------------------------------------------------------
def test_points_either_side_of_a_bin_boundary_still_weld():
    """The bug: coincidence was decided by ``round(x / tol)``, so two points arbitrarily
    closer than ``tol`` missed each other whenever they fell in adjacent cells.  A missed
    weld does not raise -- it leaves a seam open."""
    X = np.array([[0.5 - 5e-16, 0.0, 0.0], [0.5 + 5e-16, 0.0, 0.0]])
    assert not np.array_equal(np.round(X[0]), np.round(X[1]))     # adjacent cells
    lab = conform.coincident_clusters(X, 1.0)
    assert lab[0] == lab[1]


def test_the_lattice_reach_is_kept_as_well_as_the_radius():
    """Two points can share a cell and still be ``tol * sqrt(3)`` apart.  That reach is
    preserved deliberately: tolerances tuned by hand against the old behaviour were sized
    to it, and narrowing it reopens seams those callers rely on."""
    Y = np.array([[-0.49, -0.49, -0.49], [0.49, 0.49, 0.49]])
    assert float(np.linalg.norm(Y[0] - Y[1])) > 1.0               # further apart than tol
    assert np.array_equal(np.round(Y[0]), np.round(Y[1]))         # but one cell
    lab = conform.coincident_clusters(Y, 1.0)
    assert lab[0] == lab[1]


def test_a_pair_exactly_at_the_tolerance_is_not_coincident():
    """``tol`` is an exclusive bound on the radius half. ``cKDTree.query_pairs`` is
    inclusive, and taking it at face value fused a pair sitting at exactly 0.05 in
    ``examples/chimera_full.py`` -- a real spacing, not a seam -- collapsing the element
    between them. A lattice never welds two points exactly ``tol`` apart on an axis
    either, so the strict bound is also what keeps the two halves consistent."""
    at = conform.coincident_clusters(np.array([[0.0, 0, 0], [0.05, 0, 0]]), 0.05)
    assert at[0] != at[1]
    under = conform.coincident_clusters(np.array([[0.0, 0, 0], [0.0499, 0, 0]]), 0.05)
    assert under[0] == under[1]


def test_genuinely_distant_points_are_left_alone():
    lab = conform.coincident_clusters(np.array([[0.0, 0, 0], [5.0, 0, 0]]), 0.5)
    assert lab[0] != lab[1]


def test_coincidence_is_transitive_so_a_shared_hub_is_one_point():
    """A hub shared by four blocks arrives as several overlapping pairs, not one group."""
    H = np.zeros((4, 3))
    H[1, 0] = H[2, 1] = H[3, 2] = 1e-14
    assert len(set(conform.coincident_clusters(H, 1e-9).tolist())) == 1


# -- weld_pairs: stated, never measured ---------------------------------------
def test_weld_pairs_fuses_what_it_is_told_however_far_apart():
    pos = [np.array([[0.0, 0, 0], [1.0, 0, 0]]), np.array([[9.0, 0, 0], [2.0, 0, 0]])]
    pts, new_id = conform.weld_pairs(pos, np.array([[1, 2]]))
    assert pts.shape[0] == 3                       # 4 in, one pair fused
    assert new_id[1] == new_id[2]                  # welded despite being 8 apart
    assert np.array_equal(pts[new_id[1]], pos[0][1])   # the lower index's coordinate wins


def test_weld_pairs_rejects_an_out_of_range_pair():
    with pytest.raises(ValueError, match="outside the"):
        conform.weld_pairs([np.zeros((2, 3))], np.array([[0, 7]]))


# -- hexmesh.attach -----------------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
def test_attach_joins_into_one_conformal_body(order):
    a = _block(order)
    b = _stub(a, order)
    m = hexmesh.attach(a, b, "outlet", "join")
    rep = hexmesh.topology_report(m)
    assert rep.watertight and rep.conformal and rep.n_components == 1
    seam_pts = np.unique(a.quad_mesh.corners[hexmesh.tagged_faces(a, "outlet")]).size
    assert m.n_points == a.n_points + b.n_points - seam_pts


@pytest.mark.parametrize("order", ORDERS)
def test_the_seam_is_joined_across_a_gap_no_global_tolerance_could_take(order):
    """The property that distinguishes ``attach`` from ``merge``.  The two sides are a
    long way apart -- far beyond any tolerance it would be safe to hand ``merge`` for an
    assembly -- and ``attach`` takes no tolerance at all: the pairing is proved by
    bijectivity, so the separation is irrelevant."""
    a = _block(order)
    b = _stub(a, order, shift=0.4)                 # 80% of the stub's own length
    m = hexmesh.attach(a, b, "outlet", "join")
    rep = hexmesh.topology_report(m)
    assert rep.watertight and rep.conformal and rep.n_components == 1


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("own", ["a", "b"])
def test_the_owner_side_keeps_its_nodes_bit_for_bit(order, own):
    """``own`` decides whose geometry survives, and it is a copy, not an average: the
    shared-node re-scatter inside the stitch checks the two sides against
    ``conform.entity_tol``, far tighter than any pairing radius, and a merely-close seam
    fails it."""
    a = _block(order)
    b = _stub(a, order, shift=0.05)
    keeper, tag = (a, "outlet") if own == "a" else (b, "join")
    m = hexmesh.attach(a, b, "outlet", "join", own=own)
    seam = keeper.points[np.unique(keeper.quad_mesh.corners[
        hexmesh.tagged_faces(keeper, tag)])]
    assert float(cKDTree(m.points).query(seam)[0].max()) == 0.0


def test_attach_mutates_neither_input():
    a, b = _block(3), None
    b = _stub(a, 3, shift=0.05)
    pa, pb = a.points.copy(), b.points.copy()
    hexmesh.attach(a, b, "outlet", "join")
    assert np.array_equal(pa, a.points) and np.array_equal(pb, b.points)


def test_the_joined_faces_are_cleared_by_default():
    """They are interior now, and a *named* interior face makes the exporter write one
    boundary row from each side of it -- which callers used to strip by hand."""
    a = _block(2)
    m = hexmesh.attach(a, _stub(a, 2), "outlet", "join")
    assert "join" not in m.face_tags.group_tags
    assert hexmesh.tag_report(m).n_tagged_interior == 0


def test_attach_tag_names_the_interface_instead():
    a = _block(2)
    m = hexmesh.attach(a, _stub(a, 2), "outlet", "join", attach_tag="interface")
    assert "interface" in m.face_tags.group_tags
    assert hexmesh.tag_report(m).n_tagged_interior == len(
        hexmesh.tagged_faces(a, "outlet"))


def test_groups_of_different_size_are_refused():
    a = _block(1)
    with pytest.raises(ValueError, match="different face counts"):
        hexmesh.attach(a, _stub(a, 1), "wall", "join")


def test_a_buried_face_group_is_refused():
    """Joining onto a face that already has a hex on both sides would be non-manifold."""
    a = _block(1)
    m = hexmesh.attach(a, _stub(a, 1), "outlet", "join", attach_tag="interface")
    with pytest.raises(ValueError, match="hex on both sides"):
        hexmesh.attach(m, _block(1), "interface", "inlet")


def test_mismatched_order_is_refused():
    with pytest.raises(ValueError, match="same order"):
        hexmesh.attach(_block(1), _stub(_block(2), 2), "outlet", "join")


def test_own_must_name_a_side():
    a = _block(1)
    with pytest.raises(ValueError, match="own must be"):
        hexmesh.attach(a, _stub(a, 1), "outlet", "join", own="left")


def test_face_ids_may_be_given_instead_of_a_tag():
    """``tagged_faces`` is public precisely so a caller can take the list, reorder or
    subset it, and hand it straight back."""
    a = _block(2)
    b = _stub(a, 2)
    m = hexmesh.attach(a, b, hexmesh.tagged_faces(a, "outlet"),
                       hexmesh.tagged_faces(b, "join"))
    assert hexmesh.topology_report(m).n_components == 1


# -- the tag accessors --------------------------------------------------------
def test_tagged_faces_is_ascending_and_names_the_group():
    ids = hexmesh.tagged_faces(_block(1), "outlet")
    assert ids.size and np.array_equal(ids, np.sort(ids))


def test_an_unknown_tag_names_what_is_available():
    with pytest.raises(ValueError, match="no face carries the tag 'nope'"):
        hexmesh.tagged_faces(_block(1), "nope")
    with pytest.raises(ValueError, match="no edge carries the tag 'nope'"):
        quadmesh.tagged_edges(_rect(0, 1, 1, {"right": "seam"}), "nope")


# -- quadmesh.attach ----------------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
def test_quad_attach_joins_two_sections_along_an_edge_group(order):
    a = _rect(0, 1, order, {"right": "seam", "left": "west"})
    b = _rect(1, 2, order, {"left": "seam", "right": "east"})
    m = quadmesh.attach(a, b, "seam", "seam")
    assert m.n_quads == a.n_quads + b.n_quads
    assert m.n_points == a.n_points + b.n_points - 4        # the shared column
    assert quadmesh.area(m) == pytest.approx(2.0)
    assert "seam" not in m.edge_tags.group_tags


def test_quad_attach_tag_names_the_seam():
    a = _rect(0, 1, 2, {"right": "seam"})
    b = _rect(1, 2, 2, {"left": "seam"})
    assert "mid" in quadmesh.attach(a, b, "seam", "seam",
                                    attach_tag="mid").edge_tags.group_tags


def test_quad_attach_refuses_unequal_groups():
    """A differently refined side cannot be the same interface, and says so by count."""
    a = _rect(0, 1, 1, {"right": "seam"})
    b = quadmesh.rectangle([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], 3, 5,
                           side_tags={"left": "seam"})
    with pytest.raises(ValueError, match="different edge counts"):
        quadmesh.attach(a, b, "seam", "seam")


def test_quad_attach_refuses_a_group_that_is_not_the_same_curve():
    """Same edge count, but the two groups are different curves: the nearest-point map
    collapses several of one side's points onto one of the other's."""
    a = _rect(0, 1, 1, {"right": "seam", "bottom": "b"})
    b = _rect(1, 2, 1, {"left": "seam"})
    with pytest.raises(ValueError, match="not one-to-one"):
        quadmesh.attach(a, b, "b", "seam")


# -- merge is untouched -------------------------------------------------------
def test_merge_is_still_the_proximity_join():
    """``attach`` is an addition, not a replacement: ``merge`` still infers its seams."""
    a = _block(2)
    b = _stub(a, 2)
    joined = hexmesh.merge([hexmesh.retag_face(a, {"outlet": ""}),
                            hexmesh.retag_face(b, {"join": ""})])
    assert hexmesh.topology_report(joined).n_components == 1
