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
from nekmeshpy.hexmesh import Seam
from nekmeshpy.linemesh import Seam as PointSeam
from nekmeshpy.quadmesh import Seam as EdgeSeam

ORDERS = [1, 3]


def _join(a, b, tag_a, tag_b, **kw):
    """Two blocks through the n-ary API -- what a two-block join spells now."""
    return hexmesh.attach([a, b], [Seam(0, tag_a, 1, tag_b, **kw)])


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


# -- coincidence is a radius, and only a radius -------------------------------
def test_points_either_side_of_a_bin_boundary_still_weld():
    """The bug that started this: coincidence was decided by ``round(x / tol)``, so two
    points arbitrarily closer than ``tol`` missed each other whenever they fell in
    adjacent cells.  A missed weld does not raise -- it leaves a seam open."""
    X = np.array([[0.5 - 5e-16, 0.0, 0.0], [0.5 + 5e-16, 0.0, 0.0]])
    assert not np.array_equal(np.round(X[0]), np.round(X[1]))     # adjacent cells
    lab = conform.coincident_clusters(X, 1.0)
    assert lab[0] == lab[1]


def test_sharing_a_cell_is_not_enough_to_weld():
    """The inverse of the test this replaces, and the guard against the lattice coming
    back.  Two points can share a ``round(x / tol)`` cell and still be ``tol * sqrt(3)``
    apart; the lattice welded them anyway, 1.73x further than the caller asked for."""
    Y = np.array([[-0.49, -0.49, -0.49], [0.49, 0.49, 0.49]])
    assert float(np.linalg.norm(Y[0] - Y[1])) > 1.0               # further apart than tol
    assert np.array_equal(np.round(Y[0]), np.round(Y[1]))         # but one cell
    lab = conform.coincident_clusters(Y, 1.0)
    assert lab[0] != lab[1]


def test_welding_does_not_depend_on_where_the_model_sits():
    """Why the lattice had to go, not merely that it did.  Its cell edges are fixed in
    absolute space, so *which* pairs fused changed when the whole model was translated --
    and a shift of ``tol/2`` was enough to flip one.  A radius is translation-invariant,
    so the same two points weld the same way wherever they are put."""
    Y = np.array([[-0.49, -0.49, -0.49], [0.49, 0.49, 0.49]])       # one cell, d > tol
    Z = np.array([[0.5 - 5e-16, 0.0, 0.0], [0.5 + 5e-16, 0.0, 0.0]])  # two cells, d ~ 0
    for shift in (0.0, 0.5, 1.0 / 3.0, 13.7, -101.25):
        off = np.array([shift, shift, shift])
        far = conform.coincident_clusters(Y + off, 1.0)
        near = conform.coincident_clusters(Z + off, 1.0)
        assert far[0] != far[1], shift
        assert near[0] == near[1], shift


def test_a_pair_exactly_at_the_tolerance_is_not_coincident():
    """``tol`` is an exclusive bound on the radius half. ``cKDTree.query_pairs`` is
    inclusive, and taking it at face value fused a pair sitting at exactly 0.05 in
    ``examples/chimera_full.py`` -- a real spacing, not a seam -- collapsing the element
    between them."""
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
    m = _join(a, b, "outlet", "join")
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
    m = _join(a, b, "outlet", "join")
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
    m = _join(a, b, "outlet", "join", own=own)
    seam = keeper.points[np.unique(keeper.quad_mesh.corners[
        hexmesh.tagged_faces(keeper, tag)])]
    assert float(cKDTree(m.points).query(seam)[0].max()) == 0.0


def test_attach_mutates_neither_input():
    a, b = _block(3), None
    b = _stub(a, 3, shift=0.05)
    pa, pb = a.points.copy(), b.points.copy()
    _join(a, b, "outlet", "join")
    assert np.array_equal(pa, a.points) and np.array_equal(pb, b.points)


def test_the_joined_faces_are_cleared_by_default():
    """They are interior now, and a *named* interior face makes the exporter write one
    boundary row from each side of it -- which callers used to strip by hand."""
    a = _block(2)
    m = _join(a, _stub(a, 2), "outlet", "join")
    assert "join" not in m.face_tags.group_tags
    assert hexmesh.tag_report(m).n_tagged_interior == 0


def test_attach_tag_names_the_interface_instead():
    a = _block(2)
    m = _join(a, _stub(a, 2), "outlet", "join", attach_tag="interface")
    assert "interface" in m.face_tags.group_tags
    assert hexmesh.tag_report(m).n_tagged_interior == len(
        hexmesh.tagged_faces(a, "outlet"))


def test_groups_of_different_size_are_refused():
    a = _block(1)
    with pytest.raises(ValueError, match="different face counts"):
        _join(a, _stub(a, 1), "wall", "join")


def _collapse(block, tag, k):
    """``block`` with ``k`` of its ``tag`` group's points moved onto a single coordinate.

    Mutates in place -- ``points`` is a derived view over the ladder, so writing through
    it is how a caller repositions a mesh, and the callers here build a throwaway block."""
    sp = np.unique(block.quad_mesh.corners[hexmesh.tagged_faces(block, tag)])
    P = block.points.copy()
    P[sp[:k]] = P[sp[0]]
    block.points[:] = P
    return block


def test_coincident_points_within_a_group_are_refused():
    """Three of ``a``'s seam points sitting on one coordinate cannot pair one-for-one
    with three distinct points of ``b`` -- there is no bijection to find.  Nearest
    neighbour sends all three to the same point of ``b``, and the injectivity check is
    what turns that into an error instead of a silently collapsed seam."""
    a = _collapse(_block(1), "outlet", 3)
    b = _stub(_block(1), 1)
    with pytest.raises(ValueError, match="not one-to-one"):
        _join(a, b, "outlet", "join")


def test_the_message_counts_how_many_points_collapsed():
    """``k`` coincident points leave ``k-1`` unmatched, and the error says so -- the
    number is the first thing that tells you how degenerate the seam is."""
    a = _collapse(_block(1), "outlet", 4)
    b = _stub(_block(1), 1)
    with pytest.raises(ValueError, match=r"3 of a's \d+ seam points"):
        _join(a, b, "outlet", "join")


def test_matching_degeneracy_on_both_sides_is_still_refused():
    """The subtle one: if *both* seams have the same three points collapsed, the two
    sides genuinely correspond as point *sets* -- but a bijection still does not exist,
    so this must be refused rather than quietly welding three nodes into one."""
    a = _collapse(_block(1), "outlet", 3)
    b = _collapse(_stub(_block(1), 1), "join", 3)
    with pytest.raises(ValueError, match="not one-to-one"):
        _join(a, b, "outlet", "join")


def test_a_non_corresponding_group_is_refused_at_the_hex_rung():
    """Equal face counts and equal point counts, but the two groups are not the same
    surface: ``b``'s seam is collapsed to a point, so every one of ``a``'s maps to it."""
    a = _block(1)
    b = _stub(a, 1)
    sp = np.unique(b.quad_mesh.corners[hexmesh.tagged_faces(b, "join")])
    P = b.points.copy()
    P[sp] = P[sp[0]]
    b.points[:] = P
    with pytest.raises(ValueError, match="not one-to-one"):
        _join(a, b, "outlet", "join")


def test_a_failed_pairing_names_which_seam_failed():
    """With nine seams in one call, "the pairing is not one-to-one" on its own does not
    say which of them to look at.  The tag lookup already named the seam; the geometric
    checks have to as well, or the index has to be found by bisection."""
    a = _block(1)
    b = _stub(a, 1)
    sp = np.unique(b.quad_mesh.corners[hexmesh.tagged_faces(b, "join")])
    P = b.points.copy()
    P[sp] = P[sp[0]]
    b.points[:] = P
    c = _stub(_block(1), 1)
    with pytest.raises(ValueError, match=r"seams\[1\]: the pairing is not one-to-one"):
        hexmesh.attach([_block(1), c, a, b],
                       [Seam(0, "outlet", 1, "join"), Seam(2, "outlet", 3, "join")])


def test_a_buried_face_group_is_refused():
    """Joining onto a face that already has a hex on both sides would be non-manifold."""
    a = _block(1)
    m = _join(a, _stub(a, 1), "outlet", "join", attach_tag="interface")
    with pytest.raises(ValueError, match="hex on both sides"):
        _join(m, _block(1), "interface", "inlet")


def test_mismatched_order_is_refused():
    with pytest.raises(ValueError, match="same order"):
        _join(_block(1), _stub(_block(2), 2), "outlet", "join")


def test_own_must_name_a_side():
    a = _block(1)
    with pytest.raises(ValueError, match="own must be"):
        _join(a, _stub(a, 1), "outlet", "join", own="left")


def test_face_ids_may_be_given_instead_of_a_tag():
    """``tagged_faces`` is public precisely so a caller can take the list, reorder or
    subset it, and hand it straight back."""
    a = _block(2)
    b = _stub(a, 2)
    m = hexmesh.attach([a, b], [Seam(0, hexmesh.tagged_faces(a, "outlet"),
                                     1, hexmesh.tagged_faces(b, "join"))])
    assert hexmesh.topology_report(m).n_components == 1


# -- the n-ary form -----------------------------------------------------------
def _stack(n, order=2):
    ring = linemesh.circle(0.5, 8, element_tag="wall", order=order)
    sec = quadmesh.ogrid(ring, 2, uniform_spacing(2), wall_tag="wall")
    return [hexmesh.translate(
        hexmesh.extrude(sec, 1.0, 2, first_tag="lo", last_tag="hi"), (0, 0, float(k)))
        for k in range(n)]


def test_one_pass_equals_chaining_two_block_joins():
    """The n-ary form exists for speed, not for a different answer: welding a chain in
    one pass must give the same mesh as folding two-block joins across it."""
    bl = _stack(4)
    one = hexmesh.attach(bl, [Seam(k, "hi", k + 1, "lo") for k in range(3)])
    acc = bl[0]
    for k in range(1, 4):
        acc = _join(acc, bl[k], "hi", "lo")
    assert one.n_points == acc.n_points and one.n_hexes == acc.n_hexes
    assert np.array_equal(one.points, acc.points)
    assert np.array_equal(one.hexes, acc.hexes)
    assert np.array_equal(one.quad_mesh.quads, acc.quad_mesh.quads)


def test_a_block_may_carry_several_seams():
    """A middle block is welded on both faces in the same pass, so its ``own`` copies
    have to accumulate rather than the last one winning."""
    bl = _stack(3)
    m = hexmesh.attach(bl, [Seam(0, "hi", 1, "lo"), Seam(1, "hi", 2, "lo")])
    r = hexmesh.topology_report(m)
    assert r.watertight and r.conformal and r.n_components == 1
    assert m.n_hexes == sum(b.n_hexes for b in bl)


def test_seams_may_name_blocks_by_object_or_by_index():
    bl = _stack(3)
    by_obj = hexmesh.attach(bl, [Seam(bl[0], "hi", bl[1], "lo"),
                                 Seam(bl[1], "hi", bl[2], "lo")])
    by_idx = hexmesh.attach(bl, [Seam(0, "hi", 1, "lo"), Seam(1, "hi", 2, "lo")])
    assert np.array_equal(by_obj.points, by_idx.points)


def test_each_seam_carries_its_own_name_and_owner():
    bl = _stack(3)
    m = hexmesh.attach(bl, [Seam(0, "hi", 1, "lo", attach_tag="first"),
                            Seam(1, "hi", 2, "lo")])
    assert "first" in m.face_tags.group_tags
    # the unnamed seam is buried, so exactly one interface is tagged
    assert hexmesh.tag_report(m).n_tagged_interior == len(
        hexmesh.tagged_faces(bl[0], "hi"))


def test_a_seam_naming_a_mesh_outside_the_list_is_refused():
    bl = _stack(2)
    stray = _stack(1)[0]
    with pytest.raises(ValueError, match="not in the meshes list"):
        hexmesh.attach(bl, [Seam(stray, "hi", 1, "lo")])


def test_a_seam_naming_a_block_out_of_range_is_refused():
    with pytest.raises(ValueError, match="names block"):
        hexmesh.attach(_stack(2), [Seam(0, "hi", 7, "lo")])


def test_a_single_mesh_with_no_seams_is_returned_unchanged():
    only = _stack(1)[0]
    assert hexmesh.attach([only], []) is only


def test_errors_name_which_seam_failed():
    """With several seams the message has to say *which* one, or it is unactionable."""
    bl = _stack(3)
    with pytest.raises(ValueError, match=r"seams\[1\]"):
        hexmesh.attach(bl, [Seam(0, "hi", 1, "lo"), Seam(1, "hi", 2, "nope")])


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
    m = quadmesh.attach([a, b], [EdgeSeam(0, "seam", 1, "seam")])
    assert m.n_quads == a.n_quads + b.n_quads
    assert m.n_points == a.n_points + b.n_points - 4        # the shared column
    assert quadmesh.area(m) == pytest.approx(2.0)
    assert "seam" not in m.edge_tags.group_tags


def test_quad_attach_tag_names_the_seam():
    a = _rect(0, 1, 2, {"right": "seam"})
    b = _rect(1, 2, 2, {"left": "seam"})
    assert "mid" in quadmesh.attach([a, b], [EdgeSeam(0, "seam", 1, "seam",
                                             attach_tag="mid")]).edge_tags.group_tags


def test_quad_attach_refuses_unequal_groups():
    """A differently refined side cannot be the same interface, and says so by count."""
    a = _rect(0, 1, 1, {"right": "seam"})
    b = quadmesh.rectangle([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], 3, 5,
                           side_tags={"left": "seam"})
    with pytest.raises(ValueError, match="different edge counts"):
        quadmesh.attach([a, b], [EdgeSeam(0, "seam", 1, "seam")])


def test_quad_attach_refuses_a_group_that_is_not_the_same_curve():
    """Same edge count, but the two groups are different curves: the nearest-point map
    collapses several of one side's points onto one of the other's."""
    a = _rect(0, 1, 1, {"right": "seam", "bottom": "b"})
    b = _rect(1, 2, 1, {"left": "seam"})
    with pytest.raises(ValueError, match="not one-to-one"):
        quadmesh.attach([a, b], [EdgeSeam(0, "b", 1, "seam")])


# -- the stated-join fast path ------------------------------------------------
@pytest.mark.parametrize("order", [2, 3, 4])
def test_the_stated_node_path_equals_the_general_one(order):
    """``attach`` knows which entities fuse, so it renumbers the shared high-order node
    tables instead of gathering every element's nodes and scattering them back.  That
    shortcut must produce *the same mesh* as the general path ``merge`` uses -- it is an
    optimisation, not a different answer.

    The reference is built with the seam tags cleared, because the general path is
    ``merge``'s and rightly refuses two different names on one welded face."""
    from nekmeshpy.core import conform
    from nekmeshpy.hexmesh import assemble as asm

    sec = quadmesh.rectangle([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], 3, 2,
                             order=order)
    a = hexmesh.extrude(sec, 1.0, 4, first_tag="lo", last_tag="hi")
    b = hexmesh.translate(
        hexmesh.extrude(sec, 1.0, 3, first_tag="lo", last_tag="hi"), (0.0, 0.0, 1.0))
    fa, fb = hexmesh.tagged_faces(a, "hi"), hexmesh.tagged_faces(b, "lo")
    pairs, _ = asm._pair_seam(a, fa, b, fb)
    b2 = asm._adopt_seam(b, fb, pairs[:, 1], a, fa, pairs[:, 0])
    cat = np.stack([pairs[:, 0], a.n_points + pairs[:, 1]], axis=1)
    pts, pid = conform.weld_pairs([a.points, b2.points], cat)

    fast = asm._stitch([a, b2], pts, pid, who="t", seam_faces={0: fa, 1: fb})
    slow = asm._stitch([hexmesh.retag_face(a, {"hi": ""}),
                        hexmesh.retag_face(b2, {"lo": ""})], pts, pid, who="t")

    qf, qs = fast.quad_mesh, slow.quad_mesh
    for name, x, y in (("points", fast.points, slow.points),
                       ("hexes", fast.hexes, slow.hexes),
                       ("hex orient", fast.orient, slow.orient),
                       ("hex interior", fast.interior, slow.interior),
                       ("quads", qf.quads, qs.quads),
                       ("quad orient", qf.orient, qs.orient),
                       ("face interior", qf.interior, qs.interior),
                       ("edges", qf.line_mesh.lines, qs.line_mesh.lines),
                       ("edge interior", qf.line_mesh.interior, qs.line_mesh.interior)):
        assert np.array_equal(np.asarray(x), np.asarray(y)), name


def test_the_stated_path_still_catches_a_non_conforming_seam():
    """The shortcut drops the whole-mesh verification, so it has to keep the seam's:
    two sides that disagree on a shared face's interior must still raise."""
    from nekmeshpy.core import conform
    from nekmeshpy.hexmesh import assemble as asm

    a = _block(3)
    b = _stub(a, 3)
    fa, fb = hexmesh.tagged_faces(a, "outlet"), hexmesh.tagged_faces(b, "join")
    pairs, _ = asm._pair_seam(a, fa, b, fb)
    # corners agree, but b's shared face interiors are moved -- own= would have copied
    # them from a; here we deliberately skip that and expect the guard to fire
    bad = hexmesh.translate(b, (0.0, 0.0, 0.0))
    fi = np.array(bad.quad_mesh.interior, dtype=float, copy=True)
    fi[fb] += 0.05
    q = bad.quad_mesh
    bad = hexmesh.HexMesh(quadmesh.QuadMesh(q.line_mesh, q.quads, q.orient, fi,
                                            q.element_tags),
                          bad.hexes, bad.orient, bad.interior, bad.element_tags)
    cat = np.stack([pairs[:, 0], a.n_points + pairs[:, 1]], axis=1)
    pts, pid = conform.weld_pairs([a.points, bad.points], cat)
    with pytest.raises(ValueError, match="non-conforming high-order"):
        asm._stitch([a, bad], pts, pid, who="t", seam_faces={0: fa, 1: fb})


# -- linemesh.attach ----------------------------------------------------------
def _chain3():
    a = linemesh.loft(np.array([[0.0, 0, 0], [1, 0, 0]]), last_tag="j1")
    b = linemesh.loft(np.array([[1.0, 0, 0], [2, 0, 0]]), first_tag="j1", last_tag="j2")
    c = linemesh.loft(np.array([[2.0, 0, 0], [3, 0, 0]]), first_tag="j2")
    return a, b, c


def test_line_attach_matches_merge():
    """A slice at the line rung is a single point, so ``loft``'s ``first_tag`` /
    ``last_tag`` name the chain ends and that is what makes a seam addressable here."""
    a, b, c = _chain3()
    m = linemesh.merge([a, b, c])
    n = linemesh.attach([a, b, c], [PointSeam(0, "j1", 1, "j1"),
                                    PointSeam(1, "j2", 2, "j2")])
    assert np.array_equal(m.points, n.points)
    assert np.array_equal(m.lines, n.lines)


def test_line_attach_closes_a_loop_when_both_ends_are_stated():
    """Two shared-endpoint arcs into a ring -- the ``join_arcs`` idiom, which is one
    ``merge`` welding at both ends and so two stated seams here."""
    th = np.linspace(0.0, np.pi, 5)
    up = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    lo = np.stack([np.cos(th + np.pi), np.sin(th + np.pi), np.zeros_like(th)], axis=1)
    p = linemesh.loft(up, first_tag="A1", last_tag="A2")
    q = linemesh.loft(lo, first_tag="A2", last_tag="A1")
    ring = linemesh.attach([p, q], [PointSeam(0, "A1", 1, "A1"),
                                    PointSeam(0, "A2", 1, "A2")])
    assert len(linemesh.boundary_points(ring)) == 0          # a cycle has no free end
    assert np.array_equal(ring.points, linemesh.merge([p, q]).points)


def test_line_attach_clears_the_joined_point_names():
    a, b, c = _chain3()
    n = linemesh.attach([a, b, c], [PointSeam(0, "j1", 1, "j1"),
                                    PointSeam(1, "j2", 2, "j2")])
    assert n.point_tags.group_tags == []
    named = linemesh.attach([a, b, c], [PointSeam(0, "j1", 1, "j1", attach_tag="j1"),
                                        PointSeam(1, "j2", 2, "j2")])
    assert named.point_tags.group_tags == ["j1"]


def test_line_attach_refuses_an_unknown_tag_naming_the_seam():
    a, b, _ = _chain3()
    with pytest.raises(ValueError, match=r"seams\[0\].tag_a: no point carries"):
        linemesh.attach([a, b], [PointSeam(0, "nope", 1, "j1")])


def test_line_attach_refuses_groups_of_different_size():
    a, b, _ = _chain3()
    with pytest.raises(ValueError, match="different point counts"):
        linemesh.attach([a, b], [PointSeam(0, "j1", 1, np.array([0, 1]))])


# -- a block joined to itself ------------------------------------------------
def test_a_block_can_be_attached_to_itself():
    """Closing a block onto its own other end -- a torus, a periodic domain, or the
    line rung's ring.  This is the case that forced the seam names to be dropped
    *before* the renumber: both sides live in one block, so two tagged entities collapse
    onto one merged id and ``renumber`` refuses two names on one entity."""
    a = _block(1)
    m = hexmesh.attach([a], [Seam(0, "outlet", 0, "inlet")])
    rep = hexmesh.topology_report(m)
    assert rep.n_components == 1 and rep.watertight
    assert m.n_points < a.n_points                    # the two caps fused
    assert "inlet" not in m.face_tags.group_tags and "outlet" not in m.face_tags.group_tags


def test_a_self_attached_seam_can_still_be_named():
    a = _block(1)
    m = hexmesh.attach([a], [Seam(0, "outlet", 0, "inlet", attach_tag="periodic")])
    assert "periodic" in m.face_tags.group_tags


def test_quad_and_line_rungs_attach_to_themselves_too():
    sec = quadmesh.rectangle([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], 3, 2,
                             side_tags={"left": "L", "right": "R"})
    q = quadmesh.attach([sec], [EdgeSeam(0, "L", 0, "R")])
    assert q.n_points < sec.n_points

    th = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    pts = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    chain = linemesh.loft(np.vstack([pts, pts[:1]]), first_tag="r0", last_tag="r1")
    ring = linemesh.attach([chain], [PointSeam(0, "r0", 0, "r1")])
    assert len(linemesh.boundary_points(ring)) == 0
    # the same closure ``merge`` makes when handed the one open chain.  The reference is
    # built untagged on purpose: ``merge`` is right to refuse a chain whose two
    # coincident ends carry different names, which is the very thing that makes the ends
    # addressable here.
    plain = linemesh.loft(np.vstack([pts, pts[:1]]))
    assert np.array_equal(ring.points, linemesh.merge([plain]).points)


# -- merge is untouched -------------------------------------------------------
def test_merge_is_still_the_proximity_join():
    """``attach`` is an addition, not a replacement: ``merge`` still infers its seams."""
    a = _block(2)
    b = _stub(a, 2)
    joined = hexmesh.merge([hexmesh.retag_face(a, {"outlet": ""}),
                            hexmesh.retag_face(b, {"join": ""})])
    assert hexmesh.topology_report(joined).n_components == 1
