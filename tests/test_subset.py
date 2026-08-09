"""``select`` / ``remove`` / ``components`` -- the operations that take a mesh apart.

They are the inverse of ``merge`` and sit beside it in ``assemble`` for the same
reason: they **manufacture a global index space**.  So the properties to pin are the
bookkeeping ones -- every kept element keeps its geometry, its order and its tags, the
entities under it come across whole, and nothing that was dropped leaves a dangling id
behind.  The pair is pinned as a partition (``select`` and ``remove`` of one argument
reconstruct the whole), and ``components`` against a ``merge`` of known pieces.
"""

import numpy as np
import pytest

from nekmeshpy import ElementTags, hexmesh, linemesh, quadmesh
from nekmeshpy.core import conform

RADIAL = np.linspace(0.5, 1.0, 3)


def _rungs(order):
    """One mesh per rung with its package and the name of its element count."""
    ring = linemesh.circle(1.0, 8, element_tag="wall", order=order)
    section = quadmesh.ogrid(ring, 2, RADIAL, wall_tag="wall")
    block = hexmesh.extrude(section, length=2.0, layers=2,
                            first_tag="inlet", last_tag="outlet")
    return ((ring, linemesh, "n_lines"), (section, quadmesh, "n_quads"),
            (block, hexmesh, "n_hexes"))


def _n(mesh, name):
    return getattr(mesh, name)


# -- select and remove partition the mesh -------------------------------
def test_select_and_remove_partition_every_rung():
    for order in (1, 3):
        for mesh, pkg, count in _rungs(order):
            n = _n(mesh, count)
            ids = np.arange(0, n, 2)
            a, b = pkg.select(mesh, ids), pkg.remove(mesh, ids)
            assert _n(a, count) == ids.shape[0]
            assert _n(a, count) + _n(b, count) == n


def test_a_subset_keeps_the_order_it_was_cut_from():
    """The private interior tables are indexed, never rebuilt, so the order rides
    through -- including on an empty selection, where only the node block's shape is
    left to carry it."""
    for order in (1, 4):
        for mesh, pkg, count in _rungs(order):
            assert pkg.select(mesh, [0]).order == order
            assert pkg.remove(mesh, np.arange(_n(mesh, count))).order == order


def test_selected_elements_keep_their_geometry_node_for_node():
    """Not just the corners: the whole node block of a kept element must come across
    unchanged, which is what proves the shared edge / face tables were carried and not
    re-derived."""
    for order in (1, 3):
        for mesh, pkg, _count in _rungs(order):
            ids = np.array([1, 3, 4])
            blocks = pkg.element_blocks(mesh)[ids]
            assert np.allclose(pkg.element_blocks(pkg.select(mesh, ids)), blocks)


def test_select_preserves_relative_order():
    block = _rungs(1)[2][0]
    ids = np.array([7, 2, 5])          # given out of order
    got = hexmesh.element_blocks(hexmesh.select(block, ids))
    want = hexmesh.element_blocks(block)[np.sort(ids)]
    assert np.allclose(got, want)


def test_a_subset_drops_the_points_nothing_kept_touches():
    """A subset is a mesh in its own right, not a view with holes: the surviving
    numbering is dense and every point is referenced."""
    for mesh, pkg, count in _rungs(2):
        part = pkg.select(mesh, [0, 1])
        conn = {"n_lines": "lines", "n_quads": "quads", "n_hexes": "hexes"}[count]
        used = np.unique(getattr(part, conn))
        assert used.shape[0] == part.n_points
        assert used[0] == 0 and used[-1] == part.n_points - 1


# -- tags ---------------------------------------------------------------
def test_select_by_tag_takes_exactly_the_tagged_elements():
    section = quadmesh.ogrid(linemesh.circle(1.0, 8), 2, RADIAL)
    tagged = quadmesh.QuadMesh(
        section.lines, section.quad, section.flip, section.interior,
        section.edge_tags,
        ElementTags(np.arange(0, section.n_quads, 3),
                    np.full(len(np.arange(0, section.n_quads, 3)), "core")))
    got = quadmesh.select(tagged, "core")
    assert got.n_quads == len(np.arange(0, section.n_quads, 3))
    assert got.element_group_tags == ["core"]
    assert len(got.element_tags) == got.n_quads


def test_an_unknown_tag_is_an_error_not_an_empty_mesh():
    """A silent empty selection is almost always a typo, so the tag form insists the
    name is in the vocabulary."""
    block = _rungs(1)[2][0]
    with pytest.raises(ValueError, match="no element carries the tag"):
        hexmesh.select(block, "nosuchtag")


def test_side_tags_follow_their_elements_and_drop_with_them():
    """A ``face_tags`` row names an element, so a subset must renumber the rows it
    keeps and shed the rest -- never leave a row pointing at an element that is gone."""
    block = _rungs(1)[2][0]
    ids = np.arange(block.n_hexes // 2)
    part = hexmesh.select(block, ids)
    assert len(part.face_tags) < len(block.face_tags)
    assert part.face_tags.elements.max() < part.n_hexes
    # every kept row is one the parent had, on the same face of the same element
    parent = block.face_tags.as_dict()
    for e, s, t in part.face_tags:
        assert parent[(int(ids[e]), s)] == t


def test_removal_exposes_untagged_boundary():
    """The faces a removal opens up are new topological boundary and carry no tag --
    the documented behaviour, pinned so it cannot change silently.  (The block stays
    *watertight*: a cavity is still a closed surface.  What grows is the boundary.)"""
    block = _rungs(1)[2][0]
    part = hexmesh.remove(block, [0])
    assert len(part.face_tags) < len(block.face_tags)
    grew = (hexmesh.boundary_faces(part).shape[0]
            - hexmesh.boundary_faces(block).shape[0])
    assert grew > 0
    # the new rows are boundary the tag table says nothing about
    named = set(map(tuple, part.face_tags.rows.tolist()))
    assert any(tuple(r) not in named for r in hexmesh.boundary_faces(part).tolist())


# -- the whole mesh, and nothing -----------------------------------------
def test_selecting_everything_reproduces_the_mesh():
    for order in (1, 3):
        for mesh, pkg, count in _rungs(order):
            same = pkg.select(mesh, np.arange(_n(mesh, count)))
            assert _n(same, count) == _n(mesh, count)
            assert np.allclose(pkg.element_blocks(same), pkg.element_blocks(mesh))


def test_selecting_nothing_is_an_empty_mesh_not_an_error():
    """A mask or id list that names nothing is explicit, so it is allowed -- unlike a
    tag that is not in the vocabulary."""
    for mesh, pkg, count in _rungs(1):
        empty = pkg.select(mesh, [])
        assert _n(empty, count) == 0 and empty.n_points == 0


def test_a_boolean_mask_must_cover_the_mesh():
    block = _rungs(1)[2][0]
    with pytest.raises(ValueError, match="must cover all"):
        hexmesh.select(block, np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="element ids must lie"):
        hexmesh.select(block, [block.n_hexes])


# -- components, and the round trip through merge ------------------------
def test_components_finds_the_bodies_a_merge_put_together():
    for order in (1, 3):
        for mesh, pkg, count in _rungs(order):
            far = pkg.translate(mesh, (50.0, 0.0, 0.0))
            two = pkg.merge([mesh, far])
            parts = pkg.components(two)
            assert len(parts) == 2
            assert [_n(p, count) for p in parts] == [_n(mesh, count)] * 2
            assert np.allclose(pkg.element_blocks(parts[0]),
                               pkg.element_blocks(mesh))


def test_a_connected_mesh_is_one_component():
    for mesh, pkg, count in _rungs(1):
        parts = pkg.components(mesh)
        assert len(parts) == 1 and _n(parts[0], count) == _n(mesh, count)


def test_merge_of_components_reconstructs_the_mesh():
    """``merge(components(m))`` is the round trip that ties the inverse pair together."""
    block = _rungs(3)[2][0]
    two = hexmesh.merge([block, hexmesh.translate(block, (50.0, 0.0, 0.0))])
    back = hexmesh.merge(hexmesh.components(two))
    assert back.n_hexes == two.n_hexes and back.n_points == two.n_points
    assert hexmesh.volume(back, high_order=True) == pytest.approx(
        hexmesh.volume(two, high_order=True), rel=1e-12)
    # a two-body mesh is not "watertight" (that reading includes single-body), so the
    # round trip is pinned against what it was cut from rather than against True
    assert hexmesh.is_watertight(back) == hexmesh.is_watertight(two)
    assert hexmesh.is_conforming(back) and hexmesh.is_conforming(two)


def test_a_subset_of_a_block_is_still_a_valid_mesh():
    """Watertight and conforming over the piece that was kept -- the B-rep comes back
    complete, with no orphaned shared face left behind."""
    block = _rungs(3)[2][0]
    half = hexmesh.select(block, np.arange(block.n_hexes // 2))
    assert hexmesh.is_watertight(half) and hexmesh.is_conforming(half)
    assert hexmesh.quality_summary(half).n_inverted == 0


def test_renumber_map_is_the_bookkeeping_it_claims():
    keep = np.array([True, False, True, True, False])
    ids, new_of = conform.renumber_map(keep)
    assert ids.tolist() == [0, 2, 3]
    assert new_of.tolist() == [0, -1, 1, 2, -1]
