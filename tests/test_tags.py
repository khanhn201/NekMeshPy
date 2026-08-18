"""``core.tags`` -- the sparse element-tag table every rung stores, and the two
vocabularies a mesh keeps in it.

There is one table now. A rung's side tags *are* the rung below's ``element_tags``, so
what used to be three ``(element, side)`` tables -- ``PointTags`` / ``EdgeTags`` /
``FaceTags`` -- is the same ``ElementTags`` seen one rung down, addressed by the id of
the entity it names. It is deliberately *not* called "the boundary": it names a chosen
subset of entities, which is a different set from the topological domain boundary that
``boundary_faces`` computes.

Its operations are all defined as "the sparse form of" a dense numpy expression, so
most of these tests assert exactly that: build a random dense reference, run the sparse
op, and compare ``dense()`` against the expression it replaces.
"""

from collections import Counter

import numpy as np
import pytest
from conftest import face_rows, read_re2_boundary

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core.tags import ElementTags

VOCAB = ["", "wall", "inlet", "outlet", "a_much_longer_region_name"]


def random_dense(rng, n, p_tagged=0.4):
    """A dense ``(n,)`` tag array with roughly ``p_tagged`` of the slots named."""
    pick = rng.random(n) < p_tagged
    out = np.full(n, "", dtype="<U32")
    out[pick] = rng.choice(VOCAB[1:], size=int(pick.sum()))
    return out


# -- ElementTags: construction / normalization ---------------------------
def test_element_tags_empty_allocates_nothing():
    empty = ElementTags.empty()
    assert len(empty) == 0 and not empty
    assert empty.ids.nbytes == 0 and empty.tags.nbytes == 0
    assert empty.group_tags == []


def test_from_dense_drops_empties():
    t = ElementTags.from_dense(["", "wall", "", "inlet"])
    assert np.array_equal(t.ids, [1, 3])
    assert np.array_equal(t.tags, ["wall", "inlet"])
    assert len(t) == 2                     # tagged count, NOT element count


def test_normalization_sorts_and_rejects_duplicates():
    t = ElementTags([3, 1], ["c", "a"])
    assert np.array_equal(t.ids, [1, 3]) and np.array_equal(t.tags, ["a", "c"])
    assert len(ElementTags([0, 1], ["", "x"])) == 1          # "" dropped
    with pytest.raises(ValueError, match="tagged more than once"):
        ElementTags([2, 2], ["a", "b"])
    with pytest.raises(ValueError, match="negative element id"):
        ElementTags([-1], ["a"])
    with pytest.raises(ValueError, match="same length"):
        ElementTags([0, 1], ["a"])


def test_uniform_does_not_clip_the_tag():
    """The ``<U1`` footgun: ``np.full(n, tag, dtype=np.str_)`` would give ``'w'``."""
    t = ElementTags.uniform(3, "wall")
    assert t.tags.tolist() == ["wall"] * 3
    assert t.dense(3).tolist() == ["wall"] * 3
    assert len(ElementTags.uniform(4, "")) == 0


def test_concat_promotes_string_width():
    a = ElementTags.uniform(1, "a")
    b = ElementTags([1], ["a_much_longer_region_name"])
    both = ElementTags.concat([a, b])
    assert both.tags.tolist() == ["a", "a_much_longer_region_name"]


def test_element_tags_read_only_and_repr():
    t = ElementTags.uniform(2, "wall")
    with pytest.raises(ValueError):
        t.ids[0] = 7
    assert repr(t) == "<ElementTags 2 tagged {wall}>"


# -- ElementTags: each op against its dense reference --------------------
def test_dense_roundtrip():
    rng = np.random.default_rng(1)
    for n in (0, 1, 7, 64):
        d = random_dense(rng, n)
        assert ElementTags.from_dense(d).dense(n).tolist() == d.tolist()


def test_gather_matches_dense_indexing():
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = int(rng.integers(1, 30))
        d = random_dense(rng, n)
        idx = rng.integers(0, n, size=int(rng.integers(0, 40))).astype(np.int64)
        got = ElementTags.from_dense(d).gather(idx).dense(idx.shape[0])
        assert got.tolist() == d[idx].tolist()


def test_gather_does_not_leak_a_neighbours_tag():
    """A searchsorted miss must not pick up the neighbouring id's tag."""
    t = ElementTags([5], ["wall"])                  # nothing tagged below id 5
    assert t.gather(np.array([0, 1, 5], dtype=np.int64)).dense(3).tolist() == [
        "", "", "wall"]


def test_repeat_blocks_matches_tile():
    rng = np.random.default_rng(3)
    for _ in range(30):
        m = int(rng.integers(1, 8))
        nz = int(rng.integers(1, 6))
        d = random_dense(rng, m)
        got = ElementTags.from_dense(d).repeat_blocks(nz, m).dense(nz * m)
        assert got.tolist() == np.tile(d, nz).tolist()


def test_blocks_matches_repeat():
    rng = np.random.default_rng(4)
    for _ in range(30):
        m = int(rng.integers(1, 8))
        nz = int(rng.integers(1, 6))
        layers = random_dense(rng, nz)
        got = ElementTags.blocks(layers, m).dense(nz * m)
        assert got.tolist() == np.repeat(layers, m).tolist()


def test_overlay_matches_np_where():
    """``overlay`` is the sparse form of ``np.where(over != "", over, base)``."""
    rng = np.random.default_rng(5)
    for _ in range(80):
        n = int(rng.integers(1, 40))
        base_d = random_dense(rng, n, 0.5)
        over_d = random_dense(rng, n, 0.3)
        got = ElementTags.from_dense(base_d).overlay(
            ElementTags.from_dense(over_d)).dense(n)
        assert got.tolist() == np.where(over_d != "", over_d, base_d).tolist()


def test_overlay_edge_cases():
    a = ElementTags.uniform(2, "base")
    assert a.overlay(ElementTags.empty()) is a
    assert ElementTags.empty().overlay(a) is a


def test_offset_and_concat_for_merge():
    a = ElementTags.from_dense(["wall", ""])
    b = ElementTags.from_dense(["", "inlet"])
    m = ElementTags.concat([a, b.offset(2)])
    assert m.dense(4).tolist() == ["wall", "", "", "inlet"]
    assert ElementTags.concat([]).group_tags == []


def test_renumber_reverses_with_the_elements():
    t = ElementTags.from_dense(["a", "", "c"])
    n = 3
    rev = t.renumber((n - 1 - np.arange(n)).astype(np.int64))
    assert rev.dense(n).tolist() == ["c", "", "a"]


def test_is_uniform():
    assert ElementTags.uniform(4, "wall").is_uniform(4)
    assert not ElementTags.uniform(4, "wall").is_uniform(5)       # partly tagged
    assert not ElementTags.from_dense(["a", "b"]).is_uniform(2)   # two vocabularies
    assert not ElementTags.empty().is_uniform(3)


def test_group_tags_sorted_unique():
    t = ElementTags.from_dense(["b", "a", "b", ""])
    assert t.group_tags == ["a", "b"]


# -- validation the table does for itself ---------------------------------
def test_check_within_is_the_only_thing_needing_the_mesh():
    """Element *count* is the mesh's, not the table's -- so it is the one check a
    container passes in. Everything the three side-tag types used to validate for
    themselves (a side in 1..SIDES, the rung's own noun in the message) went away with
    them: a tag names an entity id now, and an id out of range is this same check."""
    et = ElementTags([3], ["fluid"])
    et.check_within(4)                                           # 0..3 -> fine
    with pytest.raises(ValueError, match="element_tags names element 3"):
        et.check_within(3)
    ElementTags.empty().check_within(0)                          # empty is always fine


def test_the_container_still_rejects_an_out_of_range_element():
    from nekmeshpy import HexMesh
    ring = linemesh.circle(1.0, 8)
    sec = quadmesh.ogrid(ring, 2, np.linspace(0.5, 1.0, 3))
    blk = hexmesh.extrude(sec, length=1.0, layers=2)
    with pytest.raises(ValueError, match="element_tags names element"):
        HexMesh(blk.quad_mesh, blk.hexes, blk.orient, None,
                ElementTags([blk.n_hexes], ["fluid"]))


# -- renaming: the tag.py rung operations --------------------------------
def test_renamed_applies_simultaneously_and_can_widen():
    """The map is read off the *original* tags, so a swap is a swap rather than two
    sequential overwrites collapsing both groups onto one -- and the result is re-sized
    to the new names, not written into the old array's fixed width."""
    t = ElementTags([0, 1, 2, 3], ["a", "b", "a", "wall"])
    got = t.renamed({"a": "b", "b": "a"})
    assert got.tags.tolist() == ["b", "a", "b", "wall"]
    assert t.renamed({"a": "a_much_longer_region_name"}).tags.tolist() == [
        "a_much_longer_region_name", "b", "a_much_longer_region_name", "wall"]


def test_renamed_merges_and_keeps_row_order():
    """Two keys may share an image. Row order is untouched -- ``.re2`` writes rows in
    it, so a rename must not become a re-sort."""
    t = ElementTags([4, 0, 2], ["inlet", "outlet", "wall"])
    got = t.renamed({"inlet": "open", "outlet": "open"})
    assert list(got) == [(0, "open"), (2, "wall"), (4, "open")]


def test_renamed_to_no_tag_drops_side_rows_but_keeps_the_rest():
    t = ElementTags([0, 1, 2], ["inlet", "wall", "outlet"])
    got = t.renamed({"inlet": "", "outlet": ""})
    assert list(got) == [(1, "wall")]


def test_renamed_to_no_tag_untags_elements():
    t = ElementTags([0, 2, 5], ["fluid", "solid", "fluid"])
    got = t.renamed({"fluid": ""})
    assert got.ids.tolist() == [2] and got.tags.tolist() == ["solid"]


def test_renamed_rejects_a_key_that_names_nothing():
    """A rename matching nothing is almost always a typo, and a mis-spelled boundary
    name is not visible again until the solver reads it."""
    t = ElementTags([0], ["wall"])
    with pytest.raises(ValueError, match="nothing is tagged 'wal'"):
        t.renamed({"wal": "wall"}, "quadmesh.retag_edge")
    assert t.renamed({}).tags.tolist() == ["wall"]
    assert ElementTags.empty().renamed({}).tags.tolist() == []


def _ladder():
    """One mesh per rung, each carrying **both** tables, built by lifting the one
    below -- so the side tags a rung renames really are the ones it inherited.

    Tagged the way the rungs are meant to be: a section's ``element_tags`` is the
    *boundary* name the face it becomes will carry one rung up ("cap"), never a region
    name.  A region ("fluid") belongs to the top rung alone, because only there is an
    element a piece of the domain rather than a piece of some domain's surface."""
    ln = linemesh.loft(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                       element_tags="wall", first_tag="inlet", last_tag="outlet")
    quad = quadmesh.extrude(ln, 1.0, 2, axis=(0, 1, 0), element_tags="cap")
    return {linemesh: ln, quadmesh: quad,
            hexmesh: hexmesh.extrude(quad, 1.0, 2, axis=(0, 0, 1),
                                     element_tags="fluid")}


@pytest.mark.parametrize("rung, retag_side, side_slot", [
    (linemesh, "retag_point", "point_tags"),
    (quadmesh, "retag_edge", "edge_tags"),
    (hexmesh, "retag_face", "face_tags"),
])
def test_retag_side_is_geometry_preserving_at_every_rung(rung, retag_side, side_slot):
    """Each rung's ``tag.py`` renames its own side table and touches nothing else --
    that is the whole reason these are not in ``morph``."""
    mesh = _ladder()[rung]
    before = getattr(mesh, side_slot).group_tags
    assert "inlet" in before
    got = getattr(rung, retag_side)(mesh, {"inlet": "supply"})

    assert getattr(got, side_slot).group_tags == sorted(
        "supply" if t == "inlet" else t for t in before)
    assert len(getattr(got, side_slot)) == len(getattr(mesh, side_slot))
    assert np.array_equal(got.points, mesh.points)
    assert got.order == mesh.order
    assert got.element_tags.group_tags == mesh.element_tags.group_tags
    assert getattr(mesh, side_slot).group_tags == before      # original untouched


@pytest.mark.parametrize("rung", [linemesh, quadmesh, hexmesh])
def test_retag_element_at_every_rung(rung):
    mesh = _ladder()[rung]
    side_slot = {linemesh: "point_tags", quadmesh: "edge_tags",
                 hexmesh: "face_tags"}[rung]
    before = getattr(mesh, side_slot).group_tags
    old = mesh.element_tags.group_tags[0]
    got = rung.retag_element(mesh, {old: "renamed"})

    assert got.element_tags.group_tags == ["renamed"]
    assert len(got.element_tags) == len(mesh.element_tags)
    assert np.array_equal(got.points, mesh.points)
    assert getattr(got, side_slot).group_tags == before


def test_retag_element_leaves_a_shared_word_in_the_side_table(built_mesh):
    """The region table and the side table are different slots, so a word they happen
    to share is renamed in one and not the other.  Contrived here -- a section's
    ``element_tags`` should be the boundary name it becomes, not a region -- but the
    two vocabularies are only kept apart by convention, so the separation is worth
    holding to."""
    mesh = built_mesh["mesh"]
    assert "wall" in mesh.face_tags.group_tags
    collided = hexmesh.HexMesh(mesh.quad_mesh, mesh.hexes, mesh.orient, mesh.interior,
                               ElementTags.uniform(mesh.hexes.shape[0], "wall"))
    got = hexmesh.retag_element(collided, {"wall": "fluid"})
    assert got.element_tags.group_tags == ["fluid"]
    assert got.face_tags.group_tags == mesh.face_tags.group_tags


def test_retag_face_drops_a_name_welded_shut(built_mesh):
    """Renaming to ``NO_TAG`` retires a boundary name without disturbing the rows
    around it -- what ``tag_report`` flags after a weld makes a tagged face interior."""
    mesh = built_mesh["mesh"]
    assert "trunk_outlet" in mesh.face_tags.group_tags
    n_drop = mesh.face_tags.count("trunk_outlet")
    got = hexmesh.retag_face(mesh, {"trunk_outlet": ""})
    assert "trunk_outlet" not in got.face_tags.group_tags
    assert len(got.face_tags) == len(mesh.face_tags) - n_drop
    assert hexmesh.tag_report(got).n_untagged_boundary == n_drop


# -- asymmetric boundary conditions --------------------------------------
def _two_region_block():
    """Two stacked hexes, the interface between them named -- fluid below, solid above.

    The face is one shared object with one name; what differs is the *region* on
    either side of it, which is the only place an asymmetry can live now."""
    g = np.zeros((2, 2, 3))
    for i, x in enumerate((0.0, 1.0)):
        for j, y in enumerate((0.0, 1.0)):
            g[i, j] = (x, y, 0.0)
    sec = quadmesh.from_grid(g)
    lo = hexmesh.extrude(sec, 1.0, 1, axis=(0, 0, 1), element_tags="fluid")
    hi = hexmesh.translate(
        hexmesh.extrude(sec, 1.0, 1, axis=(0, 0, 1), element_tags="solid"), (0, 0, 1.0))
    mesh = hexmesh.merge([lo, hi])
    iface = mesh.face_tags.ids if len(mesh.face_tags) else None
    assert iface is None
    shared = np.flatnonzero(np.bincount(np.asarray(mesh.hexes).ravel()) == 2)
    return hexmesh.tag_faces(mesh, shared, "interface")


def test_per_region_codes_split_the_two_sides_of_one_face(tmp_path):
    """The interface is one named face carried by two hexes, so it reconstructs to two
    rows -- and each takes its code from the region of the hex that owns it."""
    mesh = _two_region_block()
    assert len(mesh.face_tags) == 1                     # one face, one name
    assert len(face_rows(mesh)) == 2                    # two hexes carry it

    out = str(tmp_path / "m.re2")
    writer.to_re2(mesh, out, groups={"interface": {"fluid": "W  ", "solid": "I  "}})
    got = read_re2_boundary(out)
    assert got == Counter({(1, 6, "W  "): 1, (2, 5, "I  "): 1})


def test_a_none_side_code_writes_no_row_at_all(tmp_path):
    """How a face gets a condition from one side only -- what a conjugate interface
    keeping just the fluid's wall needs."""
    mesh = _two_region_block()
    out = str(tmp_path / "m.re2")
    writer.to_re2(mesh, out, groups={"interface": {"fluid": "W  ", "solid": None}})
    assert read_re2_boundary(out) == Counter({(1, 6, "W  "): 1})


def test_a_region_the_codes_do_not_name_is_an_error(tmp_path):
    mesh = _two_region_block()
    with pytest.raises(ValueError, match="borders an element in region 'solid'"):
        writer.to_re2(mesh, str(tmp_path / "m.re2"),
                      groups={"interface": {"fluid": "W  "}})
