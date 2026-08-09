"""``model.tags`` -- the three side-tag tables and the sparse element-tag table.

``PointTags`` / ``EdgeTags`` / ``FaceTags`` share one implementation, so the row
semantics are exercised once (on ``EdgeTags``) plus a parity check that every operation
returns the caller's own subclass.  They are deliberately *not* called "boundaries":
they name a chosen subset of sides, which is a different set from the topological
domain boundary that ``boundary_faces`` computes.

The sparse element-tag operations are all defined as "the sparse form of" a dense numpy
expression, so most of these tests assert exactly that: build a random dense reference,
run the sparse op, and compare ``dense()`` against the expression it replaces.  The
side-tag tests concentrate on the two things the mesh's output depends on -- that
``ordered()`` is the same permutation the old ``_order_bnd`` applied, and that nothing
else ever reorders rows.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core.tags import (
    EdgeTags,
    ElementTags,
    FaceTags,
    PointTags,
    TagBuilder,
)

VOCAB = ["", "wall", "inlet", "outlet", "a_much_longer_region_name"]


def random_dense(rng, n, p_tagged=0.4):
    """A dense ``(n,)`` tag array with roughly ``p_tagged`` of the slots named."""
    pick = rng.random(n) < p_tagged
    out = np.full(n, "", dtype="<U32")
    out[pick] = rng.choice(VOCAB[1:], size=int(pick.sum()))
    return out


# -- the side-tag tables -------------------------------------------------------
def test_side_tags_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        EdgeTags([0, 1], [1, 2], ["wall"])


def test_side_tags_empty():
    t = EdgeTags.empty()
    assert len(t) == 0 and not t
    assert t.group_tags == [] and t.rows.shape == (0, 2)
    assert list(t) == []


def test_side_tags_from_pairs_roundtrip():
    t = EdgeTags.from_pairs([[3, 2], [0, 1]], ["b", "a"])
    assert np.array_equal(t.elements, [3, 0])
    assert np.array_equal(t.sides, [2, 1])
    assert np.array_equal(t.rows, [[3, 2], [0, 1]])
    assert list(t) == [(3, 2, "b"), (0, 1, "a")]


def test_ordered_matches_lexsort_including_ties():
    """``ordered()`` must be exactly the old ``_order_bnd`` permutation.

    Ties on ``(element, side)`` are the case that matters: lexsort is stable, and the
    ``.vtu`` writer resolves a repeated node to the *last* row, so a different tie order
    would silently change exported ``bc_id``s."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(0, 40))
        el = rng.integers(0, 5, size=n).astype(np.int64)
        sd = rng.integers(1, 4, size=n).astype(np.int64)
        tg = rng.choice(VOCAB[1:], size=n)
        t = EdgeTags(el, sd, tg).ordered()
        p = np.lexsort((sd, el))
        assert np.array_equal(t.elements, el[p])
        assert np.array_equal(t.sides, sd[p])
        assert np.array_equal(t.tags, tg[p])


def test_construction_never_reorders():
    """Unsorted tables legitimately reach storage; only ``ordered()`` may sort."""
    t = EdgeTags.from_pairs([[0, 1], [0, 3], [0, 4], [0, 2]], list("abcd"))
    assert np.array_equal(t.sides, [1, 3, 4, 2])
    assert np.array_equal(t.ordered().sides, [1, 2, 3, 4])
    assert np.array_equal(t.ordered().tags, ["a", "d", "b", "c"])


def test_offset_concat_select_count():
    a = EdgeTags.from_pairs([[0, 1], [1, 2]], ["wall", "inlet"])
    b = EdgeTags.from_pairs([[0, 3]], ["wall"])
    m = EdgeTags.concat([a, b.offset(2)])
    assert np.array_equal(m.elements, [0, 1, 2])
    assert m.count("wall") == 2 and m.count("inlet") == 1
    assert m.group_tags == ["inlet", "wall"]
    only_wall = m.select(m.mask_for("wall"))
    assert list(only_wall) == [(0, 1, "wall"), (2, 3, "wall")]
    assert EdgeTags.concat([]).group_tags == []
    assert len(EdgeTags.empty().offset(5)) == 0


def test_as_dict_is_last_row_wins_in_row_order():
    t = EdgeTags.from_pairs([[0, 1], [0, 1]], ["first", "second"])
    assert t.as_dict() == {(0, 1): "second"}


def test_side_tags_columns_are_read_only():
    t = EdgeTags.from_pairs([[0, 1]], ["wall"])
    with pytest.raises(ValueError):
        t.elements[0] = 9


def test_side_tags_repr_summarises():
    r = repr(EdgeTags.from_pairs([[0, 1], [1, 2]], ["wall", "inlet"]))
    assert r == "<EdgeTags 2 rows {inlet,wall}>"


# -- TagBuilder -----------------------------------------------------
def test_builder_preserves_insertion_order_and_duplicates():
    bb = TagBuilder(EdgeTags)
    bb.add(2, 1, "b")
    bb.add(0, 1, "a")
    bb.add(2, 1, "b")            # a genuine duplicate must survive
    assert len(bb) == 3 and bb
    assert list(bb.build()) == [(2, 1, "b"), (0, 1, "a"), (2, 1, "b")]
    assert list(bb.build_ordered()) == [(0, 1, "a"), (2, 1, "b"), (2, 1, "b")]


def test_builder_add_if_tagged_skips_empty():
    bb = TagBuilder(EdgeTags)
    bb.add_if_tagged(0, 1, "")
    bb.add_if_tagged(1, 2, "wall")
    assert list(bb.build()) == [(1, 2, "wall")]
    assert len(TagBuilder(EdgeTags).build()) == 0


def test_builder_extend():
    bb = TagBuilder(EdgeTags)
    bb.add(0, 1, "a")
    bb.extend(EdgeTags.from_pairs([[5, 2]], ["b"]))
    assert list(bb.build()) == [(0, 1, "a"), (5, 2, "b")]


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


# -- the three subclasses are distinct and self-propagating ---------------
@pytest.mark.parametrize("cls", [PointTags, EdgeTags, FaceTags])
def test_every_operation_returns_the_callers_own_subclass(cls):
    """A rung's table must stay its own type through every derivation -- otherwise a
    sorted or offset table would no longer satisfy its container's annotation."""
    t = cls.from_pairs([[1, 2], [0, 1]], ["b", "a"])
    for got in (t.ordered(), t.offset(3), t.select(t.mask_for("a")),
                cls.concat([t, t]), cls.empty()):
        assert type(got) is cls
    assert type(TagBuilder(cls).build()) is cls
    bb = TagBuilder(cls)
    bb.add(0, 1, "x")
    assert type(bb.build_ordered()) is cls
    assert repr(t).startswith("<%s 2 rows" % cls.__name__)


def test_length_mismatch_names_the_rungs_own_type():
    with pytest.raises(ValueError, match="FaceTags: elements"):
        FaceTags([0, 1], [1, 2], ["wall"])


# -- validation the table does for itself ---------------------------------
@pytest.mark.parametrize("cls,n_sides", [(PointTags, 2), (EdgeTags, 4), (FaceTags, 6)])
def test_side_range_is_enforced_by_the_type_with_no_mesh_present(cls, n_sides):
    """``SIDES`` is what makes the three types worth having as separate types: the
    valid side range is a property of the rung, so the table checks it itself rather
    than waiting for a container to be built from it."""
    assert cls.SIDES == n_sides
    cls.from_pairs([[0, 1], [0, n_sides]], ["a", "b"])          # both ends are fine
    with pytest.raises(ValueError, match=r"side %d is outside 1\.\.%d"
                                         % (n_sides + 1, n_sides)):
        cls.from_pairs([[0, n_sides + 1]], ["a"])
    with pytest.raises(ValueError, match=r"side 0 is outside 1\.\.%d" % n_sides):
        cls.from_pairs([[0, 0]], ["a"])
    with pytest.raises(ValueError, match="negative element id"):
        cls.from_pairs([[-1, 1]], ["a"])


def test_a_side_valid_one_rung_up_is_rejected_one_rung_down():
    """The check the shared base could not make: side 6 is a legal hex face and an
    illegal quad edge, so routing an EdgeTags through a 6-sided check would pass."""
    FaceTags.from_pairs([[0, 6]], ["top"])
    with pytest.raises(ValueError, match=r"EdgeTags: side 6 is outside 1\.\.4"):
        EdgeTags.from_pairs([[0, 6]], ["top"])


def test_check_within_is_the_only_thing_needing_the_mesh():
    """Element *count* is the mesh's, not the table's -- so it is the one check a
    container passes in, and the *only* thing it passes: what a row names is the
    table's own business (a FaceTags row names a hex), so no noun crosses over."""
    ft = FaceTags.from_pairs([[3, 1]], ["wall"])
    ft.check_within(4)                                           # 0..3 -> fine
    with pytest.raises(ValueError, match="FaceTags names element 3 but there are only "
                                         "3 hexes"):
        ft.check_within(3)
    et = ElementTags([3], ["fluid"])
    et.check_within(4)
    with pytest.raises(ValueError, match="element_tags names element 3"):
        et.check_within(3)
    FaceTags.empty().check_within(0)                             # empty is always fine
    ElementTags.empty().check_within(0)


def test_the_container_still_rejects_an_out_of_range_element():
    from nekmeshpy import HexMesh
    ring = linemesh.circle(1.0, 8)
    sec = quadmesh.ogrid(ring, 2, np.linspace(0.5, 1.0, 3))
    blk = hexmesh.extrude(sec, length=1.0, layers=2)
    with pytest.raises(ValueError, match="FaceTags names element"):
        HexMesh(blk.quads, blk.hex, blk.face_orient, None,
                FaceTags.from_pairs([[blk.n_hexes, 1]], ["wall"]))


# -- TagBuilder broadcasting -------------------------------------------
def test_tag_builder_broadcasts_its_three_arguments():
    """``element`` / ``side`` / ``tag`` each accept a scalar or an array and broadcast,
    so a caller names a whole column of rows without a Python loop."""
    from nekmeshpy.core.tags import EdgeTags, TagBuilder
    ids = np.array([3, 5, 7])
    bb = TagBuilder(EdgeTags)
    bb.add(ids, 2, "wall")                      # array x scalar x scalar
    bb.add(9, np.array([1, 3]), "in")           # scalar x array x scalar
    bb.add(ids, 4, np.array(["a", "b", "c"]))   # array x scalar x array
    got = bb.build()
    assert got.elements.tolist() == [3, 5, 7, 9, 9, 3, 5, 7]
    assert got.sides.tolist() == [2, 2, 2, 1, 3, 4, 4, 4]
    assert got.tags.tolist() == ["wall"] * 3 + ["in", "in"] + ["a", "b", "c"]
    assert len(bb) == 8


def test_add_if_tagged_drops_the_untagged_rows_of_an_array():
    """A per-element tag array goes over whole -- the empty entries simply do not
    become rows, which is what lets ``loft`` hand its cap tags straight across."""
    from nekmeshpy.core.tags import EdgeTags, TagBuilder
    bb = TagBuilder(EdgeTags)
    bb.add_if_tagged(np.arange(4), 1, np.array(["a", "", "c", ""]))
    bb.add_if_tagged(np.arange(4), 2, "")        # a scalar NO_TAG adds nothing
    assert not TagBuilder(EdgeTags)
    got = bb.build()
    assert got.elements.tolist() == [0, 2]
    assert got.tags.tolist() == ["a", "c"]
    assert len(bb) == 2
