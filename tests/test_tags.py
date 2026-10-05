"""``tags`` -- the sparse element-tag table every rung stores, and the two
vocabularies a mesh keeps in it.

There is one table now. A rung's side tags *are* the rung below's ``element_tags``, so
what used to be three ``(element, side)`` tables -- ``PointTags`` / ``EdgeTags`` /
``FaceTags`` -- is the same ``Tags`` seen one rung down, addressed by the id of
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
from nekmeshpy.quadmesh import QuadMesh
from nekmeshpy.tags import Tags

VOCAB = ["", "wall", "inlet", "outlet", "a_much_longer_region_name"]


def random_dense(rng, n, p_tagged=0.4):
    """A dense ``(n,)`` tag array with roughly ``p_tagged`` of the slots named."""
    pick = rng.random(n) < p_tagged
    out = np.full(n, "", dtype="<U32")
    out[pick] = rng.choice(VOCAB[1:], size=int(pick.sum()))
    return out


# -- Tags: construction / normalization ---------------------------
def test_element_tags_empty_allocates_nothing():
    empty = Tags.empty(5)
    assert len(empty) == 0 and not empty and empty.size == 5
    assert empty.ids.nbytes == 0 and empty.tags.nbytes == 0
    assert empty.unique() == []


def test_from_dense_drops_empties():
    t = Tags.from_dense(["", "wall", "", "inlet"])
    assert np.array_equal(t.ids, [1, 3])
    assert np.array_equal(t.tags, ["wall", "inlet"])
    assert len(t) == 2                     # tagged count, NOT element count


def test_normalization_sorts_and_rejects_duplicates():
    t = Tags([3, 1], ["c", "a"], 4)
    assert np.array_equal(t.ids, [1, 3]) and np.array_equal(t.tags, ["a", "c"])
    assert len(Tags([0, 1], ["", "x"], 2)) == 1          # "" dropped
    with pytest.raises(ValueError, match="tagged more than once"):
        Tags([2, 2], ["a", "b"], 3)
    with pytest.raises(ValueError, match="negative element id"):
        Tags([-1], ["a"], 3)
    with pytest.raises(ValueError, match="same length"):
        Tags([0, 1], ["a"], 2)


def test_size_is_validated_at_construction():
    assert Tags([3], ["fluid"], 4).size == 4              # 0..3 -> fine
    with pytest.raises(ValueError, match="element 3 is tagged but the table is over "
                                         "only 3"):
        Tags([3], ["fluid"], 3)
    with pytest.raises(ValueError, match="non-negative"):
        Tags.empty(-1)
    Tags.empty(0)                                          # empty is always fine
    assert Tags.from_dense(["a", "", ""]).size == 3
    assert Tags.full(4, "").size == 4 and len(Tags.full(4, "")) == 0


def test_uniform_does_not_clip_the_tag():
    """The ``<U1`` footgun: ``np.full(n, tag, dtype=np.str_)`` would give ``'w'``."""
    t = Tags.full(3, "wall")
    assert t.tags.tolist() == ["wall"] * 3
    assert t.to_dense().tolist() == ["wall"] * 3
    assert len(Tags.full(4, "")) == 0 and Tags.full(4, "wall").size == 4


def test_concat_promotes_string_width():
    a = Tags.full(1, "a")
    b = Tags([0], ["a_much_longer_region_name"], 1)
    both = Tags.concatenate([a, b])
    assert both.tags.tolist() == ["a", "a_much_longer_region_name"]


def test_element_tags_read_only_and_repr():
    t = Tags.full(2, "wall")
    with pytest.raises(ValueError):
        t.ids[0] = 7
    assert repr(t) == "<Tags 2 tagged {wall}>"


# -- Tags: each op against its dense reference --------------------
def test_dense_roundtrip():
    rng = np.random.default_rng(1)
    for n in (0, 1, 7, 64):
        d = random_dense(rng, n)
        assert Tags.from_dense(d).to_dense().tolist() == d.tolist()


def test_gather_matches_dense_indexing():
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = int(rng.integers(1, 30))
        d = random_dense(rng, n)
        idx = rng.integers(0, n, size=int(rng.integers(0, 40))).astype(np.int64)
        got = Tags.from_dense(d).take(idx).to_dense()
        assert got.tolist() == d[idx].tolist()


def test_gather_does_not_leak_a_neighbours_tag():
    """A searchsorted miss must not pick up the neighbouring id's tag."""
    t = Tags([5], ["wall"], 6)               # nothing tagged below id 5
    assert t.take(np.array([0, 1, 5], dtype=np.int64)).to_dense().tolist() == [
        "", "", "wall"]


def test_tile_matches_np_tile():
    rng = np.random.default_rng(3)
    for _ in range(30):
        m = int(rng.integers(1, 8))
        nz = int(rng.integers(1, 6))
        d = random_dense(rng, m)
        tiled = Tags.from_dense(d).tile(nz)
        assert tiled.size == nz * m
        assert tiled.to_dense().tolist() == np.tile(d, nz).tolist()


def test_concatenate_offsets_each_table_by_the_sizes_before_it():
    a = Tags.from_dense(["wall", ""])
    b = Tags.from_dense(["", "inlet"])
    m = Tags.concatenate([a, b])
    assert m.size == 4
    assert m.to_dense().tolist() == ["wall", "", "", "inlet"]
    assert Tags.concatenate([]).unique() == [] and Tags.concatenate([]).size == 0
    assert Tags.concatenate([Tags.empty(2), Tags.empty(3)]).size == 5


def test_renumber_reverses_with_the_elements():
    t = Tags.from_dense(["a", "", "c"])
    n = 3
    rev = t.renumber((n - 1 - np.arange(n)).astype(np.int64), n)
    assert rev.to_dense().tolist() == ["c", "", "a"]
    grown = t.renumber(np.array([0, 1, 4]), 6)           # into a larger id space
    assert grown.size == 6 and grown.to_dense().tolist() == ["a", "", "", "", "c", ""]
    with pytest.raises(ValueError, match="only 3 elements"):
        t.renumber(np.array([0, 1, 5]), 3)


def test_is_uniform():
    assert Tags.full(4, "wall").is_uniform()
    assert not Tags.from_dense(["wall", ""]).is_uniform()       # partly tagged
    assert not Tags.from_dense(["a", "b"]).is_uniform()         # two vocabularies
    assert not Tags.empty(3).is_uniform()


def test_group_tags_sorted_unique():
    t = Tags.from_dense(["b", "a", "b", ""])
    assert t.unique() == ["a", "b"]


# -- the container checks the table against its own element count ---------
def test_the_container_rejects_a_table_over_the_wrong_element_count():
    from nekmeshpy import HexMesh
    ring = linemesh.circle(1.0, 8)
    sec = quadmesh.ogrid(ring, 2, np.linspace(0.5, 1.0, 3))
    blk = hexmesh.extrude(sec, length=1.0, layers=2)
    with pytest.raises(ValueError, match="element_tags must be over the"):
        HexMesh(blk.quad_mesh, blk.hexes, blk.orient, None,
                Tags([0], ["fluid"], blk.n_hexes + 1))
    assert HexMesh(blk.quad_mesh, blk.hexes, blk.orient, None).element_tags.size \
        == blk.n_hexes


# -- renaming: the tag.py rung operations --------------------------------
def test_rename_applies_simultaneously_and_can_widen():
    """The map is read off the *original* tags, so a swap is a swap rather than two
    sequential overwrites collapsing both groups onto one -- and the result is re-sized
    to the new names, not written into the old array's fixed width."""
    t = Tags([0, 1, 2, 3], ["a", "b", "a", "wall"], 4)
    got = t.rename({"a": "b", "b": "a"})
    assert got.tags.tolist() == ["b", "a", "b", "wall"]
    assert t.rename({"a": "a_much_longer_region_name"}).tags.tolist() == [
        "a_much_longer_region_name", "b", "a_much_longer_region_name", "wall"]


def test_rename_merges_and_keeps_row_order():
    """Two keys may share an image. Row order is untouched -- ``.re2`` writes rows in
    it, so a rename must not become a re-sort."""
    t = Tags([4, 0, 2], ["inlet", "outlet", "wall"], 5)
    got = t.rename({"inlet": "open", "outlet": "open"})
    assert list(got) == [(0, "open"), (2, "wall"), (4, "open")]


def test_rename_to_no_tag_drops_side_rows_but_keeps_the_rest():
    t = Tags([0, 1, 2], ["inlet", "wall", "outlet"], 3)
    got = t.rename({"inlet": "", "outlet": ""})
    assert list(got) == [(1, "wall")]


def test_rename_to_no_tag_untags_elements():
    t = Tags([0, 2, 5], ["fluid", "solid", "fluid"], 6)
    got = t.rename({"fluid": ""})
    assert got.ids.tolist() == [2] and got.tags.tolist() == ["solid"]


def test_rename_rejects_a_key_that_names_nothing():
    """A rename matching nothing is almost always a typo, and a mis-spelled boundary
    name is not visible again until the solver reads it."""
    t = Tags([0], ["wall"], 1)
    with pytest.raises(ValueError, match="nothing is tagged 'wal'"):
        t.rename({"wal": "wall"})
    assert t.rename({}).tags.tolist() == ["wall"]
    assert Tags.empty(1).rename({}).tags.tolist() == []


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
    before = getattr(mesh, side_slot).unique()
    assert "inlet" in before
    got = getattr(rung, retag_side)(mesh, {"inlet": "supply"})

    assert getattr(got, side_slot).unique() == sorted(
        "supply" if t == "inlet" else t for t in before)
    assert len(getattr(got, side_slot)) == len(getattr(mesh, side_slot))
    assert np.array_equal(got.points, mesh.points)
    assert got.order == mesh.order
    assert got.element_tags.unique() == mesh.element_tags.unique()
    assert getattr(mesh, side_slot).unique() == before      # original untouched


@pytest.mark.parametrize("rung", [linemesh, quadmesh, hexmesh])
def test_retag_element_at_every_rung(rung):
    mesh = _ladder()[rung]
    side_slot = {linemesh: "point_tags", quadmesh: "edge_tags",
                 hexmesh: "face_tags"}[rung]
    before = getattr(mesh, side_slot).unique()
    old = mesh.element_tags.unique()[0]
    got = rung.retag_element(mesh, {old: "renamed"})

    assert got.element_tags.unique() == ["renamed"]
    assert len(got.element_tags) == len(mesh.element_tags)
    assert np.array_equal(got.points, mesh.points)
    assert getattr(got, side_slot).unique() == before


def test_retag_element_leaves_a_shared_word_in_the_side_table(built_mesh):
    """The region table and the side table are different slots, so a word they happen
    to share is renamed in one and not the other.  Contrived here -- a section's
    ``element_tags`` should be the boundary name it becomes, not a region -- but the
    two vocabularies are only kept apart by convention, so the separation is worth
    holding to."""
    mesh = built_mesh["mesh"]
    assert "wall" in mesh.face_tags.unique()
    collided = hexmesh.HexMesh(mesh.quad_mesh, mesh.hexes, mesh.orient, mesh.interior,
                               Tags.full(mesh.hexes.shape[0], "wall"))
    got = hexmesh.retag_element(collided, {"wall": "fluid"})
    assert got.element_tags.unique() == ["fluid"]
    assert got.face_tags.unique() == mesh.face_tags.unique()


def test_retag_face_drops_a_name_welded_shut(built_mesh):
    """Renaming to ``NO_TAG`` retires a boundary name without disturbing the rows
    around it -- what ``tag_report`` flags after a weld makes a tagged face interior."""
    mesh = built_mesh["mesh"]
    assert "trunk_outlet" in mesh.face_tags.unique()
    n_drop = mesh.face_tags.count("trunk_outlet")
    got = hexmesh.retag_face(mesh, {"trunk_outlet": ""})
    assert "trunk_outlet" not in got.face_tags.unique()
    assert len(got.face_tags) == len(mesh.face_tags) - n_drop
    assert hexmesh.tag_report(got).n_untagged_boundary == n_drop


def _row_of_three():
    """Three unit hexes in a row along x, as three separate blocks."""
    sec = quadmesh.from_grid(np.array([[[0.0, 0, 0], [0, 1, 0]],
                                       [[1.0, 0, 0], [1, 1, 0]]]))
    return [hexmesh.translate(hexmesh.extrude(sec, 1.0, 1, axis=(0, 0, 1)),
                              (0, 0, float(k))) for k in range(3)]


def test_merge_clear_seam_tags_true_drops_every_buried_name():
    """A block named on its whole boundary, welded shut on two sides: with
    ``clear_seam_tags=True`` only the faces that stayed on the boundary keep the name."""
    a, b, c = _row_of_three()
    a = hexmesh.tag_faces(a, np.arange(a.quad_mesh.n_quads), "skin")   # every face
    plain = hexmesh.merge([a, b, c])
    kept = hexmesh.merge([a, b, c], clear_seam_tags=True)
    assert plain.face_tags.count("skin") == 6            # 5 boundary + 1 buried
    assert kept.face_tags.count("skin") == 5             # the buried one is gone
    assert hexmesh.tag_report(kept).n_tagged_interior == 0
    on_bdry = hexmesh.query.boundary_face_ids(kept)
    assert on_bdry[np.asarray(kept.face_tags.ids)].all()


def test_merge_clear_seam_tags_is_name_scoped_and_still_conflict_checks():
    """A list clears only the named tags on buried faces; another tag on the same
    buried face survives, and two different surviving names still raise."""
    a, b, _ = _row_of_three()                            # stacked along z

    def _cap(m, z):
        fc = m.quad_mesh.points[np.asarray(m.quad_mesh.corners)].mean(axis=1)
        return np.flatnonzero(np.isclose(fc[:, 2], z))

    a = hexmesh.tag_faces(a, _cap(a, 1.0), "inlet")      # a's +z cap (welds to b)
    b = hexmesh.tag_faces(b, _cap(b, 1.0), "wall")       # b's -z cap (same seam)
    with pytest.raises(ValueError, match="two different names"):
        hexmesh.merge([a, b])
    got = hexmesh.merge([a, b], clear_seam_tags=["inlet"])
    assert got.face_tags.unique() == ["wall"]          # inlet cleared, wall kept
    with pytest.raises(ValueError, match="two different names"):
        hexmesh.merge([a, b], clear_seam_tags=["something-else"])


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


# -- element_tag in the .vtu -------------------------------------------------
# A region belongs to the element, so it is written as *cell* data, unlike ``bc_id``,
# which is per point.  That is not a stylistic choice: on a conjugate mesh every
# interface node is shared by both regions, so a per-point region would have to pick
# one of them and would be wrong on the other side.


def _vtu_arrays(path):
    """``{name: values}`` for every ``DataArray`` in a binary ``.vtu``."""
    import base64
    import xml.etree.ElementTree as ET
    dtypes = {"Float64": "<f8", "Int64": "<i8", "Int32": "<i4", "UInt8": "u1"}
    out = {}
    for da in ET.parse(path).getroot().iter("DataArray"):
        raw = base64.b64decode(da.text.strip())
        n = int(np.frombuffer(raw[:8], "<u8")[0])
        out[da.get("Name") or "Points"] = np.frombuffer(raw[8:8 + n],
                                                        dtypes[da.get("type")])
    return out


def _two_region_section():
    """Two stacked unit squares, the lower ``solid`` and the upper ``fluid``."""
    lower = quadmesh.rectangle([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)], 2, 1)
    upper = quadmesh.rectangle([(0, 1, 0), (2, 1, 0), (2, 2, 0), (0, 2, 0)], 2, 1)
    section = quadmesh.merge([lower, upper])
    return QuadMesh(section.line_mesh, section.quads, section.orient,
                    section.interior,
                    Tags.from_dense(["solid", "solid", "fluid", "fluid"]))


def test_element_tag_ids_are_the_sorted_vocabulary_one_based():
    """Ids are a function of the mesh alone -- the sorted vocabulary, 1-based, with 0
    for untagged -- so a reader recovers the legend without the file carrying one."""
    tags = Tags.from_dense(["fluid", "", "solid", "fluid"])
    ids, names = writer.element_tag_ids(tags, 4)
    assert names == ["fluid", "solid"]
    assert list(ids) == [1, 0, 2, 1]


def test_quad_vtu_carries_element_tag_per_cell(tmp_path):
    section = _two_region_section()
    out = str(tmp_path / "section.vtu")
    writer.quad_to_vtu(section, out)
    ids, names = writer.element_tag_ids(section.element_tags, section.n_quads)
    assert names == ["fluid", "solid"]
    assert np.array_equal(_vtu_arrays(out)["element_tag"], ids)


def test_hex_vtu_carries_element_tag_per_cell(tmp_path):
    """One value per hex, not per node: the swept column keeps its section quad's
    region, so the count is ``n_quads * layers``."""
    mesh = hexmesh.extrude(_two_region_section(), 1.0, 3,
                           element_tags=_two_region_section().element_tags,
                           first_tag="front", last_tag="back")
    out = str(tmp_path / "block.vtu")
    writer.to_vtu(mesh, out, groups={"front": "SYM", "back": "SYM"})
    got = _vtu_arrays(out)["element_tag"]
    assert got.shape == (mesh.n_hexes,)
    assert np.array_equal(got, writer.element_tag_ids(mesh.element_tags,
                                                      mesh.n_hexes)[0])


def test_untagged_mesh_writes_no_cell_data(tmp_path):
    """An untagged mesh gets no ``CellData`` at all rather than a column of zeros."""
    plain = quadmesh.rectangle([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], 2, 2)
    out = str(tmp_path / "plain.vtu")
    writer.quad_to_vtu(plain, out)
    assert "element_tag" not in _vtu_arrays(out)


# -- the authoring bridges take one name or one per row -----------------------
def _plain_section():
    return quadmesh.rectangle([(0, 0, 0), (3, 0, 0), (3, 1, 0), (0, 1, 0)], 3, 1)


def test_tag_edges_broadcasts_one_name_over_every_row():
    """A bare string is a sequence *of characters*: zipped against the rows it would tag
    only the first edge, and truncated to a one-character dtype it would name it ``'w'``.
    Both are silent, and a seam that lost 2 of its 3 faces is not refused by anything
    until a weld comes up short."""
    rows = np.array([[q, 1] for q in range(3)], dtype=np.int64)
    tagged = quadmesh.tag_edges(_plain_section(), rows, "wall")
    assert tagged.element_group_tags == []          # elements untouched
    assert sorted(tagged.edge_tags.unique()) == ["wall"]
    assert len(quadmesh.tagged_edges(tagged, "wall")) == 3


def test_tag_edges_takes_one_name_per_row():
    rows = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int64)
    tagged = quadmesh.tag_edges(_plain_section(), rows, ["a", "b", "b"])
    assert sorted(tagged.edge_tags.unique()) == ["a", "b"]
    assert len(quadmesh.tagged_edges(tagged, "b")) == 2


def test_tag_edges_refuses_a_count_that_is_neither():
    rows = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="3 rows but 2 tags"):
        quadmesh.tag_edges(_plain_section(), rows, ["a", "b"])


def test_quadrant_ogrid_names_its_own_elements():
    """A quadrant is one patch of a disc or one side of a tetra, so it has to be able to
    carry a name of its own -- that is what lets a four-quadrant disc hand a *different*
    cap name to each quadrant through ``first_tag``."""
    ring = linemesh.circle(1.0, 8, order=1)
    arc = linemesh.select(ring, [0, 1])
    fr = quadmesh.quadrant_seam_fractions(1, 2, 0.7)
    s1 = linemesh.line(np.zeros(3), arc.points[0], fr)
    s2 = linemesh.line(np.zeros(3), arc.points[-1], fr)
    q = quadmesh.quadrant_ogrid(arc, s1, s2, 2, center_scale=0.7, element_tag="patch")
    assert q.element_group_tags == ["patch"]
    assert q.element_tags.is_uniform()
    plain = quadmesh.quadrant_ogrid(arc, s1, s2, 2, center_scale=0.7)
    assert plain.element_group_tags == []


def test_compress_follows_np_compress_over_the_stored_rows():
    t = Tags.from_dense(["a", "", "b", "c"])           # rows: (0,a) (2,b) (3,c)
    assert t.compress([True, False, True]).ids.tolist() == [0, 3]
    assert t.compress([False, True]).ids.tolist() == [2]       # shorter than the table
    assert t.compress([]).ids.tolist() == []
    assert t.compress([True, False, False, False, False]).ids.tolist() == [0]
    with pytest.raises(IndexError):
        t.compress([False, False, False, True])
    with pytest.raises(ValueError):
        t.compress([[True]])


def test_isin_takes_one_name_or_several():
    t = Tags.from_dense(["a", "b", "", "c"])
    assert t.isin("b").tolist() == [False, True, False]
    assert t.isin(["a", "c"]).tolist() == [True, False, True]


def test_take_follows_np_take_bounds():
    t = Tags.from_dense(["a", "", "b"])
    assert t.take([-1, 0]).to_dense().tolist() == ["b", "a"]      # negatives wrap
    assert t.take([2, 2]).size == 2                               # size = len(indices)
    with pytest.raises(IndexError):
        t.take([3])
    with pytest.raises(IndexError):
        t.take([-4])
    assert Tags.empty(3).take([1, 2]).size == 2


def test_selection_by_several_tag_names():
    from nekmeshpy.core.selection import mask_for_selection
    t = Tags.from_dense(["a", "b", "", "c", "a"])
    assert mask_for_selection(["a", "c"], t).tolist() == [True, False, False, True, True]
    assert mask_for_selection(np.array(["b"]), t).tolist() == [False, True, False, False, False]
    assert mask_for_selection("a", t).tolist() == [True, False, False, False, True]
    with pytest.raises(ValueError, match="'zzz'"):
        mask_for_selection(["a", "zzz"], t)


def test_flatnonzero_gives_the_ids_carrying_any_name():
    t = Tags.from_dense(["a", "b", "", "c", "a"])
    assert t.flatnonzero("a").tolist() == [0, 4]
    assert t.flatnonzero(["c", "b"]).tolist() == [1, 3]
    assert t.flatnonzero("zzz").tolist() == []


def test_put_overwrites_names_and_skips_empty_values():
    t = Tags.from_dense(["a", "", "b", ""])
    got = t.put([1, 2, 3], ["x", "", "a_much_longer_name"])
    assert got.to_dense().tolist() == ["a", "x", "b", "a_much_longer_name"]
    assert t.put([0, 3], "w").to_dense().tolist() == ["w", "", "b", "w"]
    assert t.put([1, 1], ["p", "q"]).to_dense().tolist() == ["a", "q", "b", ""]
    assert t.put([], "z") is t
    with pytest.raises(ValueError):
        t.put([0, 1], ["x"])


def test_clear_untags_and_ignores_untagged_ids():
    t = Tags.from_dense(["a", "", "b", "c"])
    got = t.clear([0, 1, 3])
    assert got.to_dense().tolist() == ["", "", "b", ""] and got.size == 4
    assert t.clear([]) is t
    with pytest.raises(IndexError):
        t.clear([4])


def test_take_dense_reads_names_by_id():
    t = Tags.from_dense(["a", "", "b"])
    assert t.take_dense([2, 1, 0, 2, -1]).tolist() == ["b", "", "a", "b", "b"]
    assert t.take_dense([]).tolist() == []
    with pytest.raises(IndexError):
        t.take_dense([9])


def test_unique_return_inverse_codes_each_row():
    t = Tags.from_dense(["b", "", "a", "b"])
    names, inverse = t.unique(return_inverse=True)
    assert names == ["a", "b"] and inverse.tolist() == [1, 0, 1]
    assert Tags.empty(3).unique(return_inverse=True)[1].tolist() == []


def test_put_refuses_an_id_outside_the_table():
    t = Tags.from_dense(["a", "", "b"])
    assert t.put([-1], "z").to_dense().tolist() == ["a", "", "z"]    # negative wraps
    with pytest.raises(IndexError):
        t.put([3], "z")
    assert t.put([0], "z").size == 3


def test_weld_requires_one_element_count():
    from nekmeshpy.tags import weld
    a = Tags.from_dense(["w", ""])
    b = Tags.from_dense(["", "i"])
    assert weld([a, b]).to_dense().tolist() == ["w", "i"]
    with pytest.raises(ValueError, match="different element counts"):
        weld([a, Tags.empty(3)])


def test_sweep_arguments_must_be_over_one_slice():
    from nekmeshpy.tags import sweep_cap, sweep_tags
    t = Tags.from_dense(["a", "b"])
    assert sweep_tags(t, 3, 2).size == 6
    assert sweep_tags(None, 3, 2).size == 6 and sweep_tags("x", 3, 2).size == 6
    assert sweep_cap(t, Tags.empty(2), 2).tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="over the 3 elements of one slice"):
        sweep_tags(t, 2, 3)
    with pytest.raises(ValueError, match="over the 3 elements of one slice"):
        sweep_cap(t, Tags.empty(3), 3)


def test_repeat_matches_np_repeat():
    d = np.array(["a", "", "b"])
    got = Tags.from_dense(d).repeat(3)
    assert got.size == 9 and got.to_dense().tolist() == np.repeat(d, 3).tolist()
    assert Tags.empty(2).repeat(4).size == 8


def test_astype_bool_marks_tagged_elements():
    t = Tags.from_dense(["a", "", "b"])
    assert t.astype(bool).tolist() == [True, False, True]
    with pytest.raises(TypeError):
        t.astype(int)
