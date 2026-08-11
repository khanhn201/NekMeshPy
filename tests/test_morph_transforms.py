"""The unary rung-preserving operations: the placements ``translate`` / ``rotate`` /
``scale`` / ``transform`` on ``LineMesh`` / ``QuadMesh`` / ``HexMesh``, and
``LineMesh.reverse``.

The placements are one affine map applied at three rungs, so the properties to pin are
the same at each: only *coordinates* move (connectivity and tags ride
through verbatim), the map reaches **every** node -- the private high-order
``interior`` tables as well as the corners -- and a rigid map leaves element quality
untouched.  ``reverse`` is the mirror image and is pinned the same way from the other
side: it moves no coordinate and only relabels, carrying the high-order nodes with it.
"""

import numpy as np
import pytest
from conftest import assert_same_side_tags

from nekmeshpy import ElementTags, LineMesh, QuadMesh, hexmesh, linemesh, quadmesh

RADIAL = np.linspace(0.4, 1.0, 3)


def _meshes(order):
    """One mesh per rung, all curved at ``order > 1`` (the circle's interior nodes sit
    on the true arc), so a transform that skipped a table would show up."""
    ring = linemesh.circle(2.0, 8, element_tag="wall", order=order)
    section = quadmesh.ogrid(ring, 2, RADIAL, wall_tag="wall")
    block = hexmesh.extrude(section, length=1.0, layers=np.linspace(0.0, 1.0, 3),
                            first_tag="inlet", last_tag="outlet")
    return ring, section, block


def _rungs(order):
    """Each mesh paired with the package holding its operations.

    Operations are free functions in per-rung namespaces, so a call site names its
    rung -- the pairing is spelled out here rather than dispatched on ``type(mesh)``."""
    ring, section, block = _meshes(order)
    return ((ring, linemesh), (section, quadmesh), (block, hexmesh))


def _tables(mesh):
    """Every coordinate table the mesh owns, top rung down."""
    if isinstance(mesh, LineMesh):
        return [mesh.points, mesh.interior]
    if isinstance(mesh, QuadMesh):
        return [mesh.interior, *_tables(mesh.lines)]
    return [mesh.interior, *_tables(mesh.quads)]


@pytest.fixture(params=[1, 3], ids=["order1", "order3"])
def order(request):
    return request.param


# -- translation --------------------------------------------------------------
def test_translate_moves_every_table_exactly(order):
    """A translation is added without a matmul, so it is bit-exact at every rung and
    on every table -- corners *and* the private interiors."""
    v = np.array([1.5, -2.0, 0.25])
    for mesh, ns in _rungs(order):
        moved = ns.morph.translate(mesh, v)
        for before, after in zip(_tables(mesh), _tables(moved)):
            assert np.array_equal(after, before + v)


def test_translate_by_zero_is_a_strict_no_op(order):
    for mesh, ns in _rungs(order):
        moved = ns.morph.translate(mesh, (0.0, 0.0, 0.0))
        for before, after in zip(_tables(mesh), _tables(moved)):
            assert np.array_equal(after, before)


def test_placement_keeps_topology_and_tags(order):
    """Only coordinates move: incidence and tags are carried verbatim."""
    rungs = _rungs(order)
    (ring, _), (section, _), (block, _) = rungs
    for (mesh, ns), attr in zip(rungs, ("point_tags", "edge_tags", "face_tags")):
        out = ns.morph.translate(mesh, (1.0, 0.0, 0.0))
        assert np.array_equal(out.element_tags.ids, mesh.element_tags.ids)
        assert np.array_equal(out.element_tags.tags, mesh.element_tags.tags)
        assert_same_side_tags(getattr(out, attr), getattr(mesh, attr))
        assert out.order == mesh.order
    assert np.array_equal(linemesh.translate(ring, (1.0, 0, 0)).lines, ring.lines)
    assert np.array_equal(quadmesh.rotate(section, 0.3).quads, section.quads)
    assert np.array_equal(hexmesh.scale(block, 2.0).hexes, block.hexes)


# -- rotation -----------------------------------------------------------------
def test_rotate_is_rigid(order):
    """Rigid: every pairwise distance -- and therefore element quality -- survives."""
    _, (section, _), (block, _) = _rungs(order)
    for mesh, ns in ((section, quadmesh), (block, hexmesh)):
        out = ns.morph.rotate(mesh, 0.7, axis=(1.0, 2.0, 3.0), center=(0.5, 0.0, -1.0))
        d0 = np.linalg.norm(mesh.points[:, None, :] - mesh.points[None, :, :], axis=2)
        d1 = np.linalg.norm(out.points[:, None, :] - out.points[None, :, :], axis=2)
        assert np.allclose(d0, d1, atol=1e-12)
        assert np.allclose(np.sort(ns.query.scaled_jacobian(out)),
                           np.sort(ns.query.scaled_jacobian(mesh)), atol=1e-12)


def test_rotate_keeps_high_order_nodes_on_the_true_circle():
    """The interior nodes are mapped by the same rigid map as the corners, so a
    high-order circle stays an exact circle after placement."""
    ring = linemesh.circle(2.0, 8, order=4)
    out = linemesh.rotate(ring, np.pi / 3, axis=(0.0, 1.0, 0.0))
    r = np.linalg.norm(
        np.vstack([out.points, out.interior.reshape(-1, 3)]), axis=1)
    assert np.allclose(r, 2.0, atol=1e-12)


def test_rotate_fixes_its_center():
    ring = linemesh.circle(1.0, 6, center=(3.0, 0.0, 0.0))
    out = linemesh.rotate(ring, np.pi, center=(3.0, 0.0, 0.0))
    assert np.allclose(np.mean(out.points, axis=0), np.mean(ring.points, axis=0))


def test_rotate_axis_need_not_be_normalized():
    ring = linemesh.circle(1.0, 6, order=3)
    a = linemesh.rotate(ring, 0.4, axis=(0.0, 3.0, 0.0))
    b = linemesh.rotate(ring, 0.4, axis=(0.0, 1.0, 0.0))
    assert np.allclose(a.points, b.points, atol=1e-14)
    assert np.allclose(a.interior, b.interior, atol=1e-14)


# -- scaling ------------------------------------------------------------------
def test_scale_uniform_and_per_axis(order):
    for mesh, ns in _rungs(order):
        for factor in (2.0, (1.0, 2.0, 3.0)):
            out = ns.morph.scale(mesh, factor)
            for before, after in zip(_tables(mesh), _tables(out)):
                assert np.allclose(after, before * np.asarray(factor), atol=1e-14)


def test_scale_about_a_center_fixes_it():
    c = np.array([1.0, -2.0, 0.5])
    ring = linemesh.circle(1.0, 6, center=c)
    out = linemesh.scale(ring, 3.0, center=c)
    assert np.allclose(np.linalg.norm(out.points - c, axis=1), 3.0)


# -- the general affine -------------------------------------------------------
def test_transform_is_the_general_case():
    """``transform`` with the rotation's own matrix reproduces ``rotate`` exactly."""
    from nekmeshpy.core import affine

    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, RADIAL)
    matrix, offset = affine.rotation(0.6, axis=(0.0, 1.0, 1.0), center=(1.0, 0, 0))
    out = quadmesh.transform(section, matrix, offset)
    ref = quadmesh.rotate(section, 0.6, axis=(0.0, 1.0, 1.0), center=(1.0, 0, 0))
    assert np.array_equal(out.points, ref.points)
    assert np.array_equal(out.interior, ref.interior)


# -- composition down the ladder ----------------------------------------------
def test_quad_and_hex_delegate_to_the_rung_below(order):
    """A quad's shared corners and edge nodes *are* its edge ``LineMesh``, so the
    quad map must equal the line map on that mesh; likewise hex -> quad."""
    ring, section, block = _meshes(order)
    v = (0.0, 1.0, -0.5)
    assert np.array_equal(quadmesh.translate(section, v).lines.points,
                          linemesh.translate(section.lines, v).points)
    assert np.array_equal(quadmesh.translate(section, v).lines.interior,
                          linemesh.translate(section.lines, v).interior)
    assert np.array_equal(hexmesh.rotate(block, 0.2).quads.points,
                          quadmesh.rotate(block.quads, 0.2).points)


def test_extrude_is_a_stack_of_translations(order):
    """``extrude`` places its slices through ``translate``; doing it by hand and
    lofting reproduces the block exactly."""
    _, section, _ = _meshes(order)
    axis = np.array([0.0, 0.0, 1.0])
    ref = hexmesh.extrude(section, length=2.0, layers=np.linspace(0.0, 1.0, 3))
    manual = hexmesh.loft([quadmesh.translate(section, d * axis)
                           for d in np.linspace(0.0, 1.0, 3) * 2.0])
    assert np.array_equal(manual.points, ref.points)
    assert np.array_equal(manual.hexes, ref.hexes)
    assert np.array_equal(manual.interior, ref.interior)


# -- rejections ---------------------------------------------------------------
def test_rejections():
    ring = linemesh.circle(1.0, 6)
    with pytest.raises(ValueError, match="axis must be non-zero"):
        linemesh.rotate(ring, 0.5, axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="must be positive"):
        linemesh.scale(ring, 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        linemesh.scale(ring, (1.0, -1.0, 1.0))
    with pytest.raises(ValueError, match=r"\(3,\) displacement"):
        linemesh.translate(ring, (1.0, 2.0))
    with pytest.raises(ValueError, match="scalar or a"):
        linemesh.scale(ring, (1.0, 2.0))


# -- reverse: a relabel, not a move -------------------------------------------
@pytest.mark.parametrize("order", [1, 4])
def test_reverse_relabels_without_moving_anything(order):
    """Point ``i`` becomes ``N-1-i`` and every coordinate is carried over, so the
    reversed curve is the identical geometry with the opposite orientation."""
    lm = linemesh.arc(2.0, 4, start_theta=0.0, end_theta=np.pi / 2, order=order)
    out = linemesh.reverse(lm)
    assert np.array_equal(out.points, lm.points[::-1])
    assert out.lines.tolist() == lm.lines.tolist()          # still the same chain
    assert np.array_equal(out.interior, lm.interior[::-1, ::-1, :])


def test_reverse_keeps_high_order_nodes_on_the_true_arc():
    """The defect ``reverse`` exists to close: re-lofting the reversed *points*
    straight-subdivides the interior and leaves the true arc."""
    lm = linemesh.arc(2.0, 4, start_theta=0.0, end_theta=np.pi / 2, order=4)
    rev = linemesh.reverse(lm)
    good = np.vstack([rev.points, rev.interior.reshape(-1, 3)])
    assert np.allclose(np.linalg.norm(good, axis=1), 2.0, atol=1e-13)
    trap = linemesh.loft(lm.points[::-1], order=lm.order)
    bad = np.linalg.norm(trap.interior.reshape(-1, 3), axis=1)
    assert np.max(np.abs(bad - 2.0)) > 1e-3                 # the chord, not the arc


@pytest.mark.parametrize("order", [1, 3])
def test_reverse_is_an_involution(order):
    lm = linemesh.circle(1.0, 6, element_tag="wall", order=order)
    back = linemesh.reverse(linemesh.reverse(lm))
    assert np.array_equal(back.points, lm.points)
    assert np.array_equal(back.lines, lm.lines)
    assert np.array_equal(back.interior, lm.interior)
    assert np.array_equal(back.element_tags.ids, lm.element_tags.ids)
    assert np.array_equal(back.element_tags.tags, lm.element_tags.tags)


def test_reverse_remaps_element_and_point_tags_to_the_same_physical_points():
    chain = linemesh.loft(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                          first_tag="in", last_tag="out")
    lm = LineMesh(chain.vertices, chain.lines, chain.interior,
                  ElementTags.from_dense(["a", "b"]))
    out = linemesh.reverse(lm)
    assert out.element_tags.dense(out.n_lines).tolist() == ["b", "a"]
    # the tag that named the x=0 end still names it after the relabel
    tagged = {t: out.points[i].tolist() for i, t in out.point_tags}
    assert tagged["in"] == [0.0, 0.0, 0.0]
    assert tagged["out"] == [2.0, 0.0, 0.0]


def test_reverse_keeps_a_loop_closed():
    """It relabels rather than re-lofting, so it works on any connectivity."""
    assert linemesh.boundary_points(linemesh.reverse(linemesh.circle(1.0, 8))).size == 0


# -- reverse on a *closed* loop: the relabel that keeps the ring canonical -----
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_reverse_leaves_a_ring_canonically_ordered(order):
    """An open chain reverses onto ``i -> N-1-i``; a ring cannot, because that also
    rotates it -- line ``k`` would come back spanning ``k-1 -> k``.  A loop pins point
    ``0`` instead, so line ``k`` still leaves point ``k``, which is the ordering every
    factory that meshes a loop exactly reads it through."""
    rev = linemesh.reverse(linemesh.circle(1.0, 8, order=order))
    n = rev.n_points
    want = np.column_stack([np.arange(n), (np.arange(n) + 1) % n])
    assert np.array_equal(rev.lines, want)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_a_reversed_ring_still_fills(order):
    """The defect this closes: the rotated ring handed ``ogrid`` its high-order nodes
    one segment out, and the elevation raised a non-conforming-edge error rather than
    meshing.  The fill must now match the unreversed one exactly."""
    ring = linemesh.circle(1.0, 8, order=order)
    got = quadmesh.ogrid(linemesh.reverse(ring), 2, 2)
    assert quadmesh.area(got, high_order=True) == pytest.approx(
        quadmesh.area(quadmesh.ogrid(ring, 2, 2), high_order=True), rel=1e-12)
    assert quadmesh.quality_summary(got).n_inverted == 0


def test_a_reversed_rings_per_segment_tags_stay_on_their_segments():
    """The order-1 face of the same defect, and the quieter one: the rotation shifted
    every per-segment wall tag by one segment without erroring."""
    ring = linemesh.circle(1.0, 8)
    named = LineMesh(ring.vertices, ring.lines, ring.interior,
                     ElementTags(np.arange(8),
                                 np.array(["s%d" % k for k in range(8)])))
    rev = linemesh.reverse(named)
    section = quadmesh.ogrid(rev, 2, 2)
    where = dict(zip(rev.element_tags.tags.tolist(), rev.element_tags.ids.tolist()))
    for eid, tag in section.edge_tags:
        mid = section.points[section.lines.lines[eid]].mean(axis=0)
        assert np.allclose(mid, rev.points[rev.lines[where[tag]]].mean(axis=0))


@pytest.mark.parametrize("order", [1, 3])
def test_reverse_still_walks_an_open_chain_off_its_last_point(order):
    """The open case is untouched -- point ``i`` becomes ``N-1-i``, as documented."""
    lm = linemesh.arc(2.0, 4, start_theta=0.0, end_theta=np.pi / 2, order=order)
    assert np.array_equal(linemesh.reverse(lm).points, lm.points[::-1])


# -- cap-tag shape parity across the three rungs ------------------------------
def test_line_loft_caps_are_one_name_each():
    """A chain's cap is one node, so the per-element form the rungs above accept
    reduces to a single string here -- and anything else is refused outright."""
    P = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    scalar = linemesh.loft(P, first_tag="in", last_tag="out")
    assert list(scalar.point_tags) == [(0, "in"), (2, "out")]
    with pytest.raises(TypeError, match="single tag string"):
        linemesh.loft(P, first_tag=["in"])
