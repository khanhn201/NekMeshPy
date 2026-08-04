"""The unary rung-preserving operations: the placements ``translate`` / ``rotate`` /
``scale`` / ``transform`` on ``LineMesh`` / ``QuadMesh`` / ``HexMesh``, and
``LineMesh.reverse``.

The placements are one affine map applied at three rungs, so the properties to pin are
the same at each: only *coordinates* move (connectivity, tags and boundaries ride
through verbatim), the map reaches **every** node -- the private high-order
``interior`` tables as well as the corners -- and a rigid map leaves element quality
untouched.  ``reverse`` is the mirror image and is pinned the same way from the other
side: it moves no coordinate and only relabels, carrying the high-order nodes with it.
"""

import numpy as np
import pytest
from conftest import assert_same_boundaries

from nekmeshpy import HexMesh, LineMesh, QuadMesh

RADIAL = np.linspace(0.4, 1.0, 3)


def _meshes(order):
    """One mesh per rung, all curved at ``order > 1`` (the circle's interior nodes sit
    on the true arc), so a transform that skipped a table would show up."""
    ring = LineMesh.circle(2.0, 8, element_tags=["wall"] * 8, order=order)
    section = QuadMesh.ogrid(ring, 2, RADIAL, wall_tag="wall")
    block = HexMesh.extrude(section, length=1.0, layers=np.linspace(0.0, 1.0, 3),
                            first_tag="inlet", last_tag="outlet")
    return ring, section, block


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
    for mesh in _meshes(order):
        moved = mesh.translate(v)
        for before, after in zip(_tables(mesh), _tables(moved)):
            assert np.array_equal(after, before + v)


def test_translate_by_zero_is_a_strict_no_op(order):
    for mesh in _meshes(order):
        moved = mesh.translate((0.0, 0.0, 0.0))
        for before, after in zip(_tables(mesh), _tables(moved)):
            assert np.array_equal(after, before)


def test_placement_keeps_topology_and_tags(order):
    """Only coordinates move: incidence, tags and boundaries are carried verbatim."""
    ring, section, block = _meshes(order)
    for mesh in (ring, section, block):
        out = mesh.translate((1.0, 0.0, 0.0))
        assert np.array_equal(out.element_tags.ids, mesh.element_tags.ids)
        assert np.array_equal(out.element_tags.tags, mesh.element_tags.tags)
        assert_same_boundaries(out.boundaries, mesh.boundaries)
        assert out.order == mesh.order
    assert np.array_equal(ring.translate((1.0, 0, 0)).lines, ring.lines)
    assert np.array_equal(section.rotate(0.3).quads, section.quads)
    assert np.array_equal(block.scale(2.0).hexes, block.hexes)


# -- rotation -----------------------------------------------------------------
def test_rotate_is_rigid(order):
    """Rigid: every pairwise distance -- and therefore element quality -- survives."""
    _, section, block = _meshes(order)
    for mesh in (section, block):
        out = mesh.rotate(0.7, axis=(1.0, 2.0, 3.0), center=(0.5, 0.0, -1.0))
        d0 = np.linalg.norm(mesh.points[:, None, :] - mesh.points[None, :, :], axis=2)
        d1 = np.linalg.norm(out.points[:, None, :] - out.points[None, :, :], axis=2)
        assert np.allclose(d0, d1, atol=1e-12)
        assert np.allclose(np.sort(out.scaled_jacobian()),
                           np.sort(mesh.scaled_jacobian()), atol=1e-12)


def test_rotate_keeps_high_order_nodes_on_the_true_circle():
    """The interior nodes are mapped by the same rigid map as the corners, so a
    high-order circle stays an exact circle after placement."""
    ring = LineMesh.circle(2.0, 8, order=4)
    out = ring.rotate(np.pi / 3, axis=(0.0, 1.0, 0.0))
    r = np.linalg.norm(
        np.vstack([out.points, out.interior.reshape(-1, 3)]), axis=1)
    assert np.allclose(r, 2.0, atol=1e-12)


def test_rotate_fixes_its_center():
    ring = LineMesh.circle(1.0, 6, center=(3.0, 0.0, 0.0))
    out = ring.rotate(np.pi, center=(3.0, 0.0, 0.0))
    assert np.allclose(np.mean(out.points, axis=0), np.mean(ring.points, axis=0))


def test_rotate_axis_need_not_be_normalized():
    ring = LineMesh.circle(1.0, 6, order=3)
    a = ring.rotate(0.4, axis=(0.0, 3.0, 0.0))
    b = ring.rotate(0.4, axis=(0.0, 1.0, 0.0))
    assert np.allclose(a.points, b.points, atol=1e-14)
    assert np.allclose(a.interior, b.interior, atol=1e-14)


# -- scaling ------------------------------------------------------------------
def test_scale_uniform_and_per_axis(order):
    for mesh in _meshes(order):
        for factor in (2.0, (1.0, 2.0, 3.0)):
            out = mesh.scale(factor)
            for before, after in zip(_tables(mesh), _tables(out)):
                assert np.allclose(after, before * np.asarray(factor), atol=1e-14)


def test_scale_about_a_center_fixes_it():
    c = np.array([1.0, -2.0, 0.5])
    ring = LineMesh.circle(1.0, 6, center=c)
    out = ring.scale(3.0, center=c)
    assert np.allclose(np.linalg.norm(out.points - c, axis=1), 3.0)


# -- the general affine -------------------------------------------------------
def test_transform_is_the_general_case():
    """``transform`` with the rotation's own matrix reproduces ``rotate`` exactly."""
    from nekmeshpy.model import affine

    section = QuadMesh.ogrid(LineMesh.circle(1.0, 8, order=2), 2, RADIAL)
    matrix, offset = affine.rotation(0.6, axis=(0.0, 1.0, 1.0), center=(1.0, 0, 0))
    out = section.transform(matrix, offset)
    ref = section.rotate(0.6, axis=(0.0, 1.0, 1.0), center=(1.0, 0, 0))
    assert np.array_equal(out.points, ref.points)
    assert np.array_equal(out.interior, ref.interior)


# -- composition down the ladder ----------------------------------------------
def test_quad_and_hex_delegate_to_the_rung_below(order):
    """A quad's shared corners and edge nodes *are* its edge ``LineMesh``, so the
    quad map must equal the line map on that mesh; likewise hex -> quad."""
    ring, section, block = _meshes(order)
    v = (0.0, 1.0, -0.5)
    assert np.array_equal(section.translate(v).lines.points,
                          section.lines.translate(v).points)
    assert np.array_equal(section.translate(v).lines.interior,
                          section.lines.translate(v).interior)
    assert np.array_equal(block.rotate(0.2).quads.points,
                          block.quads.rotate(0.2).points)


def test_extrude_is_a_stack_of_translations(order):
    """``extrude`` places its slices through ``translate``; doing it by hand and
    lofting reproduces the block exactly."""
    _, section, _ = _meshes(order)
    axis = np.array([0.0, 0.0, 1.0])
    ref = HexMesh.extrude(section, length=2.0, layers=np.linspace(0.0, 1.0, 3))
    manual = HexMesh.loft([section.translate(d * axis)
                           for d in np.linspace(0.0, 1.0, 3) * 2.0])
    assert np.array_equal(manual.points, ref.points)
    assert np.array_equal(manual.hexes, ref.hexes)
    assert np.array_equal(manual.interior, ref.interior)


# -- rejections ---------------------------------------------------------------
def test_rejections():
    ring = LineMesh.circle(1.0, 6)
    with pytest.raises(ValueError, match="axis must be non-zero"):
        ring.rotate(0.5, axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="must be positive"):
        ring.scale(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        ring.scale((1.0, -1.0, 1.0))
    with pytest.raises(ValueError, match=r"\(3,\) displacement"):
        ring.translate((1.0, 2.0))
    with pytest.raises(ValueError, match="scalar or a"):
        ring.scale((1.0, 2.0))


# -- reverse: a relabel, not a move -------------------------------------------
@pytest.mark.parametrize("order", [1, 4])
def test_reverse_relabels_without_moving_anything(order):
    """Point ``i`` becomes ``N-1-i`` and every coordinate is carried over, so the
    reversed curve is the identical geometry with the opposite orientation."""
    lm = LineMesh.arc(2.0, 4, start_theta=0.0, end_theta=np.pi / 2, order=order)
    out = lm.reverse()
    assert np.array_equal(out.points, lm.points[::-1])
    assert out.lines.tolist() == lm.lines.tolist()          # still the same chain
    assert np.array_equal(out.interior, lm.interior[::-1, ::-1, :])


def test_reverse_keeps_high_order_nodes_on_the_true_arc():
    """The defect ``reverse`` exists to close: re-lofting the reversed *points*
    straight-subdivides the interior and leaves the true arc."""
    lm = LineMesh.arc(2.0, 4, start_theta=0.0, end_theta=np.pi / 2, order=4)
    good = np.vstack([lm.reverse().points, lm.reverse().interior.reshape(-1, 3)])
    assert np.allclose(np.linalg.norm(good, axis=1), 2.0, atol=1e-13)
    trap = LineMesh.loft(lm.points[::-1], order=lm.order)
    bad = np.linalg.norm(trap.interior.reshape(-1, 3), axis=1)
    assert np.max(np.abs(bad - 2.0)) > 1e-3                 # the chord, not the arc


@pytest.mark.parametrize("order", [1, 3])
def test_reverse_is_an_involution(order):
    lm = LineMesh.circle(1.0, 6, element_tags=["wall"] * 6, order=order)
    back = lm.reverse().reverse()
    assert np.array_equal(back.points, lm.points)
    assert np.array_equal(back.lines, lm.lines)
    assert np.array_equal(back.interior, lm.interior)
    assert np.array_equal(back.element_tags.ids, lm.element_tags.ids)
    assert np.array_equal(back.element_tags.tags, lm.element_tags.tags)


def test_reverse_remaps_tags_and_boundaries_to_the_same_physical_points():
    lm = LineMesh.loft(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                       element_tags=["a", "b"], first_tag="in", last_tag="out")
    out = lm.reverse()
    assert out.element_tags.dense(out.n_lines).tolist() == ["b", "a"]
    # the tag that named the x=0 end still names it after the relabel
    tagged = {t: out.points[out.lines[e, s - 1]].tolist()
              for e, s, t in out.boundaries}
    assert tagged["in"] == [0.0, 0.0, 0.0]
    assert tagged["out"] == [2.0, 0.0, 0.0]


def test_reverse_keeps_a_loop_closed():
    """It relabels rather than re-lofting, so it works on any connectivity."""
    assert LineMesh.circle(1.0, 8).reverse().boundary_points().size == 0


# -- cap-tag shape parity across the three rungs ------------------------------
def test_line_loft_caps_accept_the_array_form_like_the_rungs_above():
    """A chain's cap is one node, so the per-element form is a one-element array --
    the point is that the same argument shapes work at every rung."""
    P = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]])
    scalar = LineMesh.loft(P, first_tag="in", last_tag="out")
    array = LineMesh.loft(P, first_tag=["in"], last_tag=np.array(["out"]))
    assert_same_boundaries(array.boundaries, scalar.boundaries)
    with pytest.raises(ValueError, match="must match cap nodes"):
        LineMesh.loft(P, first_tag=["in", "also-in"])
