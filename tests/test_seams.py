"""Unit tests for the three operations that make a seam between independently built
pieces *exact* rather than merely close: ``quadmesh.reindex``, ``hexmesh.adapter`` and
``hexmesh.bridge``.

Why exactness rather than a tolerance is the property under test: at ``order > 1``
``HexMesh.merge`` verifies shared high-order edge and face nodes against
``conform.entity_tol`` (~1e-9 of the model extent), far tighter than any coordinate
weld.  A seam that is "close" at order 1 fails outright at order 2.  So:

1. **``reindex`` is a pure relabelling.**  The returned coordinate *set* must be the
   target's, bit for bit -- that is what lets the result weld against the target's own
   real pattern later.  A rotated *copy* would be close, and close is what fails.
2. **``adapter`` leaves ``a`` untouched and lands exactly on both ends.**  Its first
   slice must be ``a``'s own points and its last ``b``'s own coordinates.
3. **``bridge`` is one loft with no internal seam**, so it comes out conformal by
   construction across patterns too far apart for a blend to pair naively.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh

RADIAL = np.array([0.0, 0.5, 1.0])


def _disc(radius=1.0, center=(0.0, 0.0, 0.0), n_side=2, order=2, start_theta=0.0):
    ring = linemesh.circle(radius, 4 * n_side, center=center, start_theta=start_theta,
                           element_tag="wall", order=order)
    return quadmesh.ogrid(ring, n_side, RADIAL, wall_tag="wall")


def _identity_sigma(mesh):
    return np.arange(mesh.points.shape[0], dtype=np.int64)


# -- reindex ------------------------------------------------------------------
def test_reindex_with_the_identity_reproduces_the_target():
    a, b = _disc(), _disc(radius=1.3)
    out = quadmesh.reindex(a, b, _identity_sigma(a))
    assert np.array_equal(out.points, b.points)
    assert np.array_equal(out.line_mesh.interior, b.line_mesh.interior)
    assert np.array_equal(out.interior, b.interior)


def test_reindex_keeps_the_targets_coordinate_set_exactly():
    """The property the whole operation exists for: relabelled, not moved."""
    a, b = _disc(), _disc(radius=1.3)
    sigma = _shift_sigma(a)
    out = quadmesh.reindex(a, b, sigma)
    assert np.array_equal(np.sort(out.points, axis=0), np.sort(b.points, axis=0))
    assert np.array_equal(out.points, b.points[sigma])


def _shift_sigma(mesh):
    """A non-trivial permutation that is still a symmetry of the disc: the 90-degree
    self-map, found the same way ``adapter`` finds it."""
    from scipy.spatial import cKDTree
    c = mesh.points.mean(axis=0)
    spun = quadmesh.rotate(mesh, np.pi / 2.0, axis=(0.0, 0.0, 1.0), center=c)
    _, sigma = cKDTree(spun.points).query(mesh.points)
    return np.asarray(sigma, dtype=np.int64)


def test_reindex_carries_the_structures_numbering():
    a, b = _disc(), _disc(radius=1.3)
    out = quadmesh.reindex(a, b, _shift_sigma(a))
    assert np.array_equal(out.quad, a.quad)
    assert np.array_equal(out.orient, a.orient)
    assert np.array_equal(out.line_mesh.lines, a.line_mesh.lines)


def test_reindex_result_can_be_blended_against_the_structure():
    """``blend`` demands identical connectivity paired by index; making two
    independently built sections satisfy that is what ``reindex`` is for."""
    a, b = _disc(), _disc(radius=1.3)
    out = quadmesh.reindex(a, b, _shift_sigma(a))
    mid = quadmesh.blend(a, out, [0.0, 0.5, 1.0])
    assert np.array_equal(mid[0].points, a.points)
    assert np.array_equal(mid[-1].points, out.points)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_reindex_moves_the_high_order_nodes_with_their_edges(order):
    a, b = _disc(order=order), _disc(radius=1.3, order=order)
    out = quadmesh.reindex(a, b, _shift_sigma(a))
    assert out.order == order
    assert np.array_equal(np.sort(out.line_mesh.interior.reshape(-1, 3), axis=0),
                          np.sort(b.line_mesh.interior.reshape(-1, 3), axis=0))


def test_reindex_rejects_a_non_permutation():
    a, b = _disc(), _disc(radius=1.3)
    sigma = _identity_sigma(a)
    sigma[1] = sigma[0]
    with pytest.raises(ValueError, match="not a permutation"):
        quadmesh.reindex(a, b, sigma)


def test_reindex_rejects_mismatched_connectivity():
    a, b = _disc(n_side=2), _disc(n_side=3)
    with pytest.raises(ValueError, match="identical quad/flip|one entry per point"):
        quadmesh.reindex(a, b, np.arange(a.points.shape[0]))


# -- adapter ------------------------------------------------------------------
def _rolled_pair(order=2):
    """Two discs off the same recipe whose patterns differ slightly, with ``b``'s
    index pairing rolled a quarter turn relative to ``a``'s."""
    a = _disc(radius=1.0, order=order)
    b = quadmesh.translate(
        quadmesh.rotate(_disc(radius=1.04, order=order), np.pi / 2.0,
                        axis=(0.0, 0.0, 1.0)), (0.0, 0.0, 1.0))
    return a, b


@pytest.mark.parametrize("order", [1, 2])
def test_adapter_end_faces_are_bit_exact(order):
    a, b = _rolled_pair(order)
    block = hexmesh.adapter(a, b, axis=(0.0, 0.0, 1.0), layers=2)
    pts = block.points
    for name, section in (("a", a), ("b", b)):
        d = np.linalg.norm(pts[:, None, :] - section.points[None, :, :],
                           axis=2).min(axis=0)
        assert d.max() == 0.0, "the %s end of the adapter is not bit-exact" % name


def test_adapter_is_a_valid_uninverted_block():
    a, b = _rolled_pair()
    block = hexmesh.adapter(a, b, axis=(0.0, 0.0, 1.0), layers=2)
    assert hexmesh.is_watertight(block)
    assert hexmesh.is_conforming(block)
    assert hexmesh.scaled_jacobian(block).min() > 0.0


def test_adapter_layers_controls_the_count():
    a, b = _rolled_pair()
    n2 = hexmesh.adapter(a, b, axis=(0.0, 0.0, 1.0), layers=2).n_hexes
    n5 = hexmesh.adapter(a, b, axis=(0.0, 0.0, 1.0), layers=5).n_hexes
    assert n5 == n2 // 2 * 5


def test_adapter_refuses_patterns_no_roll_aligns():
    """Far-apart patterns are ``bridge``'s job; blending them twists the block."""
    a = _disc(radius=1.0)
    b = quadmesh.translate(_disc(radius=4.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="no 90-degree roll"):
        hexmesh.adapter(a, b, axis=(0.0, 0.0, 1.0))


# -- bridge -------------------------------------------------------------------
def _far_pair(order=2):
    """Same radius, genuinely different station *distribution* -- the case no rotation
    fixes, because it is not an orientation mismatch."""
    a = _disc(radius=1.0, order=order)
    b = quadmesh.translate(_disc(radius=1.0, order=order, start_theta=0.37),
                           (0.0, 0.0, 6.0))
    return a, b


@pytest.mark.parametrize("order", [1, 2])
def test_bridge_is_conformal_by_construction(order):
    """One loft, so there is no internal seam for merge to verify."""
    a, b = _far_pair(order)
    block = hexmesh.bridge(a, b)
    assert hexmesh.is_conforming(block)
    assert hexmesh.is_watertight(block)


@pytest.mark.parametrize("order", [1, 2])
def test_bridge_near_ends_stay_bonded_to_their_own_discs(order):
    a, b = _far_pair(order)
    pts = hexmesh.bridge(a, b).points
    for name, section in (("a", a), ("b", b)):
        d = np.linalg.norm(pts[:, None, :] - section.points[None, :, :],
                           axis=2).min(axis=0)
        assert d.max() == 0.0, "the %s end of the bridge is not bit-exact" % name


def test_bridge_is_not_inverted():
    a, b = _far_pair()
    assert hexmesh.scaled_jacobian(hexmesh.bridge(a, b)).min() > 0.0


def test_bridge_merges_with_blocks_built_off_its_own_ends():
    """The whole point: a bridge welds into the assembly at both ends at order > 1."""
    a, b = _far_pair(order=2)
    left = hexmesh.extrude(a, 1.0, 2, axis=(0.0, 0.0, -1.0), last_tag="inlet")
    right = hexmesh.extrude(b, 1.0, 2, axis=(0.0, 0.0, 1.0), last_tag="outlet")
    whole = hexmesh.merge([left, hexmesh.bridge(a, b), right])
    rep = hexmesh.topology_report(whole)
    assert rep.watertight and rep.conformal and rep.n_components == 1


def test_bridge_rejects_patterns_that_do_not_pair_one_for_one():
    a = _disc(radius=1.0, n_side=2)
    b = quadmesh.translate(_disc(radius=1.0, n_side=2), (0.0, 0.0, 6.0))
    # collapse a cluster of b's points onto one another -- the section keeps its plane
    # and its extent, but nearest-neighbour matching can no longer be one-for-one
    pts = np.array(b.points, dtype=float)
    pts[1:6] = pts[0]
    b.points[:] = pts
    with pytest.raises(ValueError, match="not a permutation"):
        hexmesh.bridge(a, b)


def test_bridge_and_adapter_are_reachable_from_the_namespace_module():
    assert hexmesh.lift.bridge is hexmesh.bridge
    assert hexmesh.lift.adapter is hexmesh.adapter
    assert quadmesh.morph.reindex is quadmesh.reindex
