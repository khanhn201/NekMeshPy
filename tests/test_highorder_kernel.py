"""Phase 1 tests for the order-N infrastructure: the GLL reference-node helpers
(``model.fields``), the shared tensor-product kernel (``model.interp``), and the
container ``order``/``curved`` plumbing.  The load-bearing property is that at
``order == 1`` every primitive reduces **exactly** to the existing linear data --
that is the golden-invariant contract."""

import numpy as np
import pytest

from nekmeshpy import HexMesh, LineMesh, QuadMesh
from nekmeshpy.model.fields import (
    gll_nodes,
    gll_weights,
    lagrange_derivative_matrix,
    lagrange_matrix,
)
from nekmeshpy.model.interp import (
    blend_ho,
    coons_grid,
    corner_indices,
    nodes_per_element,
    scaled_jacobian_ho,
    subdivide_element,
    tensor_nodes,
)


# -- GLL reference nodes ------------------------------------------------
@pytest.mark.parametrize("order", [1, 2, 3, 5, 8])
def test_gll_nodes_endpoints_and_symmetry(order):
    g = gll_nodes(order)
    assert g.shape == (order + 1,)
    assert g[0] == 0.0 and g[-1] == 1.0                 # endpoints pinned exactly
    assert np.all(np.diff(g) > 0)                        # strictly ascending
    assert np.allclose(g, 1.0 - g[::-1])                 # symmetric about 0.5


def test_gll_order1_is_linear_endpoints():
    assert np.array_equal(gll_nodes(1), np.array([0.0, 1.0]))


@pytest.mark.parametrize("order", [1, 2, 4, 6])
def test_gll_weights_positive_and_sum_to_one(order):
    w = gll_weights(order)
    assert w.shape == (order + 1,)
    assert np.all(w > 0)
    assert np.isclose(w.sum(), 1.0)


def test_gll_nodes_cached_identity():
    assert gll_nodes(5) is gll_nodes(5)                  # cache returns same array


def test_gll_nodes_rejects_bad_order():
    with pytest.raises(ValueError):
        gll_nodes(0)


# -- Lagrange basis -----------------------------------------------------
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_lagrange_partition_of_unity(order):
    nodes = gll_nodes(order)
    ev = np.linspace(0, 1, 17)
    M = lagrange_matrix(nodes, ev)
    assert M.shape == (ev.size, nodes.size)
    assert np.allclose(M.sum(axis=1), 1.0)               # partition of unity


def test_lagrange_is_cardinal_on_its_nodes():
    nodes = gll_nodes(4)
    M = lagrange_matrix(nodes, nodes)
    assert np.allclose(M, np.eye(nodes.size))            # L_k(x_j) = delta_kj


def test_lagrange_interpolates_a_polynomial_exactly():
    # a degree-<=order polynomial is reproduced exactly by order-N interpolation
    nodes = gll_nodes(3)

    def f(x):
        return 2 * x**3 - x**2 + 0.5 * x - 3

    ev = np.linspace(0, 1, 11)
    got = lagrange_matrix(nodes, ev) @ f(nodes)
    assert np.allclose(got, f(ev))


# -- Lagrange derivative operator (order-N quality metric) --------------
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_lagrange_derivative_rows_sum_to_zero(order):
    nodes = gll_nodes(order)
    D = lagrange_derivative_matrix(nodes, nodes)
    assert D.shape == (nodes.size, nodes.size)
    assert np.allclose(D.sum(axis=1), 0.0)               # d/dx of partition of unity


def test_lagrange_derivative_differentiates_a_polynomial_exactly():
    # order-N interpolation differentiates a degree-<=order polynomial exactly
    nodes = gll_nodes(4)

    def f(x):
        return 2 * x**4 - x**3 + 0.5 * x - 3

    def df(x):
        return 8 * x**3 - 3 * x**2 + 0.5

    ev = np.linspace(0, 1, 11)
    got = lagrange_derivative_matrix(nodes, ev) @ f(nodes)
    assert np.allclose(got, df(ev))


# -- order-N scaled-Jacobian metric reduces to the corner metric --------
def test_scaled_jacobian_ho_quad_reduces_at_order1():
    from nekmeshpy.quadmesh import quality as qq

    box = QuadMesh.box(1.0, (2, 2, 2))                    # order-1 closed surface
    corner = qq.scaled_jacobian(box.points, box.quads)
    ho = scaled_jacobian_ho(box.curved, box.order, dim=2)
    assert np.allclose(corner, ho, atol=1e-12)


def test_scaled_jacobian_ho_hex_reduces_at_order1():
    from nekmeshpy.hexmesh import quality as hq
    from nekmeshpy.model.fields import uniform_spacing

    loop = LineMesh.circle(1.0, 24)
    qm = QuadMesh.ogrid(loop, n_side=6, radial=uniform_spacing(4),
                        smoothing_method="bilinear")
    blk = HexMesh.extrude(qm, axis=(0, 0, 1), length=5.0, layers=uniform_spacing(6))
    corner = hq.scaled_jacobian(blk.points, blk.hexes)
    ho = scaled_jacobian_ho(blk.curved, blk.order, dim=3)
    assert np.allclose(corner, ho, atol=1e-12)


# -- tensor lattice + corner indexing -----------------------------------
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_tensor_nodes_shape_and_range(dim, order):
    t = tensor_nodes(order, dim)
    assert t.shape == (nodes_per_element(order, dim), dim)
    assert t.min() == 0.0 and t.max() == 1.0


def test_tensor_nodes_i_fastest():
    # dim=2, order=1: i (axis 0) varies fastest -> (0,0),(1,0),(0,1),(1,1)
    t = tensor_nodes(1, 2)
    assert np.array_equal(t, [[0, 0], [1, 0], [0, 1], [1, 1]])


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_corner_indices_pick_out_corners(dim, order):
    idx = corner_indices(order, dim)
    assert idx.shape == (2**dim,)
    lattice = tensor_nodes(order, dim)
    # every selected lattice node is a genuine corner (all coords in {0,1})
    picked = lattice[idx]
    assert np.all((picked == 0.0) | (picked == 1.0))
    assert len({tuple(row) for row in picked}) == 2**dim   # all distinct corners


# -- straight subdivision reduces at N=1 --------------------------------
def test_subdivide_element_line_reduces_at_order1():
    corners = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
    block = subdivide_element(corners, 1, 1)
    # block[corner_indices] must reproduce the winding-order corners
    assert np.allclose(block[corner_indices(1, 1)], corners)


def test_subdivide_element_quad_reduces_at_order1():
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    block = subdivide_element(corners, 1, 2)
    assert block.shape == (4, 3)
    assert np.allclose(block[corner_indices(1, 2)], corners)


def test_subdivide_element_hex_reduces_at_order1():
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    block = subdivide_element(corners, 1, 3)
    assert block.shape == (8, 3)
    assert np.allclose(block[corner_indices(1, 3)], corners)


def test_subdivide_element_midpoints_lie_on_straight_edges():
    corners = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
    block = subdivide_element(corners, 2, 1)             # order 2: a midpoint
    assert np.allclose(block[1], [1, 0, 0])             # centre of the segment


# -- coons_grid reduces to the current linear structured algebra --------
def test_coons_grid_reduces_to_bilinear_at_order1():
    # a planar unit square via its four edges sampled at u=v={0,1}
    u = np.array([0.0, 1.0])
    v = np.array([0.0, 1.0])
    cb = np.array([[0, 0, 0], [1, 0, 0]], float)         # c0->c1
    ct = np.array([[0, 1, 0], [1, 1, 0]], float)         # c3->c2
    cl = np.array([[0, 0, 0], [0, 1, 0]], float)         # c0->c3
    cr = np.array([[1, 0, 0], [1, 1, 0]], float)         # c1->c2
    grid = coons_grid(cb, ct, cl, cr, u, v)
    assert grid.shape == (2, 2, 3)
    assert np.allclose(grid[0, 0], [0, 0, 0])
    assert np.allclose(grid[1, 0], [1, 0, 0])
    assert np.allclose(grid[0, 1], [0, 1, 0])
    assert np.allclose(grid[1, 1], [1, 1, 0])


# -- blend_ho -----------------------------------------------------------
def test_blend_ho_endpoints_and_midpoint():
    a = np.zeros((3, 4, 3))
    b = np.ones((3, 4, 3))
    assert np.allclose(blend_ho(a, b, 0.0), a)
    assert np.allclose(blend_ho(a, b, 1.0), b)
    assert np.allclose(blend_ho(a, b, 0.5), 0.5)


# -- container plumbing / validation ------------------------------------
def test_default_order_is_one_and_curved_materialized():
    # order 1 now always materializes the 2^d corner block (never None),
    # corner-consistent with points[conn].
    lm = LineMesh.open([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert lm.order == 1 and lm.curved.shape == (2, 2, 3)
    assert np.allclose(lm.curved, lm.points[lm.lines])
    qm = QuadMesh.from_grid(_unit_grid())
    assert qm.order == 1 and qm.curved.shape == (qm.n_quads, 4, 3)
    assert np.allclose(qm.curved[:, corner_indices(1, 2), :], qm.points[qm.quads])
    hm = HexMesh.from_grid(_unit_hex_grid())
    assert hm.order == 1 and hm.curved.shape == (hm.n_hexes, 8, 3)
    assert np.allclose(hm.curved[:, corner_indices(1, 3), :], hm.points[hm.hexes])


def test_factory_meshes_default_to_order_one():
    circ = LineMesh.circle(1.0, 8)
    assert circ.order == 1 and circ.curved.shape == (8, 2, 3)
    assert np.allclose(circ.curved, circ.points[circ.lines])
    og = QuadMesh.ogrid(circ, 2, np.array([0.0, 0.5, 1.0]))
    assert og.order == 1 and og.curved.shape == (og.n_quads, 4, 3)


# The factories don't forward order/curved yet (Phase 2+); the constructor is the
# Phase 1 plumbing site, so these drive LineMesh(...) directly.
def test_curved_at_order1_accepted_when_corner_consistent():
    # order 1 with a supplied corner block is now validated (not rejected): a
    # corner-consistent block is accepted...
    good = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=float)
    lm = LineMesh([[0, 0, 0], [1, 0, 0]], order=1, curved=good)
    assert lm.order == 1 and np.allclose(lm.curved, good)
    # ...but corners that disagree with points[lines] are rejected.
    with pytest.raises(ValueError, match="corners disagree"):
        LineMesh([[0, 0, 0], [1, 0, 0]], order=1, curved=np.zeros((1, 2, 3)))


def test_order_gt1_requires_curved():
    with pytest.raises(ValueError, match="requires a curved block"):
        LineMesh([[0, 0, 0], [1, 0, 0]], order=3)


def test_curved_wrong_shape_rejected():
    with pytest.raises(ValueError, match=r"\(1,3,3\)"):
        LineMesh([[0, 0, 0], [1, 0, 0]], order=2, curved=np.zeros((1, 5, 3)))


def test_curved_corner_mismatch_rejected():
    # order-2 line, but the endpoint nodes don't match points[lines]
    bad = np.array([[[0, 0, 0], [0.5, 0, 0], [9, 9, 9]]], dtype=float)
    with pytest.raises(ValueError, match="corners disagree"):
        LineMesh([[0, 0, 0], [1, 0, 0]], order=2, curved=bad)


def test_valid_curved_line_accepted():
    # order-2 line with a straight interior midpoint: corner-consistent
    curved = subdivide_element(np.array([[0, 0, 0], [1, 0, 0]], float), 2, 1)
    lm = LineMesh([[0, 0, 0], [1, 0, 0]], order=2, curved=curved[None])
    assert lm.order == 2
    assert lm.curved is not None and lm.curved.shape == (1, 3, 3)


# -- helpers ------------------------------------------------------------
def _unit_grid():
    from nekmeshpy.model.fields import uniform_spacing  # noqa: F401
    xs = np.linspace(0, 1, 3)
    ys = np.linspace(0, 1, 3)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx, gy, np.zeros_like(gx)], axis=-1)


def _unit_hex_grid():
    xs = np.linspace(0, 1, 3)
    g = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.stack([g[0], g[1], g[2]], axis=-1)
