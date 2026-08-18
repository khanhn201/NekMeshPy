"""Shared order-N interpolation kernel (tensor-product Lagrange on GLL nodes)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from .fields import gll_nodes, lagrange_derivative_matrix, lagrange_matrix

# Corner parametric tuples (0 = node 0, 1 = node N along that axis) in the
# connectivity winding order of each container: line [v0,v1], quad CCW, hex Nek
# (bottom quad CCW, then top quad CCW) -- matches QuadMesh.EDGE_POINTS /
# HexMesh.from_grid corner order.
_CORNER_IJK: dict[int, list[tuple[int, ...]]] = {
    1: [(0,), (1,)],
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
}


def nodes_per_element(order: int, dim: int) -> int:
    """``(order+1)**dim`` -- the high-order node count for a ``dim``-D element."""
    return (order + 1) ** dim


def tensor_nodes(order: int, dim: int) -> FloatArray:
    """The ``((order+1)**dim, dim)`` reference lattice of GLL nodes on ``[0,1]^dim``,
    in lexicographic order with axis 0 (``i``) varying fastest."""
    if dim not in (1, 2, 3):
        raise ValueError("dim must be 1, 2 or 3")
    g = gll_nodes(order)
    grids = np.meshgrid(*([g] * dim), indexing="ij")
    return np.stack([G.ravel(order="F") for G in grids], axis=1)


def corner_indices(order: int, dim: int) -> IntArray:
    """Lexicographic indices of the ``2**dim`` element corners, in connectivity
    winding order -- so ``block[corner_indices(N,d)]`` is the corner sub-slice that
    must equal ``points[conn]``."""
    if dim not in _CORNER_IJK:
        raise ValueError("dim must be 1, 2 or 3")
    n = order
    strides = [(n + 1) ** a for a in range(dim)]
    idx = [sum((n if b else 0) * strides[a] for a, b in enumerate(bits))
           for bits in _CORNER_IJK[dim]]
    return np.array(idx, dtype=np.int64)


def subdivide_element(corners: PointArray, order: int, dim: int) -> PointArray:
    """Straight-sided order-N block: multilinear map of the reference lattice through
    the ``2**dim`` ``corners`` (given in connectivity winding order). Returns
    ``((order+1)**dim, 3)`` in lexicographic order."""
    c = np.asarray(corners, dtype=float).reshape(-1, 3)
    if c.shape[0] != 2 ** dim:
        raise ValueError("subdivide_element needs %d corners for dim %d, got %d"
                         % (2 ** dim, dim, c.shape[0]))
    params = tensor_nodes(order, dim)                 # (M, dim) in [0,1]
    m = params.shape[0]
    out: PointArray = np.zeros((m, 3))
    for ci, bits in enumerate(_CORNER_IJK[dim]):
        w = np.ones(m)
        for a in range(dim):
            u = params[:, a]
            w *= u if bits[a] else (1.0 - u)
        out += w[:, None] * c[ci]
    return out


def hex_face_indices(face: int, order: int) -> IntArray:
    """Lexicographic (``i`` fastest) block indices of the ``(order+1)**2`` nodes on
    Nek hex ``face`` (1-6): faces 1/3 are the ``j=0``/``j=n`` planes, 2/4 the
    ``i=n``/``i=0`` planes, 5/6 the ``k=0``/``k=n`` planes (matching
    ``HexMesh.FACE_POINTS``).  Used to tag a whole boundary face's nodes."""
    n = order
    row = n + 1
    m: IntArray = np.arange(row ** 3, dtype=np.int64)     # lexicographic index == value
    im = m % row
    jm = (m // row) % row
    km = m // (row * row)
    planes = {1: jm == 0, 2: im == n, 3: jm == n,
              4: im == 0, 5: km == 0, 6: km == n}
    if face not in planes:
        raise ValueError("hex face must be 1-6, got %d" % face)
    return m[planes[face]]


#: Hex local edges as corner-index pairs into ``_CORNER_IJK[3]`` -- the 12 edges in
#: Nek order (bottom quad, top quad, then the four verticals), matching the ``he``
#: table used by :mod:`nekmeshpy.hexmesh.smoothing`.
_HEX_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)]


def hex_edge_indices(edge: int, order: int) -> IntArray:
    """Lexicographic (``i`` fastest) block indices of the ``order+1`` nodes along hex
    ``edge`` (0-11, in :data:`_HEX_EDGES` order), ordered start-corner -> end-corner."""
    if not 0 <= edge < 12:
        raise ValueError("hex edge must be 0-11, got %d" % edge)
    n = order
    row = n + 1
    strides = (1, row, row * row)
    ca, cb = _HEX_EDGES[edge]
    a = tuple(v * n for v in _CORNER_IJK[3][ca])          # start corner (0/n per axis)
    b = tuple(v * n for v in _CORNER_IJK[3][cb])          # end corner
    ax = next(k for k in range(3) if a[k] != b[k])        # the varying axis
    base = sum(a[k] * strides[k] for k in range(3) if k != ax)
    seq = (np.arange(n + 1) if a[ax] < b[ax]
           else np.arange(n, -1, -1))                     # start -> end direction
    return (base + seq * strides[ax]).astype(np.int64)


def quad_edge_indices(side: int, order: int) -> IntArray:
    """Lexicographic (``i`` fastest) block indices of the ``order+1`` nodes along quad
    ``side`` (1-4), ordered start-corner -> end-corner to match ``QuadMesh.EDGE_POINTS``
    (side 1 ``v0->v1``, 2 ``v1->v2``, 3 ``v2->v3``, 4 ``v3->v0``)."""
    n = order
    row = n + 1
    if side == 1:
        return np.arange(row, dtype=np.int64)                    # j=0, i 0..n
    if side == 2:
        return (n + row * np.arange(row)).astype(np.int64)       # i=n, j 0..n
    if side == 3:
        return (row * n + np.arange(n, -1, -1)).astype(np.int64)  # j=n, i n..0
    if side == 4:
        return (row * np.arange(n, -1, -1)).astype(np.int64)     # i=0, j n..0
    raise ValueError("quad side must be 1-4, got %d" % side)


def coons_grid(cb: PointArray, ct: PointArray, cl: PointArray, cr: PointArray,
               u: FloatArray, v: FloatArray) -> PointArray:
    """Transfinite (Coons-patch) blend, factored out of ``QuadMesh.structured``."""
    uu = np.asarray(u, dtype=float).reshape(-1, 1, 1)
    vv = np.asarray(v, dtype=float).reshape(1, -1, 1)
    P00, P10, P01, P11 = cb[0], cb[-1], ct[0], ct[-1]
    return ((1 - vv) * cb[:, None, :] + vv * ct[:, None, :]
            + (1 - uu) * cl[None, :, :] + uu * cr[None, :, :]
            - ((1 - uu) * (1 - vv) * P00 + uu * (1 - vv) * P10
               + (1 - uu) * vv * P01 + uu * vv * P11))


def coons_grid_fn(
    cb: Callable[[FloatArray], PointArray], ct: Callable[[FloatArray], PointArray],
    cl: Callable[[FloatArray], PointArray], cr: Callable[[FloatArray], PointArray],
) -> Callable[[FloatArray, FloatArray], PointArray]:
    """The continuous form of :func:`coons_grid`: ``cb``/``ct``/``cl``/``cr`` are
    boundary *functions* rather than pre-sampled arrays, evaluable at any node lattice
    rather than only where the boundaries happen to already be meshed."""
    def f(x: FloatArray, y: FloatArray) -> PointArray:
        xa = np.asarray(x, dtype=float)
        y0 = np.array([float(np.ravel(np.asarray(y, dtype=float))[0])])
        return coons_grid(cb(xa), ct(xa), cl(y0), cr(y0), xa, y0)[:, 0, :]

    return f


def blend_ho(a: PointArray, b: PointArray, t: float) -> PointArray:
    """``(1-t)*a + t*b`` on two index-paired high-order blocks."""
    return (1.0 - t) * a + t * b


#: Sampled points per batch in :func:`sampled_scaled_jacobian`. Peak allocation is
#: measured at **0.176 MB per thousand sampled points** -- not the 0.024 the resampled
#: block alone would suggest, because the kernel holds about eight arrays that size at
#: once (the three parametric tangents, their norms, and the cross product). So this is
#: roughly a 60 MB working set, flat in both mesh size and sampling order.
#:
#: Sized at the knee: batching 8x smaller costs 18% more time, 8x larger buys 3%.
#: Element-by-element would hold nothing at all and take **25x longer** -- 67 s against
#: 2.7 s on 8k hexes -- because numpy's per-call dispatch dwarfs the arithmetic on one
#: element's worth of nodes.
CHUNK_POINTS = 350_000


def sampled_scaled_jacobian(curved: PointArray, order: int, new_order: int,
                            dim: int) -> FloatArray:
    """:func:`scaled_jacobian` of an order-``order`` block read on the ``new_order``
    lattice, in chunks.

    Only the per-element minimum survives the resampling, so materialising the whole
    finer block is pure peak memory -- 500k hexes at order 8 is 8.7 GB for the block
    alone, and the kernel wants eight such arrays. The batch is sized in *sampled
    points* rather than elements, so it shrinks automatically as the order climbs and
    the working set stays flat in both mesh size and order -- see
    :data:`CHUNK_POINTS`."""
    if new_order == order:
        return scaled_jacobian(curved, order, dim)
    per = max(1, CHUNK_POINTS // (new_order + 1) ** dim)
    n = curved.shape[0]
    out: FloatArray = np.empty(n, dtype=float)
    for lo in range(0, n, per):
        hi = min(lo + per, n)
        out[lo:hi] = scaled_jacobian(
            resample_block(curved[lo:hi], order, new_order, dim), new_order, dim)
    return out


def resample_block(curved: PointArray, order: int, new_order: int,
                   dim: int) -> PointArray:
    """``(E,(order+1)**dim,3)`` element blocks re-read on the ``new_order`` GLL lattice.

    A change of nodal basis, **not** a refinement: the polynomial map each element
    already carries is evaluated at more points, so no geometry is invented and
    ``new_order == order`` is the identity. What it buys is the ability to ask what a
    solver running at *its* polynomial order will compute from this mesh -- see
    :func:`scaled_jacobian`, whose sampling is the whole reason the question has an
    answer other than "the same thing".

    Downward is refused. Reading an order-4 map on an order-2 lattice would throw away
    the very nodes that make it curved and report a mesh that is not the one stored --
    and it would do so by returning a *better* number, which is the wrong direction for
    a check to be wrong in."""
    if new_order < order:
        raise ValueError(
            "resample_block: cannot read an order-%d block on an order-%d lattice; "
            "sampling below the stored order hides the nodes it drops" % (order, new_order))
    if new_order == order:
        return curved
    a = lagrange_matrix(gll_nodes(order), gll_nodes(new_order))   # (new+1, order+1)
    e = curved.shape[0]
    b = curved.reshape((e,) + (order + 1,) * dim + (3,))
    for axis in range(1, dim + 1):
        b = np.moveaxis(np.tensordot(a, b, axes=([1], [axis])), 0, axis)
    out: PointArray = np.ascontiguousarray(b).reshape(e, (new_order + 1) ** dim, 3)
    return out


def _element_tangents(curved: PointArray, order: int,
                      dim: int) -> tuple[FloatArray, ...]:
    """The ``dim`` parametric tangent vectors of each element's mapping, evaluated at
    every one of its ``(order+1)**dim`` GLL nodes."""
    g = gll_nodes(order)
    d1 = lagrange_derivative_matrix(g, g)             # (row,row): d1[a,i]=L'_i(g[a])
    row = order + 1
    e = curved.shape[0]
    if dim == 1:
        t = np.einsum("mi,eic->emc", d1, curved)
        return (t,)
    if dim == 2:
        b = curved.reshape(e, row, row, 3)            # axes (j, i)
        t_i = np.einsum("mi,ejid->ejmd", d1, b).reshape(e, row * row, 3)
        t_j = np.einsum("mj,ejid->emid", d1, b).reshape(e, row * row, 3)
        return t_i, t_j
    if dim == 3:
        b = curved.reshape(e, row, row, row, 3)       # axes (k, j, i)
        t_i = np.einsum("mi,ekjid->ekjmd", d1, b).reshape(e, row ** 3, 3)
        t_j = np.einsum("mj,ekjid->ekmid", d1, b).reshape(e, row ** 3, 3)
        t_k = np.einsum("mk,ekjid->emjid", d1, b).reshape(e, row ** 3, 3)
        return t_i, t_j, t_k
    raise ValueError("_element_tangents supports dim 1, 2 or 3, got %d" % dim)


def scaled_jacobian(curved: PointArray, order: int, dim: int) -> FloatArray:
    """Per-element minimum scaled Jacobian sampled at the ``(order+1)**dim`` GLL nodes
    of a ``curved`` block, shape ``(E,)``.

    **Sampled**, and that word is load-bearing: the value is exact at those nodes and
    says nothing between them. An element's map is degree ``order`` per direction, but
    its Jacobian determinant is a polynomial of much higher degree, so a positive
    reading at the nodes is not a certificate that the element is not folded -- a
    triquadratic hex reading ``+0.35`` on its own 27 nodes has been measured at
    ``-0.12`` on 729. Whatever order the *solver* will run at is the order to ask for;
    :func:`resample_block` is how."""
    tang = _element_tangents(curved, order, dim)
    if dim == 3:
        ti, tj, tk = tang
        num = np.sum(np.cross(ti, tj) * tk, axis=2)                   # (E,M)
        L = (np.linalg.norm(ti, axis=2) * np.linalg.norm(tj, axis=2)
             * np.linalg.norm(tk, axis=2))
        sj = np.divide(num, L, out=np.zeros_like(num), where=L > 0)
        return np.min(sj, axis=1)
    ti, tj = tang
    cross = np.cross(ti, tj)                                          # (E,M,3)
    nref = np.sum(cross, axis=1)                                      # (E,3)
    nmag = np.linalg.norm(nref, axis=1)                              # (E,)
    good = nmag > 0
    nu = np.divide(nref, nmag[:, None], out=np.zeros_like(nref), where=good[:, None])
    num = np.sum(cross * nu[:, None, :], axis=2)                      # (E,M)
    L = np.linalg.norm(ti, axis=2) * np.linalg.norm(tj, axis=2)
    sj = np.divide(num, L, out=np.zeros_like(num), where=L > 0)
    return np.where(good, np.min(sj, axis=1), 0.0)
