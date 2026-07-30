"""Shared order-N interpolation kernel (tensor-product Lagrange on GLL nodes).

Order-N geometry is stored as an entity B-rep (see :mod:`nekmeshpy.model.conform`);
where a *per-element* view is needed -- the ``(E,(N+1)**d,3)`` node block in
**lexicographic (i,j,k) order with i fastest** at the
:func:`~nekmeshpy.model.fields.gll_nodes` reference positions -- it is gathered
transiently from that B-rep.  This module holds the dimension-general primitives that
build and combine such blocks, so the order-N logic lives in one place and each
factory contributes only its geometry map:

* :func:`tensor_nodes` -- the ``(N+1)**d`` reference lattice.
* :func:`corner_indices` -- the ``2**d`` corner slots in the element's connectivity
  winding order (line ``[v0,v1]``, quad CCW, hex Nek).  Corners are owned by
  ``points[conn]``; the non-corner nodes are decomposed into topological entities
  (shared edges / faces + private interiors) by :mod:`nekmeshpy.model.conform`.
* :func:`subdivide_element` -- straight multilinear subdivision through the corners
  (the "elevate" atom for straight-sided factories).
* :func:`quad_edge_indices` / :func:`hex_edge_indices` -- the block slots along a quad
  side / hex edge, for overlaying a true boundary curve and for entity decomposition.
* :func:`hex_face_indices` -- the block slots on a hex face, for tagging boundary nodes.
* :func:`coons_grid` -- the transfinite (Coons) blend factored out of ``structured``.
* :func:`blend_ho` -- ``(1-t)A + tB`` on blocks.
* :func:`scaled_jacobian_ho` -- the order-N scaled-Jacobian quality metric, sampled at
  a block's GLL nodes (reduces to the corner metric at order 1).

At ``order == 1`` every primitive reduces to the linear corner data, which is what
keeps the golden byte-identical.
"""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from .fields import gll_nodes, lagrange_derivative_matrix

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
    the ``2**dim`` ``corners`` (given in connectivity winding order).  Returns
    ``((order+1)**dim, 3)`` in lexicographic order.  At ``order == 1`` the block is
    the corners re-expressed in lexicographic order."""
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
    ``edge`` (0-11, in :data:`_HEX_EDGES` order), ordered start-corner -> end-corner.

    The dim-3 sibling of :func:`quad_edge_indices`: each edge varies one reference axis
    with the other two pinned; the sequence runs from the first corner tuple to the
    second (descending when the first corner sits at node ``n``), so overlaying an
    ordered curve onto a shared edge is orientation-consistent."""
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
    """Lexicographic (``i`` fastest) block indices of the ``order+1`` nodes along
    quad ``side`` (1-4), ordered start-corner -> end-corner to match
    ``QuadMesh.EDGE_POINTS`` (side 1 ``v0->v1``, 2 ``v1->v2``, 3 ``v2->v3``, 4
    ``v3->v0``).  Used to overlay a true boundary curve onto a straight block."""
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
    """Transfinite (Coons-patch) blend, factored out of ``QuadMesh.structured``.

    ``cb``/``ct`` are the bottom (``c0->c1``) / top (``c3->c2``) edge points sampled
    at parameters ``u``; ``cl``/``cr`` the left (``c0->c3``) / right (``c1->c2``)
    edges sampled at ``v``.  Returns the ``(len(u), len(v), 3)`` interior.  The blend
    weights ``u``/``v`` must be the parameters the edges were sampled at (uniform for
    a linear grid, GLL-refined for order N) -- the current linear ``structured`` is
    the ``u = v = linspace`` special case, so this reproduces it exactly."""
    uu = np.asarray(u, dtype=float).reshape(-1, 1, 1)
    vv = np.asarray(v, dtype=float).reshape(1, -1, 1)
    P00, P10, P01, P11 = cb[0], cb[-1], ct[0], ct[-1]
    return ((1 - vv) * cb[:, None, :] + vv * ct[:, None, :]
            + (1 - uu) * cl[None, :, :] + uu * cr[None, :, :]
            - ((1 - uu) * (1 - vv) * P00 + uu * (1 - vv) * P10
               + (1 - uu) * vv * P01 + uu * vv * P11))


def blend_ho(a: PointArray, b: PointArray, t: float) -> PointArray:
    """``(1-t)*a + t*b`` on two index-paired high-order blocks."""
    return (1.0 - t) * a + t * b


def _element_tangents(curved: PointArray, order: int,
                      dim: int) -> tuple[FloatArray, ...]:
    """The ``dim`` parametric tangent vectors of each element's mapping, evaluated at
    every one of its ``(order+1)**dim`` GLL nodes.  Returns ``dim`` arrays of shape
    ``(E, M, 3)`` (``M = (order+1)**dim``), one per reference axis, formed by
    contracting the 1-D Lagrange derivative operator along that axis of the curved
    block (a tensor-product interpolant differentiated node-by-node)."""
    g = gll_nodes(order)
    d1 = lagrange_derivative_matrix(g, g)             # (row,row): d1[a,i]=L'_i(g[a])
    row = order + 1
    e = curved.shape[0]
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
    raise ValueError("scaled-Jacobian metric supports dim 2 or 3, got %d" % dim)


def scaled_jacobian_ho(curved: PointArray, order: int, dim: int) -> FloatArray:
    """Per-element minimum scaled Jacobian sampled at the ``(order+1)**dim`` GLL nodes
    of a ``curved`` block, shape ``(E,)``.

    The scaled Jacobian at a node is ``det(J) / prod(|tangent|)`` from the mapping's
    parametric tangents there: for ``dim == 3`` a true ``(ti x tj) . tk`` over the
    tangent-length product; for ``dim == 2`` the surface form ``(ti x tj) . n_hat``
    signed against the element's mean normal (so folded nodes read negative on
    non-planar quads).  ``1`` is a perfect right-angled node, ``<= 0`` degenerate /
    inverted.  At ``order == 1`` the GLL nodes are the corners and this reduces to the
    corner metric in :func:`nekmeshpy.quadmesh.quality.scaled_jacobian` /
    :func:`nekmeshpy.hexmesh.quality.scaled_jacobian`."""
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
