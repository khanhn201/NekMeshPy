"""Per-element node blocks, as arrays: the rung-agnostic half of smoothing a curved (order ``N``) mesh.

A curved mesh stores its nodes once per *entity* (a shared vertex, a shared edge's interior, a shared face's
interior, a private element interior); the smoother thinks in per-element **blocks** ``(E, (N+1)**dim, 3)`` in
nekmeshpy's lexicographic Gauss-Lobatto order (axis 0 fastest).  A rung turns a mesh into blocks with its own
``element_blocks`` and back with its own ``with_blocks``; this module holds what is the same at every rung:

* :func:`scatter_blocks` -- blocks back onto the flat node table, through an id table that the rung gets by
  running its own ``element_blocks`` on **node ids** instead of coordinates (so every orientation convention of
  the storage is reused, none re-derived, and no coordinate search is made);
* :func:`carry_blocks` -- the curved nodes follow a displacement of the element's corners, interpolated
  multilinearly, so a shared edge or face (which sees only its own corners) stays conformal;
* :func:`node_fixed` -- which nodes inherit a vertex constraint: a node on an entity is held in a component
  when *every* vertex of that entity holds it, and an element's own interior is never held;
* :func:`min_det_blocks` -- the Jacobian determinant of a block on a Gauss-Lobatto lattice (the quadrature
  points of the TMOP run).
"""

from __future__ import annotations

import numpy as np

from ..._typing import BoolArray, FloatArray, IntArray
from ..fields import gll_nodes, lagrange_derivative_matrix, lagrange_matrix
from ..interp import corner_indices, subdivide_element

__all__ = ["lattice_weights", "carry_blocks", "scatter_blocks", "node_fixed", "min_det_blocks"]


def lattice_weights(order: int, dim: int) -> FloatArray:
    """``((order+1)**dim, 2**dim)`` multilinear weights of an element's corners at its Gauss-Lobatto lattice
    (corners in the rung's connectivity order)."""
    W: FloatArray = np.empty(((order + 1) ** dim, 2 ** dim))
    for j in range(2 ** dim):
        c = np.zeros((2 ** dim, 3))
        c[j, 0] = 1.0
        W[:, j] = subdivide_element(c, order, dim)[:, 0]
    return W


def carry_blocks(blocks: FloatArray, displacement: FloatArray, order: int, dim: int) -> FloatArray:
    """``blocks`` ``(E, M, 3)`` moved by the corner ``displacement`` ``(E, 2**dim, 3)``, interpolated
    multilinearly to every node: ``x_new = x_old + sum_c w_c(x) d_c``.  A node on an edge or face sees only that
    entity's corners (the other weights are exactly zero), so two elements sharing it agree; an element whose
    corners did not move is returned bit for bit."""
    W = lattice_weights(order, dim)
    out: FloatArray = blocks + np.einsum("mc,eck->emk", W, displacement)
    still = ~displacement.any(axis=(1, 2))
    out[still] = blocks[still]
    return out


def scatter_blocks(ids: IntArray, blocks: FloatArray, flat: FloatArray) -> FloatArray:
    """The flat node table ``flat`` ``(n, 3)`` with ``blocks`` ``(E, M, 3)`` written at ``ids`` ``(E, M)``.  A
    node shared by several elements is written by each; the blocks of a conformal mesh agree (to rounding), and the
    last writer wins.  Nodes no element references keep their value in ``flat``.  A copy."""
    out = np.array(flat, copy=True)
    out[ids.reshape(-1)] = blocks.reshape(-1, 3)
    return out


def node_fixed(vertex_fixed: BoolArray, cells: IntArray, order: int, dim: int) -> BoolArray:
    """``(E, M, k)`` the components held at every node of every block, from ``vertex_fixed`` ``(P, k)`` (the
    components held at each vertex) and the corner ids ``cells`` ``(E, 2**dim)``.

    A node lies on one lowest-dimensional entity of its element -- a vertex, an edge, a face, or the interior.
    It is held in a component when every vertex of that entity holds it; an element's interior is held in a
    component only when *all* its corners do (a frozen element stays as it is), so it is free otherwise.  The entity is read from the lattice index: an index at an end (``0`` or ``order``) pins that
    axis, and the entity's vertices are the corners agreeing on every pinned axis."""
    row = order + 1
    m = np.arange(row ** dim)
    idx = np.stack([(m // row ** a) % row for a in range(dim)], axis=1)       # (M, dim) lattice index, axis 0 fastest
    pinned = (idx == 0) | (idx == order)                              # (M, dim)
    bit = idx == order                                                # which end, where pinned
    cb = bit[corner_indices(order, dim)]                              # (2**dim, dim) the corners' own ends
    on = np.all(~pinned[:, None, :] | (cb[None, :, :] == bit[:, None, :]), axis=2)       # (M, 2**dim)
    vm = vertex_fixed[cells]                                          # (E, 2**dim, k)
    open_ = ~vm[:, None, :, :] & on[None, :, :, None]                 # a vertex of the entity that is not held
    held = ~open_.any(axis=2)                                         # (E, M, k)
    inside = ~pinned.any(axis=1)                                      # the element's own interior: no entity of its own,
    held[:, inside, :] = vertex_fixed[cells].all(axis=1)[:, None, :]    # held only where all its corners hold (a frozen element)
    return held


def min_det_blocks(coords: FloatArray, order: int, dim: int, npts: int) -> FloatArray:
    """``(E,)`` the smallest Jacobian determinant of each block over the ``npts``-per-axis Gauss-Lobatto lattice.

    ``coords`` ``(E, M, dim)`` are the node coordinates in a frame of the element's own dimension; the map is the
    order-``order`` tensor Lagrange interpolant, differentiated analytically, the determinant taken with respect
    to the reference cube ``[0, 1]**dim``."""
    g = gll_nodes(order)
    t = gll_nodes(npts - 1) if npts > 1 else np.array([0.5])
    A = lagrange_matrix(g, t)
    D = lagrange_derivative_matrix(g, t)
    e = coords.shape[0]
    row = order + 1
    if dim == 2:
        b = coords.reshape(e, row, row, 2)                            # axes (j, i)
        xi = np.einsum("pj,qi,ejic->epqc", A, D, b)
        xj = np.einsum("pj,qi,ejic->epqc", D, A, b)
        det = xi[..., 0] * xj[..., 1] - xi[..., 1] * xj[..., 0]
    elif dim == 3:
        b = coords.reshape(e, row, row, row, 3)                       # axes (k, j, i)
        xi = np.einsum("rk,pj,qi,ekjic->erpqc", A, A, D, b, optimize=True)
        xj = np.einsum("rk,pj,qi,ekjic->erpqc", A, D, A, b, optimize=True)
        xk = np.einsum("rk,pj,qi,ekjic->erpqc", D, A, A, b, optimize=True)
        det = np.einsum("erpqc,erpqc->erpq", xi, np.cross(xj, xk))
    else:
        raise ValueError("min_det_blocks: dim must be 2 or 3, got %d" % dim)
    return np.asarray(det.reshape(e, -1).min(axis=1))
