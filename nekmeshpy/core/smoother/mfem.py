"""The MFEM / TMOP bridge: one smoothing method for quads and hexes, written once.

`TMOP <https://mfem.org/howto/mesh_optimizer/>`_ (the Target-Matrix Optimization Paradigm of MFEM) minimises a
sum over elements of a *metric* of the map from a **target** element to the actual one, with a Newton or
L-BFGS solver.  This module hands a mesh to it as plain arrays and takes the vertex positions back, so the
quad rung and the hex rung each register a three-line wrapper (``("tmop", {...})`` in a schedule) and share
everything else here.

**Optional.** MFEM is imported lazily, inside the functions that need it, so this module (and every
smoother module that registers it) imports without PyMFEM; calling it without raises an
:class:`ImportError` naming the install (``pip install mfem``, the optional ``smooth`` extra).  PyMFEM is a
large binary wheel and wraps a C++ library that *aborts the process* on some invalid input, so the bridge
checks what it can before handing over (:func:`tmop` refuses an inverted input for every metric but the
untangling 313).

What it does, in order:

* **Arrays in.**  ``points`` ``(N, 3)`` and ``cells`` ``(C, 4)`` quads or ``(C, 8)`` hexes in nekmeshpy's
  corner order, which is MFEM's ``SQUARE`` / ``CUBE`` vertex order (:data:`NEK_TO_MFEM` states the permutation
  explicitly, and the tests check that a round trip keeps every Jacobian positive).  TMOP needs the element
  dimension to equal the space dimension, so a quad mesh must be **planar**: its points are rewritten in
  an orthonormal frame of that plane (:func:`plane_frame`) and a quad mesh on a curved 3-D surface is
  refused -- smooth it with the rung's own optimizer.
* **Constraints to essential DOFs** (:func:`fixed_components`), one rule per mobility kind of
  :class:`~nekmeshpy.core.smoother.constraints.Resolved`:

  - ``FROZEN`` -> every component fixed;
  - ``ON_SURFACE`` on a **plane whose normal is a coordinate axis** -> that one component fixed, the vertex
    slides in the plane (the quad rung has no surface constraint to speak of: a vertex "on" the mesh's own
    plane stays free);
  - ``ON_CURVE`` on a **line along a coordinate axis** -> the other components fixed, the vertex slides on the line;
  - **anything else** -- a curved surface or curve, ``TANGENT``, a rigid ``slide_column`` -- is held **fixed**
    (for now: the optimizer cannot yet follow a curved target); the count is in the stage's ``reason``.

  Every constrained vertex is first projected onto its target (:func:`~nekmeshpy.core.smoother.constraints.snap`),
  so a component is fixed at the *target's* coordinate, not wherever the input happened to be.
* **Targets** -- what "good" means per element, as the target Jacobian ``W`` of the metric:

  - ``"input"``: ``W`` is a box with the element's own edge lengths and ideal (orthogonal) skew -- the element is
    asked to become *orthogonal at the size and aspect it already has*, so a graded mesh stays graded
    (MFEM's ``GIVEN_SHAPE_AND_SIZE`` from per-element size and aspect-ratio fields);
  - ``"input_smoothed"``: the same, with each element's edge lengths first relaxed ``sweeps`` times toward its
    face neighbours' (weight ``sweep_weight``), so the *grading itself* is smoothed too;
  - ``"ideal"``: the ideal shape at the element's own size (``IDEAL_SHAPE_GIVEN_SIZE``): wants cubes /
    squares, which inflates a thin layer -- use with ``designed`` and a quality you can afford.

  ``designed`` elements (a boundary layer, a region to leave alone) are instead given **their own input
  Jacobian** as target, so only their shape is asked to stay and a layer keeps its thickness when its two
  faces are held.  An element whose every vertex is fully fixed carries zero weight.
* **Metric**: ``metric`` is a TMOP metric id; the defaults are ``2`` (2-D shape) and ``303`` (3-D shape).  ``80`` and
  ``333`` (shape + size, ``metric_weight``) work; ``313`` untangles (shifted barrier, finite differences, one
  integrator: ``target="ideal"``).
* **Solver**: ``"newton"`` (MINRES, l1-Jacobi) or ``"lbfgs"``.

Order 1 only (the vertices are the nodes); higher-order MFEM meshes are not wired in yet.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from ..._typing import BoolArray, FloatArray, IntArray, PointArray
from .constraints import FROZEN, ON_CURVE, ON_SURFACE, TANGENT, TIED, Resolved, snap
from .quality import history_row
from .schedule import StageLog

__all__ = ["available", "NEK_TO_MFEM", "DEFAULT_METRIC", "Frame", "plane_frame", "FixedMap", "fixed_components",
           "element_extents", "regions", "to_mfem_mesh", "from_mfem_mesh", "min_det", "TmopEnergy", "tmop"]

#: The MFEM vertex order of a quad (``SQUARE``) and a hex (``CUBE``), as nekmeshpy's own corner indices:
#: ``mfem_vertex[k] = corner[NEK_TO_MFEM[dim][k]]``.  It is the identity today (a quad runs 0-1-2-3 cyclic, a
#: hex 0-3 on the bottom face and 4-7 above them); it is a table so that the one place the two libraries'
#: conventions meet is stated, and tested, instead of assumed.
NEK_TO_MFEM: dict[int, IntArray] = {2: np.arange(4, dtype=np.int64), 3: np.arange(8, dtype=np.int64)}

#: Reference coordinates of the corners in MFEM's vertex order (``Geometry::SQUARE`` / ``CUBE``).
_REF: dict[int, IntArray] = {
    2: np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64),
    3: np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                dtype=np.int64)}

# The default TMOP metric per element dimension: the 2-D / 3-D shape (condition-number) metric.
DEFAULT_METRIC: dict[int, int] = {2: 2, 3: 303}

#: Metrics that take a weight in their constructor (``metric_weight``): 80 (shape + size), 333 (3-D shape + size).
_WEIGHTED = {80: "gamma", 333: "gamma"}
_UNTANGLE = 313
_TARGETS = ("input", "input_smoothed", "ideal")
_SOLVERS = ("newton", "lbfgs")
_TOL_AXIS = 1e-9


def available() -> bool:
    """Whether PyMFEM can be imported (nothing is imported to find out)."""
    try:
        return importlib.util.find_spec("mfem") is not None and importlib.util.find_spec("mfem.ser") is not None
    except (ImportError, ValueError):
        return False


def _mfem() -> Any:
    """The serial ``mfem`` module, or a clear :class:`ImportError`."""
    try:
        import mfem.ser as mfem
    except ImportError as err:
        raise ImportError(
            "the 'tmop' smoothing method needs PyMFEM (the Python wrapper of MFEM's TMOP), which is not "
            "installed: run `pip install mfem` (or `pip install nekmeshpy[smooth]`)") from err
    return mfem


def _dim_of(cells: IntArray) -> int:
    k = cells.shape[1]
    if k not in (4, 8):
        raise ValueError("tmop: cells must be (C, 4) quads or (C, 8) hexes, got %s" % (cells.shape,))
    return 2 if k == 4 else 3


# -- the planar frame --------------------------------------------------------------------------------
class Frame(NamedTuple):
    """An orthonormal frame of the mesh's space: ``coords = (points - origin) @ basis`` ``(N, dim)``, and
    ``points = origin + coords @ basis.T`` for a mesh that lies in the frame's span."""

    origin: FloatArray
    basis: FloatArray            # (3, dim)


def plane_frame(points: PointArray, cells: IntArray) -> Frame:
    """The frame the optimizer works in.  A hex mesh keeps the coordinate axes.  A quad mesh must lie in one
    plane (to ``1e-9`` of its extent; otherwise a :class:`ValueError`, since TMOP needs the element dimension
    to equal the space dimension): when that plane is a coordinate plane the frame is its two axes, else the
    plane's principal axes; the second one is flipped if needed so the quads keep positive area."""
    P = np.asarray(points, dtype=float)
    C = np.asarray(cells, dtype=np.int64)
    if _dim_of(C) == 3:
        return Frame(np.zeros(3), np.eye(3))
    used = np.unique(C)
    X = P[used]
    c = X.mean(axis=0)
    ext = float(np.ptp(X, axis=0).max())
    n = np.linalg.svd(X - c, full_matrices=False)[2][-1]
    off = float(np.abs((X - c) @ n).max())
    if ext > 0 and off > 1e-9 * ext:
        raise ValueError(
            "tmop: the quad mesh is not planar (its vertices sit up to %.2e from the best-fit plane, %.1e of "
            "the extent): TMOP needs element dimension == space dimension.  A quad mesh on a curved surface "
            "is smoothed with the rung's own optimizer (`optimize`)" % (off, off / ext))
    k = int(np.argmax(np.abs(n)))
    if abs(n[k]) >= 1.0 - 1e-12:                  # a coordinate plane: keep its axes, and exactly their numbers
        basis = np.eye(3)[:, [a for a in range(3) if a != k]]
    else:
        basis = np.linalg.svd(X - c, full_matrices=False)[2][:2].T
    a = (P[C[:, 1]] - P[C[:, 0]]) @ basis
    b = (P[C[:, 3]] - P[C[:, 0]]) @ basis
    if float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()) < 0.0:
        basis = basis[:, ::-1].copy()
    return Frame(np.zeros(3) if abs(n[k]) >= 1.0 - 1e-12 else c, basis)


# -- constraints -> essential components -------------------------------------------------------------
class FixedMap(NamedTuple):
    """The essential components of a :class:`~nekmeshpy.core.smoother.constraints.Resolved`."""

    #: ``(N, 3)`` the input with every constrained vertex projected onto its target.
    points: PointArray
    #: ``(N, dim)`` ``True`` where the component (in the working frame) is held.
    fixed: BoolArray
    #: One line per rule that applied, e.g. ``"12 vertices slide in axis-aligned planes"``.
    notes: tuple[str, ...]
    #: ``{rule: vertex count}``: ``frozen``, ``plane``, ``line``, ``held_curved``, ``held_other``.
    counts: dict[str, int]


def fixed_components(points: PointArray, r: Resolved, basis: FloatArray | None = None) -> FixedMap:
    """Map ``r`` to essential components in the frame ``basis`` ``(3, dim)`` (default the 3 coordinate axes).

    The rules are in the module docstring.  A target counts as an *axis-aligned plane* when its normal is
    the same unit coordinate axis at every vertex it holds (a sphere or a triangulated surface is not, even where
    one vertex happens to sit on an axis), and an *axis-aligned line* likewise for its tangent."""
    E = np.eye(3) if basis is None else np.asarray(basis, float)
    dim = E.shape[1]
    P = snap(np.asarray(points, float), r, 1.0)
    fixed: BoolArray = np.zeros((P.shape[0], dim), bool)
    counts = {"frozen": 0, "plane": 0, "line": 0, "held_curved": 0, "held_other": 0}
    fixed[r.kind == FROZEN] = True
    counts["frozen"] = int((r.kind == FROZEN).sum())
    other = (r.kind == TANGENT) | (r.kind == TIED)
    fixed[other] = True
    counts["held_other"] = int(other.sum())

    def axis_of_unit(v: FloatArray) -> int:
        w = v @ E                                  # components in the working frame (|w| = 1 for an in-frame vector)
        if float(np.ptp(w, axis=0).max()) > _TOL_AXIS:
            return -1
        k = int(np.argmax(np.abs(w[0])))
        return k if abs(float(w[0, k])) >= 1.0 - _TOL_AXIS else -1

    for t, s in enumerate(r.surfaces):
        m = np.flatnonzero((r.kind == ON_SURFACE) & (r.target == t))
        if not m.size:
            continue
        nrm = s.normal(s.project(P[m]))
        w = nrm @ E
        if float(np.ptp(nrm, axis=0).max()) <= _TOL_AXIS and float(np.linalg.norm(w[0])) < _TOL_AXIS:
            continue                               # the mesh's own plane: no in-plane restriction at all
        k = axis_of_unit(nrm)
        if k >= 0:
            fixed[m, k] = True
            counts["plane"] += int(m.size)
        else:
            fixed[m] = True
            counts["held_curved"] += int(m.size)
    for t, c in enumerate(r.curves):
        m = np.flatnonzero((r.kind == ON_CURVE) & (r.target == t))
        if not m.size:
            continue
        tan = c.tangent(c.project(P[m]))
        k = axis_of_unit(tan)
        if k >= 0:
            keep = np.arange(dim) != k
            fixed[np.ix_(m, np.flatnonzero(keep))] = True
            counts["line"] += int(m.size)
        else:
            fixed[m] = True
            counts["held_curved"] += int(m.size)
    notes = []
    if counts["frozen"]:
        notes.append("%d frozen" % counts["frozen"])
    if counts["plane"]:
        notes.append("%d slide in axis-aligned planes" % counts["plane"])
    if counts["line"]:
        notes.append("%d slide on axis-aligned lines" % counts["line"])
    if counts["held_curved"]:
        notes.append("%d on a curved or non-axis-aligned target held fixed (curved targets not supported yet)" % counts["held_curved"])
    if counts["held_other"]:
        notes.append("%d tangent / slide-column held fixed (not supported yet)" % counts["held_other"])
    return FixedMap(P, fixed, tuple(notes), counts)


# -- element geometry (numpy side) -------------------------------------------------------------------
def _corner_edges(dim: int) -> tuple[IntArray, IntArray]:
    """``(edges (n, 2) corner pairs, axis (n,))`` of the reference cell: corners differing in one coordinate."""
    R = _REF[dim]
    pairs, axes = [], []
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            d = np.flatnonzero(R[i] != R[j])
            if d.size == 1:
                pairs.append((i, j))
                axes.append(int(d[0]))
    return np.array(pairs, dtype=np.int64), np.array(axes, dtype=np.int64)


def _facets(dim: int) -> IntArray:
    """``(n_facets, 2**(dim-1))`` corner indices of the reference cell's facets (edges of a quad, faces of a hex)."""
    R = _REF[dim]
    return np.array([np.flatnonzero(R[:, d] == s) for d in range(dim) for s in (0, 1)], dtype=np.int64)


def element_extents(coords: PointArray, cells: IntArray) -> FloatArray:
    """``(C, dim)`` the mean edge length of each element along each of its reference axes."""
    C = np.asarray(cells, dtype=np.int64)
    dim = _dim_of(C)
    pairs, axes = _corner_edges(dim)
    X = np.asarray(coords, float)
    Cm = C[:, NEK_TO_MFEM[dim]]
    L = np.linalg.norm(X[Cm[:, pairs[:, 0]]] - X[Cm[:, pairs[:, 1]]], axis=2)          # (C, n_edges)
    return np.stack([L[:, axes == a].mean(axis=1) for a in range(dim)], axis=1)


def min_det(coords: PointArray, cells: IntArray, n: int = 3) -> float:
    """The smallest Jacobian determinant of the multilinear elements over an ``n``-per-axis lattice (Gauss-Lobatto-like:
    ends included) -- the check made before MFEM is entered, since a negative determinant makes
    most TMOP metrics abort the process.  ``n = 3`` covers the corners and the midpoints."""
    C = np.asarray(cells, dtype=np.int64)
    dim = _dim_of(C)
    R = _REF[dim].astype(float)
    g = np.linspace(0.0, 1.0, n)
    xi = np.stack(np.meshgrid(*([g] * dim), indexing="ij"), axis=-1).reshape(-1, dim)       # (p, dim)
    # shape functions' gradients at every lattice point: (p, corners, dim)
    f = np.where(R[None, :, :] > 0.5, xi[:, None, :], 1.0 - xi[:, None, :])                # (p, corners, dim)
    W = np.empty((xi.shape[0], R.shape[0], dim))
    for d in range(dim):
        others = np.prod(np.delete(f, d, axis=2), axis=2)
        W[:, :, d] = (2.0 * R[None, :, d] - 1.0) * others
    X = np.asarray(coords, float)[C[:, NEK_TO_MFEM[dim]]]                                   # (C, corners, dim)
    J = np.einsum("ecr,pcd->eprd", X, W)
    return float(np.linalg.det(J).min())


def regions(element_tags: Any, designed: Any = None) -> tuple[IntArray, BoolArray | None]:
    """``(attributes, designed_mask)`` from a mesh's dense element tags (``(C,)`` strings, ``""`` untagged): each
    distinct tag is an integer attribute (1-based, in sorted order), and ``designed`` -- a tag name, a list of
    names, a boolean ``(C,)`` mask or ``None`` -- becomes the mask :func:`tmop` takes.  A name no element
    carries is an error."""
    dense = np.asarray(element_tags)
    _, inv = np.unique(dense, return_inverse=True)
    attr = np.asarray(inv, dtype=np.int64).reshape(-1) + 1
    if designed is None:
        return attr, None
    if isinstance(designed, str) or (not isinstance(designed, np.ndarray) and len(designed) > 0
                                     and all(isinstance(d, str) for d in designed)):
        names = [designed] if isinstance(designed, str) else list(designed)
        missing = [n for n in names if not (dense == n).any()]
        if missing:
            raise ValueError("tmop: no element carries the tag(s) %s" % missing)
        return attr, np.isin(dense, names)
    return attr, np.asarray(designed, dtype=bool) if np.asarray(designed).dtype == bool else np.isin(
        attr, np.asarray(designed, dtype=np.int64))


# -- MFEM meshes -------------------------------------------------------------------------------------
def to_mfem_mesh(coords: PointArray, cells: IntArray, attributes: IntArray | None = None) -> Any:
    """An in-memory ``mfem.Mesh`` of ``coords`` ``(N, dim)`` and ``cells`` (quads or hexes in nekmeshpy corner order),
    vertex ``i`` of the array being vertex ``i`` of the mesh and element ``e`` being element ``e``.  ``attributes``
    ``(C,)`` ints become the element attributes (default ``1``)."""
    mfem = _mfem()
    C = np.asarray(cells, dtype=np.int64)
    dim = _dim_of(C)
    X = np.asarray(coords, float)
    if X.ndim != 2 or X.shape[1] != dim:
        raise ValueError("to_mfem_mesh: coords must be (N, %d) for %d-corner cells, got %s" % (dim, C.shape[1], X.shape))
    attr = np.ones(C.shape[0], dtype=np.int64) if attributes is None else np.asarray(attributes, dtype=np.int64)
    if attr.shape != (C.shape[0],):
        raise ValueError("to_mfem_mesh: attributes must be (%d,), got %s" % (C.shape[0], attr.shape))
    if attr.size and attr.min() < 1:
        raise ValueError("to_mfem_mesh: MFEM attributes are 1-based, got %d" % int(attr.min()))
    mesh = mfem.Mesh(dim, int(X.shape[0]), int(C.shape[0]), 0, dim)
    for p in X:
        mesh.AddVertex(*[float(v) for v in p])
    add = mesh.AddQuad if dim == 2 else mesh.AddHex
    for c, a in zip(C[:, NEK_TO_MFEM[dim]], attr):
        add([int(v) for v in c], int(a))
    mesh.FinalizeTopology()
    mesh.Finalize(False, False)
    return mesh


def from_mfem_mesh(mesh: Any) -> tuple[PointArray, IntArray, IntArray]:
    """``(coords (N, dim), cells in nekmeshpy corner order, attributes)`` read back from an ``mfem.Mesh`` --
    the inverse of :func:`to_mfem_mesh` (the vertex coordinates, not any nodal grid function)."""
    dim = int(mesh.Dimension())
    nv, ne = int(mesh.GetNV()), int(mesh.GetNE())
    X = np.array([np.array(mesh.GetVertexArray(i))[:dim] for i in range(nv)], dtype=float)
    perm = NEK_TO_MFEM[dim]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    cells = np.array([list(mesh.GetElementVertices(e)) for e in range(ne)], dtype=np.int64)[:, inv]
    attr = np.array([mesh.GetAttribute(e) for e in range(ne)], dtype=np.int64)
    return X, cells, attr


# -- the smoothed target field (V1) ------------------------------------------------------------------
def _axis_neighbours(cells: IntArray, dim: int) -> list[tuple[int, int, dict[int, int]]]:
    """``(p, q, axis_map)`` for every pair of elements sharing a facet: ``axis_map[a]`` is the axis of ``q`` that
    continues axis ``a`` of ``p`` across the facet (read off the edges they share, the rest by elimination)."""
    Cm = cells[:, NEK_TO_MFEM[dim]]
    pairs, axes = _corner_edges(dim)
    nC = Cm.shape[0]
    ek = np.sort(Cm[:, pairs], axis=2)                                                     # (C, n_edges, 2)
    big = int(Cm.max()) + 1
    key = ek[..., 0] * big + ek[..., 1]
    _, eid = np.unique(key.ravel(), return_inverse=True)
    eid = eid.reshape(nC, -1)
    fac = _facets(dim)
    fk = np.sort(Cm[:, fac], axis=2).reshape(nC * len(fac), -1)
    order = np.lexsort(fk.T[::-1])
    fs = fk[order]
    same = (fs[1:] == fs[:-1]).all(axis=1)
    pe, qe = order[:-1][same] // len(fac), order[1:][same] // len(fac)
    out = []
    for p, q in zip(pe.tolist(), qe.tolist()):
        ep = {int(eid[p, i]): int(axes[i]) for i in range(len(axes))}
        eq = {int(eid[q, i]): int(axes[i]) for i in range(len(axes))}
        mp = {ep[s]: eq[s] for s in set(ep) & set(eq)}
        if len(mp) != dim - 1:
            continue
        rest_p = set(range(dim)) - set(mp)
        rest_q = set(range(dim)) - set(mp.values())
        mp[rest_p.pop()] = rest_q.pop()
        out.append((p, q, mp))
    return out


def _smooth_extents(ext: FloatArray, cells: IntArray, free: BoolArray, sweeps: int, weight: float) -> FloatArray:
    """``ext`` with the extents of the ``free`` elements relaxed ``sweeps`` times toward the mean of their face
    neighbours' (matched axis to axis); elements that are not ``free`` are neither changed nor averaged in."""
    dim = ext.shape[1]
    nbr: list[list[tuple[int, dict[int, int]]]] = [[] for _ in range(ext.shape[0])]
    for p, q, mp in _axis_neighbours(cells, dim):
        if free[p] and free[q]:
            nbr[p].append((q, mp))
            nbr[q].append((p, {v: k for k, v in mp.items()}))
    E = ext.copy()
    todo = [e for e in np.flatnonzero(free).tolist() if nbr[e]]
    for _ in range(sweeps):
        En = E.copy()
        for e in todo:
            for a in range(dim):
                En[e, a] = (1.0 - weight) * E[e, a] + weight * float(np.mean([E[q, mp[a]] for q, mp in nbr[e]]))
        E = En
    return E


# -- the TMOP run ------------------------------------------------------------------------------------
class TmopEnergy(NamedTuple):
    """The reading a stage logs when no rung ``quality`` was supplied: the TMOP energy only."""

    energy: float
    n_inverted: int = -1
    sj_min: float = float("nan")
    objective_mean: float = float("nan")

    def __str__(self) -> str:
        return "TMOP energy %.6e" % self.energy


def _make_metric(mfem: Any, metric: int, weight: float, dim: int, mind: Any) -> Any:
    t = mfem.tmop
    if (metric >= 300) != (dim == 3):
        raise ValueError("tmop: metric %d is a %d-D metric but the cells are %s" % (metric, 3 if metric >= 300 else 2,
                                                                                    "hexes" if dim == 3 else "quads"))
    if metric == _UNTANGLE:
        return t.TMOP_Metric_313(mind)
    name = "TMOP_Metric_%03d" % metric
    cls = getattr(t, name, None)
    if cls is None:
        raise ValueError("tmop: this MFEM build has no metric %d (%s)" % (metric, name))
    return cls(weight) if metric in _WEIGHTED else cls()


def tmop(points: PointArray, cells: IntArray, resolved: Resolved, *, attributes: IntArray | None = None,
         designed: BoolArray | Sequence[int] | None = None, target: str = "input", metric: int | None = None,
         metric_weight: float = 0.5, solver: str = "newton", iters: int = 50, rtol: float = 1e-8,
         sweeps: int = 5, sweep_weight: float = 0.5, quad_order: int = 4, order: int = 1,
         quality: Callable[[PointArray], Any] | None = None, verbose: bool = False
         ) -> tuple[PointArray, StageLog]:
    """One TMOP run on the order-1 mesh ``(points, cells)`` under ``resolved``: the new vertex positions
    ``(N, 3)`` and a :class:`~nekmeshpy.core.smoother.schedule.StageLog` (its ``reason`` carries the energy
    before and after, the Newton / L-BFGS iterations and what was fixed).  The rules are in the module docstring.

    ``attributes`` ``(C,)`` ints label the elements (regions); ``designed`` selects the elements that keep their own
    input shape as target: a boolean ``(C,)`` mask, or integer attribute values (needs ``attributes``).  ``target``
    is ``"input"``, ``"input_smoothed"`` (``sweeps`` relaxations of weight ``sweep_weight`` on the target edge
    lengths) or ``"ideal"``.  ``metric`` defaults to ``DEFAULT_METRIC``; ``metric_weight`` is the weight
    of the combined metrics 80 and 333.  ``solver`` is ``"newton"`` or ``"lbfgs"``, stopping at ``iters`` or a
    relative residual ``rtol``.  ``quad_order`` is the Gauss-Lobatto quadrature order (4 includes the corners, so
    the determinant check of the line search covers the geometry a ``.re2`` file keeps).  ``quality(points)``,
    when given, reads the rung's own ``Quality`` for the log (and enables the rollback: a run that ends with
    more inverted elements than it began with is refused and the input returned).

    Raises :class:`ImportError` without PyMFEM, :class:`ValueError` for a non-planar quad mesh, an inverted input
    (except metric 313) or a bad knob, :class:`NotImplementedError` for ``order > 1``."""
    if order != 1:
        raise NotImplementedError("tmop: only order 1 is wired in (got order=%r); smooth the linear mesh and "
                                  "rebuild the high-order nodes afterwards" % (order,))
    if target not in _TARGETS:
        raise ValueError("tmop: target must be one of %s, got %r" % (_TARGETS, target))
    if solver not in _SOLVERS:
        raise ValueError("tmop: solver must be one of %s, got %r" % (_SOLVERS, solver))
    if iters < 1:
        raise ValueError("tmop: iters must be >= 1, got %r" % (iters,))
    P0 = np.asarray(points, float)
    C = np.asarray(cells, dtype=np.int64)
    dim = _dim_of(C)
    nP, nC = P0.shape[0], C.shape[0]
    metric_id = DEFAULT_METRIC[dim] if metric is None else int(metric)
    untangle = metric_id == _UNTANGLE
    mfem = _mfem()
    t = mfem.tmop

    frame = plane_frame(P0, C)
    U0 = (P0 - frame.origin) @ frame.basis
    fm = fixed_components(P0, resolved, frame.basis)
    Us = (fm.points - frame.origin) @ frame.basis
    if not untangle:
        md = min_det(Us, C)
        if not md > 0.0:
            raise ValueError(
                "tmop: the input has an inverted element (smallest Jacobian determinant %.3e): TMOP's shape metrics "
                "are undefined there and MFEM aborts the process.  Untangle first (the barrier optimizer, "
                "`untangle_schedule`) or use metric=313" % md)
    if untangle and target != "ideal":
        raise ValueError("tmop: metric 313 untangles toward the ideal unit-size element: use target='ideal'")
    fixed = fm.fixed
    inactive = fixed.all(axis=1)[C].all(axis=1)                          # every vertex fully fixed: nothing to move
    if designed is None:
        dmask: BoolArray = np.zeros(nC, bool)
    else:
        d = np.asarray(designed)
        if d.dtype == bool:
            if d.shape != (nC,):
                raise ValueError("tmop: a boolean `designed` must be (%d,), got %s" % (nC, d.shape))
            dmask = d
        else:
            if attributes is None:
                raise ValueError("tmop: `designed` given as attribute values needs `attributes`")
            dmask = np.isin(np.asarray(attributes), d.astype(np.int64))
    if untangle and dmask.any():
        raise ValueError("tmop: metric 313 is a single integrator; `designed` cannot be combined with it")
    role = np.where(inactive, 3, np.where(dmask, 2, 1)).astype(np.int64)
    n_active = int((role < 3).sum())
    if n_active == 0:
        q0 = quality(P0) if quality is not None else TmopEnergy(0.0)
        return P0.copy(), StageLog("tmop", iters, 0, "no_elements", "every element is fully constrained", q0, q0, ())

    mesh = to_mfem_mesh(Us, C, role)
    fec = mfem.H1_FECollection(1, dim)
    fes = mfem.FiniteElementSpace(mesh, fec, dim, mfem.Ordering.byNODES)
    mesh.SetNodalFESpace(fes)
    x = mfem.GridFunction(fes)
    mesh.SetNodalGridFunction(x)
    got = np.array(x.GetDataArray(), copy=True).reshape(dim, nP).T
    if float(np.abs(got - Us).max(initial=0.0)) != 0.0:
        raise RuntimeError("tmop: MFEM renumbered the vertices on construction; the bridge cannot continue")
    x0 = mfem.GridFunction(fes)
    x0.Assign(x)

    keep: list[Any] = [mesh, fec, fes, x0]
    irules = mfem.IntegrationRules(0, mfem.Quadrature1D.GaussLobatto)
    TC = t.TargetConstructor
    mind = None
    if untangle:                    # the shifted barrier updates one shared minimum determinant: a double* in MFEM
        from mfem._ser.cpointers import doublep
        md0 = min_det(Us, C)
        mind = doublep()
        mind.assign(md0 - 0.01 * abs(md0) if md0 < 0 else 0.0)
        keep.append(mind)

    def integrator(met: Any, tc: Any, w: list[float], over_target: bool = False) -> Any:
        it = t.TMOP_Integrator(met, tc)
        it.SetIntegrationRules(irules, quad_order)
        it.IntegrateOverTarget(over_target)
        wv = mfem.Vector(np.array(w, dtype=float))
        coef = mfem.PWConstCoefficient(wv)
        it.SetCoefficient(coef)
        keep.extend([met, tc, it, wv, coef])
        return it

    its: list[Any] = []
    if (role == 1).any():
        if target == "ideal":
            tc = TC(TC.IDEAL_SHAPE_UNIT_SIZE if untangle else TC.IDEAL_SHAPE_GIVEN_SIZE)
            tc.SetNodes(x0)
            its.append(integrator(_make_metric(mfem, metric_id, metric_weight, dim, mind), tc, [1.0, 0.0, 0.0]))
        else:
            ext = element_extents(U0, C)
            if target == "input_smoothed":
                ext = _smooth_extents(ext, C, role == 1, sweeps, sweep_weight)
            V = ext.prod(axis=1)
            if dim == 3:
                rho = (ext / np.cbrt(V)[:, None]) ** 1.5                  # three aspect-ratio fields, product 1
                rho_vec = np.ascontiguousarray(rho.T.ravel())
            else:
                rho_vec = np.ascontiguousarray(ext[:, 1] / ext[:, 0])       # y over x: MFEM's 2-D aspect ratio
            l2 = mfem.L2_FECollection(0, dim)
            sfes = mfem.FiniteElementSpace(mesh, l2)
            afes = mfem.FiniteElementSpace(mesh, l2, 3 if dim == 3 else 1)
            size_gf = mfem.GridFunction(sfes)
            size_gf.Assign(mfem.Vector(np.ascontiguousarray(V)))
            ar_gf = mfem.GridFunction(afes)
            ar_gf.Assign(mfem.Vector(rho_vec))
            dtc = t.DiscreteAdaptTC(TC.GIVEN_SHAPE_AND_SIZE)
            dtc.SetAdaptivityEvaluator(t.AdvectorCG())
            dtc.SetSerialDiscreteTargetSize(size_gf)
            dtc.SetMinSizeForTargets(float(V.min()))
            dtc.SetSerialDiscreteTargetAspectRatio(ar_gf)
            dtc.SetNodes(x0)
            keep.extend([l2, sfes, afes, size_gf, ar_gf, dtc])
            its.append(integrator(_make_metric(mfem, metric_id, metric_weight, dim, mind), dtc, [1.0, 0.0, 0.0]))
    if (role == 2).any():                                        # designed: the element's own input Jacobian
        tc2 = TC(TC.GIVEN_SHAPE_AND_SIZE)
        tc2.SetNodes(x0)
        its.append(integrator(_make_metric(mfem, metric_id, metric_weight, dim, mind), tc2, [0.0, 1.0, 0.0]))
    form = mfem.NonlinearForm(fes)
    if len(its) == 2:
        combo = t.TMOPComboIntegrator()
        for it in its:
            combo.AddTMOPIntegrator(it)
            it.thisown = 0                                       # the combo deletes its members: no double free
        form.AddDomainIntegrator(combo)
        keep.append(combo)
    else:
        form.AddDomainIntegrator(its[0])
    if untangle:
        its[0].EnableFiniteDifferences(x)                        # 313 has no analytic EvalP in MFEM 4.8
    keep.append(form)
    vd = sorted(int(fes.DofToVDof(int(v), int(c))) for v, c in zip(*np.nonzero(fixed)))
    if vd:
        form.SetEssentialVDofs(mfem.intArray(vd))
    e0 = float(form.GetGridFunctionEnergy(x))

    ir = irules.Get(fes.GetFE(0).GetGeomType(), quad_order)
    sol = t.TMOPNewtonSolver(ir, 0 if solver == "newton" else 1)
    sol.SetIntegrationRules(irules, quad_order)
    if solver == "newton":
        minres = mfem.MINRESSolver()
        minres.SetMaxIter(100)
        minres.SetRelTol(1e-12)
        minres.SetAbsTol(0.0)
        minres.SetPrintLevel(-1)
        ds = mfem.DSmoother(1, 1.0, 1)
        ds.SetPositiveDiagonal(True)
        minres.SetPreconditioner(ds)
        sol.SetPreconditioner(minres)
        keep.extend([minres, ds])
    if mind is not None:
        sol.SetMinDetPtr(mind)
    sol.SetMaxIter(int(iters))
    sol.SetRelTol(float(rtol))
    sol.SetAbsTol(0.0)
    sol.SetPrintLevel(1 if verbose else -1)
    sol.SetOperator(form)
    x.SetTrueVector()
    sol.Mult(mfem.Vector(0), x.GetTrueVector())
    x.SetFromTrueVector()
    e1 = float(form.GetGridFunctionEnergy(x))
    done = int(sol.GetNumIterations())
    converged = bool(sol.GetConverged())
    Un = np.array(x.GetDataArray(), copy=True).reshape(dim, nP).T
    Un[fixed] = Us[fixed]                                        # fixed components: the bits we were given
    Pn = fm.points + (Un - Us) @ frame.basis.T
    while keep:                                                  # release in reverse order of creation
        keep.pop()

    q0 = quality(fm.points) if quality is not None else TmopEnergy(e0)
    q1 = quality(Pn) if quality is not None else TmopEnergy(e1)
    detail = ("metric %d, target %s, %s: %d of %d elements active%s; fixed: %s"
              % (metric_id, target, solver, n_active, nC, ", %d designed" % int((role == 2).sum()) if dmask.any() else "",
                 ", ".join(fm.notes) or "nothing"))
    if not np.isfinite(Pn).all() or not np.isfinite(e1) or (not untangle and e1 > e0 * (1 + 1e-9) + 1e-300):
        return P0.copy(), StageLog("tmop", iters, 0, "rolled_back", "TMOP energy %.6e -> %.6e (not a descent); stage "
                                   "refused (%s)" % (e0, e1, detail), q0, q0, ())
    if quality is not None and q1.n_inverted > q0.n_inverted:
        return P0.copy(), StageLog("tmop", iters, 0, "rolled_back",
                                   "ended with %d inverted elements vs %d at the start; stage refused (%s)"
                                   % (q1.n_inverted, q0.n_inverted, detail), q0, q0, ())
    energies = ("the 313 barrier energy is shifted as it goes and is not comparable" if untangle
                else "TMOP energy %.6e -> %.6e" % (e0, e1))
    reason = "%s in %d iterations; %s (%s)" % (
        "converged" if converged else "stopped at the iteration limit", done, energies, detail)
    hist = ((0, *history_row(q0)), (done, *history_row(q1)))
    return Pn, StageLog("tmop", iters, done, "converged" if converged else "completed", reason, q0, q1, hist)
