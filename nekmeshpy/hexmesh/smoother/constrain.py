"""Hex smoother constraints: what each vertex may do, and how the target geometry holds it.

The vocabulary is the quad smoother's own, reused rather than copied: the mobility kinds
(``FREE`` / ``ON_SURFACE`` / ``ON_CURVE`` / ``FROZEN`` / ``TIED``), the targets (``plane``,
``sphere``, ``surface_from_trimesh``, ``line``, ``circle``, ``spline_curve``), the constraints
(``Frozen``, ``OnSurface``, ``OnCurve``, ``KeepLayer``, ``SlideColumns``), :class:`Resolved`,
:func:`snap` and :func:`constrain_direction` are the very objects of
:mod:`nekmeshpy.quadmesh.smoother.constrain` -- they act on a vertex array and a target, never
on an element, so the dimension does not enter.  What is hex-specific lives here:

* a **string** selection names a *face* tag (the points of every face carrying it), where the
  quad rung reads an edge tag; ``KeepLayer`` counts *hex* rings;
* :func:`resolve` -- the same precedence (``FROZEN > ON_CURVE > ON_SURFACE > FREE``, a vertex
  claimed twice is frozen and listed), with ``boundary="frozen" | "free" | "slide"``;
* ``boundary="slide"`` -- the three-layer boundary treatment: boundary faces are grouped into
  **patches** by dihedral angle (:func:`boundary_patches`); a vertex inside one patch slides on
  it (a plane when the patch is planar, else the input faces as a triangle surface), a vertex
  where two patches meet slides along their intersection (a line when straight, else a spline
  through the input vertices), and a vertex where three or more meet, or where that edge turns
  sharply, is frozen;
* :func:`frozen_region` -- every vertex of the hexes carrying some element tags;
* :func:`vertex_length` / :func:`on_target_report` over hex edges.

Resolution itself runs through the quad :func:`~nekmeshpy.quadmesh.smoother.constrain.resolve`
on the mesh's shared-face ``QuadMesh`` (its points *are* the hex points), after every
selection has been turned into explicit vertex ids here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..._typing import BoolArray, FloatArray, IntArray, PointArray
from ...core.smoother.constraints import (
    FREE,
    FROZEN,
    ON_CURVE,
    ON_SURFACE,
    TANGENT,
    TIED,
    Constraint,
    Curve,
    Frozen,
    KeepLayer,
    OnCurve,
    OnSurface,
    OnTargetReport,
    Resolved,
    Select,
    SlideColumns,
    Surface,
    Tangent,
    circle,
    constrain_direction,
    line,
    plane,
    snap,
    sphere,
    spline_curve,
    surface_from_trimesh,
    unit_rows,
)
from ...quadmesh.smoother import constrain as _qc
from ..hexmesh import HexMesh
from ..query import boundary_points, tagged_faces

__all__ = ["FREE", "ON_SURFACE", "ON_CURVE", "FROZEN", "TIED", "TANGENT", "Tangent", "CORNER_DEG", "Surface", "Curve",
           "plane", "sphere", "line", "circle", "spline_curve", "surface_from_trimesh",
           "Frozen", "OnSurface", "OnCurve", "KeepLayer", "SlideColumns", "Resolved", "OnTargetReport",
           "layer_rings", "frozen_region", "BoundaryPatches", "boundary_faces_outward", "boundary_patches",
           "resolve", "snap", "constrain_direction", "vertex_length", "on_target_report"]

#: Angle (degrees) above which two boundary faces belong to different patches, and above which
#: the edge where two patches meet turns sharply enough to end its curve at a frozen corner.
CORNER_DEG = 30.0

#: Relative flatness (distance from the best-fit plane over the patch extent) under which a patch
#: is taken as a plane rather than a triangle surface.
_PLANAR_TOL = 1e-8


def _ids(mesh: HexMesh, which: Select) -> IntArray:
    """Sorted unique vertex ids of a selection: ids, a ``(P,)`` boolean mask, or a face-tag name."""
    n = mesh.n_points
    if isinstance(which, str):
        f = mesh.quad_mesh.corners[tagged_faces(mesh, which)]
        return np.unique(f) if f.size else np.zeros(0, dtype=np.int64)
    a = np.asarray(which)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError("a boolean point mask must have %d entries, got %s" % (n, a.shape))
        return np.flatnonzero(a).astype(np.int64)
    ids = a.astype(np.int64).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= n):
        raise IndexError("point ids must lie in [0, %d)" % n)
    return np.unique(ids)


def layer_rings(mesh: HexMesh, seeds: Select, rings: int) -> IntArray:
    """Sorted ids of every vertex within ``rings`` hex-rings of ``seeds`` (``seeds`` included, a
    face-tag name allowed): ring ``k`` is the set of vertices sharing a hex with ring ``k - 1``."""
    if rings < 0:
        raise ValueError("rings must be >= 0, got %r" % (rings,))
    reach: BoolArray = np.zeros(mesh.n_points, dtype=bool)
    reach[_ids(mesh, seeds)] = True
    c = mesh.corners
    for _ in range(rings):
        touched = reach[c].any(axis=1)
        reach[c[touched].ravel()] = True
    return np.flatnonzero(reach).astype(np.int64)


def frozen_region(mesh: HexMesh, tags: str | Sequence[str]) -> Frozen:
    """A :class:`Frozen` holding every vertex of every hex whose element tag is in ``tags`` -- a
    region (a solid, a designed block) kept exactly as meshed.  Its interface with a smoothed
    neighbour is frozen with it, so the neighbour adapts and the region does not."""
    names = [tags] if isinstance(tags, str) else list(tags)
    dense = mesh.element_tags.to_dense()
    missing = [t for t in names if not (dense == t).any()]
    if missing:
        raise ValueError("frozen_region: no hex carries the element tag(s) %s" % missing)
    sel = np.isin(dense, names)
    return Frozen(np.unique(mesh.corners[sel]).astype(np.int64))


def _explicit(mesh: HexMesh, c: Constraint) -> Constraint:
    """``c`` with its selection turned into explicit vertex ids; a ``KeepLayer`` becomes the
    ``Frozen`` of its rings.  What the quad resolver then reads is rung-independent."""
    if isinstance(c, KeepLayer):
        return Frozen(layer_rings(mesh, c.seeds, c.rings))
    if isinstance(c, (Frozen, Tangent, OnSurface, OnCurve)):
        return dataclasses.replace(c, points=_ids(mesh, c.points))
    if isinstance(c, SlideColumns):
        return c
    raise TypeError("unknown constraint %r" % (c,))


# -- the boundary as patches -----------------------------------------------------------------
def boundary_faces_outward(mesh: HexMesh) -> IntArray:
    """``(B, 4)`` corner ids of every face on the domain boundary, wound so the right-hand
    normal points **out of** the mesh (each is read from the one hex that carries it)."""
    F = mesh.corners[:, HexMesh.FACE_POINTS].reshape(-1, 4)
    _, inv, cnt = np.unique(np.sort(F, axis=1), axis=0, return_inverse=True, return_counts=True)
    return np.asarray(F[cnt[inv.reshape(-1)] == 1], dtype=np.int64)


def _face_normals(P: PointArray, F: IntArray) -> PointArray:
    """Unit normal of each quad ``F`` ``(B, 4)``: the cross product of its diagonals."""
    n = np.cross(P[F[:, 2]] - P[F[:, 0]], P[F[:, 3]] - P[F[:, 1]])
    return unit_rows(n)


class BoundaryPatches(NamedTuple):
    """The boundary cut into smooth patches -- what ``resolve(..., boundary="slide")`` builds on."""

    #: ``(B, 4)`` outward-wound boundary faces (:func:`boundary_faces_outward`).
    faces: IntArray
    #: ``(B,)`` patch id of each face.
    patch: IntArray
    #: ``(n_patches,)`` whether each patch is planar (within a relative ``1e-8`` of its extent).
    planar: BoolArray
    #: ``(P,)`` how many patches each vertex touches: ``0`` interior, ``1`` on a patch,
    #: ``2`` on the edge where two meet, ``3+`` a corner.
    n_patches: IntArray
    #: ``(K, 2)`` boundary edges where two different patches meet, or that do not border
    #: exactly two boundary faces.
    feature_edges: IntArray


def boundary_patches(mesh: HexMesh, corner_deg: float = CORNER_DEG) -> BoundaryPatches:
    """Group the boundary faces into patches: two faces sharing a boundary edge are in one patch
    when their normals differ by at most ``corner_deg`` degrees (transitively, so a smoothly
    curved wall is one patch however much it turns overall)."""
    P = np.asarray(mesh.points, float)
    F = boundary_faces_outward(mesh)
    B = F.shape[0]
    nrm = _face_normals(P, F)
    e = np.sort(np.stack([F, np.roll(F, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)   # (4B, 2)
    owner: IntArray = np.repeat(np.arange(B), 4)
    keys, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    two = cnt == 2
    # the two faces of every manifold boundary edge
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    fa = owner[order[starts[two]]]
    fb = owner[order[starts[two] + 1]]
    cos = np.einsum("ij,ij->i", nrm[fa], nrm[fb])
    join = cos >= np.cos(np.radians(corner_deg))
    g = coo_matrix((np.ones(int(join.sum())), (fa[join], fb[join])), shape=(B, B))
    n_p, patch = connected_components(g, directed=False)
    patch = np.asarray(patch, dtype=np.int64)
    feat = np.r_[np.flatnonzero(two)[patch[fa] != patch[fb]], np.flatnonzero(~two)]
    feature_edges = np.asarray(keys[np.sort(feat)], dtype=np.int64).reshape(-1, 2)
    # patches per vertex
    vp = np.unique(np.stack([F.ravel(), np.repeat(patch, 4)], axis=1), axis=0)
    n_patches: IntArray = np.bincount(vp[:, 0], minlength=mesh.n_points).astype(np.int64)
    planar = np.zeros(n_p, bool)
    for p in range(n_p):
        X = P[np.unique(F[patch == p])]
        c = X - X.mean(axis=0)
        ext = float(np.ptp(X, axis=0).max())
        normal = np.linalg.svd(c, full_matrices=False)[2][-1]
        planar[p] = ext == 0.0 or float(np.abs(c @ normal).max()) <= _PLANAR_TOL * ext
    return BoundaryPatches(F, patch, planar, n_patches, feature_edges)


def _patch_surface(P: PointArray, faces: IntArray, planar: bool, name: str) -> Surface:
    """The target of one patch: its best-fit plane, or its own faces as a triangle surface."""
    ids = np.unique(faces)
    X = P[ids]
    if planar:
        n = np.linalg.svd(X - X.mean(axis=0))[2][-1]
        return dataclasses.replace(plane(X.mean(axis=0), n), name="plane")
    # the input faces themselves, as triangles: an approximation (chord error remains)
    tris = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    return surface_from_trimesh(P, tris, name=name)


def _walk_chains(nb: dict[int, list[int]], brk: set[int]) -> tuple[list[list[int]], list[list[int]]]:
    """``(open, closed)`` chains of the graph ``nb`` cut at the vertices ``brk``: each open chain
    runs break to break (ends included), each closed one is a break-free loop."""
    seen: set[tuple[int, int]] = set()
    opened: list[list[int]] = []
    closed: list[list[int]] = []

    def walk(a: int, b: int) -> list[int]:
        chain = [a, b]
        seen.update(((a, b), (b, a)))
        while chain[-1] not in brk and chain[-1] != chain[0]:
            n2 = nb[chain[-1]]
            nxt = n2[0] if n2[0] != chain[-2] else n2[1]
            seen.update(((chain[-1], nxt), (nxt, chain[-1])))
            chain.append(nxt)
        return chain

    for v in sorted(brk):
        for w in nb.get(v, []):
            if (v, w) not in seen:
                opened.append(walk(v, w))
    for v in sorted(nb):
        for w in nb[v]:
            if (v, w) not in seen:
                closed.append(walk(v, w)[:-1])
    return opened, closed


def _edge_curve(X: PointArray, closed: bool) -> Curve:
    """A line when the chain ``X`` is straight, else a spline through it."""
    if not closed:
        d = X[-1] - X[0]
        ln = float(np.linalg.norm(d))
        if ln > 0:
            t = d / ln
            off = (X - X[0]) - np.outer((X - X[0]) @ t, t)
            if float(np.linalg.norm(off, axis=1).max()) <= _PLANAR_TOL * ln:
                return line(X[0], t)
    return spline_curve(X, closed, name="edge")


def _feature_chains(mesh: HexMesh, bp: BoundaryPatches, breaks: BoolArray, corner_deg: float
                    ) -> tuple[dict[int, list[int]], set[int], list[list[int]], list[list[int]]]:
    """``(adjacency, brk, open, closed)`` of the feature-edge graph (the edges where two patches meet), cut at
    ``brk``: every ``breaks`` vertex, every vertex that is not on exactly two patches or two feature edges,
    and every corner of the graph (:func:`~nekmeshpy.quadmesh.smoother.constrain.boundary_corners`)."""
    P = np.asarray(mesh.points, float)
    nb: dict[int, list[int]] = {}
    for a, b in bp.feature_edges:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    brk = {v for v in nb if len(nb[v]) != 2 or bp.n_patches[v] != 2 or breaks[v]}
    brk |= _qc.boundary_corners(P, nb, corner_deg)
    opened, closed = _walk_chains(nb, brk)
    return nb, brk, opened, closed


def _expand(mesh: HexMesh, cons: Sequence[Constraint], corner_deg: float) -> list[Constraint]:
    """``cons`` with every geometry-less :class:`OnCurve` / :class:`OnSurface` fitted to the input mesh:
    a curve through each feature-edge chain (patch-edge vertices only; corners :class:`Frozen`), a plane or
    triangle surface per patch (patch-interior vertices only)."""
    P = np.asarray(mesh.points, float)
    out: list[Constraint] = []
    bp: BoundaryPatches | None = None
    for c in cons:
        fit_curve = isinstance(c, OnCurve) and c.curve is None
        fit_surf = isinstance(c, OnSurface) and c.surface is None
        if not (fit_curve or fit_surf):
            out.append(c)
            continue
        assert isinstance(c, (OnCurve, OnSurface))
        bp = bp or boundary_patches(mesh, corner_deg)
        ids = np.asarray(c.points, dtype=np.int64)
        named: BoolArray = np.zeros(mesh.n_points, bool)
        named[ids] = True
        if fit_curve:
            _, brk, opened, closed = _feature_chains(mesh, bp, ~named, corner_deg)
            for ch in opened:
                if len(ch) > 2:
                    out.append(OnCurve(np.asarray(ch[1:-1], dtype=np.int64), _edge_curve(P[ch], False), c.slide))
            for ch in closed:
                out.append(OnCurve(np.asarray(ch, dtype=np.int64), _edge_curve(P[ch], True), c.slide))
            corners = np.array(sorted(v for v in brk if named[v]), dtype=np.int64)
            corners = corners[bp.n_patches[corners] >= 2] if corners.size else corners
            if corners.size:
                out.append(Frozen(corners))
        else:
            for p in range(bp.planar.size):
                fp = bp.faces[bp.patch == p]
                v = np.unique(fp)
                v = v[(bp.n_patches[v] == 1) & named[v]]
                if v.size:
                    out.append(OnSurface(v, _patch_surface(P, fp, bool(bp.planar[p]), "patch%d" % p), c.slide))
    return out


def resolve(mesh: HexMesh, constraints: Sequence[Constraint], *,
            boundary: str = "frozen", corner_deg: float = CORNER_DEG) -> Resolved:
    """Resolve ``constraints`` over ``mesh``'s vertices (precedence as in the quad smoother:
    ``FROZEN > ON_CURVE > ON_SURFACE > TANGENT > FREE``; a vertex claimed by two surfaces or two curves
    is frozen and listed in ``notes``).

    The boundary is read as **patches** (:func:`boundary_patches`; ``corner_deg`` is the dihedral angle that
    separates them): a vertex inside one patch is a *face* vertex, one where two meet is an *edge* vertex on a
    chain of such vertices, and one where three or more meet, or where the chain turns by a jump (the quad
    rung's corner rule), is a *corner*.  Geometry-less constraints are fitted to the input mesh:
    ``OnSurface(ids)`` gives each patch's face vertices a plane or triangle surface of that patch,
    ``OnCurve(ids)`` gives each edge chain a line or spline, ``Tangent(ids)`` lets a face vertex move in the
    tangent plane of its boundary faces and an edge vertex along the circle through it and its two chain
    neighbours.  A corner named by a fitted constraint is frozen.

    ``boundary`` is the shorthand for what to state about every boundary vertex no constraint names:
    ``"frozen"`` (default) -> :class:`Frozen`; ``"free"`` -> nothing (it moves as the method proposes, so a
    convex boundary shrinks under Laplace, and corners are not held); ``"tangent"`` -> :class:`Tangent`;
    ``"slide"`` -> ``OnSurface`` on face vertices and ``OnCurve`` on edge vertices, both fitted.  Corners
    are frozen in every mode but ``"free"``.  Linear and high-order meshes alike: only the corners are read."""
    if boundary not in ("frozen", "free", "tangent", "slide"):
        raise ValueError("boundary must be 'frozen', 'free', 'tangent' or 'slide', got %r" % (boundary,))
    qm = mesh.quad_mesh                      # same points, same ids: the quad resolver reads only ids
    tang: BoolArray = np.zeros(mesh.n_points, bool)
    plain: list[Constraint] = []
    for c in constraints:
        e = _explicit(mesh, c)
        if isinstance(e, Tangent):
            tang[np.asarray(e.points, dtype=np.int64)] = True
        else:
            plain.append(e)
    cons = _expand(mesh, plain, corner_deg)
    first = _qc.resolve(qm, cons, boundary="free")
    bnd: BoolArray = np.zeros(mesh.n_points, bool)
    bnd[boundary_points(mesh)] = True
    if tang.any() and not bnd[tang].all():
        raise ValueError("Tangent names %d vertices that are not on the boundary" % int((tang & ~bnd).sum()))
    loose = bnd & (first.kind == FREE) & ~tang
    notes = [f"boundary={boundary!r}: {int(loose.sum())} unnamed boundary vertice(s)"]
    if boundary == "frozen" and loose.any():
        cons.append(Frozen(np.flatnonzero(loose).astype(np.int64)))
    elif boundary == "tangent":
        tang |= loose
    elif boundary == "slide" and loose.any():
        ids: IntArray = np.flatnonzero(loose).astype(np.int64)
        cons += _expand(mesh, [OnCurve(ids), OnSurface(ids)], corner_deg)
        mid = _qc.resolve(qm, cons, boundary="free")
        left = loose & (mid.kind == FREE)               # on no patch or chain: three or more patches meet
        if left.any():
            cons.append(Frozen(np.flatnonzero(left).astype(np.int64)))
    bfaces: IntArray = np.zeros((0, 4), dtype=np.int64)
    nbr: IntArray = np.full((mesh.n_points, 2), -1, np.int64)
    if tang.any():
        bp = boundary_patches(mesh, corner_deg)
        bfaces = bp.faces
        mid = _qc.resolve(qm, cons, boundary="free")
        tang &= mid.kind == FREE                        # a stronger constraint wins
        nb, brk, _, _ = _feature_chains(mesh, bp, np.zeros(mesh.n_points, bool), corner_deg)
        on_edge = tang & (bp.n_patches == 2)
        corner = tang & (bp.n_patches >= 3)
        for v in np.flatnonzero(on_edge).tolist():
            if v in brk or v not in nb:
                corner[v] = True
            else:
                nbr[v] = nb[v]
        tang &= ~corner
        if corner.any():
            cons.append(Frozen(np.flatnonzero(corner).astype(np.int64)))
            notes.append("%d Tangent vertice(s) are corners -> frozen" % int(corner.sum()))
    out = _qc.resolve(qm, cons, boundary="free")
    kind = out.kind.copy()
    if tang.any():
        kind[tang & (kind == FREE)] = TANGENT
        nbr[kind != TANGENT] = -1
        out = out._replace(kind=kind, nbr=nbr, bfaces=bfaces)
    if boundary == "slide":
        notes.append("boundary='slide': %d patch(es) (%d planar), %d edge curve(s) where two patches meet, "
                     "%d corner vertex(es) frozen" % (len(out.surfaces), sum(f.name == "plane" for f in out.surfaces),
                                                      len(out.curves), int((out.kind == FROZEN).sum())))
    notes.append(_qc._count_note(out))
    return out._replace(notes=(*out.notes, *notes))


def vertex_length(points: PointArray, hexes: IntArray) -> FloatArray:
    """``(P,)`` the shortest hex edge at each vertex -- the length unit of every step, finite
    difference and tolerance of the optimizer.  The six faces of a hex cover its twelve edges,
    so this is the quad reading over the face cycles."""
    cycles = np.asarray(hexes, dtype=np.int64)[:, HexMesh.FACE_POINTS].reshape(-1, 4)
    return _qc.vertex_length(np.asarray(points, float), cycles)


def on_target_report(points: PointArray, hexes: IntArray, r: Resolved, *,
                     tol: float = 1e-6) -> OnTargetReport:
    """Distance of every constrained vertex to its target and the largest move of a frozen one,
    in absolute units and in local hex-edge lengths (the quad check, over the hex edges)."""
    cycles = np.asarray(hexes, dtype=np.int64)[:, HexMesh.FACE_POINTS].reshape(-1, 4)
    return _qc.on_target_report(np.asarray(points, float), cycles, r, tol=tol)
