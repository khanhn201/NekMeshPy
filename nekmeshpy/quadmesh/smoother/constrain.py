"""Smoother constraints (M2): what each vertex may do, and how the target geometry holds it.

A smoother never guesses.  Every vertex gets one *mobility* -- ``FREE``, ``ON_SURFACE``,
``ON_CURVE``, ``TANGENT`` or ``FROZEN`` -- resolved once from the constraints the caller states, and the
two operations the smoothing methods share act on it:

* :func:`snap` projects constrained vertices back onto their target (``x += w (P(x) - x)``;
  ``w = 1`` is the full projection) and restores frozen ones;
* :func:`constrain_direction` removes from a gradient or a step whatever the target forbids --
  the component along the surface normal, everything but the tangent on a curve, all of it
  on a frozen vertex -- using the **target's own** normal, never the mesh's.

Strict geometry mode (the default, as in the user's pebble code) is both, every step.
Precedence is ``FROZEN > ON_CURVE > ON_SURFACE > TANGENT > FREE``; a vertex claimed by two surfaces,
or by two curves (a corner), is frozen and listed in the report -- it is never given an
order-dependent projection implicitly.  Unclaimed *boundary* vertices are frozen by default,
so a block's seam cannot drift (``boundary="free"`` opens them, ``"tangent"`` lets them move along the boundary only).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ..._typing import BoolArray, IntArray, PointArray
from ...core.smoother.constraints import (
    CORNER_DEG,
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
    boundary_corners,
    circle,
    circle_tangent,
    claimed_surfaces,
    constrain_direction,
    line,
    on_target_report,
    plane,
    snap,
    sphere,
    spline_curve,
    surface_from_trimesh,
    turning_deg,
    vertex_length,
)
from ..quadmesh import QuadMesh
from ..query import boundary_edge_ids, boundary_points, tagged_edges

__all__ = ["FREE", "ON_SURFACE", "ON_CURVE", "FROZEN", "TIED", "TANGENT", "circle_tangent", "Surface", "Curve", "plane", "sphere",
           "line", "circle", "Frozen", "Tangent", "OnSurface", "OnCurve", "KeepLayer", "layer_rings", "SlideColumns", "grid_columns", "Resolved", "OnTargetReport",
           "resolve", "snap", "constrain_direction", "on_target_report", "vertex_length", "surface_from_trimesh",
           "NormalReport", "surface_normals", "claimed_surfaces", "CORNER_DEG", "boundary_adjacency",
           "turning_deg", "spline_curve", "boundary_corners", "vertex_normals"]




def _ids(mesh: QuadMesh, which: Select) -> IntArray:
    n = mesh.n_points
    if isinstance(which, str):
        e = mesh.edges[tagged_edges(mesh, which)]
        return np.unique(e) if e.size else np.zeros(0, dtype=np.int64)
    a = np.asarray(which)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError("a boolean point mask must have %d entries, got %s" % (n, a.shape))
        return np.flatnonzero(a).astype(np.int64)
    ids = a.astype(np.int64).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= n):
        raise IndexError("point ids must lie in [0, %d)" % n)
    return np.unique(ids)


def layer_rings(mesh: QuadMesh, seeds: Select, rings: int) -> IntArray:
    """Sorted ids of every vertex within ``rings`` quad-rings of ``seeds`` (``seeds`` included): ring
    ``k`` is the set of vertices sharing a quad with ring ``k - 1``."""
    if rings < 0:
        raise ValueError("rings must be >= 0, got %r" % (rings,))
    reach: BoolArray = np.zeros(mesh.n_points, dtype=bool)
    reach[_ids(mesh, seeds)] = True
    c = mesh.corners
    for _ in range(rings):
        touched = reach[c].any(axis=1)
        reach[c[touched].ravel()] = True
    return np.flatnonzero(reach).astype(np.int64)


def grid_columns(mesh: QuadMesh, seeds: Select, direction: Sequence[float], *,
                 depth_in: int | None = None, depth_out: int | None = None,
                 within: BoolArray | None = None, outside_depth: int = 1,
                 across: float = 0.5) -> list[IntArray]:
    """The mesh's own **grid lines across a layer**, found from topology alone.

    From every seed vertex (a vertex on an interface that slides along ``direction``), step along each
    incident edge that crosses the interface (its angle to ``direction`` is more than ``arccos(across)``
    -- 60 degrees by default) and keep walking *straight* through regular (valence-4) interior vertices:
    at such a vertex the straight continuation is the one edge that shares no quad with the edge you
    arrived on.  A walk ends at a boundary vertex or an irregular vertex (both included) or after its
    depth limit.  The seed and its two walks form one column; ``depth_in`` / ``depth_out`` bound the
    walk on the first / second across-edge (``None`` = until it ends; which side is "in" is simply the
    edge order, so pass the same bound to both sides when it does not matter).

    Nothing here knows a thickness, a wall or an axis: only the connectivity and ``direction``.  It
    finds columns only where the mesh is structured across the layer; elsewhere a walk ends at once.

    ``within`` (a ``(P,)`` boolean mask, e.g. the vertices of one region) replaces the two depth
    bounds with a region rule: a walk goes on without limit through vertices that are ``within`` and
    is allowed ``outside_depth`` further vertices once it has left it -- so the layers of a region
    ride along with its interface and only the first rows of its neighbour do."""
    P = np.asarray(mesh.points, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    nv = mesh.n_points
    corners = mesh.corners
    nbr: list[set[int]] = [set() for _ in range(nv)]
    vq: list[list[int]] = [[] for _ in range(nv)]
    for qi, q in enumerate(corners):
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            nbr[a].add(b)
            nbr[b].add(a)
            vq[a].append(qi)
    bnd: BoolArray = np.zeros(nv, dtype=bool)
    bnd[boundary_points(mesh)] = True

    def quads_with(a: int, b: int) -> list[int]:
        return [qi for qi in vq[a] if b in corners[qi]]

    def step(v: int, w: int) -> int | None:
        """The straight continuation through ``w`` of the edge ``v -> w``, or None."""
        if bnd[w] or len(nbr[w]) != 4:
            return None
        adj: set[int] = {v}
        for qi in quads_with(v, w):
            q = [int(x) for x in corners[qi]]
            k = q.index(w)
            adj.update((q[(k + 1) % 4], q[(k - 1) % 4]))
        rest = nbr[w] - adj
        return next(iter(rest)) if len(rest) == 1 else None

    out: list[IntArray] = []
    for v in _ids(mesh, seeds):
        v = int(v)
        col = [v]
        sides = []
        for w in nbr[v]:
            e = P[w] - P[v]
            ln = np.linalg.norm(e)
            if ln > 0 and abs(float(e @ d)) / ln < across:
                sides.append(w)
        for side, w in enumerate(sorted(sides)[:2]):
            limit = depth_in if side == 0 else depth_out
            prev = v
            cur: int | None = w
            n = 0
            n_out = 0
            while cur is not None:
                if within is not None:
                    if not within[cur]:
                        n_out += 1
                        if n_out > outside_depth:
                            break
                elif limit is not None and n >= limit:
                    break
                col.append(cur)
                n += 1
                nxt = step(prev, cur)
                prev = cur
                cur = nxt
        out.append(np.unique(np.asarray(col, dtype=np.int64)))
    return out


def boundary_adjacency(mesh: QuadMesh) -> dict[int, list[int]]:
    """Each boundary vertex's neighbours along the boundary edges (two on a manifold boundary)."""
    nb: dict[int, list[int]] = {}
    for a, b in np.asarray(mesh.edges)[boundary_edge_ids(mesh)]:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    return nb



def vertex_normals(P: PointArray, corners: IntArray) -> PointArray:
    """``(n, 3)`` unit normal per vertex: the mean of its quads' unit normals (the cross product of the
    diagonals); a zero row where they cancel."""
    c = np.asarray(corners, dtype=np.int64)
    n = np.cross(P[c[:, 2]] - P[c[:, 0]], P[c[:, 3]] - P[c[:, 1]])
    n = n / np.where(np.linalg.norm(n, axis=1, keepdims=True) > 0, np.linalg.norm(n, axis=1, keepdims=True), 1.0)
    v = np.zeros((P.shape[0], 3))
    for k in range(4):
        np.add.at(v, c[:, k], n)
    return np.asarray(v / np.where(np.linalg.norm(v, axis=1, keepdims=True) > 0,
                                   np.linalg.norm(v, axis=1, keepdims=True), 1.0))




def _boundary_chains(mesh: QuadMesh, claimed: BoolArray, corner_deg: float) -> tuple[list[list[int]], list[list[int]], IntArray]:
    """``(open, closed, corners)``: the boundary cut at every break -- a corner (:func:`boundary_corners`), a
    vertex without exactly two boundary neighbours, or a ``claimed`` vertex.  Each open chain runs break to
    break (ends included); each closed one is a break-free loop.  ``corners`` are the unclaimed breaks."""
    nb = boundary_adjacency(mesh)
    P = np.asarray(mesh.points, float)
    brk = {v for v in nb if claimed[v]} | boundary_corners(P, nb, corner_deg, vertex_normals(P, mesh.corners))
    seen: set[tuple[int, int]] = set()
    opened: list[list[int]] = []
    closed: list[list[int]] = []

    def walk(a: int, b: int) -> list[int]:
        chain = [a, b]
        seen.add((a, b))
        seen.add((b, a))
        while chain[-1] not in brk and chain[-1] != chain[0]:
            n2 = nb[chain[-1]]
            nxt = n2[0] if n2[0] != chain[-2] else n2[1]
            seen.add((chain[-1], nxt))
            seen.add((nxt, chain[-1]))
            chain.append(nxt)
        return chain

    for v in sorted(brk):
        for w in nb[v]:
            if (v, w) not in seen:
                opened.append(walk(v, w))
    for v in sorted(nb):
        for w in nb[v]:
            if (v, w) not in seen:
                loop = walk(v, w)
                closed.append(loop[:-1])
    corners = np.array(sorted(v for v in brk if not claimed[v]), dtype=np.int64)
    return opened, closed, corners


def _expand(mesh: QuadMesh, constraints: Sequence[Constraint], corner_deg: float) -> list[Constraint]:
    """``constraints`` with every geometry-less :class:`OnCurve` / :class:`OnSurface` turned into concrete
    ones fitted to the input mesh (and the corners of a fitted curve into :class:`Frozen`)."""
    P = np.asarray(mesh.points, float)
    out: list[Constraint] = []
    for c in constraints:
        if isinstance(c, OnCurve) and c.curve is None:
            ids = _ids(mesh, c.points)
            bnd: BoolArray = np.zeros(mesh.n_points, bool)
            bnd[boundary_points(mesh)] = True
            if not bnd[ids].all():
                raise ValueError("OnCurve without a curve fits the boundary: %d of its vertices are interior"
                                 % int((~bnd[ids]).sum()))
            named: BoolArray = np.zeros(mesh.n_points, bool)
            named[ids] = True
            opened, closed, corners = _boundary_chains(mesh, bnd & ~named, corner_deg)
            for ch in opened:
                if len(ch) > 2:
                    out.append(OnCurve(np.asarray(ch[1:-1], dtype=np.int64), spline_curve(P[ch], False), c.slide))
            for ch in closed:
                out.append(OnCurve(np.asarray(ch, dtype=np.int64), spline_curve(P[ch], True), c.slide))
            if corners.size:
                out.append(Frozen(corners))
        elif isinstance(c, OnSurface) and c.surface is None:
            ids = _ids(mesh, c.points)
            hit = np.isin(mesh.corners, ids).any(axis=1)
            q = mesh.corners[hit]
            tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            out.append(OnSurface(ids, surface_from_trimesh(P, tris, name="input faces"), c.slide))
        else:
            out.append(c)
    return out


def _count_note(r: Resolved) -> str:
    return "resolved: " + ", ".join("%d %s" % (n, k) for k, n in r.counts().items() if n)


def resolve(mesh: QuadMesh, constraints: Sequence[Constraint], *,
            boundary: str = "frozen", corner_deg: float = CORNER_DEG) -> Resolved:
    """Resolve ``constraints`` over ``mesh``'s vertices (see the module docstring for the precedence).

    ``boundary`` is only a shorthand for what to state about every boundary vertex no constraint names:
    ``"frozen"`` (default) -> :class:`Frozen`; ``"free"`` -> nothing (it moves as the method proposes, so a
    convex boundary shrinks under Laplace); ``"tangent"`` -> :class:`Tangent`; ``"slide"`` ->
    ``OnCurve(ids)`` fitted to the input boundary (3-D on a surface mesh).  Corners (:func:`boundary_corners`,
    ``corner_deg`` and the local-jump test) are frozen in every mode but ``"free"``, which has no
    notion of them."""
    if boundary not in ("frozen", "free", "tangent", "slide"):
        raise ValueError("boundary must be 'frozen', 'free', 'tangent' or 'slide', got %r" % (boundary,))
    cons = _expand(mesh, constraints, corner_deg)
    first = _resolve_core(mesh, cons, corner_deg)
    if boundary == "free":
        return first._replace(notes=(*first.notes, _count_note(first)))
    bnd: BoolArray = np.zeros(mesh.n_points, bool)
    bnd[boundary_points(mesh)] = True
    loose = np.flatnonzero(bnd & (first.kind == FREE)).astype(np.int64)
    shorthand: list[Constraint] = []
    note = "boundary=%r: %d unnamed boundary vertice(s)" % (boundary, loose.size)
    if loose.size:
        shorthand = [Frozen(loose) if boundary == "frozen" else Tangent(loose) if boundary == "tangent"
                     else OnCurve(loose)]
    out = _resolve_core(mesh, [*cons, *_expand(mesh, shorthand, corner_deg)], corner_deg)
    return out._replace(notes=(*out.notes, note, _count_note(out)))


def _resolve_core(mesh: QuadMesh, constraints: Sequence[Constraint], corner_deg: float) -> Resolved:
    """The resolver proper: every constraint concrete, nothing assumed about the boundary."""
    n = mesh.n_points
    P0 = np.asarray(mesh.points, float)
    n_surf: IntArray = np.zeros(n, np.int64)
    n_curve: IntArray = np.zeros(n, np.int64)
    claim_surf: IntArray = np.full(n, -1, np.int64)
    claim_curve: IntArray = np.full(n, -1, np.int64)
    frozen: BoolArray = np.zeros(n, bool)
    pin_surf: BoolArray = np.zeros(n, bool)          # slide=False: project once, then hold
    pin_curve: BoolArray = np.zeros(n, bool)
    tie: IntArray = np.full(n, -1, np.int64)
    tie_dir: PointArray = np.zeros((n, 3))
    tie_claims: IntArray = np.zeros(n, np.int64)
    tang: BoolArray = np.zeros(n, bool)
    n_tie = 0
    surfaces: list[Surface] = []
    curves: list[Curve] = []
    for c in constraints:
        if isinstance(c, SlideColumns):
            ids: IntArray = np.zeros(0, dtype=np.int64)
        else:
            ids = _ids(mesh, c.seeds if isinstance(c, KeepLayer) else c.points)
        if isinstance(c, Frozen):
            frozen[ids] = True
        elif isinstance(c, KeepLayer):
            frozen[layer_rings(mesh, c.seeds, c.rings)] = True
        elif isinstance(c, SlideColumns):
            d_ = np.asarray(c.direction, float)
            d_ = d_ / np.linalg.norm(d_)
            for col in c.columns:
                col = np.unique(np.asarray(col, dtype=np.int64))
                tie_claims[col] += 1
                tie[col] = n_tie
                tie_dir[col] = d_
                n_tie += 1
        elif isinstance(c, Tangent):
            tang[ids] = True
        elif isinstance(c, OnSurface):
            assert c.surface is not None
            surfaces.append(c.surface)
            n_surf[ids] += 1
            claim_surf[ids] = len(surfaces) - 1
            pin_surf[ids] |= not c.slide
        elif isinstance(c, OnCurve):
            assert c.curve is not None
            curves.append(c.curve)
            n_curve[ids] += 1
            claim_curve[ids] = len(curves) - 1
            pin_curve[ids] |= not c.slide
        else:
            raise TypeError("unknown constraint %r" % (c,))

    notes: list[str] = []
    multi_s = n_surf > 1
    multi_c = n_curve > 1
    if multi_s.any():
        notes.append("%d vertices on more than one surface -> frozen (default; no order-dependent "
                     "projection is chosen implicitly)" % int(multi_s.sum()))
    if multi_c.any():
        notes.append("%d vertices on more than one curve (corners) -> frozen" % int(multi_c.sum()))
    both = (n_surf > 0) & (n_curve > 0) & ~multi_c & ~multi_s
    if both.any():
        notes.append("%d vertices on a surface and a curve -> the curve wins (stricter)"
                     % int(both.sum()))
    multi_t = tie_claims > 1
    mixed_t = (tie_claims > 0) & ((n_surf > 0) | (n_curve > 0) | frozen)
    if multi_t.any() or mixed_t.any():
        notes.append("%d vertices in a rigid column are also claimed by another constraint -> frozen (the column too)"
                     % int((multi_t | mixed_t).sum()))
    nbr: IntArray = np.full((n, 2), -1, np.int64)
    notes_tang: list[str] = []
    if tang.any():
        adj = boundary_adjacency(mesh)
        if not tang[list(adj)].sum() == tang.sum():
            raise ValueError("Tangent names %d vertices that are not on the boundary" % int(tang.sum() - tang[list(adj)].sum()))
        corner = boundary_corners(P0, adj, corner_deg, vertex_normals(P0, mesh.corners))
        n_corner = 0
        for v in np.flatnonzero(tang).tolist():
            if v in corner:
                tang[v] = False
                frozen[v] = True
                n_corner += 1
            else:
                nbr[v] = adj[v]
        if n_corner:
            notes_tang.append("%d Tangent vertice(s) are corners or lack two boundary neighbours -> frozen" % n_corner)

    kind: IntArray = np.zeros(n, np.int64)
    target: IntArray = np.full(n, -1, np.int64)
    anchor = P0.copy()
    on_s = (n_surf == 1) & (tie_claims == 0)
    on_c = (n_curve == 1) & (tie_claims == 0)
    kind[on_s] = ON_SURFACE
    target[on_s] = claim_surf[on_s]
    kind[on_c] = ON_CURVE
    target[on_c] = claim_curve[on_c]
    hold = frozen | multi_s | multi_c | multi_t | mixed_t
    # a rigid column with one pinned member is pinned as a whole
    if (tie >= 0).any():
        pinned_cols = np.unique(tie[hold & (tie >= 0)])
        if pinned_cols.size:
            members = np.isin(tie, pinned_cols) & (tie >= 0)
            notes.append("%d rigid column(s) frozen because a member is frozen (corner / boundary / other claim)"
                         % pinned_cols.size)
            hold = hold | members
    kind[(tie >= 0) & ~hold] = TIED
    kind[tang & ~hold & (kind == FREE)] = TANGENT
    kind[hold] = FROZEN
    target[hold] = -1
    # slide=False: one projection now, then the vertex is held where it landed.
    for pin, claim, objs in ((pin_surf & on_s & ~on_c & ~hold, claim_surf, surfaces),
                             (pin_curve & on_c & ~hold, claim_curve, curves)):
        for t in np.unique(claim[pin]):
            m = pin & (claim == t)
            anchor[m] = objs[t].project(P0[m])
            kind[m] = FROZEN
            target[m] = -1
    tie[hold] = -1
    notes.extend(notes_tang)
    nbr[kind != TANGENT] = -1
    return Resolved(kind, target, anchor, tuple(surfaces), tuple(curves), tuple(notes), tie, tie_dir, nbr,
                    np.zeros((0, 4), np.int64))






# -- normals for the metric: from the surface the quads sit on, not from the quads ----------
class NormalReport(NamedTuple):
    """Where the per-quad normals came from, and whether the mesh's own winding agrees.

    ``cos`` is the cosine between each quad's own (winding) normal and the surface normal that was
    used; ``n_disagree`` counts quads below ``cos_tol`` -- a folded or reversed quad, or a surface
    that is not the one the mesh lies on.  ``n_from_surface`` quads took the surface normal; the
    ``n_fallback`` others had no claimed corner and kept their own."""

    n_quads: int
    n_from_surface: int
    n_fallback: int
    n_disagree: int
    min_cos: float
    disagree_ids: IntArray
    cos_tol: float
    #: the sign vote of each surface: ``+1`` the surface normal follows the mesh winding, ``-1`` it was flipped
    flips: tuple[int, ...]



def surface_normals(points: PointArray, corners: IntArray,
                    groups: Sequence[tuple[IntArray, Surface]], *,
                    check: bool = True, cos_tol: float = 0.5) -> tuple[PointArray, NormalReport]:
    """``(Q, 3)`` unit normals for the metric from the surface the quads sit on.

    Each group is ``(vertex ids, surface)``; the unit normal of the surface is read at every such
    vertex and a quad takes the mean of its claimed corners.  A surface normal has no sign of its
    own, so each surface gets **one** sign, voted against the mesh's winding -- never one per quad,
    which would quietly forgive a reversed quad.  Quads with no claimed corner keep the normal of
    their own winding.  ``check=False`` skips the comparison with the winding (and the report's
    ``n_disagree`` / ``min_cos``); the sign vote is still taken."""
    P = np.asarray(points, dtype=float)
    c = np.asarray(corners, dtype=np.int64)
    a = P[c[:, [1, 2, 3, 0]]] - P[c]
    b = P[c[:, [3, 0, 1, 2]]] - P[c]
    own = np.cross(a, b).sum(axis=1)
    own = own / np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-300)
    nv = np.zeros_like(P)
    has = np.zeros(P.shape[0], dtype=bool)
    flips = []
    for ids, srf in groups:
        n = np.asarray(srf.normal(P[ids]), dtype=float)
        nv_full = np.zeros_like(P)
        nv_full[ids] = n
        in_group = np.zeros(P.shape[0], dtype=bool)
        in_group[ids] = True
        slots = in_group[c]                                          # (Q, 4) corners of this group
        vote = float((np.einsum("qkj,qj->qk", nv_full[c], own) * slots).sum())
        sign = -1.0 if vote < 0 else 1.0
        flips.append(int(sign))
        nv[ids] = sign * n
        has[ids] = True
    cnt = has[c].sum(axis=1)
    q = (nv[c] * has[c][:, :, None]).sum(axis=1)
    from_surface = cnt > 0
    ln = np.linalg.norm(q, axis=1)
    from_surface &= ln > 1e-12
    out = own.copy()
    out[from_surface] = q[from_surface] / ln[from_surface, None]
    cos = np.einsum("ij,ij->i", own, out)
    cos_fs = cos[from_surface]
    bad = np.flatnonzero(from_surface & (cos < cos_tol)) if check else np.zeros(0, dtype=np.int64)
    rep = NormalReport(c.shape[0], int(from_surface.sum()), int((~from_surface).sum()), int(bad.size),
                       float(cos_fs.min(initial=1.0)) if check else float("nan"), bad, cos_tol, tuple(flips))
    return out, rep
