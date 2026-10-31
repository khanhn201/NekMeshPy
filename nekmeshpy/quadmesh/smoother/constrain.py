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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from ..._typing import BoolArray, FloatArray, IntArray, PointArray
from ..quadmesh import QuadMesh
from ..query import boundary_edge_ids, boundary_points, tagged_edges

__all__ = ["FREE", "ON_SURFACE", "ON_CURVE", "FROZEN", "TIED", "TANGENT", "circle_tangent", "Surface", "Curve", "plane", "sphere",
           "line", "circle", "Frozen", "Tangent", "OnSurface", "OnCurve", "KeepLayer", "layer_rings", "SlideColumns", "grid_columns", "Resolved", "OnTargetReport",
           "resolve", "snap", "constrain_direction", "on_target_report", "vertex_length", "surface_from_trimesh",
           "NormalReport", "surface_normals", "claimed_surfaces", "CORNER_DEG", "boundary_adjacency",
           "turning_deg", "spline_curve", "boundary_corners", "vertex_normals"]

#: Turning angle (degrees) above which a boundary vertex is a corner: frozen by ``resolve(...,
#: boundary="tangent")``, and frozen, ending its chains, by ``boundary="slide"``.
CORNER_DEG = 30.0

FREE, ON_SURFACE, ON_CURVE, FROZEN, TIED, TANGENT = 0, 1, 2, 3, 4, 5
_KIND = {FREE: "free", ON_SURFACE: "on_surface", ON_CURVE: "on_curve", FROZEN: "frozen", TIED: "slide_column",
         TANGENT: "tangent"}

#: Which vertices a constraint names: point ids, a boolean mask, or an edge-tag name
#: (the points of every edge carrying it -- "this tagged loop lies on this curve").
Select = Union[str, IntArray, BoolArray]


@dataclass(frozen=True)
class Surface:
    """A target surface: ``project(P)`` the closest point, ``normal(P)`` the unit normal at
    that foot point, both ``(N, 3) -> (N, 3)``."""

    project: Callable[[PointArray], PointArray]
    normal: Callable[[PointArray], PointArray]
    name: str = "surface"


@dataclass(frozen=True)
class Curve:
    """A target curve: ``project(P)`` the closest point, ``tangent(P)`` the unit tangent
    at that foot point."""

    project: Callable[[PointArray], PointArray]
    tangent: Callable[[PointArray], PointArray]
    name: str = "curve"


def _unit(v: PointArray) -> PointArray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / np.where(n > 0, n, 1.0))


def plane(point: PointArray, normal: PointArray) -> Surface:
    """The plane through ``point`` with ``normal``."""
    p0 = np.asarray(point, float)
    n0 = _unit(np.asarray(normal, float))
    return Surface(lambda P: P - np.outer((P - p0) @ n0, n0),
                   lambda P: np.broadcast_to(n0, P.shape).copy(), "plane")


def sphere(center: PointArray, radius: float) -> Surface:
    """The sphere of ``radius`` about ``center``; the normal points outward."""
    c = np.asarray(center, float)
    return Surface(lambda P: c + radius * _unit(P - c), lambda P: _unit(P - c), "sphere")


def line(point: PointArray, direction: PointArray) -> Curve:
    """The straight line through ``point`` along ``direction``."""
    p0 = np.asarray(point, float)
    t0 = _unit(np.asarray(direction, float))
    return Curve(lambda P: p0 + np.outer((P - p0) @ t0, t0),
                 lambda P: np.broadcast_to(t0, P.shape).copy(), "line")


def circle(center: PointArray, radius: float, axis: PointArray = (0.0, 0.0, 1.0)) -> Curve:
    """The circle of ``radius`` about ``center`` in the plane normal to ``axis``; the
    tangent runs counter-clockwise about ``axis``."""
    c = np.asarray(center, float)
    z = _unit(np.asarray(axis, float))

    def radial(P: PointArray) -> PointArray:
        d = P - c
        return _unit(d - np.outer(d @ z, z))

    return Curve(lambda P: c + radius * radial(P),
                 lambda P: _unit(np.cross(z, radial(P))), "circle")


@dataclass(frozen=True)
class Frozen:
    """Never moves."""

    points: Select


@dataclass(frozen=True)
class Tangent:
    """Moves only along the boundary: a boundary vertex whose move is cut down to the tangent of the circle
    through it and its two boundary neighbours (:func:`circle_tangent`, from the current positions).  Nothing
    pulls it back onto a curve -- that is :class:`OnCurve`.  A corner (see :func:`boundary_corners`) or a vertex
    without two boundary neighbours is frozen instead; a vertex that is not on the boundary is an error."""

    points: Select


@dataclass(frozen=True)
class OnSurface:
    """Stays on ``surface``.  ``slide=True`` moves tangentially then re-projected;
    ``slide=False`` projects once onto the surface and then holds still.  Without a ``surface`` it is fitted:
    the input quads touching these vertices, as a triangle surface (:func:`surface_from_trimesh`)."""

    points: Select
    surface: Surface | None = None
    slide: bool = True


@dataclass(frozen=True)
class OnCurve:
    """Stays on ``curve``, moving along it only when ``slide=True``.  Without a ``curve`` it is fitted: the
    boundary chains through these vertices, as meshed, are cut at corners (:func:`boundary_corners`) and at
    boundary vertices not named here, and each piece gets a spline through it (:func:`spline_curve`, closed
    for a loop); a corner is frozen."""

    points: Select
    curve: Curve | None = None
    slide: bool = True


@dataclass(frozen=True)
class KeepLayer:
    """Keep a boundary layer exactly as meshed: every vertex within ``rings`` quad-rings of ``seeds``
    (the interface or wall vertices, ``seeds`` included) is frozen, so the layer's thickness, its
    growth ratio and its tangential spacing cannot change.  The layers keep their geometry exactly
    at the price of not sliding; one ring protects the first row of cells off the seeds, two the
    first two.  A square-seeking metric will otherwise inflate a thin layer (measured on the case_bz
    slices: edges next to an interface grew up to 34x)."""

    seeds: Select
    rings: int = 1


@dataclass(frozen=True)
class SlideColumns:
    """Rigid columns that slide together along ``direction``: every column is a set of vertices
    (e.g. the rows across a wall's thickness) with **one** shared degree of freedom, so they all move
    by the same amount along ``direction`` -- the cells between the rows stay rectangles while the
    spacing along the wall adapts.  Each column also stays on its own straight line.  If any member of a
    column ends up frozen (a corner, a vertex that must not leave the boundary), the whole column is
    frozen: a rigid unit cannot move with one member pinned."""

    columns: Sequence[IntArray]
    direction: Sequence[float]


Constraint = Union[Frozen, Tangent, OnSurface, OnCurve, KeepLayer, SlideColumns]


class Resolved(NamedTuple):
    """Per-vertex mobility, resolved once.  ``kind[i]`` is ``FREE/ON_SURFACE/ON_CURVE/FROZEN/TANGENT``;
    ``target[i]`` indexes ``surfaces`` or ``curves`` for the first two; ``anchor`` is where a
    frozen vertex lives (the input position, or its one-time projection for ``slide=False``)."""

    kind: IntArray
    target: IntArray
    anchor: PointArray
    surfaces: tuple[Surface, ...]
    curves: tuple[Curve, ...]
    #: Human-readable notes: every conflict and how it was resolved, never silently.
    notes: tuple[str, ...]
    #: ``tie[i]`` is the rigid column vertex ``i`` belongs to (``-1`` none); ``tie_dir[i]`` its slide direction.
    tie: IntArray
    tie_dir: PointArray
    #: ``nbr[i]`` the two boundary neighbours of a ``TANGENT`` vertex (``-1`` for every other vertex).
    nbr: IntArray
    #: ``(F, 4)`` the outward boundary faces a hex ``TANGENT`` vertex with no ``nbr`` slides in the tangent plane of
    #: (empty on a quad mesh, whose ``TANGENT`` vertices all have two boundary neighbours).
    bfaces: IntArray

    def counts(self) -> dict[str, int]:
        return {name: int((self.kind == k).sum()) for k, name in _KIND.items()}


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


def circle_tangent(xp: PointArray, x: PointArray, xn: PointArray) -> PointArray:
    """Unit tangent at ``x`` of the circle through ``xp``, ``x``, ``xn`` (rows), oriented ``xp -> xn``.

    With ``a = xp - x`` and ``c = xn - x`` the tangent at the middle point of the circumscribed circle is
    ``|a|^2 c - |c|^2 a``: perpendicular to the radius, in the plane of the three points, exact on a circle
    and second-order on uneven spacing.  Collinear points give the edge direction; the degenerate case
    ``xp == xn`` (no direction at all) returns a zero row."""
    a, c = xp - x, xn - x
    t = np.einsum("ij,ij->i", a, a)[:, None] * c - np.einsum("ij,ij->i", c, c)[:, None] * a
    n = np.linalg.norm(t, axis=-1, keepdims=True)
    return np.asarray(t / np.where(n > 0, n, 1.0))


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


def turning_deg(P: PointArray, ids: IntArray, nb: IntArray, normals: PointArray | None = None) -> FloatArray:
    """Turning angle (degrees) of the boundary at ``ids``, whose neighbours are ``nb`` ``(n, 2)``.  With
    ``normals`` (``(len(P), 3)``, e.g. :func:`vertex_normals`) both edges are first projected onto the tangent
    plane at the vertex, so the turn is the one *within* the surface and its own bending does not count."""
    u = P[ids] - P[nb[:, 0]]
    v = P[nb[:, 1]] - P[ids]
    if normals is not None:
        nn = normals[ids]
        u = u - np.einsum("ij,ij->i", u, nn)[:, None] * nn
        v = v - np.einsum("ij,ij->i", v, nn)[:, None] * nn
    c = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
    return np.asarray(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def boundary_corners(P: PointArray, adj: dict[int, list[int]], corner_deg: float = CORNER_DEG,
                     normals: PointArray | None = None) -> set[int]:
    """The corners of the boundary graph ``adj`` (vertex -> neighbours along it): a vertex without exactly
    two neighbours, or one whose turn is a **jump** -- above ``corner_deg`` *and* above three times the mean turn of
    its two neighbours.  The second test is what separates a corner from a coarsely sampled smooth curve: on
    a circle of ``N`` points every vertex turns ``360/N``, which can be far above ``corner_deg``, but its
    neighbours turn about the same (a 3:1 ellipse on 12 points peaks at 2.4), so the ratio is near 1; at a real corner the neighbours (edge interiors) turn
    little, so the ratio is large.  (A closed loop of only four points is such a circle.)  ``normals``
    makes the turns geodesic, see :func:`turning_deg`."""
    two = np.array([v for v in sorted(adj) if len(adj[v]) == 2], dtype=np.int64)
    out = {v for v in adj if len(adj[v]) != 2}
    if not two.size:
        return out
    turn = dict(zip(two.tolist(), turning_deg(P, two, np.array([adj[int(v)] for v in two], dtype=np.int64),
                                              normals).tolist()))
    for v in two.tolist():
        if turn[v] > corner_deg:
            around = [turn[w] for w in adj[v] if w in turn]
            if turn[v] > 3.0 * (sum(around) / len(around) if around else 0.0):
                out.add(v)
    return out


def spline_curve(X: PointArray, closed: bool, name: str = "boundary") -> Curve:
    """A smooth :class:`Curve` through the ordered points ``X`` (2-D or 3-D): a cubic spline in the chord-length
    parameter, periodic when ``closed`` and not-a-knot otherwise (a straight line for two points).  The foot
    point is the nearest of a dense sampling refined by a few Newton steps on the spline itself, kept only
    where it lowers the distance, so a projection never ends farther than the sample it started from."""
    X = np.asarray(X, float)
    Y = np.vstack([X, X[:1]]) if closed else X
    t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Y, axis=0), axis=1))]
    total = float(t[-1])
    cs = None if len(Y) == 2 else CubicSpline(t, Y, axis=0, bc_type="periodic" if closed else "not-a-knot")
    p0 = Y[0]
    d0 = (Y[-1] - Y[0]) / total

    def at(tt: FloatArray, der: int = 0) -> PointArray:
        if cs is not None:
            return np.asarray(cs(tt, der))
        return p0 + tt[:, None] * d0 if der == 0 else np.broadcast_to(d0, (len(tt), 3)).copy()

    ts = np.linspace(0.0, total, 64 * (len(Y) - 1) + 1)
    tree = cKDTree(at(ts))

    def foot(P: PointArray) -> FloatArray:
        P = np.asarray(P, float)
        _, j = tree.query(P)
        t0 = ts[j]
        if cs is None:
            return np.asarray(np.clip((P - p0) @ d0, 0.0, total))
        tt = t0.copy()
        for _ in range(8):
            r = at(tt) - P
            d1, d2 = at(tt, 1), at(tt, 2)
            g = np.einsum("ij,ij->i", r, d1)
            gp = np.einsum("ij,ij->i", d1, d1) + np.einsum("ij,ij->i", r, d2)
            tt = tt - g / np.where(np.abs(gp) > 1e-300, gp, 1.0)
            tt = np.mod(tt, total) if closed else np.clip(tt, 0.0, total)
        worse = np.linalg.norm(at(tt) - P, axis=1) > np.linalg.norm(at(t0) - P, axis=1)
        return np.asarray(np.where(worse, t0, tt))

    def tangent(P: PointArray) -> PointArray:
        return _unit(at(foot(P), 1))

    return Curve(lambda P: at(foot(P)), tangent, name)


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




def _targets(r: Resolved) -> list[tuple[int, Sequence[Union[Surface, Curve]]]]:
    """``(kind, targets)`` for the two kinds that project onto a geometric object."""
    return [(ON_SURFACE, r.surfaces), (ON_CURVE, r.curves)]


def snap(points: PointArray, r: Resolved, weight: float = 1.0) -> PointArray:
    """``points`` with every constrained vertex moved ``weight`` of the way to its target
    (``1`` = onto it) and every frozen vertex put back at its anchor.  A new array."""
    X = np.asarray(points, float).copy()
    X[r.kind == FROZEN] = r.anchor[r.kind == FROZEN]
    tied = (r.tie >= 0) & (r.kind == TIED)
    if tied.any():                               # rigid columns: one shared slide parameter per column
        s_i = np.einsum("ij,ij->i", X[tied] - r.anchor[tied], r.tie_dir[tied])
        g = r.tie[tied]
        mean = np.bincount(g, weights=s_i, minlength=int(r.tie.max()) + 1) / np.maximum(
            np.bincount(g, minlength=int(r.tie.max()) + 1), 1)
        X[tied] = r.anchor[tied] + mean[g][:, None] * r.tie_dir[tied]
    for k, objs in _targets(r):
        for t, obj in enumerate(objs):
            m = (r.kind == k) & (r.target == t)
            if m.any():
                Y = obj.project(X[m])
                if not np.isfinite(Y).all():
                    raise ValueError("target %r returned non-finite coordinates for %d of %d vertices; "
                                     "refusing to continue with a corrupted mesh"
                                     % (obj.name, int((~np.isfinite(Y).all(axis=1)).sum()), len(Y)))
                X[m] += weight * (Y - X[m])
    return X


def constrain_direction(d: PointArray, points: PointArray, r: Resolved) -> PointArray:
    """``d`` ``(P, 3)`` (a gradient or a step) with the forbidden part removed, using the
    target's own normal/tangent at each vertex's foot point.  A new array."""
    out = np.asarray(d, float).copy()
    out[r.kind == FROZEN] = 0.0
    for t, s in enumerate(r.surfaces):
        m = (r.kind == ON_SURFACE) & (r.target == t)
        if m.any():
            nrm = s.normal(s.project(points[m]))
            out[m] -= np.einsum("ij,ij->i", out[m], nrm)[:, None] * nrm
    for t, c in enumerate(r.curves):
        m = (r.kind == ON_CURVE) & (r.target == t)
        if m.any():
            tan = c.tangent(c.project(points[m]))
            out[m] = np.einsum("ij,ij->i", out[m], tan)[:, None] * tan
    tg = (r.kind == TANGENT) & (r.nbr[:, 0] >= 0)
    if tg.any():                                 # boundary vertex: along the circle through it and its neighbours
        tan = circle_tangent(points[r.nbr[tg, 0]], points[tg], points[r.nbr[tg, 1]])
        out[tg] = np.einsum("ij,ij->i", out[tg], tan)[:, None] * tan
    pl = (r.kind == TANGENT) & (r.nbr[:, 0] < 0)
    if pl.any():                                 # patch vertex (hex): in the tangent plane of its boundary faces
        F = r.bfaces
        nf = np.cross(points[F[:, 2]] - points[F[:, 0]], points[F[:, 3]] - points[F[:, 1]])
        nf = nf / np.where(np.linalg.norm(nf, axis=1, keepdims=True) > 0, np.linalg.norm(nf, axis=1, keepdims=True), 1.0)
        N = np.zeros_like(points)
        for k in range(4):
            np.add.at(N, F[:, k], nf)
        nv = N[pl]
        nv = nv / np.where(np.linalg.norm(nv, axis=1, keepdims=True) > 0, np.linalg.norm(nv, axis=1, keepdims=True), 1.0)
        out[pl] -= np.einsum("ij,ij->i", out[pl], nv)[:, None] * nv
    tied = (r.tie >= 0) & (r.kind == TIED)
    if tied.any():                               # a rigid column moves by the mean of its members' slide components
        s_i = np.einsum("ij,ij->i", out[tied], r.tie_dir[tied])
        g = r.tie[tied]
        mean = np.bincount(g, weights=s_i, minlength=int(r.tie.max()) + 1) / np.maximum(
            np.bincount(g, minlength=int(r.tie.max()) + 1), 1)
        out[tied] = mean[g][:, None] * r.tie_dir[tied]
    return out


def vertex_length(points: PointArray, corners: IntArray) -> FloatArray:
    """``(P,)`` the shortest edge at each vertex -- the *length unit* every step, finite
    difference and tolerance of the optimizer is measured in (notes/06 section 3)."""
    ell = np.full(points.shape[0], np.inf)
    for k in range(4):
        a, b = corners[:, k], corners[:, (k + 1) % 4]
        e = np.linalg.norm(points[a] - points[b], axis=1)
        np.minimum.at(ell, a, e)
        np.minimum.at(ell, b, e)
    return ell


class OnTargetReport(NamedTuple):
    """The geometry self-check of one state: how far constrained vertices sit from their
    targets and how far frozen ones moved, in absolute units and in local edge lengths."""

    n_checked: int
    max_distance: float
    mean_distance: float
    max_distance_in_edges: float
    max_frozen_move: float
    tol: float
    passed: bool

    def __str__(self) -> str:
        return ("on-target: %d constrained, max dist %.3e (%.2e edges), mean %.3e, "
                "max frozen move %.3e, tol %.1e x edge -> %s"
                % (self.n_checked, self.max_distance, self.max_distance_in_edges,
                   self.mean_distance, self.max_frozen_move, self.tol,
                   "PASS" if self.passed else "FAIL"))


def on_target_report(points: PointArray, corners: IntArray, r: Resolved, *,
                     tol: float = 1e-6) -> OnTargetReport:
    """Distance of every ``ON_SURFACE``/``ON_CURVE`` vertex to its target, and the largest
    displacement of a frozen vertex from its anchor; passes when the former is within ``tol``
    local edge lengths and the latter is exactly ``0`` (to ``tol`` of the edge)."""
    ell = vertex_length(points, corners)
    dist = np.zeros(points.shape[0])
    for k, objs in _targets(r):
        for t, obj in enumerate(objs):
            m = (r.kind == k) & (r.target == t)
            if m.any():
                dist[m] = np.linalg.norm(obj.project(points[m]) - points[m], axis=1)
    tied = (r.kind == TIED) & (r.tie >= 0)
    if tied.any():                               # a tied vertex must stay on anchor + s * direction
        off = points[tied] - r.anchor[tied]
        off = off - np.einsum("ij,ij->i", off, r.tie_dir[tied])[:, None] * r.tie_dir[tied]
        dist[tied] = np.linalg.norm(off, axis=1)
    cons = (r.kind == ON_SURFACE) | (r.kind == ON_CURVE) | tied
    fro = r.kind == FROZEN
    fmove = np.linalg.norm(points[fro] - r.anchor[fro], axis=1)
    rel = dist[cons] / ell[cons] if cons.any() else np.zeros(0)
    fmax = float(fmove.max()) if fmove.size else 0.0
    ok = bool((rel.max() if rel.size else 0.0) <= tol
              and (fmove / ell[fro]).max(initial=0.0) <= tol)
    return OnTargetReport(int(cons.sum()), float(dist[cons].max(initial=0.0)),
                          float(dist[cons].mean()) if cons.any() else 0.0,
                          float(rel.max(initial=0.0)), fmax, tol, ok)


# -- a surface from a triangle mesh (CAD tessellation, scanned wall) -------------------------
def _closest_on_triangles(p: PointArray, a: PointArray, b: PointArray,
                          c: PointArray) -> tuple[PointArray, PointArray]:
    """Closest point of triangle ``abc`` to ``p`` and its barycentric weights, row by row."""
    n = np.cross(b - a, c - a)
    nn = np.maximum(np.einsum("ij,ij->i", n, n), 1e-300)
    wa = np.einsum("ij,ij->i", np.cross(c - b, p - b), n) / nn
    wb = np.einsum("ij,ij->i", np.cross(a - c, p - c), n) / nn
    wc = 1.0 - wa - wb
    inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
    best_q = wa[:, None] * a + wb[:, None] * b + wc[:, None] * c
    best_w = np.stack([wa, wb, wc], axis=1)
    best_d = np.where(inside, np.einsum("ij,ij->i", p - best_q, p - best_q), np.inf)
    for u, v, iu, iv in ((a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)):
        e = v - u
        t = np.clip(np.einsum("ij,ij->i", p - u, e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-300), 0.0, 1.0)
        q = u + t[:, None] * e
        d = np.einsum("ij,ij->i", p - q, p - q)
        w = np.zeros_like(best_w)
        w[:, iu], w[:, iv] = 1.0 - t, t
        better = d < best_d
        best_q = np.where(better[:, None], q, best_q)
        best_w = np.where(better[:, None], w, best_w)
        best_d = np.where(better, d, best_d)
    return best_q, best_w


def surface_from_trimesh(points: PointArray, tris: IntArray, *, k: int = 8,
                         name: str = "trimesh") -> Surface:
    """A :class:`Surface` read off a triangle mesh -- a CAD tessellation or a scanned wall.

    ``project`` returns the closest point on the ``k`` triangles whose centroids are nearest the
    query (exact point-triangle distance within that candidate set); ``normal`` blends the area-weighted
    vertex normals with the barycentric weights of that foot point, so it is continuous across
    triangles.  It is an **approximation** of the true surface: the triangles' chord error remains,
    sharp edges are smoothed over by the blended normal, and a ``k`` that is too small on a very
    graded tessellation can miss the nearest triangle.  Cost: one KD-tree query and ``k`` triangle
    distances per point."""
    from scipy.spatial import cKDTree

    V = np.asarray(points, dtype=float)
    T = np.asarray(tris, dtype=np.int64)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])      # |fn| = 2 * area
    vn = np.zeros_like(V)
    for j in range(3):
        np.add.at(vn, T[:, j], fn)
    vn = _unit(vn)
    tree = cKDTree(V[T].mean(axis=1))
    kk = min(k, T.shape[0])

    def _foot(P: PointArray) -> tuple[PointArray, IntArray, PointArray]:
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        _, cand = tree.query(P, k=kk)
        cand = np.asarray(cand).reshape(P.shape[0], -1)
        rep = np.repeat(P, cand.shape[1], axis=0)
        t = T[cand.ravel()]
        q, w = _closest_on_triangles(rep, V[t[:, 0]], V[t[:, 1]], V[t[:, 2]])
        d = np.einsum("ij,ij->i", rep - q, rep - q).reshape(P.shape[0], -1)
        pick = np.argmin(d, axis=1) + np.arange(P.shape[0]) * cand.shape[1]
        return q[pick], t[pick], w[pick]

    def project(P: PointArray) -> PointArray:
        return _foot(P)[0]

    def normal(P: PointArray) -> PointArray:
        _, t, w = _foot(P)
        return _unit(np.einsum("ik,ikj->ij", w, vn[t]))

    return Surface(project, normal, name)


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


def claimed_surfaces(r: Resolved) -> list[tuple[IntArray, Surface]]:
    """``(vertex ids, surface)`` for every surface that holds vertices in ``r`` -- the groups
    :func:`surface_normals` reads its normals from."""
    out = []
    for t, srf in enumerate(r.surfaces):
        ids = np.flatnonzero((r.kind == ON_SURFACE) & (r.target == t))
        if ids.size:
            out.append((ids, srf))
    return out


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
