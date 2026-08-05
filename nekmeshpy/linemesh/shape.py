"""Shape factories for the ``LineMesh`` rung -- the ones owning a *shape model*
rather than being generic over any input.

Open and closed shapes are merged into one namespace here: the split was a
storage distinction, not a caller-facing one.  Each factory meshes its input
**exactly** -- no factory resamples what it is given -- which is why the
samplings a caller has to derive for one live here beside it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .._typing import FloatArray, Point, PointArray, StrArray, Vec3
from ..model.paths import SpacePath
from ..model.surfaces import SurfaceCurve, SurfaceMap
from ._plane import _arc_interior, _arc_points, _in_plane_axes
from .assemble import _eval_curve, loft, loft_fn
from .linemesh import LineMesh


def line(start: Point, end: Point, fractions: float | FloatArray, *,
         element_tag: str = "", order: int = 1) -> LineMesh:
    """A straight open line from ``start`` to ``end`` sampled at normalized
    arc-length ``fractions`` in ``[0, 1]`` (``0`` = start, ``1`` = end): the
    graded-edge sibling of ``circle``/``rectangle``. The points are placed
    exactly at ``start + f*(end - start)`` -- no resampling. ``element_tag``
    names every resulting line element (e.g. to tag a structured edge as one
    wall).

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each line element carries ``order+1`` GLL nodes placed on the straight
    segment -- the two endpoints are the corners in ``points``, and the
    ``order-1`` nodes strictly between them are the straight GLL blend
    :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>` places by default
    (read by high-order ``vtu`` export)."""
    frac = np.atleast_1d(np.asarray(fractions, dtype=float))
    s: Point = np.asarray(start, dtype=float).ravel()
    e: Point = np.asarray(end, dtype=float).ravel()
    pts = s + frac[:, None] * (e - s)
    tags = [element_tag] * (pts.shape[0] - 1) if element_tag else None
    # the segment is straight, so ``loft``'s default straight GLL interior is exact
    return loft(pts, loop=False, element_tags=tags, order=order)


def arc(radius: float, n: int, *,
        center: Point = (0.0, 0.0, 0.0),
        normal: Vec3 = (0.0, 0.0, 1.0),
        start_theta: float = 0.0,
        end_theta: float = np.pi,
        element_tags: StrArray | Sequence[str] | None = None,
        order: int = 1) -> LineMesh:
    """An **open** circular arc: ``n`` line elements over ``n+1`` points evenly
    spaced in angle from ``start_theta`` to ``end_theta`` on the circle of
    ``radius`` about ``center``, in the plane with the given ``normal`` (default
    ``+z``).  Point ``k`` sits at ``start_theta + k*(end_theta - start_theta)/n``
    measured from the in-plane ``e1`` axis, so ``end_theta < start_theta`` runs the
    arc clockwise.  ``element_tags`` (length ``n``) tags its line elements at
    construction, e.g. to name the whole arc ``wall`` for a section factory.

    This is the open sibling of
    :func:`linemesh.shape.circle <nekmeshpy.linemesh.shape.circle>` -- the analytic
    curve to hand to :func:`QuadMesh.structured <nekmeshpy.quadmesh.shape.structured>` (or to
    weld into a composite edge with
    :func:`linemesh.assemble.merge <nekmeshpy.linemesh.assemble.merge>`) instead of sampling
    points and calling :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>`,
    which can only subdivide straight between the samples.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1`` each
    element carries ``order+1`` GLL nodes placed on the **true arc** (not the
    chord) -- the two endpoints are the corners in ``points`` and the ``order-1``
    nodes strictly between them are the element's private ``interior``, so a
    high-order ``vtu`` export renders the exact arc.

    ``circle`` does **not** delegate here: over a full turn the angular step is
    exactly ``2*pi/n``, while this factory must form ``(end_theta - start_theta)/n``
    -- and ``(s + 2*pi) - s`` is not ``2*pi`` in floating point for a general ``s``,
    so delegating would move ``circle``'s nodes by a ulp.  The two share the node
    placement instead (``_plane._arc_points`` / ``_arc_interior``)."""
    ni = int(n)
    if ni < 1:
        raise ValueError("arc needs n >= 1 elements, got %d" % ni)
    s, e = float(start_theta), float(end_theta)
    if s == e:
        raise ValueError("arc needs start_theta != end_theta (got %g twice)" % s)
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    th: FloatArray = np.linspace(s, e, ni + 1)
    pts: PointArray = _arc_points(radius, c, e1, e2, th)
    if order == 1:
        return loft(pts, element_tags=element_tags)
    # element l spans th[l] .. th[l] + dth; its private interior rides the exact arc,
    # overriding ``loft``'s default straight chord blend.
    interior: PointArray = _arc_interior(
        radius, c, e1, e2, th[:-1], (e - s) / ni, order)     # (n, order-1, 3)
    return loft(pts, interior=interior, element_tags=element_tags,
                         order=order)


def _arclength_params(f: Callable[[FloatArray], PointArray], t0: float, t1: float,
                      samples: int) -> tuple[FloatArray, FloatArray]:
    """``(t_dense, s_dense)`` -- ``samples`` parameters spanning ``[t0, t1]`` and their
    normalized cumulative **chord** length in ``[0, 1]``, the table used to invert
    arc length into the curve parameter."""
    td: FloatArray = np.linspace(t0, t1, samples)
    Pd: PointArray = _eval_curve(f, td)
    s: FloatArray = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))])
    if s[-1] <= 0.0:
        raise ValueError(
            "curve has zero length over t_range -- cannot space by arc length")
    return td, s / s[-1]


def arclength_fractions(f: Callable[[FloatArray], PointArray], n: int, *,
                        t_range: tuple[float, float] = (0.0, 1.0),
                        samples: int = 1001) -> FloatArray:
    """The ``(n+1,)`` **parameter values** spanning ``t_range`` -- from ``t_range[0]``
    to ``t_range[1]`` -- at which ``f`` must be evaluated for the resulting ``n+1``
    points to be evenly spaced by **arc length**: hand the result straight to
    :func:`linemesh.assemble.loft_fn <nekmeshpy.linemesh.assemble.loft_fn>` as its
    ``fractions``, with no further scaling
    (``loft_fn(f, arclength_fractions(f, n, t_range=...), order=N)``).

    ``t_range`` is the parameter interval to invert over -- unlike ``loft_fn``, this
    helper genuinely needs a domain, because the chord table is built by sampling it
    densely.  A descending range needs no special handling: the returned values simply
    run from ``t_range[0]`` down to ``t_range[1]``, which meshes the curve backwards.

    The inversion goes through a cumulative **chord**-length table of ``samples`` dense
    evaluations of ``f``, so only *where along* the curve the nodes end up inherits that
    table's discretization error.  Every node of the resulting mesh still lies on the
    curve to machine precision, because ``loft_fn`` places it by evaluating ``f`` and
    never by interpolating this table -- raise ``samples`` for a more even spacing, not
    for a more accurate curve."""
    ni = int(n)
    if ni < 1:
        raise ValueError("arclength_fractions needs n >= 1 elements, got %d" % ni)
    if samples < 2:
        raise ValueError("arclength_fractions needs samples >= 2, got %d" % samples)
    t0, t1 = float(t_range[0]), float(t_range[1])
    if t0 == t1:
        raise ValueError(
            "arclength_fractions needs t_range endpoints to differ (got %g twice)" % t0)
    td, s = _arclength_params(f, t0, t1, samples)
    t: FloatArray = np.interp(np.linspace(0.0, 1.0, ni + 1), s, td)
    return t


def sweep_fractions(breaks: FloatArray | Sequence[float], total_length: float,
                    target: float) -> FloatArray:
    """Normalized sweep stations in ``[0, 1]`` that put a node **exactly on every
    junction** of a piecewise path, subdividing each piece at roughly ``target``.

    ``breaks`` are the cumulative arc-length positions (in the same units as
    ``total_length``, strictly ascending, strictly inside ``(0, total_length)``) of the
    path's interior junctions -- for a turtle-walked centerline, ``cumsum(seg_len)``
    with its first and last entries dropped.  ``target`` is the desired element length
    along the sweep.

    Each interval between consecutive breaks (and the two end intervals against ``0``
    and ``total_length``) is split into ``max(1, round(interval / target))`` equal
    steps *on its own*, so the breaks reappear in the output bit-for-bit rather than
    being approached by a global ``linspace``.  That is the whole point: a path's
    curvature is piecewise constant and **jumps** at a junction, so an element that
    straddled one would be fitted across two different geometries -- visible as a kink
    in the wall of a swept bend.  The result is strictly ascending, opens with ``0.0``,
    contains every ``break / total_length``, and closes with ``1.0``.

    Hand it straight to the ``fractions`` of
    :func:`HexMesh.sweep <nekmeshpy.hexmesh.lift.sweep>` /
    :func:`QuadMesh.sweep <nekmeshpy.quadmesh.lift.sweep>` (or of
    :func:`linemesh.assemble.loft_fn <nekmeshpy.linemesh.assemble.loft_fn>`) whose path is
    parametrized by normalized arc length.  Like
    :func:`arclength_fractions <nekmeshpy.linemesh.shape.arclength_fractions>` it is a ``HELPERS`` entry, not a factory: it answers a
    question about a sweep's input contract and returns a plain array, since the sweep
    itself meshes exactly at the stations given."""
    L = float(total_length)
    if not L > 0.0:
        raise ValueError("sweep_fractions needs total_length > 0, got %g" % L)
    tgt = float(target)
    if not tgt > 0.0:
        raise ValueError("sweep_fractions needs target > 0, got %g" % tgt)
    br: FloatArray = np.asarray(breaks, dtype=float).ravel()
    if br.size and (br[0] <= 0.0 or br[-1] >= L):
        raise ValueError(
            "sweep_fractions breaks must lie strictly inside (0, %g) -- they are the "
            "path's *interior* junctions, and 0 / total_length are always stations "
            "anyway (got %g .. %g)" % (L, br[0], br[-1]))
    if br.size > 1 and not np.all(np.diff(br) > 0.0):
        raise ValueError(
            "sweep_fractions breaks must be strictly ascending cumulative arc lengths")
    s: FloatArray = br / L
    pieces: list[FloatArray] = []
    for a, b in zip(np.concatenate([[0.0], s]), np.concatenate([s, [1.0]])):
        # round-to-nearest on the piece's own length, floored at one element: a piece
        # shorter than ``target`` still gets an element rather than vanishing.
        n = max(1, int(round((b - a) * L / tgt)))
        pieces.append(np.linspace(a, b, n + 1)[:-1])   # drop the shared end station
    out: FloatArray = np.concatenate(pieces + [np.array([1.0])])
    return out


def path_fractions(path: SpacePath, *, target_length: float | None = None,
                   layers: int | None = None,
                   fractions: FloatArray | Sequence[float] | None = None) -> FloatArray:
    """Resolve a :class:`SpacePath <nekmeshpy.model.paths.SpacePath>` and exactly one of
    ``target_length`` / ``layers`` / ``fractions`` into the sweep stations themselves.

    ``target_length`` and ``layers`` both go through
    :func:`sweep_fractions <nekmeshpy.linemesh.shape.sweep_fractions>` on the path's own
    ``break_fractions``, so every straight<->arc junction still carries a station;
    ``layers`` is just ``target_length = total_length / layers``, which is the *average*
    element length, not a guaranteed count -- each piece is rounded on its own.
    ``fractions`` hands stations in verbatim, for a path graded piece by piece rather
    than to one length (a U-turn given its own layer count, say).

    Factored out of the ``sweep_path`` at each rung so the three-way choice is
    spelled and validated once."""
    given = [n for n, x in (("target_length", target_length), ("layers", layers),
                            ("fractions", fractions)) if x is not None]
    if len(given) != 1:
        raise ValueError(
            "path_fractions: give exactly one of target_length / layers / fractions, "
            "got %s" % (", ".join(given) if given else "none"))
    if fractions is not None:
        return np.asarray(fractions, dtype=float).ravel()
    L = float(path.total_length)
    if layers is not None:
        n = int(layers)
        if n < 1:
            raise ValueError("path_fractions: layers must be >= 1, got %d" % n)
        target_length = L / n
    return sweep_fractions(np.asarray(path.break_fractions, dtype=float) * L, L,
                           float(target_length))   # type: ignore[arg-type]


def on_surface(curve: SurfaceCurve, surface: SurfaceMap, *, order: int = 1,
               element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
    """Mesh a :class:`SurfaceCurve <nekmeshpy.model.surfaces.SurfaceCurve>` by
    evaluating ``surface`` on it -- one element between consecutive nodes of
    ``curve.fr``, exact on the surface at **every** node.

    The surface map reaches the private GLL interiors as well as the corners, because
    this is a :func:`loft_fn <nekmeshpy.linemesh.assemble.loft_fn>` and not a
    ``loft`` of sampled points: sampling the curve into an array first would
    straight-subdivide between the samples and put the interior nodes off the surface
    at ``order > 1``.

    ``curve.fr`` may descend, which traverses the curve backwards."""
    return loft_fn(lambda x: surface(curve.g(x)), curve.fr, order=order,
                   element_tags=element_tags)


def circle(radius: float, n: int, *,
           center: Point = (0.0, 0.0, 0.0),
           normal: Vec3 = (0.0, 0.0, 1.0),
           start_theta: float = 0.0,
           element_tags: StrArray | Sequence[str] | None = None,
           order: int = 1) -> LineMesh:
    """A closed loop of ``n`` points evenly spaced on a circle of ``radius``
    about ``center`` in the plane with the given ``normal`` (default ``+z``).
    Point ``k`` sits at angle ``2*pi*k/n + start_theta`` from the in-plane
    ``e1`` axis, so ``start_theta`` rotates the whole loop -- e.g. to align its
    index 0 with a :func:`rectangle <nekmeshpy.linemesh.shape.rectangle>` far-field box's lower-left corner before an
    index-paired :func:`QuadMesh.annulus <nekmeshpy.quadmesh.lift.annulus>`.
    ``element_tags`` tags the loop's line elements at construction.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each arc element carries ``order+1`` GLL nodes placed on the **true circle**
    (not the chord) -- the two endpoints are the corners in ``points`` and the
    ``order-1`` nodes strictly between them are built here, still on the exact
    circle, as the element's private ``interior``, so a high-order ``vtu`` export
    renders the exact arc.

    The open sibling is :func:`linemesh.shape.arc <nekmeshpy.linemesh.shape.arc>`; the
    two share the node
    placement (``_plane._arc_points`` / ``_arc_interior``) but not the angle sampling:
    a full turn's step is exactly ``2*pi/n`` here, whereas ``arc`` must form
    ``(end_theta - start_theta)/n``, so ``circle`` keeps its own ``linspace`` rather
    than delegating (see ``arc``'s docstring)."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(start_theta)
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    pts: PointArray = _arc_points(radius, c, e1, e2, th)
    if order == 1:
        return loft(pts, loop=True, element_tags=element_tags)
    # element l spans angle th[l] .. th[l] + dth; place the interior GLL nodes on
    # the arc (the two endpoint nodes are already the loop's corner points), so the
    # explicit ``interior`` overrides ``loft``'s default straight chord blend.
    interior: PointArray = _arc_interior(
        radius, c, e1, e2, th, 2.0 * np.pi / n, order)             # (n, order-1, 3)
    return loft(pts, loop=True, interior=interior,
                         element_tags=element_tags, order=order)


def rectangle(width: float, height: float, n: int, *,
              center: Point = (0.0, 0.0, 0.0),
              normal: Vec3 = (0.0, 0.0, 1.0),
              side_tags: Mapping[str, str] | None = None,
              order: int = 1) -> LineMesh:
    """A closed rectangle loop of the given ``width`` x ``height`` about
    ``center`` in the plane with the given ``normal`` (default ``+z``),
    discretized into ``n`` line elements (``n`` must be a positive multiple of
    4): ``n // 4`` evenly spaced per side, running CCW from the lower-left corner
    (bottom / right / top / left), corners always landing on a point.

    With ``n`` points it feeds
    :func:`QuadMesh.annulus <nekmeshpy.quadmesh.lift.annulus>` directly as
    the outer far-field loop against a ``circle(radius, n)`` body -- rotate the
    circle with ``start_theta`` so its index 0 meets the lower-left corner
    (``atan2(-height, -width)``) and the two loops pair index-for-index (the
    radial spokes are not straight, but the mesh conforms).

    ``side_tags`` (keyed ``bottom`` / ``right`` / ``top`` / ``left``) names each
    side's line elements; an absent key leaves that side untagged and
    ``side_tags=None`` leaves the whole loop untagged.  The keys -- rather than a
    positional 4-sequence -- are what make this spelling identical to its one-rung-up
    twin :func:`QuadMesh.rectangle <nekmeshpy.quadmesh.shape.rectangle>`, which
    takes the same keyword with the same four names; an unrecognized key is a loud
    ``ValueError`` because a silent typo would otherwise just lose a wall.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each element carries ``order+1`` GLL nodes on its straight side -- the two
    endpoints are the corners in ``points`` and the ``order-1`` nodes strictly
    between them are the straight GLL blend
    :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>` places by default
    (read by high-order ``vtu`` export)."""
    ni = int(n)
    if ni < 4 or ni % 4 != 0:
        raise ValueError("rectangle n must be a positive multiple of 4, "
                         "got %d" % ni)
    m = ni // 4
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    hw, hh = width / 2.0, height / 2.0
    bl = c - hw * e1 - hh * e2
    br = c + hw * e1 - hh * e2
    tr = c + hw * e1 + hh * e2
    tl = c - hw * e1 + hh * e2
    f = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]  # corner..next, open

    def _side(p: Point, q: Point) -> PointArray:
        return p + f * (q - p)
    pts = np.concatenate([_side(bl, br), _side(br, tr),
                          _side(tr, tl), _side(tl, bl)])
    tags: list[str] | None = None
    if side_tags is not None:
        sides = ("bottom", "right", "top", "left")   # the loop's CCW traversal order
        for key in side_tags:
            if key not in sides:
                raise ValueError(
                    "rectangle side_tags key must be one of "
                    "bottom/right/top/left, got %r" % key)
        tags = [t for side in sides for t in [side_tags.get(side, "")] * m]
    # every side is straight, so ``loft``'s default straight GLL interior is exact
    return loft(pts, loop=True, element_tags=tags, order=order)

__all__ = [
    "arc",
    "arclength_fractions",
    "circle",
    "line",
    "on_surface",
    "path_fractions",
    "rectangle",
    "sweep_fractions",
]
