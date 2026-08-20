"""Shape factories for the ``LineMesh`` rung -- the ones owning a *shape model* rather
than being generic over any input."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3
from ..core.paths import Path
from ..core.surfaces import SurfaceCurve, SurfaceMap
from ..core.tags import ElementTags
from ._plane import _arc_interior, _arc_points, _in_plane_axes
from .assemble import _eval_curve, loft, loft_fn
from .linemesh import LineMesh


def line(start: Point, end: Point, fractions: float | FloatArray, *,
         element_tag: str = "", first_tag: str | None = None,
         last_tag: str | None = None, order: int = 1) -> LineMesh:
    """A straight open line from ``start`` to ``end`` sampled at normalized arc-length
    ``fractions`` in ``[0, 1]`` (``0`` = start, ``1`` = end): the graded-edge sibling of
    ``circle``/``rectangle``.

    ``first_tag`` / ``last_tag`` name the two end **points**, as on
    :func:`loft <nekmeshpy.linemesh.assemble.loft>` -- a slice at this rung is a single
    point.  That is what makes an end addressable by
    :func:`linemesh.attach <nekmeshpy.linemesh.assemble.attach>`."""
    frac = np.atleast_1d(np.asarray(fractions, dtype=float))
    s: Point = np.asarray(start, dtype=float).ravel()
    e: Point = np.asarray(end, dtype=float).ravel()
    pts = s + frac[:, None] * (e - s)
    # the segment is straight, so ``loft``'s default straight GLL interior is exact
    return loft(pts, loop=False, element_tags=element_tag or None, order=order,
                first_tag=first_tag, last_tag=last_tag)


def arc(radius: float, n: int, *,
        center: Point = (0.0, 0.0, 0.0),
        normal: Vec3 = (0.0, 0.0, 1.0),
        start_theta: float = 0.0,
        end_theta: float = np.pi,
        element_tag: str = "",
        first_tag: str | None = None,
        last_tag: str | None = None,
        order: int = 1) -> LineMesh:
    """An **open** circular arc: ``n`` line elements over ``n+1`` points evenly spaced
    in angle from ``start_theta`` to ``end_theta`` on the circle of ``radius`` about
    ``center``, in the plane with the given ``normal`` (default ``+z``).

    ``first_tag`` / ``last_tag`` name the two end **points**, which is what makes an end
    addressable by :func:`linemesh.attach <nekmeshpy.linemesh.assemble.attach>`."""
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
        return loft(pts, element_tags=element_tag or None,
                    first_tag=first_tag, last_tag=last_tag)
    # element l spans th[l] .. th[l] + dth; its private interior rides the exact arc,
    # overriding ``loft``'s default straight chord blend.
    interior: PointArray = _arc_interior(
        radius, c, e1, e2, th[:-1], (e - s) / ni, order)     # (n, order-1, 3)
    return loft(pts, interior=interior, element_tags=element_tag or None,
                first_tag=first_tag, last_tag=last_tag, order=order)


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
    ``fractions``, with no further scaling (``loft_fn(f, arclength_fractions(f, n,
    t_range=...), order=N)``)."""
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
    junction** of a piecewise path, subdividing each piece at roughly ``target``."""
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


def path_fractions(path: Path, *, target_length: float | None = None,
                   layers: int | None = None,
                   fractions: FloatArray | Sequence[float] | None = None) -> FloatArray:
    """Resolve a :class:`Path <nekmeshpy.core.paths.Path>` and exactly one of
    ``target_length`` / ``layers`` / ``fractions`` into the sweep stations themselves.
    """
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
               element_tag: str = "") -> LineMesh:
    """Mesh a :class:`SurfaceCurve <nekmeshpy.core.surfaces.SurfaceCurve>` by
    evaluating ``surface`` on it -- one element between consecutive nodes of
    ``curve.fr``, exact on the surface at **every** node."""
    return loft_fn(lambda x: surface(curve.g(x)), curve.fr, order=order,
                   element_tags=element_tag or None)


def circle(radius: float, n: int, *,
           center: Point = (0.0, 0.0, 0.0),
           normal: Vec3 = (0.0, 0.0, 1.0),
           start_theta: float = 0.0,
           element_tag: str = "",
           order: int = 1) -> LineMesh:
    """A closed loop of ``n`` points evenly spaced on a circle of ``radius`` about
    ``center`` in the plane with the given ``normal`` (default ``+z``)."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(start_theta)
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    pts: PointArray = _arc_points(radius, c, e1, e2, th)
    if order == 1:
        return loft(pts, loop=True, element_tags=element_tag or None)
    # element l spans angle th[l] .. th[l] + dth; place the interior GLL nodes on
    # the arc (the two endpoint nodes are already the loop's corner points), so the
    # explicit ``interior`` overrides ``loft``'s default straight chord blend.
    interior: PointArray = _arc_interior(
        radius, c, e1, e2, th, 2.0 * np.pi / n, order)             # (n, order-1, 3)
    return loft(pts, loop=True, interior=interior,
                element_tags=element_tag or None, order=order)


def rectangle(width: float, height: float, n: int, *,
              center: Point = (0.0, 0.0, 0.0),
              normal: Vec3 = (0.0, 0.0, 1.0),
              side_tags: Mapping[str, str] | None = None,
              order: int = 1) -> LineMesh:
    """A closed rectangle loop of the given ``width`` x ``height`` about ``center`` in
    the plane with the given ``normal`` (default ``+z``), discretized into ``n`` line
    elements (``n`` must be a positive multiple of 4): ``n // 4`` evenly spaced per
    side, running CCW from the lower-left corner (bottom / right / top / left), corners
    always landing on a point."""
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
    lm = loft(pts, loop=True, order=order)
    if tags is None:
        return lm
    return LineMesh(lm.point_mesh, lm.lines, lm.interior,
                    ElementTags.from_dense(tags))

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
