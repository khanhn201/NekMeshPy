"""Fixed-arity ``QuadMesh`` operations that raise a rung (delta +1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np

from .._typing import (
    FloatArray,
    Point,
    PointArray,
    Vec3,
)
from ..core import frames, stations
from ..core.fields import validate_layers
from ..core.paths import Orientation, SpacePath, UpSpec, resolve_frame, sample_up
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from ..linemesh.assemble import loft as line_loft
from ..linemesh.morph import blend as line_blend
from ..linemesh.morph import transform as line_transform
from ..linemesh.morph import translate
from ..linemesh.shape import path_fractions
from ..pointmesh import PointMesh
from ._helpers import _check_boundary
from .assemble import _loft_evaluated, loft
from .quadmesh import _GRID_SIDES, _ORIGIN, _Z_AXIS, QuadMesh


def extrude(
    line: LineMesh,
    length: float,
    layers: int | FloatArray,
    *,
    axis: Vec3 = _Z_AXIS,
    origin: Point = _ORIGIN,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> QuadMesh:
    """Sweep a ``LineMesh`` a distance ``length`` along ``axis`` into a quad section
    (the straight special case of :func:`loft <nekmeshpy.quadmesh.assemble.loft>`)."""
    axis_u: Vec3 = np.asarray(axis, dtype=float)
    axis_u = axis_u / np.linalg.norm(axis_u)
    offsets = validate_layers(layers, "extrude layers") * float(length)
    # the sweep is a rigid translation, so it *is* the rung-preserving ``translate``:
    # every node -- corner and private high-order interior alike -- rides the same
    # vector, and the tags / connectivity come through verbatim.  Each slice reuses
    # ``line.lines``, so the swept strip wraps exactly when the input curve's own
    # connectivity does -- there is no flag to carry.
    placed = translate(line, np.asarray(origin, dtype=float))
    slices = [translate(placed, d * axis_u) for d in offsets]
    return loft(slices, element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag)

def annulus(inner: LineMesh, outer: LineMesh, radial: int | FloatArray, *,
            inner_tag: str | None = None, outer_tag: str | None = None,
            ) -> QuadMesh:
    """Ring O-grid filling the region between an inner and an outer closed loop -- e.g.
    a circular body inside a square far-field box."""
    radial = validate_layers(radial, "annulus radial")
    A: PointArray = _check_boundary(inner, "annulus inner", 3)   # (N,3)
    B: PointArray = _check_boundary(outer, "annulus outer", 3)   # (N,3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "annulus: inner and outer loops must have equal point counts "
            "(got %d, %d); build both with the same count, "
            "e.g. LineMesh.rectangle(w, h, N) against circle(r, N)"
            % (A.shape[0], B.shape[0]))
    if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
        raise ValueError("annulus: inner and outer loops touch or cross")
    order = inner.order
    if outer.order != order:
        raise ValueError("annulus: inner and outer loops must share the same order")

    # tags from each loop's per-segment element_tags, which a scalar inner_tag /
    # outer_tag overrides for the whole ring.  ``None`` is "not asked for" and
    # inherits; ``NO_TAG`` is an explicit override *to* untagged, so it suppresses
    # the loop's own tags rather than falling through to them.
    inner_caps: str | ElementTags | None = (
        inner_tag if inner_tag is not None else inner.element_tags or None)
    outer_caps: str | ElementTags | None = (
        outer_tag if outer_tag is not None else outer.element_tags or None)

    # Blend the loops (carrying their curved blocks) and loft directly -- ring k =
    # blend_ho(inner, outer, t_k), so a high-order annulus is curved throughout, not
    # just on the two walls; loft builds the curved Coons columns.  blend copies the
    # ring topology but drops element_tags (the wall tags ride in inner_caps /
    # outer_caps as the loft's cap tags).
    rings = line_blend(inner, outer, radial)
    return loft(rings, first_tag=inner_caps, last_tag=outer_caps)

def from_grid(
    P: PointArray,
    *,
    side_tags: Mapping[str, str] | None = None,
    element_tag: str = "",
    order: int = 1,
) -> QuadMesh:
    """Build quads from a structured point grid ``P`` ``(ni+1,nj+1,3)``."""
    P = np.asarray(P, dtype=float)
    _, nj1, _ = P.shape
    tags = {s: n for s, n in (side_tags or {}).items() if n}
    for side in tags:
        _GRID_SIDES[side]        # reject an unknown side name (KeyError)

    # -- the column profile, shared by every level -----------------------
    # tagged profile end points -> the two swept walls (loft: vertex 1 -> quad side
    # 4, vertex 2 -> side 2), which is exactly x_min / x_max.
    pnamed = np.full(P.shape[0], "", dtype=object)
    for side in ("x_min", "x_max"):
        if side in tags:
            # loft carries the profile's first point onto quad side 4, its last onto 2
            pnamed[0 if _GRID_SIDES[side][1] == 4 else -1] = tags[side]
    pbnd_t = ElementTags.from_dense(np.asarray(pnamed, dtype=np.str_))
    # each profile is itself a ``LineMesh.loft`` of its ``i`` points: the rung below
    # builds the open ``i = 0..ni`` chain and, at order > 1, each segment's private
    # interior as the straight GLL blend of its two endpoints.  ``loft`` here builds
    # the sweep-direction rungs the same way and the quad interiors as the Coons
    # patch of the two, so a flat grid cell stays exact.
    def _profile(j: int) -> LineMesh:
        lm = line_loft(P[:, j, :], order=order)
        return LineMesh(PointMesh(lm.points, pbnd_t), lm.lines, lm.interior)
    slices = [_profile(j) for j in range(nj1)]
    # the loft *is* the result: its sweep-major numbering is carried up unchanged.
    return loft(slices, element_tags=element_tag or None,
                first_tag=tags.get("y_min"), last_tag=tags.get("y_max"))


def sweep(
    profile: LineMesh,
    path: Callable[[FloatArray], PointArray],
    fractions: FloatArray,
    *,
    origin: Point | Sequence[float],
    tangent: Callable[[FloatArray], PointArray] | None = None,
    orientation: Orientation = "transport",
    up: UpSpec | None = None,
    twist: float = 0.0,
    close_twist: bool = True,
    normal: Vec3 | Sequence[float] | None = None,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> QuadMesh:
    """A strip swept from one ``LineMesh`` ``profile`` along the curve ``path``."""
    order = profile.order
    _, t = stations.sweep_lattice(fractions, order, loop=loop, name="sweep")
    tv: FloatArray = t[:-1] if loop else t
    P, T = stations.sweep_path(path, tangent, tv)
    places = frames.sweep_placements(
        profile.points, P, orientation=orientation, up=sample_up(up, tv), twist=twist,
        close_twist=close_twist, loop=loop, origin=origin, normal=normal,
        path_tangents=T)
    profs: list[LineMesh] = [line_transform(profile, M, o) for M, o in places]
    if loop:
        profs.append(profs[0])          # the seam profile *is* the first placement
    return _loft_evaluated(profs, order, loop=loop, element_tags=element_tags,
                           first_tag=first_tag, last_tag=last_tag, name="sweep")


def sweep_path(
    profile: LineMesh,
    path: SpacePath,
    *,
    origin: Point | Sequence[float],
    target_length: float | None = None,
    layers: int | None = None,
    fractions: FloatArray | Sequence[float] | None = None,
    orientation: Orientation | None = None,
    up: UpSpec | None = None,
    twist: float = 0.0,
    close_twist: bool = True,
    normal: Vec3 | Sequence[float] | None = None,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> QuadMesh:
    """:func:`sweep <nekmeshpy.quadmesh.lift.sweep>` driven by a :class:`SpacePath
    <nekmeshpy.core.paths.SpacePath>`, which carries its own analytic tangent and
    junction table -- so this asks for an element length along the sweep instead of a
    station array.

    A path that also carries its own frame (anything from :func:`paths.walk
    <nekmeshpy.core.paths.walk>`) needs no ``orientation``: with none asked for, that
    frame is held per station, so a bend out of plane and a distributed ``roll`` arrive
    here as authored.  Naming an ``orientation`` (or an ``up``) overrides it."""
    fr = path_fractions(path, target_length=target_length, layers=layers,
                        fractions=fractions)
    orientation, up = resolve_frame(path, orientation, up)
    return sweep(profile, path.centerline, fr, origin=origin, tangent=path.tangent,
                 orientation=orientation, up=up, twist=twist, close_twist=close_twist,
                 normal=normal, loop=loop, element_tags=element_tags,
                 first_tag=first_tag, last_tag=last_tag)


__all__ = [
    "annulus",
    "extrude",
    "from_grid",
    "sweep",
    "sweep_path",
]
