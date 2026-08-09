"""Fixed-arity ``HexMesh`` operations that raise a rung (delta +1)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Callable, Literal

import numpy as np
from scipy.spatial import cKDTree

from .._typing import (
    FloatArray,
    IntArray,
    Point,
    PointArray,
    Vec3,
)
from ..core import frames, stations
from ..core.fields import validate_layers
from ..core.paths import SpacePath
from ..core.tags import ElementTags
from ..linemesh.shape import path_fractions
from ..quadmesh import QuadMesh
from ..quadmesh.lift import from_grid as quad_from_grid
from ..quadmesh.morph import blend as quad_blend
from ..quadmesh.morph import reindex as quad_reindex
from ..quadmesh.morph import rotate as quad_rotate
from ..quadmesh.morph import transform as quad_transform
from ..quadmesh.morph import translate
from ..quadmesh.ports import Port
from ..quadmesh.query import plane_normal as quad_plane_normal
from .assemble import _loft_evaluated, loft
from .hexmesh import _GRID_SIDES, _ORIGIN, _Z_AXIS, HexMesh

_log = logging.getLogger(__name__)


def extrude(
    section: QuadMesh,
    length: float,
    layers: int | FloatArray,
    *,
    axis: Vec3 = _Z_AXIS,
    origin: Point = _ORIGIN,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> HexMesh:
    """Sweep a single quad ``section`` a distance ``length`` along ``axis`` into a hex
    block."""
    axis_u: Vec3 = np.asarray(axis, dtype=float)
    axis_u = axis_u / np.linalg.norm(axis_u)
    offsets = validate_layers(layers, "extrude layers") * float(length)
    # the sweep is a rigid translation, so it *is* the rung-preserving ``translate``:
    # every node -- corners, shared edge-interior, private quad-interior -- rides the
    # same vector (and ``origin`` shifts all three alike), because that map is itself
    # composed down the ladder.  Each slice reuses the section's B-rep verbatim (same
    # edge LineMesh connectivity, same per-quad edge indices / flips); only
    # coordinates move.
    placed = translate(section, np.asarray(origin, dtype=float))
    slices = [translate(placed, d * axis_u) for d in offsets]
    return loft(slices, element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag)

def annulus(
    inner: QuadMesh,
    outer: QuadMesh,
    radial: int | FloatArray,
    *,
    inner_tag: str | None = None,
    outer_tag: str | None = None,
) -> HexMesh:
    """Shell O-grid filling the region between an inner and an outer closed quad surface
    (e.g. a sphere inside a cubic far-field box)."""
    radial = validate_layers(radial, "annulus radial")
    A: PointArray = np.asarray(inner.points, dtype=float).reshape(-1, 3)
    B: PointArray = np.asarray(outer.points, dtype=float).reshape(-1, 3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "annulus: inner and outer surfaces must have equal point counts "
            "(got %d, %d); build one from the other's points so they pair by "
            "index" % (A.shape[0], B.shape[0]))
    if not np.array_equal(inner.quads, outer.quads):
        raise ValueError(
            "annulus: inner and outer surfaces must share identical quad "
            "connectivity (they are paired by index)")
    if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
        raise ValueError("annulus: inner and outer surfaces touch or cross")

    # shell t is the straight-chord blend inner -> outer sharing inner's quads;
    # consecutive shells loft into hex layers.
    shells = quad_blend(inner, outer, radial)
    # wall tags from the surfaces' per-quad element_tags, which a scalar arg
    # overrides.  ``None`` is "not asked for" and inherits; ``NO_TAG`` is an
    # explicit override *to* untagged, so it suppresses the surface's own tags
    # rather than falling through to them.
    inner_caps: str | ElementTags | None = (
        inner_tag if inner_tag is not None else inner.element_tags or None)
    outer_caps: str | ElementTags | None = (
        outer_tag if outer_tag is not None else outer.element_tags or None)
    return loft(shells, first_tag=inner_caps, last_tag=outer_caps)

def from_grid(
    P: PointArray,
    *,
    side_tags: Mapping[str, str] | None = None,
    element_tag: str = "",
    order: int = 1,
) -> HexMesh:
    """Build hexes from a structured point grid ``P`` ``(ni+1,nj+1,nk+1,3)``."""
    P = np.asarray(P, dtype=float)
    tags = {s: n for s, n in (side_tags or {}).items() if n}
    for side in tags:
        _GRID_SIDES[side]        # reject an unknown side name (KeyError)
    # x/y sides are the section's own edges; z sides are the sweep's end caps.
    side_map = {s: n for s, n in tags.items() if _GRID_SIDES[s][0] == "side"}
    slices = [quad_from_grid(P[:, :, k, :], side_tags=side_map,
                             element_tag=element_tag, order=order)
              for k in range(P.shape[2])]
    # the loft *is* the result: its sweep-major numbering is carried up unchanged.
    return loft(slices, first_tag=tags.get("z_min"),
                last_tag=tags.get("z_max"))


def sweep(
    section: QuadMesh,
    path: Callable[[FloatArray], PointArray],
    fractions: FloatArray,
    *,
    origin: Point | Sequence[float],
    tangent: Callable[[FloatArray], PointArray] | None = None,
    orientation: Literal["transport", "fixed", "frenet"] = "transport",
    up: Vec3 | Sequence[float] | PointArray | None = None,
    twist: float = 0.0,
    close_twist: bool = True,
    normal: Vec3 | Sequence[float] | None = None,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> HexMesh:
    """A block swept from one ``QuadMesh`` ``section`` along the curve ``path`` -- a
    round pipe bent through a 90-degree elbow or a U-turn, from one O-grid disc."""
    order = section.order
    _, t = stations.sweep_lattice(fractions, order, loop=loop, name="sweep")
    tv: FloatArray = t[:-1] if loop else t
    P, T = stations.sweep_path(path, tangent, tv)
    places = frames.sweep_placements(
        section.points, P, orientation=orientation, up=up, twist=twist,
        close_twist=close_twist, loop=loop, origin=origin, normal=normal,
        path_tangents=T)
    profs: list[QuadMesh] = [quad_transform(section, M, o) for M, o in places]
    if loop:
        profs.append(profs[0])          # the seam profile *is* the first placement
    return _loft_evaluated(profs, t, order, loop=loop, element_tags=element_tags,
                           first_tag=first_tag, last_tag=last_tag, name="sweep")


def sweep_path(
    section: QuadMesh,
    path: SpacePath,
    *,
    origin: Point | Sequence[float],
    target_length: float | None = None,
    layers: int | None = None,
    fractions: FloatArray | Sequence[float] | None = None,
    orientation: Literal["transport", "fixed", "frenet"] = "transport",
    up: Vec3 | Sequence[float] | PointArray | None = None,
    twist: float = 0.0,
    close_twist: bool = True,
    normal: Vec3 | Sequence[float] | None = None,
    loop: bool = False,
    element_tags: str | ElementTags | None = None,
    first_tag: str | ElementTags | None = None,
    last_tag: str | ElementTags | None = None,
) -> HexMesh:
    """:func:`sweep <nekmeshpy.hexmesh.lift.sweep>` driven by a :class:`SpacePath
    <nekmeshpy.core.paths.SpacePath>` rather than by a loose ``(centerline, tangent,
    fractions)`` triple."""
    fr = path_fractions(path, target_length=target_length, layers=layers,
                        fractions=fractions)
    return sweep(section, path.centerline, fr, origin=origin, tangent=path.tangent,
                 orientation=orientation, up=up, twist=twist, close_twist=close_twist,
                 normal=normal, loop=loop, element_tags=element_tags,
                 first_tag=first_tag, last_tag=last_tag)


def _find_roll(a: QuadMesh, b: QuadMesh, axis: Vec3 | Sequence[float]) -> int:
    """The quarter turn ``k`` about ``axis`` that minimizes the index-wise deviation
    between ``a`` and ``b`` about their own centres."""
    ca, cb = a.points.mean(axis=0), b.points.mean(axis=0)
    best_k, best_d = 0, np.inf
    for k in range(4):
        cand = quad_rotate(a, k * np.pi / 2.0, axis=axis, center=ca)
        d = float(np.linalg.norm((cand.points - ca) - (b.points - cb), axis=1).max())
        if d < best_d:
            best_k, best_d = k, d
    return best_k


def _self_map(a: QuadMesh, k: int, axis: Vec3 | Sequence[float]) -> IntArray:
    """``a``'s own near-4-fold self-map under ``k`` quarter turns: which of ``a``'s own
    points does point ``i`` land closest to, rotated.  Entirely about ``a``'s own
    geometry -- nothing to do with the section it is about to be paired against."""
    ca = a.points.mean(axis=0)
    cand = quad_rotate(a, k * np.pi / 2.0, axis=axis, center=ca)
    _, sigma = cKDTree(cand.points).query(a.points)
    return np.asarray(sigma, dtype=np.int64)


def adapter(a: QuadMesh | Port, b: QuadMesh | Port, *,
            axis: Vec3 | Sequence[float] | None = None, layers: int = 2,
            max_deviation: float = 0.2, radius_tol: float = 0.05) -> HexMesh:
    """A short block morphing between two same-connectivity sections whose *node
    patterns* differ slightly -- and whose **both** end faces are bit-exact."""
    if layers < 1:
        raise ValueError("adapter: layers must be >= 1, got %d" % layers)
    sec_a = a.section if isinstance(a, Port) else a
    sec_b = b.section if isinstance(b, Port) else b
    pa, a_stated = _as_port(a, sec_b.points.mean(axis=0), "adapter")
    pb, b_stated = _as_port(b, sec_a.points.mean(axis=0), "adapter")
    _check_facing(pa, pb, a_stated and b_stated, "adapter", radius_tol)
    if axis is None:
        if not a_stated:
            raise ValueError(
                "adapter: give axis=, the direction the 90-degree roll between the two "
                "index patterns is measured about (normally the seam normal). It "
                "defaults to a's own normal only when a is a Port, which states one.")
        axis = pa.normal
    a, b = sec_a, sec_b
    ca, cb = a.points.mean(axis=0), b.points.mean(axis=0)
    k = _find_roll(a, b, axis)
    b_aligned = quad_reindex(a, b, _self_map(a, k, axis))
    dev = float(np.linalg.norm((b_aligned.points - cb) - (a.points - ca), axis=1).max())
    _log.debug("adapter: roll k=%d, residual index-wise deviation %.3e", k, dev)
    if dev > max_deviation:
        raise ValueError(
            "adapter: no 90-degree roll aligns these two node patterns -- the best of "
            "the four leaves an index-wise deviation of %.3g about each section's own "
            "centre (max_deviation %.3g). Blending across it would twist the block "
            "into inverted elements; use bridge() for patterns this far apart."
            % (dev, max_deviation))
    return loft(quad_blend(a, b_aligned, np.linspace(0.0, 1.0, layers + 1)))


def _as_port(x: QuadMesh | Port, toward: Point, who: str) -> tuple[Port, bool]:
    """``(port, was_stated)`` -- promote a bare ``QuadMesh`` by *guessing* its outward
    direction as the one pointing at ``toward``, or take a ``Port``'s stated one."""
    if isinstance(x, Port):
        return x, True
    n = quad_plane_normal(x, check=False)
    c = x.points.mean(axis=0)
    d = np.asarray(toward, dtype=float) - c
    n = n if float(n @ d) > 0.0 else -n
    r = float(np.linalg.norm(np.asarray(x.points, dtype=float) - c, axis=1).max())
    return Port(x, n, c, r), False


def _check_facing(pa: Port, pb: Port, both_stated: bool, who: str,
                  radius_tol: float) -> None:
    """The two checks a guessed direction cannot make.  Skipped unless *both* sides
    stated theirs -- a guess is derived from the very geometry being checked, so
    checking it would only ever confirm itself."""
    if not both_stated:
        return
    if not pa.faces(pb):
        raise ValueError(
            "%s: the two ports do not face each other (normals %s and %s, cosine "
            "%+.3f). A connector between them would have to fold back through one of "
            "the components; reverse whichever port is stated the wrong way round."
            % (who, np.array2string(pa.normal, precision=3),
               np.array2string(pb.normal, precision=3),
               float(pa.normal @ pb.normal)))
    big = max(pa.radius, pb.radius)
    if abs(pa.radius - pb.radius) > radius_tol * big:
        raise ValueError(
            "%s: the two ports are different sizes (radius %.6g and %.6g, %.1f%% "
            "apart). This joins same-radius sections; blend or loft between them "
            "instead if a taper is what you want."
            % (who, pa.radius, pb.radius,
               100.0 * abs(pa.radius - pb.radius) / big))


def _stub_sections(disc: QuadMesh, direction: Vec3, distance: float,
                   count: int) -> list[QuadMesh]:
    """``count`` copies of ``disc``'s own exact pattern, rigidly carried a total
    ``distance`` along ``direction`` from its own centroid."""
    c = disc.points.mean(axis=0)
    if count < 2 or distance <= 0.0:
        return [disc]
    up = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (1.0, 0.0, 0.0)
    s = np.linspace(0.0, 1.0, count)
    P: PointArray = c + s[:, None] * distance * direction
    T: PointArray = np.tile(direction, (count, 1))
    places = frames.sweep_placements(disc.points, P, orientation="fixed", up=up,
                                     origin=c, path_tangents=T)
    return [quad_transform(disc, M, o) for M, o in places]


def bridge(a: QuadMesh | Port, b: QuadMesh | Port, *, layers: int = 4,
           stub_fraction: float = 0.3, stub_max: float = 1.5, blend_layers: int = 6,
           radius_tol: float = 0.05) -> HexMesh:
    """A connector between two same-radius sections whose node patterns are too far
    apart for :func:`adapter <nekmeshpy.hexmesh.lift.adapter>` -- two legs of different
    T-junctions, built by different algorithms."""
    if blend_layers < 1:
        raise ValueError("bridge: blend_layers must be >= 1, got %d" % blend_layers)
    sec_a = a.section if isinstance(a, Port) else a
    sec_b = b.section if isinstance(b, Port) else b
    pa, a_stated = _as_port(a, sec_b.points.mean(axis=0), "bridge")
    pb, b_stated = _as_port(b, sec_a.points.mean(axis=0), "bridge")
    _check_facing(pa, pb, a_stated and b_stated, "bridge", radius_tol)
    ca, cb = sec_a.points.mean(axis=0), sec_b.points.mean(axis=0)
    length = float(np.linalg.norm(cb - ca))
    stub = min(stub_max, stub_fraction * length)
    n_stub = max(2, layers // 2)
    a_secs = _stub_sections(sec_a, pa.normal, stub, n_stub)
    b_secs_raw = _stub_sections(sec_b, pb.normal, stub, n_stub)[::-1]

    a_end, b_end = a_secs[-1], b_secs_raw[0]
    _, sigma = cKDTree(b_end.points - b_end.points.mean(axis=0)).query(
        a_end.points - a_end.points.mean(axis=0))
    if len(set(sigma.tolist())) != sigma.size:
        raise ValueError(
            "bridge: nearest-neighbour matching of the two node patterns is not a "
            "permutation -- they are too dissimilar to pair one-for-one, so the blend "
            "would collapse several of one section's nodes onto one of the other's")
    b_secs = [quad_reindex(a_end, s, sigma) for s in b_secs_raw]
    blend_secs = quad_blend(a_end, b_secs[0], np.linspace(0.0, 1.0, blend_layers + 1))
    return loft(a_secs[:-1] + blend_secs + b_secs[1:])


__all__ = [
    "adapter",
    "annulus",
    "bridge",
    "extrude",
    "from_grid",
    "sweep",
    "sweep_path",
]
