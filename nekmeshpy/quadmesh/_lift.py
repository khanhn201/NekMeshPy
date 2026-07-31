"""Fixed-arity ``QuadMesh`` operations that raise a rung (delta +1).

``extrude`` sweeps one ``LineMesh``; ``annulus`` fills between two index-paired closed
loops; ``from_grid`` builds a section from one structured point grid.  All three are
thin: they position profiles and hand them to
:func:`~nekmeshpy.quadmesh._assemble.loft`, which owns the index space, so the
numbering they expose is the loft's carried up unchanged.  ``annulus`` lives here --
not in ``_open.py`` -- because it owns no *shape* model: it is generic over whatever
two conformal loops it is handed, exactly like its sibling one rung up,
:func:`~nekmeshpy.hexmesh._lift.annulus`.  (The region fills -- ``structured`` /
``ogrid`` / ``half_ogrid`` / ``spined_ogrid`` -- are also delta +1 but do own a shape
model, so they stay in ``_open.py``.)

Free functions bound onto :class:`~nekmeshpy.QuadMesh` by ``quadmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``QuadMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._typing import (
    FloatArray,
    IntArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..linemesh import LineMesh
from ..linemesh._assemble import loft as line_loft
from ..linemesh._morph import blend as line_blend
from ..linemesh._morph import translate
from ..model.fields import validate_layers
from ._assemble import loft
from ._helpers import _apply_smoothing, _check_boundary
from .quadmesh import _GRID_SIDES, _ORIGIN, _Z_AXIS, QuadMesh


def extrude(
    line: LineMesh,
    *,
    axis: Vec3 = _Z_AXIS,
    length: float,
    layers: FloatArray,
    origin: Point = _ORIGIN,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> QuadMesh:
    """Sweep a ``LineMesh`` a distance ``length`` along ``axis`` into a quad
    section (the straight special case of :meth:`loft`).

    The ``line`` is translated rigidly along ``axis``; ``origin`` shifts the whole
    section.  ``layers`` are normalized positions along ``axis`` (strictly
    increasing in ``[0, 1]`` with the last ``1``), giving ``layers.size - 1``
    layers.  The line's ``element_tags`` ride onto the swept quads and its tagged
    boundary points onto the side-wall edges; ``first_tag`` / ``last_tag`` name
    the near / far cap edges."""
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
    return loft(slices, first_tag=first_tag, last_tag=last_tag)

def annulus(inner: LineMesh, outer: LineMesh, radial: FloatArray, *,
            smoothing_method: str | None = None,
            inner_tag: str = "", outer_tag: str = "",
            ) -> QuadMesh:
    """Ring O-grid filling the region between an inner and an outer closed loop
    -- e.g. a circular body inside a square far-field box.

    The two loops are paired by index: they must carry the same number of points
    ``N``, and point ``i`` of ``inner`` joins radially to point ``i`` of
    ``outer`` (no resampling; build the outer loop with the same point count and
    aligned index 0, e.g. a ``LineMesh.rectangle(w, h, N)`` box against a
    ``LineMesh.circle(r, N, start_theta=...)`` body).  ``radial`` are the ring positions with
    the initial position explicit (strictly increasing in ``[0, 1]``,
    ``radial[0]`` = inner ring, last = ``1`` = outer loop), giving
    ``radial.size - 1`` ring layers.  ``smoothing_method`` relaxes the ring
    interior with the inner/outer rings held fixed.

    Boundary tags come from the loops' per-line ``element_tags`` (each ring edge
    tagged from the matching loop segment, so a named box splits the outer ring
    into distinct sides).  A non-empty scalar ``inner_tag`` / ``outer_tag``
    overrides that for the whole inner / outer ring.

    The rings are always a genuine high-order blend: ``LineMesh.blend`` interpolates
    the two loops' curved blocks (``blend_ho``) so every ring carries curved
    tangential edges, and :meth:`loft` sweeps them radially -- the sibling of
    :meth:`HexMesh.annulus <nekmeshpy.hexmesh.HexMesh.annulus>` one dimension down.
    A repositioning ``smoothing_method`` (``conduction`` / ``winslow``) relaxes the
    corner grid, which cannot ride a curved block, so it is rejected at ``order > 1``
    (high-order smoothing is not implemented -- use ``order=1`` or drop the smoother);
    at ``order 1`` it relaxes the ring interior as usual.

    Built by :meth:`loft`-ing the blended rings; the inner / outer rings are the
    loft's near / far caps.  Gives ``N x (radial.size - 1)`` quads."""
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

    # tags from each loop's per-segment element_tags; a non-empty scalar
    # inner_tag / outer_tag overrides that for the whole ring.
    inner_caps: str | StrArray = (
        inner_tag if inner_tag
        else (inner.element_tags if inner.element_group_tags else ""))
    outer_caps: str | StrArray = (
        outer_tag if outer_tag
        else (outer.element_tags if outer.element_group_tags else ""))

    # Blend the loops (carrying their curved blocks) and loft directly -- ring k =
    # blend_ho(inner, outer, t_k), so a high-order annulus is curved throughout, not
    # just on the two walls; loft builds the curved Coons columns.  blend copies the
    # ring topology but drops element_tags (the wall tags ride in inner_caps /
    # outer_caps as the loft's cap tags).  A requested repositioning smoother rejects
    # order > 1 (high-order smoothing not implemented); at order 1 it relaxes the
    # linear loft as before.
    rings = line_blend(inner, outer, radial)
    qm = loft(rings, first_tag=inner_caps, last_tag=outer_caps)
    return _apply_smoothing(qm, smoothing_method)

def from_grid(
    P: PointArray,
    *,
    edge_tags: Mapping[str, str] | None = None,
    element_tag: str = "",
    order: int = 1,
) -> QuadMesh:
    """Build quads from a structured point grid ``P`` ``(ni+1,nj+1,3)``.
    ``edge_tags`` maps side names (``x_min`` / ``x_max`` / ``y_min`` / ``y_max``)
    to boundary tags on the four outer edges; a side left out (or mapped to
    ``NO_BOUNDARY``) emits no boundary row.  ``element_tag`` is written to every
    quad's dense ``element_tags``.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each quad carries ``(order+1)**2`` straight-sided GLL nodes (a flat grid cell
    is exact under this subdivision).

    Built as a :meth:`loft` of the grid's **column profiles**, each of which is
    itself a :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` of the
    grid's ``i`` points -- profile ``j`` is the open chain lofted from
    ``P[:, j, :]`` running ``i = 0..ni`` -- and the sweep runs ``j = 0..nj``, so
    the whole ladder ``HexMesh.from_grid -> from_grid -> LineMesh.loft`` is
    composed at every rung.  That is the orientation whose column quad
    ``[a_j, b_j, b_{j+1}, a_{j+1}]`` *is* the grid cell's CCW corner list
    ``[(i,j), (i+1,j), (i+1,j+1), (i,j+1)]``, and it routes all four tagged sides
    through channels ``loft`` already has: the profile's tagged **end points**
    become the ``x_min`` / ``x_max`` walls and the sweep's caps the ``y_min`` /
    ``y_max`` ones.  So the shared edges come out of the layer-by-layer B-rep
    assembly rather than a ``unique_edges`` re-derivation from corners -- this rung
    is composed from the one below like every other.

    **Ordering is the loft's, carried up unchanged** -- composing the rung below
    means accepting its numbering, so nothing is relabelled here.  The grid is
    therefore numbered **sweep-major** (``i`` fastest): grid node ``(i, j)`` is
    point ``j*(ni+1) + i`` and grid cell ``(i, j)`` is quad ``j*ni + i``, i.e.
    ``points`` equals ``P.transpose(1, 0, 2).reshape(-1, 3)`` -- *not* the
    ``P.reshape(-1, 3)`` (``j``-fastest) order this factory used historically.
    ``boundaries`` stays lexsorted by ``(quad, side)``, so its row order follows
    the quad ids; each tagged row still names the same physical side."""
    P = np.asarray(P, dtype=float)
    ni1, nj1, _ = P.shape
    ni = ni1 - 1
    tags = {s: n for s, n in (edge_tags or {}).items() if n}
    for side in tags:
        _GRID_SIDES[side]        # reject an unknown side name (KeyError)

    # -- the column profile, shared by every level -----------------------
    # np.full width-infers from the fill value (dtype=np.str_ would clip to <U1)
    line_tags: StrArray = np.full(ni, element_tag)
    # tagged profile end points -> the two swept walls (loft: vertex 1 -> quad side
    # 4, vertex 2 -> side 2), which is exactly x_min / x_max.
    pbnd: list[list[int]] = []
    pnames: list[str] = []
    for side in ("x_min", "x_max"):
        if side in tags:
            # loft carries profile end point side 1 -> quad side 4, side 2 -> 2
            vertex = 1 if _GRID_SIDES[side][1] == 4 else 2
            pbnd.append([0 if vertex == 1 else ni - 1, vertex])
            pnames.append(tags[side])
    pbnd_a: IntArray = np.asarray(pbnd, dtype=np.int64).reshape(-1, 2)
    # each profile is itself a ``LineMesh.loft`` of its ``i`` points: the rung below
    # builds the open ``i = 0..ni`` chain and, at order > 1, each segment's private
    # interior as the straight GLL blend of its two endpoints.  ``loft`` here builds
    # the sweep-direction rungs the same way and the quad interiors as the Coons
    # patch of the two, so a flat grid cell stays exact.
    slices = [line_loft(P[:, j, :], element_tags=line_tags,
                            boundaries=pbnd_a, boundary_tags=pnames, order=order)
              for j in range(nj1)]
    # the loft *is* the result: its sweep-major numbering is carried up unchanged.
    return loft(slices, first_tag=tags.get("y_min", ""),
                    last_tag=tags.get("y_max", ""))


#: Rung-raising combinators bound onto ``QuadMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "extrude": extrude,
    "annulus": annulus,
    "from_grid": from_grid,
}
