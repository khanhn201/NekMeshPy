"""Fixed-arity ``HexMesh`` operations that raise a rung (delta +1).

``extrude`` sweeps one section straight; ``annulus`` fills between two index-paired
closed surfaces; ``from_grid`` builds a block from one structured point grid.  All
three are thin: they position profiles and hand them to
:func:`~nekmeshpy.hexmesh._assemble.loft`, which owns the index space, so the numbering
they expose is the loft's carried up unchanged.

Free functions bound onto :class:`~nekmeshpy.HexMesh` by ``hexmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``HexMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import (
    FloatArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..model.fields import validate_layers
from ..quadmesh import QuadMesh
from ..quadmesh._lift import from_grid as quad_from_grid
from ..quadmesh._morph import blend as quad_blend
from ..quadmesh._morph import translate
from ._assemble import loft
from .hexmesh import _GRID_SIDES, _ORIGIN, _Z_AXIS, HexMesh


def extrude(
    section: QuadMesh,
    *,
    axis: Vec3 = _Z_AXIS,
    length: float,
    layers: FloatArray,
    origin: Point = _ORIGIN,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> HexMesh:
    """Sweep a single quad ``section`` a distance ``length`` along ``axis`` into
    a hex block.

    The section is translated rigidly along ``axis`` (its placement is
    preserved); ``origin`` shifts the whole block. ``layers`` are the normalized
    copy-plane positions in ``[0, 1]``, strictly increasing, last ``1``;
    ``layers[0]`` is the near cap and ``layers.size - 1`` hex layers span it to
    ``1``. ``first_tag`` / ``last_tag`` name the caps. The straight special case
    of ``loft``."""
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
    return loft(slices, first_tag=first_tag, last_tag=last_tag)

def annulus(
    inner: QuadMesh,
    outer: QuadMesh,
    radial: FloatArray,
    *,
    inner_tag: str = "",
    outer_tag: str = "",
) -> HexMesh:
    """Shell O-grid filling the region between an inner and an outer closed quad
    surface (e.g. a sphere inside a cubic far-field box).

    The two surfaces are paired by index: equal point count ``P`` and identical
    ``quads`` connectivity, with point ``p`` of ``inner`` joined radially to point
    ``p`` of ``outer``. ``radial`` are the shell positions in ``[0, 1]``, strictly
    increasing; ``radial[0]`` is the inner shell and the last is ``1``, so
    ``radial.size - 1`` shell layers blend inner -> outer directly in 3-D.

    Wall faces are tagged from the surfaces' per-quad ``element_tags`` (a closed
    surface has no free boundary edges): inner caps face 5, outer caps face 6. A
    non-empty scalar ``inner_tag`` / ``outer_tag`` overrides and names the whole
    wall."""
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
    # wall tags from the surfaces' per-quad element_tags; scalar arg overrides
    inner_caps: str | StrArray = (
        inner_tag if inner_tag
        else (inner.element_tags if inner.element_group_tags else ""))
    outer_caps: str | StrArray = (
        outer_tag if outer_tag
        else (outer.element_tags if outer.element_group_tags else ""))
    return loft(shells, first_tag=inner_caps, last_tag=outer_caps)

def from_grid(
    P: PointArray,
    *,
    face_tags: dict[str, str] | None = None,
    element_tag: str = "",
    order: int = 1,
) -> HexMesh:
    """Build hexes from a structured point grid ``P`` ``(ni+1,nj+1,nk+1,3)``.
    ``face_tags`` maps side names (``x_min``/``x_max``/``y_min``/``y_max``/
    ``z_min``/``z_max``) to boundary names on the six outer sides; a side left out
    or mapped to ``NO_BOUNDARY`` emits no boundary row. ``element_tag`` is written
    to every hex's ``element_tags``.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each hex carries ``(order+1)**3`` straight-sided (trilinear) GLL nodes.

    Built as a :meth:`loft` of the grid's **``k``-sections**: section ``k`` is
    the :meth:`QuadMesh.from_grid <nekmeshpy.quadmesh.QuadMesh.from_grid>` of the
    slab ``P[:, :, k, :]`` (itself a ``LineMesh`` loft), and the sweep runs
    ``k = 0..nk``.  Every tagged side rides a channel the rung below already has:
    the section's four ``edge_tags`` become the ``x_min`` / ``x_max`` / ``y_min`` /
    ``y_max`` swept side faces (section side ``s`` -> hex face ``s``, which is
    exactly the ``_GRID_SIDES`` Nek face numbering) and the sweep's caps the
    ``z_min`` / ``z_max`` ones; ``element_tag`` rides the section's per-quad tags.
    So corners, shared edges and shared faces all come out of the layer-by-layer
    B-rep assembly instead of a ``unique_edges`` re-derivation.

    **Ordering is the loft's, carried up unchanged** -- composing the rung below
    means accepting its numbering, so nothing is relabelled here.  The grid is
    numbered ``i`` fastest, ``k`` slowest: grid node ``(i, j, k)`` is point
    ``(k*(nj+1) + j)*(ni+1) + i`` and grid cell ``(i, j, k)`` is hex
    ``(k*nj + j)*ni + i``, i.e. ``points`` equals
    ``P.transpose(2, 1, 0, 3).reshape(-1, 3)`` -- *not* the ``P.reshape(-1, 3)``
    (``k``-fastest) order this factory used historically.  ``boundaries`` stays
    lexsorted by ``(element, face)``, so its row order follows the hex ids; each
    tagged row still names the same physical side."""
    P = np.asarray(P, dtype=float)
    tags = {s: n for s, n in (face_tags or {}).items() if n}
    for side in tags:
        _GRID_SIDES[side]        # reject an unknown side name (KeyError)
    # x/y sides are the section's own edges; z sides are the sweep's end caps.
    edge_tags = {s: n for s, n in tags.items() if _GRID_SIDES[s][0] == "side"}
    slices = [quad_from_grid(P[:, :, k, :], edge_tags=edge_tags,
                                 element_tag=element_tag, order=order)
              for k in range(P.shape[2])]
    # the loft *is* the result: its sweep-major numbering is carried up unchanged.
    return loft(slices, first_tag=tags.get("z_min", ""),
                    last_tag=tags.get("z_max", ""))


#: Rung-raising combinators bound onto ``HexMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "extrude": extrude,
    "annulus": annulus,
    "from_grid": from_grid,
}
