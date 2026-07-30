"""Shared internals for the planar :class:`~nekmeshpy.LineMesh` factories.

``_in_plane_axes`` returns an orthonormal ``(e1, e2)`` frame spanning the plane with a
given normal, so a factory can place a planar loop (``circle`` / ``rectangle``) or an
open curve (``arc``) in any plane. The axes are world-aligned so an axis-aligned plane
is not rotated.

``_arc_points`` / ``_arc_interior`` place nodes on the **exact** circle of a given
radius in that frame; ``circle`` (closed, ``_closed.py``) and ``arc`` (open,
``_open.py``) share them so both put their high-order GLL nodes on the true arc rather
than on the chord.
"""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3
from ..model.fields import gll_nodes


def _in_plane_axes(normal: Vec3) -> tuple[Vec3, Vec3]:
    """Orthonormal in-plane axes ``(e1, e2)`` for a unit ``normal``, world-aligned
    so an axis-aligned plane is unrotated. ``e2 = normal x e1``."""
    n: Vec3 = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref: Vec3 = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1: Vec3 = ref - np.dot(ref, n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2: Vec3 = np.cross(n, e1)
    return e1, e2


def _arc_points(radius: float, center: Point, e1: Vec3, e2: Vec3,
                th: FloatArray) -> PointArray:
    """``(len(th), 3)`` points on the exact circle of ``radius`` about ``center`` in
    the ``(e1, e2)`` frame, one per angle in ``th`` (measured from ``e1``)."""
    local = radius * np.cos(th)[:, None] * e1 + radius * np.sin(th)[:, None] * e2
    pts: PointArray = center + local
    return pts


def _arc_interior(radius: float, center: Point, e1: Vec3, e2: Vec3,
                  seg_th: FloatArray, dth: float, order: int) -> PointArray:
    """``(len(seg_th), order-1, 3)`` private high-order nodes for the line elements
    that start at the angles ``seg_th`` and span ``dth`` radians each.

    Element ``l`` carries the ``order-1`` interior GLL nodes of its own arc, placed on
    the **true circle** (its two endpoint nodes are the corners in ``points``), so a
    high-order export renders the exact arc instead of the chord."""
    ang = seg_th[:, None] + gll_nodes(order)[1:order][None, :] * dth
    arc = (radius * np.cos(ang)[:, :, None] * e1
           + radius * np.sin(ang)[:, :, None] * e2)          # (L, order-1, 3)
    interior: PointArray = center + arc
    return interior
