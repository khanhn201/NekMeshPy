"""World-aligned in-plane axes for a plane given by its normal.

A private free function that returns an orthonormal ``(e1, e2)`` frame spanning the
plane with the given normal, so a factory can place a planar loop (``circle`` /
``rectangle`` / ``far_field_box``) in any plane. The axes are world-aligned so an
axis-aligned plane is not rotated.
"""

from __future__ import annotations

import numpy as np

from .._typing import Vec3


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
