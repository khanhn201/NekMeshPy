"""Best-fit plane frame for a coplanar 3-D point set.

The section factories that build a grid *inside a boundary* (butterfly O-grid,
annulus ring blend) are naturally 2-D algorithms, but a boundary may live in any
plane in 3-D.  These private free functions map a coplanar ``(P,3)`` point set
into an orthonormal in-plane frame and back, so the 2-D algorithm runs in the
boundary's own plane instead of a flattened ``xy`` copy.

The in-plane axes are chosen **world-aligned** so an already axis-aligned
boundary is not rotated: for a normal ``n``, ``e1`` is the world axis most
orthogonal to ``n`` projected into the plane, and ``e2 = n x e1``.  Hence a
boundary in the ``xy`` plane (``n = +z``) gets ``e1 = +x``, ``e2 = +y`` (the
identity, so ``ogrid``/``annulus`` reproduce their old ``xy`` output exactly),
and a boundary in the ``yz`` plane (``n = +x``) gets ``e1 = +y``, ``e2 = +z``.
"""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3


def _in_plane_axes(normal: Vec3) -> tuple[Vec3, Vec3]:
    """Orthonormal in-plane axes ``(e1, e2)`` for a unit ``normal``, world-aligned
    so an axis-aligned plane is unrotated (``+z`` -> ``+x,+y``; ``+x`` ->
    ``+y,+z``).  ``e1`` is the world axis most orthogonal to ``normal`` projected
    into the plane; ``e2 = normal x e1``."""
    n: Vec3 = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref: Vec3 = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1: Vec3 = ref - np.dot(ref, n) * n
    e1 = e1 / np.linalg.norm(e1)
    e2: Vec3 = np.cross(n, e1)
    return e1, e2


def plane_frame(pts: PointArray) -> tuple[Point, Vec3, Vec3, Vec3]:
    """Best-fit plane frame of coplanar ``(P,3)`` ``pts``: returns
    ``(centroid, e1, e2, normal)``.  ``normal`` is computed by **Newell's method**
    over the point sequence (consistent with loop winding, robust for near-planar
    loops) and normalized; ``e1``/``e2`` are the world-aligned in-plane axes."""
    P: PointArray = np.asarray(pts, dtype=float).reshape(-1, 3)
    centroid: Point = P.mean(axis=0)
    nxt = np.roll(P, -1, axis=0)
    # Newell's method: area-weighted normal of the (possibly non-planar) polygon.
    normal: Vec3 = np.array([
        np.sum((P[:, 1] - nxt[:, 1]) * (P[:, 2] + nxt[:, 2])),
        np.sum((P[:, 2] - nxt[:, 2]) * (P[:, 0] + nxt[:, 0])),
        np.sum((P[:, 0] - nxt[:, 0]) * (P[:, 1] + nxt[:, 1])),
    ])
    normal = normal / np.linalg.norm(normal)
    e1, e2 = _in_plane_axes(normal)
    return centroid, e1, e2, normal


def to_plane(pts: PointArray, centroid: Point, e1: Vec3, e2: Vec3) -> FloatArray:
    """Project ``(P,3)`` ``pts`` into plane coordinates ``(P,2)``:
    ``[(p-centroid).e1, (p-centroid).e2]``."""
    d: PointArray = np.asarray(pts, dtype=float).reshape(-1, 3) - centroid
    return np.column_stack([d @ e1, d @ e2])


def from_plane(uv: FloatArray, centroid: Point, e1: Vec3, e2: Vec3) -> PointArray:
    """Lift plane coordinates ``(P,2)`` ``uv`` back to ``(P,3)`` world points:
    ``centroid + u*e1 + v*e2``."""
    a: FloatArray = np.asarray(uv, dtype=float).reshape(-1, 2)
    return centroid + a[:, 0:1] * e1 + a[:, 1:2] * e2
