"""Closed 3-D surface :class:`~nekmeshpy.QuadMesh` factories: watertight surfaces
built by welding face patches (``box`` / ``sphere``).

These are plain free functions returning a ``QuadMesh``; ``quadmesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use ``QuadMesh.box(...)`` /
``QuadMesh.sphere(...)``.  They build on the core ``from_grid`` / ``merge``
constructors, referenced by a lazy in-function import to avoid the import cycle with
``quadmesh.py`` (which the package imports to assemble the class).  Internal toolkit
code calls these free functions directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray, IntArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from .quadmesh import QuadMesh

# the six box faces: outward normal n with right-handed tangents (u x v = n),
# each mapped to its {x,y,z}_{min,max} side key.
_BOX_FACES = [
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "x_max"),
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "x_min"),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "y_max"),
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "y_min"),
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "z_max"),
    ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), "z_min"),
]


def box(half_sizes: float | Sequence[float] | FloatArray,
        n: int | Sequence[int] | IntArray, *,
        face_tags: Mapping[str, str] | None = None,
        order: int = 1) -> QuadMesh:
    """Closed box surface centred at the origin: six quad patches welded with
    :meth:`merge`.  ``half_sizes`` is a scalar (cube) or ``(sx, sy, sz)``; ``n``
    is a scalar or ``(nx, ny, nz)`` cells per axis.  ``face_tags`` (keyed
    ``x_min`` / ``x_max`` / ... / ``z_max``) writes each face's dense per-quad
    ``element_tags`` -- e.g. the far-field side it forms; an absent face stays
    untagged so ``merge`` welds shared edges cleanly.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each flat face patch carries ``(order+1)**2`` straight-sided GLL nodes (exact,
    the faces are planar)."""
    from .quadmesh import QuadMesh
    hs: FloatArray = np.asarray(half_sizes, dtype=float).ravel()
    if hs.size == 1:
        hs = np.full(3, float(hs[0]))
    elif hs.size != 3:
        raise ValueError("half_sizes must be a scalar or 3 values (sx, sy, sz)")
    na: IntArray = np.asarray(n, dtype=np.int64).ravel()
    if na.size == 1:
        n_axis = (int(na[0]), int(na[0]), int(na[0]))
    elif na.size == 3:
        n_axis = (int(na[0]), int(na[1]), int(na[2]))
    else:
        raise ValueError("n must be a scalar or 3 counts (nx, ny, nz)")
    ft = face_tags or {}
    patches: list[QuadMesh] = []
    for nrm, u, v, key in _BOX_FACES:
        nv: FloatArray = np.asarray(nrm, dtype=float)
        uv: FloatArray = np.asarray(u, dtype=float)
        vv: FloatArray = np.asarray(v, dtype=float)
        au = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(uv)))] + 1)
        av = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(vv)))] + 1)
        A: FloatArray
        B: FloatArray
        A, B = np.meshgrid(au, av, indexing="ij")
        face = hs * (nv + A[..., None] * uv + B[..., None] * vv)
        patches.append(QuadMesh.from_grid(face, element_tag=ft.get(key, ""),
                                          order=order))
    return QuadMesh.merge(patches)


def sphere(radius: float, n: int | Sequence[int] | IntArray, *,
           element_tag: str = "sphere", order: int = 1) -> QuadMesh:
    """Closed cubed-sphere surface of ``radius`` about the origin: a unit
    :meth:`box` projected radially onto the sphere (same connectivity, so it
    pairs by index with a same-``n`` box for
    :meth:`HexMesh.annulus <nekmeshpy.hexmesh.HexMesh.annulus>`).  Every
    quad carries ``element_tag`` (default ``sphere``).

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    **every** ``(order+1)**2`` node of each face is projected onto the true sphere
    (not just the corners), so the high-order surface is the exact sphere -- the
    motivating high-order case.

    The projection is applied **entity-wise** straight onto the cube's B-rep -- its
    corner ``points``, its shared edge-interior table and its private per-quad
    ``interior`` -- never to a reassembled per-quad block.  Radial projection is
    node-wise, so a shared edge lands in the same place seen from either incident
    quad and the result stays structurally conformal with nothing to reconcile."""
    from ..linemesh import LineMesh
    from .quadmesh import QuadMesh
    cube = box(1.0, n, order=order)

    def project(a: FloatArray) -> FloatArray:
        """Push every node of ``a`` (last axis = xyz) radially onto the sphere."""
        return radius * a / np.linalg.norm(a, axis=-1, keepdims=True)

    etags = np.full(cube.n_quads, element_tag)
    # the cube's B-rep is reused verbatim (same topology, same edge numbering); only
    # the node coordinates move, so there is nothing to re-derive or reconcile.
    lines = LineMesh(project(cube.points), cube.lines.lines, order=order,
                     interior=project(cube.lines.interior) if order > 1 else None)
    return QuadMesh(lines, cube.quad, cube.flip,
                    project(cube.interior) if order > 1 else None,
                    element_tags=etags, order=order)


#: Closed-surface factories bound onto ``QuadMesh`` by ``quadmesh/__init__.py``.
FACTORIES: dict[str, Callable[..., QuadMesh]] = {
    "box": box,
    "sphere": sphere,
}
