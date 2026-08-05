"""Parametric 3-D surface :class:`~nekmeshpy.QuadMesh` factories: surfaces built by
welding face patches from a handful of numbers -- ``box`` / ``sphere`` (watertight)
and their ground-plane halves ``half_box`` / ``hemisphere`` (open at the ``z = 0``
rim).

The four live together here rather than in ``_open.py`` because the split that matters
for a *surface* factory is "shape from parameters" (these) versus "fill a region I hand
you a boundary curve for" (``_open.py``: ``structured`` / ``ogrid`` / ``annulus``).
``half_box`` and ``hemisphere`` are ``box`` and ``sphere`` with the ``-z`` patch
dropped, share their code and their index pairing, and would be orphaned from it in
``_open.py``; their rim is a boundary of the surface, not an input to it.

These are plain free functions returning a ``QuadMesh``; ``quadmesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use ``QuadMesh.box(...)`` /
``QuadMesh.sphere(...)``.  They build on the core ``from_grid`` / ``merge``
constructors, referenced by a lazy in-function import to avoid the import cycle with
``quadmesh.py`` (which the package imports to assemble the class).  Internal toolkit
code calls these free functions directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .._typing import FloatArray, IntArray, PointArray, Vec3
from ..model.tags import EdgeTags, ElementTags
from ._assemble import merge
from ._lift import from_grid
from ._query import boundary_edges
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

# the four upright side faces of a half box: outward normal n, horizontal tangent u,
# vertical tangent +z (u x v = n), so restricting the vertical coordinate to [0, 1]
# keeps the patch in z >= 0 and puts its lower edge on the ground plane.
_HALF_BOX_SIDES = [
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "x_max"),
    ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "x_min"),
    ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), "y_max"),
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), "y_min"),
]
_VZ: Vec3 = np.array([0.0, 0.0, 1.0])


def _axis_params(half_sizes: float | Sequence[float] | FloatArray,
                 n: int | Sequence[int] | IntArray,
                 ) -> tuple[FloatArray, tuple[int, int, int]]:
    """Normalize a box-like factory's ``half_sizes`` / ``n`` to a ``(3,)`` float array
    of half extents and a ``(nx, ny, nz)`` cell-count triple (each input a scalar or
    three values)."""
    hs: FloatArray = np.asarray(half_sizes, dtype=float).ravel()
    if hs.size == 1:
        hs = np.full(3, float(hs[0]))
    elif hs.size != 3:
        raise ValueError("half_sizes must be a scalar or 3 values (sx, sy, sz)")
    na: IntArray = np.asarray(n, dtype=np.int64).ravel()
    if na.size == 1:
        return hs, (int(na[0]), int(na[0]), int(na[0]))
    if na.size == 3:
        return hs, (int(na[0]), int(na[1]), int(na[2]))
    raise ValueError("n must be a scalar or 3 counts (nx, ny, nz)")


def box(half_sizes: float | Sequence[float] | FloatArray,
        n: int | Sequence[int] | IntArray, *,
        patch_tags: Mapping[str, str] | None = None,
        order: int = 1) -> QuadMesh:
    """Closed box surface centred at the origin: six quad patches welded with
    :func:`merge <nekmeshpy.quadmesh.assemble.merge>`.  ``half_sizes`` is a scalar (cube) or ``(sx, sy, sz)``; ``n``
    is a scalar or ``(nx, ny, nz)`` cells per axis.  ``patch_tags`` (keyed
    ``x_min`` / ``x_max`` / ... / ``z_max``) writes each face's dense per-quad
    ``element_tags`` -- e.g. the far-field side it forms; an absent face stays
    untagged so ``merge`` welds shared edges cleanly.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each flat face patch carries ``(order+1)**2`` straight-sided GLL nodes (exact,
    the faces are planar)."""
    hs, n_axis = _axis_params(half_sizes, n)
    ft = patch_tags or {}
    patches: list[QuadMesh] = []
    for nrm, u, v, key in _BOX_FACES:
        nv: Vec3 = np.asarray(nrm, dtype=float)
        uv: Vec3 = np.asarray(u, dtype=float)
        vv: Vec3 = np.asarray(v, dtype=float)
        au = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(uv)))] + 1)
        av = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(vv)))] + 1)
        A: FloatArray
        B: FloatArray
        A, B = np.meshgrid(au, av, indexing="ij")
        face = hs * (nv + A[..., None] * uv + B[..., None] * vv)
        patches.append(from_grid(face, element_tag=ft.get(key, ""),
                                          order=order))
    return merge(patches)


def sphere(radius: float, n: int | Sequence[int] | IntArray, *,
           element_tag: str = "sphere", order: int = 1) -> QuadMesh:
    """Closed cubed-sphere surface of ``radius`` about the origin: a unit
    :func:`box <nekmeshpy.quadmesh.surface.box>` projected radially onto the sphere (same connectivity, so it
    pairs by index with a same-``n`` box for
    :func:`HexMesh.annulus <nekmeshpy.hexmesh.lift.annulus>`).  Every
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
    cube = box(1.0, n, order=order)

    def project(a: PointArray) -> PointArray:
        """Push every node of ``a`` (last axis = xyz) radially onto the sphere."""
        return radius * a / np.linalg.norm(a, axis=-1, keepdims=True)

    etags = ElementTags.uniform(cube.n_quads, element_tag)
    # the cube's B-rep is reused verbatim (same topology, same edge numbering); only
    # the node coordinates move, so there is nothing to re-derive or reconcile.
    lines = LineMesh(project(cube.points), cube.lines.lines, order=order,
                     interior=project(cube.lines.interior) if order > 1 else None)
    return QuadMesh(lines, cube.quad, cube.flip,
                    project(cube.interior) if order > 1 else None,
                    element_tags=etags, order=order)


def _tag_rim(qm: QuadMesh, rim_tag: str) -> QuadMesh:
    """Return ``qm`` with every free (single-quad) edge tagged ``rim_tag``.

    A half box / hemisphere is open at ``z = 0``, so its rim is exactly the surface's
    free edge set.  Sweeping the surface with
    :func:`HexMesh.annulus <nekmeshpy.hexmesh.lift.annulus>` turns those edges into
    the shell's side faces -- the ground annulus between body and far field."""
    if not rim_tag:
        return qm
    rows = boundary_edges(qm)
    bnd = EdgeTags.from_pairs(rows, [rim_tag] * rows.shape[0]).ordered()
    return QuadMesh(qm.lines, qm.quad, qm.flip, qm.interior if qm.order > 1 else None,
                    bnd, qm.element_tags, order=qm.order)


def half_box(half_sizes: float | Sequence[float] | FloatArray,
             n: int | Sequence[int] | IntArray, *,
             n_vertical: int | None = None,
             patch_tags: Mapping[str, str] | None = None,
             rim_tag: str = "",
             order: int = 1) -> QuadMesh:
    """The upper half of a :func:`box <nekmeshpy.quadmesh.surface.box>`: the five patches of the box surface that
    bound ``[-sx, sx] x [-sy, sy] x [0, sz]``, welded with :func:`merge <nekmeshpy.quadmesh.assemble.merge>` and left
    **open at the ``z = 0`` rim** (the ``z_min`` patch is dropped).

    ``half_sizes`` is a scalar (cube) or ``(sx, sy, sz)``; ``n`` is a scalar or
    ``(nx, ny, nz)`` horizontal cells per axis, and ``n_vertical`` the cells over
    ``z in [0, sz]`` on the four upright side patches (default: ``nz``).  The top
    patch is ``nx x ny``.  ``patch_tags`` (keyed ``x_min`` / ``x_max`` / ``y_min`` /
    ``y_max`` / ``z_max``) writes each patch's dense per-quad ``element_tags``;
    ``rim_tag`` names the ``z = 0`` rim edges (see :func:`hemisphere <nekmeshpy.quadmesh.surface.hemisphere>`).

    This is the far-field partner of :func:`hemisphere <nekmeshpy.quadmesh.surface.hemisphere>`: a ``hemisphere(R, n,
    n_vertical=v)`` has identical ``quads`` and point count, so the two pair by index
    for :func:`HexMesh.annulus <nekmeshpy.hexmesh.lift.annulus>`.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1`` each
    flat patch carries ``(order+1)**2`` straight-sided GLL nodes (exact -- the
    patches are planar)."""
    hs, n_axis = _axis_params(half_sizes, n)
    nv = n_axis[2] if n_vertical is None else int(n_vertical)
    if nv < 1:
        raise ValueError("half_box needs n_vertical >= 1, got %d" % nv)
    ft = patch_tags or {}
    patches: list[QuadMesh] = []
    b_side = np.linspace(0.0, 1.0, nv + 1)                 # upper half only
    for nrm, u, key in _HALF_BOX_SIDES:
        nvv: Vec3 = np.asarray(nrm, dtype=float)
        uv: Vec3 = np.asarray(u, dtype=float)
        au = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(uv)))] + 1)
        A: FloatArray
        B: FloatArray
        A, B = np.meshgrid(au, b_side, indexing="ij")
        face = hs * (nvv + A[..., None] * uv + B[..., None] * _VZ)
        patches.append(from_grid(face, element_tag=ft.get(key, ""),
                                          order=order))
    # the flat top patch at z = sz, spanning x and y in full
    ax = np.linspace(-1.0, 1.0, n_axis[0] + 1)
    ay = np.linspace(-1.0, 1.0, n_axis[1] + 1)
    AX: FloatArray
    AY: FloatArray
    AX, AY = np.meshgrid(ax, ay, indexing="ij")
    top = hs * (_VZ + AX[..., None] * np.array([1.0, 0.0, 0.0])
                + AY[..., None] * np.array([0.0, 1.0, 0.0]))
    patches.append(from_grid(top, element_tag=ft.get("z_max", ""),
                                      order=order))
    return _tag_rim(merge(patches), rim_tag)


def hemisphere(radius: float, n: int | Sequence[int] | IntArray, *,
               n_vertical: int | None = None,
               element_tag: str = "hemisphere",
               rim_tag: str = "",
               order: int = 1) -> QuadMesh:
    """Cubed-**hemisphere** surface of ``radius`` sitting on the ground plane
    ``z = 0``: a unit :func:`half_box <nekmeshpy.quadmesh.surface.half_box>` projected radially onto the sphere.  It is
    :func:`sphere <nekmeshpy.quadmesh.surface.sphere>` with the ``-z`` patch dropped, so it is open at the ``z = 0``
    rim -- a body resting on the floor rather than a closed surface.

    Same connectivity as ``half_box(s, n, n_vertical=v)`` for any ``s``, so the two
    pair point-for-point (identical ``quads``, equal point count) and
    :func:`HexMesh.annulus <nekmeshpy.hexmesh.lift.annulus>` fills the shell
    between them -- the ground-plane analogue of the
    ``sphere`` / ``box`` pairing in ``flow_past_sphere.py``.  The rim is the equator
    ``x^2 + y^2 = radius^2``, and ``half_box``'s rim is the ground square, so the
    shell's side faces are the flat ground annulus at ``z = 0``.

    Every quad carries ``element_tag`` (default ``hemisphere``), which
    ``HexMesh.annulus`` turns into the inner wall faces.  ``rim_tag`` (e.g.
    ``ground``) names the rim edges, which the same sweep turns into the shell's
    side faces -- pass it on whichever surface is the ``annulus`` *inner* argument,
    since that is the one whose ``edge_tags`` the shells carry.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    **every** ``(order+1)**2`` node of each patch is projected onto the true sphere
    (not just the corners), so the high-order wall is the exact hemisphere.  The
    projection is applied entity-wise straight onto the half box's B-rep -- its
    corner ``points``, its shared edge-interior table and its private per-quad
    ``interior`` -- so a shared edge lands in the same place seen from either
    incident quad and the result stays structurally conformal."""
    from ..linemesh import LineMesh
    cube = half_box(1.0, n, n_vertical=n_vertical, order=order)

    def project(a: PointArray) -> PointArray:
        """Push every node of ``a`` (last axis = xyz) radially onto the sphere."""
        return radius * a / np.linalg.norm(a, axis=-1, keepdims=True)

    etags = ElementTags.uniform(cube.n_quads, element_tag)
    lines = LineMesh(project(cube.points), cube.lines.lines, order=order,
                     interior=project(cube.lines.interior) if order > 1 else None)
    qm = QuadMesh(lines, cube.quad, cube.flip,
                  project(cube.interior) if order > 1 else None,
                  element_tags=etags, order=order)
    return _tag_rim(qm, rim_tag)
