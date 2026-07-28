"""All-hex mesh container.

``HexMesh`` is a pure hex container and a true sibling of
:class:`~nekmeshpy.quadmesh.QuadMesh` / :class:`~nekmeshpy.trimesh.TriMesh`:
storage is a plain ``(P,3)`` NumPy array ``points``
plus ``hexes`` (N,8) integer connectivity in Nek point order, ``boundaries``
(Nbc,2) = ``[element id (0-based), face (1-6)]`` and a parallel string array
``boundary_tags`` (Nbc,) naming each tagged face.  A boundary is identified by a
plain **tag** at build time; the tag is mapped to a Nek BC code / integer id
only at export (see :mod:`nekmeshpy.io.export`).

It also carries a dense per-hex ``element_tags`` ``(N,)`` (``""`` = untagged) --
the top of the region/material tag ladder: a
:class:`~nekmeshpy.linemesh.LineMesh`'s per-line ``element_tags`` ride
up through :meth:`~nekmeshpy.quadmesh.QuadMesh.extrude` onto the section
quads, and thence through :meth:`extrude` / :meth:`loft` onto every hex in that
quad's column.  Element tags have **no export path** yet (they never touch the
``.re2`` / ``.rea`` / ``.vtk`` bytes).

It is **not** built incrementally.  A mesh is constructed complete, either from
arrays (``HexMesh(points, hexes, boundaries)``) or through one of the factory
classmethods, named after the gmsh/CAD operations they mirror:

* :meth:`loft` -- recombine a stack of conformal
  :class:`~nekmeshpy.quadmesh.QuadMesh` cross-section profiles into a hex
  block (CAD *loft* through profiles).  Shared-point *by construction*: the
  profiles are conformal, so connectivity is index arithmetic -- no coordinate weld.
* :meth:`extrude` -- translate a single ``QuadMesh`` section along a straight axis
  into a hex block (gmsh ``Extrude`` + ``Layers`` + ``Recombine``); the straight
  special case of :meth:`loft`.
* :meth:`merge` -- stitch several hex blocks into one, coordinate-welding
  coincident seam points in a single explicit pass.
* :meth:`from_grid` -- a structured ``(ni+1,nj+1,nk+1)`` point grid to hexes.

The topology is fixed at construction; :meth:`weld` exposes the shared-point view
``(points, hexes, n_points)`` and its coordinates may still be repositioned in
place (interior repositioning, smoothing).  Everything that operates *on* a
finished mesh lives in dedicated modules taking the mesh as first argument:
:mod:`nekmeshpy.quadmesh.smoothing`,
:mod:`nekmeshpy.hexmesh.smoothing`,
:mod:`nekmeshpy.hexmesh.quality`, :mod:`nekmeshpy.io.export`,
:mod:`nekmeshpy.io.viz`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Point, PointArray, StrArray, Vec3
from ..model.fields import validate_layers
from ..quadmesh import NO_BOUNDARY, QuadMesh

# default sweep axis / origin for extrude (module-level singletons; read-only)
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# side name -> (Nek face number, axis, which end) for structured grids
_GRID_SIDES = {
    "x_min": (4, 0, 0), "x_max": (2, 0, -1),
    "y_min": (1, 1, 0), "y_max": (3, 1, -1),
    "z_min": (5, 2, 0), "z_max": (6, 2, -1),
}


class HexMesh:
    """An all-hexahedral volume mesh in shared-point form.

    Stores ``points`` ``(P,3)`` and ``hexes`` ``(N,8)`` integer connectivity (Nek
    point order), a sparse tagged-boundary list ``boundaries`` ``(Nbc,2)`` =
    ``[element id, face 1-6]`` with a parallel ``boundary_tags``, and a dense
    per-hex ``element_tags``.  It is **immutable by construction**: build it with a
    factory (:meth:`extrude` / :meth:`loft` / :meth:`annulus` / :meth:`merge` /
    :meth:`from_grid`) or the array constructor; coordinates may still be
    repositioned in place (smoothing) but the topology is fixed.  See
    :mod:`nekmeshpy.io.export` for writing ``.re2`` / ``.vtk`` / meshio output."""

    # Nek face -> the 4 corner point positions (0-based); row f is face f+1.
    FACE_POINTS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                           [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    def __init__(
        self,
        points: PointArray,
        hexes: IntArray,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
    ) -> None:
        """Construct from a shared-point representation: ``points`` ``(P,3)``,
        ``hexes`` ``(N,8)`` indices (Nek order), optional ``boundaries``
        ``(Nbc,2)`` = ``[elem, face]`` with a parallel ``boundary_tags``
        ``(Nbc,)`` naming each tagged face, and an optional dense ``element_tags``
        ``(N,)`` (``""`` = untagged; length must equal ``len(hexes)``).  Use the
        :meth:`extrude` / :meth:`merge` / :meth:`from_grid` factories for the usual
        build paths."""
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.hexes = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        self.boundaries = (np.zeros((0, 2), np.int64) if boundaries is None
                           else np.asarray(boundaries, np.int64).reshape(-1, 2))
        self.boundary_tags = (np.empty(0, dtype=np.str_) if boundary_tags is None
                              else np.asarray(boundary_tags, dtype=np.str_).reshape(-1))
        if self.boundary_tags.shape[0] != self.boundaries.shape[0]:
            raise ValueError("boundary_tags length (%d) must match boundaries (%d)"
                             % (self.boundary_tags.shape[0], self.boundaries.shape[0]))
        if element_tags is None:
            self.element_tags: StrArray = np.full(self.hexes.shape[0], "", dtype=np.str_)
        else:
            et = np.asarray(element_tags, dtype=np.str_).reshape(-1)
            if et.shape[0] != self.hexes.shape[0]:
                raise ValueError("element_tags length (%d) must match hexes (%d)"
                                 % (et.shape[0], self.hexes.shape[0]))
            self.element_tags = et

    # -- sizes -----------------------------------------------------------
    @property
    def n_hexes(self) -> int:
        """Number of hexahedra."""
        return self.hexes.shape[0]

    @property
    def n_points(self) -> int:
        """Number of (shared) points."""
        return self.points.shape[0]

    @property
    def n_boundaries(self) -> int:
        """Number of tagged boundary faces."""
        return self.boundaries.shape[0]

    @property
    def boundary_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged boundary faces (the physical groups
        present on this mesh).  A Nek BC code / integer id is assigned to each
        only at export -- see :mod:`nekmeshpy.io.export`."""
        return sorted(set(self.boundary_tags.tolist()))

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-hex element tags present on the mesh."""
        return sorted({t for t in self.element_tags.tolist() if t})

    # -- quality ---------------------------------------------------------
    def scaled_jacobian(self) -> FloatArray:
        """Per-hex minimum corner scaled Jacobian ``(n_hexes,)`` (see
        :func:`nekmeshpy.hexmesh.quality.scaled_jacobian`)."""
        from . import quality
        return quality.scaled_jacobian(self.points, self.hexes)

    def quality_summary(self) -> dict[str, Any]:
        """Aggregate scaled-Jacobian statistics (see
        :func:`nekmeshpy.hexmesh.quality.summary`)."""
        from . import quality
        return quality.summary(self.points, self.hexes)

    # -- orientation -----------------------------------------------------
    @staticmethod
    def _cap_tags(cap: str | Sequence[str] | StrArray, M: int) -> list[str]:
        """Normalize a cap tag to one tag per section quad (length ``M``).  A scalar
        ``str`` tags the whole cap (``""`` = untagged everywhere); an array-like is a
        per-quad tag (``""`` entries stay untagged), used by :meth:`annulus` to tag a
        cap from the section's per-quad ``element_tags``."""
        if isinstance(cap, str):
            return [cap] * M
        arr = np.asarray(cap, dtype=np.str_).reshape(-1)
        if arr.shape[0] != M:
            raise ValueError("cap tags length (%d) must match section quads (%d)"
                             % (arr.shape[0], M))
        return [str(x) for x in arr.tolist()]

    @staticmethod
    def _signed_vol(P: PointArray) -> float:
        """Sign proxy of the trilinear Jacobian at the hex centre (Nek order)."""
        P = np.asarray(P, dtype=float)
        r = P[[1, 2, 5, 6], :].mean(axis=0) - P[[0, 3, 4, 7], :].mean(axis=0)
        s = P[[2, 3, 6, 7], :].mean(axis=0) - P[[0, 1, 4, 5], :].mean(axis=0)
        t = P[[4, 5, 6, 7], :].mean(axis=0) - P[[0, 1, 2, 3], :].mean(axis=0)
        return float(np.dot(np.cross(r, s), t))

    # -- factories -------------------------------------------------------
    @classmethod
    def extrude(
        cls,
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
        a hex block (gmsh ``Extrude`` + ``Layers`` + ``Recombine``).

        The ``section`` is taken as a real plane in 3-D and translated **rigidly**
        along ``axis`` -- its own placement and orientation are preserved, so build
        the section in the plane you want swept (an xy section sweeps along ``z``;
        for a ``y`` sweep, author the section in xz).  ``origin`` shifts the whole
        block by a constant offset.

        ``layers`` are the normalized copy-plane positions along ``axis`` as
        fractions of ``length`` -- strictly increasing values in ``[0, 1]`` with the
        last ``1`` (the far cap).  Unlike the O-grid ``radial``, the initial position
        is *explicit*: ``layers[0]`` is the near cap -- ``0`` for a full sweep, or
        e.g. ``0.5`` to sweep only the far half of ``length`` -- so
        ``layers.size - 1`` hex layers span ``layers[0]..1``.  Pass
        ``uniform_spacing(k)`` for uniform layers, ``geometric_spacing(k, ratio)``
        for a geometric grading, or ``numpy.linspace(a, 1, k + 1)`` to start at
        ``a``.  ``first_tag`` / ``last_tag`` name the inlet/outlet caps; the swept
        side faces are named by the section's own ``boundary_tags`` and every hex
        inherits the section quad's ``element_tags`` (see :meth:`loft`).  This is the
        straight special case of :meth:`loft` -- a pure translation of the given
        profile; for a curved centreline or otherwise
        non-uniform profiles, position the profiles yourself and call :meth:`loft`.
        """
        base = np.asarray(section.points, dtype=float).reshape(-1, 3) \
            + np.asarray(origin, dtype=float)
        axis_u: Vec3 = np.asarray(axis, dtype=float)
        axis_u = axis_u / np.linalg.norm(axis_u)
        offsets = validate_layers(layers, "extrude layers") * float(length)
        slices = [QuadMesh(base + d * axis_u[None, :],
                           section.quads, boundaries=section.boundaries,
                           boundary_tags=section.boundary_tags,
                           element_tags=section.element_tags)
                  for d in offsets]
        return cls.loft(slices, first_tag=first_tag, last_tag=last_tag)

    @classmethod
    def loft(
        cls,
        slices: Sequence[QuadMesh],
        *,
        first_tag: str | Sequence[str] | StrArray = "",
        last_tag: str | Sequence[str] | StrArray = "",
    ) -> HexMesh:
        """Loft a stack of conformal quad cross-section profiles into a hex block
        (CAD *loft* through profiles; the general primitive behind :meth:`extrude`).

        ``slices`` is ``nz+1`` :class:`~nekmeshpy.quadmesh.QuadMesh`
        profiles sharing the same quad connectivity, ``boundary_tags``, and
        ``element_tags``; consecutive profiles form ``nz`` hex layers.  The first
        profile's bottom cap (face 5) is named ``first_tag``, the last profile's top
        cap (face 6) ``last_tag`` -- each a scalar ``str`` tagging the whole cap, or
        a per-quad array (one tag per section quad, ``""`` = untagged) so a cap can be
        tagged from the section's own ``element_tags`` (see :meth:`annulus`).  Each
        side face is named after its section edge
        via the section's ``boundary_tags`` (built at section time -- e.g.
        :meth:`~nekmeshpy.quadmesh.QuadMesh.structured` ``boundary_tags=`` or
        :meth:`~nekmeshpy.quadmesh.QuadMesh.ogrid` ``wall_tag=``).  An
        unnamed edge and the ``NO_BOUNDARY``
        sentinel are skipped, so a face can stay untagged (e.g. one that will be
        welded by :meth:`merge`).  Every hex in a quad's column inherits that quad's
        dense ``element_tags``.

        To tag an *interior* plane (e.g. a flux-measurement plane), loft the
        two segments either side of it separately -- with the plane as a cap of
        one of them -- and :meth:`merge`; the named cap then becomes the shared
        interior face.

        Points are shared by construction (index arithmetic over the conformal
        profile grid) -- no coordinate welding.
        """
        slices = list(slices)
        quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
        # section (quad, side) -> name; each swept side face inherits its section
        # edge (side s spans QuadMesh.EDGE_POINTS[s-1]).
        sec_bnd = np.asarray(slices[0].boundaries, dtype=np.int64).reshape(-1, 2)
        sec_tags = slices[0].boundary_tags
        side_name: dict[tuple[int, int], str] = {
            (int(sec_bnd[r, 0]), int(sec_bnd[r, 1])): str(sec_tags[r])
            for r in range(sec_bnd.shape[0])}
        tag_sides = bool(side_name)
        qtag = np.asarray(slices[0].element_tags, dtype=np.str_).reshape(-1)
        M = quads.shape[0]
        nz = len(slices) - 1
        S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                      for s in slices], axis=0)             # (nz+1, nn, 3)
        nn = S.shape[1]
        points = S.reshape((nz + 1) * nn, 3)                 # global id = i*nn + v

        # A consistently-wound section has one handedness, so decide it once from
        # the first layer and flip the whole quad template if the stack is
        # left-handed -- instead of testing every hex.  A mixed section (some
        # quads wound the other way) would silently produce inverted elements, so
        # it is rejected here rather than papered over.
        signs = np.array([cls._signed_vol(np.vstack([S[0, quads[q], :], S[1, quads[q], :]]))
                          for q in range(M)]) if nz else np.zeros(0)
        if nz and not (np.all(signs > 0) or np.all(signs < 0)):
            raise ValueError(
                "extrude: section is not consistently wound (mixed hex "
                "orientation) -- the section mesher must emit uniform winding")
        flip = bool(nz and signs[0] < 0)
        qw = quads[:, [0, 3, 2, 1]] if flip else quads

        # caps: scalar tags the whole cap, an array tags per section quad q (the
        # flip only reorders a quad's 4 corners, so caps stay faces 5/6 by q).
        first_caps = cls._cap_tags(first_tag, M)
        last_caps = cls._cap_tags(last_tag, M)

        hexes = np.empty((nz * M, 8), dtype=np.int64)
        etags: StrArray = np.empty(nz * M, dtype=np.str_)
        bnd: list[list[int]] = []
        names: list[str] = []
        e = 0
        for i in range(nz):
            for q in range(M):
                v = qw[q, :]
                hexes[e] = np.concatenate([i * nn + v, (i + 1) * nn + v])
                etags[e] = qtag[q] if qtag.size else ""
                if tag_sides:
                    # section side s -> hex side face: s when the quad kept its
                    # winding, 5-s when it was flipped (the flip reverses the edge
                    # cycle).  An unnamed edge / NO_BOUNDARY / "" stays untagged.
                    for s in (1, 2, 3, 4):
                        nm = side_name.get((q, s))
                        if nm is None or nm == NO_BOUNDARY:
                            continue
                        bnd.append([e, (5 - s) if flip else s])
                        names.append(nm)
                if i == 0 and first_caps[q]:
                    bnd.append([e, 5])
                    names.append(first_caps[q])
                if i == nz - 1 and last_caps[q]:
                    bnd.append([e, 6])
                    names.append(last_caps[q])
                e += 1
        return cls(points, hexes, *cls._order_bnd(bnd, names), element_tags=etags)

    @classmethod
    def annulus(
        cls,
        inner: QuadMesh,
        outer: QuadMesh,
        radial: FloatArray,
        *,
        inner_tag: str = "",
        outer_tag: str = "",
    ) -> HexMesh:
        """Shell O-grid filling the region *between* an inner and an outer closed
        quad surface -- e.g. a spherical body inside a cubic far-field box for a
        flow-past-sphere domain (the sibling of
        :meth:`~nekmeshpy.quadmesh.QuadMesh.annulus` one dimension up, and
        the surface-to-surface general case behind a spherical :meth:`extrude`).

        The two surfaces are paired **by index**: they must carry the same number of
        points ``P`` and identical ``quads`` connectivity, and point ``p`` of
        ``inner`` is joined radially to point ``p`` of ``outer`` (build ``inner`` from
        ``outer``'s points -- or vice versa -- so the pairing holds; see
        ``examples/flow_past_sphere.py``, where the sphere surface is ``R *
        normalize(cube surface points)`` on the same connectivity).

        ``radial`` are the shell positions with the **initial position explicit**
        (same convention as :meth:`extrude`'s ``layers`` and
        :meth:`~nekmeshpy.quadmesh.QuadMesh.annulus`): strictly increasing
        values in ``[0, 1]`` -- ``radial[0]`` is the inner shell (``0`` flush with the
        inner body) and the last is ``1`` (the outer surface) -- so
        ``radial.size - 1`` shell layers blend ``radial[0]`` -> outer.  Pass
        ``geometric_spacing(k, ratio)`` to cluster shells toward the inner body for a
        boundary layer.  The blend runs directly in 3-D (no projection), so a curvy /
        non-planar pair of surfaces keeps its true shape.

        **Wall faces are tagged from the surfaces' per-quad ``element_tags``**, not
        their ``boundary_tags`` -- a closed surface has no free boundary edges.  The
        inner (radial-in) cap of every shell column is tagged from
        ``inner.element_tags[q]`` and the outer (radial-out) cap from
        ``outer.element_tags[q]`` (``""`` stays untagged), so a cube far-field whose
        six faces carry ``inlet`` / ``outlet`` / ``top`` / ... element tags splits the
        outer wall into those groups automatically.  A non-empty scalar ``inner_tag``
        / ``outer_tag`` is the **override** -- it replaces the surface's per-quad tags
        and names the whole inner / outer wall.  An **open**
        surface's tagged ``boundaries`` still ride onto the swept lateral faces via
        :meth:`loft` (empty for the closed case here).  The hexes themselves are left
        region-untagged (the surface ``element_tags`` are wall designators)."""
        radial = validate_layers(radial, "annulus radial")
        A: FloatArray = np.asarray(inner.points, dtype=float).reshape(-1, 3)
        B: FloatArray = np.asarray(outer.points, dtype=float).reshape(-1, 3)
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

        # shell k is the straight-chord blend inner(radial[0]) -> outer(radial[-1]=1),
        # sharing inner's quad connectivity; consecutive shells loft into hex layers.
        # No element_tags on the shells -> hexes stay region-untagged; inner's tagged
        # boundaries ride onto the lateral faces (empty for a closed surface).
        shells = [QuadMesh((1.0 - t) * A + t * B, inner.quads,
                           boundaries=inner.boundaries,
                           boundary_tags=inner.boundary_tags)
                  for t in radial]
        # wall tags come from the lowest level -- the surfaces' per-quad element_tags,
        # consumed as the inner (face 5) / outer (face 6) caps.  A non-empty scalar
        # inner_tag / outer_tag OVERRIDES that and names the whole wall.
        inner_caps: str | StrArray = (
            inner_tag if inner_tag
            else (inner.element_tags if inner.element_group_tags else ""))
        outer_caps: str | StrArray = (
            outer_tag if outer_tag
            else (outer.element_tags if outer.element_group_tags else ""))
        return cls.loft(shells, first_tag=inner_caps, last_tag=outer_caps)

    @classmethod
    def merge(
        cls,
        meshes: Sequence[HexMesh],
        *,
        tol: float | None = None,
    ) -> HexMesh:
        """Stitch several hex blocks into one mesh, coordinate-welding coincident
        seam points in a single pass.  ``tol`` is the absolute coincidence
        distance (default ``1e-7`` x the merged bounding-box extent).

        Only points on each block's **domain boundary** (faces carried by a single
        hex) are weld candidates -- interior points are exclusive to their block
        and always kept distinct, so a stray interior coincidence can never
        silently collapse the mesh."""
        meshes = list(meshes)
        pos = [m.points for m in meshes]
        counts = [p.shape[0] for p in pos]
        P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
        total = P.shape[0]

        # remap: concat point index -> representative concat index (self by default)
        remap = np.arange(total, dtype=np.int64)
        is_bnd: BoolArray = np.zeros(total, dtype=bool)
        noff = 0
        for m, c in zip(meshes, counts):
            is_bnd[noff + cls._boundary_points(m.hexes)] = True
            noff += c
        bidx = np.flatnonzero(is_bnd)
        if bidx.size:
            scl = float(np.max(P.max(axis=0) - P.min(axis=0)))
            t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
            keys = np.round(P[bidx, :] / t).astype(np.int64)
            _, first_local, inverse = np.unique(
                keys, axis=0, return_index=True, return_inverse=True)
            remap[bidx] = bidx[first_local][inverse.ravel()]

        survivors = np.unique(remap)                    # concat indices kept
        new_id: IntArray = np.empty(total, dtype=np.int64)
        new_id[survivors] = np.arange(survivors.size)
        point_id = new_id[remap]                         # concat index -> final id
        points = P[survivors, :]

        hex_list, bnd_list, name_list, etag_list = [], [], [], []
        noff = eoff = 0
        for m, c in zip(meshes, counts):
            hex_list.append(point_id[m.hexes + noff])    # local -> concat -> welded id
            etag_list.append(np.asarray(m.element_tags, dtype=np.str_).reshape(-1))
            if m.boundaries.shape[0]:
                b: IntArray = m.boundaries.copy()
                b[:, 0] += eoff
                bnd_list.append(b)
                name_list.append(m.boundary_tags)
            noff += c
            eoff += m.hexes.shape[0]
        hexes = (np.concatenate(hex_list, axis=0) if hex_list
                 else np.zeros((0, 8), np.int64))
        etags = (np.concatenate(etag_list) if etag_list
                 else np.empty(0, dtype=np.str_))
        bnd = np.concatenate(bnd_list, axis=0) if bnd_list else np.zeros((0, 2), np.int64)
        names = (np.concatenate(name_list) if name_list
                 else np.empty(0, dtype=np.str_))
        return cls(points, hexes, *cls._order_bnd(bnd, names), element_tags=etags)

    # -- boundary queries (topological domain surface) ------------------
    @staticmethod
    def _boundary_mask(hexes: IntArray) -> tuple[IntArray, BoolArray]:
        """``(faces, is_boundary)``: every hex quad face ``(6N,4)`` in Nek order,
        element-major (row ``6e+f`` is element ``e``, local face ``f``), and a
        boolean mask of those carried by a single hex (the domain boundary)."""
        HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        faces: IntArray = HC[:, HexMesh.FACE_POINTS].reshape(-1, 4)
        keys = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        return faces, counts[inverse.ravel()] == 1

    @staticmethod
    def _boundary_points(hexes: IntArray) -> IntArray:
        faces, mask = HexMesh._boundary_mask(hexes)
        bf = faces[mask]
        return np.unique(bf) if bf.size else np.zeros(0, dtype=np.int64)

    def boundary_faces(self) -> IntArray:
        """``(K,2)`` array of ``[element id, local face (1-6)]`` for every face on
        the **topological** domain boundary (a quad carried by a single hex).

        Distinct from the *tagged* ``boundaries``, which may also carry
        interior planes (e.g. the bifurcation flux caps).  A face's four point ids
        are ``self.hexes[e, self.FACE_POINTS[f - 1]]``."""
        _, mask = self._boundary_mask(self.hexes)
        rows = np.flatnonzero(mask)
        return np.column_stack([rows // 6, rows % 6 + 1]).astype(np.int64)

    def boundary_elements(self) -> IntArray:
        """Sorted unique element ids with at least one face on the domain boundary."""
        return np.unique(self.boundary_faces()[:, 0])

    def boundary_points(self) -> IntArray:
        """Sorted unique point ids lying on the domain boundary."""
        return self._boundary_points(self.hexes)

    @classmethod
    def from_grid(
        cls,
        P: FloatArray,
        *,
        face_tags: dict[str, str] | None = None,
        element_tag: str = "",
    ) -> HexMesh:
        """Build hexes from a structured point grid ``P`` ``(ni+1,nj+1,nk+1,3)``.
        ``face_tags`` maps side names (``x_min``/``x_max``/``y_min``/``y_max``/
        ``z_min``/``z_max``) to boundary **names** on the six outer sides.  A side
        left out (or mapped to ``NO_BOUNDARY``)
        emits no boundary row -- use that for a side that will be welded away by
        :meth:`merge`, so merge stays a plain concatenate with no stale tag.
        ``element_tag`` (default untagged) is written to every hex's dense
        ``element_tags``."""
        P = np.asarray(P, dtype=float)
        ni1, nj1, nk1, _ = P.shape
        ni, nj, nk = ni1 - 1, nj1 - 1, nk1 - 1
        points = P.reshape(-1, 3)
        ids = np.arange(ni1 * nj1 * nk1, dtype=np.int64).reshape(ni1, nj1, nk1)

        hexes = np.empty((ni * nj * nk, 8), dtype=np.int64)
        e = 0
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    hexes[e] = [ids[i, j, k], ids[i + 1, j, k],
                                ids[i + 1, j + 1, k], ids[i, j + 1, k],
                                ids[i, j, k + 1], ids[i + 1, j, k + 1],
                                ids[i + 1, j + 1, k + 1], ids[i, j + 1, k + 1]]
                    e += 1
        bnd: list[list[int]] = []
        names: list[str] = []
        cell = np.arange(ni * nj * nk).reshape(ni, nj, nk)
        for side, name in (face_tags or {}).items():
            if not name:
                continue
            face, axis, end = _GRID_SIDES[side]
            plane: IntArray = cell.take(0 if end == 0 else -1, axis=axis).ravel()
            for eid in plane:
                bnd.append([int(eid), face])
                names.append(name)
        # np.full width-infers from the fill value (dtype=np.str_ would clip to <U1)
        etags: StrArray = np.full(hexes.shape[0], element_tag)
        return cls(points, hexes, *cls._order_bnd(bnd, names), element_tags=etags)

    @staticmethod
    def _order_bnd(
        bnd: Sequence[Sequence[int]] | IntArray,
        names: Sequence[str] | StrArray,
    ) -> tuple[IntArray, StrArray]:
        """Stably order boundary rows by ``(element id, face)`` so the exported
        block is independent of insertion order, applying the same permutation to
        the parallel ``names`` array."""
        b: IntArray = np.asarray(bnd, dtype=np.int64).reshape(-1, 2)
        nm: StrArray = np.asarray(names, dtype=np.str_).reshape(-1)
        if b.shape[0]:
            order = np.lexsort((b[:, 1], b[:, 0]))
            b = b[order]
            nm = nm[order]
        return b, nm

    # -- shared-point view ------------------------------------------------
    def weld(self) -> tuple[PointArray, IntArray, int]:
        """Shared-point view ``(points, hexes, n_points)``.  Returns the live
        positions array, so mutating it in place repositions the mesh."""
        return self.points, self.hexes, self.n_points

    def classify_points(self, wall: str) -> tuple[BoolArray, BoolArray]:
        """Flag welded points: ``(is_wall, is_fixed)``.  Faces named ``wall`` are
        wall; all other tagged faces are fixed.  A point on both is treated as
        fixed."""
        _, HC, nu = self.weld()
        is_wall: BoolArray = np.zeros(nu, dtype=bool)
        is_fixed: BoolArray = np.zeros(nu, dtype=bool)
        for b in range(self.boundaries.shape[0]):
            elem = int(self.boundaries[b, 0])
            face = int(self.boundaries[b, 1])
            ids = HC[elem, self.FACE_POINTS[face - 1, :]]
            if self.boundary_tags[b] == wall:
                is_wall[ids] = True
            else:
                is_fixed[ids] = True
        is_wall[is_fixed] = False
        return is_wall, is_fixed

    # -- topology / validity --------------------------------------------
    def topology_report(self) -> dict[str, Any]:
        """Watertightness / connectivity report of the welded mesh (see
        :func:`nekmeshpy.model.topology.hex_report`)."""
        from ..model import topology
        X, HC, _ = self.weld()
        return topology.hex_report(X, HC)

    def is_watertight(self) -> bool:
        """``True`` if the mesh boundary is a closed, leak-tight 2-manifold and
        the mesh is a single connected component.  Note this does *not* imply
        conformity: a T-junction is watertight -- use :meth:`is_conforming`."""
        rep = self.topology_report()
        return bool(rep["watertight"] and rep["n_components"] == 1)

    def is_conforming(self) -> bool:
        """``True`` if the mesh has no hanging points (no non-conformal
        T-junctions between coarse and fine elements)."""
        return bool(self.topology_report()["conformal"])

    def report(self) -> str:
        """One-call human-readable summary: element/point counts, scaled-Jacobian
        quality, per-name tagged-face counts, and the topology report.  Uses the
        mesh's own boundary tags -- no BC-code mapping needed (that is applied
        only at export)."""
        from ..model import topology
        from . import quality
        lines = ["%d hex elements, %d points" % (self.n_hexes, self.n_points)]
        lines.append(quality.format_report(quality.summary(self.points, self.hexes)))
        for name in self.boundary_group_tags:
            n = int(np.sum(self.boundary_tags == name))
            lines.append("  %-14s : %d faces" % (name, n))
        lines.append(topology.format_report(topology.hex_report(self.points, self.hexes)))
        return "\n".join(lines)

    # -- connectivity helpers (used by interior / smoothing) ------------
    @staticmethod
    def _unique_edges(HC: IntArray, he: IntArray) -> IntArray:
        Ei = HC[:, he[:, 0]].ravel()
        Ej = HC[:, he[:, 1]].ravel()
        return np.unique(np.sort(np.column_stack([Ei, Ej]), axis=1), axis=0)
