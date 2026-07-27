"""All-hex mesh container.

``HexMesh`` is a pure hex container and a true sibling of
:class:`~nekmeshpy.geometry.quadmesh.QuadMesh` / :class:`~nekmeshpy.geometry.trimesh.TriMesh`:
storage is a plain ``(P,3)`` NumPy array ``points``
plus ``hexes`` (N,8) integer connectivity in Nek point order, ``boundaries``
(Nbc,2) = ``[element id (0-based), face (1-6)]`` and a parallel string array
``boundary_names`` (Nbc,) naming each tagged face.  A boundary is identified by a
plain **name** at build time; the name is mapped to a Nek BC code / integer id
only at export (see :mod:`nekmeshpy.io.export`).

It is **not** built incrementally.  A mesh is constructed complete, either from
arrays (``HexMesh(points, hexes, boundaries)``) or through one of the factory
classmethods, named after the gmsh/CAD operations they mirror:

* :meth:`loft` -- recombine a stack of conformal
  :class:`~nekmeshpy.geometry.quadmesh.QuadMesh` cross-section profiles into a hex
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
:mod:`nekmeshpy.ops.interior`, :mod:`nekmeshpy.ops.smoothing`,
:mod:`nekmeshpy.model.quality`, :mod:`nekmeshpy.io.export`, :mod:`nekmeshpy.io.viz`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Point, PointArray, StrArray, Vec3
from ..model.fields import validate_layers
from .quadmesh import NO_BOUNDARY, QuadMesh

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
    # Nek face -> the 4 corner point positions (0-based); row f is face f+1.
    FACE_POINTS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                           [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    def __init__(
        self,
        points: PointArray,
        hexes: IntArray,
        boundaries: IntArray | None = None,
        boundary_names: StrArray | Sequence[str] | None = None,
    ) -> None:
        """Construct from a shared-point representation: ``points`` ``(P,3)``,
        ``hexes`` ``(N,8)`` indices (Nek order), optional ``boundaries``
        ``(Nbc,2)`` = ``[elem, face]`` with a parallel ``boundary_names``
        ``(Nbc,)`` naming each tagged face.  Use the :meth:`extrude` /
        :meth:`merge` / :meth:`from_grid` factories for the usual build paths."""
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.hexes = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        self.boundaries = (np.zeros((0, 2), np.int64) if boundaries is None
                           else np.asarray(boundaries, np.int64).reshape(-1, 2))
        self.boundary_names = (np.empty(0, dtype=np.str_) if boundary_names is None
                               else np.asarray(boundary_names, dtype=np.str_).reshape(-1))
        if self.boundary_names.shape[0] != self.boundaries.shape[0]:
            raise ValueError("boundary_names length (%d) must match boundaries (%d)"
                             % (self.boundary_names.shape[0], self.boundaries.shape[0]))

    # -- sizes -----------------------------------------------------------
    @property
    def n_hexes(self) -> int:
        return self.hexes.shape[0]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_boundaries(self) -> int:
        return self.boundaries.shape[0]

    @property
    def boundary_group_names(self) -> list[str]:
        """Sorted unique names of the tagged boundary faces (the physical groups
        present on this mesh).  A Nek BC code / integer id is assigned to each
        only at export -- see :mod:`nekmeshpy.io.export`."""
        return sorted(set(self.boundary_names.tolist()))

    # -- quality ---------------------------------------------------------
    def scaled_jacobian(self) -> FloatArray:
        """Per-hex minimum corner scaled Jacobian ``(n_hexes,)`` (see
        :func:`nekmeshpy.model.quality.scaled_jacobian`)."""
        from ..model import quality
        return quality.scaled_jacobian(self.points, self.hexes)

    def quality_summary(self) -> dict[str, Any]:
        """Aggregate scaled-Jacobian statistics (see
        :func:`nekmeshpy.model.quality.summary`)."""
        from ..model import quality
        return quality.summary(self.points, self.hexes)

    # -- orientation -----------------------------------------------------
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
        first_cap: str = "",
        last_cap: str = "",
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
        ``a``.  ``first_cap`` / ``last_cap`` name the inlet/outlet caps; the swept
        side faces are named by the section's own ``boundary_names`` (see
        :meth:`loft`).  This is the straight special case of :meth:`loft` -- a pure
        translation of the given profile; for a curved centreline or otherwise
        non-uniform profiles, position the profiles yourself and call :meth:`loft`.
        """
        base = np.asarray(section.points, dtype=float).reshape(-1, 3) \
            + np.asarray(origin, dtype=float)
        axis_u: Vec3 = np.asarray(axis, dtype=float)
        axis_u = axis_u / np.linalg.norm(axis_u)
        offsets = validate_layers(layers, "extrude layers") * float(length)
        slices = [QuadMesh(base + d * axis_u[None, :],
                           section.quads, boundaries=section.boundaries,
                           boundary_names=section.boundary_names)
                  for d in offsets]
        return cls.loft(slices, first_cap=first_cap, last_cap=last_cap)

    @classmethod
    def loft(
        cls,
        slices: Sequence[QuadMesh],
        *,
        first_cap: str = "",
        last_cap: str = "",
    ) -> HexMesh:
        """Loft a stack of conformal quad cross-section profiles into a hex block
        (CAD *loft* through profiles; the general primitive behind :meth:`extrude`).

        ``slices`` is ``nz+1`` :class:`~nekmeshpy.geometry.quadmesh.QuadMesh`
        profiles sharing the same quad connectivity and ``boundary_names``;
        consecutive profiles form ``nz`` hex layers.  The first profile's bottom
        cap (face 5) is named ``first_cap``, the last profile's top cap
        (face 6) ``last_cap``, and each side face is named after its section edge
        via the section's ``boundary_names`` (built at section time -- e.g.
        :meth:`~nekmeshpy.geometry.quadmesh.QuadMesh.structured` ``boundary_names=`` or
        :meth:`~nekmeshpy.geometry.quadmesh.QuadMesh.ogrid` ``wall_name=``).  An
        unnamed edge and the :data:`~nekmeshpy.geometry.quadmesh.NO_BOUNDARY`
        sentinel are skipped, so a face can stay untagged (e.g. one that will be
        welded by :meth:`merge`).

        To tag an *interior* plane (e.g. a flux-measurement plane), loft the
        two segments either side of it separately -- with the plane as a cap of
        one of them -- and :meth:`merge`; the named cap then becomes the shared
        interior face.

        Points are shared by construction (index arithmetic over the conformal
        profile grid) -- no coordinate welding.
        """
        slices = list(slices)
        quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
        boundary_names = slices[0].boundary_names
        tag_sides = bool(boundary_names)
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
        qw = quads[:, [0, 3, 2, 1]] if (nz and signs[0] < 0) else quads

        hexes = np.empty((nz * M, 8), dtype=np.int64)
        bnd: list[list[int]] = []
        names: list[str] = []
        e = 0
        for i in range(nz):
            for q in range(M):
                v = qw[q, :]
                hexes[e] = np.concatenate([i * nn + v, (i + 1) * nn + v])
                if tag_sides:
                    for f in range(4):
                        edge = frozenset((int(v[f]), int(v[(f + 1) % 4])))
                        # named side face; an unnamed edge / NO_BOUNDARY / "" is
                        # left untagged
                        nm = boundary_names.get(edge, "")
                        if nm != NO_BOUNDARY:
                            bnd.append([e, f + 1])
                            names.append(nm)
                if first_cap and i == 0:
                    bnd.append([e, 5])
                    names.append(first_cap)
                if last_cap and i == nz - 1:
                    bnd.append([e, 6])
                    names.append(last_cap)
                e += 1
        return cls(points, hexes, *cls._order_bnd(bnd, names))

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

        hex_list, bnd_list, name_list = [], [], []
        noff = eoff = 0
        for m, c in zip(meshes, counts):
            hex_list.append(point_id[m.hexes + noff])    # local -> concat -> welded id
            if m.boundaries.shape[0]:
                b: IntArray = m.boundaries.copy()
                b[:, 0] += eoff
                bnd_list.append(b)
                name_list.append(m.boundary_names)
            noff += c
            eoff += m.hexes.shape[0]
        hexes = (np.concatenate(hex_list, axis=0) if hex_list
                 else np.zeros((0, 8), np.int64))
        bnd = np.concatenate(bnd_list, axis=0) if bnd_list else np.zeros((0, 2), np.int64)
        names = (np.concatenate(name_list) if name_list
                 else np.empty(0, dtype=np.str_))
        return cls(points, hexes, *cls._order_bnd(bnd, names))

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

        Distinct from the *tagged* :attr:`boundaries`, which may also carry
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
    ) -> HexMesh:
        """Build hexes from a structured point grid ``P`` ``(ni+1,nj+1,nk+1,3)``.
        ``face_tags`` maps side names (``x_min``/``x_max``/``y_min``/``y_max``/
        ``z_min``/``z_max``) to boundary **names** on the six outer sides.  A side
        left out (or mapped to :data:`~nekmeshpy.geometry.quadmesh.NO_BOUNDARY`)
        emits no boundary row -- use that for a side that will be welded away by
        :meth:`merge`, so merge stays a plain concatenate with no stale tag."""
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
        return cls(points, hexes, *cls._order_bnd(bnd, names))

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
            if self.boundary_names[b] == wall:
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
        mesh's own boundary names -- no BC-code mapping needed (that is applied
        only at export)."""
        from ..model import quality, topology
        lines = ["%d hex elements, %d points" % (self.n_hexes, self.n_points)]
        lines.append(quality.format_report(quality.summary(self.points, self.hexes)))
        for name in self.boundary_group_names:
            n = int(np.sum(self.boundary_names == name))
            lines.append("  %-14s : %d faces" % (name, n))
        lines.append(topology.format_report(topology.hex_report(self.points, self.hexes)))
        return "\n".join(lines)

    # -- connectivity helpers (used by interior / smoothing) ------------
    @staticmethod
    def _unique_edges(HC: IntArray, he: IntArray) -> IntArray:
        Ei = HC[:, he[:, 0]].ravel()
        Ej = HC[:, he[:, 1]].ravel()
        return np.unique(np.sort(np.column_stack([Ei, Ej]), axis=1), axis=0)
