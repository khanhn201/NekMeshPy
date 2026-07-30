"""All-hex mesh container.

``HexMesh`` stores ``points`` ``(P,3)``, ``hexes`` ``(N,8)`` connectivity in Nek
order, a sparse tagged ``boundaries`` ``(Nbc,2)`` = ``[element id, face 1-6]`` with
parallel ``boundary_tags``, and a dense per-hex ``element_tags``. Boundary tags map
to Nek BC codes only at export.

It is built complete, not incrementally: from arrays or via the factory
classmethods ``loft`` / ``extrude`` / ``annulus`` / ``merge`` / ``from_grid``. The
topology is fixed at construction, but coordinates may be repositioned in place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..linemesh import LineMesh
from ..model import conform
from ..model.fields import gll_nodes, reject_loop_caps, validate_layers
from ..model.interp import corner_indices
from ..quadmesh import NO_BOUNDARY, QuadMesh

# default sweep axis / origin for extrude
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# side name -> (Nek face number, axis, which end) for structured grids.  ``from_grid``
# now routes each side through ``QuadMesh.from_grid``'s edge tags / ``loft``'s caps, so
# this is the *reference* mapping the composition reproduces (and the set of legal side
# names it validates against), not a lookup it indexes a plane with.
_GRID_SIDES = {
    "x_min": (4, 0, 0), "x_max": (2, 0, -1),
    "y_min": (1, 1, 0), "y_max": (3, 1, -1),
    "z_min": (5, 2, 0), "z_max": (6, 2, -1),
}


# -- native (block-free) high-order helpers -----------------------------
def _slice_block(s: QuadMesh, order: int) -> PointArray:
    """``(Q,(order+1)**2,3)`` in-plane high-order block of one loft profile, assembled
    natively from that section's B-rep (shared corners ++ shared edge-interior nodes in
    element traversal order ++ private quad interiors).  A loft column's geometry is a
    straight sweep between two such in-plane blocks, so this is the only intermediate
    :meth:`HexMesh.loft` needs -- no per-element hex block is ever materialized."""
    row = order + 1
    out: PointArray = np.empty((s.quads.shape[0], row * row, 3), dtype=float)
    out[:, corner_indices(order, 2), :] = s.points[s.quads]
    out[:, conform._edge_slots(2, order)[:, 1:-1], :] = conform.gather_edge_nodes(
        s.lines.interior, s.quad, s.flip)
    out[:, conform._interior_slots(2, order), :] = s.interior
    return out


def _sweep_at(bottom: PointArray, top: PointArray, g: FloatArray,
              slots: IntArray, m2: int) -> PointArray:
    """A loft column's straight GLL sweep evaluated at the hex block ``slots``.

    Hex lexicographic slot ``s`` decomposes as ``s = k*m2 + m`` (in-plane index ``m``
    fastest, sweep index ``k`` slowest), so the node there is
    ``(1-g[k])*bottom[m] + g[k]*top[m]`` -- the same elementwise expression the full
    ``(E,order+1,m2,3)`` sweep uses, restricted to the requested slots.  Returns
    ``(E, len(slots), 3)``."""
    k = slots // m2
    m = slots % m2
    gg = g[k].reshape(1, -1, 1)
    return (1.0 - gg) * bottom[:, m, :] + gg * top[:, m, :]


class HexMesh:
    """An all-hexahedral volume mesh in shared-point form.

    Stores ``points`` ``(P,3)``, ``hexes`` ``(N,8)`` connectivity (Nek order), a
    sparse tagged ``boundaries`` with parallel ``boundary_tags``, and a dense
    per-hex ``element_tags``. Immutable topology; build via a factory or the array
    constructor."""

    # Nek face -> the 4 corner point positions (0-based); row f is face f+1.
    FACE_POINTS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                           [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    def __init__(
        self,
        quads: QuadMesh,
        hex: IntArray,
        face_orient: IntArray,
        interior: PointArray | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
        *,
        order: int = 1,
    ) -> None:
        """Construct from the B-rep directly: ``quads`` (a ``QuadMesh`` holding every
        shared face -- its ``points`` are the shared corners, its ``quads`` the shared
        face connectivity, its edges / ``interior`` the shared face-boundary / interior
        HO nodes), ``hex`` ``(E,6)`` face indices into ``quads.quads`` (Nek local-face
        order), ``face_orient`` ``(E,6)`` D4 codes (element-local face frame ->
        canonical), and ``interior`` ``(E,(order-1)**3,3)`` private per-hex nodes (omit /
        ``None`` at order 1).  Also an optional dense per-hex ``element_tags`` ``(E,)``
        and a tagged-boundary list ``boundaries`` ``(Nbc,2)`` = ``[hex id, face 1-6]``
        with a parallel ``boundary_tags``.

        ``.points`` / ``.hexes`` are **derived** views over this B-rep, so
        a shared face is literally one stored object referenced by every incident hex
        (structural conformality).  :meth:`from_corners` is the linear corner -> B-rep
        bridge for callers that only hold corner connectivity; a caller that already
        owns the shared-face ``QuadMesh`` builds through here directly.  ``re2`` export
        stays linear; only ``vtu`` reads the high-order nodes."""
        if not isinstance(quads, QuadMesh):
            raise TypeError("HexMesh: quads must be a QuadMesh, got %s"
                            % type(quads).__name__)
        self._order = int(order)
        if quads.order != self._order:
            raise ValueError("HexMesh: quads.order (%d) must match order (%d)"
                             % (quads.order, self._order))
        self.quads = quads
        self.hex: IntArray = np.asarray(hex, dtype=np.int64).reshape(-1, 6)
        self.face_orient: IntArray = np.asarray(
            face_orient, dtype=np.int64).reshape(-1, 6)
        if self.face_orient.shape[0] != self.hex.shape[0]:
            raise ValueError("HexMesh: face_orient length (%d) must match hex (%d)"
                             % (self.face_orient.shape[0], self.hex.shape[0]))
        E = self.hex.shape[0]
        k = (self._order - 1) ** 3
        if interior is None:
            if self._order > 1:
                raise ValueError(
                    "HexMesh: order %d > 1 requires interior nodes" % self._order)
            self.interior: PointArray = np.zeros((E, 0, 3), dtype=float)
        else:
            ia: PointArray = np.asarray(interior, dtype=float)
            if ia.shape != (E, k, 3):
                raise ValueError(
                    "HexMesh: interior must be (E,(order-1)**3,3) = (%d,%d,3), got %s"
                    % (E, k, ia.shape))
            self.interior = ia
        if element_tags is None:
            self.element_tags: StrArray = np.full(E, "", dtype=np.str_)
        else:
            et = np.asarray(element_tags, dtype=np.str_).reshape(-1)
            if et.shape[0] != E:
                raise ValueError("element_tags length (%d) must match hexes (%d)"
                                 % (et.shape[0], E))
            self.element_tags = et
        self.boundaries: IntArray = (
            np.zeros((0, 2), np.int64) if boundaries is None
            else np.asarray(boundaries, np.int64).reshape(-1, 2))
        self.boundary_tags: StrArray = (
            np.empty(0, dtype=np.str_) if boundary_tags is None
            else np.asarray(boundary_tags, dtype=np.str_).reshape(-1))
        if self.boundary_tags.shape[0] != self.boundaries.shape[0]:
            raise ValueError("boundary_tags length (%d) must match boundaries (%d)"
                             % (self.boundary_tags.shape[0], self.boundaries.shape[0]))

        # corner connectivity + per-hex edge incidence are derived from the shared
        # faces and immutable post-construction (point moves don't change them), so
        # memoize once.
        self._corners: IntArray = self._derive_corners()
        _, self._elem_edges, self._edge_flip = conform.unique_edges(self._corners, 3)

    @classmethod
    def from_corners(
        cls,
        points: PointArray,
        hexes: IntArray,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
        *,
        order: int = 1,
    ) -> HexMesh:
        """Build a **linear** ``HexMesh`` from corner ``points`` ``(P,3)`` + Nek-order
        ``hexes`` ``(E,8)`` connectivity -- the corner -> B-rep bridge every factory
        routes through.  Decomposes the shared faces with ``conform.canonical_faces``
        (lossless, so ``.hexes`` round-trips the input exactly).

        Corners are all a linear mesh has, so this is an ``order == 1`` constructor:
        at ``order > 1`` there is no way to invent the shared edge / face / private
        interior nodes without silently straight-subdividing, so it raises.  Build with
        a factory (``extrude`` / ``loft`` / ``annulus`` / ``from_grid``) at ``order=N``,
        which places those nodes on the true geometry, or construct
        ``HexMesh(quads, hex, face_orient, interior, ..., order=N)`` directly from the
        entity fields if you already hold them."""
        if order != 1:
            raise ValueError(
                "HexMesh.from_corners builds the linear (order 1) B-rep from corner "
                "points alone; got order=%d, whose shared edge / face / private "
                "interior nodes it cannot know. Build with a factory at order=%d "
                "(e.g. HexMesh.extrude(section, ...) / HexMesh.loft(slices) from "
                "order-%d sections), which places those nodes on the true geometry, "
                "or construct HexMesh(quads, hex, face_orient, interior=..., "
                "order=%d) directly from the entity fields (the shared-face QuadMesh "
                "carries the edge / face nodes)."
                % (order, order, order, order))
        pts: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        conn: IntArray = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        canonical_conn, elem_faces, face_orient = conform.canonical_faces(conn)
        quads = QuadMesh.from_corners(pts, canonical_conn)
        return cls(quads, elem_faces, face_orient, None, boundaries,
                   boundary_tags, element_tags, order=1)

    def _derive_corners(self) -> IntArray:
        """Corner connectivity ``(E,8)`` (Nek order) recovered from the shared faces via
        ``conform.hex_corners_from_faces`` -- the lossless inverse of
        ``conform.canonical_faces``, so it reproduces the connectivity the mesh was built
        from byte-for-byte."""
        return conform.hex_corners_from_faces(
            self.quads.quads, self.hex, self.face_orient)

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return self._order

    @property
    def points(self) -> PointArray:
        """The ``(P,3)`` shared corner points -- a live view of the shared-face
        ``QuadMesh``'s ``points`` (the single source of truth), so an in-place edit
        (``mesh.points[:] = X``) moves the shared corners for every hex."""
        return self.quads.points

    @property
    def hexes(self) -> IntArray:
        """``(E,8)`` Nek-order corner connectivity, derived (memoized) from the shared
        faces.  Read-only; the B-rep ``quads`` / ``hex`` / ``face_orient`` are the source
        of truth."""
        return self._corners

    @property
    def edges(self) -> IntArray:
        """``(Ne,2)`` unique undirected hex edges (canonical: min corner id first) -- the
        shared edge topology (the ``edges`` of the shared-face ``QuadMesh``).  Non-empty
        at every order (edges are first-class B-rep storage)."""
        return self.quads.edges

    @property
    def edge_nodes(self) -> PointArray:
        """``(Ne, order-1, 3)`` shared high-order interior nodes of each unique
        :attr:`edges` entry, in canonical (min->max corner) order.  Empty at order 1."""
        return self.quads.edge_nodes

    @property
    def faces(self) -> IntArray:
        """``(Nf,4)`` unique hex faces (canonical: sorted corner ids) -- the shared face
        topology.  Non-empty at every order (faces are first-class B-rep storage)."""
        return np.sort(self.quads.quads, axis=1)

    @property
    def face_nodes(self) -> PointArray:
        """``(Nf, (order-1)**2, 3)`` shared high-order interior nodes of each unique
        :attr:`faces` entry, in the canonical D4-normalized frame.  Empty at order 1; a
        shared face resolves to the same nodes from either incident hex."""
        return self.quads.interior

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
        """Sorted unique tags of the tagged boundary faces."""
        return sorted(set(self.boundary_tags.tolist()))

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-hex element tags present on the mesh."""
        return sorted({t for t in self.element_tags.tolist() if t})

    # -- quality ---------------------------------------------------------
    def scaled_jacobian(self, *, high_order: bool = False) -> FloatArray:
        """Per-hex minimum scaled Jacobian ``(n_hexes,)``.

        Defaults to the corner metric (the pinned linear numbers).  With
        ``high_order=True`` it is sampled at the ``(order+1)**3`` GLL nodes of the
        curved block (:func:`~nekmeshpy.hexmesh.quality.scaled_jacobian_ho`); at order
        1 the two agree."""
        from . import quality
        if high_order:
            return quality.scaled_jacobian_ho(self, self.order)
        return quality.scaled_jacobian(self.points, self.hexes)

    def quality_summary(self, *, high_order: bool = False) -> dict[str, Any]:
        """Aggregate scaled-Jacobian statistics (see :meth:`scaled_jacobian` for the
        ``high_order`` flag)."""
        from . import quality
        if high_order:
            return quality.summary_ho(self, self.order)
        return quality.summary(self.points, self.hexes)

    # -- orientation -----------------------------------------------------
    @staticmethod
    def _cap_tags(cap: str | Sequence[str] | StrArray, M: int) -> list[str]:
        """Normalize a cap tag to one tag per section quad (length ``M``): a scalar
        ``str`` tags the whole cap, an array-like is per-quad."""
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
        a hex block.

        The section is translated rigidly along ``axis`` (its placement is
        preserved); ``origin`` shifts the whole block. ``layers`` are the normalized
        copy-plane positions in ``[0, 1]``, strictly increasing, last ``1``;
        ``layers[0]`` is the near cap and ``layers.size - 1`` hex layers span it to
        ``1``. ``first_tag`` / ``last_tag`` name the caps. The straight special case
        of ``loft``."""
        shift: Vec3 = np.asarray(origin, dtype=float)
        base = np.asarray(section.points, dtype=float).reshape(-1, 3) + shift
        axis_u: Vec3 = np.asarray(axis, dtype=float)
        axis_u = axis_u / np.linalg.norm(axis_u)
        offsets = validate_layers(layers, "extrude layers") * float(length)
        # the sweep is a rigid translation, so each slice's shared edge-interior and
        # private quad-interior nodes ride along by the same vector as its corner
        # points -- no block anywhere (and ``origin`` shifts all three alike).
        # each slice reuses the section's B-rep verbatim (same edge LineMesh
        # connectivity, same per-quad edge indices / flips); only coordinates move.
        ho = section.order > 1
        slices = [QuadMesh(
                      LineMesh(base + d * axis_u[None, :], section.lines.lines,
                               order=section.order,
                               interior=(section.lines.interior
                                         + (shift + d * axis_u)[None, None, :]
                                         if ho else None)),
                      section.quad, section.flip,
                      (section.interior + (shift + d * axis_u)[None, None, :]
                       if ho else None),
                      boundaries=section.boundaries,
                      boundary_tags=section.boundary_tags,
                      element_tags=section.element_tags,
                      order=section.order)
                  for d in offsets]
        return cls.loft(slices, first_tag=first_tag, last_tag=last_tag)

    @classmethod
    def loft(
        cls,
        slices: Sequence[QuadMesh],
        *,
        loop: bool = False,
        first_tag: str | Sequence[str] | StrArray = "",
        last_tag: str | Sequence[str] | StrArray = "",
    ) -> HexMesh:
        """Loft a stack of conformal quad profiles into a hex block (the general
        primitive behind ``extrude``, and the top rung of the uniform sweep shared
        with :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` and
        :meth:`QuadMesh.loft <nekmeshpy.quadmesh.QuadMesh.loft>`).

        ``slices`` is ``nz+1`` profiles sharing the same quad connectivity,
        ``boundary_tags``, and ``element_tags``; consecutive profiles form ``nz`` hex
        layers. ``first_tag`` names the first bottom cap (face 5), ``last_tag`` the
        last top cap (face 6) -- each a scalar or a per-quad array. Side faces are
        named from the section's ``boundary_tags`` (unnamed or ``NO_BOUNDARY`` edges
        stay untagged), and every hex inherits its quad's ``element_tags``. Points
        are shared by construction.

        ``loop=True`` makes the sweep **periodic**: the last profile is joined back to
        the *first*, so ``M`` profiles give ``M`` layers instead of ``M-1`` -- one
        extra layer whose top corners are profile 0's own points.  No profile is
        duplicated, so the seam faces are genuine shared entities (``unique_edges`` /
        ``canonical_faces`` resolve them from the shared corner ids) and the closed
        solid is watertight in the sweep direction, e.g. a solid torus lofted from
        disc sections.  A closed sweep has no bottom/top cap, so ``first_tag`` /
        ``last_tag`` with ``loop=True`` raise ``ValueError`` rather than being
        silently dropped, and no cap boundary row is emitted; side faces from the
        section's ``boundary_tags`` are unaffected."""
        slices = list(slices)
        if loop:
            reject_loop_caps("HexMesh.loft", first_tag, last_tag)
        quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
        # section (quad, side) -> name; each swept side face inherits its section edge
        sec_bnd = np.asarray(slices[0].boundaries, dtype=np.int64).reshape(-1, 2)
        sec_tags = slices[0].boundary_tags
        side_name: dict[tuple[int, int], str] = {
            (int(sec_bnd[r, 0]), int(sec_bnd[r, 1])): str(sec_tags[r])
            for r in range(sec_bnd.shape[0])}
        tag_sides = bool(side_name)
        qtag = np.asarray(slices[0].element_tags, dtype=np.str_).reshape(-1)
        M = quads.shape[0]
        n_prof = len(slices)
        # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
        nz = n_prof if loop else n_prof - 1
        # ``nxt[i]`` is the profile layer ``i`` sweeps *to*: i+1 normally, wrapping to
        # 0 for the closing layer of a periodic sweep.
        nxt: IntArray = (np.arange(1, nz + 1, dtype=np.int64) % n_prof if nz
                         else np.zeros(0, dtype=np.int64))
        S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                      for s in slices], axis=0)             # (n_prof, nn, 3)
        nn = S.shape[1]
        points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v

        # Decide handedness once from the first layer and flip the quad template if
        # left-handed; reject a mixed-winding section rather than invert elements.
        signs = np.array([cls._signed_vol(np.vstack([S[0, quads[q], :], S[1, quads[q], :]]))
                          for q in range(M)]) if nz else np.zeros(0)
        if nz and not (np.all(signs > 0) or np.all(signs < 0)):
            raise ValueError(
                "extrude: section is not consistently wound (mixed hex "
                "orientation) -- the section mesher must emit uniform winding")
        flip = bool(nz and signs[0] < 0)
        qw = quads[:, [0, 3, 2, 1]] if flip else quads

        # caps stay faces 5/6 by q (the flip only reorders a quad's 4 corners); a
        # periodic sweep has no cap at all, so it emits none (and rejected the tags).
        first_caps = ([""] * M if loop else cls._cap_tags(first_tag, M))
        last_caps = ([""] * M if loop else cls._cap_tags(last_tag, M))

        hexes = np.empty((nz * M, 8), dtype=np.int64)
        # every layer repeats the section's per-quad tags (hex ``e = i*M + q``).  Tiling
        # keeps the section's own string width: ``np.empty(..., dtype=np.str_)`` is
        # ``<U1`` and would clip each tag to its first character on assignment.
        etags: StrArray = (np.tile(qtag, nz) if qtag.size
                           else np.full(nz * M, "", dtype=np.str_))
        bnd: list[list[int]] = []
        names: list[str] = []
        e = 0
        for i in range(nz):
            j = int(nxt[i])                     # the profile this layer sweeps to
            for q in range(M):
                v = qw[q, :]
                hexes[e] = np.concatenate([i * nn + v, j * nn + v])
                if tag_sides:
                    # section side s -> hex face s, or 5-s when the quad was flipped
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

        # order-N: each hex column is a straight GLL sweep between the two bounding
        # profiles' in-plane blocks, evaluated only at the entity slots -- the two
        # z-face interiors come from the slices' own quad interiors, the four side-face
        # interiors and the cell interior are swept from the slices' edge / quad nodes,
        # the in-slice edges are the slices' shared edge nodes and the vertical edges
        # straight corner blends.  ``scatter_*`` then canonicalizes.
        order = slices[0].order
        if any(s.order != order for s in slices):
            raise ValueError("loft: all slices must share the same order")
        edges, elem_edges, eflip = conform.unique_edges(hexes, 3)
        canonical_conn, elem_faces, face_orient = conform.canonical_faces(hexes)
        edge_nodes: PointArray | None = None
        face_nodes: PointArray | None = None
        interior: PointArray | None = None
        if order > 1:
            g = gll_nodes(order)
            row = order + 1
            m2 = row * row
            SC = np.stack([_slice_block(s, order) for s in slices], axis=0)
            # (nz, M, m2, 3) in-plane blocks of the bottom/top slice of each layer,
            # flattened to hex order e = i*M + q.  ``nxt`` picks the top slice, so the
            # periodic closing layer sweeps back onto profile 0's own block.
            bottom = SC[np.arange(nz, dtype=np.int64)].reshape(nz * M, m2, 3)
            top = SC[nxt].reshape(nz * M, m2, 3)
            if flip:
                kk = np.arange(m2)
                trans = (kk // row) + row * (kk % row)    # transpose the in-plane grid
                bottom = bottom[:, trans, :]
                top = top[:, trans, :]
            E = nz * M
            k2 = (order - 1) ** 2
            eslots = conform._edge_slots(3, order)[:, 1:-1]         # (12, order-1)
            local_e = _sweep_at(bottom, top, g, eslots.ravel(), m2).reshape(
                E, 12, order - 1, 3)
            fslots = conform._face_interior_slots(order)            # (6, k2)
            local_f = _sweep_at(bottom, top, g, fslots.ravel(), m2).reshape(
                E, 6, k2, 3)
            interior = _sweep_at(bottom, top, g,
                                 conform._interior_slots(3, order), m2)
            tol = conform.entity_tol(points)
            edge_nodes = conform.scatter_edge_nodes(
                local_e, elem_edges, eflip, edges.shape[0], tol, "HexMesh.loft")
            face_nodes = conform.scatter_face_nodes(
                local_f, elem_faces, face_orient, canonical_conn.shape[0], tol,
                "HexMesh.loft")
        # the hex edge table unique_edges(hexes, 3) and the shared-face table
        # unique_edges(canonical_conn, 2) are the same array (both canonicalize
        # min-corner-id first over the same global corner ids), so ``edge_nodes``
        # scattered with the hex incidence indexes the shared-face QuadMesh directly.
        q_edges, q_elem_edges, q_flip = conform.unique_edges(canonical_conn, 2)
        edge_lm = LineMesh(points, q_edges, order=order, interior=edge_nodes)
        faces = QuadMesh(edge_lm, q_elem_edges, q_flip, face_nodes, order=order)
        return cls(faces, elem_faces, face_orient, interior,
                   *cls._order_bnd(bnd, names),
                   element_tags=etags, order=order)

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
        shells = QuadMesh.blend(inner, outer, radial)
        # wall tags from the surfaces' per-quad element_tags; scalar arg overrides
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
        """Stitch several hex blocks into one, coordinate-welding coincident seam
        points in a single pass.  ``tol`` is the absolute coincidence distance
        (default ``1e-7`` x the merged bounding-box extent).

        Only points on each block's domain boundary (faces carried by a single hex)
        are weld candidates; interior points are always kept distinct."""
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
        # order-N: the private per-hex interiors just concatenate, but the shared edge /
        # face tables must be rebuilt against the *merged* topology -- gather each
        # block's nodes into its own element-local order, concatenate in merged element
        # order, then re-scatter.  Those scatters are the conformal-weld guard: two
        # blocks that disagree on a welded shared edge / face raise instead of silently
        # welding.
        order = meshes[0].order if meshes else 1
        if any(mm.order != order for mm in meshes):
            raise ValueError("merge: all blocks must share the same order")
        edges, elem_edges, eflip = conform.unique_edges(hexes, 3)
        canonical_conn, elem_faces, face_orient = conform.canonical_faces(hexes)
        edge_nodes: PointArray | None = None
        face_nodes: PointArray | None = None
        interior: PointArray | None = None
        if order > 1:
            local_e: PointArray = np.concatenate(
                [conform.gather_edge_nodes(mm.quads.lines.interior, mm._elem_edges,
                                           mm._edge_flip)
                 for mm in meshes], axis=0)                    # (E,12,order-1,3)
            local_f: PointArray = np.concatenate(
                [conform.gather_face_nodes(mm.quads.interior, mm.hex, mm.face_orient)
                 for mm in meshes], axis=0)                    # (E,6,(order-1)**2,3)
            tol = conform.entity_tol(points)
            edge_nodes = conform.scatter_edge_nodes(
                local_e, elem_edges, eflip, edges.shape[0], tol, "HexMesh.merge")
            face_nodes = conform.scatter_face_nodes(
                local_f, elem_faces, face_orient, canonical_conn.shape[0], tol,
                "HexMesh.merge")
            interior = np.concatenate([mm.interior for mm in meshes], axis=0)
        q_edges, q_elem_edges, q_flip = conform.unique_edges(canonical_conn, 2)
        edge_lm = LineMesh(points, q_edges, order=order, interior=edge_nodes)
        faces = QuadMesh(edge_lm, q_elem_edges, q_flip, face_nodes, order=order)
        return cls(faces, elem_faces, face_orient, interior,
                   *cls._order_bnd(bnd, names),
                   element_tags=etags, order=order)

    @classmethod
    def blend(cls, a: HexMesh, b: HexMesh,
              fractions: FloatArray | Sequence[float]) -> list[HexMesh]:
        """Linearly morph between two conformal blocks ``a`` and ``b`` (identical
        ``hexes``, equal point count), one block per fraction ``t`` with points
        ``(1-t)*a + t*b`` -- ``t=0`` reproduces ``a``, ``t=1`` reproduces ``b``.  Each
        result carries ``a``'s ``hexes``, ``boundaries`` and ``boundary_tags``
        (positional BC markers follow the morph); per-hex ``element_tags`` are left
        for the caller to assign.  The 3-D sibling of
        :meth:`QuadMesh.blend <nekmeshpy.quadmesh.QuadMesh.blend>`.

        The morph is delegated one rung **down the B-rep ladder**: the shared corners,
        shared edge nodes and shared face nodes are exactly the shared-face
        ``QuadMesh`` (whose own corners and edge nodes are in turn its edge
        ``LineMesh``), so
        :meth:`QuadMesh.blend <nekmeshpy.quadmesh.QuadMesh.blend>` produces the blended
        face mesh and this method only lerps what a hex owns privately -- its per-hex
        ``interior`` -- while keeping ``a``'s ``hex`` / ``face_orient`` incidence
        verbatim."""
        A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
        B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "blend: blocks must have equal point counts (got %d, %d); build one "
                "from the other's points so they pair by index"
                % (A.shape[0], B.shape[0]))
        if not np.array_equal(a.hexes, b.hexes):
            raise ValueError(
                "blend: blocks must share identical connectivity (paired by index)")
        if a.order != b.order:
            raise ValueError("blend: blocks must share the same order")
        # identical connectivity => identical edge / face tables, so a's and b's shared
        # edge nodes, shared face nodes and private interiors already pair one-for-one
        # and each morphs with the same lerp the corners get from the blended points.
        # The shared corners, shared edge nodes and shared face nodes *are* the
        # shared-face QuadMesh, so that whole part of the morph is one
        # ``QuadMesh.blend`` of the rung below (which in turn delegates the corners and
        # edge nodes to ``LineMesh.blend``); the result reuses ``a``'s per-hex face
        # indices and D4 codes verbatim -- a blend is a pure point-space morph, so
        # nothing is re-derived.  At order 1 all three tables are empty and this is
        # exactly the plain point blend.
        ho = a.order > 1
        ai, bi = a.interior, b.interior
        fr: FloatArray = np.asarray(fractions, dtype=float).ravel()
        return [cls(faces, a.hex, a.face_orient,
                    (1.0 - t) * ai + t * bi if ho else None,
                    a.boundaries, a.boundary_tags, order=a.order)
                for t, faces in zip(fr, QuadMesh.blend(a.quads, b.quads, fr))]

    # -- boundary queries (topological domain surface) ------------------
    @staticmethod
    def _boundary_mask(hexes: IntArray) -> tuple[IntArray, BoolArray]:
        """``(faces, is_boundary)``: every hex quad face ``(6N,4)`` in Nek order,
        element-major (row ``6e+f``), and a mask of those on the domain boundary."""
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
        """``(K,2)`` of ``[element id, local face (1-6)]`` for every face on the
        topological domain boundary (a quad carried by a single hex). Distinct from
        the tagged ``boundaries``, which may also carry interior planes."""
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
        edge_tags = {s: n for s, n in tags.items() if not s.startswith("z")}
        slices = [QuadMesh.from_grid(P[:, :, k, :], edge_tags=edge_tags,
                                     element_tag=element_tag, order=order)
                  for k in range(P.shape[2])]
        # the loft *is* the result: its sweep-major numbering is carried up unchanged.
        return cls.loft(slices, first_tag=tags.get("z_min", ""),
                        last_tag=tags.get("z_max", ""))

    @staticmethod
    def _order_bnd(
        bnd: Sequence[Sequence[int]] | IntArray,
        names: Sequence[str] | StrArray,
    ) -> tuple[IntArray, StrArray]:
        """Stably order boundary rows by ``(element id, face)``, applying the same
        permutation to the parallel ``names`` array."""
        b: IntArray = np.asarray(bnd, dtype=np.int64).reshape(-1, 2)
        nm: StrArray = np.asarray(names, dtype=np.str_).reshape(-1)
        if b.shape[0]:
            order = np.lexsort((b[:, 1], b[:, 0]))
            b = b[order]
            nm = nm[order]
        return b, nm

    # -- shared-point view ------------------------------------------------
    def weld(self) -> tuple[PointArray, IntArray, int]:
        """Shared-point view ``(points, hexes, n_points)``; the live positions array
        can be mutated in place to reposition the mesh."""
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
        """Watertightness / connectivity report of the welded mesh."""
        from ..model import topology
        X, HC, _ = self.weld()
        return topology.hex_report(X, HC)

    def is_watertight(self) -> bool:
        """``True`` if the mesh boundary is a closed, leak-tight 2-manifold and the
        mesh is a single connected component. Does not imply conformity."""
        rep = self.topology_report()
        return bool(rep["watertight"] and rep["n_components"] == 1)

    def is_conforming(self) -> bool:
        """``True`` if the mesh has no hanging points (no T-junctions)."""
        return bool(self.topology_report()["conformal"])

    def report(self) -> str:
        """Human-readable summary: element/point counts, scaled-Jacobian quality,
        per-name tagged-face counts, and the topology report."""
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
