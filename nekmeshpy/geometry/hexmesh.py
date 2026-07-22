"""All-hex mesh container.

``HexMesh`` is a pure hex container.  Canonical storage is ``elements`` (N,8,3)
in Nek node order plus ``boundaries`` (Nbc,3) = ``[element id (0-based),
face (1-6), tag]``.  It is built up by extruding stacks of
:class:`~nekmeshpy.geometry.quadmesh.QuadMesh` cross-section slices into hexes
(:meth:`add_extruded_section`) or by appending single hexes (:meth:`add_hex`),
then frozen with :meth:`finalize`.  A welded shared-node view is produced on
demand by :meth:`weld` (with :meth:`_write_back` its inverse), and
:meth:`classify_nodes` / :meth:`_unique_edges` expose the connectivity the
algorithm modules need.

Everything that operates *on* a finished mesh lives in dedicated modules, taking
the mesh as their first argument: interior repositioning
(:mod:`nekmeshpy.ops.interior`), smoothing (:mod:`nekmeshpy.ops.smoothing`), quality
metrics (:mod:`nekmeshpy.model.quality`), export / generic views
(:mod:`nekmeshpy.io.export`), and plotting (:mod:`nekmeshpy.io.viz`).
"""

import numpy as np

from ..model.physical import PhysicalGroups


class HexMesh:
    # Nek face -> the 4 corner node positions (0-based); row f is face f+1.
    FACE_NODES = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                           [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    def __init__(self, elements=None, boundaries=None, groups=None):
        self._weld_cache = None
        # named physical-group registry (tag <-> name <-> Nek BC code)
        self.groups = groups
        if elements is not None:
            self.elements = np.asarray(elements, dtype=float).reshape(-1, 8, 3)
            self.boundaries = (np.zeros((0, 3), np.int64) if boundaries is None
                               else np.asarray(boundaries, np.int64).reshape(-1, 3))
            self._final = True
            self._hexes = None
            self._bnd = None
        else:
            self.elements = None
            self.boundaries = None
            self._final = False
            self._hexes = []
            self._bnd = []

    # -- sizes -----------------------------------------------------------
    @property
    def n_elements(self):
        self.finalize()
        return self.elements.shape[0]

    @property
    def n_boundaries(self):
        self.finalize()
        return self.boundaries.shape[0]

    @property
    def physical_groups(self):
        """The :class:`~nekmeshpy.model.physical.PhysicalGroups` registry used for
        naming/BC-code lookup; defaults to the built-in Nek table."""
        if self.groups is None:
            self.groups = PhysicalGroups.nek_default()
        return self.groups

    # -- building --------------------------------------------------------
    @staticmethod
    def _signed_vol(P):
        """Sign proxy of the trilinear Jacobian at the hex centre (Nek order)."""
        P = np.asarray(P, dtype=float)
        r = P[[1, 2, 5, 6], :].mean(axis=0) - P[[0, 3, 4, 7], :].mean(axis=0)
        s = P[[2, 3, 6, 7], :].mean(axis=0) - P[[0, 1, 4, 5], :].mean(axis=0)
        t = P[[4, 5, 6, 7], :].mean(axis=0) - P[[0, 1, 2, 3], :].mean(axis=0)
        return float(np.dot(np.cross(r, s), t))

    def add_hex(self, nodes8, orient=True):
        """Append one hex (8,3, Nek order); if ``orient`` and it is left-handed,
        flip to a positive Jacobian.  Returns the element id."""
        if self._final:
            raise RuntimeError("HexMesh already finalized; cannot add hexes")
        P = np.asarray(nodes8, dtype=float)
        if orient and self._signed_vol(P) < 0:
            P = P[[0, 3, 2, 1, 4, 7, 6, 5], :]
        self._hexes.append(P)
        return len(self._hexes) - 1

    def _tag(self, eid, face, tag):
        self._bnd.append([eid, face, tag])

    def tag_face(self, eid, face, tag):
        """Tag one face (``1``-``6``, Nek convention) of element ``eid`` with an
        integer ``tag``.  Building state only (call before :meth:`finalize`);
        lets callers add boundary tags the extruder does not know about (e.g.
        flux-measurement caps on an interior slice).  Returns ``self``."""
        if self._final:
            raise RuntimeError("HexMesh already finalized; cannot tag faces")
        self._tag(int(eid), int(face), int(tag))
        return self

    def add_extruded_section(self, slices, first_cap_tag=0, last_cap_tag=0, wall_tag=0):
        """Extrude a stack of :class:`~nekmeshpy.geometry.quadmesh.QuadMesh` cross-section
        slices into hexes.

        ``slices`` is a list of ``nz+1`` slices that all share the same quad
        connectivity and ``wall_edges``; consecutive slices are connected into
        ``nz`` layers of hexes.  The first slice's bottom cap (face 5) is tagged
        ``first_cap_tag``, the last slice's top cap (face 6) ``last_cap_tag``,
        and any side face whose in-section edge is in the slices' ``wall_edges``
        is tagged ``wall_tag``.  Tags of ``0`` are skipped.

        Returns an ``(nz, M)`` int array of the element ids created (layer i,
        quad q), so callers can tag interior caps themselves (see
        :meth:`tag_face`; used by the bifurcation flux planes).
        """
        if self._final:
            raise RuntimeError("HexMesh already finalized; cannot add hexes")
        quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
        wall_edges = slices[0].wall_edges or set()
        M = quads.shape[0]
        nz = len(slices) - 1
        S = np.stack([np.asarray(s.nodes, dtype=float).reshape(-1, 3) for s in slices], axis=0)
        base = len(self._hexes)
        for i in range(nz):
            for q in range(M):
                v = quads[q, :]
                bseq = v
                hexn = np.vstack([S[i, bseq, :], S[i + 1, bseq, :]])
                if self._signed_vol(hexn) < 0:
                    bseq = v[[0, 3, 2, 1]]
                    hexn = np.vstack([S[i, bseq, :], S[i + 1, bseq, :]])
                self._hexes.append(hexn)
                eid = len(self._hexes) - 1
                if wall_tag > 0:
                    for f in range(4):
                        if frozenset((int(bseq[f]), int(bseq[(f + 1) % 4]))) in wall_edges:
                            self._tag(eid, f + 1, wall_tag)
                if first_cap_tag > 0 and i == 0:
                    self._tag(eid, 5, first_cap_tag)
                if last_cap_tag > 0 and i == nz - 1:
                    self._tag(eid, 6, last_cap_tag)
        return np.arange(base, base + nz * M, dtype=np.int64).reshape(nz, M)

    def finalize(self):
        """Freeze the accumulated hexes/boundaries into arrays (idempotent).

        Boundaries are stably ordered by ``(element id, face)`` so the exported
        block is independent of the order faces were tagged in (interior caps
        tagged after the extrude land in their natural interleaved position)."""
        if not self._final:
            self.elements = np.array(self._hexes, dtype=float).reshape(-1, 8, 3)
            bnd = np.array(self._bnd, dtype=np.int64).reshape(-1, 3)
            if bnd.shape[0]:
                order = np.lexsort((bnd[:, 1], bnd[:, 0]))   # by element, then face
                bnd = bnd[order]
            self.boundaries = bnd
            self._final = True
            self._weld_cache = None
        return self

    # -- welded view -----------------------------------------------------
    def weld(self):
        """Weld coincident corner nodes into one shared vertex set.
        Returns ``(X, HC, nu)`` (cached)."""
        self.finalize()
        if self._weld_cache is not None:
            return self._weld_cache
        elements = self.elements
        N = elements.shape[0]
        P = elements.reshape(8 * N, 3)                  # element-major, node-inner
        scl = np.max(P.max(axis=0) - P.min(axis=0))
        tol = 1e-7 * scl
        keys = np.round(P / tol).astype(np.int64)
        # Vectorized replacement for the per-node dict loop.  It must reproduce
        # the dict's element-major INSERTION-ORDER node labelling exactly: the
        # sparse direct solves and the accept/reject smoothing branches are not
        # permutation-invariant in floating point, so a different node ordering
        # would perturb results at ~1e-13 and break byte-identity.  We relabel
        # np.unique's sorted groups by first-occurrence index to recover it.
        _, first_idx, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(first_idx, kind="stable")     # groups in first-seen order
        new_label = np.empty(order.size, dtype=np.int64)
        new_label[order] = np.arange(order.size)
        ic = new_label[inverse]
        X = P[first_idx[order], :].copy()
        HC = ic.reshape(N, 8)
        self._weld_cache = (X, HC, X.shape[0])
        return self._weld_cache

    def _write_back(self, X, HC):
        for e in range(HC.shape[0]):
            self.elements[e, :, :] = X[HC[e, :], :]
        self._weld_cache = None

    def classify_nodes(self, twall):
        """Flag welded nodes: ``(is_wall, is_fixed)``.  A node on both is
        treated as fixed."""
        _, HC, nu = self.weld()
        is_wall = np.zeros(nu, dtype=bool)
        is_fixed = np.zeros(nu, dtype=bool)
        for b in range(self.boundaries.shape[0]):
            elem = int(self.boundaries[b, 0])
            face = int(self.boundaries[b, 1])
            tag = int(self.boundaries[b, 2])
            ids = HC[elem, self.FACE_NODES[face - 1, :]]
            if tag == twall:
                is_wall[ids] = True
            else:
                is_fixed[ids] = True
        is_wall[is_fixed] = False
        return is_wall, is_fixed

    # -- connectivity helpers (used by interior / smoothing) ------------
    _HE_SECTION = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                            [4, 5], [5, 6], [6, 7], [7, 4]], dtype=np.int64)
    _HE_AXIAL = np.array([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)

    @staticmethod
    def _unique_edges(HC, he):
        Ei = HC[:, he[:, 0]].ravel()
        Ej = HC[:, he[:, 1]].ravel()
        return np.unique(np.sort(np.column_stack([Ei, Ej]), axis=1), axis=0)
