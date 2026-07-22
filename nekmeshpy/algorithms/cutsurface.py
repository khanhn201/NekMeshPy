"""Surface partitioned into the three bifurcation legs.

``CutSurface`` cuts a :class:`~nekmeshpy.geometry.trimesh.TriMesh` into legs A/B/C along
the seam fields, smooths the seams, solves each leg's own Laplace field, and
builds the shared conformal seam rings + central spine.  Ports of
``cut_surface_by_legs`` / ``smooth_cut_contour`` / ``leg_field`` /
``bifurcation_seam_rings`` (the column-major ``argmax`` tie-break for A1/A2 is
preserved).
"""

import numpy as np

from ..geometry.polyline import Arc, Polyline, Ring
from ..geometry.trimesh import TriMesh
from ..ops import trisurf


class CutSurface:
    def __init__(self, parent, V, faces, cut_edges):
        self.parent = parent                        # original TriMesh
        self.V = np.asarray(V, dtype=float)
        self.faces = [np.asarray(f, dtype=np.int64).reshape(-1, 3) for f in faces]
        self.cut_edges = np.asarray(cut_edges, dtype=np.int64)

    # -----------------------------------------------------------------
    @staticmethod
    def _leg_label(F):
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        lab = np.zeros(a.shape[0], dtype=np.int64)
        lab[(b > 0) & (c < 0)] = 1                  # leg A (trunk)
        lab[(a < 0) & (c > 0)] = 2                  # leg B
        lab[(a > 0) & (b < 0)] = 3                  # leg C
        return lab

    @classmethod
    def from_fields(cls, surface, F):
        """Cut ``surface`` into three legs defined by seam fields ``F`` (nv,3),
        retriangulating every triangle a seam passes through."""
        xyz = surface.xyz
        tri = surface.tri
        nv = xyz.shape[0]
        V = [xyz[i, :].copy() for i in range(nv)]
        cut_edges = [np.array([0, 0], dtype=np.int64) for _ in range(nv)]
        faces = {1: [], 2: [], 3: []}
        lab = cls._leg_label(F)
        ecache = {}

        def edge_pt(vi, vj, fi):
            key = (min(vi, vj), max(vi, vj), fi)
            if key in ecache:
                return ecache[key]
            fi_i = F[vi, fi]
            fi_j = F[vj, fi]
            t = fi_i / (fi_i - fi_j)
            V.append(xyz[vi, :] + t * (xyz[vj, :] - xyz[vi, :]))
            idv = len(V) - 1
            cut_edges.append(np.array([vi, vj], dtype=np.int64))
            ecache[key] = idv
            return idv

        def triple_pt(v):
            Aeq = np.array([[F[v[0], 0], F[v[1], 0], F[v[2], 0]],
                            [F[v[0], 1], F[v[1], 1], F[v[2], 1]],
                            [1.0, 1.0, 1.0]])
            lam = np.linalg.solve(Aeq, np.array([0.0, 0.0, 1.0]))
            V.append(lam @ xyz[v, :])
            idv = len(V) - 1
            cut_edges.append(np.array([-1, -1], dtype=np.int64))
            return idv

        for e in range(tri.shape[0]):
            v = tri[e, :]
            l = lab[v]
            ul = np.unique(l)
            if ul.size == 1:
                faces[int(ul[0])].append([v[0], v[1], v[2]])
            elif ul.size == 2:
                cnt = np.array([np.sum(l == u) for u in ul])
                lone = int(ul[cnt == 1][0])
                pair = int(ul[cnt == 2][0])
                p = int(v[l == lone][0])
                qr = v[l == pair]
                q = int(qr[0])
                r = int(qr[1])
                fi = (6 - lone - pair) - 1
                e1 = edge_pt(p, q, fi)
                e2 = edge_pt(p, r, fi)
                faces[lone].append([p, e1, e2])
                faces[pair].append([q, r, e2])
                faces[pair].append([q, e2, e1])
            else:
                T = triple_pt(v)
                l0, l1, l2 = int(l[0]), int(l[1]), int(l[2])
                m12 = edge_pt(int(v[0]), int(v[1]), (6 - l0 - l1) - 1)
                m23 = edge_pt(int(v[1]), int(v[2]), (6 - l1 - l2) - 1)
                m31 = edge_pt(int(v[2]), int(v[0]), (6 - l2 - l0) - 1)
                faces[l0].append([int(v[0]), m12, T])
                faces[l0].append([int(v[0]), T, m31])
                faces[l1].append([int(v[1]), m23, T])
                faces[l1].append([int(v[1]), T, m12])
                faces[l2].append([int(v[2]), m31, T])
                faces[l2].append([int(v[2]), T, m23])

        Varr = np.array(V, dtype=float)
        ce = np.array(cut_edges, dtype=np.int64)
        faces_list = [np.array(faces[k], dtype=np.int64).reshape(-1, 3) for k in (1, 2, 3)]
        return cls(surface, Varr, faces_list, ce)

    # -----------------------------------------------------------------
    def smooth_seams(self, niter=10, lam=0.6):
        """Slide each inserted cut point along its host edge toward the average
        of its seam neighbours (constrained Laplacian curve smoothing)."""
        V = self.V.copy()
        xyz = self.parent.xyz
        cut_edges = self.cut_edges
        is_cut = np.any(cut_edges != 0, axis=1)

        emap = {}
        for leg in range(3):
            f = self.faces[leg]
            for t in range(f.shape[0]):
                tv = f[t, :]
                for a in range(3):
                    i = int(tv[a])
                    j = int(tv[(a + 1) % 3])
                    if is_cut[i] and is_cut[j]:
                        key = (min(i, j), max(i, j))
                        emap.setdefault(key, []).append(leg)

        nV = V.shape[0]
        adj = [[] for _ in range(nV)]
        for (i, j), legs in emap.items():
            if len(set(legs)) >= 2:
                adj[i].append(j)
                adj[j].append(i)

        movable = [p for p in range(nV) if adj[p] and cut_edges[p, 0] > 0]
        for _ in range(niter):
            Vnew = V.copy()
            for p in movable:
                tgt = V[adj[p], :].mean(axis=0)
                xi = xyz[cut_edges[p, 0], :]
                xj = xyz[cut_edges[p, 1], :]
                d = xj - xi
                tt = np.dot(tgt - xi, d) / np.dot(d, d)
                tt = min(max(tt, 0.02), 0.98)
                Vnew[p, :] = (1 - lam) * V[p, :] + lam * (xi + tt * d)
            V = Vnew
        self.V = V
        return self

    # -----------------------------------------------------------------
    def leg_field(self, leg, gloops):
        """Extract one leg as a standalone sub-mesh and solve Laplace on it: 0
        on its opening, 1 on its seam.  Returns
        ``(sub_TriMesh, us, opening, seam, vids)``."""
        sub, vids = TriMesh.from_faces(self.V, self.faces[leg])
        sloops = [c for c in trisurf.boundary_loops(sub) if c.size >= 3]
        gset = set(int(x) for x in gloops[leg])
        opencnt = np.array([np.sum([1 for x in vids[c] if int(x) in gset]) for c in sloops])
        oi = int(np.argmax(opencnt))
        rest = [i for i in range(len(sloops)) if i != oi]
        si = rest[int(np.argmax([sloops[i].size for i in rest]))]
        opening = sloops[oi]
        seam = sloops[si]
        us = trisurf.solve_dirichlet(
            sub,
            np.concatenate([opening, seam]),
            np.concatenate([np.zeros(opening.size), np.ones(seam.size)]))
        return sub, us, opening, seam, vids

    # -----------------------------------------------------------------
    def seam_rings(self, gloops, n_half):
        """Build the three conformal seam rings + spine.  Returns
        ``(rings: list[Ring], A1, A2, spine: Polyline)``."""
        assert n_half % 4 == 0, "seam_rings: n_half must be a multiple of 4"
        V = self.V

        segV = [None, None, None]
        ordV = [None, None, None]
        for leg in range(3):
            sub, _, _, seam, vids = self.leg_field(leg, gloops)
            segV[leg] = vids[seam]
            ordV[leg] = vids[trisurf.order_boundary_loop(sub, seam)]

        common = np.intersect1d(np.intersect1d(segV[0], segV[1]), segV[2])
        Pc = V[common, :]
        diff = Pc[:, None, :] - Pc[None, :, :]
        Dc = np.sum(diff ** 2, axis=2)
        # MATLAB max(Dc(:)) scans column-major; replicate its tie-break.
        mx = int(np.argmax(Dc.ravel(order="F")))
        ia, ib = np.unravel_index(mx, Dc.shape, order="F")
        iA1 = int(common[ia])
        iA2 = int(common[ib])
        A1 = V[iA1, :]
        A2 = V[iA2, :]

        arcAB, arcAC = self._split_two_arcs(ordV[0], iA1, iA2, segV[1], segV[2])
        bc1, bc2 = self._split_two_arcs(ordV[1], iA1, iA2, segV[2], segV[0])
        segAset = set(int(x) for x in segV[0])
        arcBC = bc1 if all(int(x) in segAset for x in bc2) else bc2

        abP = self._arc_resample(V, arcAB, iA1, n_half + 1)
        acP = self._arc_resample(V, arcAC, iA1, n_half + 1)
        bcP = self._arc_resample(V, arcBC, iA1, n_half + 1)

        rings = [Ring(self._join_arcs(abP, acP, n_half)),
                 Ring(self._join_arcs(abP, bcP, n_half)),
                 Ring(self._join_arcs(acP, bcP, n_half))]
        spine = Polyline((abP + acP + bcP) / 3.0)
        return rings, A1, A2, spine

    @staticmethod
    def _split_two_arcs(ordv, iA1, iA2, segX, segY):
        ordv = np.asarray(ordv).ravel()
        k1 = int(np.flatnonzero(ordv == iA1)[0])
        ordv = np.roll(ordv, -k1)
        k2 = int(np.flatnonzero(ordv == iA2)[0])
        a = ordv[0:k2 + 1]
        b = np.concatenate([ordv[k2:], ordv[:1]])[::-1]
        interiorA = a[1:-1]
        segXset = set(int(x) for x in segX)
        if all(int(x) in segXset for x in interiorA):
            return a, b
        return b, a

    @staticmethod
    def _arc_resample(V, arcverts, iA1, n):
        arcverts = np.asarray(arcverts).ravel()
        if arcverts[0] != iA1:
            arcverts = arcverts[::-1]
        return Arc(V[arcverts, :]).resample(n)

    @staticmethod
    def _join_arcs(p, q, nh):
        qr = q[::-1]
        return np.vstack([p[0:nh, :], qr[0:nh, :]])
