"""Conforming-O-grid leg builder.

``OGridLeg`` turns a stack of fine interior cross-section rings (opening -> seam)
into the two half-O-grid meshes of one leg, conformal at the junction.  Its
:meth:`build` returns two lists of :class:`~nekmeshpy.geometry.quadmesh.QuadMesh` slices
(shared topology), which :class:`~nekmeshpy.geometry.hexmesh.HexMesh` extrudes into hexes.
Port of the original ``ogrid_leg_from_rings``.
"""

import numpy as np

from ..geometry.polyline import Polyline, Ring
from ..geometry.quadmesh import QuadMesh
from ..ops import trisurf


def half_ogrid(arc, spine, radial, center_scale):
    """Structured HALF-circle O-grid over a half-disk, split along the ``spine``
    curve (A1..A2); the dedicated section mesher that produces one O-grid
    :class:`~nekmeshpy.geometry.quadmesh.QuadMesh` slice.

    arc          : (4*Ntheta+1,3) wall points, arc[0]=A1, arc[-1]=A2
    spine        : Polyline (or (>=2,3) array), central curve A1..A2
    radial       : increasing O-ring layer positions in (0,1], last = 1
    center_scale : inner-block extent as a fraction of the spine

    The outermost O-ring is the open wall arc; its consecutive node pairs are
    recorded as the returned mesh's ``wall_edges`` (the ordered ``arc_ids`` path
    is an implementation detail kept here, not exposed on the QuadMesh).
    """
    arc = np.asarray(arc, dtype=float)
    if not isinstance(spine, Polyline):
        spine = Polyline(spine)
    na = arc.shape[0]
    if (na - 1) % 4 != 0:
        raise ValueError("half_ogrid: arc must have 4*Ntheta+1 points")
    Nt = (na - 1) // 4
    radial = np.asarray(radial, dtype=float).ravel()
    Nr = radial.size
    cs = center_scale

    O = spine.point_at_fraction(0.5)[0]
    sN = (1 - cs) / 2
    sS = (1 + cs) / 2

    # inner rectangle: Coons patch (flat edge on spine, apex on arc)
    fe = spine.point_at_fraction(np.linspace(sN, sS, 2 * Nt + 1))
    Q_N = O + cs * (arc[Nt, :] - O)
    Q_S = O + cs * (arc[3 * Nt, :] - O)
    ae = Q_N + (np.arange(2 * Nt + 1)[:, None] / (2 * Nt)) * (Q_S - Q_N)
    P_N = fe[0, :]
    P_S = fe[-1, :]

    ni = 2 * Nt
    nj = Nt
    rid = np.zeros((ni + 1, nj + 1), dtype=np.int64)
    nodes = []
    for i in range(ni + 1):
        u = i / ni
        for j in range(nj + 1):
            v = j / nj
            left = (1 - v) * P_N + v * Q_N
            right = (1 - v) * P_S + v * Q_S
            bott = fe[i, :]
            top = ae[i, :]
            C = ((1 - v) * bott + v * top + (1 - u) * left + u * right
                 - ((1 - u) * (1 - v) * P_N + u * (1 - v) * P_S
                    + (1 - u) * v * Q_N + u * v * Q_S))
            nodes.append(C)
            rid[i, j] = len(nodes) - 1

    quads = []
    for i in range(ni):
        for j in range(nj):
            quads.append([rid[i, j], rid[i + 1, j], rid[i + 1, j + 1], rid[i, j + 1]])

    # inner perimeter facing the arc, N -> S (4Nt+1 nodes)
    peri = np.concatenate([rid[0, 0:nj + 1],
                           rid[1:ni + 1, nj],
                           rid[ni, nj - 1::-1]])
    nodes = np.array(nodes, dtype=float)
    peripts = nodes[peri, :]

    # outer O-ring: blend perimeter out to the arc
    lid = [peri]
    for r in range(Nr):
        tau = radial[r]
        pts = (1 - tau) * peripts + tau * arc
        pts[0, :] = spine.point_at_fraction((1 - tau) * sN)[0]
        pts[-1, :] = spine.point_at_fraction(sS + tau * (1 - sS))[0]
        base = nodes.shape[0]
        nodes = np.vstack([nodes, pts])
        lid.append(base + np.arange(pts.shape[0]))

    for r in range(Nr):
        a = lid[r]
        b = lid[r + 1]
        for k in range(4 * Nt):
            quads.append([a[k], a[k + 1], b[k + 1], b[k]])

    arc_ids = lid[Nr]                          # outermost layer = the open wall arc
    wall_edges = {frozenset((int(arc_ids[k]), int(arc_ids[k + 1])))
                  for k in range(arc_ids.shape[0] - 1)}
    return QuadMesh(nodes, np.array(quads, dtype=np.int64), wall_edges=wall_edges)


class OGridLeg:
    def __init__(self, fine_rings, seam_ring, spine, surface, params):
        """
        fine_rings : list[Ring] fine interior rings (opening -> seam)
        seam_ring  : Ring or (M,3) conformal seam cross-section (M = 2*nh)
        spine      : Polyline or (nh+1,3) shared central spine at the seam
        surface    : TriMesh to project wall/ring nodes onto
        params     : dict with radial, center_scale, resample_spline, project_to_stl
        """
        self.fine_rings = [r if isinstance(r, Ring) else Ring(r) for r in fine_rings]
        self.frlev = np.asarray(params["frlev"], dtype=float) if "frlev" in params else None
        self.seam_ring = np.asarray(seam_ring.points if isinstance(seam_ring, Polyline)
                                    else seam_ring, dtype=float)
        self.spine_pts = np.asarray(spine.points if isinstance(spine, Polyline)
                                    else spine, dtype=float)
        self.surface = surface
        self.p = params

    def build(self, frlev):
        """Build the leg.  ``frlev`` is the field level of each fine ring in
        (0,1), opening -> seam.  Returns ``(half1, half2)``, each a list of
        ``nr`` :class:`QuadMesh` slices (opening -> seam)."""
        p = self.p
        frlev = np.asarray(frlev, dtype=float)
        seam_ring = self.seam_ring
        M = seam_ring.shape[0]
        nh = M // 2
        Nfine = 4 * M

        La = Polyline(seam_ring[0:nh + 1, :]).length
        Lb = Polyline(np.vstack([seam_ring[nh:M, :], seam_ring[0:1, :]])).length
        f_seam = La / (La + Lb)

        # normalise fine rings to Nfine points
        fr = [r.resample(Nfine) for r in self.fine_rings]

        # align fine rings back from the seam (index 0 -> A1, consistent winding)
        ref = Ring(seam_ring).resample(Nfine)
        for k in range(len(fr) - 1, -1, -1):
            fr[k] = fr[k].align_to(ref)
            ref = fr[k]

        # M-point interior rings with a slowly ramping split fraction
        rings = []
        for k in range(len(fr)):
            f = 0.5 + (f_seam - 0.5) * frlev[k]
            rings.append(fr[k].split_by_fraction(f, nh))
        ni = len(rings)
        nr = ni + 1

        RS = np.zeros((nr, M, 3))
        for k in range(ni):
            RS[k, :, :] = rings[k]
        RS[nr - 1, :, :] = seam_ring
        if p["resample_spline"]:
            RS = self._spline_stack(RS, nr)

        if p["project_to_stl"]:
            sub = RS[0:nr - 1, :, :].reshape((nr - 1) * M, 3)
            sub = trisurf.project_points(self.surface, sub)
            RS[0:nr - 1, :, :] = sub.reshape(nr - 1, M, 3)
        RS[nr - 1, :, :] = seam_ring

        A1 = self.spine_pts[0, :]
        A2 = self.spine_pts[-1, :]
        dev = self.spine_pts - (A1 + (np.arange(nh + 1)[:, None] / nh) * (A2 - A1))
        ringlev = np.arange(nr) / (nr - 1)

        half1 = []
        half2 = []
        for k in range(nr):
            R = RS[k, :, :]
            e1 = R[0, :]
            e2 = R[nh, :]
            spine = (e1 + (np.arange(nh + 1)[:, None] / nh) * (e2 - e1)) + ringlev[k] * dev
            arc1 = R[0:nh + 1, :]
            arc2 = np.vstack([R[nh:M, :], R[0:1, :]])
            half1.append(half_ogrid(arc1, Polyline(spine),
                                    p["radial"], p["center_scale"]))
            half2.append(half_ogrid(arc2, Polyline(spine[::-1, :]),
                                    p["radial"], p["center_scale"]))
        return half1, half2

    @staticmethod
    def _spline_stack(H, Nout):
        """Spline-smooth each node's path down the leg, resample to Nout
        stations.  H is (nslices,Nnode,3)."""
        Nn = H.shape[1]
        Hout = np.zeros((Nout, Nn, 3))
        for j in range(Nn):
            Hout[:, j, :] = Polyline(H[:, j, :]).resample_spline(Nout)
        return Hout
