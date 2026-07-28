"""
Loads a triangulated vessel surface, solves three intrinsic-Laplacian seam
fields, cuts it into legs A/B/C, builds conformal seam rings + a central spine,
extrudes each leg's O-grid cross-sections into hexes, welds the legs, smooths,
and exports.  Everything is composed from the ``nekmeshpy`` toolkit here -- there
is no mesher class; edit the constants below and re-run::

    PYTHONPATH=. python examples/bifurcation.py

Writes ``bifurcation.re2`` / ``.rea`` (Nek5000/NekRS) and ``bifurcation.vtk``.
"""

import logging
import os

import numpy as np

from nekmeshpy import (
    HexMesh,
    LineMesh,
    PhysicalGroups,
    QuadMesh,
    TriMesh,
    export,
    smoothing,
    trisurf,
    viz,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters (defaults reproduce the bundled ``car`` case) ----------------
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VTX = os.path.join(_DATA, "car.vtx")     # vertices  (N x 3)
TRI = os.path.join(_DATA, "car.tri")     # triangles (M x 3, 1-based)

N_HALF = 8                    # half-ring resolution; MULTIPLE OF 4
N_SLICES = 20                 # cross-sections per leg (hex layers = n_slices)
MIN_LOOP_PTS = 6              # ignore isocontour loops smaller than this
CENTER_SCALE = 0.5            # inner square-core size (fraction of diameter)
RADIAL = np.array([0.0, 0.4, 0.6, 0.8, 1.0])   # O-ring layer positions (first 0, last 1.0)
RESAMPLE_SPLINE = True
PROJECT_TO_STL = True
SMOOTHING_METHOD = "bilinear"    # "bilinear" | "conduction" | "winslow"
SMOOTH_ITERS = 8              # post-assembly untangle/polish sweeps (0 = off)
SMOOTH_LAMBDA = 0.5
FLUX_OFFSET = 2              # hex layers in from the outlet cap (0 = off)
OUT_NAME = "bifurcation"
EXPORT_RE2 = True
EXPORT_VTK = True
PLOT = False


# -- seam / opening solvers --------------------------------------------------
def order_openings(surf):
    """Order the three boundary loops A/B/C: A = trunk (lowest mean Z), B/C =
    branches by mean X.  Returns a list of 3 vertex-index arrays."""
    loops = surf.boundary_loops()
    assert len(loops) == 3, "expected exactly 3 boundary loops, got %d" % len(loops)
    Z = surf.points[:, 2]
    X = surf.points[:, 0]
    meanZ = np.array([Z[c].mean() for c in loops])
    iA = int(np.argmin(meanZ))
    rest = [i for i in range(3) if i != iA]
    meanX = np.array([X[loops[i]].mean() for i in rest])
    order = np.argsort(meanX, kind="stable")
    return [loops[iA], loops[rest[order[0]]], loops[rest[order[1]]]]


def conduction_field(surf, gloops, neumann, dvals):
    """One conduction problem: Laplace with natural Neumann on loop ``neumann``,
    Dirichlet ``dvals`` on the other two, shifted to zero mean on the free loop."""
    dpoints, dv = [], []
    for k in range(3):
        if k == neumann:
            continue
        g = np.asarray(gloops[k]).ravel()
        dpoints.append(g)
        dv.append(dvals[k] * np.ones(g.size))
    u = trisurf.solve_dirichlet(surf, np.concatenate(dpoints), np.concatenate(dv))
    return u - np.mean(u[gloops[neumann]])


def seam_fields(surf, gloops):
    """The three conduction seam fields U (nv, 3)."""
    nan = np.nan
    dvals = [[nan, 0, 1], [1, nan, 0], [0, 1, nan]]
    U = np.zeros((surf.n_points, 3))
    for k in range(3):
        U[:, k] = conduction_field(surf, gloops, k, dvals[k])
    return U


# -- cut the surface into legs -----------------------------------------------
def leg_label(F):
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    lab = np.zeros(a.shape[0], dtype=np.int64)
    lab[(b > 0) & (c < 0)] = 1                   # leg A (trunk)
    lab[(a < 0) & (c > 0)] = 2                   # leg B
    lab[(a > 0) & (b < 0)] = 3                   # leg C
    return lab


def cut_surface_from_fields(surf, F):
    """Cut ``surf`` into three legs defined by seam fields ``F`` (nv, 3),
    retriangulating every triangle a seam passes through.  Returns
    ``(V, faces)``."""
    xyz = surf.points
    tri = surf.tris
    nv = xyz.shape[0]
    V = [xyz[i, :].copy() for i in range(nv)]
    faces = {1: [], 2: [], 3: []}
    lab = leg_label(F)
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
        ecache[key] = idv
        return idv

    def triple_pt(v):
        Aeq = np.array([[F[v[0], 0], F[v[1], 0], F[v[2], 0]],
                        [F[v[0], 1], F[v[1], 1], F[v[2], 1]],
                        [1.0, 1.0, 1.0]])
        lam = np.linalg.solve(Aeq, np.array([0.0, 0.0, 1.0]))
        V.append(lam @ xyz[v, :])
        idv = len(V) - 1
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
    faces_list = [np.array(faces[k], dtype=np.int64).reshape(-1, 3) for k in (1, 2, 3)]
    return Varr, faces_list


def leg_field(V, faces, leg, gloops):
    """Extract one leg as a sub-mesh and solve Laplace (0 on opening, 1 on seam).
    Returns ``(sub, us, opening, seam, vids)``."""
    sub, vids = TriMesh.from_faces(V, faces[leg])
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


def _arc_resample(V, arcverts, iA1, n):
    arcverts = np.asarray(arcverts).ravel()
    if arcverts[0] != iA1:
        arcverts = arcverts[::-1]
    return LineMesh.open(V[arcverts, :]).resample(np.linspace(0.0, 1.0, n))


def _join_arcs(p, q, nh):
    return np.vstack([p.points[0:nh], q.points[::-1][0:nh]])


def seam_rings(V, faces, gloops, n_half):
    """Build the three conformal seam rings + spine.  Returns
    ``(rings, A1, A2, spine)``."""
    segV = [None, None, None]
    ordV = [None, None, None]
    for leg in range(3):
        sub, _, _, seam, vids = leg_field(V, faces, leg, gloops)
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

    arcAB, arcAC = _split_two_arcs(ordV[0], iA1, iA2, segV[1], segV[2])
    bc1, bc2 = _split_two_arcs(ordV[1], iA1, iA2, segV[2], segV[0])
    segAset = set(int(x) for x in segV[0])
    arcBC = bc1 if all(int(x) in segAset for x in bc2) else bc2

    abP = _arc_resample(V, arcAB, iA1, n_half + 1)
    acP = _arc_resample(V, arcAC, iA1, n_half + 1)
    bcP = _arc_resample(V, arcBC, iA1, n_half + 1)

    rings = [LineMesh.loop(_join_arcs(abP, acP, n_half)),
             LineMesh.loop(_join_arcs(abP, bcP, n_half)),
             LineMesh.loop(_join_arcs(acP, bcP, n_half))]
    spine = LineMesh.open((abP.points + acP.points + bcP.points) / 3.0)
    return rings, A1, A2, spine


# -- O-grid leg builder ------------------------------------------------------
def _spline_stack(H, Nout):
    """Spline-smooth each point's path down the leg, resample to Nout stations."""
    Nn = H.shape[1]
    Hout = np.zeros((Nout, Nn, 3))
    for j in range(Nn):
        Hout[:, j, :] = LineMesh.open(H[:, j, :]).resample_spline(Nout).points
    return Hout


def ogrid_leg(fine_rings, seam_ring, spine, surface, frlev, *,
              radial, center_scale, resample_spline, project_to_stl, smoothing_method):
    """Turn a stack of fine interior rings (opening -> seam) into a list of ``nr``
    full-disk :class:`QuadMesh` cross-section slices: each station's two
    half-O-grids are repositioned then merged along the shared spine."""
    fine_rings = [r if isinstance(r, LineMesh) else LineMesh.loop(r) for r in fine_rings]
    seam_ring = seam_ring.points
    spine_pts = spine.points
    frlev = np.asarray(frlev, dtype=float)
    M = seam_ring.shape[0]
    nh = M // 2
    Nfine = 4 * M

    La = LineMesh.open(seam_ring[0:nh + 1, :]).length
    Lb = LineMesh.open(np.vstack([seam_ring[nh:M, :], seam_ring[0:1, :]])).length
    f_seam = La / (La + Lb)

    fr = [r.resample(np.linspace(0.0, 1.0, Nfine, endpoint=False)) for r in fine_rings]
    ref = LineMesh.loop(seam_ring).resample(np.linspace(0.0, 1.0, Nfine, endpoint=False))
    for k in range(len(fr) - 1, -1, -1):
        fr[k] = fr[k].align_to(ref)
        ref = fr[k]

    rings = []
    for k in range(len(fr)):
        f = 0.5 + (f_seam - 0.5) * frlev[k]
        rings.append(fr[k].split_by_fraction(f, nh).points)
    ni = len(rings)
    nr = ni + 1

    RS = np.zeros((nr, M, 3))
    for k in range(ni):
        RS[k, :, :] = rings[k]
    RS[nr - 1, :, :] = seam_ring
    if resample_spline:
        RS = _spline_stack(RS, nr)

    if project_to_stl:
        sub = RS[0:nr - 1, :, :].reshape((nr - 1) * M, 3)
        sub = trisurf.project_points(surface, sub)
        RS[0:nr - 1, :, :] = sub.reshape(nr - 1, M, 3)
    RS[nr - 1, :, :] = seam_ring

    A1 = spine_pts[0, :]
    A2 = spine_pts[-1, :]
    dev = spine_pts - (A1 + (np.arange(nh + 1)[:, None] / nh) * (A2 - A1))
    ringlev = np.arange(nr) / (nr - 1)

    half1, half2 = [], []
    for k in range(nr):
        R = RS[k, :, :]
        e1 = R[0, :]
        e2 = R[nh, :]
        spn = (e1 + (np.arange(nh + 1)[:, None] / nh) * (e2 - e1)) + ringlev[k] * dev
        arc1 = R[0:nh + 1, :]
        arc2 = np.vstack([R[nh:M, :], R[0:1, :]])
        # reposition interior stations; leave the opening cap (k=0) and the
        # pinned seam (k=nr-1) as the raw algebraic fill
        m = smoothing_method if 0 < k < nr - 1 else None
        # the arc IS the wall: tag it at the line level (one tag per arc segment),
        # so half_ogrid rides it onto the wall edges (see flow_past_cylinder.py).
        half1.append(QuadMesh.half_ogrid(
            LineMesh.open(arc1, element_tags=["wall"] * (len(arc1) - 1)),
            LineMesh.open(spn), radial,
            center_scale=center_scale, smoothing_method=m))
        half2.append(QuadMesh.half_ogrid(
            LineMesh.open(arc2, element_tags=["wall"] * (len(arc2) - 1)),
            LineMesh.open(spn[::-1, :]), radial,
            center_scale=center_scale, smoothing_method=m))
    return [QuadMesh.merge([half1[k], half2[k]]) for k in range(nr)]


def flux_name_for(outlet_name):
    """Flux-plane boundary name for a leg's outlet name, or "" if it has none."""
    if FLUX_OFFSET <= 0:
        return ""
    if outlet_name == "top_outlet_2":
        return "flux_1"
    if outlet_name == "top_outlet_1":
        return "flux_2"
    return ""


# -- pipeline (flat driver) --------------------------------------------------
surf = TriMesh.from_files(VTX, TRI)

gloops = order_openings(surf)
U = seam_fields(surf, gloops)

V, faces = cut_surface_from_fields(surf, U)
rings, A1, A2, spine = seam_rings(V, faces, gloops, N_HALF)

outlet_name = ["trunk_outlet", "top_outlet_1", "top_outlet_2"]
levels = np.linspace(0, 1, N_SLICES + 2)[1:-1]
# name -> Nek BC code / integer id, applied only at export (byte-exact reference)
GROUPS = PhysicalGroups.nek_default()

blocks = []
for leg in range(3):
    sub, us, _, _, _ = leg_field(V, faces, leg, gloops)
    fr, frlev = trisurf.extract_rings(sub, us, levels, MIN_LOOP_PTS)
    slices = ogrid_leg(fr, rings[leg], spine, surf, frlev,
                       radial=RADIAL, center_scale=CENTER_SCALE,
                       resample_spline=RESAMPLE_SPLINE, project_to_stl=PROJECT_TO_STL,
                       smoothing_method=SMOOTHING_METHOD)
    flux_name = flux_name_for(outlet_name[leg])
    off = FLUX_OFFSET
    # opening cap = the leg outlet; the seam end is interior (no far cap).  When
    # there is a flux plane, split the leg there so it becomes a cap of the
    # downstream segment; merge re-joins the two into a shared interior face.
    if flux_name and 0 < off < len(slices) - 1:
        blocks.append(HexMesh.loft(slices[:off + 1], first_tag=outlet_name[leg]))
        blocks.append(HexMesh.loft(slices[off:], first_tag=flux_name))
    else:
        blocks.append(HexMesh.loft(slices, first_tag=outlet_name[leg]))

mesh = HexMesh.merge(blocks)

if SMOOTH_ITERS > 0:
    smoothing.smooth(mesh, surf, smooth_iters=SMOOTH_ITERS, smooth_lambda=SMOOTH_LAMBDA,
                     wall="wall", project_to_stl=PROJECT_TO_STL)

export.summary(mesh)
if EXPORT_VTK:
    export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
if EXPORT_RE2:
    export.to_re2(mesh, OUT_NAME, groups=GROUPS)
if PLOT:
    viz.plot(mesh, ["wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"], OUT_NAME)
print("bifurcation: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
