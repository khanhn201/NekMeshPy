"""
Bifurcation vessel mesher: load a triangulated surface, solve three Laplacian
seam fields, cut into legs A/B/C, build conformal seam rings + a central spine,
refit each station's scanned wall ring as a truncated Fourier series so the
high-order nodes sit on a genuine curve, extrude each leg's O-grid sections into
hexes, weld, smooth, export::

    PYTHONPATH=. python examples/bifurcation.py

Writes ``bifurcation.re2`` (Nek5000/NekRS) and ``bifurcation.vtu``.
"""

import logging
import os

import numpy as np

from nekmeshpy import (
    PhysicalGroups,
    TriMesh,
    export,
    hexmesh,
    linemesh,
    quadmesh,
    smoothing,
    trimesh,
    viz,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters (defaults reproduce the bundled ``car`` case) ----------------
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VTX = os.path.join(_DATA, "car.vtx")     # vertices  (N x 3)
TRI = os.path.join(_DATA, "car.tri")     # triangles (M x 3, 1-based)

N_HALF = 8                    # half-ring resolution; MULTIPLE OF 4
N_SLICES = 30                 # cross-sections per leg (hex layers = n_slices)
MIN_LOOP_PTS = 6              # ignore isocontour loops smaller than this
CENTER_SCALE = 0.5            # inner square-core size (fraction of diameter)
RADIAL = np.array([0.0, 0.4, 0.7, 0.9, 1.0])   # O-ring layer positions (first 0, last 1.0)
PROJECT_TO_STL = True
# polynomial order.  The wall is genuinely curved at ORDER > 1: each interior
# station's scanned ring is refit as a truncated Fourier series (``fourier_ring``
# below) and meshed with ``LineMesh.loft_fn``, so the wall's high-order nodes sit on
# that analytic loop rather than on chords between the scanned samples.  The spine
# stays linear (``LineMesh.loft`` straight-subdivides it), which is all a flat
# half-disc seam needs.
ORDER = 3
FOURIER_KEEP = 0.5            # fraction of the rFFT modes kept in the wall refit
# Smoothing is off: both relaxers move corner nodes only and reject order > 1, and
# the Fourier wall already removes the facet-scale noise they were there to polish.
# To exercise them, set ORDER = 1 with "conduction" / SMOOTH_ITERS = 8.
SMOOTHING_METHOD = None    # "none" | "bilinear" | "conduction" | "winslow"
SMOOTH_ITERS = 0             # post-assembly untangle/polish sweeps (0 = off)
SMOOTH_LAMBDA = 0.5
FLUX_OFFSET = 2              # hex layers in from the outlet cap (0 = off)
OUT_NAME = "bifurcation"
EXPORT_RE2 = True
EXPORT_VTK = True
EXPORT_FLD = True            # Nek field file: the GLL nodes .re2 cannot carry
PLOT = False


# -- seam / opening solvers --------------------------------------------------
def order_openings(surf):
    """Order the three boundary loops: A = trunk (lowest mean Z), B/C by mean X.
    Returns 3 vertex-index arrays."""
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
    """Laplace with Neumann on loop ``neumann``, Dirichlet ``dvals`` on the other
    two, shifted to zero mean on the free loop."""
    dpoints, dv = [], []
    for k in range(3):
        if k == neumann:
            continue
        g = np.asarray(gloops[k]).ravel()
        dpoints.append(g)
        dv.append(dvals[k] * np.ones(g.size))
    u = trimesh.ops.solve_dirichlet(surf, np.concatenate(dpoints), np.concatenate(dv))
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
    retriangulating triangles a seam crosses.  Returns ``(V, faces)``."""
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
    """One leg as a sub-mesh with Laplace solved (0 on opening, 1 on seam).
    Returns ``(sub, us, opening, seam, vids)``."""
    sub, vids = TriMesh.from_faces(V, faces[leg])
    sloops = [c for c in trimesh.ops.boundary_loops(sub) if c.size >= 3]
    gset = set(int(x) for x in gloops[leg])
    opencnt = np.array([np.sum([1 for x in vids[c] if int(x) in gset]) for c in sloops])
    oi = int(np.argmax(opencnt))
    rest = [i for i in range(len(sloops)) if i != oi]
    si = rest[int(np.argmax([sloops[i].size for i in rest]))]
    opening = sloops[oi]
    seam = sloops[si]
    us = trimesh.ops.solve_dirichlet(
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


def _arc_curve(V, arcverts, iA1, n, keep=FOURIER_KEEP):
    """Analytic ``A1 -> A2`` seam arc: the scanned polyline's deviation from its own
    chord, refit as a truncated **sine** series in the normalized arc-length parameter
    ``s``, and returned as a callable ``p(s)`` on ``[0, 1]``.

    This is ``fourier_ring``'s open-curve sibling, and the basis is chosen for the one
    property the seam needs: every ``sin(k*pi*s)`` vanishes at both ends, so ``A1`` and
    ``A2`` -- the triple points where all three arcs meet -- stay **bit-exact** however
    many modes are kept.  That is what lets the three legs keep welding: each arc is
    fitted once, globally, and the *same* mesh is handed to both legs that share it, so
    no leg ever sees a privately-refitted seam (which is why the seam ring could not
    simply go through ``fourier_wall`` -- refitting per leg would mix the two arcs that
    leg happens to see).

    Dropping the upper modes removes the STL facet noise for the same reason it does on
    a station ring, and ``keep`` is applied to the modes the *mesh* can resolve
    (``n`` elements -> ``n-1`` interior nodes -> modes ``1..n-1``), mirroring
    ``fourier_ring``'s ``keep * (M // 2 + 1)``.
    """
    arcverts = np.asarray(arcverts).ravel()
    if arcverts[0] != iA1:
        arcverts = arcverts[::-1]
    N = 4 * n                                  # dense uniform-arc-length fit samples
    s = np.linspace(0.0, 1.0, N + 1)
    Q = trimesh.ops.resample_polyline(V[arcverts, :], s)
    a1, a2 = Q[0, :], Q[-1, :]
    dev = Q - (a1 + s[:, None] * (a2 - a1))    # vanishes at both ends by construction
    K = max(1, int(keep * (n - 1)))
    kk = np.arange(1, K + 1)
    # discrete sine transform (type I) of the interior samples, truncated to K modes
    S = np.sin(np.pi * np.outer(np.arange(1, N), kk) / N)      # (N-1, K)
    B = (2.0 / N) * (S.T @ dev[1:N, :])                        # (K,3) sine coefficients

    def p(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return (a1 + t[:, None] * (a2 - a1)
                + np.sin(np.pi * np.outer(t, kk)) @ B)

    return p


def _arc_mesh(p, n, element_tag):
    """Mesh an ``_arc_curve`` into ``n`` high-order elements, evenly spaced by arc
    length.  ``LineMesh.loft_fn`` evaluates ``p`` at every node -- corners *and* the
    private GLL interiors -- so no node lands on a chord."""
    return linemesh.loft_fn(p, linemesh.arclength_fractions(p, n),
                          order=ORDER, element_tags=[element_tag] * n)


def _ring(p, q):
    """Close two shared-endpoint ``A1 -> A2`` arcs into one loop by welding them at
    ``A1`` and ``A2`` (:meth:`LineMesh.merge`); ``q`` is reversed so the traversal
    runs ``A1 -> A2`` down ``p`` then ``A2 -> A1`` back up ``q`` without crossing.

    ``reverse`` carries ``q``'s high-order nodes with it; re-lofting its points
    would straight-subdivide them and lose the curve at ``ORDER > 1``."""
    return linemesh.merge([p, linemesh.reverse(q)])


def seam_rings(V, faces, gloops, n_half):
    """Build the three conformal seam rings + spine.
    Returns ``(rings, A1, A2, spine)``."""
    segV = [None, None, None]
    ordV = [None, None, None]
    for leg in range(3):
        sub, _, _, seam, vids = leg_field(V, faces, leg, gloops)
        segV[leg] = vids[seam]
        ordV[leg] = vids[trimesh.ops.order_boundary_loop(sub, seam)]

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

    # Each shared arc is refit and meshed exactly once here, so the two legs that
    # share it are handed bit-identical geometry and the blocks still weld.
    abP = _arc_mesh(_arc_curve(V, arcAB, iA1, n_half), n_half, "wall")
    acP = _arc_mesh(_arc_curve(V, arcAC, iA1, n_half), n_half, "wall")
    bcP = _arc_mesh(_arc_curve(V, arcBC, iA1, n_half), n_half, "wall")

    rings = [_ring(abP, acP), _ring(abP, bcP), _ring(acP, bcP)]
    spine = linemesh.loft((abP.points + acP.points + bcP.points) / 3.0, order=ORDER)
    return rings, A1, A2, spine


# -- analytic wall from a scanned ring ---------------------------------------
def fourier_ring(P, keep=FOURIER_KEEP):
    """Refit a closed scanned ring as a truncated Fourier series ``p(t)``, periodic
    on ``[0, 2*pi)`` with sample ``j`` at ``t = 2*pi*j/M``.

    ``x``, ``y`` and ``z`` are transformed independently (``np.fft.rfft`` against the
    uniform parameter) and only the lowest ``keep`` fraction of the modes is retained;
    the rest -- facet-scale noise from the STL projection, which a high-order wall would
    otherwise resolve faithfully -- is dropped.  The result is a genuine closed-form
    curve, so ``LineMesh.loft_fn`` can place every GLL node *on* it instead of on the
    chords between the samples (see the straight-GLL-subdivision trap in CLAUDE.md).
    """
    P = np.asarray(P, dtype=float)
    M = P.shape[0]
    nk = max(2, int(keep * (M // 2 + 1)))     # modes 0..nk-1; never the Nyquist mode
    C = np.fft.rfft(P, axis=0)[:nk, :] / M
    C[1:, :] *= 2.0                           # each retained mode is a conjugate pair
    k = np.arange(nk)

    def p(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.real(np.exp(1j * t[:, None] * k[None, :]) @ C)

    return p


def fourier_wall(P, order, element_tag):
    """Closed high-order wall ``LineMesh`` through ``fourier_ring(P)``, sampled at the
    same ``M`` parameters as ``P`` so index 0 / ``M//2`` still land on the spine rails.
    ``LineMesh.loft_fn`` only ever makes an open chain, so weld the two ends into a loop
    the way the toolkit expects (``LineMesh.merge``)."""
    M = np.asarray(P).shape[0]
    return linemesh.merge([linemesh.loft_fn(
        fourier_ring(P), np.linspace(0.0, 2.0 * np.pi, M + 1),
        order=order, element_tags=[element_tag] * M)])


# -- O-grid leg builder ------------------------------------------------------
def ogrid_leg(fine_rings, seam_ring, spine, surface, frlev, *,
              radial, center_scale, project_to_stl, smoothing_method):
    """Turn a stack of fine interior rings (opening -> seam) into ``nr`` full-disk
    :class:`QuadMesh` slices: each station's two half-O-grids are repositioned
    then merged along the shared spine."""
    seam_pts = seam_ring.points
    spine_pts = spine.points
    frlev = np.asarray(frlev, dtype=float)
    M = seam_pts.shape[0]
    nh = M // 2

    # conformalize the scanned interior rings onto the seam ring's topology (the
    # one intrinsic interpolation that can't be pushed to the caller)
    rings = trimesh.ops.conform_ring_stack(fine_rings, seam_pts, frlev, nh)
    ni = len(rings)
    nr = ni + 1

    RS = np.zeros((nr, M, 3))
    for k in range(ni):
        RS[k, :, :] = rings[k]
    RS[nr - 1, :, :] = seam_pts

    if project_to_stl:
        sub = RS[0:nr - 1, :, :].reshape((nr - 1) * M, 3)
        sub = trimesh.ops.project_points(surface, sub)
        RS[0:nr - 1, :, :] = sub.reshape(nr - 1, M, 3)
    RS[nr - 1, :, :] = seam_pts

    A1 = spine_pts[0, :]
    A2 = spine_pts[-1, :]
    dev = spine_pts - (A1 + (np.arange(nh + 1)[:, None] / nh) * (A2 - A1))
    ringlev = np.arange(nr) / (nr - 1)

    slices = []
    for k in range(nr):
        R = RS[k, :, :]
        # The seam ring (k = nr-1) is *shared* with the two other legs, so it cannot be
        # refit here -- a per-leg refit would mix the two seam arcs this leg happens to
        # see and the blocks would no longer weld.  It arrives already curved instead:
        # ``seam_rings`` fits each of the three shared arcs once, globally
        # (``_arc_curve``), so use that ring verbatim.  Re-lofting its *points* would
        # straight-subdivide it and throw the curve away at ORDER > 1 -- which is
        # exactly what used to leave this one station with 63 degrees of corner at its
        # element joints while every other station sat within 0.2 degrees.
        # Every interior station is private, so refit it into an analytic curve.
        if k == nr - 1:
            wall = seam_ring
        else:
            wall = fourier_wall(R, ORDER, "wall")
        wpts = wall.points
        e1 = wpts[0, :]
        e2 = wpts[nh, :]
        # this station's spine is the (nh+1)-point deviating diameter A1..A2;
        # spined_ogrid splits the wall loop along it into two half-O-grids and merges.
        spn = (e1 + (np.arange(nh + 1)[:, None] / nh) * (e2 - e1)) + ringlev[k] * dev
        # spined_ogrid meshes the spine exactly at the points given, so resample this
        # scanned diameter to the sampling it consumes here rather than hand it a
        # curve at the wrong count (there is no analytic form to evaluate: the
        # deviation comes off the STL, so the chord is the honest interpolant).
        spn = trimesh.ops.resample_polyline(
            spn, quadmesh.spine_fractions(nh // 4, radial, center_scale))
        # reposition interior stations; leave opening cap (k=0) and pinned seam
        # (k=nr-1) as raw algebraic fill.  Wall tagged on the loop (see
        # flow_past_cylinder.py) so spined_ogrid rides it onto the wall edges.
        m = smoothing_method if 0 < k < nr - 1 else None
        slices.append(quadmesh.spined_ogrid(
            wall, radial, spine=linemesh.loft(spn, order=ORDER),
            center_scale=center_scale, smoothing_method=m))
    return slices


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
    fr, frlev = trimesh.ops.extract_rings(sub, us, levels, MIN_LOOP_PTS)
    slices = ogrid_leg(fr, rings[leg], spine, surf, frlev,
                       radial=RADIAL, center_scale=CENTER_SCALE,
                       project_to_stl=PROJECT_TO_STL,
                       smoothing_method=SMOOTHING_METHOD)
    flux_name = flux_name_for(outlet_name[leg])
    off = FLUX_OFFSET
    # opening cap = leg outlet; seam end is interior.  With a flux plane, split
    # the leg there (a cap of the downstream segment); merge re-joins them.
    if flux_name and 0 < off < len(slices) - 1:
        blocks.append(hexmesh.loft(slices[:off + 1], first_tag=outlet_name[leg]))
        blocks.append(hexmesh.loft(slices[off:], first_tag=flux_name))
    else:
        blocks.append(hexmesh.loft(slices, first_tag=outlet_name[leg]))

mesh = hexmesh.merge(blocks)

if SMOOTH_ITERS > 0:
    smoothing.smooth(mesh, surf, smooth_iters=SMOOTH_ITERS, smooth_lambda=SMOOTH_LAMBDA,
                     wall="wall", project_to_stl=PROJECT_TO_STL)

print(hexmesh.report(mesh))
if EXPORT_VTK:
    export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
if EXPORT_RE2:
    export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
if EXPORT_FLD:
    # .re2 is corner-only at any order; the field file carries the full GLL block,
    # so this is the export that actually preserves the ORDER = 3 geometry.
    export.to_fld(mesh, OUT_NAME + "0.f00001")
if PLOT:
    viz.plot(mesh, ["wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"], OUT_NAME)
print("bifurcation: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
