"""Femoral-style oblique T-junction: generate the surface, then mesh it exactly the way
``carotid.py`` meshes a scanned one.

``carotid.py`` loads a triangulated vessel and finds everything about the junction by
solving on it.  This mesher does the same, on a surface it **builds** rather than loads:
a main vessel (axis ``X``, radius ``R_MAIN``) and a branch (radius ``R_BRANCH``) teeing
off at ``BRANCH_ANGLE``, joined on a flat elliptical seam and ringed by a trough.  The
surface is written out as ``femoral.stl`` and then handed to the same pipeline --
conduction seam fields, a sign-based cut into three legs, conformal seam rings from
Fourier-refit arcs, an O-grid per station, loft, weld::

    PYTHONPATH=. python examples/femoral.py

Writes ``femoral.stl``, ``femoral.re2`` (Nek5000/NekRS) and ``femoral.vtu``.

**The seam is flat.** Where the two cylinders truly intersect, the seam wanders in ``z``
-- lowest at the saddles, highest fore and aft -- so a trough cut a fixed depth below it
undulates too.  Cutting the branch cylinder with the plane ``z = Z_SEAM`` instead gives
an exact ellipse at one height, and the trough around it is then the same depth
everywhere.  ``Z_SEAM`` sits ``SEAM_OFFSET`` below the true intersection's lowest point,
so the branch is sunk slightly into the main pipe rather than sitting on it.

**The trough** falls ``TROUGH_DEPTH`` below that rim onto a level floor and climbs back
onto the cylinder ``TROUGH_WIDTH`` out.  Its two stages are deliberately decoupled: the
descent uses only the rim and floor heights, so nothing can pull the floor off level, and
the climb back is flat at both ends so it contributes nothing at the bottom.  The descent
is a half-period sine, which leaves the rim with a finite slope -- the trough is cut into
the mouth rather than shelved around it.

**What the conduction solve decides.** Everything about *where the mesh is cut* comes
from three Laplace fields, conceptually the same idea as the carotid's -- except solved
volumetrically over the tet scaffold rather than on the surface, so it is which *tet*
belongs to which leg, where the three seam arcs run, and where the two triple points
sit.  The geometry above only says what the wall *is*; it never says where to put a
seam.
"""

import logging
import os
import struct
import sys

import numpy as np

from nekmeshpy import (
    TetMesh,
    TriMesh,
    fields,
    hexmesh,
    linemesh,
    quadmesh,
    tetmesh,
    trimesh,
    writer,
)
from nekmeshpy.core import conform

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import femoral_vol as fvol  # noqa: E402  (needs the path above)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- geometry ----------------------------------------------------------------
R_MAIN = 4.0                  # main vessel radius
R_BRANCH = 3.0                # branch radius (must be < R_MAIN: the branch has to fit)
BRANCH_ANGLE = 45.0           # degrees between the branch axis and +x (0 < a <= 90)
SEAM_OFFSET = 0.0             # how far the flat seam sinks below the true intersection's
                              # lowest point
TROUGH_DEPTH = 1.0            # how far the trough floor sits below that rim
TROUGH_WIDTH = 3.0            # trough width outward from the rim
RUN_MAIN = 10 * 2 * R_MAIN    # straight main run each side of the junction, 10 diameters
RUN_BRANCH = 10 * 2 * R_BRANCH

# -- Total depth is TROUGH_DEPTH+SEAM_OFFSET+1.35

# -- surface tessellation (the "scan" this mesher is handed) -----------------
NT = 24                       # nodes across each half of the mouth
NU = NDN = 250                # nodes along the main pipe, up- and downstream.  **This is
                              # the one that resolves the indent**: the trough is only
                              # TROUGH_WIDTH wide and lies along the axis, so it is the
                              # axial cell that has to be small.  At 60 the cell was 1.33
                              # and the whole dent spanned two of them.
NB = 24                       # nodes round the un-holed band of the main pipe.  Angular
                              # resolution buys nothing for the trough -- raising this
                              # only makes the cells *more* elongated the wrong way.
NS = 150                      # nodes along the branch (axial there too)
GRADE = 1.0                   # **total** axial stretch of the tessellation: the cells at
                              # an opening are this many times those at the rim.  1.0 is
                              # uniform.  Stated as a total, not a per-cell ratio, so it
                              # cannot compound with the node count -- a per-cell 1.5 over
                              # 60 cells is a 2e10 stretch, which collapses the cells at
                              # the rim to nothing.

# -- mesh --------------------------------------------------------------------
SCAFFOLD = dict(NT=40, NU=180, NDN=180, NB=72, NS=150, GRADE=1.0)  # tet scaffold:
                              # graded like the volume field, near-isotropic at the seam.
                              # gmsh cannot refine a *discrete* surface, so these set the
                              # floor on tet size at the wall -- ``TET_NEAR`` below it
                              # does nothing.  The grading is what keeps that affordable:
                              # 0.11 triangles at the junction, 1.25 down the legs.
TET_NEAR = 0.357              # tet size at the junction -- 0.45/2**(1/3), i.e. twice the
TET_FAR = 0.357               # element density.  Equal means **no grading**: one
TET_RAMP = 30.0               # size everywhere, so ``TET_RAMP`` does nothing.
                              # The binding constraint is that a hex layer cannot be
                              # thinner than the tets it is cut from: an isosurface of a
                              # P1 field is C0, with kinks the size of a tet, so two
                              # stations closer together than that interleave and the
                              # sweep inverts.  With the layers graded into the seam the
                              # thinnest is ~0.34, so the tets there must be ~0.11.
DELAUNAY_ROUNDS = 20          # edge-flip sweeps toward the surface Delaunay condition
RELAX_PASSES = 30             # relaxation sweeps over the junction, to take the shear out
RELAX_RADIUS = 15.0            # ... within this far of the seam.  0 passes disables.
SPLIT_TOL = 0.50              # wall/interior cut-off when splitting an interface               # ... over this distance from the junction
N_HALF = 12                   # half-ring resolution; MULTIPLE OF 4
NEAR_LEN = 2.0                # the uniform run, in leg **diameters** out from the
                              # junction.  The junction is where the wall actually does
                              # something -- the crater, the rim, the three-way weld --
                              # and constant layer thickness across it is what draws the
                              # trough; the legs beyond are diameters of nothing much.
NEAR_LEN_BRANCH = 3.0         # ...except down the branch, which gets three times the run
                              # for the same layer count, so its layers are three times
                              # thicker.  A node carried onto the wall moves a fixed
                              # distance, so what decides whether that distortion matters
                              # is how big it is *relative to the layer* -- and the branch
                              # is where the snap has furthest to reach, off the crater
                              # and round the rim.
N_UNIFORM = 12                # layers in that uniform run
N_GRADED = 20                 # layers from there out to the outlet.  Their growth is not
                              # a knob: it is solved for, so the first graded layer equals
                              # the uniform one and the last lands on the cap.  One
                              # grading across the whole leg cannot serve both ends --
                              # fine enough for the trough wastes elements down a straight
                              # pipe, coarse enough for the pipe draws the trough with
                              # three layers.
FLUX_OFFSET = 2               # hex layers in from the outlet cap to the flux plane
                              # (0 = off).  Splitting the loft there is what names it.
MIN_LOOP_PTS = 6              # ignore isocontour loops smaller than this
CENTER_SCALE = 0.8            # inner square-core size (fraction of diameter)
RADIAL = np.array([0.0, 0.6, 1.0])   # O-ring layers (first 0, last 1.0)
PROJECT_TO_STL = True
SNAP_MAX = 0.20               # farthest a node may be carried onto the analytic wall.
                              # Beyond this it is not being projected, it is being moved
                              # somewhere unrelated to its neighbours -- leave it be.
SNAP_AMBIG = 0.10             # when the crater and the branch tube are this close to
                              # equidistant the side is a coin toss, and the two choices
                              # tear the surface between adjacent nodes.  Leave those be
                              # too: near the corner, not snapping beats snapping wrong.
ORDER = 2                     # the wall is genuinely curved above 1: each station's
                              # ring is refit as a Fourier series and meshed with
                              # ``loft_fn``, so its nodes sit on that loop, not on chords
FOURIER_KEEP = 0.5            # fraction of the rFFT modes kept in the wall refit
RIM_KEEP = 0.5                # ditto for a *station's* wall ring, which is the boundary
                              # of a marching-tets isosurface and so wanders along the
                              # pipe by about a tet.  Lower than FOURIER_KEEP because
                              # this is removing tet-scale noise, not resolving a shape:
                              # the ring is a near-circle and its real content is in the
                              # first few modes.
TET_CACHE = "data/femoral_tets.npz"          # under ``examples/data/``, beside this
                              # script rather than in whatever directory it was run from:
                              # the cache belongs to the mesher, not to the caller.
                              # The tet mesh and its conduction fields,
                              # keyed on the scaffold knobs and the tet size.  gmsh plus
                              # six CG solves is ~2.5 minutes and none of it changes while
                              # the O-grid is being worked on.
SURFACE_CACHE = "data/femoral_%s.npz"        # likewise; the built surfaces, keyed on the knobs
                              # above.  Relaxing the junction costs real time and the
                              # surface is settled, so rebuilding both of them on every
                              # run is pure waste.  Delete the files (or change a knob,
                              # which invalidates the key) to rebuild.
OUT_NAME = "femoral"
EXPORT_STL = False
PLOT_STL = False              # show the surface and stop, without meshing.  For tuning
                              # NT/NU/NB/NS by eye: change a knob, run, look.
EXPORT_RE2 = True
EXPORT_VTK = True
EXPORT_FLD = True

FLUX_UPSTREAM, FLUX_DOWNSTREAM = "flux_upstream", "flux_downstream"
# the flux planes are interior surfaces, so they carry Nek's own ``f1``/``f2`` codes
# rather than a flow condition -- naming a plane is not constraining it
GROUPS = {
    "wall": "W  ",
    "inlet": "v  ",
    "outlet": "int",
    "branch": "O  ",
    "flux_1": {FLUX_UPSTREAM: "f1 ", FLUX_DOWNSTREAM: None},
    "flux_2": {FLUX_UPSTREAM: "f2 ", FLUX_DOWNSTREAM: None},
}

# -- derived -----------------------------------------------------------------
if not 0.0 < R_BRANCH < R_MAIN:
    raise ValueError("femoral needs 0 < R_BRANCH < R_MAIN (the branch must fit)")
_A = np.radians(BRANCH_ANGLE)
COS_A, SIN_A = np.cos(_A), np.sin(_A)
if SIN_A <= 0.0:
    raise ValueError("femoral needs 0 < BRANCH_ANGLE <= 90 degrees")

def _per_cell(total, n):
    """The per-cell ratio that spreads a *total* stretch over ``n`` cells.

    ``fields.geometric_spacing`` takes a per-cell ratio, which compounds: the same number
    that grades 26 cells sensibly annihilates 60 of them.  Feeding it a total keeps the
    grading meaning the same thing however finely the surface is sampled."""
    return float(total) ** (1.0 / max(int(n) - 1, 1))


#: The branch axis: through the origin, on the main axis.
AXIS = np.array([COS_A, 0.0, SIN_A])

Z_NAT = np.sqrt(R_MAIN ** 2 - R_BRANCH ** 2)   # lowest point of the true intersection
Z_SEAM = Z_NAT - SEAM_OFFSET                   # the flat seam plane
Z_FLOOR = Z_SEAM - TROUGH_DEPTH                # the trough's level floor
THETA_1 = np.arccos(R_BRANCH / R_MAIN)         # the mouth spans |y| <= R_BRANCH, so its
THETA_2 = np.pi - THETA_1                      # angular span survives the flattening


# -- the surface -------------------------------------------------------------
def branch_point(t, s):
    """A point of the branch cylinder at angle ``t``, ``s`` along its axis."""
    t = np.asarray(t, dtype=float)
    s = np.broadcast_to(np.asarray(s, dtype=float), t.shape)
    return np.stack([-R_BRANCH * np.sin(t) * SIN_A + s * COS_A,
                     R_BRANCH * np.cos(t),
                     R_BRANCH * np.sin(t) * COS_A + s * SIN_A], axis=-1)


def seam_axial(t):
    """``s`` where the branch cylinder crosses ``z = Z_SEAM``: substitute the branch
    point's ``z`` and solve.  Every seam point then lands on the plane exactly."""
    t = np.asarray(t, dtype=float)
    return (Z_SEAM - R_BRANCH * np.sin(t) * COS_A) / SIN_A


def seam_point(t):
    """The seam -- a plane through a cylinder, so an exact ellipse, all at one ``z``."""
    return branch_point(t, seam_axial(t))


def branch_angle_at(theta):
    """The branch angle whose seam point sits at main-pipe angle ``theta``.  Both
    surfaces give the same ``y``, so ``R_BRANCH cos t = R_MAIN cos theta`` fixes ``t`` up
    to the sign that picks the leading or trailing half of the mouth."""
    return np.arccos(np.clip(R_MAIN * np.cos(np.asarray(theta, dtype=float)) / R_BRANCH,
                             -1.0, 1.0))


def rim_x(theta, t_sign):
    """Where the mouth's rim sits in ``x`` at main-pipe angle ``theta``."""
    return seam_point(t_sign * branch_angle_at(theta))[..., 0]


_TC = np.linspace(-np.pi, np.pi, 4001)[:-1]
_SEAM_XY = seam_point(_TC)[:, :2]


_SEAM_TREE = None


def _seam_distance(xy):
    """Distance from each ``(x,y)`` to the seam ellipse.

    Through a tree, not brute force.  One call against a few thousand ellipse samples is
    cheap either way, but ``main_wall`` calls this and the junction relaxation calls
    ``main_wall`` twenty-five times per pass -- the brute-force version turned a minute of
    relaxation into many.  The ellipse never moves, so the tree is built once."""
    global _SEAM_TREE
    if _SEAM_TREE is None:
        from scipy.spatial import cKDTree
        _SEAM_TREE = cKDTree(_SEAM_XY)
    return _SEAM_TREE.query(np.asarray(xy, dtype=float).reshape(-1, 2))[0]


def crater_z(u, zc):
    """Crown height at distance ``u`` outward from the rim, over a cylinder standing
    ``zc`` high at that ``y``.

    Two decoupled stages.  ``0 -> W/2`` is the trough alone -- a half-period sine from
    the rim down to ``Z_FLOOR``, written in terms of those two heights and nothing else,
    so the floor is that level and the climb cannot pull it off.  ``W/2 -> W`` adds the
    return to the cylinder through a raised cosine that is flat at both ends and so
    contributes nothing at the bottom.

    Running one ramp the whole way instead, minus a sine, pins only the *value* at
    ``W/2``: the ramp tilts the curve, the true minimum slides inboard of it and dips
    past the floor by however steep the climb is.  That is a floor following the crown
    again, which is what the flat seam exists to stop."""
    w = np.clip(u / TROUGH_WIDTH, 0.0, 1.0)
    climb = np.where(w <= 0.5, 0.0, 0.5 * (1.0 - np.cos(2.0 * np.pi * (w - 0.5))))
    return Z_SEAM - TROUGH_DEPTH * np.sin(np.pi * w) + (zc - Z_SEAM) * climb


def main_wall(theta, x):
    """The main pipe, cratered down onto the flat rim and the level floor beyond it.

    ``minimum(..., z)`` keeps the crater one-directional -- it cuts the wall down, never
    lifts it -- which is what makes it fade out by itself down the flanks, where the pipe
    already stands below the floor, and leave the belly alone."""
    theta = np.asarray(theta, dtype=float)
    x = np.broadcast_to(np.asarray(x, dtype=float), theta.shape)
    y = R_MAIN * np.cos(theta)
    z0 = R_MAIN * np.sin(theta)
    u = _seam_distance(np.column_stack([x.ravel(), y.ravel()])).reshape(theta.shape)
    z = crater_z(u, np.abs(z0))
    return np.stack([x, y, np.where(z0 > 0.0, np.minimum(z, z0), z0)], axis=-1)


def _nearest_on_main(Q, iters=16):
    """The genuinely nearest point of the main wall, by Gauss-Newton on ``(theta, x)``.

    Taking the wall at the query point's *own* ``(theta, x)`` is a radial reset, not a
    projection: it is the nearest point only where the surface is a plain cylinder.  Over
    the crater, where the wall dives in ``z``, the nearest point sits at a different
    ``theta`` and a different ``x`` -- so a radial reset shears whatever moved the point
    there, which is what made smoothing tear the mesh up around the mouth."""
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    th = np.arctan2(Q[:, 2], Q[:, 1])
    x = Q[:, 0].copy()
    # Away from the crater the wall is a plain cylinder, where the radial reset already
    # *is* the nearest point -- so only iterate on the points the crater actually reaches.
    # That is a few percent of a leg, and the iteration is the expensive part.
    near = _seam_distance(np.column_stack([Q[:, 0], Q[:, 1]])) < TROUGH_WIDTH + 1.0
    if not near.any():
        return main_wall(th, x)
    out = main_wall(th, x)
    th, x, Qn = th[near], x[near], Q[near]
    h = 1e-6
    for _ in range(int(iters)):
        S = main_wall(th, x)
        r = S - Qn
        j0 = (main_wall(th + h, x) - main_wall(th - h, x)) / (2.0 * h)
        j1 = (main_wall(th, x + h) - main_wall(th, x - h)) / (2.0 * h)
        a11 = np.einsum("ij,ij->i", j0, j0)
        a12 = np.einsum("ij,ij->i", j0, j1)
        a22 = np.einsum("ij,ij->i", j1, j1)
        b1 = -np.einsum("ij,ij->i", j0, r)
        b2 = -np.einsum("ij,ij->i", j1, r)
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        dth = (b1 * a22 - b2 * a12) / det
        dx = (a11 * b2 - a12 * b1) / det
        # the crater makes this non-smooth in places, so cap the step and let it walk
        step = np.maximum(np.abs(dth), np.abs(dx) / max(R_MAIN, 1e-30))
        damp = np.minimum(1.0, 0.3 / np.maximum(step, 1e-30))
        th = th + damp * dth
        x = x + damp * dx
    out[near] = main_wall(th, x)
    return out


def snap_to_wall(Q):
    """Snap points onto the **exact** analytic wall -- the crater or the branch tube --
    *except* where snapping is not a well-posed thing to do.

    Nothing here needs a triangulation: the surface is a closed form, so a point can be
    put on it exactly instead of on the nearest facet of a stand-in for it.  Two candidates
    are formed -- the main wall and the branch cylinder -- and the nearer wins, which sorts
    out which tube a point belongs to without asking.  On the seam the two agree, because
    the seam lies on both.

    Two cases where the nearer does *not* win, and the point is simply left where it is:

    ``SNAP_AMBIG`` -- the two candidates are near enough equidistant that the choice is a
    coin toss.  That is the medial axis of the corner where the tubes meet, and a
    nearest-point map is genuinely discontinuous across it: neighbouring nodes fall on
    opposite sides and the surface between them is torn.  There is no right answer to pick
    here, so pick nothing.

    ``SNAP_MAX`` -- the nearest wall is further away than a projection should ever reach.
    Moving a node that far is not projecting it, it is teleporting it, and it arrives
    somewhere unrelated to its neighbours.  Measured on this mesh: the median node moves
    0.013 and the 90th percentile 0.023, but the tail runs to 1.07 -- on a trough 1 deep.
    Those are the nodes that drew the creases, and they are better left on the surface they
    were lifted onto, slightly off the wall, than snapped somewhere consistent with
    nothing."""
    Q = np.asarray(Q, dtype=float).reshape(-1, 3)
    a = _nearest_on_main(Q)
    s = Q @ AXIS
    rad = Q - s[:, None] * AXIS[None, :]
    n = np.linalg.norm(rad, axis=1)
    b = (s[:, None] * AXIS[None, :]
         + R_BRANCH * rad / np.maximum(n, 1e-30)[:, None])
    da = np.linalg.norm(a - Q, axis=1)
    db = np.linalg.norm(b - Q, axis=1)
    out = np.where((da <= db)[:, None], a, b)
    undecidable = np.abs(da - db) < SNAP_AMBIG
    too_far = np.minimum(da, db) > SNAP_MAX
    return np.where((undecidable | too_far)[:, None], Q, out)


def delaunay_flip(V, F, rounds=DELAUNAY_ROUNDS):
    """Flip edges toward the surface Delaunay condition.

    This moves no nodes at all -- it only rechooses which diagonal splits each quad -- so
    the geometry stays exactly the analytic surface and nothing needs projecting back.
    That is the whole reason it succeeds where relaxing the vertices failed: the defect at
    a patch seam is not that the nodes sit badly, it is that the two structured blocks
    triangulate their quads along opposing diagonals, and no amount of moving nodes will
    change a diagonal.

    The test is the standard one for a triangulated surface: an edge is Delaunay when the
    two angles facing it sum to no more than pi, and flipping when they exceed it raises
    the smallest angle in the pair.  A flip is refused when it would duplicate an existing
    edge or turn either new triangle over, which is what keeps a non-convex pair of
    triangles from folding."""
    F = np.array(F, dtype=np.int64, copy=True)
    for _ in range(int(rounds)):
        e = np.concatenate([F[:, [1, 2]], F[:, [2, 0]], F[:, [0, 1]]])
        opp = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
        tri = np.tile(np.arange(F.shape[0]), 3)
        key = np.sort(e, axis=1)
        o = np.lexsort((key[:, 1], key[:, 0]))
        key, opp, tri = key[o], opp[o], tri[o]
        pair = np.flatnonzero((key[:-1] == key[1:]).all(axis=1))
        if pair.size == 0:
            break
        t1, t2 = tri[pair], tri[pair + 1]
        c, d = opp[pair], opp[pair + 1]
        a, b = key[pair, 0], key[pair, 1]

        # angles facing the edge, at c and at d
        def ang(p, u, v):
            x, y = V[u] - V[p], V[v] - V[p]
            cs = np.einsum("ij,ij->i", x, y) / np.maximum(
                np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-30)
            return np.arccos(np.clip(cs, -1.0, 1.0))

        want = np.flatnonzero(ang(c, a, b) + ang(d, a, b) > np.pi + 1e-9)
        if want.size == 0:
            break
        edges_now = set(map(tuple, key.tolist()))
        busy = np.zeros(F.shape[0], dtype=bool)   # one flip per triangle per round, so
        done = 0                                  # the edge map stays valid as we go
        for k in want:
            i, j, cc, dd = int(t1[k]), int(t2[k]), int(c[k]), int(d[k])
            if busy[i] or busy[j] or cc == dd:
                continue
            if (min(cc, dd), max(cc, dd)) in edges_now:
                continue                       # the flip would duplicate an edge
            # take the edge's direction from the triangle as stored, so the replacements
            # inherit its winding rather than the sorted key's arbitrary order
            r = int(np.flatnonzero(F[i] == cc)[0])
            aa, bb = int(F[i, (r + 1) % 3]), int(F[i, (r + 2) % 3])
            n = np.cross(V[bb] - V[aa], V[cc] - V[aa])
            m1 = np.cross(V[aa] - V[cc], V[dd] - V[cc])
            m2 = np.cross(V[dd] - V[cc], V[bb] - V[cc])
            if (np.dot(m1, n) <= 0.0 or np.dot(m2, n) <= 0.0
                    or np.linalg.norm(m1) < 1e-14 or np.linalg.norm(m2) < 1e-14):
                continue                       # would fold or collapse a triangle
            F[i] = (cc, aa, dd)
            F[j] = (cc, dd, bb)
            busy[i] = busy[j] = True
            edges_now.add((min(cc, dd), max(cc, dd)))
            done += 1
        if not done:
            break
    return F


def relax_junction(V, F, passes=RELAX_PASSES, radius=RELAX_RADIUS):
    """Even out the tessellation around the junction, leaving the geometry alone.

    The mouth patches lay their axial grid between the opening and ``rim_x(theta)``, and
    ``rim_x`` turns over almost vertically as theta reaches the saddle -- the mouth's
    width in ``x`` stops changing there, the way a circle's does at its tangent point.  So
    the last rows of those patches are sheared hard against the band's fixed grid, and the
    seam along ``THETA_1`` / ``THETA_2`` shows up as two lines of slivers.

    That seam carries no geometry: it is where the patches were split, not a crease, so
    the nodes are free to slide along it.  Relaxing them does exactly that.  What makes it
    safe now is that ``snap_to_wall`` is a genuine nearest-point projection -- when it was
    a radial reset it sheared back whatever the relaxation had just fixed, and smoothing
    measured worse than doing nothing.

    Pinned: the openings, and the mouth rim, which *is* a crease and has to stay put."""
    if not passes:
        return V
    V = np.array(V, dtype=float, copy=True)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = np.concatenate([e, e[:, ::-1]])

    from scipy.spatial import cKDTree
    pin = np.zeros(V.shape[0], dtype=bool)
    key = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uk, cnt = np.unique(key, axis=0, return_counts=True)
    pin[np.unique(uk[cnt == 1])] = True                     # the three openings
    rim = seam_point(np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False))
    pin[cKDTree(rim).query(V)[0] < 1e-9] = True             # the mouth rim
    near = np.linalg.norm(V - np.array([0.0, 0.0, Z_SEAM]), axis=1) < radius
    free = near & ~pin
    if not free.any():
        return V

    for _ in range(int(passes)):
        for lam in (0.5, -0.53):                            # Taubin: shrink, unshrink
            acc = np.zeros_like(V)
            tot = np.zeros(V.shape[0])
            np.add.at(acc, e[:, 0], V[e[:, 1]])
            np.add.at(tot, e[:, 0], 1.0)
            ok = free & (tot > 0)
            V[ok] += lam * (acc[ok] / tot[ok, None] - V[ok])
        V[free] = snap_to_wall(V[free])
    return V


def plot_stl(surface, radius=9.0):
    """Show the generated surface, so the tessellation knobs can be tuned by eye.

    Set ``PLOT_STL`` and the script stops here rather than going on to mesh: the point is
    to change ``NT``/``NU``/``NB``/``NS`` and look, without waiting for a tet mesh and a
    conduction solve first.

    ``radius`` is the half-width of the opening view around the junction, which is the
    only part worth looking at closely -- the legs are ten diameters of straight pipe."""
    import matplotlib
    for backend in ("qtagg", "tkagg", "gtk4agg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for k in list(matplotlib.rcParams):          # every key belongs to this viewer
        if k.startswith("keymap."):
            matplotlib.rcParams[k] = []

    V, T = surface.points, surface.tris
    e = [np.linalg.norm(V[T[:, (i + 1) % 3]] - V[T[:, i]], axis=1) for i in range(3)]
    lo, hi = np.minimum.reduce(e), np.maximum.reduce(e)
    ar = hi / np.maximum(lo, 1e-30)
    cen = V[T].mean(axis=1)
    mid = np.array([0.0, 0.0, Z_SEAM])
    rad = np.linalg.norm(cen - mid, axis=1)
    u = _seam_distance(np.column_stack([cen[:, 0], cen[:, 1]]))
    dent = (u > 0.0) & (u < TROUGH_WIDTH) & (cen[:, 2] > 0.0)
    print("surface: %d tris | aspect med %.2f p95 %.2f max %.2f | edge %.3f..%.3f"
          % (T.shape[0], np.median(ar), np.percentile(ar, 95), ar.max(),
             lo.min(), hi.max()))
    print("   indent: %d tris, median edge %.3f -> %.1f cells across a %.1f-wide trough"
          % (int(dent.sum()), np.median(hi[dent]),
             TROUGH_WIDTH / max(np.median(hi[dent]), 1e-30), TROUGH_WIDTH))

    state = {"view": "junction", "colour": "plain", "wire": True}
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    def draw():
        ax.clear()
        if state["view"] == "junction":
            sel, half, ctr = np.flatnonzero(rad < radius + 3.0), radius, mid
        elif state["view"] == "indent":
            sel, half, ctr = np.flatnonzero(dent | (rad < 4.0)), 5.5, mid
        else:
            sel = np.arange(0, T.shape[0], max(1, T.shape[0] // 40000))
            b0, b1 = V.min(0), V.max(0)
            ctr, half = 0.5 * (b0 + b1), 0.5 * float((b1 - b0).max()) * 1.02
        if state["colour"] == "ar":
            fc = matplotlib.colormaps["inferno_r"](np.clip((ar[sel] - 1.0) / 4.0, 0, 1))
        elif state["colour"] == "edge":
            fc = matplotlib.colormaps["viridis"](
                np.clip(hi[sel] / max(np.percentile(hi, 98), 1e-30), 0, 1))
        else:
            fc = "#b9c6cc"
        ax.add_collection3d(Poly3DCollection(
            V[T[sel]], facecolor=fc, edgecolor="k" if state["wire"] else "none",
            linewidth=0.12))
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("%s -- %s, %d of %d tris\n"
                     "j junction/all | i indent | a aspect | e edge | p plain | w wire "
                     "| r reset | q quit"
                     % (OUT_NAME + ".stl", state["view"], sel.size, T.shape[0]),
                     fontsize=9)
        fig.canvas.draw_idle()

    def onkey(ev):
        if ev.key == "j":
            state["view"] = "all" if state["view"] != "all" else "junction"
        elif ev.key == "i":
            state["view"] = "indent"
        elif ev.key in ("a", "e", "p"):
            state["colour"] = {"a": "ar", "e": "edge", "p": "plain"}[ev.key]
        elif ev.key == "w":
            state["wire"] = not state["wire"]
        elif ev.key == "r":
            ax.view_init(elev=22, azim=-58)
            fig.canvas.draw_idle()
            return
        elif ev.key == "q":
            plt.close(fig)
            return
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", onkey)
    ax.view_init(elev=22, azim=-58)
    draw()
    plt.show()


def wall_loop(ring, order, tag="wall"):
    """A station's wall ring as a closed ``LineMesh``, straight from the points given.

    Deliberately *not* refit as an analytic curve the way ``carotid.py`` refits its
    scanned rings.  That ring is the boundary of the station's isosurface, and the
    station's interior nodes are on that same isosurface; refitting the ring would slide
    the wall element's corners onto a different surface from its own interior, which is
    exactly the bulge that shows up along the wall layer.  Carotid has to refit because a
    surface solve gives it nothing in the interior to be consistent with."""
    R = np.asarray(ring, dtype=float).reshape(-1, 3)
    return linemesh.merge([linemesh.loft(np.vstack([R, R[:1]]), order=order,
                                         element_tags=tag)])


def build_surface():
    """The junction as one manifold ``TriMesh`` with exactly three boundary loops.

    Four structured patches -- main wall up- and downstream of the mouth, the band round
    the rest of the pipe, and the branch tube.  The band's ``x`` grid is the union of the
    other two at the saddle, so the three meet node-for-node along ``THETA_1`` /
    ``THETA_2``; the rim is computed once and *injected* into every patch that touches
    it, so they agree there exactly rather than to whatever the crater's sampled distance
    happens to give."""
    th_hole = np.linspace(THETA_1, THETA_2, NT + 1)
    # geometric along the pipe, fine at the rim: the tets inherit the surface's spacing
    # at the wall, so the surface has to be graded the same way the volume field is or
    # the two disagree and the transition fills with slivers
    xi_up = 1.0 - fields.geometric_spacing(NU, _per_cell(GRADE, NU))[::-1]
    xi_dn = fields.geometric_spacing(NDN, _per_cell(GRADE, NDN))
    th_band = np.linspace(THETA_2, THETA_1 + 2.0 * np.pi, NB + 1)

    x_saddle = float(seam_point(np.array([0.0]))[0, 0])
    x_in = float(rim_x(th_hole, +1.0).min()) - RUN_MAIN
    x_out = float(rim_x(th_hole, -1.0).max()) + RUN_MAIN
    s_branch = float(seam_axial(_TC).max()) + RUN_BRANCH

    verts, tris = [], []

    def add(P, wrap=False):
        ni, nj, _ = P.shape
        base = len(verts)
        verts.extend(P.reshape(-1, 3))
        idx = base + np.arange(ni * nj).reshape(ni, nj)
        ring = np.vstack([idx, idx[:1]]) if wrap else idx
        a, b, c, d = ring[:-1, :-1], ring[:-1, 1:], ring[1:, 1:], ring[1:, :-1]
        tris.extend(np.stack([a, b, c], -1).reshape(-1, 3))
        tris.extend(np.stack([a, c, d], -1).reshape(-1, 3))

    t_lead = branch_angle_at(th_hole)
    c_lead, c_trail = seam_point(t_lead), seam_point(-t_lead)

    xl = rim_x(th_hole, +1.0)
    p2 = main_wall(np.broadcast_to(th_hole[:, None], (NT + 1, NU + 1)),
                   x_in + xi_up[None, :] * (xl[:, None] - x_in))
    p2[:, -1, :] = c_lead
    add(p2)

    xt = rim_x(th_hole, -1.0)
    p3 = main_wall(np.broadcast_to(th_hole[:, None], (NT + 1, NDN + 1)),
                   xt[:, None] + xi_dn[None, :] * (x_out - xt[:, None]))
    p3[:, 0, :] = c_trail
    add(p3)

    xb = np.concatenate([x_in + xi_up * (x_saddle - x_in),
                         x_saddle + xi_dn[1:] * (x_out - x_saddle)])
    add(main_wall(np.broadcast_to(th_band[:, None], (NB + 1, xb.size)),
                  np.broadcast_to(xb[None, :], (NB + 1, xb.size))))

    t_ring = np.concatenate([t_lead, -t_lead[::-1][1:-1]])
    s0 = seam_axial(t_ring)
    pb = branch_point(np.broadcast_to(t_ring[:, None], (t_ring.size, NS + 1)),
                      s0[:, None] + np.linspace(0, 1, NS + 1)[None, :]
                      * (s_branch - s0[:, None]))
    pb[:, 0, :] = np.vstack([c_lead, c_trail[::-1][1:-1]])
    add(pb, wrap=True)

    V = np.array(verts, dtype=float)
    F = np.array(tris, dtype=np.int64)
    # weld the patch seams: the rim was injected identically, the band meets the other
    # two on a shared x grid, so this only ever fuses points that are already equal
    first, inv = np.unique(
        conform.coincident_clusters(V, 1e-7 * float((V.max(0) - V.min(0)).max())),
        return_inverse=True)
    V, F = V[first], inv.ravel()[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    V = relax_junction(V, F)
    F = delaunay_flip(V, F)

    # the patches are wound consistently but not necessarily outward, and an STL is read
    # as outward-normal; decide from one undented face far up the pipe, where outward
    # *is* radial, and flip the lot if it disagrees
    cen = V[F].mean(axis=1)
    f0 = int(np.flatnonzero((cen[:, 0] < x_in + 0.25 * RUN_MAIN)
                            & (np.hypot(cen[:, 1], cen[:, 2]) < R_MAIN + 1e-9))[0])
    nrm = np.cross(V[F[f0, 1]] - V[F[f0, 0]], V[F[f0, 2]] - V[F[f0, 0]])
    if float(np.dot(nrm, np.array([0.0, cen[f0, 1], cen[f0, 2]]))) < 0.0:
        F = F[:, ::-1]
    return TriMesh(V, F)


def write_stl(surface, path):
    """Binary STL, so the generated surface can be inspected in any viewer."""
    P = surface.points[surface.tris]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros((P.shape[0], 12), dtype="<f4")
    rec[:, 0:3], rec[:, 3:6], rec[:, 6:9], rec[:, 9:12] = n, P[:, 0], P[:, 1], P[:, 2]
    with open(path, "wb") as fh:
        fh.write(b"femoral junction, generated by NekMeshPy".ljust(80, b" "))
        fh.write(struct.pack("<I", P.shape[0]))
        pad = b"\x00\x00"
        fh.write(b"".join(rec[i].tobytes() + pad for i in range(P.shape[0])))
    return path


# -- seam / opening solvers ---------------------------------------------------
def order_openings(surf):
    """Order the three boundary loops: A = main inlet (lowest mean X), B = branch
    (highest mean Z), C = main outlet.

    Only the *identification* is this junction's own -- the carotid picks its trunk by
    lowest mean Z and splits the other two by X, since its layout differs.  This is the
    one piece of ``trimesh.ops``'s trifurcation-splitting pipeline that cannot be
    shared: everything downstream of the three ordered loops lives there, but here the
    surface split itself is unused (:func:`tet_scaffold` reads these loops only to find
    the volumetric caps; the actual leg cut is the tet-conduction fields below,
    :func:`tetmesh.ops.seam_fields <nekmeshpy.tetmesh.ops.seam_fields>` /
    :func:`tetmesh.ops.leg_label <nekmeshpy.tetmesh.ops.leg_label>`)."""
    loops = surf.boundary_loops()
    assert len(loops) == 3, "expected exactly 3 boundary loops, got %d" % len(loops)
    Z, X = surf.points[:, 2], surf.points[:, 0]
    meanZ = np.array([Z[c].mean() for c in loops])
    iB = int(np.argmax(meanZ))
    rest = [i for i in range(3) if i != iB]
    meanX = np.array([X[loops[i]].mean() for i in rest])
    order = np.argsort(meanX, kind="stable")
    return [loops[rest[order[0]]], loops[iB], loops[rest[order[1]]]]


# -- volumetric conduction ---------------------------------------------------
def tet_scaffold(surf):
    """Cap the wall, tet-mesh the interior, and return ``(mesh, caps, gloops)``.

    The scaffold is deliberately coarse and thrown away afterwards: it only has to carry
    a smooth field, and surface element size is what drives the tet count.  Its wall
    nodes keep their indices, so the field's trace on the wall can be read straight off
    the first ``surf.n_points`` rows."""
    gloops = order_openings(surf)
    loops = [trimesh.ops.order_boundary_loop(surf, g) for g in gloops]
    capped, _ = tetmesh.ops.cap_surface(surf, loops)
    mesh = tetmesh.ops.tet_mesh(capped, TET_NEAR, TET_FAR, TET_RAMP, (0.0, 0.0, Z_SEAM))
    # gmsh remeshed the surface, so the caps have to be found on the mesh it returned
    caps = tetmesh.ops.cap_nodes(mesh, [surf.points[q] for q in loops])
    return mesh, caps, gloops


# which two fields bound each leg: ``tetmesh.ops.leg_label`` reads a leg off two signs, so those
# two zero sets are its two cuts.  ``(index, sign)`` -- keep where ``sign * U[:, index] > 0``
LEG_CUTS = {1: ((1, +1), (2, -1)), 2: ((0, -1), (2, +1)), 3: ((0, +1), (1, -1))}


def _growth_ratio(h, length, n, lo=1.0e-3, hi=10.0, iters=80):
    """The geometric ratio whose ``n`` cells, starting at ``h``, span ``length``.

    Solving for the ratio rather than being handed one is what makes the transition
    seamless: the graded run's first cell *is* the uniform run's cell, so the two meet
    with no jump, and the growth is whatever it has to be to reach the outlet in ``n``."""
    if n < 1 or h <= 0.0:
        return 1.0

    def span(q):
        return h * (n if abs(q - 1.0) < 1e-12 else (q ** n - 1.0) / (q - 1.0))

    if span(hi) < length:
        return hi
    if span(lo) > length:
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if span(mid) < length:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def leg_arclength(walker, seed, n=400):
    """``(u, s)`` along a leg: the level, and the distance travelled to reach it.

    The conduction field is not linear in distance -- it crowds where the leg narrows and
    stretches where it opens out -- so "2 diameters from the junction" cannot be read off
    a level directly.  Walking one point down the gradient from the seam to the cap and
    accumulating what it travels gives the map between the two, and the walker is already
    the thing that knows how to do that."""
    us = np.linspace(1.0 - 1e-3, 1e-3, int(n))
    p = walker.advance(np.asarray(seed, dtype=float).reshape(1, 3), float(us[0]))
    s = [0.0]
    for u in us[1:]:
        q = walker.advance(p, float(u))
        s.append(s[-1] + float(np.linalg.norm(q - p)))
        p = q
    return us, np.asarray(s, dtype=float)


def leg_levels(walker, seed, diameter, near_len):
    """The stations of one leg: **uniform** for ``near_len`` diameters out from the
    junction, then **geometric** the rest of the way to the outlet.

    One grading across the whole leg cannot serve both ends.  Fine enough at the junction
    to hold the trough and it wastes elements down a straight pipe; coarse enough at the
    outlet and the trough is drawn with three layers, which is the transition that would
    not resolve.  Splitting the run lets each half be sized for what is actually there:
    constant thickness where the geometry turns, growth where nothing happens.

    Returned ascending in ``u`` -- cap first, junction last -- with both ends dropped,
    since ``ogrid_leg`` supplies the cap and the seam itself."""
    us, s = leg_arclength(walker, seed)
    total = float(s[-1])
    near = min(near_len * diameter, 0.8 * total)
    h = near / N_UNIFORM
    q = _growth_ratio(h, total - near, N_GRADED)
    d = np.concatenate([
        np.arange(1, N_UNIFORM + 1) * h,                       # out from the junction
        near + h * np.cumsum(q ** np.arange(N_GRADED))])       # on to the cap
    d[-1] = total                                              # land exactly on the cap
    return np.interp(d[:-1], s, us)[::-1], q


def leg_field_vol(mesh, U, caps, leg):
    """A leg's own conduction field: ``0`` on its opening cap, ``1`` on **the cut**,
    no-flux on the wall -- solved on a mesh that has been cut along the cut.

    The carotid solves this on the leg's *surface*; here it is the volume, so its level
    sets are cross-sections rather than curves -- and, the wall being no-flux, they meet
    the wall at a right angle instead of at whatever angle an algebraic fill produces.

    Pinning ``1`` on the tet faces that happen to separate this leg from the others is the
    mistake ``interface`` warns about one rung down: those faces are a *quantized* cut, a
    staircase along whatever facets the mesher laid down, 45 and 90 degrees between
    neighbours.  A harmonic field with Dirichlet data on a staircase has staircase level
    sets near it, which is why every station close to the junction came out wiggly while
    the cut itself -- read as a level set -- was smooth.

    Imposing the value on a band of nodes near the cut instead does not help: the band is
    picked by a threshold, so its edge is quantized in turn, and the near-seam level sets
    come apart into dozens of loops.  The condition has to be imposed on the cut, so the
    cut has to be in the mesh.  ``clip_tets`` puts it there -- twice, once per bounding
    interface -- and the nodes it creates lie exactly on the level set, so ``u = 1`` on
    them *is* ``u = 1`` on the smooth cut."""
    Pl, Fl, Tl, o1 = fvol.clip_tets(mesh.points, U, mesh.tets, *_cut_arg(U, leg, 0))
    Pl, Fl, Tl, o2 = fvol.clip_tets(Pl, Fl, Tl, *_cut_arg(Fl, leg, 1))
    # a node the cut made carries -1; through two clips, either clip may have made it
    back = np.maximum(o2, 0)
    oncut = (o2 == -1) | ((o2 >= 0) & (o1[back] == -1))

    inv1 = np.full(mesh.n_points, -1, np.int64)
    inv1[o1[o1 >= 0]] = np.flatnonzero(o1 >= 0)
    inv2 = np.full(o1.shape[0], -1, np.int64)
    inv2[o2[o2 >= 0]] = np.flatnonzero(o2 >= 0)
    cap = inv1[caps[leg - 1]]
    cap = inv2[cap[cap >= 0]]
    cap = cap[cap >= 0]

    nodes = np.concatenate([cap, np.flatnonzero(oncut)])
    vals = np.concatenate([np.zeros(cap.size), np.ones(int(oncut.sum()))])
    nodes, keep = np.unique(nodes, return_index=True)
    u = tetmesh.ops.solve_dirichlet(TetMesh(Pl, Tl), nodes, vals[keep])
    return Pl, Tl, u


def _cut_arg(F, leg, which):
    k, sign = LEG_CUTS[leg][which]
    return (sign * F[:, k],)


def station_discs(mesh, inleg, u, levels):
    """One isosurface per level, as ``(TriMesh, ordered boundary ring)``.

    Restricting to the leg's own tets is what keeps each level set a single disc: the
    field is monotone through the leg but says nothing useful about the others."""
    out = []
    for lv in levels:
        pts, tris = fvol.isosurface(mesh.points, mesh.tets[inleg], u, float(lv))
        if tris.shape[0] < 4:
            continue
        disc = TriMesh(pts, tris)
        loops = trimesh.ops.boundary_loops(disc)
        if not loops:
            continue
        big = max(loops, key=lambda c: c.size)
        ring = pts[trimesh.ops.order_boundary_loop(disc, big)]
        if ring.shape[0] < MIN_LOOP_PTS:
            continue
        out.append((disc, ring, float(lv)))
    return out


def drop_degenerate(slices):
    """Discard any station whose O-grid has collapsed.

    A station pushed hard enough onto the seam has a disc barely distinguishable from it;
    two of its nodes then land on top of each other, ``merge`` fuses them, and the slice
    quietly comes back one point short -- which surfaces much later as ``loft`` refusing
    to stack slices of different shapes.  Catch it here, where the cause is still
    visible."""
    n = slices[-1].n_points
    keep = [s for s in slices if s.n_points == n]
    if len(keep) != len(slices):
        logging.warning("femoral: dropped %d collapsed station(s) of %d",
                        len(slices) - len(keep), len(slices))
    return keep


def project_interior(section, disc):
    """Pull a station onto the isosurface it came from -- every node, corners and the
    high-order tables alike (see ``femoral_vol.map_to_surface``)."""
    return fvol.map_to_surface(section, disc.points, disc.tris)


# -- O-grid leg builder ------------------------------------------------------
# -- O-grid leg builder ------------------------------------------------------
def seam_pieces(mesh, U, wall_pts, n_half, radial, center_scale, fine):
    """The three shared wall arcs and the one shared spine, computed **once**.

    Each interface is a half-disc: an arc on the wall and a spine through the interior,
    meeting at the two triple points.  Every arc is shared by exactly two legs and the
    spine by all three, so each is built once here and handed to whoever needs it -- the
    rule that makes the blocks weld, and the reason ``carotid.py`` fits its arcs globally
    rather than per leg.

    The three interfaces each give their own estimate of the spine; they agree only to
    about a third of an element, so the shared one is their mean."""
    nt = n_half // 4
    spine_fr = quadmesh.spine_fractions(nt, radial, center_scale)
    PAIRS = ((1, 2), (1, 3), (2, 3))
    # the spine is read straight off the clip that made the sheet -- the vertices the
    # trim created are the intersection with the other sheets, exactly
    raw = {}
    for pair in PAIRS:
        pts, tris, cut = fvol.interface(mesh.points, mesh.tets, U, *pair, want_cut=True)
        disc = TriMesh(pts, tris)
        ring = trimesh.ops.order_boundary_loop(
            disc, max(trimesh.ops.boundary_loops(disc), key=lambda c: c.size))
        a_ids, s_ids = fvol.split_boundary_by_cut(ring, cut)
        # The sheet's own wall boundary is only as accurate as marching tets made it, so
        # put it on the wall before anything is lifted through this triangulation.  Doing
        # it here rather than on the arc alone is what keeps the sheet and its rim on the
        # *same* surface -- correcting only the rim later would shear the ring of elements
        # between them.  The sheet is shared by the two legs either side, so both still
        # see bit-identical geometry.
        pts[a_ids] = snap_to_wall(pts[a_ids])
        raw[pair] = (pts, tris, pts[a_ids], pts[s_ids])

    spine_xyz = fvol.shared_spine([raw[p][3] for p in raw], 400)
    spine_pts = trimesh.ops.resample_polyline(spine_xyz, spine_fr)
    A1, A2 = spine_pts[0], spine_pts[-1]

    arcs = {}
    for pair, (_pts, _tris, arc_xyz, _) in raw.items():
        a = trimesh.ops.resample_polyline(arc_xyz, np.linspace(0.0, 1.0, 4 * nt + 1))
        if np.linalg.norm(a[0] - A1) > np.linalg.norm(a[-1] - A1):
            a = a[::-1]
        a[0], a[-1] = A1, A2                       # pin onto the shared triple points
        # the arc came off the *scaffold*, a coarse stand-in for the wall; put it on
        # the real surface -- exactly, since we have the closed form
        a = snap_to_wall(a)
        a[0], a[-1] = A1, A2
        # ``loft`` straight-subdivides between the points it is given, so snapping only
        # ``a`` leaves every curved node of the arc on a chord -- measured at 0.10 off the
        # wall against 0.0008 for the corners it sits between.  This arc *is* the rim of
        # the cutting O-grid (``pin_curve`` puts it back verbatim), so that is the seam
        # disc sitting off the wall.  Snap the whole curve, nodes included.
        arc = linemesh.loft(a, order=ORDER, element_tags="wall")
        if arc.interior.size:
            arc.interior[:] = snap_to_wall(
                arc.interior.reshape(-1, 3)).reshape(arc.interior.shape)
        arcs[pair] = arc
    return arcs, linemesh.loft(spine_pts, order=ORDER), raw


def _rim_target(sec, uv_pts, rim):
    """Where a station's wall ring **should** be, as a function of parameter position.

    A station is the boundary of a marching-tets isosurface, so its wall ring inherits
    the tet mesh's own facets: it wanders back and forth along the pipe by about a tet,
    which is the squiggle visible between consecutive stations on the wall.  Snapping it
    to the wall does not remove that -- snapping moves a point *onto* the surface, and a
    ring that wobbles along the wall is already on it.

    So low-pass the ring first, as a closed curve in its own index parameter, and snap
    the smoothed curve.  ``fourier_ring`` gives it back in closed form, which means the
    curved rim nodes can be evaluated *on* it rather than interpolated between corners --
    the sagitta argument this function already makes for sampling.

    Refitting a ring is what ``wall_loop`` deliberately does not do, for a good reason:
    it would slide the wall corners onto a different surface from the interior they are
    attached to.  That objection is answered here rather than avoided, because this
    displacement is the one the caller carries inward."""
    if rim.size < 8:
        return lambda P, uv: snap_to_wall(P)      # too few samples to low-pass
    ruv = uv_pts[rim]
    o = np.argsort(np.arctan2(ruv[:, 1], ruv[:, 0]))
    ang = np.arctan2(ruv[o, 1], ruv[o, 0])
    M = rim.size
    tpar = 2.0 * np.pi * np.arange(M) / M
    p = trimesh.ops.fourier_ring(sec.points[rim[o]], keep=RIM_KEEP)
    # the ring's parameter runs with its angle, so a node anywhere on the rim -- corner
    # or curved -- finds its place on the refit curve by its own angle
    ang_x = np.concatenate([ang - 2.0 * np.pi, ang, ang + 2.0 * np.pi])
    t_x = np.concatenate([tpar - 2.0 * np.pi, tpar, tpar + 2.0 * np.pi])

    def at(P, uv):
        a = np.arctan2(uv[:, 1], uv[:, 0])
        return snap_to_wall(p(np.interp(a, ang_x, t_x)))

    return at


def blend_to_wall(sec, uv_pts, uv_edge, uv_face, power=1.0):
    """Put the wall rim on the **analytic** wall and carry the correction inward.

    A section is lifted onto a *triangulated* isosurface, so left alone its wall is
    high-order in storage and piecewise linear in geometry -- the curved nodes land on
    facet chords, measurably further from the cylinder than the corners they sit between
    (0.0093 against 0.0014, on a chord sagitta of 0.077).

    Moving only the rim onto the true wall is worse than leaving it: the rim then lives on
    a different surface from the interior it is attached to, the wall layer shears between
    the two, and elements inverted that had been fine.  So the rim's displacement is
    carried inward, weighted by the node's radius in the *parameter* circle -- 1 at the
    rim, 0 at the core -- which is exactly the O-grid's own radial coordinate and costs
    nothing to know.  The section stays one surface; it is just the right one at the
    wall.

    The correction has to be *sampled* at the curved rim nodes too, not interpolated
    between the corners: a corner sits near a triangulation vertex and is barely off the
    cylinder, while the curved node between two corners sits on the facet chord and is off
    it by the sagitta -- six times further.  A field built from corners alone does not know
    that error exists, and moves the curved nodes by the wrong amount."""
    rim = quadmesh.boundary_points(sec)
    wall_at = _rim_target(sec, uv_pts, rim)
    src_uv = [uv_pts[rim]]
    src_d = [wall_at(sec.points[rim], uv_pts[rim]) - sec.points[rim]]

    inter = sec.line_mesh.interior
    on = np.zeros(0, bool)
    wall_edges = np.zeros((0,) + inter.shape[1:])
    if inter.size and uv_edge is not None:
        on = np.isin(np.asarray(sec.edges), rim).all(axis=1)
        if on.any():
            wall_edges = inter[on]
            q = wall_edges.reshape(-1, 3)
            uvq = uv_edge[on].reshape(-1, 2)
            src_uv.append(uvq)
            src_d.append(wall_at(q, uvq) - q)

    ruv, disp = np.vstack(src_uv), np.vstack(src_d)
    ang = np.arctan2(ruv[:, 1], ruv[:, 0])
    o = np.argsort(ang)
    ang, disp = ang[o], disp[o]
    ang = np.concatenate([ang - 2.0 * np.pi, ang, ang + 2.0 * np.pi])
    disp = np.vstack([disp, disp, disp])

    def carried(uv):
        r = np.hypot(uv[:, 0], uv[:, 1])
        t = np.arctan2(uv[:, 1], uv[:, 0])
        d = np.column_stack([np.interp(t, ang, disp[:, k]) for k in range(3)])
        return np.minimum(r, 1.0)[:, None] ** power * d

    if wall_edges.size:
        # the rim's own nodes take their exact displacement: the parameter loop is
        # straight-subdivided, so their radius is a hair under 1 and the weight would
        # otherwise shrink the very correction this exists to apply
        flat = uv_edge.reshape(-1, 2)
        d = carried(flat).reshape(inter.shape)
        d[on] = src_d[-1].reshape(int(on.sum()), -1, 3)
        inter[:] = inter + d
    elif inter.size and uv_edge is not None:
        inter[:] = inter + carried(uv_edge.reshape(-1, 2)).reshape(inter.shape)

    face = sec.interior
    if face.size and uv_face is not None:
        face[:] = face + carried(uv_face.reshape(-1, 2)).reshape(face.shape)

    d = carried(uv_pts)
    d[rim] = src_d[0]
    sec.points[:] = sec.points + d
    return sec


def lift_section(dm, sec):
    """Lift **every** node a section stores, not only its corners.

    ``QuadMesh.points`` is the corners: a 192-quad order-3 section has 209 of them, while
    its curved nodes live in ``lines.interior`` (the edges') and ``interior`` (the faces').
    Lifting only ``points`` leaves those curved nodes holding whatever the planar O-grid
    put there -- parameter-plane coordinates, in the wrong place entirely -- and two
    sections mapped through different parametrizations then disagree about a shared edge,
    which is what stops the seam halves merging.  Each array is lifted from its own planar
    values, so all three land on the surface together."""
    for arr in (sec.line_mesh.interior, sec.interior):
        if arr.size:
            arr[:] = dm.lift(arr[..., :2].reshape(-1, 2)).reshape(arr.shape)
    sec.points[:] = dm.lift(sec.points[:, :2])
    return sec


def map_section(pts, tris, ring_ids, seam_loop, frac, *, radial, center_scale):
    """O-grid a triangulated disc by mapping onto it through a parametrization rather
    than projecting onto it.

    Projecting an O-grid onto a triangle soup takes each node to its nearest point, which
    is not a map between the two surfaces at all: where the section creases -- and it
    creases along the spine, where two interfaces meet -- neighbouring nodes land on the
    same fold and the quads between them turn inside out.  Instead embed the disc in the
    unit circle (``femoral_vol.DiscMap``, Tutte, provably fold-free), build the O-grid
    *there*, where it is fold-free for the same reason it is on any disc, and lift it
    back.  A homeomorphism composed with a fold-free grid is a fold-free grid."""
    dm = fvol.DiscMap(pts, tris, ring_ids)

    # the rim keeps the seam's node correspondence, so every station is welded the same
    # way round; only the interior is what the parametrization decides
    seam_pts = seam_loop.points
    nh = seam_pts.shape[0] // 2
    rim = trimesh.ops.conform_ring_stack([pts[ring_ids]], seam_pts,
                                         np.array([frac]), nh)[0]
    ruv = dm.ring_uv(rim)

    # the same O-grid, built in the circle: a planar loop through the rim's parameters,
    # spanned by the chord between the two triple points' parameters
    flat = np.column_stack([ruv, np.zeros(ruv.shape[0])])
    e1, e2 = flat[0], flat[nh]
    fr = quadmesh.spine_fractions(nh // 4, radial, center_scale)
    chord = e1 + fr[:, None] * (e2 - e1)
    sec = quadmesh.spined_ogrid(
        wall_loop(flat, ORDER), radial,
        spine=linemesh.loft(chord, order=ORDER),
        center_scale=center_scale, quadrant_scale=center_scale, wall_tag="wall")
    # the parameter positions, kept before the lift overwrites them -- the O-grid's own
    # radial coordinate, which is what weights the wall correction inward
    uv_pts = sec.points[:, :2].copy()
    uv_edge = (sec.line_mesh.interior[..., :2].copy()
               if sec.line_mesh.interior.size else None)
    uv_face = sec.interior[..., :2].copy() if sec.interior.size else None
    lift_section(dm, sec)
    return (blend_to_wall(sec, uv_pts, uv_edge, uv_face)
            if PROJECT_TO_STL else sec)


def seam_section(arc, spine, iface, *, radial, center_scale, flip=False):
    """One half of the seam disc, laid **on** the interface rather than across it.

    ``spined_ogrid`` fills between the rim and the spine algebraically, and the interface
    is curved, so that fill bulges off it -- measured at up to 0.20 here, against a spine
    that is itself on all three interfaces to 0.003.  The visible ledge across the seam
    disc is that bulge, not a misplaced spine.

    So build the half O-grid in the interface's *parameter* circle and lift it, exactly as
    a station is built.  Topology is untouched: ``spined_ogrid`` is itself
    ``merge(half_ogrid(arc1, spine), half_ogrid(arc2, spine2))``, so the halves still weld
    into the same mesh.  Conformality survives for the same reason it did before -- the
    arc, the spine and the interface are all shared, so both legs either side of an
    interface build a bit-identical half from bit-identical inputs."""
    pts, tris = iface
    disc = TriMesh(pts, tris)
    ring = trimesh.ops.order_boundary_loop(
        disc, max(trimesh.ops.boundary_loops(disc), key=lambda c: c.size))

    # An interface is bounded by its arc *and* its spine, so sending the whole boundary to
    # a circle would put the spine on the rim -- and a half O-grid whose spine is on the
    # rim collapses its core onto it.  Send it to a half-disc instead: the arc to the
    # semicircle, the spine to the diameter.  Still convex, so Tutte still holds, and the
    # spine now runs through the interior where the O-grid expects it.
    from scipy.spatial import cKDTree
    R = pts[ring]
    ends = cKDTree(R).query(np.vstack([spine.points[0], spine.points[-1]]))[1]
    i, j = int(min(ends)), int(max(ends))
    runs = (np.arange(i, j + 1), np.concatenate([np.arange(j, ring.size), np.arange(i + 1)]))
    # the arc is the run that lies on the wall; the spine is the other one
    dw = [np.linalg.norm(snap_to_wall(R[r]) - R[r], axis=1).mean() for r in runs]
    arc_run, spine_run = (runs[0], runs[1]) if dw[0] < dw[1] else (runs[1], runs[0])

    def chord(idx):
        q = R[idx]
        d = np.linalg.norm(np.diff(q, axis=0), axis=1)
        t = np.concatenate([[0.0], np.cumsum(d)])
        return t / max(t[-1], 1e-30)

    uv_ring = np.zeros((ring.size, 2))
    ta = chord(arc_run)
    uv_ring[arc_run] = np.column_stack([np.cos(np.pi * ta), np.sin(np.pi * ta)])
    ts = chord(spine_run)
    uv_ring[spine_run] = np.column_stack([-1.0 + 2.0 * ts, np.zeros(ts.size)])
    if np.linalg.norm(R[arc_run[0]] - R[spine_run[-1]]) > 1e-9:
        uv_ring[spine_run] = uv_ring[spine_run][::-1]
    dm = fvol.DiscMap(pts, tris, ring, uv_ring)

    flat_a = np.column_stack([dm.ring_uv(arc.points), np.zeros(arc.n_points)])
    flat_s = np.column_stack([dm.ring_uv(spine.points), np.zeros(spine.n_points)])
    half = quadmesh.half_ogrid(
        linemesh.loft(flat_a, order=ORDER, element_tags="wall"),
        linemesh.loft(flat_s, order=ORDER), radial, center_scale=center_scale,
        quadrant_scale=center_scale)
    # the parameter positions, kept before the lift overwrites them -- ``pin_curve``
    # finds its nodes by where they sit in the *parameter* plane, which is the only
    # place it can recognize them, and after the lift they are model coordinates
    uv_pts = half.points[:, :2].copy()
    uv_edge = (half.line_mesh.interior[..., :2].copy()
               if half.line_mesh.interior.size else None)
    uv_face = half.interior[..., :2].copy() if half.interior.size else None
    lift_section(dm, half)
    was = (half.points.copy(), half.line_mesh.interior.copy(), half.interior.copy())
    # Restore the shared curves exactly.  The arc and the spine bound *two* interfaces,
    # and each one's triangulation renders them slightly differently -- by the ~0.003 the
    # two soups disagree about the triple curve.  Reconstructing them is therefore never
    # bit-identical between the halves, and ``merge`` rejects a shared edge whose interior
    # nodes differ by more than the entity tolerance.  They are shared data: put them back.
    pin_curve(half, uv_pts, flat_a, arc)
    pin_curve(half, uv_pts, flat_s, spine)
    carry_pins(half, was, uv_pts, uv_edge, uv_face)
    return half


def _half_disc_boundary(uv):
    """``(s, dist)`` for parameter points in the closed half-disc.

    ``s`` walks the boundary once -- the unit semicircle from ``(1,0)`` to ``(-1,0)``
    over ``[0, pi]``, then the diameter back over ``[pi, pi+2]`` -- taken at the node's
    *nearest* boundary point, and ``dist`` is how far it is from there.  One coordinate
    for both bounding curves is the whole point: the arc and the spine are pinned by the
    same mechanism, so a carry that treats them separately would need two weights and a
    rule for the corner where they meet, and this needs neither."""
    u, v = uv[:, 0], np.maximum(uv[:, 1], 0.0)
    r = np.hypot(u, v)
    d_arc, d_dia = 1.0 - r, v
    on_arc = d_arc <= d_dia
    s = np.where(on_arc, np.arctan2(v, u), np.pi + (u + 1.0))
    return s, np.minimum(d_arc, d_dia)


#: Half the inradius reciprocal: ``min(1-r, v)`` peaks at 0.5 inside the unit half-disc
#: (at ``u=0, v=0.5``, where the two distances meet), so this normalizes the weight to
#: reach 0 exactly at the point furthest from both bounding curves.
_HALF_DISC_INRADIUS = 0.5


def carry_pins(sec, was, uv_pts, uv_edge, uv_face, power=1.0):
    """Carry what :func:`pin_curve` moved inward, instead of stopping at the curve.

    Pinning alone is the trap :func:`blend_to_wall` documents one rung down: the arc and
    the spine land on the shared data, the interior they are attached to does not move,
    and the layer between the two shears.  Measured, pinning without this: curved scaled
    Jacobian min -0.3036 -> -0.9348 and 4 inverted elements -> 8.

    So take the displacement the pins applied, read it along the boundary at each node's
    nearest boundary point, and weight it by how far in that node sits -- 1 on either
    bounding curve, 0 at the point furthest from both.  The pinned nodes keep their exact
    positions; everything else follows them by as much as it should.  Sampled at the
    curved nodes as well as the corners, for the reason ``blend_to_wall`` gives: a curved
    node sits on a facet chord and is off by the sagitta, several times further than the
    corners it lies between, so a field built from corners alone moves it by the wrong
    amount."""
    pts0, edge0, face0 = was
    d_pts = sec.points - pts0
    d_edge = sec.line_mesh.interior - edge0 if edge0.size else np.zeros((0, 0, 3))

    src_uv = [uv_pts[np.abs(d_pts).any(axis=1)]]
    src_d = [d_pts[np.abs(d_pts).any(axis=1)]]
    if d_edge.size and uv_edge is not None:
        moved = np.abs(d_edge).any(axis=(1, 2))
        if moved.any():
            src_uv.append(uv_edge[moved].reshape(-1, 2))
            src_d.append(d_edge[moved].reshape(-1, 3))
    if not sum(a.shape[0] for a in src_uv):
        return sec

    buv, bd = np.vstack(src_uv), np.vstack(src_d)
    bs, _ = _half_disc_boundary(buv)
    o = np.argsort(bs)
    bs, bd = bs[o], bd[o]
    period = np.pi + 2.0
    bs_x = np.concatenate([bs - period, bs, bs + period])
    bd_x = np.vstack([bd, bd, bd])

    def carried(uv):
        s, dist = _half_disc_boundary(uv)
        w = np.clip(1.0 - dist / _HALF_DISC_INRADIUS, 0.0, 1.0) ** power
        d = np.column_stack([np.interp(s, bs_x, bd_x[:, k]) for k in range(3)])
        return w[:, None] * d

    # the pinned nodes are already exact; everything else follows them
    keep = np.abs(d_pts).any(axis=1)
    add = carried(uv_pts)
    add[keep] = 0.0
    sec.points[:] = sec.points + add

    if d_edge.size and uv_edge is not None:
        moved = np.abs(d_edge).any(axis=(1, 2))
        add = carried(uv_edge.reshape(-1, 2)).reshape(d_edge.shape)
        add[moved] = 0.0
        sec.line_mesh.interior[:] = sec.line_mesh.interior + add
    if sec.interior.size and uv_face is not None:
        sec.interior[:] = sec.interior + carried(
            uv_face.reshape(-1, 2)).reshape(sec.interior.shape)
    return sec


def pin_curve(sec, uv, flat, curve):
    """Put a boundary curve's own nodes back, corners and curved nodes alike.

    ``flat`` is where that curve sits in the parameter plane, which is exactly where
    ``half_ogrid`` placed it, so its nodes are found by matching coordinates rather than
    by trusting an index convention.  ``uv`` is the section's own parameter positions,
    which is what those coordinates have to be matched against: they are the same numbers
    only *before* ``lift_section`` replaces them with model coordinates, and matching a
    lifted point against a parameter one silently finds nothing at all."""
    from scipy.spatial import cKDTree
    tree = cKDTree(flat[:, :2])
    d, j = tree.query(uv[:, :2])
    on = d < 1e-12
    sec.points[on] = curve.points[j[on]]
    if not sec.line_mesh.interior.size:
        return
    ends = np.asarray(sec.edges)
    a, b = j[ends[:, 0]], j[ends[:, 1]]
    good = on[ends[:, 0]] & on[ends[:, 1]] & (np.abs(a - b) == 1)
    for k in np.flatnonzero(good):
        i = min(int(a[k]), int(b[k]))
        nodes = curve.interior[i]
        sec.line_mesh.interior[k] = nodes if a[k] < b[k] else nodes[::-1]


def level_section(walker, level, seam_loop, *, radial, center_scale):
    """A station: the leg field's ``u = level`` isosurface, O-gridded."""
    pts, tris = walker.isosurface(level)
    disc = TriMesh(pts, tris)
    ring_ids = trimesh.ops.order_boundary_loop(
        disc, max(trimesh.ops.boundary_loops(disc), key=lambda c: c.size))
    return map_section(pts, tris, ring_ids, seam_loop, level,
                       radial=radial, center_scale=center_scale)


def cap_section(loop_xyz, seam_loop, *, radial, center_scale):
    """The opening itself, O-gridded.

    The stations are interior level sets -- ``u = 0`` is the cap's own Dirichlet
    boundary, where the level set degenerates -- so the leg has to be closed off with the
    opening as given rather than with a level set approaching it.  Left out, the mesh
    stops a whole diameter short of the geometry and its inlet face floats inside the
    domain.  The opening is planar, so fanning it from its centroid is exact."""
    R = np.asarray(loop_xyz, dtype=float).reshape(-1, 3)
    pts = np.vstack([R, R.mean(axis=0)])
    c = R.shape[0]
    tris = np.column_stack([np.arange(c), np.roll(np.arange(c), -1),
                            np.full(c, c)])
    return map_section(pts, tris, np.arange(c), seam_loop, 0.0,
                       radial=radial, center_scale=center_scale)


def ogrid_leg(walker, levels, cap_loop, seam_loop, spine, ifaces, *,
              radial, center_scale):
    """A leg's slices, opening first and seam last.

    Each is O-gridded through its own parametrization rather than chained to the one
    before it.  Carrying a section along ``grad u`` from the seam did keep consecutive
    stations from crossing, but it propagated each section's defects into every section
    after it, and it left the rim to ``snap_to_wall`` -- which moves rim nodes off the
    surface the interior was placed on, costing several inverted quads per near-seam
    station.  Nothing is lost by dropping the chain: the parametrization is canonical
    given the disc and the rim correspondence, and both vary smoothly with the level.

    The two ends are not level sets.  ``u = 1`` is pinned across whole tets at the seam
    and ``u = 0`` is the cap's own Dirichlet boundary, so both degenerate; the seam is
    built from its two interfaces and the cap from the opening as given."""
    # each half on the interface it is, rather than an algebraic fill across both
    seam = quadmesh.merge([
        seam_section(a, sp, f, radial=radial, center_scale=center_scale)
        # the second half traverses A2 -> A1, arc *and* spine -- reversing only the
        # spine leaves the two halves wound against each other, and every quad of one of
        # them then reads inverted in the merged disc
        for a, sp, f in ((seam_loop[0], spine, ifaces[0]),
                         (linemesh.reverse(seam_loop[1]), linemesh.reverse(spine),
                          ifaces[1]))])
    loop = linemesh.merge([seam_loop[0], linemesh.reverse(seam_loop[1])])
    slices = [cap_section(cap_loop, loop, radial=radial, center_scale=center_scale)]
    slices += [level_section(walker, float(lv), loop, radial=radial,
                             center_scale=center_scale) for lv in levels]
    slices.append(seam)
    return slices


def cached_tets(scaffold):
    """``tet_scaffold``, remembered on disk.

    Keyed on the scaffold it is built from and the tet knobs, so either changing rebuilds.
    The conduction fields are cached with it -- a pure function of the same inputs, and
    six CG solves to rebuild.  The Laplacian is not: it is sparse, so it does not
    round-trip through ``savez``, and nothing downstream wants the whole-domain one
    anyway -- each leg assembles its own, over its own cut mesh."""
    key = np.concatenate([
        np.array([NT, NU, NDN, NB, NS, GRADE, TET_NEAR, TET_FAR, TET_RAMP], dtype=float),
        scaffold.points.ravel()[::997], [scaffold.n_tris]])
    path = os.path.join(_HERE, TET_CACHE)
    if os.path.exists(path):
        z = np.load(path)
        if z["key"].shape == key.shape and np.array_equal(z["key"], key):
            logging.info("femoral: tet mesh and fields from cache")
            return (TetMesh(z["P"], z["TET"]), [z["c0"], z["c1"], z["c2"]],
                    z["U"], z["lab"])
        logging.info("femoral: tet knobs changed, rebuilding")
    mesh, caps, _gl = tet_scaffold(scaffold)
    U = tetmesh.ops.seam_fields(mesh, caps)
    lab = tetmesh.ops.leg_label(mesh, U)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, key=key, P=mesh.points, TET=mesh.tets, c0=caps[0], c1=caps[1],
             c2=caps[2], U=U, lab=lab)
    logging.info("femoral: cached the tet mesh (%d tets) and its fields", mesh.n_tets)
    return mesh, caps, U, lab


def cached_surface(tag):
    """``build_surface()``, remembered on disk under ``tag``.

    The key covers every knob the surface depends on -- the tessellation counts as they
    stand *now*, so this works for the scaffold's overridden values as well as the fine
    ones, plus the geometry and both post-processing knobs.  Change any of them and it
    rebuilds, rather than silently handing back the old geometry: tuning a parameter and
    not seeing it take effect is the failure this has to prevent."""
    key = np.array([NT, NU, NDN, NB, NS, GRADE, R_MAIN, R_BRANCH, BRANCH_ANGLE,
                    SEAM_OFFSET, TROUGH_DEPTH, TROUGH_WIDTH, RUN_MAIN, RUN_BRANCH,
                    DELAUNAY_ROUNDS, RELAX_PASSES, RELAX_RADIUS], dtype=float)
    path = os.path.join(_HERE, SURFACE_CACHE % tag)
    if os.path.exists(path):
        z = np.load(path)
        if z["key"].shape == key.shape and np.array_equal(z["key"], key):
            logging.info("femoral: %s surface from cache", tag)
            return TriMesh(z["V"], z["F"])
        logging.info("femoral: %s surface knobs changed, rebuilding", tag)
    s = build_surface()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, key=key, V=s.points, F=s.tris)
    logging.info("femoral: cached the %s surface (%d tris)", tag, s.n_tris)
    return s


# -- pipeline (flat driver) --------------------------------------------------
surf = cached_surface("fine")
print("surface: %d points, %d tris, %d boundary loops"
      % (surf.n_points, surf.n_tris, len(surf.boundary_loops())))
if EXPORT_STL:
    write_stl(surf, OUT_NAME + ".stl")
if PLOT_STL:
    # tuning the tessellation should not cost a tet mesh and three conduction solves
    plot_stl(surf)
    raise SystemExit(0)

_fine = {k: globals()[k] for k in SCAFFOLD}
globals().update(SCAFFOLD)
scaffold = cached_surface("scaffold")
globals().update(_fine)
tets, caps, Uvol, lab = cached_tets(scaffold)
print("tet scaffold: %d nodes, %d tets" % (tets.n_points, tets.n_tets))
print("leg tet counts: %s (unassigned %d)"
      % ([int((lab == k).sum()) for k in (1, 2, 3)], int((lab == 0).sum())))

arcs, spine, raw = seam_pieces(tets, Uvol, scaffold.points, N_HALF, RADIAL,
                               CENTER_SCALE, surf)
print("shared spine: %d nodes; arcs %s"
      % (spine.n_points, {k: v.n_points for k, v in arcs.items()}))

# each leg's seam ring is the pair of arcs it shares with its neighbours, welded at the
# triple points -- the same arcs, so adjacent legs meet on identical geometry
LEG_ARCS = {1: ((1, 2), (1, 3)), 2: ((1, 2), (2, 3)), 3: ((1, 3), (2, 3))}
seam_loops = {leg: linemesh.merge([arcs[p], linemesh.reverse(arcs[q])])
              for leg, (p, q) in LEG_ARCS.items()}

opening_name = ["inlet", "branch", "outlet"]
# a flux plane on each outflow leg, exactly carotid's idiom: the leg is lofted in two
# pieces there and ``merge`` re-joins them, so the shared plane is an interior surface
# carrying a name rather than a gap in the mesh
FLUX_NAME = {"branch": "flux_1", "outlet": "flux_2"}
LEG_DIAMETER = [2.0 * R_MAIN, 2.0 * R_BRANCH, 2.0 * R_MAIN]
LEG_NEAR_LEN = [NEAR_LEN, NEAR_LEN_BRANCH, NEAR_LEN]   # leg 2 is the branch

# the three openings, in the same order as the legs -- each leg is closed off with the
# one its conduction field is pinned to zero on
cap_loops = [surf.points[trimesh.ops.order_boundary_loop(surf, c)]
             for c in order_openings(surf)]

blocks = []
for leg in (1, 2, 3):
    Pl, Tl, u = leg_field_vol(tets, Uvol, caps, leg)
    walker = fvol.FieldWalker(Pl, Tl, u)
    pr, qr = LEG_ARCS[leg]
    near_len = LEG_NEAR_LEN[leg - 1]
    levels, ratio = leg_levels(walker, seam_loops[leg].points.mean(axis=0),
                               LEG_DIAMETER[leg - 1], near_len)
    slices = ogrid_leg(walker, levels, cap_loops[leg - 1],
                       (arcs[pr], arcs[qr]), spine,
                       (raw[pr][:2], raw[qr][:2]),
                       radial=RADIAL, center_scale=CENTER_SCALE)
    name = opening_name[leg - 1]
    flux, off, joint = FLUX_NAME.get(name, ""), FLUX_OFFSET, N_GRADED
    print("  %-7s: %d slices = %d uniform over %.3gD + %d graded (ratio %.4f)%s"
          % (name, len(slices), N_UNIFORM, near_len, N_GRADED, ratio,
             ", flux plane %d in" % off if flux else ""))
    # Lofted in pieces, not in one run.  The uniform and graded halves meet at a genuine
    # change of layer thickness, and ``loft_spline`` fits a cubic through the *whole*
    # stack it is given -- across that joint it would read the change as curvature and
    # overshoot.  One loft per run keeps each spline inside a region of its own spacing,
    # and the split at the flux plane is what gives that plane a name.
    if flux and 0 < off < joint:
        blocks.append(hexmesh.loft_spline(slices[:off + 1], first_tag=name,element_tags=FLUX_DOWNSTREAM))
        blocks.append(hexmesh.loft_spline(slices[off:joint + 1], first_tag=flux,element_tags=FLUX_UPSTREAM))
    else:
        blocks.append(hexmesh.loft_spline(slices[:joint + 1], first_tag=name))
    blocks.append(hexmesh.loft_spline(slices[joint:]))

mesh = hexmesh.merge(blocks)
mesh = hexmesh.scale(mesh, 1.0/(R_MAIN*2.0))

print(hexmesh.report(mesh))
print("femoral: branch %.0f deg, R %.3g / %.3g, seam z %.4f, trough %.3g deep x %.3g"
      % (BRANCH_ANGLE, R_MAIN, R_BRANCH, Z_SEAM, TROUGH_DEPTH, TROUGH_WIDTH))

if EXPORT_VTK:
    writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
if EXPORT_RE2:
    writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
if EXPORT_FLD:
    writer.to_fld(mesh, OUT_NAME + ".f00000")
print("femoral: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
