"""Hex smoothing of a curved bent pipe with MFEM's TMOP: a curved mesh stays a curved mesh.

A pipe bent through 90 degrees, order 3: an O-grid section (4 + 16 = 20 quads, 8 wall segments) swept along a quarter
circle in 8 layers, 160 hexes.  ``hexmesh.sweep`` evaluates the path on the whole node lattice, so every wall node lies
on the exact torus.  The inlet cap is in the plane ``z = 0``, the outlet cap in the plane ``x = R_BEND``; they are
tagged ``"inlet"`` / ``"outlet"``, the pipe wall ``"wall"``.  Every node that is not on the boundary -- the vertices
and the curved nodes of every edge, face and hex -- is then jittered, conformally, to make a damaged input.  Three runs
start from it, all with ``boundary="slide"`` (the caps slide in their planes, the curved wall is held):

    order1   the order-1 route: the straight-sided shadow (the corners alone) through the ``tmop`` stage; what the
             toolkit could do before a curved mesh was smoothed as curved.  Repaired, but the wall is an octagon.
    native   the native optimizer on the curved mesh: it moves the vertices and carries the curved nodes along.
    tmop     the ``tmop`` stage on the curved mesh: every node moves, the order is kept, the wall is held.

Each mesh is read three ways: at its stored nodes, on the order-``SAMPLE_ORDER`` lattice (what a solver at that order
sees; the nodal reading is silent between the nodes), and at the corners (the ``.re2`` geometry).  The wall is read
between its nodes, on a finer lattice, against the exact torus; the volume against Pappus.

    PYTHONPATH=. python examples/smoother/mfem/02_bent_pipe.py

Needs PyMFEM: ``pip install nekmeshpy[smooth]``.  Writes ``02_bent_pipe_{input,order1,native,tmop}.vtu`` and
``02_bent_pipe.re2`` (the result's corners) to the current directory and leaves ``mesh`` (the tmop result), ``meshes``, ``rows``, ``checks``, ``fields``, ``ORDER``,
``SAMPLE_ORDER``, ``R``, ``N_HEX``, and ``VERTS`` / ``SECTION_VERTS`` (vertex-id sets naming the wall, the caps and the
two face layers of the viewer) in the namespace.  With ``NEKMESHPY_BENT_PIPE_INPUT_ONLY`` set in the environment it
builds, measures and writes the input alone (no PyMFEM needed).
"""

import copy
import os
import time

import numpy as np

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core import interp
from nekmeshpy.core.smoother import mfem as tmop_bridge
from nekmeshpy.hexmesh import HexMesh
from nekmeshpy.hexmesh import smoother as hs

INPUT_ONLY = bool(os.environ.get("NEKMESHPY_BENT_PIPE_INPUT_ONLY"))
if not INPUT_ONLY and not tmop_bridge.available():
    raise ImportError("02_bent_pipe.py needs PyMFEM (MFEM's TMOP): pip install nekmeshpy[smooth]")

# -- parameters --------------------------------------------------------------
R = 0.5                          # pipe radius
R_BEND = 1.5                     # bend radius (centreline)
N_SIDE = 2                       # ogrid: 4 + 16 = 20 quads, 8 wall segments
RADIAL = 2
LAYERS = 8                       # 20 x 8 = 160 hexes
ORDER = 3
SAMPLE_ORDER = 7                 # the solver's order (core.quality.SCAN_ORDER); change to match yours
WALL_SAMPLE = 12                 # lattice order for reading the wall between its nodes
PERTURB = 0.13                   # jitter half-width: PERTURB * (shortest hex edge) / ORDER
SEED = 0
TMOP = {"target": "input"}       # the tmop stage's knobs (quad_order 2N+2 and sample_order max(N, 7) by default)
VEL_BC = {"wall": ("W  ", 1), "inlet": ("v  ", 2), "outlet": ("O  ", 3)}  # (cbc, boundaryID) for the .re2
OUTER = 3                        # outer rounds of the native schedule

# -- the bent pipe: an O-grid section swept along a quarter circle ---------------------
t_build = time.perf_counter()


# [doc:build]
def path(t):
    return np.stack([R_BEND * (1 - np.cos(np.pi * t / 2)), 0 * t, R_BEND * np.sin(np.pi * t / 2)], axis=-1)


def tangent(t):
    return np.stack([np.sin(np.pi * t / 2), 0 * t, np.cos(np.pi * t / 2)], axis=-1)


wall = linemesh.circle(R, 4 * N_SIDE, element_tag="wall", order=ORDER)    # order=: the wall nodes on the circle
section = quadmesh.ogrid(wall, N_SIDE, RADIAL)                            # the fill carries the curvature inward
exact = hexmesh.sweep(section, path, np.linspace(0.0, 1.0, LAYERS + 1), origin=(0.0, 0.0, 0.0), tangent=tangent,
                      first_tag="inlet", last_tag="outlet", element_tags="fluid")   # the path on every node
# [/doc:build]


def torus_off(x):
    """``|sqrt((rho - R_BEND)^2 + y^2) - R|``, the distance of points ``x`` ``(.., 3)`` from the wall's torus
    (``rho`` is the distance from the bend's axis)."""
    rho = np.sqrt((x[..., 0] - R_BEND) ** 2 + x[..., 2] ** 2)
    return np.abs(np.sqrt((rho - R_BEND) ** 2 + x[..., 1] ** 2) - R)

N_HEX = exact.n_hexes
assert exact.order == ORDER

# The named boundary pieces as sets of vertex ids, so any mesh numbered like this one (the straight shadow, a
# smoothed result) can find its wall and caps without tags: a face is the wall's when all 4 corners are.
VERTS = {t: np.unique(exact.quad_mesh.corners[hexmesh.tagged_faces(exact, t)]) for t in ("wall", "inlet", "outlet")}


# The two face layers the page draws, also as vertex sets: the faces at mid-bend (station LAYERS // 2) and the faces
# lying in the bend's symmetry plane y = 0.  Read on the exact mesh, before the jitter.
_phi = np.arctan2(exact.points[:, 2], R_BEND - exact.points[:, 0])
SECTION_VERTS = {"mid": np.flatnonzero(np.abs(_phi - np.pi / 2 * (LAYERS // 2) / LAYERS) < 1e-9),
                 "sym": np.flatnonzero(np.abs(exact.points[:, 1]) < 1e-9)}
assert all(np.isin(exact.quad_mesh.corners, v).all(axis=1).any() for v in SECTION_VERTS.values()), "empty section layer"


def faces_of(m, tag):
    """``(F, (order+1)**2, 3)`` node blocks of the boundary faces of ``m`` whose corners all lie in ``VERTS[tag]``."""
    on = np.isin(m.quad_mesh.corners, VERTS[tag]).all(axis=1) & hexmesh.boundary_face_ids(m)
    return quadmesh.element_blocks(m.quad_mesh)[on]


assert torus_off(faces_of(exact, "wall")).max() < 1e-12 * R, "a wall node is off the torus"

# -- the jitter: every node off the boundary, written in place on the stored entities ------------------
# Each entity is stored once (a shared vertex, a shared edge's interior, a shared face's interior, a hex's own
# interior), so writing those arrays in place is conformal by construction -- no coordinate search, no write-back.
mesh = copy.deepcopy(exact)
qm = mesh.quad_mesh
bface = hexmesh.boundary_face_ids(mesh)
b_pt = np.zeros(mesh.n_points, bool)
b_pt[qm.corners[bface]] = True                       # a point is on the boundary iff it is a corner of a boundary face
b_edge = np.zeros(len(mesh.edges), bool)
b_edge[qm.quads[bface].ravel()] = True               # an edge iff it is a side of one
h_min = np.linalg.norm(mesh.points[mesh.edges[:, 0]] - mesh.points[mesh.edges[:, 1]], axis=1).min()
amp = PERTURB * h_min / ORDER
rng = np.random.default_rng(SEED)


p_in, e_in, f_in = ~b_pt, ~b_edge, ~bface
mesh.points[p_in] += rng.uniform(-amp, amp, (int(p_in.sum()), 3))
mesh.edge_nodes[e_in] += rng.uniform(-amp, amp, mesh.edge_nodes[e_in].shape)
mesh.face_nodes[f_in] += rng.uniform(-amp, amp, mesh.face_nodes[f_in].shape)
mesh.interior += rng.uniform(-amp, amp, mesh.interior.shape)
assert hexmesh.is_conforming(mesh) and hexmesh.is_watertight(mesh)
assert torus_off(faces_of(mesh, "wall")).max() < 1e-12 * R, "the jitter moved a wall node"
build_seconds = time.perf_counter() - t_build
print("bent pipe: order %d, %d hexes, %d nodes moved by up to %.4f (h_min %.3f); built in %.2f s"
      % (ORDER, N_HEX, int(p_in.sum() + e_in.sum() * (ORDER - 1) + f_in.sum() * (ORDER - 1) ** 2
                           + mesh.interior.shape[0] * mesh.interior.shape[1]), amp, h_min, build_seconds))


# -- readings --------------------------------------------------------------------
V_EXACT = np.pi * R ** 2 * np.pi * R_BEND / 2        # Pappus


def wall_dev_pct(m):
    """Largest distance of the wall from the exact torus, in % of ``R``, read on a ``WALL_SAMPLE`` lattice between the
    wall's own nodes (the nodes alone would hide the faceting of an order-1 wall)."""
    blk = faces_of(m, "wall")
    fine = interp.resample_block(blk, m.order, max(WALL_SAMPLE, m.order), 2) if blk.shape[0] else blk
    return float(torus_off(fine).max() / R * 100)


def measure(m, seconds=0.0, passes=0, reason=""):
    """The row of one mesh: three readings of the scaled Jacobian, the wall, the volume, the cost."""
    qn = hexmesh.quality_summary(m)
    q7 = hexmesh.quality_summary(m, order=SAMPLE_ORDER)
    qc = hexmesh.corner_summary(m)
    return dict(order=m.order, sj_nodes_min=float(qn.min), sj_nodes_mean=float(qn.mean),
                sj7_min=float(q7.min), n_inv7=int(q7.n_inverted),
                sj_corner_min=float(qc.min), n_inv_corner=int(qc.n_inverted),
                wall_dev_pct=wall_dev_pct(m),
                vol_err_pct=float((hexmesh.volume(m, high_order=True) - V_EXACT) / V_EXACT * 100),
                seconds=seconds, iters=passes, reason=reason)


def field_of(m):
    """Per-hex scaled Jacobian read three ways, for the viewer."""
    return {"sj_nodes": hexmesh.scaled_jacobian(m), "sj7": hexmesh.scaled_jacobian(m, order=SAMPLE_ORDER),
            "sj_corner": hexmesh.corner_scaled_jacobian(m)}


def cap_off(m):
    """Largest distance of the cap nodes from their planes: ``|z|`` on the inlet, ``|x - R_BEND|`` on the outlet."""
    return float(max(np.abs(faces_of(m, "inlet")[..., 2]).max(), np.abs(faces_of(m, "outlet")[..., 0] - R_BEND).max()))


def wall_move(m):
    """Largest change of any wall node from the input, over every node of the wall faces (not the corners)."""
    return float(np.abs(faces_of(m, "wall") - faces_of(mesh, "wall")).max())


meshes = {"input": mesh}
rows = {"input": measure(mesh)}
fields = {"input": field_of(mesh)}
# [doc:constraints]
r = hs.resolve(mesh, [], boundary="slide")      # caps slide in their planes, the curved wall is held
# [/doc:constraints]
q0 = hs.quality(mesh.points, mesh.corners)
print("input: order 3 | min SJ nodes %.3f, order %d %.3f (%d inverted), corners %.3f (%d inverted) | wall %.1e %% off"
      % (rows["input"]["sj_nodes_min"], SAMPLE_ORDER, rows["input"]["sj7_min"], rows["input"]["n_inv7"],
         rows["input"]["sj_corner_min"], rows["input"]["n_inv_corner"], rows["input"]["wall_dev_pct"]))


# ====================================================================================================
# The three runs.  ``order1`` runs the order-1 ``tmop`` on the straight shadow; ``native`` and ``tmop`` run on the
# curved mesh itself (``run_schedule`` carries the curved nodes through native stages and keeps the order).
# ====================================================================================================
def run_variants():
    """Fills ``meshes``, ``rows``, ``fields`` with ``order1``, ``native``, ``tmop`` and returns ``checks``.

    Each row holds: ``order, sj_nodes_min, sj_nodes_mean, sj7_min, n_inv7, sj_corner_min,
    n_inv_corner, wall_dev_pct, vol_err_pct, seconds, iters, reason`` (``measure`` writes them all)."""
    def timed(m, resolved, schedule):
        t = time.perf_counter()
        out, logs = hs.run_schedule(m, resolved, schedule)
        return out, logs[-1], time.perf_counter() - t

    # order1: the straight shadow (same vertex numbering), repaired, the curve gone
    shadow = HexMesh.from_corners(mesh.points, mesh.corners, mesh.element_tags)
    assert np.array_equal(shadow.points, mesh.points)
    rs = hs.resolve(shadow, [], boundary="slide")
    out, log, sec = timed(shadow, rs, [("tmop", TMOP)])
    meshes["order1"] = out
    rows["order1"] = measure(out, sec, log.passes_done, log.reason)

    # native: vertices only, the curved nodes carried by the interpolated vertex displacement; tmop: every node
    out, log, sec = timed(mesh, r, hs.suggest_schedule(q0, outer=OUTER))
    meshes["native"] = out
    rows["native"] = measure(out, sec, log.passes_done, log.reason)
    t = time.perf_counter()
    # [doc:tmop]
    out, logs = hs.run_schedule(mesh, r, [("tmop", TMOP)])   # every node moves; out.order == mesh.order
    log = logs[-1]     # log.reason: iterations, energy, what was held; log.after: min det, scaled J at order 7
    # [/doc:tmop]
    sec = time.perf_counter() - t
    meshes["tmop"] = out
    rows["tmop"] = measure(out, sec, log.passes_done, log.reason)
    for k in ("order1", "native", "tmop"):
        fields[k] = field_of(meshes[k])
    return dict(wall_max_move=wall_move(meshes["tmop"]), cap_max_off=cap_off(meshes["tmop"]),
                conforming={k: bool(hexmesh.is_conforming(m) and hexmesh.is_watertight(m)) for k, m in meshes.items()})


checks = {}
if not INPUT_ONLY:
    checks = run_variants()
    print()
    for k in ("input", "order1", "native", "tmop"):
        d = rows[k]
        print("%-7s order %d | SJ min nodes %.3f mean %.3f, order %d %.3f (%d inv), corners %.3f (%d inv) | "
              "wall %.2e %% off | volume %+.2f %% | %.2f s"
              % (k, d["order"], d["sj_nodes_min"], d["sj_nodes_mean"], SAMPLE_ORDER, d["sj7_min"], d["n_inv7"],
                 d["sj_corner_min"], d["n_inv_corner"], d["wall_dev_pct"], d["vol_err_pct"], d["seconds"]))
        if d["reason"]:
            print("        reason: %s" % d["reason"])
    print("checks: wall max move %.1e, cap max off %.1e, conforming %s"
          % (checks["wall_max_move"], checks["cap_max_off"], checks["conforming"]))
    mesh = meshes["tmop"]
    # [doc:quality]
    q7 = hexmesh.quality_summary(mesh, order=SAMPLE_ORDER)    # the curved map, sampled at the solver's order
    qc = hexmesh.corner_summary(mesh)                          # the 8 corners alone: what .re2 writes
    print("min scaled J: %.3f at order %d, %.3f at the corners" % (q7.min, SAMPLE_ORDER, qc.min))
    # [/doc:quality]

# [doc:export]
for k, m in meshes.items():
    writer.to_vtu(m, "02_bent_pipe_%s.vtu" % k)        # VTK Lagrange cells: the curved geometry, for viewing
writer.to_re2(mesh, "02_bent_pipe.re2", bc=VEL_BC)     # Nek5000's mesh file: the 8 corners of each hex only
# [/doc:export]
