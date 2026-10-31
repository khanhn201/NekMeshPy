"""Frozen-reference cases for the shared smoother engine.

``record_quad`` / ``record_hex`` run the public smoother API of each rung on fixed inputs and flatten every
number into a ``{name: ndarray}`` dict.  ``data/reference_quad.npz`` and ``data/reference_hex.npz`` hold the
output of the code **before** it was moved into ``nekmeshpy.core.smoother``; ``test_equivalence.py`` re-runs the
same recipes and requires the same bytes.  Only the public namespaces of the two rungs are used, so the recipes
stay valid however the code underneath is organised.

Regenerate (only when a number is *meant* to change)::

    PYTHONPATH=. python tests/smoother/core/cases.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data")
for sub in ("quad", "hex"):
    path = os.path.join(_HERE, os.pardir, sub)
    if path not in sys.path:
        sys.path.append(path)

from hexblocks import block_arrays  # noqa: E402

from nekmeshpy import hexmesh, linemesh, quadmesh  # noqa: E402
from nekmeshpy.hexmesh import smoother as hs  # noqa: E402
from nekmeshpy.quadmesh.smoother import constrain as cn  # noqa: E402
from nekmeshpy.quadmesh.smoother import smooth  # noqa: E402


# -- quad inputs ---------------------------------------------------------------------------------
def _grid_quads(n: int) -> Any:
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    return np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)


def quad_cases() -> dict[str, tuple[Any, Any, Any]]:
    """``{name: (mesh, resolved, normals)}``: annulus sector, free disk, sphere, tangled square."""
    out: dict[str, tuple[Any, Any, Any]] = {}
    # annulus sector, boundary frozen
    n = 8
    u = np.linspace(0.0, 1.0, n + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    rad, ang = 0.4 + 0.6 * U, 0.5 * np.pi * V
    exact = np.stack([(rad * np.cos(ang)).ravel(), (rad * np.sin(ang)).ravel(), np.zeros(U.size)], axis=1)
    quads = _grid_quads(n)
    nrm = smooth.reference_normals(exact, quads)
    inner = ((U > 0) & (U < 1) & (V > 0) & (V < 1)).ravel()
    pts = exact.copy()
    pts[inner, :2] += np.random.default_rng(4).uniform(-0.45, 0.45, (inner.sum(), 2)) * 0.6 / n
    mesh = quadmesh.QuadMesh.from_corners(pts, quads)
    out["annulus"] = (mesh, cn.resolve(mesh, []), nrm)
    # free disk: tangent boundary, nothing pulls it back
    m0 = quadmesh.ogrid(linemesh.circle(1.0, 24, element_tag="wall"), 6, 4, wall_tag="wall")
    P = np.array(m0.points, dtype=float)
    wall = np.isclose(np.hypot(P[:, 0], P[:, 1]), 1.0, atol=1e-9)
    ell = cn.vertex_length(m0.points, m0.corners)
    P[~wall, :2] += np.random.default_rng(3).uniform(-0.3, 0.3, ((~wall).sum(), 2)) * ell[~wall, None]
    mesh = smooth.with_points(m0, P)
    out["disk_tangent"] = (mesh, cn.resolve(mesh, [], boundary="tangent"),
                           smooth.reference_normals(m0.points, m0.corners))
    # sphere: every vertex on the exact sphere, normals from the surface
    surf = cn.sphere([0, 0, 0], 1.0)
    cube = quadmesh.box(1.0, 4)
    P = surf.project(np.asarray(cube.points) / np.linalg.norm(cube.points, axis=1, keepdims=True))
    q = np.array(cube.corners)
    nq = np.cross(P[q[:, 1]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 0]])
    flip = np.einsum("ij,ij->i", nq, surf.normal(P[q].mean(1))) < 0
    q[flip] = q[flip][:, ::-1]
    ref = quadmesh.QuadMesh.from_corners(P, q)
    ell = cn.vertex_length(ref.points, ref.corners)[:, None]
    jit = surf.project(ref.points + np.random.default_rng(1).normal(size=ref.points.shape) * 0.25 * ell)
    mesh = smooth.with_points(ref, jit)
    res = cn.resolve(mesh, [cn.OnSurface(np.arange(mesh.n_points), surf)], boundary="free")
    out["sphere"] = (mesh, res, cn.surface_normals(mesh.points, mesh.corners, cn.claimed_surfaces(res))[0])
    # tangled square: three interior vertices dragged by up to 1.6 cells
    n = 8
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    quads = _grid_quads(n)
    clean = quadmesh.QuadMesh.from_corners(grid, quads)
    nrm = smooth.reference_normals(grid, quads)
    inner_ids = np.flatnonzero((grid[:, 0] > 0) & (grid[:, 0] < 1) & (grid[:, 1] > 0) & (grid[:, 1] < 1))
    rng = np.random.default_rng(2)
    while True:
        pts = grid.copy()
        for v in rng.choice(inner_ids, 3, replace=False):
            pts[v, :2] += rng.uniform(-1.6, 1.6, 2) / n
        mesh = smooth.with_points(clean, pts)
        if smooth.quality(mesh.points, mesh.corners, nrm).n_inverted:
            break
    out["tangled"] = (mesh, cn.resolve(mesh, []), nrm)
    return out


# -- hex inputs ----------------------------------------------------------------------------------
def hex_cases() -> dict[str, tuple[Any, Any]]:
    """``{name: (mesh, resolved)}``: a jittered 6x6x6 block (frozen boundary) and the cylinder of the hex examples
    (frozen, and ``boundary="tangent"``)."""
    out: dict[str, tuple[Any, Any]] = {}
    P, H = block_arrays(6, 0.3, 1)
    mesh = hexmesh.HexMesh.from_corners(P, H)
    out["block"] = (mesh, hs.resolve(mesh, []))
    r, length, sectors, layers = 1.0, 1.0, 16, 4
    cyl = hexmesh.extrude(quadmesh.ogrid(linemesh.circle(r, sectors), 4, 2), length, layers)
    exact = cyl.points.copy()
    rho = np.hypot(exact[:, 0], exact[:, 1])
    wall = np.isclose(rho, r)
    cap = np.isclose(exact[:, 2], 0.0) | np.isclose(exact[:, 2], length)
    rim = wall & cap
    wall_only, cap_only, inner = wall & ~cap, cap & ~wall, ~wall & ~cap
    h_arc, h_z, h_xy = 2 * np.pi * r / sectors, length / layers, 0.2
    u = np.random.default_rng(0).uniform(-0.25, 0.25, exact.shape)
    Q = exact.copy()
    Q[inner] += u[inner] * np.array([h_xy, h_xy, h_z])
    ang = np.arctan2(exact[:, 1], exact[:, 0])
    for m, dz in ((wall_only, True), (rim, False)):
        a = ang[m] + u[m, 0] * h_arc / r
        Q[m, 0], Q[m, 1] = r * np.cos(a), r * np.sin(a)
        if dz:
            Q[m, 2] += u[m, 1] * h_z
    Q[cap_only, :2] += u[cap_only, :2] * h_xy
    mesh = hexmesh.HexMesh.from_corners(Q, cyl.corners, cyl.element_tags)
    out["cylinder"] = (mesh, hs.resolve(mesh, []))
    out["cylinder_tangent"] = (mesh, hs.resolve(mesh, [], boundary="tangent"))
    return out


# -- flattening ----------------------------------------------------------------------------------
def _put(rec: dict[str, Any], key: str, value: Any) -> None:
    rec[key] = np.asarray(value)


def _log(rec: dict[str, Any], key: str, log: Any) -> None:
    _put(rec, key + ".text", [log.method, log.stopped, log.reason])
    _put(rec, key + ".counts", [log.passes_requested, log.passes_done])
    _put(rec, key + ".before", np.array(tuple(log.before), float))
    _put(rec, key + ".after", np.array(tuple(log.after), float))
    _put(rec, key + ".history", np.array(log.history, float).reshape(-1, 4))
    if log.points is not None:
        _put(rec, key + ".points", log.points)


def record_quad(name: str, mesh: Any, res: Any, nrm: Any) -> dict[str, np.ndarray]:
    """Everything the quad smoother computes on one case, flattened to ``{key: array}``."""
    rec: dict[str, Any] = {}
    P, c = np.asarray(mesh.points), mesh.corners
    w = np.linspace(0.5, 1.5, mesh.n_quads)
    for tag, delta in (("d0", 0.0), ("d1", 0.01)):
        m = smooth.corner_metrics(P, c, nrm, delta=delta)
        _put(rec, "phi." + tag, m.phi)
        _put(rec, "sj." + tag, m.sj)
        _put(rec, "obj." + tag, smooth.objective(P, c, nrm, delta=delta))
        _put(rec, "objw." + tag, smooth.objective(P, c, nrm, delta=delta, weights=w))
        _put(rec, "grad." + tag, smooth.gradient(P, c, nrm, delta=delta))
        _put(rec, "gradw." + tag, smooth.gradient(P, c, nrm, delta=delta, weights=w))
    _put(rec, "quality", np.array(tuple(smooth.quality(P, c, nrm)), float))
    _put(rec, "shape", np.array(tuple(smooth.shape_stats(P, c)), float))
    _put(rec, "failing", smooth.failing_elements(P, c, nrm))
    out, log = smooth.optimize(mesh, res, iters=20, normals=nrm)
    _put(rec, "optimize.points", out.points)
    _log(rec, "optimize", log)
    out, log = smooth.optimize(mesh, res, iters=8, normals=nrm, weights=w, delta=0.0, restart="none")
    _put(rec, "optimize_w.points", out.points)
    _log(rec, "optimize_w", log)
    out, log = smooth.laplace(mesh, res, passes=5, normals=nrm)
    _put(rec, "laplace.points", out.points)
    _log(rec, "laplace", log)
    if name != "sphere":
        out, log = smooth.winslow(mesh, res, sweeps=30, normals=nrm)
        _put(rec, "winslow.points", out.points)
        _log(rec, "winslow", log)
    sched = smooth.user_schedule(outer=2, n_lap=2, n_opt=6)
    if name == "tangled":
        sched = smooth.untangle_schedule(iters=8) + sched
    out, logs = smooth.run_schedule(mesh, res, sched, normals=nrm, record=True)
    _put(rec, "schedule.points", out.points)
    for i, lg in enumerate(logs):
        _log(rec, "schedule.%02d" % i, lg)
    return rec


def record_hex(name: str, mesh: Any, res: Any) -> dict[str, np.ndarray]:
    """Everything the hex smoother computes on one case, flattened to ``{key: array}``."""
    rec: dict[str, Any] = {}
    P, c = np.asarray(mesh.points), mesh.corners
    w = np.linspace(0.5, 1.5, mesh.n_hexes)
    for tag, delta in (("d0", 0.0), ("d1", 0.01)):
        m = hs.corner_metrics(P, c, delta=delta)
        _put(rec, "phi." + tag, m.phi)
        _put(rec, "sj." + tag, m.sj)
        _put(rec, "obj." + tag, hs.objective(P, c, delta=delta))
        _put(rec, "objw." + tag, hs.objective(P, c, delta=delta, weights=w))
        _put(rec, "grad." + tag, hs.gradient(P, c, delta=delta))
        _put(rec, "gradw." + tag, hs.gradient(P, c, delta=delta, weights=w))
    _put(rec, "quality", np.array(tuple(hs.quality(P, c)), float))
    _put(rec, "shape", np.array(tuple(hs.shape_stats(P, c)), float))
    _put(rec, "failing", hs.failing_elements(P, c))
    out, log = hs.optimize(mesh, res, iters=20)
    _put(rec, "optimize.points", out.points)
    _log(rec, "optimize", log)
    out, log = hs.optimize(mesh, res, iters=8, weights=w, delta=0.0, restart="none")
    _put(rec, "optimize_w.points", out.points)
    _log(rec, "optimize_w", log)
    out, log = hs.laplace(mesh, res, passes=4)
    _put(rec, "laplace.points", out.points)
    _log(rec, "laplace", log)
    out, logs = hs.run_schedule(mesh, res, hs.user_schedule(outer=2, n_lap=2, n_opt=6), record=True)
    _put(rec, "schedule.points", out.points)
    for i, lg in enumerate(logs):
        _log(rec, "schedule.%02d" % i, lg)
    return rec


def record_all() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    quad: dict[str, np.ndarray] = {}
    for name, (mesh, res, nrm) in quad_cases().items():
        quad.update({"%s/%s" % (name, k): v for k, v in record_quad(name, mesh, res, nrm).items()})
    hexd: dict[str, np.ndarray] = {}
    for name, (mesh, res) in hex_cases().items():
        hexd.update({"%s/%s" % (name, k): v for k, v in record_hex(name, mesh, res).items()})
    return quad, hexd


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    q, h = record_all()
    np.savez_compressed(os.path.join(DATA, "reference_quad.npz"), **q)
    np.savez_compressed(os.path.join(DATA, "reference_hex.npz"), **h)
    print("wrote %d quad and %d hex arrays to %s" % (len(q), len(h), DATA))
