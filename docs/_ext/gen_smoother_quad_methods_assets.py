"""Generate the assets of the smoothing *method* pages (``docs/smoother/quad/methods/``).

Not part of the ``nekmeshpy`` package -- doc-build tooling like ``gen_smoother_quad_assets.py``; run it by hand after the
smoother changes, and commit what it writes (well under a minute):

    python docs/_ext/gen_smoother_quad_methods_assets.py [--vtu DIR]

The pages explain the methods themselves, so the figures are about a method's mechanics and characteristics:

* ``Which method when`` runs the **same input** through four ways of smoothing and shows the results as small multiples
  (input | Laplace | Winslow | Optimizer | Combined, one colour scale), with a scoreboard and per-characteristic findings:

  - ``Laplace only``       ``smooth.laplace(passes=LAPLACE_PASSES, w=LAPLACE_W)``;
  - ``Winslow only``       ``smooth.winslow`` at its defaults (``sweeps=200``);
  - ``Optimizer only``     ``smooth.optimize(iters=OPTIMIZER_ITERS)`` at the default ``delta``;
  - ``Combined schedule``  ``smooth.suggest_schedule``, whose composition is read from ``user_schedule()`` and
    ``untangle_schedule()`` (a delta-continuation preset of ``optimize``) themselves.

* ``Laplacian smoothing`` gets a step-by-step diagram of one pass on a tiny patch, computed with the functions
  ``smooth.laplace`` itself calls and checked against ``smooth.laplace(passes=1)``, and the free-boundary rules side by
  side on a disk.
* ``Winslow smoothing`` gets the concave input of the comparison, Laplace against Winslow, and a short table.
* ``Optimizer`` gets the objective history of real runs (the stage logs) and the interactive phi / phi_delta plot.
* ``Constraints`` gets the one-step projection figure and ``boundary="slide"`` against ``"free"``, ``Grading`` small
  multiples of the protections.

Every number on a page is generated: parameter defaults come from ``inspect`` of the real signatures, the rest from the
runs.  Seconds are single measurements, no repeats.  Every figure is inline SVG styled by furo's CSS variables
(``docs/_static/custom.css``), so it follows the light / dark theme.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from typing import Any

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                                   # the cost figures are for serial runs

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOCS)
GENERATED = os.path.join(_DOCS, "smoother", "quad", "_generated")
sys.path.insert(0, _REPO)

from nekmeshpy import linemesh, quadmesh  # noqa: E402
from nekmeshpy.core.fields import geometric_spacing  # noqa: E402
from nekmeshpy.core.smoother import metric  # noqa: E402
from nekmeshpy.quadmesh.smoother import constrain as cn  # noqa: E402
from nekmeshpy.quadmesh.smoother import smooth  # noqa: E402

LAPLACE_PASSES, LAPLACE_W = 100, 0.95
OPTIMIZER_ITERS = 200
LAPLACE = [("laplace", {"passes": LAPLACE_PASSES, "w": LAPLACE_W})]
OPTIMIZER = [("optimize", {"iters": OPTIMIZER_ITERS})]
WINSLOW_SWEEPS = inspect.signature(smooth.winslow).parameters["sweeps"].default        # Winslow at its defaults
WINSLOW = [("winslow", {"sweeps": WINSLOW_SWEEPS})]
METHODS = [("laplace", "Laplace only"), ("winslow", "Winslow only"), ("optimizer", "Optimizer only"),
           ("combined", "Combined schedule")]
SHORT = {"laplace": "Laplace", "winslow": "Winslow", "optimizer": "Optimizer", "combined": "Combined"}
TIE = 1e-3                                                           # relative: two readings this close are a tie
MACHINE = "a laptop, running serial"
DIAGRAM_W = 0.7                                                      # the Laplace diagram's shrink factor (visible)
UP = "../../../"                                                        # from docs/smoother/quad/methods/*.md to docs/


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fid:
        fid.write(text)


def raw(html: str) -> str:
    return "```{raw} html\n" + html + "\n```\n"


def num(v: Any) -> str:
    """A knob value as it would be typed in a call."""
    if isinstance(v, bool) or isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float):
        return "%g" % v
    return repr(v)


def default(fn: Any, name: str) -> Any:
    return inspect.signature(fn).parameters[name].default


# ============================================================================================================
# inputs
# ============================================================================================================
class Input:
    """One thing to smooth: its mesh, constraints, normals, and the readings that are specific to it."""

    def __init__(self, key: str, label: str, kind: str, mesh: Any, res: Any, nrm: Any, *,
                 extras: Any = None, tangle_cases: Any = None, layer_line: Any = None) -> None:
        self.key, self.label, self.kind = key, label, kind
        self.mesh, self.res, self.nrm = mesh, res, nrm
        self.extras = extras                  # callable(points) -> {label: value}
        self.tangle_cases = tangle_cases      # callable(method key) -> (repaired, total), the tangled family
        self.layer_line = layer_line          # vertex ids along a designed layer's outer edge, drawn on the figures
        self.q0 = smooth.quality(mesh.points, mesh.corners, nrm)
        self.area0 = signed_area(mesh.corners, np.asarray(mesh.points), nrm)


def grid_quads(nx: int, ny: int) -> Any:
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    return idx, np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)


def annulus() -> Input:
    """The 90-degree annular sector of ``examples/smoother/quad/01_basics.py``: boundary frozen, interior jittered."""
    n, r_in, r_out, jitter, seed = 14, 0.4, 1.0, 0.45, 4
    u = np.linspace(0.0, 1.0, n + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    rad, ang = r_in + (r_out - r_in) * U, 0.5 * np.pi * V
    exact = np.stack([(rad * np.cos(ang)).ravel(), (rad * np.sin(ang)).ravel(), np.zeros(U.size)], axis=1)
    _, quads = grid_quads(n, n)
    nrm = smooth.reference_normals(exact, quads)
    inner = ((U > 0) & (U < 1) & (V > 0) & (V < 1)).ravel()
    pts = exact.copy()
    pts[inner, :2] += np.random.default_rng(seed).uniform(-jitter, jitter, (inner.sum(), 2)) * (r_out - r_in) / n
    mesh = quadmesh.QuadMesh.from_corners(pts, quads)
    return Input("annulus", "annulus", "planar", mesh, cn.resolve(mesh, []), nrm)


def disk() -> Input:
    """An O-grid disk with a jittered interior and a tangent boundary (a plain mesh, no geometry attached): the
    boundary moves along itself and nothing pulls it back, which is where the boundary modes differ."""
    radius, wall_n, jitter, seed = 1.0, 24, 0.3, 3
    m0 = quadmesh.ogrid(linemesh.circle(radius, wall_n, element_tag="wall"), 6, 4, wall_tag="wall")
    P = np.array(m0.points, dtype=float)
    wall = np.isclose(np.hypot(P[:, 0], P[:, 1]), radius, atol=1e-9)
    ell = cn.vertex_length(m0.points, m0.corners)
    P[~wall, :2] += np.random.default_rng(seed).uniform(-jitter, jitter, ((~wall).sum(), 2)) * ell[~wall, None]
    mesh = smooth.with_points(m0, P)
    nrm = smooth.reference_normals(m0.points, m0.corners)
    return Input("disk", "free disk", "planar", mesh, cn.resolve(mesh, [], boundary="tangent"), nrm)


def tangled() -> Input:
    """``02_untangle``'s protocol: a unit square of 8 x 8 quads, boundary frozen, three interior vertices dragged by up
    to 1.6 cells.  The mesh shown is the first draw that tangles; ``tangle_cases`` runs every one of ``cases`` draws."""
    n, cases, dragged, drag, seed = 8, 8, 3, 1.6, 2
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    _, quads = grid_quads(n, n)
    clean = quadmesh.QuadMesh.from_corners(grid, quads)
    nrm = smooth.reference_normals(grid, quads)
    inner = np.flatnonzero((grid[:, 0] > 0) & (grid[:, 0] < 1) & (grid[:, 1] > 0) & (grid[:, 1] < 1))
    rng = np.random.default_rng(seed)
    family: list[Any] = []
    while len(family) < cases:
        pts = grid.copy()
        for v in rng.choice(inner, dragged, replace=False):
            pts[v, :2] += rng.uniform(-drag, drag, 2) / n
        m = smooth.with_points(clean, pts)
        if smooth.quality(m.points, m.corners, nrm).n_inverted:
            family.append(m)

    def repaired(method: str) -> tuple[int, int]:
        ok = 0
        for m in family:
            q0 = smooth.quality(m.points, m.corners, nrm)
            _, logs = smooth.run_schedule(m, cn.resolve(m, []), schedule_of(method, q0), normals=nrm)
            ok += logs[-1].after.n_inverted == 0
        return ok, len(family)

    return Input("tangled", "tangled square", "planar", family[0], cn.resolve(family[0], []), nrm,
                 tangle_cases=repaired)


def sphere() -> Input:
    """``04_curved_surfaces``'s cubed sphere: tangential jitter, every vertex ``OnSurface`` the exact sphere, the normals
    taken from the surface."""
    surf = cn.sphere([0, 0, 0], 1.0)
    cube = quadmesh.box(1.0, 5)
    P = surf.project(np.asarray(cube.points) / np.linalg.norm(cube.points, axis=1, keepdims=True))
    q = cube.corners
    nq = np.cross(P[q[:, 1]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 0]])
    flip = np.einsum("ij,ij->i", nq, surf.normal(P[q].mean(1))) < 0
    q = q.copy()
    q[flip] = q[flip][:, ::-1]
    ref = quadmesh.QuadMesh.from_corners(P, q)
    ell = cn.vertex_length(ref.points, ref.corners)[:, None]
    jit = surf.project(ref.points + np.random.default_rng(1).normal(size=ref.points.shape) * 0.25 * ell)
    mesh = smooth.with_points(ref, jit)
    res = cn.resolve(mesh, [cn.OnSurface(np.arange(mesh.n_points), surf)], boundary="free")
    nrm = cn.surface_normals(mesh.points, mesh.corners, cn.claimed_surfaces(res))[0]
    return Input("sphere", "sphere", "surface", mesh, res, nrm)


LAYER_ROWS = (0.0, 0.004, 0.0096, 0.017, 0.030)      # y of the thin layer's rows: four layers (03_constraints)
LAYER_NX, STRIP_CELLS = 12, 8
LAYER_KEY = "layer thickness change, max (%)"


def strip(keep: bool = False) -> Input:
    """``03_constraints``'s strip: a four-layer graded wall layer under a uniform block, boundary frozen."""
    seed, keep_rings = 3, len(LAYER_ROWS) - 1
    ys = np.r_[np.asarray(LAYER_ROWS), np.linspace(LAYER_ROWS[-1], 1.0, STRIP_CELLS + 1)[1:]]
    xs = np.linspace(0.0, 1.0, LAYER_NX + 1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    S = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], axis=1)
    sidx, squads = grid_quads(len(xs) - 1, len(ys) - 1)
    nrm = smooth.reference_normals(S, squads)
    above = np.zeros(S.shape[0], bool)
    above[sidx[1:-1, len(LAYER_ROWS):-1]] = True
    S[above, :2] += np.random.default_rng(seed).uniform(-0.3, 0.3, (above.sum(), 2)) * [1 / LAYER_NX, 1 / STRIP_CELLS]
    mesh = quadmesh.QuadMesh.from_corners(S, squads)
    top0 = S[sidx[1:-1, keep_rings], 1].copy()

    def extras(pts: Any) -> dict[str, float]:
        top = np.asarray(pts)[sidx[1:-1, keep_rings], 1]
        return {LAYER_KEY: float(100 * (np.abs(top - top0) / top0).max())}

    cons = [cn.KeepLayer(sidx[:, 0], rings=keep_rings)] if keep else []
    return Input("layer", "graded layer", "planar", mesh, cn.resolve(mesh, cons), nrm, extras=extras,
                 layer_line=sidx[:, keep_rings])


REFLEX_KEY = "inverted away from the reflex corner"


def polyline(pts: list[tuple[float, float]], cells: list[int]) -> Any:
    """``(n, 2)`` points along a polyline, ``cells[s]`` equal cells on segment ``s``."""
    out = [np.asarray(pts[0], float)]
    for a, b, n in zip(pts[:-1], pts[1:], cells):
        out += [(1 - t) * np.asarray(a, float) + t * np.asarray(b, float) for t in np.arange(1, n + 1) / n]
    return np.array(out)


def lshape() -> Input:
    """An L-shaped domain meshed as one block: the Coons patch of its boundary, with the re-entrant (reflex) corner a
    point of the top side.  The patch folds near that corner; the boundary is frozen.  The normals are the plane's: a
    folded quad's own winding would read it as valid."""
    bottom, right = polyline([(0, 0), (2, 0)], [18]), polyline([(2, 0), (2, 1)], [12])
    top, left = polyline([(0, 2), (1, 2), (1, 1), (2, 1)], [6, 6, 6]), polyline([(0, 0), (0, 2)], [12])
    nx, ny = len(bottom) - 1, len(left) - 1
    u, v = (np.arange(nx + 1) / nx)[:, None, None], (np.arange(ny + 1) / ny)[None, :, None]
    X = ((1 - v) * bottom[:, None] + v * top[:, None] + (1 - u) * left[None] + u * right[None]
         - ((1 - u) * (1 - v) * bottom[0] + u * (1 - v) * bottom[-1] + (1 - u) * v * top[0] + u * v * top[-1]))
    P = np.zeros((X.shape[0] * X.shape[1], 3))
    P[:, :2] = X.reshape(-1, 2)
    idx, quads = grid_quads(nx, ny)
    nrm = np.tile([0.0, 0.0, 1.0], (len(quads), 1))
    if signed_area(quads, P, nrm) < 0:
        quads = quads[:, ::-1].copy()
    mesh = quadmesh.QuadMesh.from_corners(P, quads)
    reflex = int(idx[12, ny])                                        # the top side's 13th point, (1, 1)
    assert np.allclose(P[reflex, :2], [1.0, 1.0])
    at_reflex = np.any(mesh.corners == reflex, axis=1)

    def extras(pts: Any) -> dict[str, float]:
        bad = smooth.corner_metrics(np.asarray(pts, float), mesh.corners, nrm).sj.min(axis=1) <= 0
        return {REFLEX_KEY: int((bad & ~at_reflex).sum())}

    return Input("lshape", "L-shape", "planar", mesh, cn.resolve(mesh, []), nrm, extras=extras)


def schedule_of(method: str, q0: smooth.Quality) -> list[tuple[str, dict[str, Any]]]:
    return {"laplace": LAPLACE, "winslow": WINSLOW, "optimizer": OPTIMIZER,
            "combined": smooth.suggest_schedule(q0)}[method]


# ============================================================================================================
# measurement
# ============================================================================================================
def signed_area(corners: Any, pts: Any, nrm: Any) -> float:
    Q = pts[corners]
    cr = 0.5 * np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1])
    return float(np.einsum("ij,ij->i", cr, nrm).sum())


def measure(inp: Input, pts: Any, seconds: float | None) -> dict[str, Any]:
    """Every reading of one state of ``inp``; ``seconds`` is ``None`` for the input itself."""
    corners = inp.mesh.corners
    q = smooth.quality(pts, corners, inp.nrm)
    s = smooth.shape_stats(pts, corners)
    Q = pts[corners]
    area = np.linalg.norm(0.5 * np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1]), axis=1)
    out: dict[str, Any] = {
        "min scaled J": q.sj_min, "mean scaled J": q.sj_mean, "mean phi": q.phi_mean,
        "size ratio max": s.size_ratio_max, "CV of area": float(area.std() / area.mean()),
        "area change (%)": 100.0 * (signed_area(corners, pts, inp.nrm) - inp.area0) / inp.area0,
        "inverted": q.n_inverted, "seconds": seconds,
    }
    if inp.extras is not None:
        out.update(inp.extras(pts))
    return out


#: reading, direction (+1: higher is better, -1: lower), format
CORE = [
    ("min scaled J", +1, "%.3f"), ("mean scaled J", +1, "%.3f"), ("mean phi", -1, "%.3f"),
    ("size ratio max", -1, "%.2f"), ("CV of area", -1, "%.3f"), ("area change (%)", -1, "%+.1f"),
    ("inverted", -1, "%d"), ("seconds", -1, "%.2f"),
]
FMT = {k: f for k, _, f in CORE}
FMT[LAYER_KEY], FMT[REFLEX_KEY] = "%.0f", "%d"
DIRECTION = {k: d for k, d, _ in CORE}
DIRECTION[LAYER_KEY] = DIRECTION[REFLEX_KEY] = -1


def fmt(key: str, v: Any) -> str:
    if v is None:
        return "–"
    if key == "area change (%)" and abs(v) < 0.05:
        v = 0.0                                                      # no "-0.0"
    if key == "mean phi" and v > 1e3:
        return "%.1e" % v
    return FMT.get(key, "%.3g") % v


def run_all() -> list[dict[str, Any]]:
    """Run every input through the four methods once and return the readings and the final meshes."""
    rows = []
    for inp in (annulus(), disk(), tangled(), sphere(), strip(), lshape()):
        runs: dict[str, Any] = {}
        for key, label in METHODS:
            sched = schedule_of(key, inp.q0)
            t0 = time.perf_counter()
            out, logs = smooth.run_schedule(inp.mesh, inp.res, sched, normals=inp.nrm)
            dt = time.perf_counter() - t0
            runs[key] = {"label": label, "mesh": out, "logs": logs, "seconds": dt,
                         "reading": measure(inp, np.asarray(out.points), dt)}
            print("  %-22s %-18s %d inverted, min J %.3f, %.2f s" % (
                inp.label, label, runs[key]["reading"]["inverted"], runs[key]["reading"]["min scaled J"], dt), flush=True)
        if inp.tangle_cases is not None:
            for key, _ in METHODS:
                runs[key]["reading"]["cases repaired"] = inp.tangle_cases(key)
        rows.append({"input": inp, "start": measure(inp, np.asarray(inp.mesh.points), None), "runs": runs})
    return rows


def winners(values: dict[str, Any], direction: int) -> list[str]:
    """The methods whose value is within ``TIE`` of the best one (several on a tie)."""
    vs = {k: v for k, v in values.items() if v is not None}
    best = max(vs.values()) if direction > 0 else min(vs.values())
    tol = max(TIE * abs(best), 5e-4 if not isinstance(best, (int, np.integer)) else 0)
    return [k for k, v in vs.items() if abs(v - best) <= tol]


def best_of(row: dict[str, Any], key: str) -> list[str]:
    vals = {k: r["reading"][key] for k, r in row["runs"].items()}
    if key == "area change (%)":
        vals = {k: abs(v) for k, v in vals.items()}
    return winners(vals, DIRECTION[key])


def by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["input"].key: r["runs"] for r in rows}


def per_step(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Median seconds per Laplace pass, Winslow sweep and optimizer iteration over the inputs, from the single-method
    runs (the inputs where the stage took at least one step)."""
    out = {}
    for m in ("laplace", "winslow", "optimizer"):
        out[m] = float(np.median([r["runs"][m]["seconds"] / r["runs"][m]["logs"][0].passes_done for r in rows
                                  if r["runs"][m]["logs"][0].passes_done > 0]))
    return out


# ============================================================================================================
# the three methods, written down from the code
# ============================================================================================================
def call(method: str, knobs: dict[str, Any]) -> str:
    return "%s(%s)" % (method, ", ".join("%s=%s" % (k, num(v)) for k, v in knobs.items()))


def compress(sched: list[tuple[str, dict[str, Any]]]) -> str:
    """A schedule as a short expression: repeated blocks as ``k x [...]``, a run of one method whose knobs differ only in
    one key as ``method(..., key=v)`` over the listed values."""
    out, i = [], 0
    while i < len(sched):
        m, kn = sched[i]
        j = i + 1
        while j < len(sched) and sched[j][0] == m and set(sched[j][1]) == set(kn):
            j += 1
        varying = [k for k in kn if len({num(s[1][k]) for s in sched[i:j]}) > 1]
        if j - i > 1 and len(varying) == 1:
            key = varying[0]
            fixed = {k: v for k, v in kn.items() if k != key}
            vals = ", ".join(num(s[1][key]) for s in sched[i:j])
            out.append("%s(%s, %s=v) for v in %s" % (m, ", ".join("%s=%s" % (k, num(v)) for k, v in fixed.items()), key, vals))
            i = j
            continue
        best = (1, 1)
        for L in range(1, 5):
            block = sched[i:i + L]
            if len(block) < L:
                break
            k = 1
            while sched[i + k * L:i + (k + 1) * L] == block:
                k += 1
            if k > 1 and k * L > best[0] * best[1]:
                best = (L, k)
        L, k = best
        body = " + ".join(call(*s) for s in sched[i:i + L])
        out.append(("%d × [%s]" % (k, body)) if k > 1 and L > 1 else ("%d × %s" % (k, body)) if k > 1 else body)
        i += L * k
    return " + ".join(out)


def budget(sched: list[tuple[str, dict[str, Any]]]) -> dict[str, int]:
    """The largest number of steps a schedule may take, per method."""
    out: dict[str, int] = {}
    for m, kn in sched:
        out[m] = out.get(m, 0) + int(kn.get("passes", kn.get("iters", kn.get("sweeps", 0))))
    return out


def steps_text(b: dict[str, int]) -> str:
    names = {"laplace": "Laplace passes", "optimize": "optimizer iterations", "laplace_local": "local Laplace passes",
             "optimize_local": "local optimizer iterations", "winslow": "Winslow sweeps"}
    return ", ".join("%d %s" % (v, names[k]) for k, v in b.items())


def methods_table() -> str:
    user, untangle = smooth.user_schedule(), smooth.untangle_schedule()
    delta = default(smooth.optimize, "delta")
    body = [
        ["**Laplace only**", "`%s`" % call("laplace", LAPLACE[0][1]), "one Laplace stage", steps_text(budget(LAPLACE))],
        ["**Winslow only**", "`%s`" % call("winslow", WINSLOW[0][1]), "one Winslow stage at its defaults",
         steps_text(budget(WINSLOW))],
        ["**Optimizer only**", "`%s`" % call("optimize", OPTIMIZER[0][1]),
         "one optimizer stage at the default `delta=%s`" % num(delta), steps_text(budget(OPTIMIZER))],
        ["**Combined**", "`suggest_schedule(quality)`",
         "`user_schedule()` = `%s`; when a quad starts inverted, optimizer passes with a wide-to-narrow `delta` come first "
         "(`untangle_schedule()` = `%s`)"
         % (compress(user), compress(untangle)),
         "%s; plus %s when a quad starts inverted" % (steps_text(budget(user)), steps_text(budget(untangle)))],
    ]
    return md_table(["method", "call", "runs", "at most"], body)


# ============================================================================================================
# drawing: small multiples of quad meshes coloured by scaled Jacobian
# ============================================================================================================
_STOPS = [(0.0, (214, 64, 52)), (0.2, (240, 150, 60)), (0.45, (244, 213, 94)), (1.0, (92, 170, 98))]


def sj_colour(sj: float) -> str:
    """The case-study viewers' colour scale (``smoother_viewer.js``): red at or below 0, yellow, green at 1."""
    t = min(max(float(sj), 0.0), 1.0)
    for (a, ca), (b, cb) in zip(_STOPS[:-1], _STOPS[1:]):
        if t <= b:
            u = (t - a) / (b - a)
            return "#%02x%02x%02x" % tuple(int(round(ca[k] + u * (cb[k] - ca[k]))) for k in range(3))
    return "#5caa62"


def boundary_segments(corners: Any) -> Any:
    """``(B, 2)`` vertex pairs of the edges used by one quad only."""
    e = np.stack([corners, np.roll(corners, -1, axis=1)], axis=-1).reshape(-1, 2)
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv.ravel()] == 1]


class View:
    """How a mesh state is put on paper: planar meshes in their x-y plane, the sphere orthographically with its back
    half culled."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        d = np.array([1.0, 0.75, 0.9]) / np.linalg.norm([1.0, 0.75, 0.9])
        e1 = np.cross([0.0, 0.0, 1.0], d)
        e1 /= np.linalg.norm(e1)
        self.d, self.e1, self.e2 = d, e1, np.cross(d, e1)

    def xy(self, pts: Any) -> Any:
        pts = np.asarray(pts, float)
        if self.kind == "planar":
            return pts[:, :2]
        return np.stack([pts @ self.e1, pts @ self.e2], axis=1)

    def visible(self, pts: Any, corners: Any) -> Any:
        if self.kind == "planar":
            return np.ones(len(corners), bool)
        return np.asarray(pts, float)[corners].mean(axis=1) @ self.d > 0.05


def mesh_svg(view: View, pts: Any, corners: Any, nrm: Any, box: tuple[float, float, float, float],
             outline: Any = None, label: str = "", line: Any = None, dots: Any = None) -> str:
    """One panel: every visible quad filled by its worst-corner scaled Jacobian, inverted quads outlined, and the input's
    boundary (``outline``, 2-D points of segments) dashed on top."""
    x0, y0, x1, y1 = box
    s = 1000.0 / max(x1 - x0, y1 - y0)
    W, H = int(round(s * (x1 - x0))), int(round(s * (y1 - y0)))
    xy = view.xy(pts)
    P = np.stack([(xy[:, 0] - x0) * s, (y1 - xy[:, 1]) * s], axis=1)
    sj = smooth.corner_metrics(np.asarray(pts, float), corners, nrm).sj.min(axis=1)
    vis = view.visible(pts, corners)
    quads, bad = [], []
    for q in np.flatnonzero(vis):
        c = P[corners[q]]
        d = "M%d %dL%d %dL%d %dL%d %dZ" % tuple(np.round(c).astype(int).ravel())
        quads.append('<path d="%s" fill="%s"/>' % (d, sj_colour(sj[q])))
        if sj[q] <= 0:
            bad.append(d)
    out = ['<svg viewBox="0 0 %d %d" class="mm-mesh" role="img" aria-label="%s">' % (W, H, label),
           '<g class="q">%s</g>' % "".join(quads)]
    if bad:
        out.append('<path class="inv" d="%s"/>' % "".join(bad))
    if line is not None:
        ids, closed = line
        out.append('<path class="ly" d="M%s%s"/>' % (" L".join("%d %d" % tuple(np.round(P[i]).astype(int)) for i in ids),
                                                     " Z" if closed else ""))
    if outline is not None:
        o = np.stack([(outline[..., 0] - x0) * s, (y1 - outline[..., 1]) * s], axis=-1)
        out.append('<path class="ol" d="%s"/>' % "".join("M%d %dL%d %d" % tuple(np.round(seg).astype(int).ravel())
                                                         for seg in o))
    if dots is not None and len(dots):
        out.append("".join('<circle class="fzd" cx="%d" cy="%d" r="26"/>' % tuple(np.round(P[i]).astype(int)) for i in dots))
    return "".join(out) + "</svg>"


def panel_box(view: View, states: list[Any], corners: Any) -> tuple[float, float, float, float]:
    xy = np.concatenate([view.xy(p)[np.unique(corners[view.visible(p, corners)])] for p in states])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = 0.03 * float((hi - lo).max())
    return float(lo[0] - pad), float(lo[1] - pad), float(hi[0] + pad), float(hi[1] + pad)


def multiples_row(title: str, view: View, mesh0: Any, nrm: Any, states: list[tuple[str, Any, str]],
                  show_outline: bool, line: Any = None, dots: Any = None) -> str:
    """One row of small multiples, one shared box (so sizes compare across the row): ``states`` are ``(label, points,
    caption)``, the first the input.  ``dots`` maps a state index to vertex ids drawn as frozen-vertex dots.  A row of
    more than four panels widens its grid (``--mm-cols``, read by ``custom.css``)."""
    corners = mesh0.corners
    box = panel_box(view, [p for _, p, _ in states], corners)
    seg = boundary_segments(corners) if show_outline else None
    outline = view.xy(np.asarray(mesh0.points))[seg] if seg is not None and len(seg) else None
    cells = []
    for i, (label, pts, cap) in enumerate(states):
        svg = mesh_svg(view, pts, corners, nrm, box, outline if i > 0 else None, "%s, %s" % (title, label), line,
                       (dots or {}).get(i))
        cells.append('<figure class="mm-cell">%s<figcaption><b>%s</b><br>%s</figcaption></figure>' % (svg, label, cap))
    cols = ' style="--mm-cols:%d"' % len(states) if len(states) > 4 else ""
    return '<div class="mm-row"%s><p class="mm-name">%s</p>%s</div>' % (cols, title, "".join(cells))


LEGEND = ('<div class="mm-legend"><span>worst-corner scaled J</span><span>≤ 0</span><i></i><span>1</span>'
          '<span class="mm-dash"></span><span>input boundary</span>'
          '<span class="mm-ly"></span><span>outer edge of a designed layer</span></div>')


def caption(inp: Input, rd: dict[str, Any]) -> str:
    parts = ["min J %.2f" % rd["min scaled J"]]
    if REFLEX_KEY in rd and rd["inverted"]:
        parts.append("%d inverted, at the corner" % rd["inverted"] if rd[REFLEX_KEY] == 0 else
                     "%d inverted (%d off the corner)" % (rd["inverted"], rd[REFLEX_KEY]))
    elif inp.q0.n_inverted or rd["inverted"]:
        parts.append("%d inverted" % rd["inverted"])
    if inp.key == "disk":
        parts.append("area %s %%" % fmt("area change (%)", rd["area change (%)"]))
    if LAYER_KEY in rd:
        parts.append("layer %+.0f %%" % rd[LAYER_KEY] if rd[LAYER_KEY] > 0.05 else "layer as meshed")
    return " · ".join(parts)


def compare_figure(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        inp = row["input"]
        states = [("input", np.asarray(inp.mesh.points), caption(inp, row["start"]))]
        for key, _ in METHODS:
            r = row["runs"][key]
            states.append((SHORT[key], np.asarray(r["mesh"].points), caption(inp, r["reading"])))
        line = None if inp.layer_line is None else (inp.layer_line, False)
        out.append(multiples_row(inp.label, View(inp.kind), inp.mesh, inp.nrm, states, inp.kind == "planar", line))
    return raw('<div class="mm-grid">' + LEGEND + "".join(out) + "</div>")


# ============================================================================================================
# the comparison's tables and findings
# ============================================================================================================
SCORE_KEYS = [("shape", "min scaled J"), ("shape", "mean phi"), ("size", "size ratio max"), ("size", "CV of area")]


def score_table(rows: list[dict[str, Any]]) -> str:
    """Compact: per shape / size reading, on how many inputs each method is best (ties count for each); then the
    readings that only one input measures, and the cost."""
    n = len(rows)
    by = by_key(rows)
    start = {r["input"].key: r["start"] for r in rows}
    body = []

    def bold(cells: dict[str, str], best: list[str]) -> list[str]:
        return ["**%s**" % cells[m] if m in best and len(best) < len(METHODS) else cells[m] for m, _ in METHODS]

    for group, key in SCORE_KEYS:
        t = {m: 0 for m, _ in METHODS}
        for row in rows:
            for m in best_of(row, key):
                t[m] += 1
        top = max(t.values())
        body.append([group, key]
                    + bold({m: str(v) for m, v in t.items()}, [m for m, v in t.items() if v == top]))
    rep = {m: by["tangled"][m]["reading"]["cases repaired"] for m, _ in METHODS}
    tot = rep["laplace"][1]
    body.append(["tangled input", "tangled squares repaired (of %d)" % tot]
                + bold({m: str(v[0]) for m, v in rep.items()}, winners({m: v[0] for m, v in rep.items()}, +1)))
    left = {m: by["lshape"][m]["reading"]["inverted"] for m, _ in METHODS}
    body.append(["concave", "L-shape, inverted quads left (input %d)" % start["lshape"]["inverted"]]
                + bold({m: str(v) for m, v in left.items()}, winners(left, -1)))
    drift = {m: by["disk"][m]["reading"]["area change (%)"] for m, _ in METHODS}
    body.append(["free boundary", "area change, free disk (%)"]
                + bold({m: fmt("area change (%)", v) for m, v in drift.items()},
                       winners({m: abs(v) for m, v in drift.items()}, -1)))
    lay = {m: by["layer"][m]["reading"][LAYER_KEY] for m, _ in METHODS}
    body.append(["grading", "graded layer, thickness change (%)"]
                + bold({m: "%+.0f" % v for m, v in lay.items()}, winners(lay, -1)))
    secs = {m: float(np.mean([r["runs"][m]["seconds"] for r in rows])) for m, _ in METHODS}
    body.append(["cost", "seconds per run, mean"] + bold({m: "%.2f" % v for m, v in secs.items()}, winners(secs, -1)))
    return ("The shape and size rows count the inputs (of %d) on which each method is best; the others are one input's "
            "reading. Best in bold; readings within %g %% of each other tie, and a tie counts for each method.\n\n"
            % (n, 100 * TIE) + md_table(["", "reading"] + [lab for _, lab in METHODS], body))


def names(keys: list[str], rows: list[dict[str, Any]]) -> str:
    lab = {r["input"].key: r["input"].label for r in rows}
    ls = [lab[k] for k in keys]
    return "none" if not ls else ls[0] if len(ls) == 1 else ", ".join(ls[:-1]) + " and " + ls[-1]


def valid_keys(rows: list[dict[str, Any]]) -> list[str]:
    """The inputs every method leaves with no inverted quad."""
    return [r["input"].key for r in rows if all(r["runs"][m]["reading"]["inverted"] == 0 for m, _ in METHODS)]


UNGRADED = ("annulus", "sphere")                  # valid, ungraded inputs where Winslow stands in for the optimizer (the
                                                  # tangent-boundary disk is not one: its boundary moves, so the optimizer's shape is not Winslow's target)


def winslow_vs_optimizer(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """On the valid ungraded inputs: the largest gap in mean phi between Winslow and the optimizer, and Winslow's total
    time as a fraction of the optimizer's."""
    by = by_key(rows)
    gap = max(abs(by[i]["winslow"]["reading"]["mean phi"] - by[i]["optimizer"]["reading"]["mean phi"]) for i in UNGRADED)
    frac = sum(by[i]["winslow"]["seconds"] for i in UNGRADED) / sum(by[i]["optimizer"]["seconds"] for i in UNGRADED)
    return float(gap), float(frac)


def shows_text(rows: list[dict[str, Any]]) -> str:
    """What the comparison shows, one short line per characteristic; every number and every list of inputs is read
    from the runs (the qualitative words are re-checked by ``check_claims``)."""
    by = by_key(rows)
    rd = lambda i, m, k: by[i][m]["reading"][k]  # noqa: E731
    start = {r["input"].key: r["start"] for r in rows}
    valid = valid_keys(rows)
    lap_j = [i for i in valid if rd(i, "laplace", "min scaled J") > rd(i, "optimizer", "min scaled J")]
    opt_j = [i for i in valid if i not in lap_j]
    gap, frac = winslow_vs_optimizer(rows)
    cv = {m: [r["input"].key for r in rows if m in best_of(r, "CV of area")] for m, _ in METHODS}
    rep = {m: rd("tangled", m, "cases repaired") for m, _ in METHODS}
    stall = by["tangled"]["optimizer"]["logs"][0]
    wl = by["layer"]["winslow"]["logs"][0]
    st = per_step(rows)
    lines = [
        "- **Shape.** On the %d inputs every method leaves valid, the optimizer (alone or combined) has the lowest mean "
        "phi, the quantity it minimises. Winslow comes within %.3f of it on the %s, in %.0f %% of its time. By min "
        "scaled J, Laplace wins on the %s, the optimizer on the %s: the scaled J reads angles only."
        % (len(valid), gap, names(list(UNGRADED), rows), 100 * frac, names(lap_j, rows), names(opt_j, rows)),
        "- **Concave outline.** The L-shape's Coons patch starts with %d inverted quads. Laplace leaves %d, the "
        "optimizer alone %d, the combined schedule %d; Winslow leaves %d, %s at the reflex corner, where two quads "
        "share a 270-degree turn."
        % (start["lshape"]["inverted"], rd("lshape", "laplace", "inverted"), rd("lshape", "optimizer", "inverted"),
           rd("lshape", "combined", "inverted"), rd("lshape", "winslow", "inverted"),
           "only" if rd("lshape", "winslow", REFLEX_KEY) == 0 else "not only"),
        "- **Size.** No method evens element size everywhere: Laplace has the lowest spread of area (CV) on the %s, "
        "another method on the %s."
        % (names(cv["laplace"], rows), names([r["input"].key for r in rows if r["input"].key not in cv["laplace"]], rows)),
        "- **Tangled squares.** Of %d tangled squares, Laplace repaired %d, Winslow %d, the optimizer alone %d (at the "
        "default `delta` it stops after %d iteration), the combined schedule %d."
        % (rep["laplace"][1], rep["laplace"][0], rep["winslow"][0], rep["optimizer"][0], stall.passes_done,
           rep["combined"][0]),
        "- **Free boundary.** The disk has `boundary=\"tangent\"`: a boundary vertex slides along the boundary and never "
        "moves inward, so the area is kept (Laplace %s %%, Winslow %s %%, optimizer %s %%). The optimizer works there "
        "too: it runs %d iterations to min scaled J %.3f, against a frozen boundary's %.3f and `boundary=\"slide\"`'s "
        "%.3f. With `boundary=\"free\"` Laplace pulls the boundary onto its chords (%s %%) "
        "and the optimizer lets the rim cells grow (%s %%). Geometry you know goes on the boundary as `OnCurve` "
        "({doc}`constraints`)."
        % (fmt("area change (%)", rd("disk", "laplace", "area change (%)")),
           fmt("area change (%)", rd("disk", "winslow", "area change (%)")),
           fmt("area change (%)", rd("disk", "optimizer", "area change (%)")),
           sum(lg.passes_done for lg in by["disk"]["optimizer"]["logs"]), rd("disk", "optimizer", "min scaled J"),
           disk_optimizer("frozen")[1], disk_optimizer("slide")[1],
           fmt("area change (%)", disk_free_area(rows, "laplace")),
           fmt("area change (%)", disk_free_area(rows, "optimizer"))),
        "- **Grading.** Every method thickens the graded layer: Laplace %+.0f %%, Winslow %+.0f %% (its guard stopped "
        "it after %d sweeps), the optimizer %+.0f %%; see {doc}`grading`."
        % (rd("layer", "laplace", LAYER_KEY), rd("layer", "winslow", LAYER_KEY), wl.passes_done,
           rd("layer", "optimizer", LAYER_KEY)),
        "- **Cost.** A Laplace pass costs %.1f ms, a Winslow sweep %.1f ms, an optimizer iteration %.1f ms with its line "
        "search. Medians over the inputs; single runs on %s."
        % (1e3 * st["laplace"], 1e3 * st["winslow"], 1e3 * st["optimizer"], MACHINE),
    ]
    return "\n".join(lines) + "\n"


def disk_free_area(rows: list[dict[str, Any]], method: str) -> float:
    """The free disk's area change (%) under the comparison's Laplace stage or optimizer, with ``boundary="free"``."""
    inp = next(r["input"] for r in rows if r["input"].key == "disk")
    res = cn.resolve(inp.mesh, [], boundary="free")
    if method == "laplace":
        out, _ = smooth.laplace(inp.mesh, res, normals=inp.nrm, **LAPLACE[0][1])
    else:
        out, _ = smooth.run_schedule(inp.mesh, res, OPTIMIZER, normals=inp.nrm)
    return float(measure(inp, np.asarray(out.points), None)["area change (%)"])


# ============================================================================================================
# the Winslow page: the concave input, Laplace against Winslow, and a short table
# ============================================================================================================
def winslow_figure(rows: list[dict[str, Any]]) -> str:
    row = next(r for r in rows if r["input"].key == "lshape")
    inp = row["input"]
    states = [("input", np.asarray(inp.mesh.points), caption(inp, row["start"]))]
    for key in ("laplace", "winslow"):
        r = row["runs"][key]
        states.append((SHORT[key], np.asarray(r["mesh"].points), caption(inp, r["reading"])))
    legend = LEGEND.replace('<span class="mm-ly"></span><span>outer edge of a designed layer</span>', "")
    return raw('<div class="mm-grid">' + legend + multiples_row(inp.label, View("planar"), inp.mesh, inp.nrm, states, True)
               + "</div>")


def winslow_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Laplace and Winslow at the comparison's budgets, read from its runs, plus the graded layer with the guard
    relaxed (what a converged Winslow does to grading) and with ``KeepLayer``."""
    by = by_key(rows)
    start = {r["input"].key: r["start"] for r in rows}
    out = [
        {"label": "L-shape (concave)", "reading": "inverted quads (input %d)" % start["lshape"]["inverted"],
         **{m: "%d" % by["lshape"][m]["reading"]["inverted"] for m in ("laplace", "winslow")}},
        {"label": "L-shape (concave)", "reading": "of which away from the reflex corner",
         **{m: "%d" % by["lshape"][m]["reading"][REFLEX_KEY] for m in ("laplace", "winslow")}},
        {"label": "free disk", "reading": "area change (%)",
         **{m: fmt("area change (%)", by["disk"][m]["reading"]["area change (%)"]) for m in ("laplace", "winslow")}},
        {"label": "graded layer, frozen wall", "reading": "layer thickness change (%)",
         **{m: "%+.0f" % by["layer"][m]["reading"][LAYER_KEY] for m in ("laplace", "winslow")}},
    ]
    relaxed = 1.0
    inp = strip(keep=False)
    lap, _ = smooth.laplace(inp.mesh, inp.res, normals=inp.nrm, max_sj_drop=relaxed, **LAPLACE[0][1])
    win, wlog = smooth.winslow(inp.mesh, inp.res, normals=inp.nrm, max_sj_drop=relaxed, **WINSLOW[0][1])
    out.append({"label": "graded layer, `max_sj_drop=%g`" % relaxed, "reading": "layer thickness change (%)",
                "laplace": "%+.0f" % measure(inp, np.asarray(lap.points), None)[LAYER_KEY],
                "winslow": "%+.0f" % measure(inp, np.asarray(win.points), None)[LAYER_KEY], "sweeps": wlog.passes_done})
    inp = strip(keep=True)
    row = {"label": "graded layer, `KeepLayer(rings=%d)`" % (len(LAYER_ROWS) - 1), "reading": "layer thickness change (%)"}
    for key, (m, _) in {"laplace": smooth.laplace(inp.mesh, inp.res, normals=inp.nrm, **LAPLACE[0][1]),
                        "winslow": smooth.winslow(inp.mesh, inp.res, normals=inp.nrm, **WINSLOW[0][1])}.items():
        row[key] = "%+.0f" % measure(inp, np.asarray(m.points), None)[LAYER_KEY]
    out.append(row)
    return out


def winslow_table(runs: list[dict[str, Any]]) -> str:
    return md_table(["input", "reading", "Laplace (%d passes)" % LAPLACE_PASSES, "Winslow (%d sweeps)" % WINSLOW_SWEEPS],
                    [[r["label"], r["reading"], r["laplace"], r["winslow"]] for r in runs])


def check_winslow(runs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    """The statements the Winslow and grading pages make in words, re-checked against the table and the comparison."""
    r = {i: run for i, run in enumerate(runs)}
    bad = []
    if not float(r[4]["winslow"]) > max(row["reading"][LAYER_KEY] for row in by_key(rows)["layer"].values()):
        bad.append("with the guard relaxed, Winslow no longer thickens the layer more than every method of the comparison")
    if not (int(r[0]["winslow"]) < int(r[0]["laplace"]) and int(r[1]["winslow"]) == 0 < int(r[1]["laplace"])):
        bad.append("Winslow no longer leaves fewer inverted quads than Laplace on the L-shape, none away from the corner")
    if not abs(float(r[2]["winslow"])) < 0.1:
        bad.append("Winslow (tangent rule) no longer keeps the free disk's area")
    if not (float(r[3]["winslow"]) > 10 and float(r[3]["laplace"]) > 10):
        bad.append("an unprotected graded layer no longer thickens under both Laplace and Winslow")
    if not float(r[4]["winslow"]) > 2 * max(float(r[4]["laplace"]), float(r[3]["winslow"])):
        bad.append("with the guard relaxed, Winslow no longer erases the grading far more than Laplace")
    if not abs(float(r[5]["winslow"])) < 1.0:
        bad.append("KeepLayer no longer keeps the graded layer under Winslow")
    return bad


def input_tables(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        body = []
        keys = [k for k, _, _ in CORE] + [k for k in (LAYER_KEY, REFLEX_KEY) if k in row["start"]]
        for key in keys:
            vals = {m: row["runs"][m]["reading"][key] for m, _ in METHODS}
            best = best_of(row, key) if all(v is not None for v in vals.values()) else []
            cells = ["**%s**" % fmt(key, vals[m]) if m in best and len(best) < len(METHODS) else fmt(key, vals[m])
                     for m, _ in METHODS]
            body.append([key, fmt(key, row["start"][key])] + cells)
        if "cases repaired" in row["runs"]["laplace"]["reading"]:
            vals2 = {m: row["runs"][m]["reading"]["cases repaired"][0] for m, _ in METHODS}
            tot = row["runs"]["laplace"]["reading"]["cases repaired"][1]
            body.append(["tangled squares repaired (of %d)" % tot, "0"] + ["%d" % vals2[m] for m, _ in METHODS])
        out.append("**%s**\n\n%s\n" % (row["input"].label, md_table(["reading", "input"] + [lab for _, lab in METHODS], body)))
    return "".join(out)


def trade_table(rows: list[dict[str, Any]], wruns: list[dict[str, Any]]) -> str:
    st = per_step(rows)
    by = by_key(rows)
    left = {m: by["lshape"][m]["reading"]["inverted"] for m, _ in METHODS}
    _, frac = winslow_vs_optimizer(rows)
    body = [
        ["Laplace only", "cheap; never adds an inverted quad; best min scaled J on simple planar patches; repaired "
         "every tangled square here", "blind to phi; folds stay on a concave outline (%d left on the L-shape); relaxes "
         "grading" % left["laplace"], "%.1f ms a pass" % (1e3 * st["laplace"]), "a cheap first pass, or a valid result is "
         "enough"],
        ["Winslow only", "a concave outline: only the reflex-corner quad stays folded (%d left on the L-shape); the "
         "optimizer's shape on valid, ungraded input in %.0f %% of its time; repaired every tangled square"
         % (left["winslow"], 100 * frac),
         "erases grading once its guard lets it run (%s %% with `max_sj_drop=1`; pair with `KeepLayer`); can drift "
         "on a closed surface over many sweeps" % wruns[4]["winslow"],
         "%.1f ms a sweep" % (1e3 * st["winslow"]), "Laplace fails: a concave outline, a thin curved block ({doc}`winslow`)"],
        ["Optimizer only", "lowest phi; weights and constraints act inside it", "stalls on a tangle at the default "
         "`delta`; grows a `\"free\"` boundary; undoes grading", "%.1f ms an iteration" % (1e3 * st["optimizer"]),
         "a valid input whose shape is the goal"],
        ["Combined", "optimizer passes with a wide-to-narrow `delta` first, then its shape", "the optimizer's cost, grading and boundary "
         "behaviour", "%.2f s a run (mean)" % float(np.mean([r["runs"]["combined"]["seconds"] for r in rows])),
         "the default; any damaged or unknown input"],
    ]
    return md_table(["method", "good at", "weak at", "cost", "use when"], body)


def check_claims(rows: list[dict[str, Any]]) -> list[str]:
    """The qualitative statements the pages make in words, re-checked against the runs; returns what no longer holds."""
    by = by_key(rows)
    rd = lambda i, m, k: by[i][m]["reading"][k]  # noqa: E731
    start = {r["input"].key: r["start"] for r in rows}
    valid = valid_keys(rows)
    bad = []
    if sorted(valid) != sorted(("annulus", "disk", "sphere", "layer")):
        bad.append("the inputs every method leaves valid are no longer the annulus, disk, sphere and layer: %s" % valid)
    if not all(set(best_of(r, "mean phi")) & {"optimizer", "combined"} for r in rows if r["input"].key in valid):
        bad.append("the optimizer (alone or combined) is no longer best on mean phi on every valid input")
    if not all(rd(i, "laplace", "mean phi") > rd(i, "optimizer", "mean phi") for i in valid):
        bad.append("Laplace is no longer worse than the optimizer on mean phi on every valid input")
    gap, frac = winslow_vs_optimizer(rows)
    if not (gap < 0.01 and frac < 1.0):
        bad.append("Winslow is no longer close to the optimizer's shape, or no longer faster, on the valid ungraded inputs")
    if not (rd("annulus", "laplace", "min scaled J") > rd("annulus", "optimizer", "min scaled J")
            and rd("tangled", "laplace", "min scaled J") > 0.9):
        bad.append("Laplace no longer has the best min scaled J on the simple planar patches")
    for m in ("laplace", "winslow"):
        if rd("tangled", m, "cases repaired")[0] != rd("tangled", m, "cases repaired")[1]:
            bad.append("%s no longer repairs every tangled square" % m)
        if not all(rd(i, m, "inverted") <= start[i]["inverted"] for i in by):
            bad.append("%s added an inverted quad (the pages say it never does)" % m)
    if rd("tangled", "optimizer", "cases repaired")[0] >= rd("tangled", "combined", "cases repaired")[0]:
        bad.append("the optimizer alone repairs as many tangles as the combined schedule")
    lsh = {m: rd("lshape", m, "inverted") for m, _ in METHODS}
    if not (lsh["winslow"] < min(v for m, v in lsh.items() if m != "winslow") and rd("lshape", "winslow", REFLEX_KEY) == 0
            < rd("lshape", "laplace", REFLEX_KEY) and lsh["laplace"] > 0):
        bad.append("Winslow no longer leaves the fewest inverted quads on the L-shape, only at the reflex corner")
    if not all(rd("lshape", m, "inverted") - rd("lshape", m, REFLEX_KEY) >= 1 for m, _ in METHODS):
        bad.append("a method now repairs the L-shape's reflex-corner quad (the Winslow page says none does)")
    if not max(abs(rd("disk", m, "area change (%)")) for m in ("laplace", "winslow", "optimizer")) < 1.0:
        bad.append("a method no longer keeps the tangent-boundary disk's area")
    if not (disk_free_area(rows, "laplace") < -1.0 and disk_free_area(rows, "optimizer") > 1.0):
        bad.append("with boundary='free', Laplace no longer shrinks / the optimizer no longer grows the free disk")
    if not all(rd("layer", m, LAYER_KEY) > 10 for m, _ in METHODS):
        bad.append("an unprotected graded layer no longer thickens under every method")
    if not (rd("layer", "laplace", LAYER_KEY) < min(rd("layer", m, LAYER_KEY) for m in ("optimizer", "combined"))):
        bad.append("Laplace no longer thickens the layer less than the optimizer")
    if by["layer"]["winslow"]["logs"][0].stopped != "rolled_back":
        bad.append("Winslow's guard no longer stops it on the graded layer")
    if not all(rd(i, "combined", "min scaled J") >= rd(i, "optimizer", "min scaled J") - 1e-3 for i in valid):
        bad.append("the combined schedule no longer reaches the optimizer's min scaled J on the valid inputs")
    st = per_step(rows)
    if not LAPLACE_PASSES * st["laplace"] < OPTIMIZER_ITERS * st["optimizer"]:
        bad.append("Laplace only is no longer cheaper than the optimizer's budget")
    return bad


# ============================================================================================================
# the Laplace diagram: one pass on a tiny patch, step by step
# ============================================================================================================
ARC_C, ARC_R = np.array([0.0, -1.2, 0.0]), 4.0


def laplace_patch() -> tuple[Any, Any, int]:
    """A 3 x 3 patch: bottom and sides frozen (the default for an unclaimed boundary), the two middle top vertices
    ``OnCurve`` an arc.  The bottom and top rows are unevenly spaced and one interior vertex is displaced, so every step
    moves something.  Returns the mesh, its constraints and the vertex the figure follows."""
    nx = ny = 3
    xs = np.linspace(-1.5, 1.5, nx + 1)
    xb = xs + np.array([0.0, 0.25, 0.2, 0.0])                       # the bottom row: it slides, then is put back
    idx, quads = grid_quads(nx, ny)
    top_ang = np.arcsin(xs / ARC_R) + np.array([0.0, -0.1, -0.06, 0.0])
    P = np.zeros(((nx + 1) * (ny + 1), 3))
    for i in range(nx + 1):
        top = ARC_C[:2] + ARC_R * np.array([np.sin(top_ang[i]), np.cos(top_ang[i])])
        for j in range(ny + 1):
            t = j / ny
            P[idx[i, j], :2] = (1 - t) * np.array([xb[i], 0.0]) + t * top
    for (i, j), d in (((1, 1), (0.3, 0.18)), ((2, 1), (-0.06, 0.08)), ((1, 2), (0.08, -0.08)), ((2, 2), (-0.1, -0.04))):
        P[idx[i, j], :2] += d
    mesh = quadmesh.QuadMesh.from_corners(P, quads)
    res = cn.resolve(mesh, [cn.OnCurve(idx[1:-1, ny], cn.circle(ARC_C, ARC_R))])
    return mesh, res, int(idx[1, 1])


def laplace_steps() -> str:
    """The three steps of one pass, computed with the functions ``smooth.laplace`` calls (``_shrunk_corners``,
    ``_scatter``, ``constrain.snap``) and checked against ``smooth.laplace(passes=1)`` itself.  The snap moves are a
    fraction of an edge, so step (c) adds two magnified insets: the on-curve and the frozen vertex that move most."""
    mesh, res, v = laplace_patch()
    corners = mesh.corners
    P0 = np.asarray(mesh.points, float)
    bnd = smooth.boundary_edge_ids(mesh)[mesh.quads]
    mult = np.bincount(corners.ravel(), minlength=mesh.n_points).astype(float)
    sh = smooth._shrunk_corners(P0, corners, bnd, DIAGRAM_W)                    # (a)
    avg = metric.scatter(mesh.n_points, corners, sh) / mult[:, None]           # (b)
    snapped = cn.snap(avg, res, 1.0)                                            # (c): passes=1, so the full snap
    ref, log = smooth.laplace(mesh, res, passes=1, w=DIAGRAM_W)
    if log.passes_done != 1 or np.abs(np.asarray(ref.points) - snapped).max() > 1e-12:
        raise RuntimeError("the Laplace diagram no longer reproduces smooth.laplace(passes=1)")
    oncurve = np.flatnonzero(res.kind == cn.ON_CURVE)
    frozen = np.flatnonzero(res.kind == cn.FROZEN)
    snap_move = np.linalg.norm(snapped - avg, axis=1)
    zoom_v = [int(oncurve[np.argmax(snap_move[oncurve])]), int(frozen[np.argmax(snap_move[frozen])])]

    allp = np.concatenate([P0, avg, snapped, sh.reshape(-1, 3)])
    x0, x1 = allp[:, 0].min() - 0.2, allp[:, 0].max() + 0.2
    y0, y1 = allp[:, 1].min() - 0.2, allp[:, 1].max() + 0.25
    s = 240.0 / (x1 - x0)
    W, Hm = 240, int(round(s * (y1 - y0)))

    def main_map(p: Any) -> tuple[float, float]:
        return (p[0] - x0) * s, (y1 - p[1]) * s

    def draw(X: Any) -> dict[str, Any]:
        """The drawing primitives under one map from model to SVG coordinates."""
        def poly(pts: Any, cls: str) -> str:
            return '<path class="%s" d="M%s Z"/>' % (cls, " L".join("%.1f %.1f" % X(p) for p in pts))

        def mesh_(Q: Any, cls: str) -> str:
            return "".join(poly(Q[c], cls) for c in corners)

        def arrow(a: Any, b: Any, cls: str, fig: str) -> str:
            (ax, ay), (bx, by) = X(a), X(b)
            return '<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" marker-end="url(#lp-%s-%s)"/>' % (
                cls, ax, ay, bx, by, fig, cls)

        def dot(p: Any, cls: str, r: float = 2.6) -> str:
            return '<circle class="%s" cx="%.1f" cy="%.1f" r="%.1f"/>' % ((cls,) + X(p) + (r,))

        def arc() -> str:
            t = np.linspace(-0.6, 0.6, 241)
            pts = np.stack([ARC_C[0] + ARC_R * np.sin(t), ARC_C[1] + ARC_R * np.cos(t)], axis=1)
            return '<path class="tgt" d="M%s"/>' % " L".join("%.1f %.1f" % X(p) for p in pts)

        return {"poly": poly, "mesh": mesh_, "arrow": arrow, "dot": dot, "arc": arc}

    def defs(fig: str) -> str:
        return "<defs>" + "".join(
            '<marker id="lp-%s-%s" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
            'orient="auto"><path class="h-%s" d="M0,0 L7,3.5 L0,7 z"/></marker>' % (fig, c, c) for c in ("mv", "sn")) + "</defs>"

    def svg(fig: str, height: int, body: str) -> str:
        return ('<svg viewBox="0 0 %d %d" class="lp-fig" role="img" aria-label="Laplace pass, step %s">%s%s</svg>'
                % (W, height, fig, defs(fig), body))

    d = draw(main_map)
    copies_v = [sh[q, k] for q, k in zip(*np.nonzero(corners == v))]
    a = svg("a", Hm, d["mesh"](P0, "old") + "".join(d["poly"](sh[q], "shr") for q in range(len(corners)))
            + "".join(d["arrow"](P0[v], c, "mv", "a") for c in copies_v) + d["dot"](P0[v], "vx"))
    b = svg("b", Hm, d["mesh"](P0, "ghost") + d["mesh"](avg, "new")
            + "".join(d["dot"](c, "cp", 1.5) for c in sh.reshape(-1, 3))
            + "".join(d["dot"](c, "cpv", 2.4) for c in copies_v) + d["arrow"](P0[v], avg[v], "mv", "b")
            + d["dot"](avg[v], "vx"))
    # step (c): the snap, plus one magnified inset per followed vertex below the patch
    inset_w, inset_h, gap = 112, 92, 6
    body = d["arc"]() + d["mesh"](avg, "ghost") + d["mesh"](snapped, "new")
    body += "".join(d["dot"](snapped[i], "oc") for i in oncurve) + "".join(d["dot"](snapped[i], "fz", 2.2) for i in frozen)
    mags = []
    for k, u in enumerate(zoom_v):
        ox, oy = 8 + k * (inset_w + gap), Hm + 4
        mag = 46.0 / max(float(snap_move[u]) * s, 1e-9)               # the move drawn about 46 px long
        mid = 0.5 * (avg[u] + snapped[u])
        cx, cy = ox + inset_w / 2, oy + inset_h / 2 + 4

        def zmap(p: Any, mid: Any = mid, mag: float = mag, cx: float = cx, cy: float = cy) -> tuple[float, float]:
            return cx + (p[0] - mid[0]) * s * mag, cy - (p[1] - mid[1]) * s * mag

        z = draw(zmap)
        clip = "lp-clip-%d" % k
        body += ('<clipPath id="%s"><rect x="%d" y="%d" width="%d" height="%d"/></clipPath>' % (clip, ox, oy, inset_w, inset_h)
                 + '<rect class="ins" x="%d" y="%d" width="%d" height="%d"/>' % (ox, oy, inset_w, inset_h)
                 + '<g clip-path="url(#%s)">' % clip + (z["arc"]() if res.kind[u] == cn.ON_CURVE else "")
                 + z["mesh"](avg, "ghost") + z["mesh"](snapped, "new") + z["dot"](avg[u], "hol", 3)
                 + z["arrow"](avg[u], snapped[u], "sn", "c") + z["dot"](snapped[u], "oc" if res.kind[u] == cn.ON_CURVE else "fz", 3)
                 + "</g>"
                 + '<text class="lbl" x="%d" y="%d">%d: %s, ×%.0f</text>' % (ox + 4, oy + 11, k + 1,
                                                                            "on curve" if k == 0 else "frozen", mag)
                 + '<circle class="ring" cx="%.1f" cy="%.1f" r="7"/>' % main_map(snapped[u])
                 + '<text class="lbl" x="%.1f" y="%.1f">%d</text>' % (main_map(snapped[u])[0] + 7, main_map(snapped[u])[1] + (12 if k else -6), k + 1))
        mags.append(mag)
    c = svg("c", Hm + inset_h + 8, body)
    cap_a = "(a) shrink: each quad's corners are re-read at ±w, toward its own centre; a boundary side stays at ±1"
    cap_b = "(b) average: each vertex goes to the mean of its shrunk corner copies (four for the one followed)"
    cap_c = "(c) snap: a vertex on a curve is projected onto it, a frozen one returns to its anchor (magnified below)"
    figs = "".join('<figure>%s<figcaption>%s</figcaption></figure>' % (g, t) for g, t in ((a, cap_a), (b, cap_b), (c, cap_c)))
    ell = cn.vertex_length(P0, corners)
    txt = ("One pass on a 3 × 3 patch with w = %s (the default is %s; a smaller w is drawn so the moves show). The bottom "
           "and sides are frozen, the two middle top vertices are on the arc (red). The result is identical to "
           "`laplace(passes=1, w=%s)`. The followed vertex moves %.2f of an edge in (b); the snap moves are smaller, "
           "%.3f and %.3f of an edge in the insets."
           % (num(DIAGRAM_W), num(default(smooth.laplace, "w")), num(DIAGRAM_W), float(np.linalg.norm(avg[v] - P0[v]) / ell[v]),
              float(snap_move[zoom_v[0]] / ell[zoom_v[0]]), float(snap_move[zoom_v[1]] / ell[zoom_v[1]])))
    return raw('<div class="lp-steps">' + figs + "</div>") + "\n" + txt + "\n"


# ============================================================================================================
# the Laplace page: the free-boundary rules side by side
# ============================================================================================================
def uneven_disk() -> tuple[Input, Any]:
    """The comparison's free disk with its boundary vertices slid along the circle by up to 0.3 of their spacing (on an
    evenly spaced circle the free move has no tangential part, so the rules would not differ).  Returns the input and
    its boundary vertex ids."""
    inp = disk()
    P = np.array(inp.mesh.points)
    bnd = np.flatnonzero(np.isclose(np.hypot(P[:, 0], P[:, 1]), 1.0))
    th = np.arctan2(P[bnd, 1], P[bnd, 0]) + np.random.default_rng(5).uniform(-0.3, 0.3, bnd.size) * 2 * np.pi / bnd.size
    P[bnd, 0], P[bnd, 1] = np.cos(th), np.sin(th)
    mesh = smooth.with_points(inp.mesh, P)
    return Input("uneven", "free disk, uneven boundary", "planar", mesh, cn.resolve(mesh, [], boundary="tangent"),
                 inp.nrm), bnd


def laplace_boundary() -> tuple[str, list[str]]:
    """The uneven disk after the comparison's Laplace stage under the three boundary treatments; returns the figure with
    its caption, and the statements that no longer hold."""
    inp, bnd = uneven_disk()
    P0 = np.asarray(inp.mesh.points)
    ell = float(cn.vertex_length(P0, inp.mesh.corners)[bnd].mean())
    runs = {b: (cn.resolve(inp.mesh, [], boundary=b), {}) for b in ("free", "tangent", "slide")}
    states = [("input", P0, "min J %.2f" % inp.q0.sj_min)]
    got = {}
    for key, (res, kw) in runs.items():
        out, _ = smooth.laplace(inp.mesh, res, normals=inp.nrm, **LAPLACE[0][1], **kw)
        X = np.asarray(out.points)
        area = measure(inp, X, None)["area change (%)"]
        off = float(np.abs(np.hypot(X[bnd, 0], X[bnd, 1]) - 1.0).max() / ell)
        got[key] = (area, off)
        states.append((key, X, "area %s %% · off circle %s h" % (fmt("area change (%)", area),
                                                                  "%.1f" % off if off >= 0.1 else "%.2g" % off)))
    legend = LEGEND.replace('<span class="mm-ly"></span><span>outer edge of a designed layer</span>', "")
    fig = raw('<div class="mm-grid">' + legend
              + multiples_row("free disk, uneven boundary", View("planar"), inp.mesh, inp.nrm, states, True) + "</div>")
    cap = ("The comparison's free disk, its boundary vertices slid along the circle by up to 0.3 of their spacing, after "
           "`%s`. **free**: `boundary=\"free\"`; **tangent**: `boundary=\"tangent\"`; **slide**: `boundary=\"slide\"`. *Off circle*: the largest distance of a boundary vertex from "
           "the circle, in mean boundary edges h." % call("laplace", LAPLACE[0][1]))
    bad = []
    if not (got["free"][0] < -1.0 and abs(got["tangent"][0]) < 1.0 and abs(got["slide"][0]) < 1.0):
        bad.append("on the uneven disk, free no longer shrinks or tangent / slide no longer keep the area")
    if not (got["slide"][1] < 0.1 and got["tangent"][1] < 0.1 < got["free"][1]):
        bad.append("on the uneven disk, tangent and slide no longer both stay within 0.1 h of the circle where free does not")
    return fig + "\n" + cap + "\n", bad


# ============================================================================================================
# the constraints page: boundary="slide" against "free"
# ============================================================================================================
_DISK_OPT: dict[str, tuple[Any, float, float, int]] = {}


def disk_optimizer(boundary: str) -> tuple[Any, float, float, int]:
    """The comparison's free disk under the optimizer schedule with ``boundary=...`` and nothing else named: the final
    points, min scaled J, area change (%) and the optimizer's iterations.  Cached: the figure and the comparison text
    read the same run."""
    if boundary not in _DISK_OPT:
        inp = disk()
        res = cn.resolve(inp.mesh, [], boundary=boundary)
        out, logs = smooth.run_schedule(inp.mesh, res, OPTIMIZER, normals=inp.nrm)
        X = np.asarray(out.points)
        r = measure(inp, X, None)
        _DISK_OPT[boundary] = (X, float(r["min scaled J"]), float(r["area change (%)"]), sum(lg.passes_done for lg in logs))
    return _DISK_OPT[boundary]


def slide_figure() -> tuple[str, list[str]]:
    """The jittered annulus and the free disk under the comparison's optimizer stage with ``boundary`` ``"free"``,
    ``"tangent"`` and ``"slide"``; frozen vertices are dotted.  Returns the figure with its caption, and the statements
    that no longer hold."""
    legend = (LEGEND.replace('<span class="mm-ly"></span><span>outer edge of a designed layer</span>',
                             '<span class="mm-fz"></span><span>frozen corner</span>'))
    rows_html, got, froz, counts, iters = [], {}, {}, {}, {}
    for title, make in (("annulus", annulus), ("free disk", disk)):
        inp = make()
        P0 = np.asarray(inp.mesh.points)
        states = [("input", P0, "min J %.3f" % inp.q0.sj_min)]
        dots = {}
        for i, b in enumerate(("free", "tangent", "slide"), start=1):
            res = cn.resolve(inp.mesh, [], boundary=b)
            if title == "free disk":
                X, mj, area, n_it = disk_optimizer(b)
            else:
                out, logs = smooth.run_schedule(inp.mesh, res, OPTIMIZER, normals=inp.nrm)
                X = np.asarray(out.points)
                r = measure(inp, X, None)
                mj, area, n_it = float(r["min scaled J"]), float(r["area change (%)"]), sum(lg.passes_done for lg in logs)
            got[title, b] = (mj, area)
            iters[title, b] = n_it
            counts[title, b] = res.counts()
            froz[title, b] = int((res.kind == cn.FROZEN).sum())
            if froz[title, b]:
                dots[i] = np.flatnonzero(res.kind == cn.FROZEN)
            states.append(('boundary="%s"' % b, X, "min J %.3f · area %s %%" % (mj, fmt("area change (%)", area))))
        rows_html.append(multiples_row(title, View("planar"), inp.mesh, inp.nrm, states, True, dots=dots))
    fig = raw('<div class="mm-grid">' + legend + "".join(rows_html) + "</div>")
    ca, cd = counts["annulus", "slide"], counts["free disk", "slide"]
    frozen_j = disk_optimizer("frozen")[1]
    cap = ("Both inputs of the comparison under `%s`, no constraint named. **Annulus**: its %d boundary vertices move along "
           "the boundary and its %d corners are frozen (dots); the %d interior vertices are free. **Free disk**: a circle "
           "has no corner, so all %d boundary vertices move. Under `\"tangent\"` the optimizer runs %d iterations "
           "to min scaled J %.3f, as under `\"frozen\"` (%.3f); under `\"slide\"` it reaches %.3f with the area kept."
           % (call("optimize", OPTIMIZER[0][1]), ca["on_curve"], ca["frozen"], ca["free"], cd["on_curve"],
              iters["free disk", "tangent"], got["free disk", "tangent"][0], frozen_j, got["free disk", "slide"][0]))
    bad = []
    for t in ("annulus", "free disk"):
        if not abs(got[t, "slide"][1]) < 0.1:
            bad.append("boundary='slide' no longer keeps the %s's area (to 0.1 %%)" % t)
        if not abs(got[t, "free"][1]) > 1.0:
            bad.append("boundary='free' no longer changes the %s's area by more than 1 %%" % t)
    if froz["annulus", "slide"] != 4:
        bad.append("boundary='slide' no longer freezes exactly the annulus's 4 corners")
    if froz["free disk", "slide"] or froz["free disk", "tangent"]:
        bad.append("the free disk now has a frozen vertex under slide or tangent")
    if not got["free disk", "tangent"][0] > got["free disk", "slide"][0] - 0.05:
        bad.append("the optimizer stalls on the tangent disk again: its min scaled J is no longer close to slide's")
    if not abs(got["free disk", "tangent"][0] - frozen_j) < 0.05:
        bad.append("the tangent disk's optimizer result is no longer about the frozen result: update the wording")
    return fig + "\n" + cap + "\n", bad


# ============================================================================================================
# the constraints page: how one boundary vertex moves, setting by setting
# ============================================================================================================
BM_DEG = (0.0, 22.0, 32.0, 78.0, 90.0)       # boundary angles of the patch's outer arc (deg): uneven, every turn < CORNER_DEG
BM_RADII = (0.45, 0.72, 1.0)
BM_W = 0.1                                   # a large Laplace step, so the moves are visible
BM_SCALE, BM_X, BM_Y = 700.0, (-0.15, 0.15), (-0.22, 0.08)       # px per unit; the window (along the tangent, along the normal)
BM_CSS = (
    ".bm{margin:1rem 0 1.4rem}"
    ".bm-layers{display:flex;flex-wrap:wrap;gap:.3rem .9rem;font-size:.82em;color:var(--color-foreground-secondary);margin-bottom:.7rem}"
    ".bm-layers b{color:var(--color-foreground-primary)}"
    ".bm-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7rem .6rem}"
    ".bm-cell{margin:0;min-width:0}"
    ".bm-cell svg{width:100%;height:auto;display:block;border:1px solid var(--color-background-border);border-radius:4px;"
    "background:var(--color-background-primary)}"
    ".bm-cell figcaption{font-size:.78em;line-height:1.3;text-align:center;color:var(--color-foreground-secondary);margin-top:.25rem}"
    ".bm-cell figcaption b{color:var(--color-foreground-primary)}"
    ".bm-chips{display:inline-flex;gap:2px;vertical-align:middle;margin-left:.3rem}"
    ".bm-chips i{font-style:normal;font-size:.8em;line-height:1.35;width:1.3em;text-align:center;border-radius:3px;"
    "border:1px solid var(--color-background-border);color:var(--color-foreground-muted)}"
    ".bm-chips i.on{background:var(--color-brand-primary);border-color:var(--color-brand-primary);color:var(--color-background-primary)}"
    ".bm-legend{display:flex;flex-wrap:wrap;gap:.25rem 1rem;font-size:.8em;color:var(--color-foreground-secondary);margin-top:.7rem}"
    ".bm-legend span{display:inline-flex;align-items:center;gap:.35rem}"
    ".bm svg .mesh{fill:none;stroke:var(--color-foreground-muted);stroke-width:1;stroke-opacity:.55}"
    ".bm svg .nb{fill:var(--color-foreground-muted)}"
    ".bm svg .curve{fill:none;stroke:var(--color-foreground-primary);stroke-width:2.2}"
    ".bm svg .chord{fill:none;stroke:var(--color-foreground-muted);stroke-width:1.5;stroke-dasharray:1 4;stroke-linecap:round}"
    ".bm svg .tan{fill:none;stroke:var(--color-foreground-muted);stroke-width:1.2;stroke-dasharray:6 4}"
    ".bm svg .raw{fill:none;stroke:#d64034;stroke-width:2.4}"
    ".bm svg .ghost{fill:none;stroke:#d64034;stroke-width:1.6;stroke-opacity:.4;stroke-dasharray:3 3}"
    ".bm svg .mv{fill:none;stroke:var(--color-brand-primary);stroke-width:2.6}"
    ".bm svg .pr{fill:none;stroke:#1f9d6b;stroke-width:2.6}"
    ".bm svg .rm{fill:none;stroke:var(--color-foreground-muted);stroke-width:1.2;stroke-dasharray:2 3}"
    ".bm svg .h-raw{fill:#d64034}.bm svg .h-mv{fill:var(--color-brand-primary)}.bm svg .h-pr{fill:#1f9d6b}"
    ".bm svg .cp{fill:var(--color-background-primary);stroke:#d64034;stroke-width:1.6}"
    ".bm svg .st{fill:var(--color-background-primary);stroke:var(--color-foreground-primary);stroke-width:1.6}"
    ".bm svg .fin{fill:var(--color-foreground-primary);stroke:var(--color-background-primary);stroke-width:1.5}"
    ".bm svg .lock rect{fill:var(--color-foreground-primary)}"
    ".bm svg .lock path{fill:none;stroke:var(--color-foreground-primary);stroke-width:2}"
    ".bm svg text{font:11px sans-serif;fill:var(--color-foreground-secondary);stroke:var(--color-background-primary);stroke-width:3px;paint-order:stroke}.bm svg text.lpr{fill:#1f9d6b}"
    "@media (max-width:40rem){.bm-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}"
)


def boundary_move_figure() -> tuple[str, list[str]]:
    """One boundary vertex of a tiny annular patch, every ``boundary`` setting from the same start.  The proposed move is
    the Laplace average (``_shrunk_corners`` with the boundary sides kept at +-1, then ``_scatter``); the settings then act
    with the library's own ``laplace`` rule (frozen, free, tangent), ``constrain_direction`` + ``snap`` (slide), and
    ``resolve`` (corner).  Returns the figure with its legend, and the statements that no longer hold."""
    ang = np.radians(BM_DEG)
    idx = np.arange(len(BM_RADII) * len(ang)).reshape(len(BM_RADII), len(ang))
    _, quads = grid_quads(len(BM_RADII) - 1, len(ang) - 1)
    P0 = np.array([[r * math.cos(a), r * math.sin(a), 0.0] for r in BM_RADII for a in ang])
    mesh = quadmesh.QuadMesh.from_corners(P0, quads)
    nrm = smooth.reference_normals(P0, quads)
    corners = mesh.corners
    v, cv = int(idx[-1, 2]), int(idx[-1, 0])              # the vertex followed, and a corner of the patch
    bnd = smooth.boundary_edge_ids(mesh)[mesh.quads]
    mult = np.bincount(corners.ravel(), minlength=mesh.n_points).astype(float)
    sh = smooth._shrunk_corners(P0, corners, bnd, BM_W)                          # layer 1: sides kept at +-1
    avg = metric.scatter(mesh.n_points, corners, sh) / mult[:, None]
    res = {b: cn.resolve(mesh, [], boundary=b) for b in ("frozen", "free", "tangent", "slide")}
    tangent_avg = P0 + np.where((res["tangent"].kind == cn.TANGENT)[:, None],
                                cn.constrain_direction(avg - P0, P0, res["tangent"]), avg - P0)   # layer 2
    curve = res["slide"].curves[int(res["slide"].target[v])]
    step = cn.constrain_direction(avg - P0, P0, res["slide"])                    # layer 2 for a curve ...
    slid = cn.snap(P0 + step, res["slide"], 1.0)                                 # ... and layer 3
    grad = smooth.gradient(P0, corners, nrm)[cv]                                  # the optimizer's proposal at the corner
    d_corner = -grad / np.linalg.norm(grad) * 0.5 * float(cn.vertex_length(P0, corners)[cv])
    got: dict[str, Any] = {"frozen": cn.snap(avg, res["frozen"], 1.0), "free": avg, "tangent": tangent_avg, "slide": slid}
    bad = []
    for key in ("frozen", "free", "tangent"):
        ref, log = smooth.laplace(mesh, res[key], passes=1, w=BM_W, normals=nrm, max_sj_drop=1.0)
        if log.passes_done != 1 or np.abs(np.asarray(ref.points)[v] - got[key][v]).max() > 1e-12:
            raise RuntimeError("the boundary-move figure no longer reproduces smooth.laplace(passes=1), %s" % key)
    radius = lambda p: float(np.hypot(p[0], p[1]))                               # noqa: E731  (the patch lies on a circle of 1)
    off = {k: radius(got[k][v]) - 1.0 for k in ("free", "tangent", "slide")}
    if not (off["free"] < -3 * abs(off["tangent"]) and abs(off["tangent"]) > 5 * abs(off["slide"]) and abs(off["slide"]) < 1e-2):
        bad.append("on the boundary patch, free / tangent / slide no longer sit progressively closer to the circle")
    on_curve = float(np.abs(curve.project(got["slide"][v][None, :])[0] - got["slide"][v]).max())
    turn = float(cn.turning_deg(P0, np.array([cv]), np.array([[idx[-1, 1], idx[-2, 0]]]))[0])
    if on_curve > 1e-9 or res["slide"].kind[v] != cn.ON_CURVE:
        bad.append("the slid vertex no longer lands exactly on its curve")
    if not (turn > cn.CORNER_DEG and cv in cn.boundary_corners(P0, cn.boundary_adjacency(mesh), cn.CORNER_DEG)
            and cv in np.flatnonzero(res["tangent"].kind == cn.FROZEN) and res["slide"].kind[cv] == cn.FROZEN
            and res["frozen"].kind[cv] == cn.FROZEN):
        bad.append("the patch corner is no longer frozen under tangent, slide and frozen")
    if float(cn.turning_deg(P0, np.array([v]), np.array([[v - 1, v + 1]]))[0]) >= cn.CORNER_DEG:
        bad.append("the followed vertex is itself a corner")

    def frame(o: int) -> Any:
        """Model -> SVG map: origin at vertex ``o``, x along the boundary tangent (counter-clockwise), y outward."""
        e1 = np.array([-P0[o, 1], P0[o, 0], 0.0]) / np.hypot(P0[o, 0], P0[o, 1])
        e2 = np.array([P0[o, 0], P0[o, 1], 0.0]) / np.hypot(P0[o, 0], P0[o, 1])
        if o == cv:                                      # the corner: the outside bisector of its two boundary sides up
            u = sum((P0[n] - P0[o]) / np.linalg.norm(P0[n] - P0[o]) for n in (idx[-1, 1], idx[-2, 0]))
            e2 = -u / np.linalg.norm(u)
            e1 = np.array([e2[1], -e2[0], 0.0])

        def X(p: Any) -> tuple[float, float]:
            d = np.asarray(p, float) - P0[o]
            return (-BM_X[0] * BM_SCALE + BM_SCALE * float(d @ e1), BM_Y[1] * BM_SCALE - BM_SCALE * float(d @ e2))
        return X, e1, e2

    W, H = int(round((BM_X[1] - BM_X[0]) * BM_SCALE)), int(round((BM_Y[1] - BM_Y[0]) * BM_SCALE))
    fig_id = [0]

    def panel(o: int, mode: str) -> str:
        fig_id[0] += 1
        k = fig_id[0]
        X, e1, e2 = frame(o)
        t = np.linspace(-0.9, 0.9, 181)
        th0 = math.atan2(P0[o, 1], P0[o, 0])
        circ = np.stack([np.cos(th0 + t), np.sin(th0 + t), np.zeros_like(t)], axis=1)

        def path(pts: Any, cls: str) -> str:
            return '<path class="%s" d="M%s"/>' % (cls, " L".join("%.1f %.1f" % X(p) for p in pts))

        def arrow(a: Any, b: Any, cls: str) -> str:
            (ax, ay), (bx, by) = X(a), X(b)
            return ('<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" marker-end="url(#bm%d-%s)"/>'
                    % (cls, ax, ay, bx, by, k, cls))

        def dot(p: Any, cls: str, r: float) -> str:
            return '<circle class="%s" cx="%.1f" cy="%.1f" r="%.1f"/>' % ((cls,) + X(p) + (r,))

        def line(p: Any, d: Any, cls: str) -> str:
            a, b = p - 0.6 * d, p + 0.6 * d
            return path([a, b], cls)

        nb = [n for n in (idx[-1, 1], idx[-2, 0]) if o == cv] or [v - 1, v + 1]
        body = "".join(path(P0[corners[q]][[0, 1, 2, 3, 0]], "mesh") for q in np.flatnonzero((corners == o).any(axis=1)))
        body += "".join(dot(P0[n], "nb", 3) for n in np.unique(corners[(corners == o).any(axis=1)]) if n != o)
        spline = curve.project(circ) if mode == "slide" else circ
        body += path(spline, "curve") + "".join(path([P0[o], P0[n]], "chord") for n in nb)
        tvec = None
        if mode == "proposed":
            body += "".join(dot(sh[q, kk], "cp", 3.4) for q, kk in zip(*np.nonzero(corners == o)))
            body += arrow(P0[o], avg[o], "raw")
        elif mode == "frozen":
            body += arrow(P0[o], avg[o], "ghost")
        elif mode == "free":
            body += arrow(P0[o], avg[o], "ghost") + arrow(P0[o], avg[o], "mv")
        elif mode == "tangent":
            tvec = cn.circle_tangent(P0[[v - 1]], P0[[o]], P0[[v + 1]])[0]
            body += line(P0[o], tvec, "tan") + arrow(P0[o], avg[o], "ghost")
            body += path([got["tangent"][o], avg[o]], "rm") + arrow(P0[o], got["tangent"][o], "mv")
            x, y = X(avg[o])
            body += '<text x="%.1f" y="%.1f" text-anchor="end">normal part</text>' % (x - 6, y + 4)
        elif mode == "slide":
            tvec = curve.tangent(P0[o][None, :])[0]
            body += line(P0[o], tvec, "tan") + arrow(P0[o], avg[o], "ghost")
            body += path([P0[o] + step[o], avg[o]], "rm") + arrow(P0[o], P0[o] + step[o], "mv")
            body += arrow(P0[o] + step[o], got["slide"][o], "pr")
            x, y = X(got["slide"][o])
            body += '<text class="lpr" x="%.1f" y="%.1f" text-anchor="end">projection</text>' % (x - 5, y + 16)
        elif mode == "corner":
            body += arrow(P0[o], P0[o] + d_corner, "raw")
        body += dot(P0[o], "st", 5)
        final = {"frozen": P0[o], "free": avg[o], "tangent": got["tangent"][o], "slide": got["slide"][o], "corner": P0[o]}
        if mode in final and mode not in ("frozen", "corner"):
            body += dot(final[mode], "fin", 4.2)
        if mode in ("frozen", "corner"):
            x, y = X(P0[o])
            body += ('<g class="lock" transform="translate(%.1f %.1f)"><rect x="-6" y="0" width="12" height="9" rx="1.5"/>'
                     '<path d="M-3.5,0 v-3 a3.5,3.5 0 0 1 7,0 v3"/></g>' % (x + 11, y - 22))
        defs = "".join('<marker id="bm%d-%s" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
                       'orient="auto"><path class="h-%s" d="M0,0 L9,4.5 L0,9 z"/></marker>' % (k, c, c)
                       for c in ("raw", "mv", "pr"))
        label = {"proposed": "proposed move", "frozen": "frozen", "free": "free", "tangent": "tangent",
                 "slide": "slide / OnCurve", "corner": "corner"}[mode]
        return ('<svg viewBox="0 0 %d %d" role="img" aria-label="one boundary vertex, %s">%s%s</svg>' % (W, H, label, "<defs>" + defs + "</defs>", body))

    def chips(on: str) -> str:
        return '<span class="bm-chips">%s</span>' % "".join('<i class="%s">%s</i>' % ("on" if c in on else "", c) for c in "123")

    px = {k: float(np.linalg.norm(got[k][v] - P0[v])) for k in ("free", "tangent", "slide")}
    edge = 0.5 * float(np.linalg.norm(P0[v + 1] - P0[v]) + np.linalg.norm(P0[v] - P0[v - 1]))        # mean boundary edge at v
    adj = cn.boundary_adjacency(mesh)
    cnb = [int(idx[-1, 1]), int(idx[-2, 0])]                                      # the corner's two boundary neighbours
    nb_turn = float(np.mean([cn.turning_deg(P0, np.array([n]), np.array([adj[n]]))[0] for n in cnb]))
    cells = [
        ("proposed", "<b>a. proposed move</b>%s<br>the average of its quads' shrunk corners (rings), which lie on the chords" % chips("1"), ),
        ("frozen", "<b>frozen</b>%s<br>boundary=\"frozen\", Frozen: the move is dropped" % chips(""), ),
        ("free", "<b>free</b>%s<br>boundary=\"free\": moves as proposed, %.1f %% of the edge inside the curve"
         % (chips("1"), 100 * -off["free"] / edge), ),
        ("tangent", "<b>tangent</b>%s<br>boundary=\"tangent\", Tangent: the normal part is removed; slides along, %.1f %% of the move off the curve"
         % (chips("12"), 100 * abs(off["tangent"]) / px["tangent"]), ),
        ("slide", "<b>slide / OnCurve</b>%s<br>a curve fitted (slide) or given (OnCurve): tangent move, then projected onto it" % chips("123"), ),
        ("corner", "<b>corner</b>%s<br>turn %.0f&deg;, its neighbours' %.0f&deg;: a jump, frozen except under free" % (chips(""), turn, nb_turn), ),
    ]
    html_cells = []
    for (mode, cap), o in zip(cells, [v] * 5 + [cv]):
        html_cells.append('<figure class="bm-cell">%s<figcaption>%s</figcaption></figure>' % (panel(o, mode), cap.replace("`", "")))

    def sw(kind: str) -> str:
        parts = {"raw": '<line class="raw" x1="1" y1="6" x2="22" y2="6"/><path class="h-raw" d="M22,2 L30,6 L22,10 z"/>',
                 "mv": '<line class="mv" x1="1" y1="6" x2="22" y2="6"/><path class="h-mv" d="M22,2 L30,6 L22,10 z"/>',
                 "pr": '<line class="pr" x1="1" y1="6" x2="22" y2="6"/><path class="h-pr" d="M22,2 L30,6 L22,10 z"/>',
                 "tan": '<line class="tan" x1="1" y1="6" x2="30" y2="6"/>',
                 "curve": '<line class="curve" x1="1" y1="6" x2="30" y2="6"/>',
                 "chord": '<line class="chord" x1="1" y1="6" x2="30" y2="6"/>',
                 "st": '<circle class="st" cx="15" cy="6" r="5"/>', "fin": '<circle class="fin" cx="15" cy="6" r="4.2"/>'}[kind]
        return '<svg width="32" height="12" viewBox="0 0 32 12" aria-hidden="true">%s</svg>' % parts

    legend = ('<div class="bm-legend">'
              + "".join("<span>%s%s</span>" % (sw(k), t) for k, t in (
                  ("raw", "proposed move"), ("mv", "move taken"), ("tan", "tangent of the circle through the vertex and its neighbours"), ("pr", "projection"),
                  ("curve", "curve"), ("chord", "chord"), ("st", "start"), ("fin", "final position")))
              + "</div>")
    layers = ('<div class="bm-layers"><span>%s <b>source</b>: a boundary side is not shrunk across its normal</span>'
              '<span>%s <b>movement</b>: the move loses its component normal to the boundary</span>'
              '<span>%s <b>constraint</b>: the vertex is projected back onto its curve</span></div>'
              % (chips("1"), chips("2"), chips("3")))
    fig = raw("<style>%s</style>" % BM_CSS + '<div class="bm">' + layers + '<div class="bm-grid">' + "".join(html_cells) + "</div>"
              + legend + "</div>")
    return fig + "\n", bad


# ============================================================================================================
# the two guides: a decision tree for constraints and the smoothing workflow (HTML / CSS boxes, no images)
# ============================================================================================================
FLOW_CSS = (
    ".fc{max-width:36rem;margin:1rem 0 1.4rem}"
    ".fc *{box-sizing:border-box}"
    ".fc a{color:inherit;text-decoration:none}.fc a:hover b{text-decoration:underline}"
    ".fc .box{border:1px solid var(--color-background-border);border-left:4px solid var(--color-foreground-muted);"
    "border-radius:6px;padding:.4rem .7rem;background:var(--color-background-secondary);font-size:.88em;line-height:1.35}"
    ".fc .box b{color:var(--color-foreground-primary)}"
    ".fc .box small{display:block;color:var(--color-foreground-secondary);font-size:.92em}"
    ".fc .q{border-left-color:var(--color-brand-primary)}"
    ".fc .go{border-left-color:#1f9d6b}"
    ".fc .st{border-left-color:var(--color-brand-primary)}"
    ".fc .n{display:inline-block;min-width:1.4em;text-align:center;margin-right:.4rem;border-radius:50%;"
    "background:var(--color-brand-primary);color:var(--color-background-primary);font-weight:600}"
    ".fc .arr{text-align:center;color:var(--color-foreground-muted);font-size:.8em;line-height:1.5}"
    ".fc .row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:.2rem .6rem;align-items:center}"
    ".fc .yes{display:flex;align-items:center;gap:.4rem}.fc .yes>span{color:var(--color-foreground-muted);font-size:.8em;white-space:nowrap}"
    ".fc .yes>a,.fc .yes>.box{flex:1;min-width:0}"
    ".fc .note{margin-top:.6rem;font-size:.8em;color:var(--color-foreground-secondary)}"
    ".fc .chips a{text-decoration:underline dotted}"
    ".fc .chips{display:flex;flex-wrap:wrap;gap:.25rem;margin-top:.3rem}"
    ".fc .chips span{border:1px solid var(--color-background-border);border-radius:1rem;padding:0 .55rem;"
    "background:var(--color-background-primary);font-size:.92em}"
    "@media (max-width:40rem){.fc .row{grid-template-columns:minmax(0,1fr)}}"
)


def _box(cls: str, body: str, href: str | None = None) -> str:
    inner = '<div class="box %s">%s</div>' % (cls, body)
    return '<a href="%s">%s</a>' % (href, inner) if href else inner


def constraint_tree() -> str:
    """Which constraint for a group of vertices, as a vertical chain of questions with the answer beside each."""
    def ask(q: str, yes: str) -> str:
        return ('<div class="row">%s<div class="yes"><span>yes &#8594;</span>%s</div></div>' % (_box("q", q), yes)
                + '<div class="row"><div class="arr">&#8595; no</div><div></div></div>')

    def ans(body: str, href: str) -> str:
        return _box("go", body, href)
    rows = [
        '<div class="row">%s<div></div></div><div class="row"><div class="arr">&#8595;</div><div></div></div>' % _box("st", "<b>A group of vertices</b>"),
        ask("<b>Must it not move at all?</b>", ans("<code>Frozen(points)</code>", "#what-a-constraint-is")),
        ask("<b>Is it on geometry you know?</b><small>circle, plane, sphere, CAD or a tessellation</small>",
            ans("<b>on a surface</b>: <code>OnSurface(points, target)</code><small>on a curve or edge: "
                "<code>OnCurve(points, target)</code></small>", "#targets")),
        ask("<b>Is it a mesh boundary, with no geometry?</b>",
            ans("<code>boundary=\"slide\"</code><small>OnCurve / OnSurface fitted to the input, corners frozen; any method</small>"
                "<small><code>\"tangent\"</code>: slides along, never inward; keeps the area</small>"
                "<small>or <code>\"frozen\"</code>, the default and safest</small>", "#boundary-settings")),
        ask("<b>Is it inside a designed boundary layer?</b>",
            ans("<code>KeepLayer(seeds, rings)</code><small>a plate that must slide whole: <code>SlideColumns</code></small>",
                "#layers-and-columns")),
        '<div class="row">%s<div></div></div>' % ans(
            "<b>Otherwise free</b><small>nothing named; on the boundary, <code>boundary=\"free\"</code> lets it move as "
            "proposed</small>", "#boundary-settings"),
    ]
    note = ('<div class="note">Where claims meet: <a href="#precedence-and-conflicts"><b>Frozen &gt; OnCurve &gt; '
            'OnSurface &gt; Tangent &gt; Free</b></a>.</div>')
    return raw("<style>%s</style>" % FLOW_CSS + '<div class="fc" role="group" aria-label="choosing a constraint">'
               + "".join(rows) + note + "</div>")


def workflow_chart() -> str:
    """The smoothing workflow, one box per step, each linking to the page that explains it (hrefs relative to
    ``docs/smoother/quad/``, where the chart is included)."""
    user = "../../user/quad_smoother.html"
    steps = [
        ("Check the input", "<code>check_connectivity</code> refuses gaps; overlapping and inverted quads are reported",
         user + "#checks-before-smoothing"),
        ("Attach geometry, set constraints", "<code>OnSurface</code> / <code>OnCurve</code> / <code>boundary=</code>: "
         "the decision tree", "methods/constraints.html#choosing-constraints"),
        ("Protect designed layers", "<code>KeepLayer</code>, <code>SlideColumns</code>, <code>weights</code>",
         "methods/grading.html"),
        ("Pick the method", "<code>suggest_schedule</code> by default", "methods/index.html"),
        ("Run a round, read the report", "<code>propose</code> / <code>accept</code> / <code>undo</code>; numbers per region",
         user + "#the-round-report"),
        ("High-order mesh?", "stays curved: the nodes follow the vertices; <code>rebuild_high_order</code> re-projects "
         "them onto the targets, then export; TMOP smooths a curved mesh as curved", user + "#high-order"),
    ]
    chips = ('<div class="chips"><span><a href="methods/laplace.html">Laplace</a>: a cheap pass</span>'
             '<span><a href="methods/optimizer.html">optimizer</a>: the shape</span>'
             '<span><a href="methods/winslow.html">Winslow</a>: concave or thin curved blocks</span>'
             '<span><a href="../mfem/index.html">TMOP</a> (optional, MFEM): keeps the input\'s sizes and layers, '
             '<code>("tmop", {...})</code></span>'
             '<span>inverted quads: <code>untangle_schedule()</code>, an <code>optimize</code> preset with a '
             'wide-to-narrow <code>delta</code></span></div>')
    out = []
    for i, (title, sub, href) in enumerate(steps, start=1):
        extra = chips if i == 4 else ""
        out.append('<div class="box st"><span class="n">%d</span><a href="%s"><b>%s</b></a><small>%s</small>%s</div>'
                   % (i, href, title, sub, extra))
        if i == 5:
            out.append('<div class="arr">&#8595; good enough<br><span class="note">not yet: back to step 2, 3 or 4 '
                       '(<code>undo</code> a worse round)</span></div>')
        elif i < len(steps):
            out.append('<div class="arr">&#8595;</div>')
    return raw("<style>%s</style>" % FLOW_CSS + '<div class="fc" role="group" aria-label="smoothing workflow">'
               + "".join(out) + "</div>")


# ============================================================================================================
# the optimizer: objective history of real runs
# ============================================================================================================
def history_figure(rows: list[dict[str, Any]]) -> str:
    """Two charts from the stage logs: the annulus under ``optimize(iters=OPTIMIZER_ITERS)`` (F/quad per accepted
    iteration), and the tangled square under the delta-continuation stages of the combined schedule, one stage per delta,
    against the optimizer alone at the default delta."""
    by = by_key(rows)
    lg = by["annulus"]["optimizer"]["logs"][0]
    ann = [(0, lg.before.objective_mean)] + [(h[0], h[3]) for h in lg.history]
    deltas = list(default(smooth.untangle_schedule, "deltas"))
    tlogs = by["tangled"]["combined"]["logs"][:len(deltas)]
    tang, bounds, inv0 = [(0, tlogs[0].before.objective_mean)], [], None
    off = 0
    for d, tl in zip(deltas, tlogs):
        bounds.append((off, d))
        for h in tl.history:
            tang.append((off + h[0], h[3]))
            if inv0 is None and h[1] == 0:
                inv0 = (off + h[0], h[3])
        off += tl.passes_done
    alone_log = by["tangled"]["optimizer"]["logs"][0]
    alone = [(0, alone_log.before.objective_mean)] + [(h[0], h[3]) for h in alone_log.history]

    W, H, L, R, T, B = 300, 210, 46, 10, 26, 34
    pw, ph = W - L - R, H - T - B

    def frame(xmax: float, ylab: list[tuple[float, str]], Y: Any, title: str, xlabel: str) -> str:
        out = ['<text class="ti" x="%d" y="14">%s</text>' % (L, title)]
        for yv, lab in ylab:
            y = Y(yv)
            out.append('<line class="gr" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (L, y, L + pw, y))
            out.append('<text class="tk" x="%d" y="%.1f" text-anchor="end">%s</text>' % (L - 4, y + 3.5, lab))
        step = 10 ** math.floor(math.log10(max(xmax, 1)))
        step = step * (5 if xmax / step > 12 else 2 if xmax / step > 5 else 1)
        for xv in np.arange(0, xmax + 1e-9, step):
            x = L + pw * xv / xmax
            out.append('<text class="tk" x="%.1f" y="%d" text-anchor="middle">%d</text>' % (x, T + ph + 13, xv))
        out.append('<path class="ax" d="M%d %dV%dH%d"/>' % (L, T, T + ph, L + pw))
        out.append('<text class="tk" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (L + pw / 2, H - 4, xlabel))
        return "".join(out)

    def line(pts: list[tuple[float, float]], X: Any, Y: Any, cls: str) -> str:
        return '<path class="%s" d="M%s"/>' % (cls, " L".join("%.1f %.1f" % (X(x), Y(y)) for x, y in pts))

    # the annulus: linear y
    xa = max(x for x, _ in ann)
    ya = max(y for _, y in ann) * 1.05
    Xa = lambda x: L + pw * x / xa  # noqa: E731
    Ya = lambda y: T + ph * (1 - y / ya)  # noqa: E731
    drop = ann[0][1] - ann[-1][1]
    k90 = next(x for x, y in ann if ann[0][1] - y >= 0.9 * drop)
    tstep = [t for t in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if ya / t <= 6][0]
    ticks = [(v, "%g" % v) for v in np.arange(0.0, ya, tstep)]
    left = ('<svg viewBox="0 0 %d %d" class="hc-fig" role="img" aria-label="objective history, annulus">' % (W, H)
            + frame(xa, ticks, Ya, "annulus: optimize(iters=%d)" % OPTIMIZER_ITERS, "accepted iteration")
            + line(ann, Xa, Ya, "s1")
            + '<line class="mk" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>' % (Xa(k90), T, Xa(k90), T + ph)
            + '<text class="tk" x="%.1f" y="%d">90 %% of the drop by %d</text>' % (Xa(k90) + 4, T + 12, k90)
            + "</svg>")

    # the tangled square: log y
    allv = [y for _, y in tang + alone if y > 0]
    lo, hi = math.floor(math.log10(min(allv))), math.ceil(math.log10(max(allv)))
    xt = max(x for x, _ in tang)
    Xt = lambda x: L + pw * x / xt  # noqa: E731
    Yt = lambda y: T + ph * (1 - (math.log10(max(y, 10.0 ** lo)) - lo) / (hi - lo))  # noqa: E731
    every = max(1, int(math.ceil((hi - lo) / 5)))
    ticks_t = [(10.0 ** e, "1e%d" % e) for e in range(lo, hi + 1, every)]
    marks = []
    for k, (x, d) in enumerate(bounds):
        xe = bounds[k + 1][0] if k + 1 < len(bounds) else xt
        if k % 2:
            marks.append('<rect class="band" x="%.1f" y="%d" width="%.1f" height="%d"/>' % (Xt(x), T, Xt(xe) - Xt(x), ph))
        if Xt(xe) - Xt(x) >= 34:
            marks.append('<text class="tk" x="%.1f" y="%d">δ=%s</text>' % (Xt(x) + 3, T + ph - 5, num(d)))
    right = ('<svg viewBox="0 0 %d %d" class="hc-fig" role="img" aria-label="objective history, tangled square">' % (W, H)
             + frame(xt, ticks_t, Yt, "tangled square: delta continuation", "accepted iteration, stages in turn")
             + "".join(marks) + line(tang, Xt, Yt, "s2") + line(alone, Xt, Yt, "s3")
             + '<circle class="d3" cx="%.1f" cy="%.1f" r="3"/>' % (Xt(alone[-1][0]), Yt(alone[-1][1]))
             + '<text class="tk" x="%.1f" y="%.1f">optimize() alone: stops here</text>' % (Xt(alone[-1][0]) + 6, Yt(alone[-1][1]) + 4)
             + ('<circle class="d2" cx="%.1f" cy="%.1f" r="3"/><text class="tk" x="%.1f" y="%.1f">0 inverted</text>'
                % (Xt(inv0[0]), Yt(inv0[1]), Xt(inv0[0]) + 6, Yt(inv0[1]) - 6) if inv0 else "")
             + "</svg>")
    fig = raw('<div class="hc-pair">' + left + right + "</div>")
    stages = ", ".join("%s (%d)" % (num(d), tl.passes_done) for d, tl in zip(deltas, tlogs))
    cap = ("F/quad, the exact objective the stage log records after each accepted iteration (log scale on the right). "
           "Annulus: %.2f → %.2f in %d iterations (stopped: %s). Tangled square: %d inverted at the start; one stage per "
           "`delta`, shaded in turn, widest first: %s iterations; no inverted quad from iteration %d. The exact F can rise "
           "while the descent lowers the barrier-free one it actually minimises. The optimizer alone at `delta=%s` gets "
           "%d iteration and stops: %s."
           % (ann[0][1], ann[-1][1], lg.passes_done, lg.reason, tlogs[0].before.n_inverted, stages,
              inv0[0] if inv0 else -1, num(default(smooth.optimize, "delta")), alone_log.passes_done, alone_log.reason))
    return fig + "\n" + cap + "\n"


# ============================================================================================================
# the parameter tables, from the real signatures
# ============================================================================================================
PARAM_DOC = {
    "laplace": {
        "passes": "number of passes",
        "w": "shrink factor per pass, in (0, 1]; 1 is a no-op, smaller moves vertices more",
        "normals": "reference normals for the readings (default: the input's winding)",
        "elements": "boolean mask of quads: only their vertices move (the local variant)",
        "max_sj_drop": "a pass that adds an inverted quad or lowers the minimum scaled J by more than this is refused "
                       "and the stage stops on the last accepted pass",
        "kappa": "inversion-penalty weight, used only for the F/quad reading in the log",
        "eps": "inversion-penalty threshold on the scaled J, likewise only for the reading",
    },
    "winslow": {
        "sweeps": "maximum Gauss-Seidel sweeps",
        "omega": "under-relaxation of every move, in (0, 1]",
        "tol": "stop when the largest move of a sweep is below this many local edge lengths",
        "normals": "reference normals (default: the input's winding; pass the surface's)",
        "elements": "boolean mask of quads: only their vertices move (the local variant)",
        "max_sj_drop": "a sweep that adds an inverted quad or lowers the minimum scaled J by more than this below the "
                       "best reached is retried with `omega` halved, then refused and the stage stops",
        "retries": "number of `omega` halvings tried on a refused sweep",
        "kappa": "inversion-penalty weight, used only for the F/quad reading in the log",
        "eps": "inversion-penalty threshold on the scaled J, likewise only for the reading",
    },
    "optimize": {
        "iters": "maximum conjugate-gradient iterations",
        "normals": "reference normals (default: the input's winding; pass the surface's)",
        "elements": "boolean mask of quads: only their vertices move, plus the quads they touch count (the local variant)",
        "alpha_max": "cap on a step, in local edge lengths",
        "restart": "`'halve'` halves the Fletcher-Reeves beta on every second iteration, `'none'` is plain Fletcher-Reeves",
        "kappa": "inversion-penalty weight per corner on the sum objective",
        "eps": "scaled-J threshold below which a corner is penalised",
        "delta": "barrier-removal width of the descent metric; 0 descends on the exact phi",
        "tol_step": "stop when the accepted step is below this many local edge lengths",
        "tol_rel": "stop when the relative drop of F in one iteration is below this",
        "weights": "per-quad weights on the shape term (0: that quad's shape is irrelevant; the penalty still applies)",
    },
}


def param_table(fn: Any, doc: dict[str, str], skip: tuple[str, ...] = ("mesh", "resolved")) -> str:
    sig = inspect.signature(fn)
    names_ = [n for n in sig.parameters if n not in skip]
    missing = [n for n in names_ if n not in doc]
    if missing:
        raise RuntimeError("%s has parameters with no description: %s" % (fn.__name__, missing))
    body = []
    for n in names_:
        p = sig.parameters[n]
        d = "required" if p.default is inspect.Parameter.empty else "`%r`" % (p.default,)
        body.append(["`%s`" % n, d, doc[n]])
    return md_table(["parameter", "default", "meaning"], body)


# -- the optimizer's scale: a constant rescale of the objective is a no-op ------------------------------------
def rescale_text() -> str:
    """Two measurements on the annulus: rescaling the objective by a constant, and rescaling the model's units."""
    inp = annulus()
    P = np.asarray(inp.mesh.points)
    ref, _ = smooth.optimize(inp.mesh, inp.res, normals=inp.nrm, iters=50)
    ref_p = np.asarray(ref.points)
    e = np.asarray(inp.mesh.edges)
    edge = float(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1).min())
    ones = np.ones(inp.mesh.n_quads)
    worst = 0.0
    for c in (1e-3, 1e3):
        out, _ = smooth.optimize(inp.mesh, inp.res, normals=inp.nrm, iters=50, weights=c * ones, kappa=c * smooth.KAPPA)
        worst = max(worst, float(np.abs(np.asarray(out.points) - ref_p).max() / edge))
    worst_u = 0.0
    for f in (1e-3, 1e3):
        big = quadmesh.QuadMesh.from_corners(P * f, inp.mesh.corners)
        out, _ = smooth.optimize(big, cn.resolve(big, []), normals=inp.nrm, iters=50)
        worst_u = max(worst_u, float(np.abs(np.asarray(out.points) / f - ref_p).max() / edge))
    return ("On the annulus, 50 iterations: scaling the objective (weights and `kappa`) by 10⁻³ or 10³ moves no vertex "
            "more than %.0e shortest edges; scaling the coordinates, %.0e." % (worst, worst_u))


# ============================================================================================================
# the grading page
# ============================================================================================================
GRADING_OUTER = 5


def grading_runs() -> list[dict[str, Any]]:
    """The graded strip (no protection, ``KeepLayer``, zero weights on the layer) and the pipe section's O-grid (wall
    sliding, with and without ``KeepLayer``), every run ``user_schedule(outer=GRADING_OUTER)``, the schedule the examples use."""
    out: list[dict[str, Any]] = []
    sched = smooth.user_schedule(outer=GRADING_OUTER)

    def record(case: str, key: str, protect: str, m0: Any, res: Any, nrm: Any, weights: Any = None,
               **extra: Any) -> dict[str, Any]:
        o, logs = smooth.run_schedule(m0, res, sched, normals=nrm, weights=weights)
        s0, s1 = smooth.shape_stats(m0.points, m0.corners), smooth.shape_stats(o.points, o.corners)
        on = cn.on_target_report(o.points, o.corners, res)
        return {"case": case, "key": key, "protect": protect, "m0": m0, "nrm": nrm, "mesh": o,
                "q0": smooth.quality(m0.points, m0.corners, nrm), "q1": logs[-1].after,
                "ratio": (s0.size_ratio_max, s1.size_ratio_max), "frozen_move": on.max_frozen_move, **extra}

    keep_rings = len(LAYER_ROWS) - 1
    for key, protect in (("strip", "none"), ("strip_kept", "KeepLayer(rings=%d)" % keep_rings)):
        inp = strip(key == "strip_kept")
        r = record("graded strip", key, protect, inp.mesh, inp.res, inp.nrm, line=(inp.layer_line, False))
        r["thickness"] = inp.extras(np.asarray(r["mesh"].points))[LAYER_KEY]
        out.append(r)
    inp = strip(False)
    bottom = np.arange(inp.mesh.n_points)[np.isclose(np.asarray(inp.mesh.points)[:, 1], 0.0)]
    rings = cn.layer_rings(inp.mesh, bottom, keep_rings)
    wts = np.where(np.isin(inp.mesh.corners, rings).all(axis=1), 0.0, 1.0)
    r = record("graded strip", "strip_weights", "weights=0 on the layer quads", inp.mesh, inp.res, inp.nrm, wts,
               line=(inp.layer_line, False))
    r["thickness"] = inp.extras(np.asarray(r["mesh"].points))[LAYER_KEY]
    out.append(r)
    # the pipe section of cs_pipe_section.py: the same perturbed input with and without the kept rings
    radius, n_side, n_rad, scale, grad_, noise, seed = 0.5, 6, 8, 0.7, 0.5, 0.2, 7
    start = quadmesh.ogrid(linemesh.circle(radius, 4 * n_side, element_tag="wall"), n_side,
                           geometric_spacing(n_rad, grad_), center_scale=scale, quadrant_scale=scale)
    nrm = smooth.reference_normals(start.points, start.corners)
    wall = np.unique(start.line_mesh.lines[start.edge_tags.flatnonzero("wall")])
    circ = cn.circle([0, 0, 0], radius)
    keep_p = 3
    first_ring = np.setdiff1d(cn.layer_rings(start, wall, 1), wall)
    edge_ring = np.setdiff1d(cn.layer_rings(start, wall, keep_p), cn.layer_rings(start, wall, keep_p - 1))
    SP = np.asarray(start.points)
    edge_ring = edge_ring[np.argsort(np.arctan2(SP[edge_ring, 1], SP[edge_ring, 0]))]   # the kept layer's outer ring
    kept = cn.resolve(start, [cn.OnCurve(wall, circ), cn.KeepLayer(wall, rings=keep_p)])
    free = np.flatnonzero(kept.kind == cn.FREE)
    P = np.array(start.points, dtype=float)
    ell0 = cn.vertex_length(start.points, start.corners)
    P[free, :2] += np.random.default_rng(seed).uniform(-noise, noise, (free.size, 2)) * ell0[free, None]
    m0 = smooth.with_points(start, P)

    def first_layer(Q: Any) -> Any:
        a_w = np.arctan2(Q[wall, 1], Q[wall, 0])
        a_r = np.arctan2(Q[first_ring, 1], Q[first_ring, 0])
        near = first_ring[np.argmin(np.abs(np.angle(np.exp(1j * (a_w[:, None] - a_r[None, :])))), axis=1)]
        return np.hypot(Q[wall, 0], Q[wall, 1]) - np.hypot(Q[near, 0], Q[near, 1])

    for key, cons, label in (("pipe", [cn.OnCurve(wall, circ)], "OnCurve wall only (it slides)"),
                             ("pipe_kept", [cn.OnCurve(wall, circ), cn.KeepLayer(wall, rings=keep_p)],
                              "KeepLayer(rings=%d) + OnCurve" % keep_p)):
        res = cn.resolve(start, cons)
        r = record("pipe section", key, label, m0, res, nrm, wall_frozen=int((res.kind[wall] == cn.FROZEN).sum()),
                   wall_n=int(wall.size), line=(edge_ring, True))
        t0, t1 = first_layer(start.points), first_layer(np.asarray(r["mesh"].points))
        r["thickness"] = float(100 * (np.abs(t1 - t0) / t0).max())
        out.append(r)
    return out


def grading_table(runs: list[dict[str, Any]]) -> str:
    body = [[r["case"], r["protect"], "%.3f → %.3f" % (r["q0"].sj_min, r["q1"].sj_min), "%.2f → %.2f" % r["ratio"],
             "%.0f %%" % r["thickness"]] for r in runs]
    return md_table(["case", "protection", "min scaled J", "size ratio max", "first layer thickness change (max)"], body)


def grading_text(runs: list[dict[str, Any]]) -> str:
    keep = [r for r in runs if r["key"] == "pipe_kept"][0]
    ratio = (1.0 / LAYER_NX) / (LAYER_ROWS[1] - LAYER_ROWS[0])
    return ("Every run is `user_schedule(outer=%d)`. A first-layer cell of the strip has edge ratio %.1f : 1, so its corners "
            "score phi = %.0f. With `KeepLayer` and a sliding wall, %d of %d wall vertices resolve frozen; the largest move "
            "of a frozen vertex is %.1e."
            % (GRADING_OUTER, ratio, ((ratio + 1 / ratio) / 2) ** 2, keep["wall_frozen"], keep["wall_n"], keep["frozen_move"]))


def grading_figure(runs: list[dict[str, Any]]) -> str:
    by = {r["key"]: r for r in runs}
    short = {"strip": "unprotected", "strip_kept": "KeepLayer", "strip_weights": "weights = 0",
             "pipe": "wall slides", "pipe_kept": "KeepLayer + wall"}
    out = []
    for title, keys in (("graded strip", ("strip", "strip_kept", "strip_weights")), ("pipe section", ("pipe", "pipe_kept"))):
        r0 = by[keys[0]]
        states = [("input", np.asarray(r0["m0"].points), "min J %.2f" % r0["q0"].sj_min)]
        for k in keys:
            r = by[k]
            states.append((short[k], np.asarray(r["mesh"].points), "min J %.2f · layer %s" % (
                r["q1"].sj_min, "%+.0f %%" % r["thickness"] if r["thickness"] > 0.05 else "as meshed")))
        out.append(multiples_row(title, View("planar"), r0["m0"], r0["nrm"], states, False, r0["line"]))
    return raw('<div class="mm-grid">' + LEGEND.replace('<span class="mm-dash"></span><span>input boundary</span>', "")
               .replace("outer edge of a designed layer", "outer edge of the kept rings")
               + "".join(out) + "</div>")


def check_grading(runs: list[dict[str, Any]]) -> list[str]:
    """The statements the grading page makes in words, re-checked against its runs."""
    by = {r["key"]: r for r in runs}
    bad = []
    if by["strip_kept"]["thickness"] != 0.0 or by["pipe_kept"]["thickness"] != 0.0:
        bad.append("KeepLayer no longer holds the layer thickness exactly")
    if by["strip"]["thickness"] < 10 or by["pipe"]["thickness"] < 10:
        bad.append("an unprotected layer no longer thickens")
    if by["strip_weights"]["thickness"] < 10:
        bad.append("weights alone now protect the layer")
    if by["pipe_kept"]["wall_frozen"] != by["pipe_kept"]["wall_n"]:
        bad.append("KeepLayer and a sliding wall now combine")
    if not by["pipe_kept"]["ratio"][1] > by["pipe_kept"]["ratio"][0]:
        bad.append("the size jump at the kept layer's edge no longer grows")
    return bad


# ============================================================================================================
# the constraints page: one projected step
# ============================================================================================================
def step_figure() -> tuple[str, str]:
    """One optimizer step on the sphere, computed with the library: the shape gradient of one vertex, split by
    ``constrain_direction`` into the part the sphere allows (tangential) and the part it forbids (normal).  Returns the SVG
    block and its caption.  The vertex shown is the one whose gradient has the largest normal part."""
    inp = sphere()
    P = np.asarray(inp.mesh.points)
    g = smooth.gradient(P, inp.mesh.corners, inp.nrm)
    ell = cn.vertex_length(P, inp.mesh.corners)
    surf = inp.res.surfaces[0]
    nv = surf.normal(P)
    frac = np.abs(np.einsum("ij,ij->i", -g, nv)) / np.maximum(np.linalg.norm(g, axis=1), 1e-300)
    v = int(np.argmax(frac * (np.linalg.norm(g, axis=1) * ell > 0)))
    d = -g[v] / np.linalg.norm(g[v]) * 0.5 * ell[v]                      # half a local edge along the raw descent direction
    n = nv[v]
    t_full = cn.constrain_direction(-g, P, inp.res)[v]
    t_hat = t_full / np.linalg.norm(t_full)
    d_n, d_t = float(d @ n), float(d @ t_hat)
    x_raw = P[v] + d
    x_t = P[v] + d_t * t_hat                                              # the step the optimizer takes
    off = float(np.linalg.norm(x_t - surf.project(x_t[None, :])[0]) / ell[v])      # what snap removes afterwards
    R = 1.0

    def uv(x: Any) -> tuple[float, float]:             # (along t_hat, along n) relative to the vertex
        return float((x - P[v]) @ t_hat), float((x - P[v]) @ n)

    s = 230.0 / float(ell[v])                           # px per unit length: the raw step is about 115 px
    ox, oy = 60.0, (60.0 if d_n < 0 else 190.0)

    def X(p: tuple[float, float]) -> tuple[float, float]:
        return ox + s * p[0], oy - s * p[1]

    phis = np.linspace(-0.25, 1.0, 61) * float(ell[v]) / R
    path = "M" + " L".join("%.1f,%.1f" % X((R * np.sin(ph), R * np.cos(ph) - R)) for ph in phis)
    px = {"x": X(uv(P[v])), "raw": X(uv(x_raw)), "t": X(uv(x_t))}
    below = 1 if d_n < 0 else -1                        # the raw step leaves on this side of the surface

    def arrow(a: str, b: str, cls: str) -> str:
        return ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="%s" marker-end="url(#ah-%s)"/>'
                % (px[a][0], px[a][1], px[b][0], px[b][1], cls, cls))

    def text(x: float, y: float, label: str, cls: str = "lbl") -> str:
        return '<text x="%.1f" y="%.1f" class="%s">%s</text>' % (x, y, cls, label)

    svg = (
        '<svg viewBox="0 0 300 250" role="img" aria-label="one constrained optimizer step on a sphere" class="sm-stepfig">'
        '<style>.sm-stepfig{width:100%;max-width:440px;display:block;margin:0 auto;color:var(--color-foreground-primary)}'
        '.sm-stepfig .surf{fill:none;stroke:var(--color-foreground-muted);stroke-width:2}'
        '.sm-stepfig .tan{stroke:var(--color-foreground-muted);stroke-width:1;stroke-dasharray:4 3}'
        '.sm-stepfig .raw{stroke:#d64034;stroke-width:2}'
        '.sm-stepfig .proj{stroke:var(--color-brand-primary);stroke-width:3}'
        '.sm-stepfig .cut{stroke:#d64034;stroke-width:1.5;stroke-dasharray:4 3}'
        '.sm-stepfig .pt{fill:var(--color-foreground-primary)}'
        '.sm-stepfig .lbl{font:11px sans-serif;fill:var(--color-foreground-primary)}'
        '.sm-stepfig .mut{font:11px sans-serif;fill:var(--color-foreground-muted)}'
        '.sm-stepfig .red{font:11px sans-serif;fill:#d64034}'
        '.sm-stepfig .blue{font:11px sans-serif;fill:var(--color-brand-primary)}'
        '.sm-stepfig .h-raw{fill:#d64034}.sm-stepfig .h-proj{fill:var(--color-brand-primary)}</style>'
        '<defs><marker id="ah-raw" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
        'orient="auto"><path class="h-raw" d="M0,0 L9,4.5 L0,9 z"/></marker>'
        '<marker id="ah-proj" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
        'orient="auto"><path class="h-proj" d="M0,0 L9,4.5 L0,9 z"/></marker></defs>'
        '<path d="' + path + '" class="surf"/>'
        + '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="tan"/>' % (px["x"][0] - 40, px["x"][1], px["x"][0] + 250, px["x"][1])
        + '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="cut"/>' % (px["t"][0], px["t"][1], px["raw"][0], px["raw"][1])
        + arrow("x", "raw", "raw") + arrow("x", "t", "proj")
        + '<circle cx="%.1f" cy="%.1f" r="4" class="pt"/>' % px["x"]
        + text(px["x"][0] - 14, px["x"][1] - below * 12, "x")
        + text(px["raw"][0] + 8, px["raw"][1] + below * 4, "raw step", "red")
        + text(px["t"][0] + 4, px["t"][1] - below * 8, "constrained step", "blue")
        + text(px["t"][0] + 6, (px["t"][1] + px["raw"][1]) / 2, "normal part, removed", "mut")
        + text(px["x"][0] + 150, px["x"][1] - below * 8, "tangent plane at x", "mut")
        + text(8, 16 if below > 0 else 240, "sphere, cross-section through x", "mut")
        + '</svg>')
    cap = ("One vertex of the jittered sphere, half a local edge along its shape gradient (the vertex whose gradient has the "
           "largest normal part): the raw step (red) is %.0f %% normal. `constrain_direction` removes that part with the "
           "sphere's own normal and keeps the tangential step (blue); `snap` then removes the remaining distance to the "
           "sphere, %.3f local edges here."
           % (100 * abs(d_n) / float(np.linalg.norm(d)), off))
    return raw(svg), cap


def phi_plot_block() -> str:
    deltas = list(default(smooth.untangle_schedule, "deltas"))
    cfg = {"delta": smooth.DELTA, "kappa": smooth.KAPPA, "eps": smooth.EPS, "deltas": deltas}
    return raw('<script src="%s_static/smoother_phi_plot.js"></script>\n<div id="sm-phi-plot"></div>\n'
               '<script>window.initPhiPlot("sm-phi-plot", %s);</script>' % (UP, json.dumps(cfg)))


def phi_values() -> str:
    """The numbers the plot's explanation quotes, and the delta continuation."""
    return ("Here `eps` = %s, `kappa` = %s and the default `delta` = %s. `untangle_schedule()` is a delta-continuation preset of "
            "`optimize`, not a separate method: one stage of %d iterations per `delta` in %s, widest first; "
            "`suggest_schedule` puts it in front when a quad starts inverted. The logs always report the exact phi."
            % (num(smooth.EPS), num(smooth.KAPPA), num(smooth.DELTA), default(smooth.untangle_schedule, "iters"),
               ", ".join(num(d) for d in default(smooth.untangle_schedule, "deltas"))))


def constraint_params() -> str:
    """The constraint dataclasses: fields and defaults read from the classes."""
    import dataclasses
    doc = {
        "Frozen": "never moves",
        "Tangent": "moves along the boundary only: the normal part of each move or gradient is removed; nothing projects it "
                   "back",
        "OnSurface": "stays on a `Surface` (none: fitted to the input faces); `slide=True` moves tangentially and is "
                     "projected back every step, `False` projects once and holds",
        "OnCurve": "the same along a `Curve` (none: a spline through the input boundary)",
        "KeepLayer": "freezes every vertex within `rings` quad-rings of the seeds (seeds included)",
        "SlideColumns": "rigid columns with one shared degree of freedom along `direction`",
    }
    body = []
    for cls in (cn.Frozen, cn.Tangent, cn.OnSurface, cn.OnCurve, cn.KeepLayer, cn.SlideColumns):
        fields = ", ".join("%s" % f.name + ("" if f.default is dataclasses.MISSING else "=%r" % (f.default,))
                           for f in dataclasses.fields(cls))
        body.append(["`%s(%s)`" % (cls.__name__, fields), doc[cls.__name__]])
    b, cd = default(cn.resolve, "boundary"), default(cn.resolve, "corner_deg")
    body.append(["`resolve(mesh, constraints, boundary=%r, corner_deg=%r)`" % (b, cd),
                 "resolves the claims once; `boundary` is a shorthand for every boundary vertex no constraint names: "
                 "`'frozen'` -> `Frozen`, `'free'` -> nothing, `'tangent'` -> `Tangent`, `'slide'` -> `OnCurve` fitted to the "
                 "input; corners (a turn above `corner_deg` and 3x its neighbours') frozen except under `'free'`"])
    t = default(cn.on_target_report, "tol")
    body.append(["`on_target_report(points, corners, resolved, tol=%r)`" % (t,),
                 "distance of constrained vertices to their targets and the largest move of a frozen one, in local edge lengths"])
    return md_table(["call", "meaning"], body)


def write_vtu(rows: list[dict[str, Any]], out: str) -> None:
    from nekmeshpy.io.writer import quad_to_vtu
    os.makedirs(out, exist_ok=True)
    for row in rows:
        inp = row["input"]

        def sj(m: Any, nrm: Any = inp.nrm) -> Any:
            return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)

        quad_to_vtu(inp.mesh, os.path.join(out, "methods_%s_input.vtu" % inp.key), cell_data={"sj_min": sj(inp.mesh)})
        for key, _ in METHODS:
            m = row["runs"][key]["mesh"]
            quad_to_vtu(m, os.path.join(out, "methods_%s_%s.vtu" % (inp.key, key)),
                        point_data={"displacement": np.asarray(m.points) - np.asarray(inp.mesh.points)},
                        cell_data={"sj_min": sj(m)})


WORKFLOW = "smoothing_workflow.inc"           # the one fragment outside the ``methods_*`` family: the Smoothing index page's

#: every fragment this script writes; anything else named ``methods_*`` in ``_generated`` is stale and removed
FRAGMENTS = ("compare.methods", "compare.figure", "compare.score", "compare.shows", "compare.trade", "compare.tables",
             "laplace.steps", "laplace.boundary", "laplace.params", "optimizer.history", "optimizer.plot", "optimizer.phi",
             "optimizer.rescale", "optimizer.params", "winslow.figure", "winslow.params", "winslow.table",
             "constraints.figure", "constraints.tree", "constraints.boundary", "constraints.slide", "constraints.params",
             "grading.figure", "grading.table", "grading.text")


def build(generated: str = GENERATED, vtu: str | None = None) -> dict[str, int]:
    t0 = time.perf_counter()
    print("comparison runs ...", flush=True)
    rows = run_all()
    wruns = winslow_runs(rows)
    lap_bnd, bad_lap = laplace_boundary()
    slide, bad_slide = slide_figure()
    move, bad_move = boundary_move_figure()
    text: dict[str, str] = {
        "compare.methods": methods_table(),
        "compare.figure": compare_figure(rows),
        "compare.score": score_table(rows),
        "compare.shows": shows_text(rows),
        "compare.trade": trade_table(rows, wruns),
        "compare.tables": input_tables(rows),
        "laplace.steps": laplace_steps(),
        "laplace.boundary": lap_bnd,
        "laplace.params": param_table(smooth.laplace, PARAM_DOC["laplace"]),
        "optimizer.history": history_figure(rows),
        "optimizer.plot": phi_plot_block(),
        "optimizer.phi": phi_values() + "\n",
        "optimizer.rescale": rescale_text() + "\n",
        "optimizer.params": param_table(smooth.optimize, PARAM_DOC["optimize"]),
        "winslow.figure": winslow_figure(rows),
        "winslow.params": param_table(smooth.winslow, PARAM_DOC["winslow"]),
        "winslow.table": winslow_table(wruns),
        "constraints.slide": slide,
        "constraints.boundary": move,
        "constraints.tree": constraint_tree(),
        "constraints.params": constraint_params(),
    }
    fig, cap = step_figure()
    text["constraints.figure"] = fig + "\n" + cap + "\n"
    g = grading_runs()
    text["grading.figure"] = grading_figure(g)
    text["grading.table"] = grading_table(g)
    text["grading.text"] = grading_text(g) + "\n"
    assert set(text) == set(FRAGMENTS)
    sizes = {}
    for name, body in text.items():
        path = os.path.join(generated, "methods_" + name + ".inc")
        write(path, body)
        sizes[os.path.basename(path)] = len(body.encode())
    chart = workflow_chart()
    write(os.path.join(generated, WORKFLOW), chart)                                  # included by docs/smoother/index.md
    sizes[WORKFLOW] = len(chart.encode())
    keep = {"methods_%s.inc" % f for f in FRAGMENTS}
    for fn in os.listdir(generated):
        if fn.startswith("methods_") and fn not in keep:
            os.remove(os.path.join(generated, fn))
            print("  removed stale %s" % fn)
    stale = check_claims(rows) + check_grading(g) + check_winslow(wruns, rows) + bad_lap + bad_slide + bad_move
    for msg in stale:
        print("WARNING: a statement on the pages no longer holds: " + msg, file=sys.stderr)
    if vtu:
        write_vtu(rows, vtu)
    print("done in %.1f s" % (time.perf_counter() - t0))
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu", default=None, help="also write before/after .vtu files of every run into this directory")
    args = ap.parse_args()
    for name, size in sorted(build(vtu=args.vtu).items()):
        print("  %-40s %7.1f KB" % (name, size / 1e3))


if __name__ == "__main__":
    main()
