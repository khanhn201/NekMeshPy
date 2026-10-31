"""The full user schedule -- 10 outer loops of laplace(3) + optimize(20), then local lifts --
run to the end on every geometry, so convergence is on record."""

import numpy as np
import pytest
from grids import grid

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth


def _oriented(m, P, q):
    if np.einsum("qj,qj->q", smooth.reference_normals(m.points, m.corners), m.points[m.corners].mean(1)).mean() < 0:
        return quadmesh.QuadMesh.from_corners(P, q[:, ::-1].copy())
    return m


def case_square():
    m = quadmesh.QuadMesh.from_corners(*grid(14, jitter=0.45, seed=5))
    return m, cn.resolve(m, []), None


def case_disk():
    sec = quadmesh.ogrid(linemesh.circle(1.0, 24, element_tag="wall"), 6, 4, wall_tag="wall")
    sec = smooth.with_points(sec, sec.points + np.random.default_rng(3).normal(size=sec.points.shape) * [0.03, 0.03, 0])
    return sec, cn.resolve(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))]), None


def case_sphere_patch():
    P, q = grid(10, jitter=0.45, seed=2)
    u, v = (P[:, 0] - 0.5) * 1.4, (P[:, 1] - 0.5) * 1.4
    P = np.stack([np.sin(u) * np.cos(v), np.sin(v), np.cos(u) * np.cos(v)], 1)
    m = _oriented(quadmesh.QuadMesh.from_corners(P, q), P, q)
    r = cn.resolve(m, [cn.OnSurface(np.arange(m.n_points), cn.sphere([0, 0, 0], 1.0))], boundary="free")
    return smooth.with_points(m, cn.snap(m.points, r)), r, None


def case_closed_sphere():
    cs = quadmesh.sphere(1.0, 6)
    nrm = smooth.reference_normals(cs.points, cs.corners)
    r = cn.resolve(cs, [cn.OnSurface(np.arange(cs.n_points), cn.sphere([0, 0, 0], 1.0))])
    P = cn.snap(cs.points + np.random.default_rng(1).normal(size=cs.points.shape) * 0.04, r)
    return smooth.with_points(cs, P), r, nrm


CASES = {"square": case_square, "disk": case_disk, "sphere_patch": case_sphere_patch,
         "closed_sphere": case_closed_sphere}


def series(logs):
    """Cumulative accepted passes/iterations -> (n_inverted, min scaled J, mean objective)."""
    rows = [(0, logs[0].before.n_inverted, logs[0].before.sj_min, logs[0].before.objective_mean)]
    k = 0
    for lg in logs:
        for h in lg.history:
            k += 1
            rows.append((k, h[1], h[2], h[3]))
    return np.array(rows)


@pytest.mark.parametrize("name", list(CASES))
def test_long_schedule_converges_on_every_geometry(name):
    m, r, nrm = CASES[name]()
    out, logs = smooth.run_schedule(m, r, smooth.user_schedule(), normals=nrm)
    first, last = logs[0].before, logs[-1].after
    s = series(logs)
    assert first.n_inverted > 0 and last.n_inverted == 0          # starts tangled, ends valid
    assert last.sj_min > 0.5
    assert last.objective_mean < 0.05 * first.objective_mean       # converged by >95 % in the objective
    assert cn.on_target_report(out.points, out.corners, r).passed  # still exactly on the targets
    # the objective settles: the last quarter of the run moves less than 5 % of the first quarter
    n = len(s)
    assert abs(s[-1, 3] - s[3 * n // 4, 3]) < 0.05 * abs(s[0, 3] - s[n // 4, 3])
    # inversions never come back once they are gone
    gone = np.flatnonzero(s[:, 1] == 0)
    assert gone.size and (s[gone[0]:, 1] == 0).all()
    # frozen vertices never moved
    fr = r.kind == cn.FROZEN
    np.testing.assert_array_equal(out.points[fr], r.anchor[fr])


def _random_tangles(n=12, trials=14, seed=7):
    """Meshes with three interior vertices dragged up to 1.6 cells, keeping only the tangled ones."""
    Pg, q = grid(n)
    ref = quadmesh.QuadMesh.from_corners(Pg, q)
    nrm = smooth.reference_normals(ref.points, ref.corners)
    rng = np.random.default_rng(seed)
    inner = [i for i in range(Pg.shape[0]) if 0 < Pg[i, 0] < 1 and 0 < Pg[i, 1] < 1]
    out = []
    for _ in range(trials):
        P = Pg.copy()
        for v in rng.choice(inner, 3, replace=False):
            P[v, :2] += rng.uniform(-1.6, 1.6, 2) / n
        m = smooth.with_points(ref, P)
        if smooth.quality(m.points, m.corners, nrm).n_inverted > 0:
            out.append(m)
    return out, nrm


def test_untangle_continuation_beats_the_plain_schedule_on_random_tangles():
    """The reason `suggest_schedule` puts the continuation first on a tangled mesh."""
    meshes, nrm = _random_tangles()
    assert len(meshes) >= 10
    plain = cont = 0
    for m in meshes:
        r = cn.resolve(m, [])
        plain += smooth.run_schedule(m, r, smooth.user_schedule(), normals=nrm)[1][-1].after.n_inverted == 0
        cont += smooth.run_schedule(m, r, smooth.untangle_schedule() + smooth.user_schedule(),
                                    normals=nrm)[1][-1].after.n_inverted == 0
    assert cont == len(meshes)               # always untangled
    assert plain < len(meshes)               # the plain schedule does not (documented limit)


def test_suggest_schedule_adds_the_untangling_stage_only_when_needed():
    m = quadmesh.QuadMesh.from_corners(*grid(10, jitter=0.45))
    nrm = smooth.reference_normals(m.points, m.corners)
    tangled = smooth.quality(m.points, m.corners, nrm)
    Pg, qg = grid(10)
    clean = smooth.quality(Pg, qg, smooth.reference_normals(Pg, qg))
    assert tangled.n_inverted > 0 and smooth.suggest_schedule(tangled)[0][1]["delta"] == 1.0
    assert smooth.suggest_schedule(clean)[0] == ("laplace", {"passes": 3, "w": 0.97})


def test_a_failed_untangle_is_never_worse_than_the_input():
    """With the barrier left in (delta=0) a tangle cannot be crossed; the stage must then say so
    and not return a mesh with more inverted quads than it was given."""
    m = _random_tangles(trials=4)[0][0]
    nrm = smooth.reference_normals(*grid(12))
    out, log = smooth.optimize(m, cn.resolve(m, []), iters=60, delta=0.0, normals=nrm)
    assert log.after.n_inverted <= log.before.n_inverted
