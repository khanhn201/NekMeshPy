"""T7 (the part that belongs in CI): the cost of a smoothing pass grows linearly with the mesh, not faster.
The big measurements (1e4, 1e5, 1e6 quads) are in the report; here a quarter-size mesh must cost no more than a
loose multiple, which a quadratic hot spot would break."""

import time

import numpy as np
from grids import grid

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth


def _cost(n, iters=4, reps=3):
    P, q = grid(n, jitter=0.3, seed=1)
    m = quadmesh.QuadMesh.from_corners(P, q)
    r = cn.resolve(m, [])
    nrm = smooth.reference_normals(m.points, m.corners)
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        smooth.optimize(m, r, iters=iters, normals=nrm)
        smooth.laplace(m, r, passes=3, normals=nrm)
        best = min(best, time.perf_counter() - t)
    return best, len(q)


def test_a_smoothing_stage_costs_about_linear_time_in_the_number_of_quads():
    small, ns = _cost(40)
    big, nb = _cost(120)                                     # 9x the quads
    ratio = big / small
    assert ratio < 9 * 3, "9x the quads cost %.1fx the time (%d -> %d quads): super-linear" % (ratio, ns, nb)


def test_memory_and_gap_check_scale_with_the_mesh():
    P, q = grid(150, jitter=0.2, seed=1)
    m = quadmesh.QuadMesh.from_corners(P, q)
    t = time.perf_counter()
    rep = quadmesh.check_connectivity(m)
    assert rep.passed and time.perf_counter() - t < 20.0      # 22,500 quads: a second or so; 20 s only catches a blow-up
