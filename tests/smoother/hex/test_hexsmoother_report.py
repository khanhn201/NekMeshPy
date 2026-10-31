"""The before/after round report."""

import numpy as np
from hexblocks import block, block_arrays

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.hexmesh.smoother import constrain as cn
from nekmeshpy.hexmesh.smoother import linearize, round_report, smooth


def _tagged_block():
    P, H = block_arrays(4)
    tags = np.where(P[H].mean(axis=1)[:, 0] > 0.5, "right", "left")
    return block(4, jitter=0.3, seed=3, tags=list(tags))


def test_report_fields_regions_sampling_targets_and_frozen():
    m = _tagged_block()
    r = cn.resolve(m, [cn.frozen_region(m, "right")], boundary="slide")
    out, logs = smooth.run_schedule(m, r, smooth.user_schedule(outer=1, n_opt=10))
    rep = round_report(m, out, r, stages=logs, sample_order=3)
    assert rep.passed and not rep.failures
    assert [g.region for g in rep.regions_before] == ["left", "right"]
    assert rep.regions_before[0].n_hexes == rep.regions_after[0].n_hexes == 32
    assert len(rep.regions_before[0].below) == len(rep.thresholds) == 3
    right_b, right_a = rep.regions_before[1], rep.regions_after[1]
    assert right_b == right_a                                       # frozen region: nothing changed
    assert rep.sample_order == 3 and rep.sampled_after.orders == (3,) and rep.curved_before is None
    assert rep.after.objective_mean < rep.before.objective_mean
    assert any(t.name == "plane" and t.max_distance < 1e-14 for t in rep.targets)    # planes held
    assert rep.n_frozen > 0 and rep.n_frozen_moved == 0
    text = str(rep)
    for part in ("1. constraints", "2. what ran", "per region", "sampled at order 3", "4. geometry", "frozen:"):
        assert part in text


def test_a_moved_frozen_vertex_or_a_new_inversion_fails_the_round():
    m = block(3, jitter=0.2, seed=1)
    r = cn.resolve(m, [])
    P = m.points.copy()
    P[0] += 1e-9
    rep = round_report(m, smooth.with_points(m, P), r, sample_order=1)
    assert not rep.passed and any("bit-identical" in f for f in rep.failures)
    Q = m.points.copy()
    Q[21] = [0.05, 0.05, 0.05]                                      # drag an interior vertex to a corner
    rep = round_report(m, smooth.with_points(m, Q), r, sample_order=1)
    assert any("inverted hexes rose" in f for f in rep.failures)


def test_a_high_order_input_is_read_on_its_curved_map_too():
    cyl = hexmesh.extrude(quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 1), 1.0, 1)
    lin = linearize(cyl)
    r = cn.resolve(lin, [])
    out, logs = smooth.run_schedule(lin, r, [("laplace", {"passes": 1})])
    rep = round_report(lin, out, r, stages=logs, original=cyl, sample_order=4)
    assert rep.curved_before is not None and rep.curved_before.orders == (4,)
    assert "high-order input" in str(rep)
