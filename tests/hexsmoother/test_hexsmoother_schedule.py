"""run_schedule and the method registry a backend plugs in by."""

import numpy as np
import pytest
from hexblocks import block

from nekmeshpy.hexmesh.smoother import constrain as cn
from nekmeshpy.hexmesh.smoother import smooth


@pytest.fixture
def registry():
    saved = dict(smooth._METHODS)
    yield
    smooth._METHODS.clear()
    smooth._METHODS.update(saved)


def test_the_builtin_methods_and_a_schedule_run_in_order():
    assert smooth.methods() == ["laplace", "laplace_local", "optimize", "optimize_local"]
    m = block(4, jitter=0.3, seed=1)
    r = cn.resolve(m, [])
    out, logs = smooth.run_schedule(m, r, smooth.user_schedule(outer=1, n_opt=5), record=True)
    assert [lg.method for lg in logs] == ["laplace", "optimize", "laplace_local", "laplace_local"]
    assert all(lg.points is not None for lg in logs)
    np.testing.assert_array_equal(logs[-1].points, out.points)
    assert logs[1].after.objective_mean < logs[0].before.objective_mean


def test_a_registered_method_runs_by_name_and_receives_elements_and_weights(registry):
    seen = {}

    def nudge(mesh, resolved, *, step=0.0, elements=None, weights=None):
        seen.update(step=step, elements=elements, weights=weights)
        q = smooth.quality(mesh.points, mesh.corners)
        return mesh, smooth.StageLog("nudge", 1, 1, "completed", "did nothing", q, q, ())

    smooth.register_method("nudge", nudge, local=True, weights=True)
    assert "nudge" in smooth.methods()
    m = block(3, jitter=0.35, seed=2)
    w = np.ones(m.n_hexes)
    out, logs = smooth.run_schedule(m, cn.resolve(m, []), [("nudge", {"step": 0.5})], weights=w)
    assert out is m and logs[0].method == "nudge" and seen["step"] == 0.5
    assert seen["elements"].shape == (m.n_hexes,) and seen["weights"] is w
    with pytest.raises(ValueError, match="already registered"):
        smooth.register_method("nudge", nudge)
    smooth.register_method("nudge", nudge, replace=True)
    with pytest.raises(ValueError, match="unknown smoothing method"):
        smooth.run_schedule(m, cn.resolve(m, []), [("tmop", {})])


def test_suggest_schedule_untangles_first_only_when_needed():
    tangled = block(4, jitter=0.45, seed=0)
    clean = block(4, jitter=0.1, seed=0)
    s1 = smooth.suggest_schedule(smooth.quality(tangled.points, tangled.corners), outer=1)
    s2 = smooth.suggest_schedule(smooth.quality(clean.points, clean.corners), outer=1)
    assert s1[:5] == smooth.untangle_schedule() and s2 == smooth.user_schedule(outer=1)
