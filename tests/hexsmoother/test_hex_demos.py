"""The hex smoother demos in ``examples/hexsmoother/`` run, leave a valid linear ``mesh``, and the headline
claim each one makes holds (asserted with margin, on the numbers the demo itself computed)."""

import os

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import HexMesh
from nekmeshpy.hexmesh import smoother as hs


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    cache = {}

    def run(name):
        if name not in cache:
            out = tmp_path_factory.mktemp(name.replace(".py", ""))
            cache[name] = run_example(os.path.join("hexsmoother", name), out)
        return cache[name]
    return run


def _valid(ns):
    mesh = ns["mesh"]
    assert isinstance(mesh, HexMesh) and 0 < mesh.n_hexes <= 500 and mesh.order == 1
    assert np.isfinite(mesh.points).all()


def test_01_methods(demo):
    ns = demo("01_methods.py")
    _valid(ns)
    q = {k: v["q"] for k, v in ns["rows"].items()}
    assert q["input"].n_inverted == 0 and q["input"].sj_min < 0.4
    for k in ("laplace", "optimizer", "schedule"):
        assert q[k].n_inverted == 0
        assert q[k].sj_min > q["input"].sj_min + 0.4
    assert q["optimizer"].objective_mean < 0.98 * q["laplace"].objective_mean
    assert q["schedule"].objective_mean <= 1.01 * q["optimizer"].objective_mean


def test_02_slide_box(demo):
    ns = demo("02_slide_box.py")
    _valid(ns)
    rows = ns["rows"]
    s, f = rows["slide"], rows["frozen"]
    assert s["off_plane"] < 1e-12 and s["off_line"] < 1e-12 and s["corner_move"] == 0.0
    assert s["q"].sj_min > f["q"].sj_min + 0.3
    assert s["q"].n_inverted == 0 and f["q"].n_inverted == 0


def test_03_free_boundary(demo):
    ns = demo("03_free_boundary.py")
    _valid(ns)
    rows, sag = ns["rows"], ns["sagitta"]
    assert rows["free"]["volume_change_pct"] < -2.0
    assert abs(rows["tangent"]["volume_change_pct"]) < 0.5
    assert rows["free"]["wall_dist"] > 3 * rows["tangent"]["wall_dist"]
    assert rows["slide"]["cap_off_plane"] < 1e-12
    assert rows["slide"]["facet"] < 1e-9
    assert rows["slide"]["wall_dist"] <= 1.01 * sag
    assert rows["exact"]["wall_dist"] < 1e-9
    assert all(r["q"].n_inverted == 0 for r in rows.values())


def test_04_thin_layer(demo):
    ns = demo("04_thin_layer.py")
    _valid(ns)
    un, pr = ns["rows"]["unprotected"], ns["rows"]["protected"]
    assert un["thick_mean_pct"] > 50.0
    assert max(abs(pr["thick_mean_pct"]), abs(pr["thick_max_pct"])) < 1e-9
    assert pr["keep_move"] == 0.0
    assert pr["shape"].size_ratio_max > un["shape"].size_ratio_max
    assert all(r["q"].n_inverted == 0 for r in ns["rows"].values())


def test_05_untangle(demo):
    ns = demo("05_untangle.py")
    _valid(ns)
    q = {k: v["q"] for k, v in ns["rows"].items()}
    assert q["input"].n_inverted >= 5
    assert q["optimize0"].n_inverted > 0
    assert q["untangle"].n_inverted == 0 and q["untangle"].sj_min > 0.9
    assert q["suggested"].n_inverted == 0
    sched = hs.untangle_schedule()
    assert sched and all(name == "optimize" and "delta" in knobs for name, knobs in sched)
