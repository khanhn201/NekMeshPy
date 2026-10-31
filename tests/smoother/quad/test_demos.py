"""The smoother demos in ``examples/smoother/quad/`` run, leave a valid ``mesh``, and the headline
claim each one makes holds (asserted with margin, on the numbers the demo itself computed)."""

import json
import os

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import QuadMesh, quadmesh


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    cache = {}

    def run(name):
        if name not in cache:
            out = tmp_path_factory.mktemp(name.replace(".py", ""))
            cache[name] = (run_example(os.path.join("smoother", "quad", name), out), out)
        return cache[name]
    return run


def _valid(ns):
    mesh = ns["mesh"]
    assert isinstance(mesh, QuadMesh) and mesh.n_quads > 0 and mesh.order == 1
    assert np.isfinite(mesh.points).all()


def _frames_ok(path, n_quads):
    doc = json.load(open(path))
    assert len(doc["quads"]) == n_quads and len(doc["frames"]) >= 2
    assert doc["frames"][-1]["n_inverted"] <= doc["frames"][0]["n_inverted"]
    assert os.path.getsize(path) < 400_000                      # small enough to ship to the docs viewer


def test_01_basics(demo):
    ns, out = demo("01_basics.py")
    _valid(ns)
    for name in ("before", "laplace", "optimizer", "schedule"):
        assert (out / ("01_basics_%s.vtu" % name)).exists()
    q0 = ns["q0"]
    f = {k: v[1][-1].after for k, v in ns["results"].items()}
    assert q0.n_inverted == 0 and all(v.n_inverted == 0 for v in f.values())
    for v in f.values():                                         # every method improves every headline number
        assert v.sj_min > q0.sj_min + 0.3 and v.sj_mean > q0.sj_mean and v.sj_q05 > q0.sj_q05 + 0.2
        assert v.objective_mean < 0.5 * q0.objective_mean
    # the optimizer reaches a clearly lower objective than Laplace, and the schedule matches it
    assert f["optimizer"].objective_mean < 0.8 * f["laplace"].objective_mean
    assert f["schedule"].objective_mean < 1.05 * f["optimizer"].objective_mean
    assert ns["mesh"] is ns["results"]["schedule"][0]
    _frames_ok(out / "01_basics_frames.json", ns["before"].n_quads)


def test_02_untangle(demo):
    ns, out = demo("02_untangle.py")
    _valid(ns)
    n = ns["CASES"]
    assert len(ns["rows"]) == n and all(r[0] > 0 for r in ns["rows"])        # every case really starts tangled
    assert ns["untangle_ok"] >= n - 1                                          # the continuation repairs (nearly) all
    assert ns["untangle_ok"] > ns["plain_ok"]                                  # and strictly more than the plain schedule
    assert all(r[2] <= r[1] for r in ns["rows"])                               # never worse than plain, case by case
    q = ns["u_logs"][-1].after
    assert q.n_inverted == 0 and q.sj_min > 0.9
    _frames_ok(out / "02_untangle_frames.json", ns["tangled"].n_quads)


def test_03_constraints(demo):
    ns, out = demo("03_constraints.py")
    _valid(ns)
    # (a) the sliding boundary stays on the circle, evens out, and is no worse than frozen
    (fro, fro_logs), (sli, sli_logs) = ns["disk_out"]["frozen"], ns["disk_out"]["sliding"]
    wall = ns["wall_ids"]
    for m in (fro, sli):
        assert np.abs(np.hypot(m.points[wall, 0], m.points[wall, 1]) - ns["DISK_RADIUS"]).max() < 1e-12
        assert m.points.shape == ns["disk"].points.shape
    assert ns["wall_gaps"](sli.points) > 0.9 > 0.5 > ns["wall_gaps"](fro.points)
    assert sli_logs[-1].after.n_inverted == 0 and fro_logs[-1].after.n_inverted == 0
    assert sli_logs[-1].after.objective_mean < 0.9 * fro_logs[-1].after.objective_mean
    assert ns["mesh"] is sli
    # (b) box: corners do not move at all, creases and faces stay on their lines / planes
    assert ns["corner_ids"].size == 8 and len(ns["creases"]) == 12
    assert ns["corner_moved"] == 0.0
    assert ns["crease_dist"] < 1e-12 and ns["face_dist"] < 1e-12
    assert ns["moved_crease"] > 1e-3                                           # the creases did slide
    q = ns["box_logs"][-1].after
    assert q.n_inverted == 0 and q.sj_min > 0.9
    # (c) KeepLayer leaves the layer as meshed; without it the layer inflates
    keep, nokeep = ns["strip_out"]["KeepLayer"][0], ns["strip_out"]["no KeepLayer"][0]
    t0 = ns["thick0"]
    assert np.abs(ns["layer_top"](keep.points) - t0).max() == 0.0
    assert (np.abs(ns["layer_top"](nokeep.points) - t0) / t0).max() > 0.5
    for name in ("a_disk_frames", "c_layer_frames"):
        _frames_ok(out / ("03_%s.json" % name), ns["disk"].n_quads if name.startswith("a") else ns["strip"].n_quads)
    for name in ("03_a_disk_sliding.vtu", "03_b_box_after.vtu", "03_c_layer_KeepLayer.vtu"):
        assert (out / name).exists()


def test_04_curved_surfaces(demo):
    ns, out = demo("04_curved_surfaces.py")
    _valid(ns)
    for name, r in ns["rows"].items():
        assert r["logs"][0].before.n_inverted > 0 and r["logs"][-1].after.n_inverted == 0
        assert r["on"].passed and r["on"].max_distance < 1e-9                    # exactly on the surface
        assert abs(r["out_m"]["area"] / r["in_m"]["area"] - 1) < 0.02            # no area drift
        assert abs(float(quadmesh.angle_defect(r["out"]).sum()) - 2 * np.pi * r["chi"]) < 1e-9    # Gauss-Bonnet: topology, not quality
        assert r["out_m"]["rms"] <= r["in_m"]["rms"] and r["out_m"]["max"] <= r["in_m"]["max"]
        # (a) the floor: the smoothed error is of the order of the undisturbed mesh's, never far below it
        assert r["out_m"]["rms"] > 0.5 * r["ref_m"]["rms"] and r["out_m"]["rms"] < 2.0 * r["ref_m"]["rms"]
        for tag in ("before", "after"):
            assert (out / ("04_%s_%s.vtu" % (name, tag))).exists()
    assert ns["rows"]["torus"]["out_m"]["rms"] < 0.25 * ns["rows"]["torus"]["in_m"]["rms"]
    assert ns["rows"]["sphere"]["n_irregular"] == 8 and ns["rows"]["torus"]["n_irregular"] == 0
    # the shown limit: the ellipsoid's elements are not smaller where it curves more
    assert ns["h_out"][ns["hi"]].mean() / ns["h_out"][ns["lo"]].mean() > 0.9
    # (b) one folded quad: surface normals beat winding normals, in error and in repaired folds, and not every fold is repaired
    fo, n_fold = ns["fold_out"], len(ns["FOLD_QUADS"])
    q0 = ns["FOLD_QUADS"][0]
    assert fo[q0, "winding"][1]["rms"] > 10 * fo[q0, "surface"][1]["rms"]
    clean = {lab: sum(fo[fq, lab][2] == 0 for fq in ns["FOLD_QUADS"]) for lab in ("surface", "winding")}
    assert clean["surface"] > clean["winding"] and clean["surface"] >= 2 and clean["surface"] < n_fold + 1
    assert (out / ("04_b_ellipsoid_fold%d_surface_normals.vtu" % q0)).exists()


def test_05_high_order(demo):
    ns, out = demo("05_high_order.py")
    _valid(ns)
    curved, rep = ns["curved"], ns["rep"]
    assert curved.order == ns["ORDER"] == 4 and ns["perturbed"].order == 4
    # wall nodes: far from the circle after the carry, on it to rounding after the projection
    assert ns["wall_node_deviation"](ns["carried"]) > 1e-5 and ns["wall_node_deviation"](curved) < 1e-13
    # between the nodes the polynomial is close, not exact: a small floor, many orders below the carried error
    bc, bp = ns["wall_between_node_deviation"](ns["carried"]), ns["wall_between_node_deviation"](curved)
    assert bp < 1e-6 and bp < 1e-3 * bc and bp > 0.0
    assert np.array_equal(curved.points, ns["after"].points)                     # corners are the smoothed vertices
    assert rep.n_inverted_sampled == 0 and rep.sj_min_rebuilt > 0.5 and rep.max_untouched_move == 0.0
    assert all(sj > 0 for sj in ns["sj_orders"])                                  # read on 4 lattices
    err_lin = abs(quadmesh.area(ns["after"]) - np.pi)
    err_cur = abs(quadmesh.area(curved, high_order=True) - np.pi)
    assert err_cur < 1e-3 * err_lin
    assert ns["mesh"] is ns["after"] and ns["mesh"].order == 1
    for name in ("05_before.vtu", "05_after.vtu", "05_curved.vtu"):
        assert (out / name).exists()
    _frames_ok(out / "05_frames.json", ns["before"].n_quads)


def test_06_session(demo):
    ns, out = demo("06_session.py")
    _valid(ns)
    # (a) the round flow: a proposal changes nothing, a stale one is refused, undo restores the input bit for bit
    assert ns["p1"].report.passed and ns["p1"].report.round == 1
    assert ns["undone"] is ns["start"] and np.array_equal(ns["undone"].points, ns["saved_points"])
    assert ns["n_accepted"] >= 1 and ns["s"].round == 0
    assert ns["smoothed"].points.shape == ns["saved_points"].shape
    assert not np.array_equal(ns["smoothed"].points, ns["saved_points"])
    assert ns["p1"].report.after.objective_mean < 0.1 * ns["p1"].report.before.objective_mean
    assert ns["p_conservative"].report.after.objective_mean > ns["p1"].report.after.objective_mean
    for rnd in ("round_01", "round_02"):
        for name in ("before.vtu", "after.vtu"):
            assert (out / ns["OUT_DIR"] / rnd / name).exists()
    # (b) the gap gate refuses the open seam, and the override is written into the report
    assert "the gap check failed" in ns["gap_text"] and "gap cluster" in ns["gap_text"]
    assert any("GAP CHECK FAILED and overridden" in ln for ln in ns["rep_gap"].constraints.splitlines())
    # (c) fast paths: every round passed; auto stops early
    assert len(ns["fast_reports"]["print"][0]) == 3 and all(r.passed for r in ns["fast_reports"]["print"][0])
    auto = ns["fast_reports"]["auto"][0]
    assert 1 <= len(auto) < ns["AUTO_ROUNDS"] and all(r.passed for r in auto)
    # (d) structural refusals
    assert "collapsed edge" in ns["refusals"]["collapsed edge"]
    assert "pinch" in ns["refusals"]["pinch vertex"]
    for name in ("06_input.vtu", "06_smoothed.vtu"):
        assert (out / name).exists()
    _frames_ok(out / "06_frames.json", ns["start"].n_quads)


# -- case studies (examples/smoother/quad/cs_*.py): the real example meshes, as built and perturbed ----------------
def _recovers(run, *, size_ratio=True):
    """The stressed run ends clean and better than its input on every headline number."""
    q0, q1 = run["q"]
    assert q1.n_inverted == 0 and run["ov"][1] == 0
    assert q1.sj_min > q0.sj_min and q1.sj_mean > q0.sj_mean and q1.sj_q05 > q0.sj_q05
    if size_ratio:
        assert run["size_ratio"][1] < run["size_ratio"][0]
    assert run["on"].passed


def test_cs_channel_obstacle(demo):
    ns, out = demo("cs_channel_obstacle.py")
    _valid(ns)
    asis, stress = ns["runs"]["asis"], ns["runs"]["stress"]
    assert ns["start"].n_quads == ns["ns"]["mesh"].n_quads > 6000                   # the example's own mesh
    assert asis["q"][0].sj_min == pytest.approx(1.0) and asis["ov"] == (0, 0)       # already a perfect tensor grid
    assert asis["q"][1].n_inverted == 0 and asis["q"][1].sj_mean > 0.99             # nothing is broken, nothing gained
    assert asis["on"].passed and asis["on"].max_frozen_move == 0.0
    assert stress["q"][0].sj_min < 0.8 and stress["ov"][0] == 0
    _recovers(stress)
    assert stress["q"][1].sj_q05 > 0.99
    for run in ("asis", "stress"):
        for tag in ("before", "after"):
            assert (out / ("cs_channel_%s_%s.vtu" % (run, tag))).exists()
        path = out / ("cs_channel_%s_frames.json" % run)
        assert len(json.load(open(path))["frames"]) >= 2 and os.path.getsize(path) < 150_000


def test_cs_pipe_section(demo):
    ns, out = demo("cs_pipe_section.py")
    _valid(ns)
    asis, stress = ns["runs"]["asis"], ns["runs"]["stress"]
    for run in (asis, stress):                                                       # the protected layers did not move at all
        assert np.array_equal(run["out"].points[ns["layer"]], run["before"].points[ns["layer"]])
        assert np.abs(np.hypot(*run["out"].points[ns["wall"], :2].T) - ns["RADIUS"]).max() < 1e-12
        assert run["on"].passed and run["q"][1].n_inverted == 0 and run["ov"][1] == 0
    assert asis["q"][0].n_inverted == 0 and asis["q"][1].sj_min >= asis["q"][0].sj_min
    assert asis["size_ratio"][1] > asis["size_ratio"][0]                             # the grading below the layers is relaxed
    assert stress["q"][0].sj_min < asis["q"][0].sj_min
    _recovers(stress, size_ratio=False)
    for run in ("asis", "stress"):
        path = out / ("cs_pipe_%s_frames.json" % run)
        _frames_ok(path, ns["start"].n_quads)
        assert os.path.getsize(path) < 150_000


def test_cs_sphere_shell(demo):
    ns, out = demo("cs_sphere_shell.py")
    _valid(ns)
    runs = ns["runs"]
    assert set(runs) == {(s, v) for s in ("sphere", "cube") for v in ("asis", "stress")}
    for (name, label), run in runs.items():
        assert run["q"][1].n_inverted == 0 and run["ov"][1] == 0 and run["on"].passed
        assert run["on"].max_distance < 1e-9 and run["on"].max_frozen_move == 0.0
        for tag in ("before", "after"):
            assert (out / ("cs_shell_%s_%s_%s.vtu" % (name, label, tag))).exists()
        if label == "stress":
            _recovers(run)
    cube = runs["cube", "asis"]
    assert np.abs(cube["out"].points - cube["before"].points).max() < 1e-3        # a regular box is (to 1e-4 of an edge) a fixed point
    assert runs["sphere", "asis"]["q"][1].sj_mean > runs["sphere", "asis"]["q"][0].sj_mean
    assert ns["mesh"] is runs["sphere", "asis"]["out"]


def test_cs_hemisphere(demo):
    ns, out = demo("cs_hemisphere.py")
    _valid(ns)
    asis, stress = ns["runs"]["asis"], ns["runs"]["stress"]
    for run in (asis, stress):
        assert np.array_equal(run["out"].points[ns["rim"]], run["before"].points[ns["rim"]])     # the rim is frozen
        assert run["q"][1].n_inverted == 0 and run["ov"][1] == 0 and run["on"].passed
    assert ns["rim"].size == 24
    assert asis["size_ratio"][1] < asis["size_ratio"][0]
    _recovers(stress)
    for run in ("asis", "stress"):
        for tag in ("before", "after"):
            assert (out / ("cs_hemisphere_%s_%s.vtu" % (run, tag))).exists()


def test_cs_wall_surface(demo):
    ns, out = demo("cs_wall_surface.py")
    _valid(ns)
    asis, stress = ns["runs"]["asis"], ns["runs"]["stress"]
    assert isinstance(ns["mesh"], QuadMesh) and ns["hexes"].n_hexes == 7200          # a surface smoothed; the hexes are the example's
    assert ns["start"].n_quads == 1440 and ns["res"].counts()["on_surface"] == ns["inside"].size
    for run in (asis, stress):
        assert run["on"].passed and run["on"].max_frozen_move == 0.0
        assert run["q"][1].n_inverted == 0
        assert run["ov"][1] <= run["ov"][0]                                          # overlaps never grow (they may remain)
        assert np.array_equal(run["out"].points[ns["rim"]], run["before"].points[ns["rim"]])
    assert asis["q"][1].sj_mean > asis["q"][0].sj_mean
    assert stress["q"][1].sj_mean > stress["q"][0].sj_mean + 0.03 and stress["q"][1].sj_q05 > stress["q"][0].sj_q05
    assert stress["size_ratio"][1] < stress["size_ratio"][0]
    for run in ("asis", "stress"):
        for tag in ("before", "after"):
            assert (out / ("cs_wall_%s_%s.vtu" % (run, tag))).exists()


def _fast(demo, name):
    """Run a surface case study with one outer cycle (the hex mesh it starts from takes most of the time)."""
    os.environ["NEKMESHPY_SMOOTH_OUTER"] = "1"
    try:
        return demo(name)
    finally:
        del os.environ["NEKMESHPY_SMOOTH_OUTER"]


def _surface_case(ns, out, prefix, n_quads, n_hexes):
    _valid(ns)
    assert isinstance(ns["mesh"], QuadMesh) and ns["hexes"].n_hexes == n_hexes       # a surface; the hexes are untouched
    assert ns["start"].n_quads == n_quads and ns["res"].counts()["on_surface"] == ns["inside"].size
    for run in ns["runs"].values():
        assert run["on"].passed and run["on"].max_frozen_move == 0.0
        assert run["q"][1].n_inverted == 0 and run["ov"][1] == 0
        assert np.array_equal(run["out"].points[ns["rim"]], run["before"].points[ns["rim"]])
        assert run["q"][1].sj_mean > run["q"][0].sj_mean
    for run in ("asis", "stress"):
        for tag in ("before", "after"):
            assert (out / ("%s_%s_%s.vtu" % (prefix, run, tag))).exists()


def test_cs_wire_coil(demo):
    ns, out = _fast(demo, "cs_wire_coil.py")          # one outer cycle here; the page is built with five
    _surface_case(ns, out, "cs_coil", 1344, 9248)
    assert ns["runs"]["stress"]["q"][0].n_inverted > 0                      # the perturbation inverts quads


def test_cs_u_tube(demo):
    ns, out = _fast(demo, "cs_u_tube.py")             # one outer cycle here; the page is built with five
    _surface_case(ns, out, "cs_utube", 2040, 26652)
    asis = ns["runs"]["asis"]
    assert asis["q"][0].sj_mean > 0.9 and asis["q"][1].sj_min > asis["q"][0].sj_min
    assert asis["size_ratio"][1] < asis["size_ratio"][0]
    # the outer box: five planes' worth of quads, planar as built (little to gain), the perturbed input recovers
    box = ns["box_runs"]
    assert ns["box_start"].n_quads == 5000 and ns["box_res"].counts()["on_surface"] > 4000
    for run in box.values():
        assert run["on"].passed and run["on"].max_frozen_move == 0.0
        assert run["q"][1].n_inverted == 0 and run["ov"][1] == 0
    assert box["asis"]["q"][0].sj_mean > 0.99                             # already nearly all squares
    assert box["stress"]["q"][0].sj_mean < 0.95 < box["stress"]["q"][1].sj_mean       # damaged, then repaired
    for run in ("asis", "stress"):
        for tag in ("before", "after"):
            assert (out / ("cs_utube_box_%s_%s.vtu" % (run, tag))).exists()


def test_cs_u_tube_hex(demo):
    ns, _ = _fast(demo, "cs_u_tube_hex.py")        # one outer cycle; the carried-in wall is judged on the finer lattice too
    built, carried = ns["readings"]["as built"], ns["readings"]["smoothed wall carried in"]
    assert built["n"] == carried["n"] == 26652 and built["corner"]["n_inv"] == 0 and built["fine"]["n_inv"] == 0
    assert built["faces"].keys() == carried["faces"].keys() and built["conforming"] and carried["conforming"]
    assert ns["built"].order == 1 and ns["sj_fine"]["as built"].shape == (26652,)
    w = ns["where"]
    assert w["bad_touch"] == w["n_bad"] == w["bad_fluid"] + w["bad_solid"]      # only the hexes next to the wall suffer
    assert w["off_min_built"] == w["off_min_carried"]                             # the rest is not touched
    assert ns["valid"] == (carried["corner"]["n_inv"] == 0 and carried["fine"]["n_inv"] == 0)
