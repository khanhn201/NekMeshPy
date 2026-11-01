"""The MFEM/TMOP demos in ``examples/smoother/mfem/`` run, leave a valid ``mesh`` (linear for 01, curved for 02), and
the headline claims they make hold (asserted with margin, on the numbers the demo itself computed).  Skipped without PyMFEM."""

import os

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import HexMesh

pytest.importorskip("mfem.ser")   # not "mfem": this directory is itself importable as a namespace package


@pytest.fixture(scope="module")
def thin_layer(tmp_path_factory):
    return run_example(os.path.join("smoother", "mfem", "01_thin_layer.py"), tmp_path_factory.mktemp("01_thin_layer"))


def _valid(ns):
    mesh = ns["mesh"]
    assert isinstance(mesh, HexMesh) and 0 < mesh.n_hexes <= 500 and mesh.order == 1
    assert np.isfinite(mesh.points).all()


def test_01_thin_layer(thin_layer):
    ns = thin_layer
    _valid(ns)
    rows, side = ns["rows"], ns["side"]
    assert rows["native"]["thick_mean_pct"] > 50 and rows["ideal"]["thick_mean_pct"] > 50
    d = rows["input_designed"]
    assert abs(d["thick_mean_pct"]) < 10 and d["thick_max_abs_pct"] < 15
    assert d["q"].sj_min > max(rows["native"]["q"].sj_min, rows["ideal"]["q"].sj_min) + 0.1
    assert d["q"].sj_min > rows["input"]["q"].sj_min + 0.5
    assert all(r["q"].n_inverted == 0 for r in rows.values())
    assert side["input_plain"]["max_point_diff"] < 1e-6
    assert abs(side["ideal_designed"]["thick_mean_pct"]) < 0.5 * rows["ideal"]["thick_mean_pct"]


@pytest.fixture(scope="module")
def bent_pipe(tmp_path_factory):
    return run_example(os.path.join("smoother", "mfem", "02_bent_pipe.py"), tmp_path_factory.mktemp("02_bent_pipe"))


def test_02_bent_pipe(bent_pipe):
    """C1-C5 of the page, with the margins ``claims_curved`` in docs/_ext/gen_smoother_mfem_assets.py asserts."""
    ns = bent_pipe
    mesh = ns["mesh"]
    assert isinstance(mesh, HexMesh) and mesh.n_hexes == ns["N_HEX"] and mesh.order == ns["ORDER"]
    rows, checks = ns["rows"], ns["checks"]
    inp, o1, nat, tm = rows["input"], rows["order1"], rows["native"], rows["tmop"]
    # C1: curved TMOP repairs the mesh and keeps its order
    assert tm["order"] == ns["ORDER"] and tm["n_inv7"] == 0 and tm["n_inv_corner"] == 0
    assert tm["sj7_min"] > inp["sj7_min"] + 0.1 and tm["sj_nodes_min"] > inp["sj_nodes_min"] + 0.1
    assert tm["sj_corner_min"] > inp["sj_corner_min"] + 0.05     # backend.md: on the demo TMOP improves both readings
    # C2: the wall is held, the caps slide
    assert checks["wall_max_move"] == 0.0 and checks["cap_max_off"] < 1e-12
    assert tm["wall_dev_pct"] == inp["wall_dev_pct"]
    # C3: the order-1 route loses the wall
    assert o1["order"] == 1 and o1["wall_dev_pct"] > 100 * max(tm["wall_dev_pct"], 1e-4) and o1["vol_err_pct"] < -1
    # C4: vertex-only smoothing leaves the curved nodes' defect
    assert nat["order"] == ns["ORDER"] and nat["sj_corner_min"] > inp["sj_corner_min"] + 0.05
    assert tm["sj7_min"] > nat["sj7_min"] + 0.05 and tm["sj_nodes_mean"] > nat["sj_nodes_mean"] + 0.05
    assert nat["wall_dev_pct"] > tm["wall_dev_pct"]
    # C5: every result is conformal
    assert all(checks["conforming"].values())
    # the order-1 route reads highest (straight-sided)
    assert o1["sj7_min"] > max(tm["sj7_min"], nat["sj7_min"])


def test_02_bent_pipe_doc_excerpts():
    """The blocks the docs include from the script (``literalinclude`` between markers) are all there."""
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.join(root, "docs", "_ext"))
    from gen_smoother_mfem_assets import check_excerpts

    check_excerpts(os.path.join(root, "examples", "smoother", "mfem", "02_bent_pipe.py"))

