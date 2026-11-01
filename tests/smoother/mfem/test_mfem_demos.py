"""The MFEM/TMOP demo in ``examples/smoother/mfem/`` runs, leaves a valid linear ``mesh``, and the headline claims
it makes hold (asserted with margin, on the numbers the demo itself computed).  Skipped without PyMFEM."""

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
