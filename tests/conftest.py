"""Shared fixtures for the NekMeshPy regression suite.

The ``built_mesh`` fixture runs the full default surface pipeline exactly once
per test session (into a temp dir, plotting disabled) and hands the resulting
:class:`~nekmeshpy.geometry.hexmesh.HexMesh` plus its written ``.re2``/``.rea``/``.vtk``
paths to every test.  Golden reference outputs live in ``tests/golden/`` (a
frozen snapshot of the validated MATLAB/Octave results).
"""

import os

import numpy as np
import pytest

from nekmeshpy import BifurcationMesher, Config

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")


@pytest.fixture(scope="session")
def built_mesh(tmp_path_factory):
    out = tmp_path_factory.mktemp("mesh")
    cwd = os.getcwd()
    os.chdir(out)
    try:
        cfg = Config()
        cfg.plot = False
        cfg.out_name = "bifurcation"
        mesh = BifurcationMesher(cfg).run()
    finally:
        os.chdir(cwd)
    return {
        "mesh": mesh,
        "cfg": cfg,
        "re2": os.path.join(out, "bifurcation.re2"),
        "rea": os.path.join(out, "bifurcation.rea"),
        "vtk": os.path.join(out, "bifurcation.vtk"),
    }


def read_re2_coords(path):
    """Return (n_elem, coords[n_elem*3*8] float64, bnd_block bytes)."""
    with open(path, "rb") as f:
        hdr = f.read(80)
        f.read(4)  # test float32
        num_elem = int(hdr.split()[1])
        # each element: 1 group double + 8x + 8y + 8z doubles = 25 doubles
        elem_block = np.fromfile(f, dtype="<f8", count=num_elem * 25)
        rest = f.read()
    coords = elem_block.reshape(num_elem, 25)[:, 1:]  # drop the group double
    return num_elem, coords, rest
