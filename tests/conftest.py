"""Shared fixtures for the NekMeshPy regression suite.

The concrete geometry meshers are flat scripts in ``examples/``; the tests run
them with :func:`runpy.run_path` and read the resulting ``mesh`` global.  The
``built_mesh`` fixture runs ``examples/bifurcation.py`` once per session (into a
temp dir), returning the assembled :class:`~nekmeshpy.hexmesh.HexMesh`
plus its written ``.re2``/``.rea``/``.vtk`` paths.  Golden reference outputs live
in ``tests/golden/`` (a frozen snapshot of the validated results).
"""

import os
import runpy

import matplotlib
import numpy as np
import pytest

# the suite runs headless (viz tests import matplotlib), so pin a non-interactive
# backend here -- no MPLBACKEND=Agg needed on the command line.
matplotlib.use("Agg")

_HERE = os.path.dirname(__file__)
_EXAMPLES = os.path.join(_HERE, "..", "examples")
GOLDEN = os.path.join(_HERE, "golden")

# bundled ``car`` surface used by the bifurcation example
CAR_VTX = os.path.join(_EXAMPLES, "data", "car.vtx")
CAR_TRI = os.path.join(_EXAMPLES, "data", "car.tri")


def run_example(name, tmp_path):
    """Execute the flat example script ``examples/<name>`` in ``tmp_path`` and
    return its module namespace (``ns["mesh"]`` is the built HexMesh)."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return runpy.run_path(os.path.join(_EXAMPLES, name), run_name="__main__")
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def built_mesh(tmp_path_factory):
    out = tmp_path_factory.mktemp("mesh")
    ns = run_example("bifurcation.py", out)
    return {
        "mesh": ns["mesh"],
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
