"""Smoke tests for the docs gallery's asset pipeline.

Not a re-run of every example (``test_examples.py`` already builds all of them, at
real cost) -- this checks the two things specific to the viewer: ``writer.boundary_to_vtp``
writes well-formed, non-empty PolyData, and ``gen_viewer_assets.py``'s skip list still
agrees with ``test_examples.py``'s.
"""

import os
import sys
import xml.etree.ElementTree as ET

from conftest import run_example

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs", "_ext"))

import gen_viewer_assets  # noqa: E402
from test_examples import EXCLUDED, LIBRARY_ONLY  # noqa: E402

from nekmeshpy.io import writer  # noqa: E402


def test_boundary_to_vtp_writes_well_formed_nonempty_polydata(tmp_path):
    ns = run_example("circular_pipe.py", tmp_path)
    mesh = ns["mesh"]

    out = tmp_path / "circular_pipe.vtp"
    writer.boundary_to_vtp(mesh, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    root = ET.parse(str(out)).getroot()
    assert root.tag == "VTKFile"
    poly = root.find(".//Polys")
    assert poly is not None


def test_skip_list_is_a_superset_of_library_only():
    """`gen_viewer_assets.py` must skip everything `test_examples.py` treats as
    library-only, or asset generation crashes on a script with no `mesh` global."""
    assert LIBRARY_ONLY <= gen_viewer_assets.SKIP


def test_gen_viewer_assets_skips_the_gmsh_example():
    """``femoral`` is out of the test harness for cost; it is out of asset generation
    for the same reason, and the two lists must not drift apart."""
    assert EXCLUDED <= gen_viewer_assets.SKIP
