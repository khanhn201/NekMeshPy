"""Smoke tests for the docs gallery's asset pipeline.

Not a re-run of every example (``test_examples.py`` already builds all of them, at
real cost) -- this checks the two things specific to the viewer: ``writer.boundary_to_vtp``
writes well-formed, non-empty PolyData, and every example the gallery embeds is one
``gen_viewer_assets.py`` can build.
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
    ns = run_example("circular_pipe_tjunction.py", tmp_path)
    mesh = ns["mesh"]

    out = tmp_path / "circular_pipe_tjunction.vtp"
    writer.boundary_to_vtp(mesh, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
    root = ET.parse(str(out)).getroot()
    assert root.tag == "VTKFile"
    poly = root.find(".//Polys")
    assert poly is not None


def test_gallery_examples_are_buildable_by_the_harness():
    """``gen_viewer_assets.py`` builds exactly the examples the gallery names; each must
    exist and must not be one ``test_examples.py`` treats as library-only or excluded
    (no ``mesh`` global, or too costly to run)."""
    names = gen_viewer_assets._examples()
    assert names
    assert not set(names) & (LIBRARY_ONLY | EXCLUDED)
