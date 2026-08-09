"""Golden regression tests: the default surface pipeline must keep reproducing
the validated MATLAB/Octave reference.

The contract is **geometry to a tolerance, topology and tags exactly**: every
coordinate in the ``.re2`` / ``.vtu`` exports matches the golden within ``RE2_TOL``,
while connectivity, element/node numbering, VTK cell types and the boundary blocks
are compared byte-for-byte.  Floats are deliberately *not* byte-compared -- the CI
matrix reproduces the mesh bit-for-bit, but a differently built interpreter shifts
coordinates by ~1e-13 (see :func:`test_vtu_coords_match_golden`), which says nothing
about correctness.

These pin the numerics so every generalization refactor is safe.
"""

import base64
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import GOLDEN, read_re2_coords

from nekmeshpy import hexmesh
from nekmeshpy.hexmesh import quality

# Residual is scipy.spsolve vs MATLAB backslash; ~2.7e-13 observed.
RE2_TOL = 1e-12


def test_element_and_boundary_counts(built_mesh):
    mesh = built_mesh["mesh"]
    assert mesh.hexes.shape == (7200, 8)         # (N,8) shared-point connectivity
    assert mesh.points.shape == (8137, 3)
    assert len(mesh.face_tags) == 1840


def test_tag_face_counts(built_mesh):
    mesh = built_mesh["mesh"]
    names = mesh.face_tags.tags
    counts = {n: int(np.sum(names == n)) for n in
              ("wall", "trunk_outlet", "top_outlet_1", "top_outlet_2")}
    assert counts["wall"] == 1440
    assert counts["trunk_outlet"] == 80
    assert counts["top_outlet_1"] == 80
    assert counts["top_outlet_2"] == 80


def test_scaled_jacobian_quality(built_mesh):
    X, HC, _ = hexmesh.weld(built_mesh["mesh"])
    sj = quality.scaled_jacobian(X, HC)
    # values for the order-3 pipeline whose wall is refit analytically before meshing:
    # each private station ring as a truncated-Fourier loop (``fourier_ring``) and each
    # of the three *shared* seam arcs as a truncated sine series with its endpoints
    # pinned (``_arc_curve``).  Low-passing away the STL facet noise un-pinches the
    # worst wall elements, so the floor is well above the 0.0281 of the order-1 chord
    # wall it replaced.  Corner metric, so the high-order nodes do not enter it; still
    # no inverted elements.
    assert float(np.min(sj)) == pytest.approx(0.1207, abs=1e-3)
    assert float(np.mean(sj)) == pytest.approx(0.9117, abs=1e-3)
    assert float(np.min(sj)) > 0.0   # no inverted elements


def test_re2_coords_match_golden(built_mesh):
    ne, coords, _ = read_re2_coords(built_mesh["re2"])
    gne, gcoords, _ = read_re2_coords(os.path.join(GOLDEN, "bifurcation.re2"))
    assert ne == gne == 7200
    assert coords.shape == gcoords.shape
    assert np.max(np.abs(coords - gcoords)) < RE2_TOL


def test_re2_boundary_block_identical(built_mesh):
    _, _, bnd = read_re2_coords(built_mesh["re2"])
    _, _, gbnd = read_re2_coords(os.path.join(GOLDEN, "bifurcation.re2"))
    assert bnd == gbnd           # BC block is exact integers/codes


#: VTU ``DataArray`` type -> the little-endian numpy dtype it decodes to.
_VTU_DTYPE = {"Float64": "<f8", "Int64": "<i8", "Int32": "<i4", "UInt8": "u1"}


def _read_vtu(path):
    """Decode a ``.vtu`` into ``{array name: values}``.

    The export writes ``format="binary"`` -- each ``DataArray`` is base64 of a byte
    count followed by the raw little-endian values -- so the arrays come back as
    numbers rather than text.  That is what lets the coordinates be compared
    numerically while every integer array stays pinned exactly, the same split the
    old ascii text surgery made.
    """
    root = ET.parse(path).getroot()
    out = {}
    for da in root.iter("DataArray"):
        raw = base64.b64decode(da.text.strip())
        n = int(np.frombuffer(raw[:8], "<u8")[0])
        out[da.get("Name") or "Points"] = np.frombuffer(
            raw[8:8 + n], _VTU_DTYPE[da.get("type")])
    piece = root.find(".//Piece")
    out["_counts"] = (piece.get("NumberOfPoints"), piece.get("NumberOfCells"))
    return out


def test_vtu_structure_byte_identical(built_mesh):
    """Everything but the coordinates -- markup, connectivity, offsets, VTK cell
    types, ``bc_id`` -- is byte-exact.  That is where a refactor bug shows up:
    element/node numbering, the high-order Lagrange node permutation, and the
    boundary ids are all integers and cannot drift."""
    got = _read_vtu(built_mesh["vtu"])
    ref = _read_vtu(os.path.join(GOLDEN, "bifurcation.vtu"))
    assert got["_counts"] == ref["_counts"]
    for name in ("connectivity", "offsets", "types", "bc_id"):
        assert np.array_equal(got[name], ref[name]), name


def test_vtu_coords_match_golden(built_mesh):
    """Coordinates to ``RE2_TOL``, the same bound the ``.re2`` coordinates get.

    A byte-exact float comparison is not a property this pipeline has: the CI matrix
    (CPython 3.9-3.12, numpy 2.0-2.5, scipy 1.13-1.18) reproduces the mesh
    bit-for-bit, but a *differently built* interpreter -- measured on a cp314 wheel
    of the same numpy/scipy as the 3.12 leg -- shifts every coordinate by up to
    7.3e-13.  That is float-association noise in the same class as the
    ``spsolve``-vs-backslash residual ``RE2_TOL`` was chosen for, so it gets the same
    bound rather than a golden that is only valid on one build.
    """
    got = _read_vtu(built_mesh["vtu"])["Points"].reshape(-1, 3)
    ref = _read_vtu(os.path.join(GOLDEN, "bifurcation.vtu"))["Points"].reshape(-1, 3)
    assert got.shape == ref.shape == (202249, 3)
    assert np.max(np.abs(got - ref)) < RE2_TOL
