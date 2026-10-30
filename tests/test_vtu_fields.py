"""Extra named arrays in the ``.vtu`` writers: no meshio needed to write, shapes validated."""
import re

import numpy as np
import pytest

from nekmeshpy import quadmesh
from nekmeshpy.io import writer


def _mesh():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], float)
    return quadmesh.QuadMesh.from_corners(P, np.array([[0, 1, 2, 3], [1, 4, 5, 2]]))


@pytest.mark.parametrize("binary", [False, True])
def test_point_and_cell_arrays_are_written(tmp_path, binary):
    m = _mesh()
    f = str(tmp_path / "a.vtu")
    writer.quad_to_vtu(m, f, binary=binary,
                       point_data={"s": np.arange(6.0), "d": np.zeros((6, 3)), "k": np.arange(6)},
                       cell_data={"q": np.array([0.5, 1.5])})
    txt = open(f).read()
    for name in ("s", "d", "k", "q"):
        assert 'Name="%s"' % name in txt
    assert 'Name="d" NumberOfComponents="3"' in txt
    assert re.search(r'Int32" Name="k"', txt) and re.search(r'Float64" Name="s"', txt)
    assert '<PointData Scalars="s">' in txt and '<CellData Scalars="q">' in txt


def test_ascii_values_round_trip(tmp_path):
    m = _mesh()
    f = str(tmp_path / "a.vtu")
    writer.quad_to_vtu(m, f, binary=False, point_data={"s": np.arange(6.0)})
    body = re.search(r'Name="s" format="ascii">\n(.*?)</DataArray>', open(f).read(), re.S).group(1)
    got = np.array(body.split(), float)
    assert np.array_equal(got, np.arange(6.0)[m.corners].ravel())   # repeated on each corner copy


def test_no_extras_leaves_the_file_unchanged(tmp_path):
    m = _mesh()
    a, b = str(tmp_path / "a.vtu"), str(tmp_path / "b.vtu")
    writer.quad_to_vtu(m, a)
    writer.quad_to_vtu(m, b, point_data={}, cell_data=None)
    assert open(a).read() == open(b).read() and "PointData" not in open(a).read()


def test_wrong_length_names_the_array(tmp_path):
    with pytest.raises(ValueError, match="'bad'"):
        writer.quad_to_vtu(_mesh(), str(tmp_path / "a.vtu"), point_data={"bad": np.zeros(5)})
    with pytest.raises(ValueError, match="'bad'"):
        writer.quad_to_vtu(_mesh(), str(tmp_path / "a.vtu"), cell_data={"bad": np.zeros(3)})


def test_reserved_name_raises(tmp_path):
    with pytest.raises(ValueError, match="element_tag"):
        writer.quad_to_vtu(_mesh(), str(tmp_path / "a.vtu"), cell_data={"element_tag": np.zeros(2)})


def test_writing_does_not_import_meshio(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "meshio", None)                  # any import of it now fails
    writer.quad_to_vtu(_mesh(), str(tmp_path / "a.vtu"), cell_data={"q": np.zeros(2)})
