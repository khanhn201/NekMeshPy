"""The Nek5000 field-file writer (``export.to_fld``).

``.re2`` has no high-order format, so it ships only the 8 corners of each hex; the
field file (``<prefix>0.f00001``) stores the full ``lx1*ly1*lz1`` GLL block and is
therefore the export that actually preserves an ``order = N`` mesh.  These tests read
the bytes back with an independent reader written straight from the format spec
(``#std`` header, endian tag, ``int32`` element map, field data, min/max metadata) --
deliberately *not* by calling any toolkit code -- and check the round trip.
"""

import struct

import numpy as np
import pytest
from conftest import conformal

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.io import export

LAYERS = np.array([0.0, 0.5, 1.0])


def readnek(path):
    """Independent reader for the Nek ``#std`` field format.

    Returns the header fields plus ``data`` as ``(nel, npel, 3)`` -- i.e. transposed
    back out of the on-disk per-element ``x``-block / ``y``-block / ``z``-block layout.
    """
    with open(path, "rb") as f:
        header = f.read(132).decode("ascii")
        parts = header.split()
        assert parts[0] == "#std"
        wdsz, lx1, ly1, lz1, nel, nelf = (int(v) for v in parts[1:7])
        time, istep = float(parts[7]), int(parts[8])
        fid, nf, fields = int(parts[9]), int(parts[10]), parts[11]
        etag = struct.unpack("<f", f.read(4))[0]
        elmap = np.frombuffer(f.read(4 * nel), dtype="<i4")
        npel = lx1 * ly1 * lz1
        real = "<f8" if wdsz == 8 else "<f4"
        data = np.frombuffer(f.read(nel * npel * 3 * wdsz),
                             dtype=real).reshape(nel, 3, npel)
        minmax = np.frombuffer(f.read(nel * 3 * 2 * 4), dtype="<f4").reshape(nel, 3, 2)
        trailing = f.read()
    return dict(header=header, wdsz=wdsz, lr1=(lx1, ly1, lz1), nel=nel, nelf=nelf,
                time=time, istep=istep, fid=fid, nf=nf, fields=fields, etag=etag,
                elmap=elmap, data=data.transpose(0, 2, 1), minmax=minmax,
                trailing=trailing)


def _mesh(order):
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=order), 2, LAYERS)
    return hexmesh.extrude(section, length=1.0, layers=LAYERS)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_fld_round_trips_the_gll_block(tmp_path, order):
    """Every GLL node survives the write, in Nek's own per-element ordering.

    The expected block is the conformal walk's ``nodes[conn_ho]``, which is
    lexicographic with ``i`` fastest -- exactly Nek's node ordering -- so the writer
    must hand it over with no permutation at all.  A wrong permutation would leave the
    corners right and scramble everything else, so this only bites at ``order > 1``.
    """
    mesh = _mesh(order)
    assert mesh.order == order
    path = str(tmp_path / "m0.f00001")
    export.to_fld(mesh, path)
    got = readnek(path)
    nodes, conn_ho = conformal(mesh)
    want = nodes[conn_ho]
    assert got["lr1"] == (order + 1,) * 3
    assert got["nel"] == want.shape[0]
    assert np.array_equal(got["data"], want)     # double precision: bit-exact


def test_fld_header_and_framing(tmp_path):
    """Header layout, endian tag, element map and the trailing metadata block.

    The header is pinned as a literal against the reference ``writenek.m`` format
    string ``'#std %1i %2i %2i %2i %10i %10i %20.13E %9i %6i %6i %s\\n'`` padded to
    132 bytes -- field widths included, since Nek's reader is position-tolerant only
    because every writer agrees on them.
    """
    mesh = _mesh(3)
    path = str(tmp_path / "m0.f00001")
    export.to_fld(mesh, path, time=1.5, istep=7)
    got = readnek(path)
    assert len(got["header"]) == 132
    assert got["header"] == (
        "#std 8  4  4  4 %10d %10d  1.5000000000000E+00 %9d %6d %6d X\n"
        % (mesh.n_hexes, mesh.n_hexes, 7, 0, 1)).ljust(132)
    assert got["etag"] == pytest.approx(6.54321, abs=1e-6)   # little-endian marker
    assert np.array_equal(got["elmap"], np.arange(1, mesh.n_hexes + 1))
    assert got["fields"] == "X"      # a mesh has geometry and nothing else
    assert got["trailing"] == b""    # metadata block is the last thing in the file


def test_fld_metadata_is_the_per_element_extent(tmp_path):
    """The 3-D trailing block is per element, per component, ``min`` then ``max``,
    always single precision regardless of ``wdsz``."""
    mesh = _mesh(3)
    path = str(tmp_path / "m0.f00001")
    export.to_fld(mesh, path)
    got = readnek(path)
    nodes, conn_ho = conformal(mesh)
    want = nodes[conn_ho]
    assert got["minmax"][..., 0] == pytest.approx(want.min(axis=1), abs=1e-6)
    assert got["minmax"][..., 1] == pytest.approx(want.max(axis=1), abs=1e-6)


def test_fld_single_precision(tmp_path):
    """``wdsz = 4`` halves the coordinate block and keeps it readable to float32."""
    mesh = _mesh(2)
    p8, p4 = str(tmp_path / "d0.f00001"), str(tmp_path / "s0.f00001")
    export.to_fld(mesh, p8)
    export.to_fld(mesh, p4, wdsz=4)
    g8, g4 = readnek(p8), readnek(p4)
    assert g4["wdsz"] == 4 and g8["wdsz"] == 8
    assert g4["data"] == pytest.approx(g8["data"], abs=1e-6)
    nodes, conn_ho = conformal(mesh)
    npel, nel = conn_ho.shape[1], conn_ho.shape[0]
    import os
    assert os.path.getsize(p8) - os.path.getsize(p4) == nel * npel * 3 * 4


def test_fld_rejects_bad_word_size(tmp_path):
    with pytest.raises(ValueError, match="wdsz"):
        export.to_fld(_mesh(1), str(tmp_path / "m0.f00001"), wdsz=6)
