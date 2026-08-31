"""The Nek5000 field-file writer (``writer.to_fld``).

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
from nekmeshpy.io import writer

LAYERS = np.array([0.0, 0.5, 1.0])


def _cht_pair():
    """A fluid block and a separate solid one, solid stored **first** -- the worst
    case for the fluid-first reorder.  Mirrors ``test_re2_fluid.py``'s own fixture,
    since the two writers' ``fluid=`` must reorder identically."""
    def _sec(x0):
        X, Y = np.meshgrid(np.linspace(x0, x0 + 1.0, 3), np.linspace(0.0, 1.0, 3),
                           indexing="ij")
        return quadmesh.from_grid(np.stack([X, Y, np.zeros_like(X)], axis=-1))

    def _slab(x0, tag):
        return hexmesh.extrude(_sec(x0), 2.0, 3, element_tags=tag)

    return hexmesh.merge([_slab(5.0, "solid"), _slab(0.0, "fluid")], tol=1e-9)


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
    writer.to_fld(mesh, path)
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
    writer.to_fld(mesh, path, time=1.5, istep=7)
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
    writer.to_fld(mesh, path)
    got = readnek(path)
    nodes, conn_ho = conformal(mesh)
    want = nodes[conn_ho]
    assert got["minmax"][..., 0] == pytest.approx(want.min(axis=1), abs=1e-6)
    assert got["minmax"][..., 1] == pytest.approx(want.max(axis=1), abs=1e-6)


def test_fld_single_precision(tmp_path):
    """``wdsz = 4`` halves the coordinate block and keeps it readable to float32."""
    mesh = _mesh(2)
    p8, p4 = str(tmp_path / "d0.f00001"), str(tmp_path / "s0.f00001")
    writer.to_fld(mesh, p8)
    writer.to_fld(mesh, p4, wdsz=4)
    g8, g4 = readnek(p8), readnek(p4)
    assert g4["wdsz"] == 4 and g8["wdsz"] == 8
    assert g4["data"] == pytest.approx(g8["data"], abs=1e-6)
    nodes, conn_ho = conformal(mesh)
    npel, nel = conn_ho.shape[1], conn_ho.shape[0]
    import os
    assert os.path.getsize(p8) - os.path.getsize(p4) == nel * npel * 3 * 4


def test_fld_rejects_bad_word_size(tmp_path):
    with pytest.raises(ValueError, match="wdsz"):
        writer.to_fld(_mesh(1), str(tmp_path / "m0.f00001"), wdsz=6)


def test_fld_fluid_none_matches_the_mesh_own_order(tmp_path):
    """The default -- unpermuted, agreeing with an unpermuted ``to_re2`` (``fluid=``
    also defaulting to ``None`` there)."""
    mesh = _cht_pair()
    path = str(tmp_path / "m0.f00001")
    writer.to_fld(mesh, path)
    got = readnek(path)
    nodes, conn_ho = conformal(mesh)
    assert np.array_equal(got["data"], nodes[conn_ho])


def test_fld_fluid_reorders_to_match_to_re2(tmp_path):
    """``fluid=`` must permute ``to_fld`` exactly the way it permutes ``to_re2``: Nek
    reads a restart/IC field file element by element against the ``.re2`` it already
    loaded, so a mismatch between the two files' numbering would silently misassign
    every element's geometry (and, in a real restart, its field data) onto the wrong
    piece of the mesh -- not merely stale, since both files are the same length and
    neither one knows it disagrees with the other.

    ``mesh`` stores its solid block first (``_cht_pair``'s own worst case for the
    reorder), so a ``to_fld`` that ignored ``fluid=`` would write element 0 as a solid
    corner while ``to_re2`` writes element 0 as a fluid corner -- exactly the kind of
    disagreement this checks for, by comparing each file's own corner block directly.
    """
    mesh = _cht_pair()
    re2_path = str(tmp_path / "m.re2")
    fld_path = str(tmp_path / "m0.f00001")
    writer.to_re2(mesh, re2_path, groups={"fluid_wall": "W  "}, fluid="fluid",
                  thermal={"fluid_wall": "I  ", "solid_wall": "I  "})
    writer.to_fld(mesh, fld_path, fluid="fluid")

    with open(re2_path, "rb") as f:
        hdr = f.read(80)
        f.read(4)
        n = int(hdr.split()[1])
        nelv = int(hdr.split()[3])
        re2_corners = (np.frombuffer(f.read(n * 25 * 8), "<f8")
                      .reshape(n, 25)[:, 1:].reshape(n, 3, 8).transpose(0, 2, 1))

    got = readnek(fld_path)
    assert got["nel"] == n
    assert nelv < n                                # the reorder actually did something
    # compare per-element centroids, not node-for-node: ``.re2``'s 8 corners are in
    # Nek's own preprocessor order while ``.fld``'s GLL block is lexicographic --
    # different node orderings within one element, but the same 8 points, so the
    # centroid is what proves the two files agree on which element is which.
    assert re2_corners.mean(axis=1) == pytest.approx(got["data"].mean(axis=1), abs=1e-9)
