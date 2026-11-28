"""``.re2`` curve records: above order 1, every curved edge's midpoint as a Nek ``'m'``
record -- the stored edge polynomial read at its parametric midpoint, at any order."""

import numpy as np
import pytest
from conftest import read_re2_coords, read_re2_curves

from nekmeshpy import BoundaryConditions, hexmesh, linemesh, quadmesh, writer

_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def _section(order):
    sec = quadmesh.ogrid(linemesh.circle(1.0, 8, order=order), 2, 1)
    return quadmesh.tag_edges(sec, quadmesh.boundary_edges(sec), "wall")


def _corners(path, n, dim):
    with open(path, "rb") as f:
        f.read(84)
        rec = np.fromfile(f, dtype="<f8", count=n * (1 + dim * 2 ** dim))
    return rec.reshape(n, -1)[:, 1:].reshape(n, dim, 2 ** dim).transpose(0, 2, 1)


def _type(rec):
    return {r.tobytes().rstrip(b"\x00").decode() for r in rec[:, 7:8].copy()}


@pytest.mark.parametrize("order", [2, 3, 4])
def test_curved_by_default_and_midpoints_on_the_circle(tmp_path, order):
    mesh = hexmesh.extrude(_section(order), 1.0, 2)
    f = str(tmp_path / "c.re2")
    writer.to_re2(mesh, f, bc={"wall": "W  "})
    rec = read_re2_curves(f)
    X = _corners(f, len(mesh.corners), 3)
    assert len(rec) and _type(rec) == {"m"}
    on = 0
    for e, k, x, y, z, *_ in rec:
        a, b = X[int(e) - 1, _EDGES[int(k) - 1][0]], X[int(e) - 1, _EDGES[int(k) - 1][1]]
        assert np.linalg.norm([x, y, z] - 0.5 * (a + b)) < 0.5 * np.linalg.norm(a - b)
        if abs(np.hypot(*a[:2]) - 1) < 1e-9 and abs(np.hypot(*b[:2]) - 1) < 1e-9:
            # exact circle sampling: the stored wall polynomial interpolates the circle,
            # so its midpoint sits on it to the interpolation error
            assert abs(np.hypot(x, y) - 1.0) < (1e-12 if order == 2 else 1e-3)
            on += 1
    assert on == 8 * (1 + 2 + 1)              # 8 wall edges a ring; the middle ring is in 2 layers


def test_curved_false_and_linear_write_no_curves(tmp_path):
    ho = hexmesh.extrude(_section(3), 1.0, 2)
    lin = hexmesh.extrude(_section(1), 1.0, 2)
    writer.to_re2(ho, str(tmp_path / "a.re2"), bc={"wall": "W  "}, curved=False)
    writer.to_re2(lin, str(tmp_path / "b.re2"), bc={"wall": "W  "})
    writer.to_re2(lin, str(tmp_path / "c.re2"), bc={"wall": "W  "}, curved=True)
    for n in "abc":
        assert len(read_re2_curves(tmp_path / ("%s.re2" % n))) == 0
    assert (tmp_path / "b.re2").read_bytes() == (tmp_path / "c.re2").read_bytes()


def test_curves_leave_elements_and_boundary_untouched(tmp_path):
    mesh = hexmesh.extrude(_section(2), 1.0, 2)
    writer.to_re2(mesh, str(tmp_path / "s.re2"), bc={"wall": "W  "}, curved=False)
    writer.to_re2(mesh, str(tmp_path / "c.re2"), bc={"wall": "W  "})
    _, cs, bs = read_re2_coords(tmp_path / "s.re2")
    _, cc, bc = read_re2_coords(tmp_path / "c.re2")
    assert np.array_equal(cs, cc) and bs == bc


def test_curve_records_follow_the_fluid_first_order(tmp_path):
    def cyl(x0, region):
        m = hexmesh.extrude(_section(2), 1.0, 2, element_tags=region)
        m = hexmesh.tag_faces(m, np.flatnonzero(hexmesh.boundary_face_ids(m)), region + "_wall")
        return hexmesh.translate(m, (x0, 0.0, 0.0))
    mesh = hexmesh.merge([cyl(5.0, "solid"), cyl(0.0, "fluid")], tol=1e-9)   # solid stored first
    n = len(mesh.corners)
    f = str(tmp_path / "f.re2")
    writer.to_re2(mesh, f, fluid="fluid", bc=BoundaryConditions(
        velocity={"fluid_wall": "W  "}, temperature={"fluid_wall": "t  ", "solid_wall": "t  "}))
    X = _corners(f, n, 3)
    rec = read_re2_curves(f)
    assert len(rec)
    for e, k, x, y, z, *_ in rec:
        a, b = X[int(e) - 1, _EDGES[int(k) - 1][0]], X[int(e) - 1, _EDGES[int(k) - 1][1]]
        assert np.linalg.norm([x, y, z] - 0.5 * (a + b)) < 0.5 * np.linalg.norm(a - b)


def test_quad_re2_curved_by_default(tmp_path):
    sec = _section(3)
    f = str(tmp_path / "q.re2")
    writer.quad_to_re2(sec, f, bc={"wall": "W  "})
    rec = read_re2_curves(f)
    X = _corners(f, len(sec.corners), 2)
    assert len(rec) and set(rec[:, 1].astype(int)) <= {1, 2, 3, 4}
    on = 0
    for e, k, x, y, *_ in rec:
        a, b = X[int(e) - 1, _EDGES[int(k) - 1][0]], X[int(e) - 1, _EDGES[int(k) - 1][1]]
        assert np.linalg.norm([x, y] - 0.5 * (a + b)) < 0.5 * np.linalg.norm(a - b)
        if abs(np.hypot(*a) - 1) < 1e-9 and abs(np.hypot(*b) - 1) < 1e-9:
            assert abs(np.hypot(x, y) - 1.0) < 1e-3
            on += 1
    assert on == 8
