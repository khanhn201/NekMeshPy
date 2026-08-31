"""``hexmesh.periodic_pairs`` and ``to_re2(..., periodic=)`` -- the face-to-face
correspondence a Nek ``'P  '`` boundary row is meaningless without.

The distinction from ``attach``: both are *told* which group meets which and both prove
the pairing by bijectivity rather than by a tolerance, but ``attach``'s two halves are
meant to end up in the same place, so nearest-neighbour reads the correspondence
directly.  A periodic pair's halves sit a lattice vector apart, so the transform has to
be stated -- and stating it is what makes the pairing checkable, which is the property
these tests lean on hardest.
"""

import numpy as np
import pytest
from conftest import read_re2_boundary, read_re2_periodic

from nekmeshpy import hexmesh, writer
from nekmeshpy.core import affine
from nekmeshpy.hexmesh import Periodic

LZ = 2.0
SECTOR = np.pi / 4


def _box(nz=4, tags=None):
    """A ``1 x 1 x LZ`` grid box with every side named."""
    X = np.stack(np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3),
                             np.linspace(0.0, LZ, nz), indexing="ij"), axis=-1)
    return hexmesh.from_grid(X, element_tag="fluid", side_tags=tags or {
        "z_min": "inlet", "z_max": "outlet", "x_min": "wall", "x_max": "wall",
        "y_min": "wall", "y_max": "wall"})


def _wedge():
    """An annular sector: its two radial cut planes are each other's rotation."""
    r, th, z = (np.linspace(1.0, 2.0, 3), np.linspace(0.0, SECTOR, 5),
                np.linspace(0.0, 1.0, 2))
    R, TH, Z = np.meshgrid(r, th, z, indexing="ij")
    return hexmesh.from_grid(
        np.stack([R * np.cos(TH), R * np.sin(TH), Z], axis=-1), element_tag="fluid",
        side_tags={"y_min": "cut_lo", "y_max": "cut_hi", "x_min": "inner",
                   "x_max": "outer", "z_min": "bot", "z_max": "top"})


def _axial(dz=LZ):
    return Periodic("inlet", "outlet", affine.translation([0.0, 0.0, dz]))


# -- the pairing itself -------------------------------------------------
def test_translation_pairs_every_face_both_ways():
    """Both directions come out of the one table, which is what makes them agree."""
    mesh = _box()
    n_end = int(hexmesh.tagged_faces(mesh, "inlet").size)
    pairs = hexmesh.periodic_pairs(mesh, [_axial()])

    assert pairs.rows.shape == (2 * n_end, 4)
    assert pairs.worst == 0.0                      # a pure z shift of a z-uniform grid
    partner = pairs.partner_of()
    assert len(partner) == 2 * n_end
    # every row's partner is itself a row, and points back
    assert all(partner[v] == k for k, v in partner.items())
    # ... and the two sides really are z_min (Nek face 5) and z_max (face 6)
    assert {f for _, f in partner} == {5, 6}


def test_rotation_pairs_a_wedge():
    mesh = _wedge()
    pairs = hexmesh.periodic_pairs(
        mesh, [Periodic("cut_lo", "cut_hi", affine.rotation(SECTOR))])
    partner = pairs.partner_of()
    assert len(partner) == 2 * int(hexmesh.tagged_faces(mesh, "cut_lo").size)
    assert all(partner[v] == k for k, v in partner.items())
    assert pairs.worst < pairs.tol


def test_the_pairing_is_geometric_not_positional():
    """The correspondence follows the geometry, not the order the faces are stored in.

    Reversing one group's id order must not change a single pair -- ``locate_rows``
    reads corner *sets*, so what pairs a face is where its corners land, not which slot
    it sits in."""
    mesh = _box()
    fwd = hexmesh.periodic_pairs(mesh, [_axial()]).partner_of()
    rev = hexmesh.periodic_pairs(mesh, [Periodic(
        hexmesh.tagged_faces(mesh, "inlet"),
        hexmesh.tagged_faces(mesh, "outlet")[::-1],
        affine.translation([0.0, 0.0, LZ]))]).partner_of()
    assert fwd == rev


def test_no_specs_is_no_pairs():
    pairs = hexmesh.periodic_pairs(_box(), [])
    assert pairs.rows.shape == (0, 4)
    assert pairs.partner_of() == {}


def test_the_mesh_is_untouched():
    """A pairing states a fact about the mesh; it does not weld, rename or move."""
    mesh = _box()
    before = mesh.points.copy()
    hexmesh.periodic_pairs(mesh, [_axial()])
    assert np.array_equal(mesh.points, before)
    assert sorted(mesh.face_group_tags) == ["inlet", "outlet", "wall"]
    assert hexmesh.is_watertight(mesh)


# -- what a wrong statement does ----------------------------------------
def test_a_wrong_transform_raises_with_its_residual():
    """The whole point of stating the map: a mis-typed pitch is caught here, not by the
    solver reading a mesh welded shut in the wrong place."""
    with pytest.raises(ValueError, match="worst residual of 1.000e-01"):
        hexmesh.periodic_pairs(_box(), [_axial(dz=LZ - 0.1)])


def test_the_transform_must_carry_a_onto_b():
    with pytest.raises(ValueError, match="one-to-one|worst residual"):
        hexmesh.periodic_pairs(_box(), [_axial(dz=-LZ)])


def test_unequal_face_counts_raise():
    with pytest.raises(ValueError, match="different face counts"):
        hexmesh.periodic_pairs(_box(), [Periodic("inlet", "wall", affine.translation(
            [0.0, 0.0, LZ]))])


def test_a_group_cannot_pair_with_itself():
    """One name over both ends cannot say which end is which, so it is refused rather
    than split on a coordinate guess."""
    with pytest.raises(ValueError, match="already claimed"):
        hexmesh.periodic_pairs(_box(), [Periodic("inlet", "inlet", affine.translation(
            [0.0, 0.0, LZ]))])


def test_a_face_may_be_claimed_by_one_spec_only():
    with pytest.raises(ValueError, match=r"specs\[1\].*already claimed"):
        hexmesh.periodic_pairs(_box(), [_axial(), _axial()])


def test_a_buried_face_is_not_a_boundary():
    """A periodic face is a domain boundary by definition; an interior plane that kept
    its name is the mistake this catches."""
    X = np.stack(np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3),
                             np.linspace(0.0, LZ, 3), indexing="ij"), axis=-1)
    a = hexmesh.from_grid(X, element_tag="f", side_tags={"z_max": "seam"})
    b = hexmesh.translate(
        hexmesh.from_grid(X, element_tag="f", side_tags={"z_min": "seam"}),
        [0.0, 0.0, LZ])
    welded = hexmesh.merge([a, b], tol=1e-9)
    with pytest.raises(ValueError, match="carry a hex on both sides"):
        hexmesh.periodic_pairs(welded, [Periodic(
            "seam", "seam", affine.translation([0.0, 0.0, LZ]))])


def test_differently_refined_sides_raise():
    """Equal *face* counts with unequal point counts has no correspondence to find, and
    says so rather than reporting the counts that happen to match.

    A 2x2 patch and a 1x4 strip are both four faces, and 9 points against 10."""
    def block(nx, ny, dx, tag):
        X = np.stack(np.meshgrid(np.linspace(dx, dx + 1.0, nx + 1),
                                 np.linspace(0.0, 1.0, ny + 1),
                                 np.linspace(0.0, LZ, 2), indexing="ij"), axis=-1)
        return hexmesh.from_grid(X, element_tag="f", side_tags={"z_max": tag})

    mesh = hexmesh.merge([block(2, 2, 0.0, "patch"), block(1, 4, 9.0, "strip")],
                         tol=1e-9)
    with pytest.raises(ValueError, match="not the same surface"):
        hexmesh.periodic_pairs(mesh, [Periodic(
            "patch", "strip", affine.translation([9.0, 0.0, 0.0]))])


# -- the .re2 rows ------------------------------------------------------
GROUPS_P = {"wall": "W  ", "inlet": "P  ", "outlet": "P  "}


def test_re2_writes_the_partner_element_and_face(tmp_path):
    """``bc(1)`` / ``bc(2)`` carry the partner, 1-based on the element as every other
    re2 element id is, and the file's own rows are mutually consistent."""
    mesh = _box()
    path = str(tmp_path / "box.re2")
    writer.to_re2(mesh, path, groups=GROUPS_P, periodic=[_axial()])

    from_file = read_re2_periodic(path)
    assert len(from_file) == 2 * int(hexmesh.tagged_faces(mesh, "inlet").size)
    assert all(from_file[v] == k for k, v in from_file.items())
    # the same table the pairing produced, shifted to the file's 1-based element ids
    expected = {(e + 1, f): (pe + 1, pf) for (e, f), (pe, pf)
                in hexmesh.periodic_pairs(mesh, [_axial()]).partner_of().items()}
    assert from_file == expected
    # and every P row is a row read_re2_boundary sees too, with the P code
    codes = read_re2_boundary(path)
    assert all(codes[(e, f, "P  ")] == 1 for e, f in from_file)


def test_a_non_periodic_mesh_leaves_the_partner_fields_zero(tmp_path):
    """The widened row must not change a file that names no periodic face -- which is
    what keeps the golden regression's boundary block byte-identical."""
    mesh = _box()
    plain = str(tmp_path / "plain.re2")
    writer.to_re2(mesh, plain, groups={"wall": "W  ", "inlet": "v  ", "outlet": "O  "})
    assert read_re2_periodic(plain) == {}
    with open(plain, "rb") as f:
        raw = f.read()
    n = mesh.n_hexes
    rows = np.frombuffer(raw[84 + n * 200 + 16:], dtype="<f8").reshape(-1, 8)
    assert np.all(rows[:, 2:7] == 0.0)


# -- groups and periodic= must agree ------------------------------------
def test_a_P_code_without_a_spec_is_refused(tmp_path):
    """It would write partner element 0, face 0 -- a mesh Nek accepts and mis-solves."""
    with pytest.raises(ValueError, match="must be the same set"):
        writer.to_re2(_box(), str(tmp_path / "x.re2"), groups=GROUPS_P)


def test_a_spec_without_a_P_code_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must be the same set"):
        writer.to_re2(_box(), str(tmp_path / "x.re2"),
                      groups={"wall": "W  ", "inlet": "v  ", "outlet": "O  "},
                      periodic=[_axial()])


def test_naming_only_one_half_periodic_is_refused(tmp_path):
    with pytest.raises(ValueError, match="must be the same set"):
        writer.to_re2(_box(), str(tmp_path / "x.re2"),
                      groups={"wall": "W  ", "inlet": "P  ", "outlet": "O  "},
                      periodic=[_axial()])


def test_a_resolved_pairing_may_be_passed_straight_in(tmp_path):
    """``periodic=`` takes the report as readily as the specs, so a mesher that wants to
    inspect or log the pairing does not compute it twice."""
    mesh = _box()
    pairs = hexmesh.periodic_pairs(mesh, [_axial()])
    a, b = str(tmp_path / "a.re2"), str(tmp_path / "b.re2")
    writer.to_re2(mesh, a, groups=GROUPS_P, periodic=[_axial()])
    writer.to_re2(mesh, b, groups=GROUPS_P, periodic=pairs)
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()


def test_vtu_still_paints_a_periodic_group(tmp_path):
    """A periodic name is an ordinary group to the viewer -- ``bc_id`` by id, no code."""
    mesh = _box()
    path = str(tmp_path / "box.vtu")
    writer.to_vtu(mesh, path, groups=GROUPS_P)
    with open(path) as f:
        assert 'Name="bc_id"' in f.read()
