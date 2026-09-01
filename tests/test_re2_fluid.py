"""``to_re2(..., fluid=, thermal=)`` -- the multi-field boundary blocks a Nek5000
conjugate heat transfer run needs.

Nek's ``.re2`` header carries two element counts, ``nelgt`` (total) and ``nelgv``
(velocity mesh): a run with ``nelgv < nelgt`` solves momentum on the first ``nelgv``
elements only, listed **first**, contiguously. That alone was the first defect fixed
here (``fluid=``): before it, ``to_re2`` always wrote ``nelgv == nelgt``.

The deeper one, found by actually running a mesh through Nek5000 (``ierr=4`` reading
the boundary block): its reader (``core/reader_re2.f``) *always* reads one boundary
block per field once ``nelgt > nelgv``, regardless of the header's own field count --
so a conjugate mesh needs a **second** block (temperature), covering every element,
which ``groups=`` alone cannot describe since it is scoped to the velocity field only
once ``fluid=`` is in play. ``thermal=`` supplies that second block.
"""

import struct

import numpy as np
import pytest
from conftest import read_re2_periodic

from nekmeshpy import hexmesh, quadmesh, writer
from nekmeshpy.core import affine
from nekmeshpy.hexmesh import Periodic


def _sec(x0=0.0):
    X, Y = np.meshgrid(np.linspace(x0, x0 + 1.0, 3), np.linspace(0.0, 1.0, 3),
                       indexing="ij")
    P = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    return quadmesh.from_grid(P)


def _slab(x0, tag, **kw):
    """A ``1 x 1 x 2`` hex slab tagged ``tag``, offset to ``[x0, x0+1]`` in x."""
    return hexmesh.extrude(_sec(x0), 2.0, 3, element_tags=tag, **kw)


def _walled(mesh, name="wall"):
    """Every domain-boundary face named ``name`` -- enough to give ``to_re2`` a code
    for every row without caring which face is which.  Safe only for a single-region
    mesh: naming a face on **both** regions the same way is exactly what a scoped
    velocity ``groups=`` must not do once there is a solid region to keep out."""
    return hexmesh.tag_faces(mesh, np.flatnonzero(hexmesh.boundary_face_ids(mesh)), name)


def _cht_pair():
    """A fluid block and a separate solid one (no shared face -- a coincident one
    would have to carry one name, and each side names its *own* whole boundary here),
    solid stored **first** -- the worst case for the fluid-first reorder.  Each
    region's boundary is named separately (``fluid_wall`` / ``solid_wall``), the way a
    real velocity/thermal split needs -- ``fluid_wall`` alone is a valid *velocity*
    ``groups=`` name."""
    fluid = _slab(0.0, "fluid")
    solid = _slab(5.0, "solid")
    fluid = hexmesh.tag_faces(
        fluid, np.flatnonzero(hexmesh.boundary_face_ids(fluid)), "fluid_wall")
    solid = hexmesh.tag_faces(
        solid, np.flatnonzero(hexmesh.boundary_face_ids(solid)), "solid_wall")
    return hexmesh.merge([solid, fluid], tol=1e-9)


def _read_header(path):
    with open(path, "rb") as f:
        hdr = f.read(80).decode("ascii")
    return int(hdr[5:21]), int(hdr[24:40]), int(hdr[40:44])   # nelgt, nelgv, nBCre2


def _corners(path, n):
    with open(path, "rb") as f:
        f.read(84)
        E = np.frombuffer(f.read(n * 200), "<f8").reshape(n, 25)
    return np.stack([E[:, 1:9], E[:, 9:17], E[:, 17:25]], axis=-1)   # (n,8,3)


def _bc_blocks(path, n_hexes, n_blocks):
    """The raw ``(elem, face, ..., code)`` rows of every boundary block, in file
    order -- the general reader ``read_re2_periodic``/``read_re2_boundary`` don't
    give (they read exactly one block's worth), which is the point here."""
    with open(path, "rb") as f:
        raw = f.read()
    off = 84 + n_hexes * 200 + 8       # skip header/elements/the ncurve double
    blocks = []
    for _ in range(n_blocks):
        n_bnd = int(struct.unpack("<d", raw[off:off + 8])[0])
        off += 8
        rows = np.frombuffer(raw[off:off + n_bnd * 64], "<f8").reshape(n_bnd, 8)
        off += n_bnd * 64
        blocks.append(rows)
    return blocks


# -- the reorder itself --------------------------------------------------
def test_fluid_none_writes_every_element_as_velocity_mesh(tmp_path):
    """The default: unchanged from before ``fluid=`` existed."""
    mesh = _walled(_cht_pair())
    path = str(tmp_path / "plain.re2")
    writer.to_re2(mesh, path, groups={"wall": "W  "})
    nelgt, nelgv, nbcre2 = _read_header(path)
    assert nelgt == nelgv == mesh.n_hexes
    assert nbcre2 == 1
    # and unpermuted: element i in the file is element i of the mesh
    assert np.allclose(_corners(path, mesh.n_hexes), mesh.points[mesh.corners])


def test_fluid_puts_the_named_region_first(tmp_path):
    """Regardless of storage order (solid first here), the file's elements 1..nelgv
    are exactly the ``"fluid"``-tagged ones."""
    mesh = _cht_pair()
    regs = mesh.element_tags.dense(mesh.n_hexes)
    n_fluid = int((regs == "fluid").sum())
    path = str(tmp_path / "cht.re2")
    writer.to_re2(mesh, path, groups={"fluid_wall": "W  "}, fluid="fluid",
                  thermal={"fluid_wall": "I  ", "solid_wall": "I  "})

    nelgt, nelgv, nbcre2 = _read_header(path)
    assert nelgt == mesh.n_hexes
    assert nelgv == n_fluid
    assert nbcre2 == 2

    corners = _corners(path, mesh.n_hexes)
    fluid_x = mesh.points[mesh.corners[regs == "fluid"]][..., 0]
    solid_x = mesh.points[mesh.corners[regs == "solid"]][..., 0]
    assert corners[:nelgv, :, 0].max() <= fluid_x.max() + 1e-9
    assert corners[:nelgv, :, 0].min() >= fluid_x.min() - 1e-9
    assert corners[nelgv:, :, 0].min() >= solid_x.min() - 1e-9


def test_fluid_that_is_everything_is_the_same_as_none(tmp_path):
    """Naming a region that covers the whole mesh is a no-op reorder: nelgv == nelgt,
    same as leaving ``fluid=`` off, and no second field is required."""
    mesh = _walled(_slab(0.0, "fluid"))
    a, b = str(tmp_path / "a.re2"), str(tmp_path / "b.re2")
    writer.to_re2(mesh, a, groups={"wall": "W  "})
    writer.to_re2(mesh, b, groups={"wall": "W  "}, fluid="fluid")
    with open(a, "rb") as fa, open(b, "rb") as fb:
        assert fa.read() == fb.read()


def test_an_unknown_fluid_tag_raises(tmp_path):
    mesh = _cht_pair()
    with pytest.raises(ValueError, match="no element carries the tag"):
        writer.to_re2(mesh, str(tmp_path / "x.re2"), groups={"fluid_wall": "W  "},
                      fluid="nope")


# -- the second (thermal) field ------------------------------------------
def test_thermal_is_required_once_fluid_makes_a_solid_region(tmp_path):
    mesh = _cht_pair()
    with pytest.raises(ValueError, match="thermal= is required"):
        writer.to_re2(mesh, str(tmp_path / "x.re2"), groups={"fluid_wall": "W  "},
                      fluid="fluid")


def test_thermal_is_rejected_when_there_is_no_solid_region(tmp_path):
    mesh = _walled(_slab(0.0, "fluid"))
    with pytest.raises(ValueError, match="only one field's block"):
        writer.to_re2(mesh, str(tmp_path / "x.re2"), groups={"wall": "W  "},
                      fluid="fluid", thermal={"wall": "I  "})


def _named_interface(mesh):
    """The one shared face between the mesh's ``"fluid"`` and ``"solid"`` regions,
    named ``"interface"`` -- the conjugate-wall pattern every CHT mesher in the repo
    uses (``wire_coil.py``, ``rod_bundle.py``)."""
    regs = mesh.element_tags.dense(mesh.n_hexes)
    hexes = np.asarray(mesh.hexes)
    owners = np.full((mesh.quad_mesh.n_quads, 2), -1)
    slot = np.zeros(mesh.quad_mesh.n_quads, dtype=int)
    for e in range(mesh.n_hexes):
        for q in hexes[e]:
            owners[q, slot[q]] = e
            slot[q] += 1
    interior = owners[:, 1] >= 0
    cross = interior & (regs[owners[:, 0].clip(min=0)] != regs[owners[:, 1].clip(min=0)])
    return hexmesh.tag_faces(mesh, np.flatnonzero(cross), "interface")


def test_a_name_absent_from_thermal_gets_no_row_and_no_warning(tmp_path):
    """A conjugate interface: an explicit wall on the fluid side of velocity, and
    conformal (no row at all) for temperature -- the physically ordinary case, and
    not a typo to warn about."""
    fluid = _slab(0.0, "fluid")
    solid = _slab(1.0, "solid")
    mesh = _named_interface(hexmesh.merge([solid, fluid], tol=1e-9))
    # the two regions' *own* remaining boundary, named apart -- "fluid_wall" alone is
    # a valid velocity name, exactly as a real conjugate mesh's would be
    regs = mesh.element_tags.dense(mesh.n_hexes)
    bnd = np.flatnonzero(hexmesh.boundary_face_ids(mesh))
    hexes = np.asarray(mesh.hexes)
    face_owner = np.full(mesh.quad_mesh.n_quads, -1)
    for e in range(mesh.n_hexes):
        for q in hexes[e]:
            if face_owner[q] < 0:
                face_owner[q] = e
    fluid_bnd = bnd[regs[face_owner[bnd]] == "fluid"]
    solid_bnd = bnd[regs[face_owner[bnd]] == "solid"]
    mesh = hexmesh.tag_faces(mesh, fluid_bnd, "fluid_wall")
    mesh = hexmesh.tag_faces(mesh, solid_bnd, "solid_wall")

    path = str(tmp_path / "conj.re2")
    writer.to_re2(mesh, path, groups={"interface": {"fluid": "W  ", "solid": None},
                                      "fluid_wall": "W  "},
                  fluid="fluid",
                  thermal={"fluid_wall": "I  ", "solid_wall": "I  "})  # "interface" omitted
    _vel, therm = _bc_blocks(path, mesh.n_hexes, 2)
    codes_therm = {struct.pack("<d", r[7]).decode("ascii", "replace").rstrip("\x00 ")
                  for r in therm}
    assert codes_therm == {"I"}   # only the two walls, never a stray row for "interface"


def test_a_field_wide_code_on_the_wrong_side_raises(tmp_path):
    """Naming a solid-only face in the velocity table is the wrong field's mistake,
    not a silently-dropped one -- Nek's velocity block cannot hold it at all."""
    mesh = _cht_pair()
    with pytest.raises(ValueError, match="outside this field's"):
        writer.to_re2(mesh, str(tmp_path / "x.re2"),
                      groups={"solid_wall": "W  "},   # solid-only, in the velocity table
                      fluid="fluid", thermal={"fluid_wall": "I  ", "solid_wall": "I  "})


# -- combined with periodic -----------------------------------------------
def test_periodic_rows_follow_the_reorder(tmp_path):
    """A periodic pair's element ids in the file must be read through the same
    permutation as every other row -- both directions, own and partner."""
    fluid = _slab(0.0, "fluid", first_tag="z_lo", last_tag="z_hi")
    solid = _slab(1.0, "solid")
    mesh = hexmesh.merge([solid, fluid], tol=1e-9)     # solid first in storage
    regs = mesh.element_tags.dense(mesh.n_hexes)
    n_fluid = int((regs == "fluid").sum())

    pairs = hexmesh.periodic_pairs(
        mesh, [Periodic("z_lo", "z_hi", affine.translation([0.0, 0.0, 2.0]))])
    path = str(tmp_path / "cht_p.re2")
    writer.to_re2(mesh, path, groups={"z_lo": "P  ", "z_hi": "P  "},
                  fluid="fluid", thermal={"z_lo": "P  ", "z_hi": "P  "},
                  periodic=pairs)

    nelgt, nelgv, _ = _read_header(path)
    assert nelgv == n_fluid

    partner = read_re2_periodic(path)   # reads only the FIRST (velocity) block
    assert len(partner) == 2 * int(hexmesh.tagged_faces(mesh, "z_lo").size)
    assert all(partner[v] == k for k, v in partner.items())
    # z_lo/z_hi are fluid-only, so every periodic row must land in [1, nelgv]
    assert all(e <= nelgv and pe <= nelgv for (e, _), (pe, _) in partner.items())


def test_periodic_pairs_may_span_both_regions(tmp_path):
    """wire_coil's real shape: one periodic pair on the fluid region (velocity and
    temperature), another on the solid one (temperature only) -- each pair must stay
    inside its own domain, never crossing nelgv."""
    fluid = _slab(0.0, "fluid", first_tag="f_lo", last_tag="f_hi")
    solid = _slab(1.0, "solid", first_tag="s_lo", last_tag="s_hi")
    mesh = hexmesh.merge([solid, fluid], tol=1e-9)
    regs = mesh.element_tags.dense(mesh.n_hexes)
    n_fluid = int((regs == "fluid").sum())
    T = affine.translation([0.0, 0.0, 2.0])

    pairs = hexmesh.periodic_pairs(
        mesh, [Periodic("f_lo", "f_hi", T), Periodic("s_lo", "s_hi", T)])
    path = str(tmp_path / "both.re2")
    writer.to_re2(mesh, path, groups={"f_lo": "P  ", "f_hi": "P  "},   # velocity: fluid only
                  fluid="fluid",
                  thermal={"f_lo": "P  ", "f_hi": "P  ",
                          "s_lo": "P  ", "s_hi": "P  "},               # thermal: both
                  periodic=pairs)

    vel, therm = _bc_blocks(path, mesh.n_hexes, 2)
    assert len(vel) == 2 * int(hexmesh.tagged_faces(mesh, "f_lo").size)
    assert all(e <= n_fluid for e in vel[:, 0])

    therm_key = {(int(a), int(b)): (int(c), int(d)) for a, b, c, d in therm[:, :4]}
    assert all(therm_key[v] == k for k, v in therm_key.items())
    fluid_rows = {k: v for k, v in therm_key.items() if k[0] <= n_fluid}
    solid_rows = {k: v for k, v in therm_key.items() if k[0] > n_fluid}
    assert fluid_rows and solid_rows
    assert all(pe <= n_fluid for _, (pe, _) in fluid_rows.items())
    assert all(pe > n_fluid for _, (pe, _) in solid_rows.items())
