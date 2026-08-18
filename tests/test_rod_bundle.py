"""``examples/rod_bundle.py``: the Voronoi tiling closes exactly, and the sweep keeps it.

The mesher's whole correctness claim is a *partition* -- 61 hexagonal cells plus one
outer gap ring plus the duct wall cover the duct interior once, with nothing left over
and nothing counted twice.  That is checkable to round-off rather than by eye: the
three regions' areas must sum to the duct's outer hexagon, and the only boundary the
merged section may have left is that hexagon's own outer loop.  A cell that failed to
weld, a zigzag side collected twice, or a duct sampled off by one all break one of
those two, so they are the tests worth having.

The sweep then has to preserve all of it -- volume, region split, watertightness -- and
resolve the one thing the section leaves open: two *conjugate* interfaces, each a
single face carrying a single name but a different Nek code on each side of it.  A
missing ``side_codes`` entry there would double every interface row.
"""

import collections

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import hexmesh, quadmesh
from nekmeshpy.io.writer import _as_groups, _export_rows

REGIONS = ("rod", "fluid", "duct")


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rod_bundle")


@pytest.fixture(scope="module")
def bundle(bundle_dir):
    return run_example("rod_bundle.py", bundle_dir)


def _duct_hexagon_area(ns):
    """Area of a regular hexagon of apothem ``a``: ``2*sqrt(3)*a^2``."""
    return 2.0 * np.sqrt(3.0) * (ns["DUCT_IN"] + ns["DUCT_T"]) ** 2


def test_regions_partition_the_duct(bundle):
    section = bundle["section"]
    tags = section.element_tags.dense(section.n_quads)
    assert set(np.unique(tags)) == set(REGIONS)

    total = _duct_hexagon_area(bundle)
    areas = quadmesh.element_areas(section)
    assert areas.sum() == pytest.approx(total, abs=1e-12)
    assert sum(areas[tags == r].sum() for r in REGIONS) == pytest.approx(total, abs=1e-12)


def test_rod_region_is_the_wire_wrapped_profile(bundle):
    """``rod`` is exactly the 61 polygons inscribed in the wire-wrapped wall profile:
    ``0.5 * sum r_k r_(k+1) sin(dtheta)``, the corner-only area, since the region's
    outer wall is chordal at the corners even though ``ORDER > 1`` bows it onto the
    profile in between.

    The plain-circle area is asserted *not* to match as well, because the wire is a
    local bulge and a bump function that silently evaluated to zero -- a wrong angle
    convention, a window that never opens -- would leave a mesh that still looks
    entirely reasonable."""
    section = bundle["section"]
    n, r0 = bundle["N_CIRC"], bundle["R_ROD"]
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + np.pi / 6.0
    r = r0 + bundle["wire_bump"](theta, 0.0)
    profile = 0.5 * np.sum(r * np.roll(r, -1) * np.sin(2.0 * np.pi / n))
    circle = 0.5 * n * r0 ** 2 * np.sin(2.0 * np.pi / n)

    tags = section.element_tags.dense(section.n_quads)
    got = quadmesh.element_areas(section)[tags == "rod"].sum()
    assert got == pytest.approx(61 * profile, rel=1e-12)
    assert got > 61 * circle * (1.0 + 1e-6)


def test_the_wire_sits_where_it_was_asked_to(bundle):
    """The bulge peaks at the angle asked for, reaches ``WIRE_H``, and is gone outside
    ``WIRE_HALF`` -- so the wall is the plain circle everywhere else."""
    bump, half = bundle["wire_bump"], bundle["WIRE_HALF"]
    for wire_theta in (0.0, 0.9, -2.4):
        assert bump(np.array(wire_theta), wire_theta) == pytest.approx(bundle["WIRE_H"])
        assert bump(np.array(wire_theta + half), wire_theta) == pytest.approx(0.0)
        away = wire_theta + half + np.linspace(0.0, np.pi - half, 17)
        assert np.allclose(bump(away, wire_theta), 0.0)

    # and the wall really carries it: slice 0's farthest centre-rod wall node is
    # R_ROD + WIRE_H, on the +x ray
    section = bundle["section"]
    xy = section.points[:, :2]
    near = xy[np.linalg.norm(xy, axis=1) < bundle["R_ROD"] + bundle["WIRE_H"] + 1e-9]
    k = int(np.argmax(np.linalg.norm(near, axis=1)))
    assert np.linalg.norm(near[k]) == pytest.approx(bundle["R_ROD"] + bundle["WIRE_H"],
                                                    rel=1e-12)
    assert np.arctan2(near[k, 1], near[k, 0]) == pytest.approx(0.0, abs=1e-12)


def test_the_stack_is_a_helix_of_one_turn_per_lead(bundle):
    """Every station carries the same wall, rotated: the crest angle advances linearly
    with z and closes exactly one turn over ``WIRE_LEAD``.

    Tracked through the centre rod's own wall ids, found *topologically* -- the solid
    region split into its 61 discs, the one at the origin, its boundary. A radius filter
    would not do it: a neighbour rod's near-side wall reaches within
    ``PITCH - R_ROD - WIRE_H`` of the origin, inside this rod's own crest."""
    mesh, slices = bundle["mesh"], bundle["slices"]
    sec0, span, n = slices[0], bundle["SPAN"], bundle["N_SPAN"]
    assert bundle["SPAN"] == bundle["WIRE_LEAD"]

    discs = quadmesh.components(quadmesh.select(sec0, "rod"))
    assert len(discs) == 61
    centre = min(discs, key=lambda d: np.linalg.norm(quadmesh.centroid(d)[:2]))
    by_xy = {tuple(np.round(q, 9)): i for i, q in enumerate(sec0.points)}
    wall = np.array([by_xy[tuple(np.round(q, 9))]
                     for q in quadmesh.boundary_mesh(centre).points])
    assert wall.size == bundle["N_CIRC"]

    nn = sec0.n_points
    assert mesh.n_points == nn * (n + 1)
    angles = []
    for k in range(n + 1):
        xy = mesh.points[k * nn:(k + 1) * nn][wall, :2]
        r = np.linalg.norm(xy, axis=1)
        assert r.max() <= bundle["R_ROD"] + bundle["WIRE_H"] + 1e-12
        # bulge-weighted circular mean: exact for a lobe symmetric about its crest
        w = np.maximum(r - bundle["R_ROD"], 0.0)
        a = np.arctan2(xy[:, 1], xy[:, 0])
        angles.append(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()))

    z = np.arange(n + 1) * span / n
    got = np.unwrap(np.asarray(angles))
    # the residual is the wall's own 360/N_CIRC sampling, not the stack
    assert np.max(np.abs(got - 2.0 * np.pi * z / bundle["WIRE_LEAD"])) < np.deg2rad(1.0)
    assert got[-1] == pytest.approx(2.0 * np.pi, abs=np.deg2rad(1.0))


def test_only_the_duct_outside_is_left_on_the_section_boundary(bundle):
    """Every cell-to-cell seam welded, so the section's boundary is the outer hexagon
    alone -- and it is the run named ``outer``, one edge per boundary edge."""
    section = bundle["section"]
    edge_tags = section.edge_tags
    assert quadmesh.boundary_edges(section).shape[0] == edge_tags.count("outer")
    assert edge_tags.count("duct_surface") == edge_tags.count("outer")
    assert edge_tags.count("rod_surface") == 61 * bundle["N_CIRC"]


def test_the_stack_fills_the_duct_exactly(bundle):
    """The **total** is the duct hexagon times the span, to round-off.

    Exact despite the twist, and that is the point: the duct is the one boundary the
    wire does not move, so however the stations rotate inside it they can only trade
    volume among themselves."""
    mesh, section = bundle["mesh"], bundle["section"]
    assert hexmesh.is_watertight(mesh)
    assert mesh.n_hexes == section.n_quads * bundle["N_SPAN"]
    assert hexmesh.volume(mesh) == pytest.approx(
        _duct_hexagon_area(bundle) * bundle["SPAN"], abs=1e-9)


def test_the_twist_costs_the_regions_only_the_chord_error(bundle):
    """Region by region the volume is *not* area x span, and should not be.

    ``loft`` is straight-sided along its sweep, so between two stations a rotating
    interface is cut as a chord across its own arc: the rod loses a sliver to the
    coolant on the way round.

    The bound is *derived*, not tuned. Over one layer the wall turns
    ``dt = 2 pi / N_SPAN``, and linear interpolation puts a node at radius ``r`` on the
    chord, i.e. at ``r cos(dt/2)``; area goes as ``r^2``, so no region can lose more
    than ``1 - cos(pi / N_SPAN)^2`` of itself. A magic constant here would have to be
    re-tuned every time the wire or the lead changed -- which is exactly what it did,
    silently passing at ``WIRE_H = 0.12`` and failing at 0.16 for no reason but its
    own staleness."""
    mesh, section = bundle["mesh"], bundle["section"]
    vol = hexmesh.element_volumes(mesh)
    hex_tags = mesh.element_tags.dense(mesh.n_hexes)
    quad_tags = section.element_tags.dense(section.n_quads)
    areas = quadmesh.element_areas(section)

    defect = {}
    for r in REGIONS:
        want = areas[quad_tags == r].sum() * bundle["SPAN"]
        defect[r] = vol[hex_tags == r].sum() / want - 1.0

    # the duct never rotates, so its own sweep is exact
    assert defect["duct"] == pytest.approx(0.0, abs=1e-12)
    # the rod loses, the coolant gains it back, and the two cancel into the total
    bound = 1.0 - np.cos(np.pi / bundle["N_SPAN"]) ** 2
    assert -bound < defect["rod"] < 0.0
    assert 0.0 < defect["fluid"] < bound


def test_no_inverted_elements(bundle):
    """Read off the **curved** block, which is the reading that can disagree."""
    assert hexmesh.quality_summary(bundle["mesh"]).n_inverted == 0
    assert quadmesh.quality_summary(bundle["section"]).n_inverted == 0


def test_conjugate_interfaces_export_from_the_fluid_side_only(bundle):
    """Each interface face is carried by two hexes, but ``side_codes`` drops the metal
    one -- so the rod walls export ``61 * N_CIRC * N_SPAN`` rows, not twice that."""
    mesh = bundle["mesh"]
    rows = _export_rows(mesh, _as_groups(mesh, bundle["GROUPS"]))
    n = collections.Counter(name for _, _, name, _ in rows)
    span, circ = bundle["N_SPAN"], bundle["N_CIRC"]

    assert n["rod_surface"] == 61 * circ * span
    assert n["duct_surface"] == n["outer"]

    section = bundle["section"]
    tags = section.element_tags.dense(section.n_quads)
    assert n["inlet"] == n["outlet"] == int((tags == "fluid").sum())
    assert n["cut"] == 2 * int((tags != "fluid").sum())


def test_vtu_carries_the_three_regions_per_cell(bundle, bundle_dir):
    """The point of the region split is to be able to pull the coolant out of the
    bundle in ParaView, so it has to survive into the file -- per cell, one value per
    hex, in the swept counts the section's own split implies."""
    import base64
    import xml.etree.ElementTree as ET

    from nekmeshpy.io import writer

    mesh, section = bundle["mesh"], bundle["section"]
    da = ET.parse(str(bundle_dir / "rod_bundle.vtu")).getroot().find(
        ".//CellData/DataArray[@Name='element_tag']")
    assert da is not None
    raw = base64.b64decode(da.text.strip())
    got = np.frombuffer(raw[8:8 + int(np.frombuffer(raw[:8], "<u8")[0])], "<i4")

    ids, names = writer.element_tag_ids(mesh.element_tags, mesh.n_hexes)
    assert names == sorted(REGIONS)
    assert np.array_equal(got, ids)

    quad_tags = section.element_tags.dense(section.n_quads)
    for i, name in enumerate(names):
        assert int((got == i + 1).sum()) == \
            int((quad_tags == name).sum()) * bundle["N_SPAN"]
