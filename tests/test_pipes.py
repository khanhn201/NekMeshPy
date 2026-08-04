"""Run the flat pipe example scripts and check the meshes they produce."""

import numpy as np
import pytest
from conftest import run_example
from scipy import integrate

from nekmeshpy import topology
from nekmeshpy.hexmesh import quality
from nekmeshpy.model import conform, fields, interp


def _scaled_jac(mesh):
    return quality.scaled_jacobian(*mesh.weld()[:2])


def _tag_count(mesh, name):
    return mesh.face_tags.count(name)


def test_circular_pipe(tmp_path):
    mesh = run_example("circular_pipe.py", tmp_path)["mesh"]
    # wall sides + inlet / outlet caps all present
    assert _tag_count(mesh, "wall") > 0
    assert (_tag_count(mesh, "inlet") > 0
            and _tag_count(mesh, "inlet") == _tag_count(mesh, "outlet"))
    # O-grid: no collapsed centre cell, all positive Jacobian
    assert float(np.min(_scaled_jac(mesh))) > 0.5
    assert mesh.is_watertight() and mesh.is_conforming()
    assert set(mesh.face_group_tags) >= {"wall", "inlet", "outlet"}


def test_rectangular_pipe(tmp_path):
    mesh = run_example("rectangular_pipe.py", tmp_path)["mesh"]
    # a structured axis-aligned duct is exact -> scaled Jacobian 1 everywhere
    assert float(np.min(_scaled_jac(mesh))) == pytest.approx(1.0, abs=1e-9)
    assert mesh.is_watertight() and mesh.is_conforming()
    assert _tag_count(mesh, "inlet") == _tag_count(mesh, "outlet")  # caps match
    assert set(mesh.face_group_tags) >= {"wall", "inlet", "outlet"}


def test_circular_pipe_tjunction(tmp_path):
    mesh = run_example("circular_pipe_tjunction.py", tmp_path)["mesh"]
    # three legs welded into one conformal, watertight block at the junction
    assert mesh.is_watertight() and mesh.is_conforming()
    assert topology.hex_report(*mesh.weld()[:2]).n_components == 1
    # wall + three circular openings, each opening the same face count
    assert _tag_count(mesh, "wall") > 0
    for name in ("outlet", "branch"):
        assert _tag_count(mesh, name) == _tag_count(mesh, "inlet")
    # analytic O-grid junction stays well away from degenerate
    assert float(np.min(_scaled_jac(mesh))) > 0.3
    assert set(mesh.face_group_tags) >= {"wall", "inlet", "outlet", "branch"}


def _wall_nodes(mesh):
    """**Every** node of every ``wall``-tagged boundary face, high-order ones too.

    Corner-only would pass on a mesh that is high-order in storage and linear in
    geometry, which is exactly the failure mode worth guarding here."""
    nodes, conn = conform.conformal_hex(
        mesh.points, mesh.hexes, mesh._elem_edges, mesh._edge_flip,
        mesh.quads.lines.interior, mesh.hex, mesh.face_orient,
        mesh.quads.interior, mesh.interior, mesh.order)
    ids = [conn[e][interp.hex_face_indices(s, mesh.order)]
           for e, s, tag in mesh.face_tags
           if tag == "wall"]
    return nodes[np.unique(np.concatenate(ids))]


def _volume(mesh):
    """Sum of element volumes: GLL quadrature of ``det(J)`` over each curved block.

    Reads the geometry the elements actually carry, so it sees curvature that a
    corner-based measure cannot."""
    o = mesh.order
    nodes, conn = conform.conformal_hex(
        mesh.points, mesh.hexes, mesh._elem_edges, mesh._edge_flip,
        mesh.quads.lines.interior, mesh.hex, mesh.face_orient,
        mesh.quads.interior, mesh.interior, o)
    blk = nodes[conn].reshape(-1, o + 1, o + 1, o + 1, 3)
    g = fields.gll_nodes(o)
    d = fields.lagrange_derivative_matrix(g, g)
    ti = np.einsum("mi,ekjid->ekjmd", d, blk)
    tj = np.einsum("mj,ekjid->ekmid", d, blk)
    tk = np.einsum("mk,ekjid->emjid", d, blk)
    det = np.einsum("...d,...d->...", np.cross(ti, tj), tk)
    x = g * 2.0 - 1.0
    w = 2.0 / (o * (o + 1) * np.polynomial.legendre.legval(x, [0] * o + [1]) ** 2)
    return float(np.einsum("ekji,kji->", det, np.einsum("k,j,i->kji", w, w, w) / 8.0))


def test_quadrant_pipe_tjunction(tmp_path):
    ns = run_example("quadrant_pipe_tjunction.py", tmp_path)
    mesh = ns["mesh"]
    # the whole point of the reference topology: one quadrant of the main pipe *is*
    # a quadrant of the branch, so the junction welds into a single closed block.
    rep = topology.hex_report(*mesh.weld()[:2])
    assert rep.watertight and rep.conformal
    assert rep.n_components == 1
    assert rep.n_open_edges == 0 and rep.n_hanging_points == 0
    assert _tag_count(mesh, "inlet") == _tag_count(mesh, "outlet")
    assert _tag_count(mesh, "branch") == _tag_count(mesh, "inlet")
    assert _tag_count(mesh, "wall") > 0
    assert set(mesh.face_group_tags) == {"wall", "inlet", "outlet", "branch"}
    assert float(np.min(_scaled_jac(mesh))) > 0.2

    # every wall node -- corner and high-order alike -- is on one of the two
    # cylinders.  Nothing in this mesh is straight-subdivided: the wall curves are
    # meshed from their parametrizations, the leg transition is a loft_fn rather
    # than a loft (which is straight along the sweep), and the crotch caps are
    # evaluated at every node rather than blended from corners.
    assert mesh.order > 1, "the point of this assertion is the high-order nodes"
    p = _wall_nodes(mesh)
    on_main = np.abs(np.hypot(p[:, 0], p[:, 1]) - ns["R_MAIN"]) < 1e-12
    on_branch = np.abs(np.hypot(p[:, 1], p[:, 2]) - ns["R_BRANCH"]) < 1e-12
    assert np.all(on_main | on_branch)
    assert on_main.any() and on_branch.any()
    # and the order-N geometry is not merely stored, it is sound
    assert float(np.min(mesh.scaled_jacobian(high_order=True))) > 0.2

    # The mesh fills exactly the union of the two cylinders.  This is the check that
    # separates high-order *geometry* from high-order storage: straight-subdivided
    # order-N elements would stall at the order-1 faceting error of ~1.2e-2, whereas
    # the real thing converges spectrally (1.1e-5 / 1.5e-7 / 2.6e-10 at order 2/3/4).
    # What is left at order 3 is quadrature error on det(J), not geometry: the volume
    # enclosed depends only on the boundary, and every wall node is on its cylinder.
    rm, rb, h = ns["R_MAIN"], ns["R_BRANCH"], ns["H_BRANCH"]
    exact = np.pi * rm**2 * 2 * ns["L_MAIN"] + integrate.dblquad(
        lambda z, y: h - np.sqrt(rm**2 - y**2), -rb, rb,
        lambda y: -np.sqrt(rb**2 - y**2), lambda y: np.sqrt(rb**2 - y**2))[0]
    assert _volume(mesh) == pytest.approx(exact, rel=1e-6)

    # the 45 degree rotation: the whole footprint reaches only asin(Rb/Rm) either
    # side of theta = 0, so it stays inside the branch-facing quadrant of the plain
    # disc each leg morphs into, whose seams sit at +-45 degrees.
    foot = np.vstack([a.points for a in ns["FQ"]])
    theta = np.abs(np.arctan2(foot[:, 1], foot[:, 0]))
    assert theta.max() == pytest.approx(np.arcsin(ns["R_BRANCH"] / ns["R_MAIN"]),
                                        abs=1e-12)
    assert theta.max() < np.pi / 4 - np.deg2rad(10.0)
