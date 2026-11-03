"""Normals for the metric come from the surface the quads sit on, not from the (possibly folded) input quads."""
import logging

import numpy as np
import surfaces as S

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth
from nekmeshpy.quadmesh.smoother.session import SmoothSession


def _ellipsoid(seed=0):
    an, m, ref = S.jittered_case("ellipsoid", seed=seed)
    return an, m, ref


def _torus():
    """A clean input: no folded quad (the jittered sphere-like cases have a few, which is the point of the check)."""
    an, m, _ = S.jittered_case("torus")
    return an, m


def _subdivide(V, T):
    """One midpoint subdivision of a triangulated unit sphere (new points pushed onto the sphere)."""
    V = list(V)
    cache = {}
    nt = []

    def mid(i, j):
        k = (min(i, j), max(i, j))
        if k not in cache:
            v = (V[i] + V[j]) / 2
            V.append(v / np.linalg.norm(v))
            cache[k] = len(V) - 1
        return cache[k]
    for a, b, c in T:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nt += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(V), np.array(nt)


def _icosphere(n=3):
    """A triangulated unit sphere (octahedron subdivided ``n`` times), outward oriented."""
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    T = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    for _ in range(n):
        V, T = _subdivide(V, T)
    return V, T


def test_surface_normals_fix_the_ellipsoid():
    an, m, ref = _ellipsoid()
    nrm_ref = smooth.reference_normals(ref.points, ref.corners)
    r = cn.resolve(m, [cn.OnSurface(np.arange(m.n_points), an.surface)], boundary="free")
    nrm, rep = cn.surface_normals(m.points, m.corners, cn.claimed_surfaces(r))
    exact = an.surface.normal(m.points)[m.corners].mean(axis=1)
    exact /= np.linalg.norm(exact, axis=1, keepdims=True)
    assert np.einsum("ij,ij->i", nrm, exact).min() > 0.999            # it IS the surface normal
    assert np.einsum("ij,ij->i", nrm_ref, exact).min() < 0            # the reference had a folded quad
    assert rep.n_from_surface == m.n_quads and rep.n_fallback == 0


def test_global_flip_is_handled_but_a_reversed_quad_is_reported():
    an, m = _torus()
    flipped = cn.Surface(an.surface.project, lambda P: -an.surface.normal(P), "inward")
    nrm, rep = cn.surface_normals(m.points, m.corners, [(np.arange(m.n_points), flipped)])
    nrm0, _ = cn.surface_normals(m.points, m.corners, [(np.arange(m.n_points), an.surface)])
    assert np.allclose(nrm, nrm0) and rep.flips == (-1,) and rep.n_disagree == 0
    c = m.corners.copy()
    c[::5] = c[::5, ::-1]                                              # a fifth of the quads wound backwards
    bad = quadmesh.QuadMesh.from_corners(m.points, c)
    _, rep = cn.surface_normals(bad.points, bad.corners, [(np.arange(m.n_points), an.surface)])
    assert rep.n_disagree == len(c[::5]) and rep.min_cos < -0.9


def test_check_can_be_opted_out():
    an, m, _ = _ellipsoid()
    _, rep = cn.surface_normals(m.points, m.corners, [(np.arange(m.n_points), an.surface)], check=False)
    assert rep.n_disagree == 0 and np.isnan(rep.min_cos)


def test_trimesh_surface_approximates_the_sphere():
    V, T = _icosphere(3)
    srf = cn.surface_from_trimesh(V, T)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    X = X / np.linalg.norm(X, axis=1, keepdims=True) * rng.uniform(0.8, 1.2, (500, 1))
    foot = srf.project(X)
    assert np.abs(np.linalg.norm(foot, axis=1) - 1).max() < 0.02       # chord error of the tessellation
    assert np.all(np.linalg.norm(srf.project(foot) - foot, axis=1) < 1e-9)       # idempotent
    n = srf.normal(X)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert (n * foot).sum(1).min() > 0.97                              # within ~14 degrees of the exact normal


def test_trimesh_normals_do_the_same_job():
    an, m, _ = _ellipsoid()
    V, T = _icosphere(4)
    tm = cn.surface_from_trimesh(V * [1, 1.5, 2], T)                   # a tessellation of the very ellipsoid
    nrm, rep = cn.surface_normals(m.points, m.corners, [(np.arange(m.n_points), tm)])
    exact = an.surface.normal(m.points)[m.corners].mean(axis=1)
    exact /= np.linalg.norm(exact, axis=1, keepdims=True)
    assert np.einsum("ij,ij->i", nrm, exact).min() > 0.9
    _, rep_exact = cn.surface_normals(m.points, m.corners, [(np.arange(m.n_points), an.surface)])
    assert rep.n_disagree == rep_exact.n_disagree > 0                  # the jittered input has real folds; both see them


def test_session_takes_normals_from_its_surface_constraint_and_warns_on_a_folded_quad(caplog):
    an, m = _torus()
    cons = [cn.OnSurface(np.arange(m.n_points), an.surface)]
    with caplog.at_level(logging.WARNING):
        s = SmoothSession(m, cons, boundary="free")
    assert s.normal_report is not None and s.normal_report.n_fallback == 0 and s.normal_report.n_disagree == 0
    assert "disagree" not in caplog.text
    exact = an.surface.normal(m.points)[m.corners].mean(axis=1)
    assert np.einsum("ij,ij->i", s.normals, exact / np.linalg.norm(exact, axis=1, keepdims=True)).min() > 0.999
    c = m.corners.copy()
    c[3] = c[3, ::-1]
    bad = quadmesh.QuadMesh.from_corners(m.points, c)
    with caplog.at_level(logging.WARNING):
        s = SmoothSession(bad, cons, boundary="free")
    assert s.normal_report.n_disagree == 1 and "disagree with the surface normal" in caplog.text


def test_precedence_explicit_normals_then_surface_then_constraints_then_reference():
    an, m, _ = _ellipsoid()
    cons = [cn.OnSurface(np.arange(m.n_points), an.surface)]
    given = np.tile([0.0, 0.0, 1.0], (m.n_quads, 1))
    s = SmoothSession(m, cons, boundary="free", normals=given)
    assert s.normals is given and s.normal_report is None
    plain = SmoothSession(m, [], boundary="free")                       # nothing known: the quads' own winding
    assert plain.normal_report is None
    assert np.allclose(plain.normals, smooth.reference_normals(m.points, m.corners))
    viasurf = SmoothSession(m, [], boundary="free", surface=an.surface)
    assert viasurf.normal_report is not None and viasurf.normal_report.n_from_surface == m.n_quads
