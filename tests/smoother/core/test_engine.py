"""The shared smoother engine on its own: the metric against the one determinant formula at d = 2 and d = 3,
its gradient against finite differences, the registry, the descent driver, and the layering rule."""

import ast
import os
import subprocess
import sys

import numpy as np
import pytest

from nekmeshpy.core.smoother import constraints, descent, metric, schedule

_PKG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "nekmeshpy")
_ENGINE = os.path.join(_PKG, "core", "smoother")


# -- layering ----------------------------------------------------------------------------------------
def test_engine_imports_no_rung_in_a_fresh_interpreter():
    """``core.smoother`` imported with the package ``__init__`` bypassed pulls in neither ``quadmesh`` nor ``hexmesh``."""
    code = (
        "import sys, types\n"
        "pkg = types.ModuleType('nekmeshpy'); pkg.__path__ = [%r]; sys.modules['nekmeshpy'] = pkg\n"
        "import nekmeshpy.core.smoother\n"
        "bad = sorted(m for m in sys.modules if m.startswith(('nekmeshpy.quadmesh', 'nekmeshpy.hexmesh')))\n"
        "assert not bad, bad\n"
        "print('ok', len([m for m in sys.modules if m.startswith('nekmeshpy.core.smoother')]))\n" % os.path.abspath(_PKG))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ok")


def test_engine_source_names_no_rung():
    for name in os.listdir(_ENGINE):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_ENGINE, name)).read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = [("." * node.level) + (node.module or "")] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            assert not any(m.endswith(("quadmesh", "hexmesh")) or "quadmesh." in m or "hexmesh." in m for m in mods), (name, mods)


# -- the metric ------------------------------------------------------------------------------------
def _jacobians(edges):
    """``(n, k, d, d)`` Jacobians whose columns are the edge vectors (planar: the first d coordinates)."""
    return np.stack(edges, axis=-1)


def _mean_ratio(J, d):
    det = np.linalg.det(J)
    fro2 = (J * J).sum(axis=(-1, -2))
    return (fro2 / d) ** d / det ** 2, det / np.prod(np.linalg.norm(J, axis=-2), axis=-1)


def _random_edges(d, n=5, k=4, seed=0):
    rng = np.random.default_rng(seed)
    base = np.eye(d)[None, None] * rng.uniform(0.5, 2.0, (n, k, 1, 1))
    J = base + rng.normal(size=(n, k, d, d)) * 0.15
    J[..., 0] *= np.sign(np.linalg.det(J))[..., None]            # right-handed
    return J


@pytest.mark.parametrize("d", [2, 3])
def test_metric_is_the_mean_ratio_of_the_corner_jacobian(d):
    J = _random_edges(d)
    edges = [np.concatenate([J[..., j], np.zeros(J.shape[:2] + (3 - d,))], axis=-1) for j in range(d)]
    normals = np.tile([0.0, 0.0, 1.0], (J.shape[0], 1))
    m = metric.corner_metrics(edges, normals if d == 2 else None)
    phi, sj = _mean_ratio(J, d)
    assert np.allclose(m.phi, phi, rtol=1e-12) and np.allclose(m.sj, sj, rtol=1e-12)
    assert (m.phi >= 1.0 - 1e-12).all()


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("delta", [0.0, 0.05])
def test_gradient_matches_finite_differences(d, delta):
    rng = np.random.default_rng(3)
    J = _random_edges(d, n=3, k=2, seed=1)
    edges = [np.concatenate([J[..., j], np.zeros(J.shape[:2] + (3 - d,))], axis=-1) for j in range(d)]
    normals = np.tile([0.0, 0.0, 1.0], (3, 1)) if d == 2 else None
    w = rng.uniform(0.5, 1.5, 3)
    kw = {"kappa": 10.0, "eps": 0.9, "delta": delta, "weights": w}
    grads = metric.corner_gradient(edges, normals, **kw)

    def f(es):
        m = metric.corner_metrics(es, normals, delta=delta)
        return metric.objective_value(m, d, kappa=10.0, eps=0.9, weights=w)

    h = 1e-6
    for j in range(d):
        for (e, k, c) in [(0, 0, 0), (1, 1, 1), (2, 0, 2)]:
            if d == 2 and c == 2:
                continue                                         # planar embedding: the z component is not a variable
            up = [x.copy() for x in edges]
            dn = [x.copy() for x in edges]
            up[j][e, k, c] += h
            dn[j][e, k, c] -= h
            fd = (f(up) - f(dn)) / (2 * h)
            assert abs(fd - grads[j][e, k, c]) <= 1e-5 * max(1.0, abs(fd)), (j, e, k, c, fd, grads[j][e, k, c])


def test_assemble_gradient_sums_corner_terms_onto_vertices():
    corner_index = np.array([[0, 1], [1, 2]])
    far = [np.array([[1, 2], [2, 0]])]
    g = [np.array([[[1.0, 0, 0], [2.0, 0, 0]], [[3.0, 0, 0], [4.0, 0, 0]]])]
    out = metric.assemble_gradient(3, corner_index, far, g)
    assert np.allclose(out[:, 0], [3.0, -4.0, 1.0])               # far ends [4, 1, 5] minus own corners [1, 5, 4]
    assert np.isclose(out.sum(), 0.0)                            # every term enters once with and once against


def test_metric_rejects_unsupported_dimension_and_missing_normals():
    e = np.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="normals"):
        metric.corner_metrics([e, e])
    with pytest.raises(ValueError, match="expected 2"):
        metric.corner_metrics([e])


# -- the registry ----------------------------------------------------------------------------------
class _Mesh:
    points = np.zeros((1, 3))


def _fn(mesh, resolved, **kw):
    _fn.seen = kw
    return mesh, schedule.StageLog("x", 1, 1, "completed", "", None, None, ())


def test_registry_registers_refuses_duplicates_and_passes_knobs():
    reg = schedule.Registry()
    reg.register("x", _fn, local=True, weights=True)
    assert reg.names() == ["x"]
    with pytest.raises(ValueError, match="already registered"):
        reg.register("x", _fn)
    reg.register("x", _fn, replace=True, weights=True, local=True)
    with pytest.raises(ValueError, match="non-empty"):
        reg.register("", _fn)
    with pytest.raises(TypeError):
        reg.register("y", 3)
    mesh, logs = reg.run(_Mesh(), None, [("x", {"a": 1})], failing=lambda m: "FAIL", weights="W",
                         common={"normals": "N", "a": 9}, record=True)
    assert _fn.seen == {"a": 1, "elements": "FAIL", "weights": "W", "normals": "N"}
    assert logs[0].points.shape == (1, 3)
    with pytest.raises(ValueError, match="unknown smoothing method"):
        reg.run(_Mesh(), None, [("nope", {})], failing=lambda m: None)


def test_presets_are_shared():
    assert schedule.user_schedule(outer=1)[-1][0] == "laplace_local"
    assert [s[1]["delta"] for s in schedule.untangle_schedule()][0] == 1.0

    class Q:
        n_inverted = 2
    assert schedule.suggest_schedule(Q(), outer=1)[0][0] == "optimize"
    Q.n_inverted = 0
    assert schedule.suggest_schedule(Q(), outer=1)[0][0] == "laplace"


# -- the descent driver -------------------------------------------------------------------------------
def test_descent_minimises_a_quadratic_with_a_frozen_vertex():
    target = np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    def evaluate(Q):
        return float(((Q - target) ** 2).sum()), 0

    def grad(Q):
        return 2.0 * (Q - target)

    P0 = np.zeros((3, 3))
    movable = np.array([True, True, False])
    res = descent.minimize(
        P0, evaluate=evaluate, gradient=grad, project=lambda d, X: d.copy(), snap=lambda X: X,
        summary=lambda X: (0, 0.0, evaluate(X)[0]), ell=np.ones((3, 1)), movable=movable,
        tangent=np.zeros(3, bool), n_used=1, iters=50)
    assert res.f < 1e-6 and np.allclose(res.P[:2], target[:2], atol=1e-3)
    assert (res.P[2] == 0).all()                                 # the frozen vertex never moved
    assert res.stopped == "converged" and res.done >= 1


def test_descent_refuses_steps_that_add_inversions():
    res = descent.minimize(
        np.zeros((1, 3)), evaluate=lambda Q: (float(Q.sum() ** 2), 0 if Q.sum() >= 0 else 1),
        gradient=lambda Q: np.ones((1, 3)), project=lambda d, X: d, snap=lambda X: X,
        summary=lambda X: (0, 0.0, 0.0), ell=np.ones((1, 1)), movable=np.ones(1, bool), tangent=np.zeros(1, bool),
        n_used=1, iters=5)
    assert res.ninv == 0 and res.P.sum() >= 0


def test_constraints_snap_and_direction_without_a_mesh():
    surf = constraints.plane([0, 0, 0], [0, 0, 1])
    r = constraints.Resolved(
        kind=np.array([constraints.ON_SURFACE, constraints.FROZEN, constraints.FREE]),
        target=np.array([0, -1, -1]), anchor=np.array([[0, 0, 0], [5.0, 5, 5], [0, 0, 0]]),
        surfaces=(surf,), curves=(), notes=(), tie=np.full(3, -1), tie_dir=np.zeros((3, 3)),
        nbr=np.full((3, 2), -1), bfaces=np.zeros((0, 4), np.int64))
    X = np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [1.0, 1.0, 1.0]])
    S = constraints.snap(X, r)
    assert np.allclose(S[0], [1, 2, 0]) and np.allclose(S[1], [5, 5, 5]) and np.allclose(S[2], X[2])
    d = constraints.constrain_direction(np.ones((3, 3)), X, r)
    assert np.allclose(d[0], [1, 1, 0]) and (d[1] == 0).all() and np.allclose(d[2], 1)
