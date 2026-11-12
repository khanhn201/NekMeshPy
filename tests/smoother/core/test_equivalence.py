"""The shared engine reproduces the smoother that existed before it: same inputs, same bytes.

``data/reference_*.npz`` were written by ``cases.py`` from the rung-local code at the commit before the engine
moved into ``nekmeshpy.core.smoother``.  Every array is compared: the metric, objective and gradient, the
quality and shape readings, one Laplace run, one Winslow run, 20 optimizer iterations, and a full schedule with
every stage's positions and log.  Integer arrays must be equal; float arrays to ``RTOL`` of their largest entry,
because the same code changes the last bits with the CPU and the numpy/BLAS build (measured: at most 8.4e-15 of
scale, over 223 of the 389 arrays, on CI runners and locally), while any real change is many orders larger.

Regenerated after the line-search fix (golden-section refinement shrinks from above when both probes are
infeasible, instead of walking toward larger steps): only the entries that ran such a search changed --
quad ``disk_tangent`` (``optimize.*``: the tangent-boundary optimizer no longer stalls after 14 iterations),
quad ``tangled`` (the ``schedule.*`` stages), and hex ``block`` (``optimize_w.*``).  Every other recorded array
(quad ``annulus`` and ``sphere``, hex ``cylinder`` and ``cylinder_tangent``, and the metric, quality, Laplace and
Winslow entries of all cases) is still bit-identical to the pre-engine code.
"""

import os

import cases
import numpy as np
import pytest

RTOL = 1e-12

QUAD = np.load(os.path.join(cases.DATA, "reference_quad.npz"))
HEX = np.load(os.path.join(cases.DATA, "reference_hex.npz"))


def _compare(ref, got, name):
    keys = sorted(k for k in ref.files if k.startswith(name + "/"))
    assert keys
    assert set(keys) == {"%s/%s" % (name, k) for k in got}, "recorded keys differ"
    for k in keys:
        a, b = ref[k], got[k.split("/", 1)[1]]
        assert a.shape == b.shape and a.dtype == b.dtype, k
        if a.dtype.kind != "f":
            assert np.array_equal(a, b), "%s differs" % k
            continue
        assert np.array_equal(np.isnan(a), np.isnan(b)), "%s: NaN pattern differs" % k
        fin = ~np.isnan(a)
        if not fin.any():
            continue
        diff = np.max(np.abs(a[fin] - b[fin]))
        assert diff <= RTOL * np.max(np.abs(a[fin])), "%s differs, max |diff| %s" % (k, diff)


@pytest.mark.parametrize("name", ["annulus", "disk_tangent", "sphere", "tangled"])
def test_quad_matches_frozen_reference(name):
    mesh, res, nrm = cases.quad_cases()[name]
    _compare(QUAD, cases.record_quad(name, mesh, res, nrm), name)


@pytest.mark.parametrize("name", ["block", "cylinder", "cylinder_tangent"])
def test_hex_matches_frozen_reference(name):
    mesh, res = cases.hex_cases()[name]
    _compare(HEX, cases.record_hex(name, mesh, res), name)
