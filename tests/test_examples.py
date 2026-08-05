"""Every example script runs and produces a valid mesh.

The examples *are* the meshers -- there are no mesher classes -- so an example that
stops building is a product defect, not a broken demo.  Fifteen of them were already
exercised indirectly by the geometry-specific suites; the other five were not, and a
refactor once left four of those with calls to methods that no longer existed while CI
stayed green.  This module closes that hole by **discovering** the scripts rather than
listing them, so a newly added example is covered the moment it lands.

The checks are deliberately generic -- structure, not geometry.  Anything specific to
one mesher belongs in that mesher's own test file, where the numbers can be asserted
properly; here we only ask whether the script still builds something a solver would
accept.
"""

import os

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import HexMesh, LineMesh, QuadMesh, hexmesh

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")

#: Scripts that are imported by other examples rather than run for their own mesh, so
#: they legitimately define no ``mesh`` global.  They must still execute cleanly.
LIBRARY_ONLY = {"tjunction_lib.py"}

#: Wall-clock over ~10 s on a warm machine, so they are deselected from a default run
#: (``-m "not slow"`` in ``addopts``) and run explicitly by one CI job.  Measured:
#: chimera 57 s, chimera_full 138 s.
SLOW = {"chimera.py", "chimera_full.py"}

#: Examples known to build an inverted element, as a strict xfail so that fixing one
#: fails this test and forces the entry out.  Empty, and worth keeping that way: the
#: only entry was the old two-manifold ``chimera.py`` (min scaled Jacobian about -0.17,
#: on ``main`` too), which has since been deleted.
KNOWN_INVERTED = set()

_CONTAINERS = (LineMesh, QuadMesh, HexMesh)


def _scripts():
    return sorted(f for f in os.listdir(_EXAMPLES) if f.endswith(".py"))


def _param(name):
    return pytest.param(name, marks=pytest.mark.slow) if name in SLOW else name


ALL = [_param(n) for n in _scripts()]


def _n_elements(mesh):
    """Element count of any container, whichever rung it sits on."""
    for attr in ("n_hexes", "n_quads", "n_lines"):
        if hasattr(mesh, attr):
            return getattr(mesh, attr)
    raise AssertionError("not a mesh container: %r" % type(mesh))


@pytest.fixture(scope="session")
def _built(tmp_path_factory):
    """Run each example at most once per session -- several take tens of seconds."""
    cache = {}

    def build(name):
        if name not in cache:
            out = tmp_path_factory.mktemp(name.replace(".py", ""))
            cache[name] = run_example(name, out)
        return cache[name]

    return build


@pytest.mark.parametrize("name", ALL)
def test_example_builds_a_valid_mesh(name, _built):
    """The script runs, and leaves a non-empty container in ``mesh``."""
    ns = _built(name)
    if name in LIBRARY_ONLY:
        assert "mesh" not in ns, (
            "%s is listed as library-only but now builds a mesh -- drop it from "
            "LIBRARY_ONLY so it gets checked like the others" % name)
        return
    assert "mesh" in ns, (
        "%s defines no `mesh` global. Examples double as integration tests and the "
        "suite reads that name; add it, or list the script in LIBRARY_ONLY." % name)
    mesh = ns["mesh"]
    assert isinstance(mesh, _CONTAINERS), type(mesh)
    assert _n_elements(mesh) > 0
    assert mesh.points.shape[1] == 3          # geometry is always 3-D
    assert np.isfinite(mesh.points).all()


@pytest.mark.parametrize("name", ALL)
def test_example_hex_mesh_is_a_closed_conformal_domain(name, _built):
    """A volume mesh must be watertight and conforming to be solvable at all."""
    ns = _built(name)
    mesh = ns.get("mesh")
    if not isinstance(mesh, HexMesh):
        pytest.skip("%s builds no volume mesh" % name)
    assert hexmesh.is_watertight(mesh)
    assert hexmesh.is_conforming(mesh)


@pytest.mark.parametrize("name", ALL)
def test_example_has_no_inverted_elements(name, _built, request):
    """Positive scaled Jacobian everywhere -- an inverted hex is unusable downstream,
    so this is the one quality bar worth enforcing on every mesher."""
    if name in KNOWN_INVERTED:
        request.node.add_marker(pytest.mark.xfail(
            strict=True, reason="pre-existing inverted element; see KNOWN_INVERTED"))
    ns = _built(name)
    mesh = ns.get("mesh")
    if not isinstance(mesh, HexMesh):
        pytest.skip("%s builds no volume mesh" % name)
    assert float(np.min(hexmesh.scaled_jacobian(mesh))) > 0.0


def test_every_example_is_covered_by_this_module():
    """The guard on the guard: the parametrization is generated from a directory
    listing, so this only fails if that discovery itself breaks."""
    assert {p.values[0] if hasattr(p, "values") else p for p in ALL} == set(_scripts())
    assert LIBRARY_ONLY <= set(_scripts())
    assert SLOW <= set(_scripts())
    assert KNOWN_INVERTED <= set(_scripts())
