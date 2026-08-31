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
#:
#: ``serpentine_pipe.py`` is deliberately **not** here: run directly (as this harness
#: does, ``run_name="__main__"``) it still builds its own mesh -- only a plain
#: ``import``, which ``chimera_full.py`` uses for two names, sees the guarded body
#: skipped.
LIBRARY_ONLY = {"tjunction_lib.py", "femoral_vol.py"}

#: Not run by this harness at all -- discovered, listed, and then dropped from the
#: parametrization.  ``femoral`` ships and is maintained as an example; it is simply
#: too demanding to be a test.  It tet-meshes with **gmsh** (the only thing in the repo
#: that does) and costs 316 s cold, which CI always is.
#:
#: Excluding it rather than marking it slow also retires a check that could not be made
#: honest: gmsh does not tetrahedralize the same way twice -- not across machines, not
#: run to run on one (measured: 157402 / 157446 / 157483 nodes for identical input).
#: ``snap_to_wall`` then moves each node a fixed distance, so the distortion it sees is
#: that distance *relative to its own layer*, and a draw that lands one layer thin
#: enough turns an element inside out.  Observed on one unchanged commit: 3 passes and 2
#: failures, at min scaled Jacobian -0.988 and -0.989.  That was carried as a non-strict
#: xfail; a check whose result is a draw of the dice is not a check.
#:
#: The underlying defect is real and unfixed -- the fix is to make the mesher
#: independent of the draw (``NEAR_LEN`` / layer thickness, see ``CLAUDE.md``).  What
#: changed is only that this suite no longer pretends to watch it.
EXCLUDED = {"femoral.py"}

#: Examples known to build an inverted element, as a strict xfail so that fixing one
#: fails this test and forces the entry out.  Empty, and worth keeping that way: the
#: only entry was the old two-manifold ``chimera.py`` (min scaled Jacobian about -0.17,
#: on ``main`` too), which has since been deleted.
KNOWN_INVERTED = set()

_CONTAINERS = (LineMesh, QuadMesh, HexMesh)


def _scripts():
    return sorted(f for f in os.listdir(_EXAMPLES) if f.endswith(".py"))


ALL = [n for n in _scripts() if n not in EXCLUDED]


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
    assert set(ALL) | EXCLUDED == set(_scripts())
    assert LIBRARY_ONLY <= set(ALL)
    assert EXCLUDED <= set(_scripts())
    assert KNOWN_INVERTED <= set(ALL)
