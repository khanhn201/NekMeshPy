"""Generate the boundary-surface ``.vtp`` assets the docs' live mesh viewer loads.

Not part of the ``nekmeshpy`` package (pyproject only packages ``nekmeshpy*``) -- this
is doc-build tooling, run once before ``sphinx-build`` (see ``.github/workflows/docs.yml``
and ``CLAUDE.md``'s Commands block), same as a human running it locally:

    python docs/_ext/gen_viewer_assets.py

Each example is executed the same way ``tests/test_examples.py`` does (``runpy.run_path``,
cwd inside a scratch directory so any files an example writes don't land in the repo),
and its ``mesh`` global is exported through ``writer.boundary_to_vtp`` -- the boundary
surface only, corners only, never the interior volume a ``.vtu`` would carry. That is
what keeps even the largest example (``chimera_full.py``, a few hundred MB as a ``.vtu``)
web-sized: a viewer only ever shows the outer surface anyway.
"""

from __future__ import annotations

import os
import runpy
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOCS)
_EXAMPLES = os.path.join(_REPO, "examples")
_OUT = os.path.join(_DOCS, "_static", "meshes")

sys.path.insert(0, _REPO)

#: Scripts that build no ``mesh`` of their own (imported as a library by another
#: example) or need an extra the ``docs`` install group doesn't pull in (``gmsh``, for
#: the tet-meshed femoral pair) -- mirrors ``tests/test_examples.py``'s
#: ``LIBRARY_ONLY`` / ``EXCLUDED`` sets, kept separate rather than imported from there since
#: ``tests/`` isn't packaged either and this script has its own, narrower reason to skip
#: each one.
SKIP = {"tjunction_lib.py", "femoral_vol.py", "femoral.py"}


def _examples() -> list[str]:
    return sorted(f for f in os.listdir(_EXAMPLES)
                  if f.endswith(".py") and f not in SKIP)


def _run(name: str, scratch: str) -> dict:
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        return runpy.run_path(os.path.join(_EXAMPLES, name), run_name="__main__")
    finally:
        os.chdir(cwd)


def main() -> None:
    from nekmeshpy import HexMesh
    from nekmeshpy.io import writer

    os.makedirs(_OUT, exist_ok=True)
    for name in _examples():
        stem = name[:-3]
        with tempfile.TemporaryDirectory(prefix="nekmeshpy-viewer-") as scratch:
            print("building %s ..." % name, flush=True)
            ns = _run(name, scratch)
        mesh = ns.get("mesh")
        if not isinstance(mesh, HexMesh):
            print("  skip: %s defines no HexMesh `mesh`" % name)
            continue
        out_path = os.path.join(_OUT, "%s.vtp" % stem)
        writer.boundary_to_vtp(mesh, out_path)
        size = os.path.getsize(out_path)
        print("  %s: %.2f MB" % (os.path.basename(out_path), size / 1e6))


if __name__ == "__main__":
    main()
