"""Sphinx configuration for the NekMeshPy documentation.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -b html -n --keep-going docs docs/_build/html
"""

import os
import sys
from importlib import metadata

# autodoc imports nekmeshpy.io.viz, which imports matplotlib -- force a headless
# backend so the build works in CI without a display.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

# make the package importable from a source checkout (editable install also works)
sys.path.insert(0, os.path.abspath(".."))
# the mesh-viewer directive lives in docs/_ext -- doc-build tooling, not a package
sys.path.insert(0, os.path.abspath("_ext"))

# -- Project information ------------------------------------------------------
project = "NekMeshPy"
author = "NekMeshPy contributors"
copyright = "NekMeshPy contributors"
try:
    release = metadata.version("nekmeshpy")
except metadata.PackageNotFoundError:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "mesh_viewer",
]

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST (Markdown) narrative
myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3

# -- autodoc -----------------------------------------------------------------
# The reference documents each *leaf* module once with ``automodule`` (see
# docs/reference/*.md). Autodoc's usual "skip imported members" rule does NOT
# apply here: every rung package (linemesh/quadmesh/hexmesh) declares an
# ``__all__``, and once a module has one, autodoc treats it as the authoritative
# member list regardless of where each name was actually defined -- so a bare
# ``automodule:: nekmeshpy.quadmesh :members:`` (even with an explicit, narrowed
# ``:members:`` list) still pulls in every re-exported operation, duplicating the
# full docstring already rendered under its owning submodule
# (``nekmeshpy.quadmesh.shape``, ``.morph``, ...). The reference pages route
# around this: each rung's own top section uses ``autoclass``/``autodata`` for
# just its container class and constants, which documents that exact object at
# that exact dotted path without going through ``__all__`` at all -- everything
# else is documented once, under the submodule that actually defines it.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# the project's numpy dtype aliases are documentation aliases with no class page
_TYPE_ALIASES = [
    "FloatArray", "IntArray", "BoolArray", "StrArray",
    "Point", "Vec3", "PointArray",
]

# nitpicky (-n) mode flags every unresolved xref.  Ignore targets we can never
# resolve: third-party objects without an inventory entry (numpy internals, scipy,
# meshio) and the project's own type aliases / typevars, which have no class page.
nitpick_ignore_regex = [
    (r"py:.*", r"numpy\..*"),
    (r"py:.*", r"scipy\..*"),
    (r"py:.*", r"sp\..*"),  # scipy.sparse imported as `sp` in annotations
    (r"py:.*", r"meshio.*"),
    (r"py:.*", r"'?(" + "|".join(_TYPE_ALIASES) + r")'?"),
    (r"py:.*", r"GroupsArg"),
    (r"py:.*", r"nekmeshpy\..*\.[A-Z]$"),  # single-letter TypeVars (F, ...)
]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"NekMeshPy {release}"
html_static_path = ["_static"]
# furo's footer_icons wants the icon as literal SVG markup (see furo's page.html
# template) -- there's no "named icon" shorthand, so pull the mark from a CDN
# (jsdelivr mirrors Simple Icons) rather than inlining the path data here.
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/khanhn201/nekmeshpy",
            "html": (
                '<img class="github-icon" '
                'src="https://cdn.jsdelivr.net/npm/simple-icons@v13/icons/github.svg" '
                'alt="" width="1em" height="1em" style="vertical-align:middle">'
            ),
        },
    ],
}
# furo styles a visited link in a separate purple brand color by default --
# custom.css makes it match an ordinary link instead, in both light and dark mode.
html_css_files = ["custom.css"]
# vtk.js loaded from jsdelivr rather than vendored -- the unscoped "vtk.js" npm
# package publishes the same prebuilt UMD bundle as @kitware/vtk.js (which itself
# ships no UMD dist), just at a plain root path. Load order matters: viewer.js
# reads window.vtk at call time (via a small retry), but declaring vtk.js first keeps
# the intent obvious.
html_js_files = [
    "https://cdn.jsdelivr.net/npm/vtk.js@36.7.1/vtk.js",
    "viewer.js",
]
