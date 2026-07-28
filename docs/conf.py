"""Sphinx configuration for the NekMeshPy documentation.

Build locally with::

    pip install -e ".[docs]"
    MPLBACKEND=Agg sphinx-build -b html -n --keep-going docs docs/_build/html
"""

import os
import sys
from importlib import metadata

# autodoc imports nekmeshpy.io.viz, which imports matplotlib -- force a headless
# backend so the build works in CI without a display.
os.environ.setdefault("MPLBACKEND", "Agg")

# make the package importable from a source checkout (editable install also works)
sys.path.insert(0, os.path.abspath(".."))

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
]

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST (Markdown) narrative
myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3

# -- autodoc -----------------------------------------------------------------
# The reference documents each *leaf* module once with ``automodule`` (see
# docs/reference/*.md); autodoc excludes imported members by default, so the
# classes re-exported from the package ``__init__``s are documented exactly once,
# at their canonical location -- no duplicate/ambiguous cross-references.
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
html_static_path: list[str] = []
