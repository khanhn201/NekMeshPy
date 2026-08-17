"""Sphinx extension registering the ``mesh-viewer`` directive.

Not part of the ``nekmeshpy`` package -- doc-build tooling, loaded by ``docs/conf.py``
via a ``sys.path`` insert of this directory (there is no installed package for it to
live in; ``pyproject.toml`` only packages ``nekmeshpy*``).

The directive itself just emits a ``<div>`` + inline ``<script>`` calling the loader
in ``docs/_static/viewer.js`` (loaded site-wide via ``conf.py``'s ``html_js_files``,
along with vtk.js itself from a CDN). The actual rendering happens entirely in
the browser -- this extension only wires the markup, it does not touch vtk.js itself.
"""

from __future__ import annotations

import itertools

from docutils import nodes
from docutils.parsers.rst import Directive, directives

_counter = itertools.count()


class MeshViewerDirective(Directive):
    """``.. mesh-viewer:: <example-stem>``

    ``<example-stem>`` names a file under ``docs/_static/meshes/<stem>.vtp``, generated
    by ``docs/_ext/gen_viewer_assets.py`` from the matching ``examples/<stem>.py``.
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec = {
        "height": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        stem = self.arguments[0]
        height = self.options.get("height", "420px")
        div_id = "mesh-viewer-%d" % next(_counter)
        env = self.state.document.settings.env
        # `_static/...` is root-relative; docnames nest (e.g. "user/gallery"), so walk
        # back up one level per "/" -- same trick Sphinx's own templates use for
        # resource links, since get_relative_uri is for docnames, not static files.
        vtp_url = "../" * env.docname.count("/") + "_static/meshes/%s.vtp" % stem
        html = (
            '<div class="mesh-viewer" id="%s" '
            'style="width:100%%;height:%s;border:1px solid var(--color-background-border, #ccc);"></div>\n'
            "<script>\n"
            "(function () {\n"
            "  var el = document.getElementById(%r);\n"
            "  function boot() {\n"
            "    if (window.initMeshViewer) {\n"
            "      window.initMeshViewer(el, %r);\n"
            "    } else {\n"
            "      window.setTimeout(boot, 50);\n"
            "    }\n"
            "  }\n"
            "  boot();\n"
            "})();\n"
            "</script>\n"
        ) % (div_id, height, div_id, vtp_url)
        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_directive("mesh-viewer", MeshViewerDirective)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
