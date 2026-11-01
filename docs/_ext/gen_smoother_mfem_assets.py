"""Generate the assets of the MFEM / TMOP pages (``docs/smoother/mfem/``).

Not part of the ``nekmeshpy`` package -- doc-build tooling like ``gen_smoother_hex_assets.py``, whose machinery it reuses; run
it by hand after the TMOP bridge or the example script changes, and commit what it writes:

    python docs/_ext/gen_smoother_mfem_assets.py [workflow thin_layer]

The ``thin_layer`` key needs PyMFEM (``pip install nekmeshpy[smooth]``) and exits with that message when it is missing;
``workflow`` needs no MFEM.  With no argument every key is built, ``workflow`` first, so without PyMFEM the run writes the
workflow chart and then exits with the install message.

* ``workflow`` writes ``docs/smoother/mfem/_generated/workflow.inc``, the step chart of using the TMOP stage.
* ``thin_layer`` runs ``examples/smoother/mfem/01_thin_layer.py`` with ``runpy`` (through the hex generator's ``build``) and
  writes, under ``docs/_static/smoother/mfem/``, ``thin_layer_<variant>_{shell,section}.vtp`` and
  ``thin_layer_manifest.json``, and under ``docs/smoother/mfem/_generated/``, ``thin_layer.{viewer,svg,table,keypoints}.inc``,
  every number read from the run.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gen_smoother_hex_assets as hexgen  # noqa: E402
from gen_smoother_hex_assets import DASH, f4, fint, fpct, ftime, table_md  # noqa: E402

_DOCS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOCS)
GENERATED = os.path.join(_DOCS, "smoother", "mfem", "_generated")
STATIC = os.path.join(_DOCS, "_static", "smoother", "mfem")
_EX = os.path.join(_REPO, "examples", "smoother", "mfem")
WORKFLOW = "workflow.inc"


def spec_thin_layer(ns):
    rows, side = ns["rows"], ns["side"]
    q = {k: rows[k]["q"] for k in rows}
    variants = [("input", "input"), ("native", "native optimizer"), ("ideal", "TMOP ideal"),
                ("input_designed", "TMOP input + designed")]
    headers = ["variant", "first layer, mean change (%)", "max abs. change (%)", "inverted", "min SJ", "mean SJ",
               "time (s)"]

    def row(k, label, ref=False):
        x, d = q[k], rows[k]
        c = [DASH] * 2 if ref else [fpct(d["thick_mean_pct"], 1), "%.1f" % d["thick_max_abs_pct"]]
        return [label] + c + [fint(x.n_inverted), f4(x.sj_min), f4(x.sj_mean), DASH if ref else ftime(d["seconds"])]
    table = table_md(headers, row("input", "input", True), [row(k, l) for k, l in variants[1:]],
                     [None, "absmin", "absmin", "min", "max", "max", "min"], True)
    N, I, D = rows["native"], rows["ideal"], rows["input_designed"]
    P, ID = side["input_plain"], side["ideal_designed"]
    d = P["max_point_diff"]
    diff = "at most %.0e" % d if d < 1e-6 else "up to %.1e" % d
    kp = [
        "**The cube-seeking methods thicken the designed layer**: the native optimizer by %+.0f %% on average (up to "
        "%.0f %%), TMOP `target=\"ideal\"` by %+.0f %% (up to %.0f %%)."
        % (N["thick_mean_pct"], N["thick_max_abs_pct"], I["thick_mean_pct"], I["thick_max_abs_pct"]),
        "**`target=\"input\"` keeps the layer and repairs the interior**: the first layer changes by %+.1f %% on average "
        "(at most %.1f %%), and the minimum scaled Jacobian rises from %.2f to %.2f, against %.2f for the native "
        "optimizer and %.2f for `ideal`."
        % (D["thick_mean_pct"], D["thick_max_abs_pct"], q["input"].sj_min, D["q"].sj_min, N["q"].sj_min, I["q"].sj_min),
        "**Here `designed` is not what holds the layer**: the layer is already box-shaped, so `target=\"input\"` alone "
        "asks for the same thing (without `designed` the result differs by %s). `designed` matters when the bulk target "
        "would reshape the layer: with `target=\"ideal\"` it holds the layer at %+.1f %% instead of %+.0f %%."
        % (diff, ID["thick_mean_pct"], I["thick_mean_pct"]),
    ]
    return dict(
        title="A thin designed layer under the native optimizer and two TMOP targets: cuts through the middle of the box",
        planes=[((0, 0.5 + 0.3 / 6, 0), (0, 1, 0)), ((0.5 + 0.3 / 6, 0, 0), (1, 0, 0))], variants=variants,
        extra={"layer": ns["first_layer"].astype(float)},
        bars=([(l, rows[k]["thick_mean_pct"]) for k, l in variants[1:]], lambda v: "%+.1f %%" % v,
              "First-layer mean thickness change, in % (zero keeps the design)",
              "First-layer mean thickness change of the native optimizer and the two TMOP runs"),
        table=table, keypoints=kp,
        fields=[{"name": "layer", "label": "First layer", "kind": "cat", "range": [0, 1],
                 "legend": "first layer | other"}])


KEYS = {
    "thin_layer": ("01_thin_layer.py", spec_thin_layer),
}


def workflow_chart() -> str:
    """Using the TMOP stage, one box per step, each linking to the section that explains it (hrefs relative to
    ``docs/smoother/mfem/``, where the chart is included).  It reuses the CSS of the quad smoothing workflow chart."""
    from gen_smoother_quad_methods_assets import FLOW_CSS, raw

    api = "backend.html"
    steps = [
        ("Install the extra", "<code>pip install nekmeshpy[smooth]</code>; MFEM is imported only when a "
         "<code>tmop</code> stage runs", api + "#install"),
        ("Set up as usual", "<code>resolve</code> and <code>run_schedule</code>, or a <code>SmoothSession</code>, with "
         "the smoothers' own constraints: frozen vertices and axis-aligned planes and lines carry over, curved targets "
         "are held", api + "#constraints"),
        ("Choose the target", "what each element is asked to become; <code>designed=</code> keeps a layer's own shape",
         api + "#targets-and-designed-elements"),
        ("Choose the metric and solver", "the defaults suit most meshes", api + "#metric-and-solver"),
        ("Propose, read the report, accept", "<code>(\"tmop\", {...})</code> alone or beside the native stages; the "
         "stage's <code>reason</code> says what was held, and a worse run is rolled back", api + "#reading-the-result"),
        ("High-order mesh?", "order 1 for now: TMOP smooths the straight-sided mesh; quads: "
         "<code>rebuild_high_order</code> afterwards, hexes: no rebuild yet", api + "#limits"),
    ]
    chips = {
        3: ('<div class="chips"><span><code>"input"</code> (default): orthogonal, at its own size and aspect</span>'
            '<span><code>"input_smoothed"</code>: the same, sizes relaxed first</span>'
            '<span><code>"ideal"</code>: a square or a cube</span></div>'),
        4: ('<div class="chips"><span>shape: <code>2</code> (quad), <code>303</code> (hex)</span>'
            '<span>shape + size: <code>80</code>, <code>333</code></span>'
            '<span>untangling: <code>313</code></span>'
            '<span><code>"newton"</code> or <code>"lbfgs"</code></span></div>'),
    }
    out = []
    for i, (title, sub, href) in enumerate(steps, start=1):
        out.append('<div class="box st"><span class="n">%d</span><a href="%s"><b>%s</b></a><small>%s</small>%s</div>'
                   % (i, href, title, sub, chips.get(i, "")))
        if i == 5:
            out.append('<div class="arr">&#8595; good enough<br><span class="note">not yet: back to step 3 or 4 '
                       '(<code>undo</code> a worse round)</span></div>')
        elif i < len(steps):
            out.append('<div class="arr">&#8595;</div>')
    out.append('<div class="note">A worked example: <a href="thin_layer.html"><b>Demo: a thin designed layer</b></a>.'
               '</div>')
    return raw("<style>%s</style>" % FLOW_CSS + '<div class="fc" role="group" aria-label="TMOP workflow">'
               + "".join(out) + "</div>")


ALL = ["workflow", "thin_layer"]


def main(argv):
    keys = argv or ALL
    unknown = [k for k in keys if k not in ALL]
    if unknown:
        raise SystemExit("unknown key(s) %s; choose from %s" % (unknown, ALL))
    os.makedirs(STATIC, exist_ok=True)
    os.makedirs(GENERATED, exist_ok=True)
    if "workflow" in keys:
        with open(os.path.join(GENERATED, WORKFLOW), "w") as f:
            f.write(workflow_chart())
        print("workflow       %s written" % WORKFLOW)
    if "thin_layer" in keys:
        from nekmeshpy.core.smoother import mfem

        if not mfem.available():
            raise SystemExit("key thin_layer needs PyMFEM: pip install nekmeshpy[smooth]")
        sizes, total = hexgen.build("thin_layer", KEYS, _EX, STATIC, GENERATED, sub="mfem")
        print("%-14s %2d vtp files, largest %6d B, %7d B written" % ("thin_layer", len(sizes), max(sizes.values()), total))


if __name__ == "__main__":
    main(sys.argv[1:])
