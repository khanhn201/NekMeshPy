"""Generate the assets of the MFEM / TMOP pages (``docs/smoother/mfem/``).

Not part of the ``nekmeshpy`` package -- doc-build tooling like ``gen_smoother_hex_assets.py``, whose machinery it reuses; run
it by hand after the TMOP bridge or the example script changes, and commit what it writes:

    python docs/_ext/gen_smoother_mfem_assets.py [workflow thin_layer bent_pipe]

The ``thin_layer`` and ``bent_pipe`` keys need PyMFEM (``pip install nekmeshpy[smooth]``) and exits with that message when it is missing;
``workflow`` needs no MFEM.  With no argument every key is built, ``workflow`` first, so without PyMFEM the run writes the
workflow chart and then exits with the install message.

* ``workflow`` writes ``docs/smoother/mfem/_generated/workflow.inc``, the step chart of using the TMOP stage.
* ``thin_layer`` runs ``examples/smoother/mfem/01_thin_layer.py`` with ``runpy`` (through the hex generator's ``build``) and
  writes, under ``docs/_static/smoother/mfem/``, ``thin_layer_<variant>_{shell,section}.vtp`` and
  ``thin_layer_manifest.json``, and under ``docs/smoother/mfem/_generated/``, ``thin_layer.{viewer,svg,table,keypoints}.inc``,
  every number read from the run.
* ``bent_pipe`` runs ``examples/smoother/mfem/02_bent_pipe.py`` and asserts the claims C1-C5 (``claims_curved``)
  on its results before writing anything, then writes ``bent_pipe_<variant>_{shell,section}.vtp``,
  ``bent_pipe_manifest.json`` and ``bent_pipe.{viewer,svg,table,keypoints}.inc`` the same way.  Its viewer
  assets are its own (``build_curved``): a boundary face is one polygon whose outline is the resampled curved edge,
  and the section is two layers of existing faces, not a plane cut.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

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


# -- bent_pipe ---------------------------------------------------------------------------------------------------
CURVED_VARIANTS = [("input", "input"), ("order1", "order-1 route"), ("native", "native optimizer"),
                   ("tmop", "TMOP, curved")]
CURVED_CAMERA = {"direction": [0.5, -1.4, 0.6], "viewUp": [0, 0, 1]}


def claims_curved(ns):
    """Assert the claims C1-C5 of the bent-pipe page on the demo's results ``ns``; each message names its claim.
    A failing claim means the key point is rewritten from what the run shows, never that the demo is tuned to save it."""
    rows, checks = ns["rows"], ns["checks"]
    inp, o1, nat, tm = rows["input"], rows["order1"], rows["native"], rows["tmop"]
    assert (tm["order"] == ns["ORDER"] and tm["sj7_min"] > inp["sj7_min"] + 0.1
            and tm["sj_nodes_min"] > inp["sj_nodes_min"] + 0.1 and tm["sj_corner_min"] > inp["sj_corner_min"] + 0.05
            and tm["n_inv7"] == 0 and tm["n_inv_corner"] == 0), \
        "C1 (TMOP curved repairs the mesh and keeps its order) failed: %s vs input %s" % (tm, inp)
    assert (checks["wall_max_move"] == 0.0 and checks["cap_max_off"] < 1e-12
            and tm["wall_dev_pct"] == inp["wall_dev_pct"]), \
        "C2 (the wall is held, the caps slide) failed: %s" % checks
    assert (o1["order"] == 1 and o1["wall_dev_pct"] > 100 * max(tm["wall_dev_pct"], 1e-4)
            and o1["vol_err_pct"] < -1), "C3 (the order-1 route loses the wall) failed: %s" % o1
    assert (nat["order"] == ns["ORDER"] and nat["sj_corner_min"] > inp["sj_corner_min"] + 0.05
            and tm["sj7_min"] > nat["sj7_min"] + 0.05 and tm["sj_nodes_mean"] > nat["sj_nodes_mean"] + 0.05
            and nat["wall_dev_pct"] > tm["wall_dev_pct"]), \
        "C4 (vertex-only smoothing leaves the curved nodes' defect) failed: native %s tmop %s" % (nat, tm)
    assert all(checks["conforming"].values()), "C5 (every result is conformal) failed: %s" % checks["conforming"]
    assert o1["sj7_min"] > max(tm["sj7_min"], nat["sj7_min"]), \
        "the order-1 route no longer reads highest (its key point says it does): %s" % o1
    # the page's intro types the case: an order-3 bent pipe, 20 quads x 8 layers = 160 hexes, read at order 7
    assert (ns["ORDER"], ns["N_HEX"], ns["LAYERS"], ns["SAMPLE_ORDER"]) == (3, 160, 8, 7), \
        "bent_pipe.md types order 3, 160 hexes, 8 layers, order 7: update the page"


def grouped_bars_svg(groups, series, fmt, aria, caption):
    """Theme-aware grouped horizontal bars: ``groups`` is ``[(label, [value per series])]``, ``series`` the legend
    ``[(name, colour)]``.  The axis runs from ``min(0, lowest)`` to 1, the zero line is drawn, every bar ends in its value.
    Colours go through furo's CSS variables, so it follows light / dark.  Inline SVG, no library."""
    vals = [v for _, vs in groups for v in vs]
    lo, hi = min(0.0, min(vals)), 1.0
    label_w, plot_w, pad_r, bar_h, gap = 130.0, 250.0, 56.0, 12.0, 12.0
    top = 26.0                                           # the legend line
    left = 40.0 if lo < 0 else 0.0                       # room for negative value labels
    x0 = label_w + 6 + left
    scale = plot_w / (hi - lo)
    zero = x0 - lo * scale
    n = len(series)
    gh = n * bar_h + gap
    width, height = x0 + plot_w + pad_r, top + gh * len(groups) + 18
    txt = "fill: var(--color-foreground-primary); font: 12px sans-serif"
    muted = "fill: var(--color-foreground-muted); font: 11px sans-serif"
    out = ['<figure class="hx-bars" style="margin: 1rem 0; max-width: 38rem">',
           '<svg viewBox="0 0 %.0f %.0f" width="100%%" role="img" aria-label="%s">' % (width, height, aria)]
    lx = label_w + 6
    for name, col in series:                             # legend
        out.append('<rect x="%.1f" y="6" width="10" height="10" rx="2" style="fill: %s"/>' % (lx, col))
        out.append('<text x="%.1f" y="15" style="%s">%s</text>' % (lx + 14, muted, name))
        lx += 14 + 6.2 * len(name) + 14
    for g, (lab, vs) in enumerate(groups):
        y = top + g * gh
        out.append('<text x="%.1f" y="%.1f" text-anchor="end" style="%s">%s</text>'
                   % (label_w, y + n * bar_h / 2 + 4, txt, lab))
        for i, (v, (_, col)) in enumerate(zip(vs, series)):
            w = abs(v) * scale
            x = zero if v >= 0 else zero - w
            by = y + i * bar_h
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" style="fill: %s"/>'
                       % (x, by + 1, max(w, 1.0), bar_h - 2, col))
            tx, anchor = (zero + w + 4, "start") if v >= 0 else (zero - w - 4, "end")
            out.append('<text x="%.1f" y="%.1f" text-anchor="%s" style="%s">%s</text>'
                       % (tx, by + bar_h - 2, anchor, muted, fmt(v)))
    ybot = top + gh * len(groups)
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" style="stroke: var(--color-background-border); '
               'stroke-width: 1"/>' % (zero, top - 4, zero, ybot - 4))
    for tv in (0.0, 1.0):
        out.append('<text x="%.1f" y="%.1f" text-anchor="middle" style="%s">%g</text>'
                   % (zero + tv * scale, ybot + 10, muted, tv))
    out.append("</svg>")
    out.append("<figcaption>%s</figcaption>" % caption)
    out.append("</figure>")
    return "```{raw} html\n%s\n```\n" % "\n".join(out)


def spec_bent_pipe(ns):
    """Everything the page shows that is not a mesh file, read from the demo's results ``ns``."""
    rows, order, so = ns["rows"], ns["ORDER"], ns["SAMPLE_ORDER"]
    headers = ["variant", "order", "min SJ, nodes", "mean SJ, nodes", "min SJ, order %d" % so,
               "inverted, order %d" % so, "min SJ, corners", "wall off the torus, max (% of R)", "time (s)"]

    def row(k, label, ref=False):
        d = rows[k]
        return [label, fint(d["order"]), f4(d["sj_nodes_min"]), f4(d["sj_nodes_mean"]), f4(d["sj7_min"]),
                fint(d["n_inv7"]), f4(d["sj_corner_min"]), "%.3f" % d["wall_dev_pct"],
                DASH if ref else ftime(d["seconds"])]
    table = table_md(headers, row("input", "input", True), [row(k, l) for k, l in CURVED_VARIANTS[1:]],
                     [None, None, "max", "max", "max", "min", "max", "min", "min"], True)
    inp, o1, nat, tm = rows["input"], rows["order1"], rows["native"], rows["tmop"]
    ck = ns["checks"]
    kp = [
        "**TMOP smooths the curved mesh as curved.** The order stays %d and every node moves: the minimum scaled "
        "Jacobian rises from %.2f to %.2f on the order-%d lattice and from %.2f to %.2f at the corners, with no hex "
        "inverted, in %d Newton iterations."
        % (tm["order"], inp["sj7_min"], tm["sj7_min"], so, inp["sj_corner_min"], tm["sj_corner_min"], tm["iters"]),
        "**The wall is held, the caps slide.** No wall node moves, and the cap nodes stay in their planes to %.0e. "
        "Between its nodes the wall stays as close to the torus as the input's order-%d wall: %.3f %% of R."
        % (max(ck["cap_max_off"], 1e-16), order, tm["wall_dev_pct"]),
        "**The order-1 route loses the wall.** Its result is straight-sided: off the torus by up to %.1f %% of R, and "
        "%.1f %% short in volume. Its scaled Jacobian reads highest, but it is the reading of a different, faceted "
        "geometry." % (o1["wall_dev_pct"], -o1["vol_err_pct"]),
        "**Moving vertices cannot repair curved nodes.** The native optimizer raises the corners from %.2f to %.2f, but "
        "the perturbation of the curved nodes rides along: %.2f on the order-%d lattice against %.2f for TMOP, and a "
        "mean of %.2f at the nodes against %.2f. Its wall vertices slide on the faceted input wall and the carried "
        "nodes are not projected back, so the wall drifts to %.3f %% of R."
        % (inp["sj_corner_min"], nat["sj_corner_min"], nat["sj7_min"], so, tm["sj7_min"], nat["sj_nodes_mean"],
           tm["sj_nodes_mean"], nat["wall_dev_pct"]),
        "**Every result is conformal and watertight.**",
    ]
    series = [("stored nodes", "var(--color-brand-primary)"), ("order %d" % so, "var(--color-foreground-muted)"),
              ("corners", "#1f9d6b")]
    chart = grouped_bars_svg(
        [(l, [rows[k]["sj_nodes_min"], rows[k]["sj7_min"], rows[k]["sj_corner_min"]]) for k, l in CURVED_VARIANTS],
        series, lambda v: "%.2f" % v,
        "Minimum scaled Jacobian of the input and the three results, at the nodes, at order %d and at the corners" % so,
        "Minimum scaled Jacobian read three ways: at the stored nodes, on the order-%d lattice, at the corners" % so)
    return dict(
        title="A bent pipe, order %d: shell and two face layers (mid-bend, symmetry plane)" % order,
        fields=[{"name": "sj7", "label": "Scaled Jacobian (order %d, worst point)" % so, "kind": "sj", "range": [0, 1]},
                {"name": "sj_nodes", "label": "Scaled Jacobian (stored nodes)", "kind": "sj", "range": [0, 1]},
                {"name": "sj_corner", "label": "Scaled Jacobian (corners, the .re2 geometry)", "kind": "sj",
                 "range": [0, 1]}],
        table=table, keypoints=kp, chart=chart)


def _perimeter(m_):
    """Indices, in order, of the perimeter of an ``(m_+1) x (m_+1)`` lattice stored axis 0 fastest (``4 * m_`` of them)."""
    r = m_ + 1
    i = np.arange(m_)
    return np.concatenate([i, m_ + i * r, m_ * r + m_ - i, (m_ - i) * r])


def _polygons(blocks, order, M):
    """The ``(F, 4M, 3)`` outlines of the face ``blocks``: each resampled to order ``M`` (a change of nodal basis, so the
    curved edge is the stored one) and its perimeter taken in order."""
    from nekmeshpy.core import interp

    fine = interp.resample_block(blocks, order, M, 2) if len(blocks) else np.zeros((0, (M + 1) ** 2, 3))
    return fine[:, _perimeter(M), :]


def _newell(poly):
    """Unnormalised normals ``(F, 3)`` of closed polygons ``(F, n, 3)``."""
    nxt = np.roll(poly, -1, axis=1)
    return np.stack([((poly[..., 1] - nxt[..., 1]) * (poly[..., 2] + nxt[..., 2])).sum(1),
                     ((poly[..., 2] - nxt[..., 2]) * (poly[..., 0] + nxt[..., 0])).sum(1),
                     ((poly[..., 0] - nxt[..., 0]) * (poly[..., 1] + nxt[..., 1])).sum(1)], axis=1)


def _poly_vtp(path, polys, cell_data):
    """``polys`` ``(F, n, 3)`` as one polygon per face, with ``cell_data`` ``{name: (values, type)}``."""
    F, n = polys.shape[:2]
    return hexgen.write_vtp(path, polys.reshape(-1, 3), np.arange(F * n), np.arange(n, F * n + 1, n), cell_data)


def curved_shell_vtp(ns, m, path):
    """Every boundary face of ``m`` as one polygon with the curved outline, wound outward (normal away from the
    owning hex's centroid)."""
    from nekmeshpy import hexmesh, quadmesh

    M = 4 * ns["ORDER"]
    qm = m.quad_mesh
    ids = np.flatnonzero(hexmesh.boundary_face_ids(m))
    polys = _polygons(quadmesh.element_blocks(qm)[ids], m.order, M)
    owner = np.full(qm.n_quads, -1)
    owner[m.hexes.ravel()] = np.repeat(np.arange(m.n_hexes), 6)
    hc = m.points[m.corners].mean(axis=1)[owner[ids]]
    out = (_newell(polys) * (polys.mean(axis=1) - hc)).sum(axis=1) > 0
    polys = np.where(out[:, None, None], polys, polys[:, ::-1])
    return _poly_vtp(path, polys, {})


def curved_section_vtp(ns, m, fields, path):
    """The two face layers of ``m`` (``plane`` 0 at mid-bend, 1 in the symmetry plane) as polygons with the curved
    outline; each face carries the minimum of its incident hexes' values of every field."""
    from nekmeshpy import quadmesh

    M = 4 * ns["ORDER"]
    qm = m.quad_mesh
    blocks = quadmesh.element_blocks(qm)
    per_face = {}
    for name, v in fields.items():
        a = np.full(qm.n_quads, np.inf)
        np.minimum.at(a, m.hexes.ravel(), np.repeat(np.asarray(v, float), 6))
        per_face[name] = a
    polys, data, plane = [], {k: [] for k in fields}, []
    for pid, key in enumerate(("mid", "sym")):
        ids = np.flatnonzero(np.isin(qm.corners, ns["SECTION_VERTS"][key]).all(axis=1))
        assert len(ids), "empty section layer %r" % key
        polys.append(_polygons(blocks[ids], m.order, M))
        plane += [pid] * len(ids)
        for k in fields:
            data[k] += per_face[k][ids].tolist()
    cd = {k: (v, "Float32") for k, v in data.items()}
    cd["plane"] = (plane, "Int32")
    return _poly_vtp(path, np.concatenate(polys), cd)


def curved_variant_files(ns, key, vk, static):
    """Write one variant's shell and section files into ``static``; returns ``{file name: bytes}``."""
    m = ns["meshes"][vk]
    shell, section = "%s_%s_shell.vtp" % (key, vk), "%s_%s_section.vtp" % (key, vk)
    return {shell: curved_shell_vtp(ns, m, os.path.join(static, shell)),
            section: curved_section_vtp(ns, m, ns["fields"][vk], os.path.join(static, section))}


def curved_variant_note(ns, vk):
    d, so = ns["rows"][vk], ns["SAMPLE_ORDER"]
    return ("order %d | min scaled J: %.2f at order %d, %.2f at the corners | %d inverted"
            % (d["order"], d["sj7_min"], so, d["sj_corner_min"], d["n_inv7"]))


#: The marked blocks of the bent-pipe script that ``bent_pipe.md`` and ``backend.md`` include verbatim (``literalinclude``
#: between ``# [doc:NAME]`` and ``# [/doc:NAME]``): the page shows the script's own code, so it cannot drift from it.
EXCERPTS = ("build", "constraints", "tmop", "quality", "export")


def check_excerpts(script, names=EXCERPTS):
    """Assert that ``script`` holds each marked block exactly once, opened before it is closed, and not empty."""
    with open(script) as f:
        lines = f.read().splitlines()
    for n in names:
        o = [i for i, ln in enumerate(lines) if ln.strip() == "# [doc:%s]" % n]
        c = [i for i, ln in enumerate(lines) if ln.strip() == "# [/doc:%s]" % n]
        assert len(o) == 1 and len(c) == 1 and o[0] + 1 < c[0], \
            "%s: the block [doc:%s] the docs include is missing, doubled or empty" % (os.path.basename(script), n)


def build_curved(key="bent_pipe", static=None, generated=None):
    """Run the bent-pipe demo, assert C1-C5, write the viewer files, manifest and fragments; ``(sizes, total bytes)``.
    ``hexgen.build`` is not used: it draws corners and cuts planes, this draws curved faces."""
    static = STATIC if static is None else static
    generated = GENERATED if generated is None else generated
    script, spec_fn = KEYS[key]
    check_excerpts(os.path.join(_EX, script))
    ns = hexgen.run_script(os.path.join(_EX, script))
    claims_curved(ns)
    sp = spec_fn(ns)
    sizes, variants = {}, []
    for vk, label in CURVED_VARIANTS:
        files = curved_variant_files(ns, key, vk, static)
        sizes.update(files)
        shell, section = files
        variants.append({
            "label": label, "note": curved_variant_note(ns, vk),
            "layers": [{"name": "wall", "role": "surface", "label": "Boundary shell", "vtp": shell},
                       {"name": "section", "role": "section", "label": "Face layers", "vtp": section}],
            "fields": sp["fields"]})
    big = {k: v for k, v in sizes.items() if v > hexgen.MAX_VTP_BYTES}
    assert not big, big
    manifest = os.path.join(static, key + "_manifest.json")
    with open(manifest, "w") as f:
        json.dump({"title": sp["title"], "camera": CURVED_CAMERA, "variants": variants}, f, indent=1)
    inc = {"viewer": hexgen.viewer_inc(key, "mfem"), "svg": sp["chart"], "table": sp["table"],
           "keypoints": "".join("- %s\n" % k for k in sp["keypoints"])}
    for name, text in inc.items():
        with open(os.path.join(generated, "%s.%s.inc" % (key, name)), "w") as f:
            f.write(text)
    return sizes, sum(sizes.values()) + os.path.getsize(manifest) + sum(map(len, inc.values()))


KEYS = {
    "thin_layer": ("01_thin_layer.py", spec_thin_layer),
    "bent_pipe": ("02_bent_pipe.py", spec_bent_pipe),
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
         "the smoothers' own constraints: frozen nodes and axis-aligned planes and lines carry over, curved targets "
         "are held", api + "#constraints"),
        ("Choose the target", "what each element is asked to become; <code>designed=</code> keeps a layer's own shape",
         api + "#targets-and-designed-elements"),
        ("Choose the metric and solver", "the defaults suit most meshes", api + "#metric-and-solver"),
        ("Propose, read the report, accept", "<code>(\"tmop\", {...})</code> alone or beside the native stages; the "
         "stage's <code>reason</code> says what was held, and a worse run is rolled back", api + "#reading-the-result"),
        ("High-order mesh?", "smoothed as curved: every node moves and the order is kept; read the report at the "
         "solver's order and at the corners", api + "#high-order"),
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
    out.append('<div class="note">Worked examples: <a href="thin_layer.html"><b>Demo: a thin designed layer</b></a>, '
               '<a href="bent_pipe.html"><b>Demo: a bent pipe at order 3</b></a>.</div>')
    return raw("<style>%s</style>" % FLOW_CSS + '<div class="fc" role="group" aria-label="TMOP workflow">'
               + "".join(out) + "</div>")


ALL = ["workflow", "thin_layer", "bent_pipe"]


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
    if "bent_pipe" in keys:
        from nekmeshpy.core.smoother import mfem

        if not mfem.available():
            raise SystemExit("key bent_pipe needs PyMFEM: pip install nekmeshpy[smooth]")
        sizes, total = build_curved("bent_pipe")
        print("%-14s %2d vtp files, largest %6d B, %7d B written" % ("bent_pipe", len(sizes), max(sizes.values()), total))


if __name__ == "__main__":
    main(sys.argv[1:])
