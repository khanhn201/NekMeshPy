"""Generate the assets of the hex smoothing deep-dive pages (``docs/hexsmoother/``).

Not part of the ``nekmeshpy`` package -- doc-build tooling like ``gen_smoother_hex_assets.py``; run it by hand after
the hex smoother or one of the example scripts changes, and commit what it writes (a few seconds):

    python docs/_ext/gen_hexsmoother_assets.py [methods slide_box free_boundary thin_layer untangle]

With no argument every key is built.  Each key runs one script of ``examples/hexsmoother/`` with ``runpy`` (cwd in a
scratch directory, stdout captured) and reads its public namespace (``rows``, ``meshes``, ``sj`` ...).  It writes, under
``docs/_static/hexsmoother/``,

* ``<key>_<variant>_shell.vtp``    every boundary face of the variant's mesh as quads;
* ``<key>_<variant>_section.vtp``  the hex cross-section: polygons where two planes cut the hexes, with the per-hex
  worst-corner scaled Jacobian ``sj`` (and ``layer`` / ``inv`` where the script has such a flag);
* ``<key>_manifest.json``          what ``smoother_hex_viewer.js`` reads (no ``groups`` layer: the view opens on the
  cross-section);

and under ``docs/hexsmoother/_generated/``, per key, ``<key>.viewer.inc``, ``<key>.svg.inc`` (a theme-aware bar chart),
``<key>.table.inc`` and ``<key>.keypoints.inc``, every number read from the run.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import runpy
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOCS)
GENERATED = os.path.join(_DOCS, "hexsmoother", "_generated")
STATIC = os.path.join(_DOCS, "_static", "hexsmoother")
_EX = os.path.join(_REPO, "examples", "hexsmoother")
MAX_VTP_BYTES = 400_000
MAX_TOTAL_BYTES = 4_000_000
CAMERA = {"direction": [1.0, -1.4, 1.1], "viewUp": [0, 0, 1]}
sys.path.insert(0, _REPO)


# -- small helpers (copies of the ones in gen_smoother_hex_assets.py) -------------------------------------------------
def b64(a):
    """One binary ``DataArray`` payload: the byte count (UInt64) then the bytes, base64'd as a single blob."""
    raw = np.ascontiguousarray(a).tobytes()
    return base64.b64encode(np.array([len(raw)], dtype="<u8").tobytes() + raw).decode()


def write_vtp(path, points, conn, offsets, cell_data):
    """Binary PolyData; ``cell_data`` is {name: (values, "Float32" | "Int32")}."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    offsets = np.asarray(offsets, dtype=np.int32)
    out = [
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
        '<PolyData>\n<Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="%d">\n'
        % (len(points), len(offsets)),
        '<Points><DataArray type="Float32" NumberOfComponents="3" format="binary">%s</DataArray></Points>\n' % b64(points),
        '<Polys><DataArray type="Int32" Name="connectivity" format="binary">%s</DataArray>\n'
        '<DataArray type="Int32" Name="offsets" format="binary">%s</DataArray></Polys>\n'
        % (b64(np.asarray(conn, dtype=np.int32)), b64(offsets)),
    ]
    if cell_data:
        out.append("<CellData>")
        for name, (vals, typ) in cell_data.items():
            dt = np.float32 if typ == "Float32" else np.int32
            out.append('<DataArray type="%s" Name="%s" format="binary">%s</DataArray>' % (typ, name, b64(np.asarray(vals, dt))))
        out.append("</CellData>\n")
    out.append("</Piece>\n</PolyData>\n</VTKFile>\n")
    with open(path, "w") as f:
        f.write("".join(out))
    return os.path.getsize(path)


def shell_vtp(m, path):
    """Every boundary face of ``m`` as a quad, wound outward, no cell data."""
    from nekmeshpy.hexmesh import smoother as hs

    q = hs.boundary_faces_outward(m)
    return write_vtp(path, m.points, q.ravel(), np.arange(4, 4 * len(q) + 1, 4), {})


def cut(m, sj, extra, origin, normal, edges):
    """Polygons where a plane cuts the hexes of ``m``: ``(polygons, sj, {name: values})`` per cut hex."""
    X = m.points[m.corners]
    normal = np.asarray(normal, float)
    d = (X - np.asarray(origin, float)) @ normal
    ref = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    polys, s, ex = [], [], {k: [] for k in extra}
    for e in np.flatnonzero((d.min(axis=1) < 0) & (d.max(axis=1) > 0)):
        da, db = d[e, edges[:, 0]], d[e, edges[:, 1]]
        x = da * db < 0
        t = (da[x] / (da[x] - db[x]))[:, None]
        pts = X[e, edges[x, 0]] + t * (X[e, edges[x, 1]] - X[e, edges[x, 0]])
        if len(pts) < 3:
            continue
        c = pts.mean(axis=0)
        ang = np.arctan2((pts - c) @ v, (pts - c) @ u)
        polys.append(pts[np.argsort(ang)])
        s.append(sj[e])
        for k in extra:
            ex[k].append(float(extra[k][e]))
    return polys, np.array(s), {k: np.array(v_) for k, v_ in ex.items()}


def section_vtp(m, sj, extra, planes, path):
    """The cuts of every ``(origin, normal)`` in ``planes`` as one PolyData with ``sj``, ``plane`` and ``extra``."""
    from nekmeshpy.core import conform

    edges = conform._LOCAL_EDGES[3]
    pts, conn, off = [], [], []
    cd = {"sj": [], "plane": []}
    cd.update({k: [] for k in extra})
    for pid, (o, n) in enumerate(planes):
        polys, s, ex = cut(m, sj, extra, o, n, edges)
        for pg in polys:
            conn += list(range(len(pts), len(pts) + len(pg)))
            pts += pg.tolist()
            off.append(len(conn))
        cd["sj"] += s.tolist()
        cd["plane"] += [pid] * len(polys)
        for k in extra:
            cd[k] += ex[k].tolist()
    data = {"sj": (cd["sj"], "Float32"), "plane": (cd["plane"], "Int32")}
    data.update({k: (cd[k], "Float32") for k in extra})
    return write_vtp(path, pts, conn, off, data)


def run_script(path):
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)
        try:
            return runpy.run_path(path, run_name="gen_hexsmoother")
        finally:
            os.chdir(cwd)


# -- number formats -------------------------------------------------------------------------------------------------
def f4(x):
    return "%.4f" % x


def fdist(x):
    return "0" if x == 0 else "%.1e" % x


def fpct(x, n=2):
    return "%+.*f" % (n, x)


def ftime(x):
    return "%.2f" % x


def fint(x):
    return "%d" % x


DASH = "—"


# -- bar chart, table ----------------------------------------------------------------------------------------------
def bars_svg(items, fmt, metric, aria):
    """A theme-aware horizontal bar chart: ``items`` is ``[(label, value)]``; a bar labelled ``input`` is the reference
    (muted).  Colours only
    through furo's CSS variables, so it follows light / dark.  Bars grow from the zero line, negatives to the left."""
    vals = [v for _, v in items]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    if hi == lo:
        hi = lo + 1.0
    label_w, plot_w, row_h, pad_l, pad_r = 140.0, 220.0, 24.0, 6.0, 64.0
    left = 64.0 if lo < 0 else 0.0
    x0 = label_w + pad_l + left
    scale = plot_w / (hi - lo)
    zero = x0 + (-lo) * scale
    width = x0 + plot_w + pad_r
    height = row_h * len(items) + 12
    txt = "fill: var(--color-foreground-primary); font: 12px sans-serif"
    muted = "fill: var(--color-foreground-muted); font: 12px sans-serif"
    out = ['<figure class="hx-bars" style="margin: 1rem 0; max-width: 34rem">',
           '<svg viewBox="0 0 %.0f %.0f" width="100%%" role="img" aria-label="%s">' % (width, height, aria)]
    for i, (lab, v) in enumerate(items):
        y = 6 + i * row_h
        w = abs(v) * scale
        x = zero if v >= 0 else zero - w
        fill = "var(--color-foreground-muted)" if lab == "input" else "var(--color-brand-primary)"
        out.append('<text x="%.1f" y="%.1f" text-anchor="end" style="%s">%s</text>'
                   % (label_w, y + 14, txt, lab))
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" style="fill: %s"/>'
                   % (x, y + 3, max(w, 1.0), row_h - 9, fill))
        tx, anchor = (zero + w + 4, "start") if v >= 0 else (zero - w - 4, "end")
        out.append('<text x="%.1f" y="%.1f" text-anchor="%s" style="%s">%s</text>'
                   % (tx, y + 14, anchor, muted, fmt(v)))
    out.append('<line x1="%.1f" y1="2" x2="%.1f" y2="%.1f" style="stroke: var(--color-background-border); stroke-width: 1"/>'
               % (zero, zero, height - 4))
    out.append("</svg>")
    out.append("<figcaption>%s</figcaption>" % metric)
    out.append("</figure>")
    return "```{raw} html\n%s\n```\n" % "\n".join(out)


BOLD_NOTE = ("**Bold**: the best value in each column among the results (highest scaled Jacobian; lowest count, "
             "objective, time, change or distance); the input row is for reference.")


def table_md(headers, input_cells, runs, rules, with_time):
    """Markdown table: the input row first, then the runs; the best value of each column among the runs in bold (ties all
    bold, compared as formatted)."""
    cols = len(headers)
    best = {}
    for c in range(1, cols):
        if rules[c] is None:
            continue
        nums = [float(r[c]) for r in runs if r[c] != DASH]
        if not nums:
            continue
        if rules[c] == "absmin":       # |x| < 1e-12 is round-off: tied with an exact 0
            nums = [0.0 if abs(x) < 1e-12 else x for x in nums]
        key = {"max": max(nums), "min": min(nums), "absmin": min(nums, key=abs)}[rules[c]]
        best[c] = key
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (cols - 1)) + "|",
             "| " + " | ".join(input_cells) + " |"]
    for r in runs:
        cells = [r[0]]
        for c in range(1, cols):
            s = r[c]
            v = float(s) if s != DASH else 0.0
            if rules[c] == "absmin" and abs(v) < 1e-12:
                v = 0.0
            if c in best and s != DASH and v == best[c]:
                s = "**%s**" % s
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    tail = ["", BOLD_NOTE]
    if with_time:
        tail.append("")
        tail.append("Times: a laptop, running serial.")
    return "\n".join(lines + tail) + "\n"


def note_of(q):
    return "min scaled J %.2f, %d inverted, F/hex %.3f" % (q.sj_min, q.n_inverted, q.objective_mean)


# -- per-key specifications ------------------------------------------------------------------------------------------
def sector_planes(h_z, theta_off=0.3, z=None, dth=np.pi / 16):
    """z plane and the vertical plane through the axis at 45 degrees (offset off the 11.25-degree vertex layers)."""
    th = np.pi / 4 + theta_off * dth
    return [((0, 0, z), (0, 0, 1)), ((0, 0, 0), (-np.sin(th), np.cos(th), 0))]


def spec_methods(ns):
    zc = 0.25 + 0.3 * 0.125
    rows = ns["rows"]
    q = {k: rows[k]["q"] for k in rows}
    t = {k: rows[k]["seconds"] for k in rows}
    inp, lap, opt, sch = q["input"], q["laplace"], q["optimizer"], q["schedule"]
    variants = [("input", "input"), ("laplace", "Laplace"), ("optimizer", "optimizer"), ("schedule", "schedule")]
    headers = ["method", "inverted", "min SJ", "mean SJ", "worst 5 % SJ", "F/hex", "time (s)"]

    def row(k, label, ref=False):
        x = q[k]
        return [label, fint(x.n_inverted), f4(x.sj_min), f4(x.sj_mean), f4(x.sj_q05), f4(x.objective_mean),
                DASH if ref else ftime(t[k])]
    table = table_md(headers, row("input", "input", True), [row(k, l) for k, l in variants[1:]],
                     [None, "min", "max", "max", "max", "min", "min"], True)
    beats = sch.objective_mean < 0.999 * opt.objective_mean
    kp = [
        "**Laplace is the cheap one**: %.0fx faster than the optimizer here; it lifts the minimum scaled Jacobian from "
        "%.2f to %.2f but leaves F/hex at %.3f." % (t["optimizer"] / t["laplace"], inp.sj_min, lap.sj_min, lap.objective_mean),
        "**The optimizer reaches the lowest objective**: F/hex %.3f, minimum scaled Jacobian %.2f."
        % (opt.objective_mean, opt.sj_min),
        "**The suggested schedule %s the optimizer** (F/hex %.4f against %.4f) in %d stages: %s"
        % ("beats" if beats else "matches", sch.objective_mean, opt.objective_mean, rows["schedule"]["stages"],
           "see its `suggest_schedule` list." if beats else
           "its value is robustness on harder starts, not a better optimum."),
        "**The optimum is not all cubes**: on a curved sector the objective stays above zero (F/hex %.3f), which is "
        "why the methods can be told apart." % opt.objective_mean,
    ]
    return dict(
        title="Three methods on one perturbed curved block: cuts at mid-height and through the axis at 45 degrees",
        planes=sector_planes(0.125, z=zc), variants=variants, extra={},
        bars=([(l, q[k].sj_min) for k, l in variants], lambda v: "%.2f" % v, "Minimum scaled Jacobian (higher is better)",
              "Minimum scaled Jacobian of the input and of the three methods"),
        table=table, keypoints=kp, fields=[])


def spec_slide_box(ns):
    rows = ns["rows"]
    q = {k: rows[k]["q"] for k in rows}
    variants = [("input", "input"), ("frozen", "frozen"), ("slide", "slide")]
    headers = ["boundary", "inverted", "min SJ", "mean SJ", "worst 5 % SJ", "face off-plane", "edge off-line",
               "corner move"]

    def row(k, ref=False):
        x = q[k]
        d = [DASH] * 3 if ref else [fdist(rows[k]["off_plane"]), fdist(rows[k]["off_line"]), fdist(rows[k]["corner_move"])]
        return [k, fint(x.n_inverted), f4(x.sj_min), f4(x.sj_mean), f4(x.sj_q05)] + d
    table = table_md(headers, row("input", True), [row("frozen"), row("slide")],
                     [None, "min", "max", "max", "max", "absmin", "absmin", "absmin"], False)
    s = rows["slide"]
    notes = " ".join(s["notes"])
    patches = int(re.search(r"(\d+) patch", notes).group(1))
    curves = int(re.search(r"(\d+) edge curve", notes).group(1))
    kp = [
        "**Face vertices stay on their planes** (largest distance %s, round-off) **and edge vertices on their lines** "
        "(%s); the 8 corners do not move (%s)." % (fdist(s["off_plane"]), fdist(s["off_line"]), fdist(s["corner_move"])),
        "**Sliding lets the boundary spacing even out**: minimum scaled Jacobian %.2f against %.2f with the boundary "
        "frozen." % (s["q"].sj_min, rows["frozen"]["q"].sj_min),
        "**The constraints come from the mesh**: with no target passed, `resolve` found %d planar patches, %d edge lines "
        "and %d frozen corners." % (patches, curves, s["counts"]["frozen"]),
        "**A box is the easy case**: every hex can become a cube, so sliding reaches %s."
        % ("a perfect grid" if s["q"].sj_min > 0.9999 else "min scaled Jacobian %.4f" % s["q"].sj_min),
    ]
    return dict(
        title="A box with a sliding boundary: cuts at mid-height and mid-depth",
        planes=[((0, 0, 0.3 + 0.3 * 0.15), (0, 0, 1)), ((0, 0.5, 0), (0, 1, 0))], variants=variants, extra={},
        bars=([(l, q[k].sj_min) for k, l in variants], lambda v: "%.2f" % v, "Minimum scaled Jacobian (higher is better)",
              "Minimum scaled Jacobian of the input, the frozen run and the sliding run"),
        table=table, keypoints=kp, fields=[])


def spec_free_boundary(ns):
    rows, sag = ns["rows"], ns["sagitta"]
    q = {k: rows[k]["q"] for k in rows}
    variants = [("input", "input"), ("free", "free"), ("tangent", "tangent"), ("slide", "slide"),
                ("exact", "exact wall")]
    headers = ["variant", "volume change (%)", "off cylinder", "off input facets (wall)", "cap off-plane", "inverted",
               "min SJ", "mean SJ"]

    def row(k, label, ref=False):
        x, d = q[k], rows[k]
        c = [DASH] * 4 if ref else [fpct(d["volume_change_pct"]), fdist(d["wall_dist"]), fdist(d["facet"]),
                                    fdist(d["cap_off_plane"])]
        return [label] + c + [fint(x.n_inverted), f4(x.sj_min), f4(x.sj_mean)]
    table = table_md(headers, row("input", "input", True), [row(k, l) for k, l in variants[1:]],
                     [None, "absmin", "absmin", "absmin", "absmin", "min", "max", "max"], False)
    ch, tg, sl, ex = rows["free"], rows["tangent"], rows["slide"], rows["exact"]
    kp = [
        "**`free` shrinks a convex wall**: the volume falls by %.1f %% and wall vertices leave the cylinder by up to %.1e."
        % (abs(ch["volume_change_pct"]), ch["wall_dist"]),
        "**`tangent` keeps the volume** (%+.2f %%) but follows no target: wall vertices drift %.1e off the cylinder and "
        "rim vertices %.1e off the cap planes." % (tg["volume_change_pct"], tg["wall_dist"], tg["cap_off_plane"]),
        "**`slide` holds the boundary as meshed**: caps exactly planar, wall vertices on the input's faceted wall (%.0e) "
        "and rim vertices on a spline through the input rim (%.1e off its chords); %.1e off the true cylinder is the "
        "facet chord error (sagitta %.1e)." % (sl["facet"], sl["facet_rim"], sl["wall_dist"], sag),
        "**An exact `Surface` holds the true wall**: with a cylinder target the wall vertices sit %.0e off it."
        % ex["wall_dist"],
    ]
    return dict(
        title="A cylinder with a free wall: cuts at mid-height and near the axis",
        planes=[((0, 0, 0.5 + 0.3 * 0.25), (0, 0, 1)), ((0.02, 0, 0), (1, 0, 0))], variants=variants, extra={},
        bars=([(l, rows[k]["volume_change_pct"]) for k, l in variants[1:]], lambda v: "%+.2f %%" % v,
              "Volume change against the input, in % (closer to zero is better)",
              "Volume change of the free, tangent, slide and exact wall runs"),
        table=table, keypoints=kp, fields=[])


def spec_thin_layer(ns):
    rows = ns["rows"]
    q = {k: rows[k]["q"] for k in rows}
    variants = [("input", "input"), ("unprotected", "unprotected"), ("protected", "KeepLayer")]
    headers = ["variant", "first layer, mean change (%)", "max change (%)", "kept vertices, max move",
               "size ratio max", "inverted", "min SJ", "mean SJ"]

    def row(k, label, ref=False):
        x, d = q[k], rows[k]
        kept = fdist(d["keep_move"]) if k == "protected" else DASH      # nothing is kept in the unprotected run
        c = [DASH] * 3 if ref else [fpct(d["thick_mean_pct"], 1), fpct(d["thick_max_pct"], 1), kept]
        return [label] + c + ["%.2f" % d["shape"].size_ratio_max, fint(x.n_inverted), f4(x.sj_min), f4(x.sj_mean)]
    table = table_md(headers, row("input", "input", True), [row(k, l) for k, l in variants[1:]],
                     [None, "absmin", "absmin", "absmin", "min", "min", "max", "max"], False)
    un, pr, inp = rows["unprotected"], rows["protected"], rows["input"]
    move = "by exactly 0" if pr["keep_move"] == 0 else "by up to %.1e" % pr["keep_move"]
    kp = [
        "**The cube-seeking metric thickens a designed layer**: unprotected, the first layer grows by %.0f %% on average "
        "and up to %.0f %%." % (un["thick_mean_pct"], un["thick_max_pct"]),
        "**`KeepLayer(wall, 1)` keeps it exactly**: %.1f %% change; the %d kept vertices move %s."
        % (pr["thick_mean_pct"], pr["keep_n"], move),
        "**Protecting a layer moves the problem to its edge**: the largest size jump between face-neighbours is %.2f with "
        "`KeepLayer` against %.2f unprotected (input %.2f)."
        % (pr["shape"].size_ratio_max, un["shape"].size_ratio_max, inp["shape"].size_ratio_max),
    ]
    return dict(
        title="A thin designed layer, unprotected and protected: cuts through the middle of the box",
        planes=[((0, 0.5 + 0.3 / 6, 0), (0, 1, 0)), ((0.5 + 0.3 / 6, 0, 0), (1, 0, 0))], variants=variants,
        extra={"layer": ns["first_layer"].astype(float)},
        bars=([(l, rows[k]["thick_mean_pct"]) for k, l in variants[1:]], lambda v: "%+.1f %%" % v,
              "First-layer mean thickness change, in % (zero keeps the design)",
              "First-layer mean thickness change of the unprotected and KeepLayer runs"),
        table=table, keypoints=kp,
        fields=[{"name": "layer", "label": "First layer", "kind": "cat", "range": [0, 1],
                 "legend": "first layer | other"}])


def spec_untangle(ns):
    from nekmeshpy.hexmesh import smoother as hs

    rows = ns["rows"]
    q = {k: rows[k]["q"] for k in rows}
    variants = [("input", "input"), ("laplace", "Laplace"), ("optimize0", "optimize δ=0"),
                ("untangle", "untangle preset"), ("suggested", "suggested")]
    headers = ["method", "stages", "inverted", "min SJ", "mean SJ", "time (s)"]

    def row(k, label, ref=False):
        x, d = q[k], rows[k]
        return [label, DASH if ref else fint(d["stages"]), fint(x.n_inverted), f4(x.sj_min), f4(x.sj_mean),
                DASH if ref else ftime(d["seconds"])]
    table = table_md(headers, row("input", "input", True), [row(k, l) for k, l in variants[1:]],
                     [None, "min", "min", "max", "max", "min"], True)
    sched = hs.untangle_schedule()
    in_inv = q["input"].n_inverted
    lap, o0, un, sg = q["laplace"], q["optimize0"], q["untangle"], q["suggested"]
    kp = [
        "**The exact metric cannot untangle**: `optimize(delta=0)` has a wall at zero Jacobian and leaves %s hexes "
        "inverted." % ("all %d" % in_inv if o0.n_inverted == in_inv else "%d of %d" % (o0.n_inverted, in_inv)),
        "**`untangle_schedule` is a preset of `optimize`, not a separate untangler**: %d `optimize` stages with `delta` "
        "shrinking from %g to %g; it %s (minimum scaled Jacobian %.2f)."
        % (len(sched), sched[0][1]["delta"], sched[-1][1]["delta"],
           "clears all %d inverted hexes" % in_inv if un.n_inverted == 0 else
           "leaves %d of %d inverted" % (un.n_inverted, in_inv), un.sj_min),
    ]
    if lap.n_inverted == 0:
        kp.append("**On this input Laplace clears them too** (minimum %.2f, %.2f s): a few displaced vertices in an "
                  "otherwise clean block are what element averaging repairs best."
                  % (lap.sj_min, rows["laplace"]["seconds"]))
    else:
        kp.append("**Laplace leaves %d inverted**: its guard refuses any pass that adds an inverted hex." % lap.n_inverted)
    kp.append("**`suggest_schedule` adds the preset by itself** when the input has inverted hexes: %d stages, minimum "
              "scaled Jacobian %.2f." % (rows["suggested"]["stages"], sg.sj_min))
    P = ns["before"].points
    h = ns["inverted_in_input"]
    zs = P[ns["pushed"], 2]
    zplane = float(np.mean(zs)) + 0.05 * 0.125
    return dict(
        title="Folded hexes and the methods that repair them: cuts through the pushed vertices and the axis plane",
        planes=sector_planes(0.125, z=zplane), variants=variants, extra={"inv": h.astype(float)},
        bars=([(l, float(q[k].n_inverted)) for k, l in variants], lambda v: "%d" % v,
              "Inverted hexes (fewer is better)", "Inverted hexes of the input and of each method"),
        table=table, keypoints=kp,
        fields=[{"name": "inv", "label": "Inverted in the input", "kind": "cat", "range": [0, 1],
                 "legend": "inverted in the input | other"}])


KEYS = {
    "methods": ("01_methods.py", spec_methods),
    "slide_box": ("02_slide_box.py", spec_slide_box),
    "free_boundary": ("03_free_boundary.py", spec_free_boundary),
    "thin_layer": ("04_thin_layer.py", spec_thin_layer),
    "untangle": ("05_untangle.py", spec_untangle),
}


def viewer_inc(key):
    return ('```{raw} html\n<script src="../_static/smoother_hex_viewer.js"></script>\n<div id="hx-%s"></div>\n'
            '<script>window.initSmootherHex("hx-%s", "../_static/hexsmoother/%s_manifest.json");</script>\n```\n'
            % (key, key, key))


def build(key):
    script, spec_fn = KEYS[key]
    ns = run_script(os.path.join(_EX, script))
    sp = spec_fn(ns)
    rows = ns["rows"]
    sizes = {}
    variants = []
    for vk, label in sp["variants"]:
        m, sj = ns["meshes"][vk], ns["sj"][vk]
        shell, section = "%s_%s_shell.vtp" % (key, vk), "%s_%s_section.vtp" % (key, vk)
        sizes[shell] = shell_vtp(m, os.path.join(STATIC, shell))
        sizes[section] = section_vtp(m, sj, sp["extra"], sp["planes"], os.path.join(STATIC, section))
        variants.append({
            "label": label, "note": note_of(rows[vk]["q"]),
            "layers": [{"name": "wall", "role": "surface", "label": "Boundary shell", "vtp": shell},
                       {"name": "section", "role": "section", "label": "Hex cross-section", "vtp": section}],
            "fields": [{"name": "sj", "label": "Scaled Jacobian (worst corner)", "kind": "sj", "range": [0, 1]}]
            + sp["fields"],
        })
    big = {k: v for k, v in sizes.items() if v > MAX_VTP_BYTES}
    assert not big, big
    manifest = {"title": sp["title"], "camera": CAMERA, "variants": variants}
    with open(os.path.join(STATIC, key + "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    items, fmt, metric, aria = sp["bars"]
    inc = {
        "viewer": viewer_inc(key),
        "svg": bars_svg(items, fmt, metric, aria),
        "table": sp["table"],
        "keypoints": "".join("- %s\n" % k for k in sp["keypoints"]),
    }
    for name, text in inc.items():
        with open(os.path.join(GENERATED, "%s.%s.inc" % (key, name)), "w") as f:
            f.write(text)
    total = sum(sizes.values()) + os.path.getsize(os.path.join(STATIC, key + "_manifest.json")) + sum(map(len, inc.values()))
    return sizes, total


def main(argv):
    keys = argv or list(KEYS)
    unknown = [k for k in keys if k not in KEYS]
    if unknown:
        raise SystemExit("unknown key(s) %s; choose from %s" % (unknown, list(KEYS)))
    os.makedirs(STATIC, exist_ok=True)
    os.makedirs(GENERATED, exist_ok=True)
    grand = 0
    for k in keys:
        sizes, total = build(k)
        grand += total
        print("%-14s %2d vtp files, largest %6d B, %7d B written" % (k, len(sizes), max(sizes.values()), total))
    assert grand < MAX_TOTAL_BYTES * len(keys), grand
    print("total written: %d bytes" % grand)


if __name__ == "__main__":
    main(sys.argv[1:])
