"""Generate the assets of the hex views on the smoothing pages (``docs/smoother/``).

Not part of the ``nekmeshpy`` package -- doc-build tooling like ``gen_smoother_assets.py``; run it by hand after the
smoother or one of the scripts below changes, and commit what it writes (about two minutes):

    python docs/_ext/gen_smoother_hex_assets.py [carotid | u_tube | wire_coil ...]

Three pages carry a hex view, each fed by one script of ``examples/smoother/`` that is executed with ``runpy`` (cwd in a
scratch directory, stdout captured) and whose public namespace is read:

* ``carotid``    ``cs_carotid_hex.py`` -- the carotid recipe re-run on the smoothed wall (variants ``original`` and
  ``smoothed (held)``, the same four layers);
* ``u_tube``     ``cs_u_tube_hex.py`` -- the as-built U-tube hexes, and the smoothed interface carried into them;
* ``wire_coil``  ``cs_wire_coil_hex.py`` -- the same for the wire surface.

For every mesh shown it writes, under ``docs/_static/smoother/``,

* ``<prefix>_<variant>_exterior.vtp``  the gallery's exterior view: every named face of the mesh as quads (order-1 corners
  only, none dropped) with a per-quad ``bc_id``; the group names travel in the manifest;
* ``<prefix>_<variant>_wall.vtp``      the wall / interface surface (translucent in the cross-section view);
* ``<prefix>_<variant>_section.vtp``   the hex cross-section: polygons where two planes cut the hexes, with ``sj`` (scaled
  Jacobian, order-``SAMPLE_ORDER`` lattice) and a per-case second field (``bl`` or ``region``);
* ``<prefix>_manifest.json``           what ``smoother_hex_viewer.js`` reads;

and under ``docs/smoother/_generated/`` the viewer block and the tables and prose numbers of each page, every figure
read from the run.  A carried mesh that is not valid (inverted hexes, other boundary names) is **not** shown: the page
states the numbers instead.  The binary ``.vtp`` files are small (Float32 / Int32, UInt64 block headers), the section ones too.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOCS)
GENERATED = os.path.join(_DOCS, "smoother", "_generated")
STATIC = os.path.join(_DOCS, "_static", "smoother")
_EX = os.path.join(_REPO, "examples", "smoother")
MAX_VTP_BYTES = 400_000
MAX_TOTAL_BYTES = 4_000_000
sys.path.insert(0, _REPO)


def b64(a):
    """One binary ``DataArray`` payload as ``boundary_to_vtp`` writes it: the byte count (UInt64) then the bytes, base64'd as
    a single blob (separately encoded blocks leave padding mid-string, which vtk.js does not read)."""
    raw = np.ascontiguousarray(a).tobytes()
    return base64.b64encode(np.array([len(raw)], dtype="<u8").tobytes() + raw).decode()


def write_vtp(path, points, conn, offsets, cell_data):
    """Binary PolyData: ``points`` (n,3), ``conn`` the flat vertex ids, ``offsets`` the end of each polygon,
    ``cell_data`` {name: (values, "Float32" | "Int32")}."""
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


def exterior_vtp(hexmesh, quadmesh, m, path):
    """Every named face of ``m`` as quads on its order-1 corners, ``bc_id`` per quad (ids over the sorted names, as the
    gallery's ``boundary_to_vtp``); returns ``(bytes, {id: name})``."""
    surf = quadmesh.select(m.quad_mesh, m.face_tags.ids)
    names = np.asarray(surf.element_tags.to_dense())
    uniq = sorted(set(names.tolist()) - {""})
    ids = np.array([uniq.index(n) + 1 if n else 0 for n in names.tolist()])
    q = np.asarray(surf.corners)
    size = write_vtp(path, surf.points, q.ravel(), np.arange(4, 4 * len(q) + 1, 4), {"bc_id": (ids, "Int32")})
    return size, {str(i + 1): n for i, n in enumerate(uniq)}


def cut(m, sj, extra, origin, normal, edges):
    """Polygons where a plane cuts the hexes of ``m``: ``(polygons, sj, {name: values})`` per cut hex.  ``extra`` is
    {name: per-hex values}; ``edges`` the 12 hex edges as corner pairs."""
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


def section_vtp(m, sj, extra, planes, edges, path):
    """The cuts of every ``(origin, normal)`` in ``planes`` as one PolyData with ``sj``, ``plane`` and the ``extra`` fields."""
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


def wall_vtp(quadmesh, hexmesh, m, tag, path):
    """One quad per physical face of the tagged wall (an interior interface comes back once per side)."""
    w = hexmesh.boundary_mesh(m, tag)
    c = np.asarray(w.corners)
    _, first = np.unique(np.sort(c, axis=1), axis=0, return_index=True)
    c = c[np.sort(first)]
    return write_vtp(path, w.points, c.ravel(), np.arange(4, 4 * len(c) + 1, 4), {})


def run_script(path):
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)
        try:
            return runpy.run_path(path, run_name="gen_hex")
        finally:
            os.chdir(cwd)


def fmt(x, n=4):
    return ("%." + str(n) + "f") % x


def verdict(a, b):
    if abs(b - a) < 5e-5:
        return "unchanged"
    return "rose" if b > a else "fell"


def viewer_inc(div, manifest):
    return (
        '```{raw} html\n<script src="../_static/smoother_hex_viewer.js"></script>\n<div id="%s"></div>\n'
        '<script>window.initSmootherHex("%s", "../_static/smoother/%s");</script>\n```\n' % (div, div, manifest)
    )


def write_variant(prefix, key, label, note, ns, m, sj, extra, extra_field, planes, tag, hidden, static):
    """Write the three .vtp of one mesh; return ``(manifest variant, {file: bytes})``."""
    from nekmeshpy import hexmesh, quadmesh
    from nekmeshpy.core import conform

    edges = conform._LOCAL_EDGES[3]
    files = {"%s_%s_exterior.vtp" % (prefix, key): None, "%s_%s_wall.vtp" % (prefix, key): None,
             "%s_%s_section.vtp" % (prefix, key): None}
    epath, wpath, spath = files
    sizes = {}
    sizes[epath], groups = exterior_vtp(hexmesh, quadmesh, m, os.path.join(static, epath))
    sizes[wpath] = wall_vtp(quadmesh, hexmesh, m, tag, os.path.join(static, wpath))
    sizes[spath] = section_vtp(m, sj, extra, planes, edges, os.path.join(static, spath))
    variant = {
        "label": label,
        "note": note,
        "layers": [
            {"name": "exterior", "role": "groups", "vtp": epath, "groups": groups, "hidden": hidden},
            {"name": "wall", "role": "surface", "label": "Wall surface", "vtp": wpath},
            {"name": "section", "role": "section", "label": "Hex cross-section", "vtp": spath},
        ],
        "fields": [{"name": "sj", "label": "Scaled Jacobian", "kind": "sj", "range": [0, 1]}, extra_field],
    }
    return variant, sizes


def finish(prefix, manifest, sizes, static):
    big = {k: v for k, v in sizes.items() if v > MAX_VTP_BYTES}
    assert not big and sum(sizes.values()) < MAX_TOTAL_BYTES, (big, sum(sizes.values()))
    with open(os.path.join(static, prefix + "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


# -- carotid -------------------------------------------------------------------------------------------------------
CAROTID = [
    ("original", "original", "original",
     "The hexes of the carotid recipe, order-1 corners (what a .re2 file holds)."),
    ("rebuilt", "smoothed wall, rebuilt (same 4 layers)", "smoothed (held)",
     "The carotid recipe re-run on the smoothed wall: the same four layers, the shared seam ring held fixed."),
]


def build_carotid(generated, static):
    ns = run_script(os.path.join(_EX, "cs_carotid_hex.py"))
    variants, layers = ns["variants"], ns["layers"]
    planes = list(ns["planes_for"](ns["rings_orig"]).values())
    (o1, n1), (o2, n2) = planes
    prefix = "carotid_hex"
    mv, sizes = [], {}
    for key, label, vkey, note in CAROTID:
        m = variants[vkey][0]
        bl = np.zeros(m.n_hexes)
        bl[layers[vkey]["wall_hex"]] = 1.0
        field = {"name": "bl", "label": "Wall layer (hexes touching the wall)", "kind": "cat", "range": [0, 1],
                 "legend": "wall layer | other"}
        v, sz = write_variant(prefix, key, label, note, ns, m, ns["fine_sj"](m), {"bl": bl}, field, planes, "wall", [], static)
        mv.append(v)
        sizes.update(sz)
    n2 = n2 / np.linalg.norm(n2)
    direction = n2 + 0.55 * n1 / np.linalg.norm(n1)
    manifest = {
        "title": "The carotid hexes: exterior by boundary group, and cuts through a leg and through the trifurcation",
        "camera": {"direction": [round(float(x), 4) for x in direction], "viewUp": [0, 0, 1]},
        "variants": mv,
    }
    finish(prefix, manifest, sizes, static)
    carotid_includes(ns, generated)
    cost_includes(ns, generated)
    return sizes


def carotid_includes(ns, generated):
    layers, readings, so = ns["layers"], ns["readings"], ns["SAMPLE_ORDER"]
    names = [
        ("original", "original (carotid recipe)"),
        ("smoothed (free)", "smoothed wall, seam ring free"),
        ("smoothed (held)", "**rebuilt on the smoothed wall** (seam ring held)"),
    ]
    rows = [
        "| mesh | hexes | min scaled J | mean | worst 5 %% | inverted | order-%d sampling: min / inverted |" % so,
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, label in names:
        r = readings[key]
        c, f = r["corner"], r["fine"]
        rows.append("| %s | %d | %s | %s | %s | %d | %s / %d |"
                    % (label, r["n"], fmt(c["min"]), fmt(c["mean"]), fmt(c["q05"]), c["n_inv"], fmt(f["min"]), f["n_inv"]))
    same = all(abs(readings[k]["corner"][s] - readings[k]["fine"][s]) < 5e-4 for k, _ in names for s in ("min", "mean", "q05"))
    quality = (
        "\n".join(rows) + "\n\n"
        "Scaled Jacobian of the order-1 hexes, read at their 8 corners (the straight-sided hex a `.re2` file holds). The last column "
        "re-reads the same trilinear map on an %d^3 lattice, the solver's order: %s.\n"
        % (so + 1, "it agrees with the corner reading to 3 decimals in every row" if same
           else "it differs from the corner reading in some row, see the table")
    )
    rad = ns["RADIAL"]
    rel0, rel1 = layers["original"]["rel"], layers["smoothed (held)"]["rel"]
    lrows = [
        "| mesh | layers | first layer, fraction of the core-to-wall gap | first layer / local radius: min | median | max |",
        "| --- | --- | --- | --- | --- |  --- |",
    ]
    for label, rel in (("original wall", rel0), ("smoothed wall, rebuilt", rel1)):
        lrows.append("| %s | %d | %s | %s | %s | %s |"
                     % (label, len(rad) - 1, fmt(rad[-1] - rad[-2], 3), fmt(rel.min()), fmt(np.median(rel)), fmt(rel.max())))
    layers_inc = (
        "`RADIAL` (constants at the top of `examples/smoother/cs_carotid_hex.py`) is the original's: `%s`, %d layers between the "
        "core square and the wall, the wall layer %s of the gap. The rebuild keeps exactly this layering.\n\n"
        % (np.round(rad, 3).tolist(), len(rad) - 1, fmt(rad[-1] - rad[-2], 2))
        + "\n".join(lrows) + "\n\n"
        "The first layer is measured as each wall hex's edge into the volume, against the mean radius of its station's wall ring "
        "(%d wall hexes in both). Its thickness over the local radius is essentially unchanged, so the smoothed wall leaves the "
        "original boundary layer intact: the layering is a property of the construction, not of the surface.\n"
        % layers["original"]["n"]
    )
    a, b = readings["original"]["corner"], readings["smoothed (held)"]["corner"]
    fa, fb = readings["original"]["fine"], readings["smoothed (held)"]["fine"]
    fr = readings["smoothed (free)"]["corner"]
    w0, w1 = ns["wall_of"]["original"], ns["wall_of"]["smoothed (held)"]
    vol = 100 * (readings["smoothed (held)"]["volume"] / readings["original"]["volume"] - 1)
    text = (
        "With the same layers, the rebuild on the smoothed wall moved the hexes' worst corner reading from %s to %s, the mean "
        "from %s to %s and the worst 5 %% from %s to %s; %d hexes are inverted at the corners before and %d after (order-%d "
        "sampling: %d and %d). The wall faces themselves improve more than the hexes do: wall scaled J (against the "
        "tessellation) min %s → %s, worst 5 %% %s → %s, largest neighbour size ratio %s → %s, overlapping wall quads %d → %d. "
        "The enclosed volume changes by %s %%.\n\n"
        "Smoothing leaves the first layer as it was: first layer over local radius, median %s → %s, min %s → %s, max %s → %s.\n\n"
        "With the shared seam ring left free, the same rebuild inverts %d hexes (min %s).\n\n"
        "The worst hex %s (min %s → %s) while the mean %s and the worst 5 %% %s: read the minimum first.\n"
        % (fmt(a["min"]), fmt(b["min"]), fmt(a["mean"]), fmt(b["mean"]), fmt(a["q05"]), fmt(b["q05"]), a["n_inv"], b["n_inv"], so,
           fa["n_inv"], fb["n_inv"], fmt(w0["q"].sj_min, 3), fmt(w1["q"].sj_min, 3), fmt(w0["q"].sj_q05, 3),
           fmt(w1["q"].sj_q05, 3), fmt(w0["size_max"], 2), fmt(w1["size_max"], 2), w0["ov"], w1["ov"], "%+.3f" % vol,
           fmt(np.median(rel0)), fmt(np.median(rel1)), fmt(rel0.min()), fmt(rel1.min()), fmt(rel0.max()), fmt(rel1.max()),
           fr["n_inv"], fmt(fr["min"], 3), verdict(a["min"], b["min"]) + " in quality", fmt(a["min"]), fmt(b["min"]),
           verdict(a["mean"], b["mean"]), verdict(a["q05"], b["q05"]))
    )
    for part, body in (("quality", quality), ("layers", layers_inc), ("text", text)):
        with open(os.path.join(generated, "carotid_hex.%s.inc" % part), "w") as f:
            f.write(body)
    with open(os.path.join(generated, "carotid_hex.viewer.inc"), "w") as f:
        f.write(viewer_inc("sm-carotid-hex", "carotid_hex_manifest.json"))


def cost_includes(ns, generated):
    """The carotid page's Cost subsection, read from the ONE run of ``cs_carotid_hex.py`` above (its ``TIMING_ROWS``): the
    table, a horizontal bar per stage (HTML/CSS, theme variables, no image) and a paragraph that reads them with no typed number."""
    rows, total, extras = ns["TIMING_ROWS"], ns["TIMING_TOTAL"], ns["TIMING_EXTRAS"]
    secs = dict(rows)
    table = ["| stage | seconds | share of the total |", "| --- | --- | --- |"]
    for name, sec in rows:
        table.append("| %s | %.2f | %.1f %% |" % (name.replace("|", "/"), sec, 100 * sec / total))
    table.append("| **total of the stages above** | **%.2f** | **100 %%** |" % total)
    table.append("")
    table.append("Smoothing (7) vs building the original hexes (4) and rebuilding them (8): **%.2f s** vs **%.2f s** and **%.2f s**." % (
        ns["T_SMOOTH"], secs[ns["S4"]], secs[ns["S8"]]))
    table.append("")
    bars = ['```{raw} html', '<div class="sm-cost-bars" style="margin:0.5rem 0 1rem">']
    top = max(sec for _, sec in rows)
    for name, sec in rows:
        w = max(0.4, 100 * sec / top)
        bars.append(
            '<div style="display:flex;align-items:center;gap:0.6rem;font-size:0.8em;margin:2px 0">'
            '<span style="flex:0 0 19rem;color:var(--color-foreground-secondary)">%s</span>'
            '<span style="flex:1 1 auto;background:var(--color-background-secondary);border-radius:2px">'
            '<span style="display:block;height:0.9em;width:%.1f%%;background:var(--color-brand-primary);border-radius:2px"></span></span>'
            '<span style="flex:0 0 3.2rem;text-align:right;color:var(--color-foreground-primary)">%.1f %%</span></div>'
            % (name, w, 100 * sec / total))
    bars += ["</div>", "```", ""]
    big = max(rows, key=lambda r: r[1])
    t_smooth, t_build, t_rebuild = ns["T_SMOOTH"], secs[ns["S4"]], secs[ns["S8"]]
    if t_smooth < t_build and t_smooth < t_rebuild:
        verdict_ = "cheaper than either building the hexes or rebuilding them"
    elif t_smooth > t_build and t_smooth > t_rebuild:
        verdict_ = "more expensive than either building the hexes or rebuilding them"
    else:
        verdict_ = "between the cost of building the hexes and of rebuilding them"
    text = (
        "The stage that takes the most time is **%s** (%.0f %% of the stages listed). Smoothing the wall (7) takes %.0f %% of "
        "them, which is %s: it costs %.1f times the original build (4) and %.1f times the rebuild (8). One run, serial, a rough "
        "estimate; the same script on another machine, or again on this one, gives other seconds and other shares. The work "
        "outside the table (other smoothing variants, the perturbed input and their readings, the figure: %.1f s here) is what "
        "makes the script's own total longer than the table's.\n"
        % (big[0][2:].strip(), 100 * big[1] / total, 100 * t_smooth / total, verdict_, t_smooth / t_build, t_smooth / t_rebuild,
           extras)
    )
    machine = ("Measured on a laptop, running serial (one process, no parallelism); numbers vary by machine and run.\n")
    for part, body in (("table", "\n".join(table)), ("bar", "\n".join(bars)), ("text", text), ("machine", machine)):
        with open(os.path.join(generated, "carotid_cost.%s.inc" % part), "w") as f:
            f.write(body)


# -- U-tube and wire coil: as built, and the smoothed surface carried in ----------------------------------------------
CASES = {
    "u_tube": dict(
        script="cs_u_tube_hex.py", prefix="u_tube_hex", tag="interface", what="interface", hidden=["outer"],
        carried="smoothed wall carried in", layers="the solid shell and the fluid shell, one hex layer each",
        title="The U-tube hexes: exterior by boundary group, and cuts along the U and across both legs",
        camera=[0.8, -0.6, 0.9],
        built_note="The hexes of examples/u_tube_box.py at order 1 (the corners a .re2 file holds): a solid box around the U, "
                   "a one-layer solid shell, a one-layer fluid shell and the fluid core. The box faces are switched off at first.",
    ),
    "wire_coil": dict(
        script="cs_wire_coil_hex.py", prefix="wire_coil_hex", tag="coil", what="wire surface", hidden=["outer", "wall"],
        carried="smoothed surface carried in", layers="one hex layer on each side of the surface",
        title="The wire-coil hexes: exterior by boundary group, and cuts across and along the pipe",
        camera=[0.4, -1.0, 0.5],
        built_note="The hexes of examples/wire_coil.py at order 1 (the corners a .re2 file holds): fluid around a helical wire. "
                   "The pipe's outer surface and its wall are switched off at first, so the wire's own surface shows.",
    ),
}


def build_case(name, generated, static):
    c = CASES[name]
    ns = run_script(os.path.join(_EX, c["script"]))
    variants, readings, sj_fine, valid = ns["variants"], ns["readings"], ns["sj_fine"], ns["valid"]
    region = (np.asarray(ns["built"].element_tags.to_dense()) == ns["FLUID"]).astype(float)
    field = {"name": "region", "label": "Region", "kind": "cat", "range": [0, 1], "legend": "fluid | solid"}
    planes = [(o, n) for _, o, n in ns["planes"]]
    keys = [("as_built", "as built", "as built", c["built_note"])]
    if valid:
        keys.append(("carried", "smoothed %s carried in" % c["what"], c["carried"],
                     "The smoothed %s put in place of the old one, nothing else moved." % c["what"]))
    mv, sizes = [], {}
    for key, label, vkey, note in keys:
        v, sz = write_variant(c["prefix"], key, label, note, ns, variants[vkey], sj_fine[vkey], {"region": region}, field,
                              planes, c["tag"], c["hidden"], static)
        mv.append(v)
        sizes.update(sz)
    d = np.asarray(c["camera"], float)
    manifest = {"title": c["title"], "camera": {"direction": d.tolist(), "viewUp": [0, 0, 1]}, "variants": mv}
    finish(c["prefix"], manifest, sizes, static)
    case_includes(c, ns, readings, valid, generated)
    return sizes


def case_includes(c, ns, readings, valid, generated):
    so, w, prefix = ns["SAMPLE_ORDER"], ns["where"], c["prefix"]
    rows = ["| mesh | hexes | min scaled J | mean | worst 5 %% | inverted | order-%d sampling: min / inverted | conforming, watertight |" % so,
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in readings.items():
        cr, f = r["corner"], r["fine"]
        rows.append("| %s | %d | %s | %s | %s | %d | %s / %d | %s |"
                    % (k, r["n"], fmt(cr["min"]), fmt(cr["mean"]), fmt(cr["q05"]), cr["n_inv"], fmt(f["min"]), f["n_inv"],
                       "yes" if r["conforming"] else "NO"))
    a, b = readings["as built"], readings[c["carried"]]
    n_wall = ns["unsmoothed"].n_points
    table = ("\n".join(rows) + "\n\nScaled Jacobian of the order-1 hexes read at their 8 corners (what a `.re2` file holds), and the same "
             "trilinear map re-read on an %d^3 lattice (order %d, the solver's)." % (so + 1, so))
    text = (
        "Carrying the smoothed %s into the hexes replaces its %d vertices by the smoothed ones (it moved %.2f local edges on average, "
        "largest %.2f) and leaves every other vertex where the example put it. At the corners that makes %d of %d hexes inverted "
        "(min scaled J %s → %s, mean %s → %s) and, re-read at order %d, %d inverted (min %s → %s). %d of the %d inverted hexes "
        "are in the solid region and %d in the fluid region; %d of them touch the %s. The %d hexes that touch the %s read min "
        "%s → %s at the corners (%s); the remaining %d hexes are unchanged (min %s → %s)."
        % (c["what"], n_wall, w["mean_move"], w["max_move"], b["corner"]["n_inv"], b["n"], fmt(a["corner"]["min"]),
           fmt(b["corner"]["min"]), fmt(a["corner"]["mean"]), fmt(b["corner"]["mean"]), so, b["fine"]["n_inv"],
           fmt(a["fine"]["min"]), fmt(b["fine"]["min"]), w["bad_solid"], w["n_bad"], w["bad_fluid"], w["bad_touch"], c["what"],
           w["n_touch"], c["what"], fmt(w["touch_min_built"]), fmt(w["touch_min_carried"]), c["layers"], a["n"] - w["n_touch"],
           fmt(w["off_min_built"]), fmt(w["off_min_carried"]))
    )
    if valid:
        text += ("\n\nThis carried mesh has no inverted hex at the corners or at order %d, conforms, and has the same boundary names; "
                 "the view offers it next to the mesh as built." % so)
    else:
        text += ("\n\n**The smoothed %s cannot be carried into the volume for this case.** The carried mesh is not valid, so the "
                 "view below shows only the mesh as built (all toggles stay). %s"
                 % (c["what"], "Every inverted hex touches the %s: it is a layer between the moved surface and the vertices beyond "
                    "it, which did not move." % c["what"] if w["bad_touch"] == w["n_bad"]
                    else "Not every inverted hex touches the %s, see the counts above." % c["what"]))
    for part, body in (("table", table), ("text", text + "\n")):
        with open(os.path.join(generated, "%s.%s.inc" % (prefix, part)), "w") as f:
            f.write(body + ("\n" if part == "table" else ""))
    with open(os.path.join(generated, prefix + ".viewer.inc"), "w") as f:
        f.write(viewer_inc("sm-" + prefix.replace("_", "-"), prefix + "_manifest.json"))


def build(which=None, generated=GENERATED, static=STATIC):
    os.makedirs(generated, exist_ok=True)
    os.makedirs(static, exist_ok=True)
    sizes = {}
    for name in which or ["carotid", "u_tube", "wire_coil"]:
        print("building %s ..." % name, flush=True)
        sizes.update(build_carotid(generated, static) if name == "carotid" else build_case(name, generated, static))
    return sizes


if __name__ == "__main__":
    out = build(sys.argv[1:] or None)
    print("wrote %s; total %d kB" % (", ".join("%s %d kB" % (k, v // 1000) for k, v in sorted(out.items())), sum(out.values()) // 1000))
