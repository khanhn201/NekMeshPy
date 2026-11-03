"""Generate the assets of the smoothing gallery pages (``docs/smoother/quad/``).

Not part of the ``nekmeshpy`` package -- doc-build tooling like ``gen_viewer_assets.py``; run it by hand after
the smoother or an example changes, and commit what it writes:

    python docs/_ext/gen_smoother_quad_assets.py [stem ...]

The scripts of ``examples/smoother/quad/`` that feed the two pages are executed the way ``tests/test_examples.py`` runs
the examples (``runpy.run_path``, cwd in a scratch directory), with stdout captured.  From the runs it writes

* ``docs/_static/smoother/quad/<name>_frames.json``  the playback frames the page viewer and its two charts read: the
  planar scripts' own JSON is copied, the 3-D ones are recorded here through the public API
  (``smooth.run_schedule(record=True)`` + ``frames.write_frames``) on the very inputs the script used;
* ``docs/smoother/_generated/<page>.viewer.inc``  the raw-HTML block that mounts the viewer on the page, with the
  list of geometries and their frames files (``SIMPLE_GEOMETRIES`` / ``CAROTID_WALL`` below are its one source);
* ``docs/smoother/_generated/<page>.table.inc``  the headline numbers as a Markdown table: read from the frames
  of the page's geometries (simple geometries) or from the script's own ``Quality`` readings (carotid wall);

The pages in ``docs/smoother/quad/*.md`` are static Markdown that ``{include}`` these fragments, so a docs build only
reads files and no number on a page is typed by hand.  The fragments use the ``.inc`` suffix so Sphinx does not
take them for documents.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
from typing import Any, NamedTuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_DOCS)
_EXAMPLES = os.path.join(_REPO, "examples", "smoother", "quad")
GENERATED = os.path.join(_DOCS, "smoother", "quad", "_generated")
STATIC = os.path.join(_DOCS, "_static", "smoother", "quad")

sys.path.insert(0, _REPO)

MAX_JSON_BYTES = 150_000          # one frames file; the page assets stay small
BOX_JSON_BYTES = 340_000          # the U-tube's outer box: 5000 quads, ~120 kB of connectivity and ~130 kB per frame, so two frames


def json_budget(name: str) -> int:
    """The size budget of one frames file (the outer box, with five times the quads, gets its own)."""
    return BOX_JSON_BYTES if "_box_" in name else MAX_JSON_BYTES


class Geometry(NamedTuple):
    """One selector entry of a page: the frames file, how to draw it, the script that built it, a one-line note."""

    key: str
    label: str
    frames: str
    kind: str          # "planar" (canvas) or "surface" (vtk.js)
    script: str
    note: str


SIMPLE_GEOMETRIES = [
    Geometry("annulus", "annulus", "01_basics_frames.json", "planar", "01_basics.py",
             "A jittered annular patch, smoothed with the user schedule (Laplace and optimizer stages alternating)."),
    Geometry("disk", "disk", "03_a_disk_frames.json", "planar", "03_constraints.py",
             "A disk with its input folded; the wall slides along the circle."),
    Geometry("pipe", "pipe section", "cs_pipe_stress_frames.json", "planar", "cs_pipe_section.py",
             "The O-grid section of a circular pipe, vertices perturbed; the wall slides on its circle, the "
             "layers next to it are kept."),
    Geometry("box", "box", "03_b_box_frames.json", "surface", "03_constraints.py",
             "The surface of a box: the 8 corners stay, the creases slide on their lines, faces stay on their planes."),
    Geometry("sphere", "sphere", "04_sphere_frames.json", "surface", "04_curved_surfaces.py",
             "A jittered cubed sphere; every vertex is pulled back onto the exact sphere."),
    Geometry("ellipsoid", "ellipsoid", "04_ellipsoid_frames.json", "surface", "04_curved_surfaces.py",
             "A jittered ellipsoid; the elements do not shrink where the surface curves more."),
    Geometry("torus", "torus", "04_torus_frames.json", "surface", "04_curved_surfaces.py",
             "A jittered torus (a donut) with a few inverted quads in the input."),
    Geometry("hemisphere", "hemisphere", "cs_hemisphere_stress_frames.json", "surface", "cs_hemisphere.py",
             "The hemisphere surface of the flow-past-hemisphere mesh, perturbed; the rim is frozen."),
]
CAROTID_WALL = [
    Geometry("asis", "as built", "cs_wall_asis_frames.json", "surface", "cs_wall_surface.py",
             "The wall of the carotid hex mesh as the example builds it; the hexes behind it are not updated."),
    Geometry("stress", "perturbed input", "cs_wall_stress_frames.json", "surface", "cs_wall_surface.py",
             "The same wall with its vertices perturbed; the hexes behind it are not updated."),
]
PAGES = {"simple_geometries": SIMPLE_GEOMETRIES, "carotid_wall": CAROTID_WALL}

# Optional pages: each is one script, one page and its frames, and is dropped from the build when its script or its
# page file is deleted (then the generated fragments and frames files of that page go with it).
WIRE_COIL = [
    Geometry("asis", "as built", "cs_coil_asis_frames.json", "surface", "cs_wire_coil.py",
             "The helical wire surface of the wire-coil hex mesh as the example builds it; the hexes are not updated."),
    Geometry("stress", "perturbed input", "cs_coil_stress_frames.json", "surface", "cs_wire_coil.py",
             "The same surface with its vertices perturbed; the hexes are not updated."),
]
U_TUBE = [
    Geometry("asis", "interface", "cs_utube_asis_frames.json", "surface", "cs_u_tube.py",
             "The U-tube wall of the hex mesh as the example builds it; the hexes are not updated."),
    Geometry("stress", "interface, perturbed input", "cs_utube_stress_frames.json", "surface", "cs_u_tube.py",
             "The same wall with its vertices perturbed; the hexes are not updated."),
    Geometry("box_asis", "outer box", "cs_utube_box_asis_frames.json", "surface", "cs_u_tube.py",
             "The six outer faces of the box as the example builds them: planar, the creases and corners held on "
             "their lines; the hexes are not updated."),
    Geometry("box_stress", "outer box, perturbed input", "cs_utube_box_stress_frames.json", "surface", "cs_u_tube.py",
             "The same box with its vertices perturbed (put back on its planes and lines); the hexes are not updated."),
]
OPTIONAL = {  # page -> (geometries, script stem, frames prefix, first printed line (unused), run labels)
    "wire_coil_surface": (WIRE_COIL, "cs_wire_coil", "cs_coil", "wire-coil wire surface", "the wire surface"),
    "u_tube_interface": (U_TUBE, "cs_u_tube", "cs_utube", "U-tube wall (tag", "the U-tube wall"),
}
for _page, (_geoms, _stem, *_rest) in OPTIONAL.items():
    if os.path.isfile(os.path.join(_EXAMPLES, _stem + ".py")) and os.path.isfile(
            os.path.join(_DOCS, "smoother", "quad", _page + ".md")):
        PAGES[_page] = _geoms

STAGES = ["01_basics", "03_constraints", "04_curved_surfaces", "cs_pipe_section", "cs_hemisphere", "cs_wall_surface"]
STAGES += [v[1] for p, v in OPTIONAL.items() if p in PAGES]
# planar frames each script writes itself (copied under docs/_static/smoother/quad/)
PLANAR_FRAMES = {
    "01_basics": ["01_basics_frames.json"],
    "03_constraints": ["03_a_disk_frames.json"],
    "cs_pipe_section": ["cs_pipe_stress_frames.json"],
}


# -- small formatting helpers --------------------------------------------------------------------------------
def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def arrow(a: float | int, b: float | int, fmt: str = "%.4f") -> str:
    return (fmt + " → " + fmt) % (a, b)


def runs_table(runs: dict[str, dict[str, Any]]) -> str:
    """One row per run: ``before → after`` of the headline numbers, then one verdict sentence per run."""
    rows = []
    for label, r in runs.items():
        (q0, q1), (o0, o1) = r["q"], r["ov"]
        rows.append([label, str(q0.n_quads), arrow(q0.n_inverted, q1.n_inverted, "%d"), arrow(o0, o1, "%d"),
                     arrow(q0.sj_min, q1.sj_min), arrow(q0.sj_mean, q1.sj_mean), arrow(q0.sj_q05, q1.sj_q05)])
    table = md_table(["run", "quads", "inverted", "overlapping", "min scaled J", "mean scaled J",
                      "worst 5 % scaled J"], rows)
    return table + "\n" + "".join(verdict(label, r) for label, r in runs.items())


def verdict(label: str, r: dict[str, Any]) -> str:
    """One generated sentence per run: what actually happened to the numbers, better or worse."""
    (q0, q1), (_, o1) = r["q"], r["ov"]

    def move(name: str, a: float, b: float) -> str:
        word = "rose" if b > a + 5e-5 else "fell" if b < a - 5e-5 else "held"
        return "%s **%s** (%.4f → %.4f)" % (name, word, a, b)

    tail = "no quad is inverted or overlapping at the end" if q1.n_inverted == 0 and o1 == 0 else (
        "%d inverted and %d overlapping quads remain" % (q1.n_inverted, o1))
    return "- %s: %s, %s, %s; %s.\n" % (label, move("min scaled J", q0.sj_min, q1.sj_min),
                                        move("mean scaled J", q0.sj_mean, q1.sj_mean),
                                        move("worst 5 %", q0.sj_q05, q1.sj_q05), tail)


def wall_table(ns: dict[str, Any], runs: str = "runs") -> str:
    """The carotid wall script's own readings, one row per run, and a verdict sentence per run."""
    names = {"asis": "as built", "stress": "perturbed input"}
    return runs_table({names[k]: r for k, r in ns[runs].items()})


def moved_lines(ns: dict[str, Any], what: str, runs: str = "runs") -> str:
    """How far the smoother moved vertices and what happened to the size jump: one generated line per run."""
    names = {"asis": "as built", "stress": "perturbed input"}
    out = []
    for k, r in ns[runs].items():
        mv = r["moved"]
        out.append("- %s: vertices of %s moved **%.2f local edges** on average (largest %.2f); the size ratio between "
                   "neighbours went %.2f → %.2f.\n" % (names[k], what, mv.mean(), mv.max(), *r["size_ratio"]))
    return "".join(out)


def frames_table(geometries: list[Geometry], static: str) -> str:
    """One row per geometry whose frames file exists: input against the last stage, read from the frames themselves."""
    rows = []
    for g in geometries:
        path = os.path.join(static, g.frames)
        if not os.path.isfile(path):
            continue
        with open(path) as fid:
            doc = json.load(fid)
        a, b = doc["frames"][0], doc["frames"][-1]
        rows.append([g.label, "`examples/smoother/quad/%s`" % g.script, str(len(doc["quads"])),
                     arrow(a["n_inverted"], b["n_inverted"], "%d"), arrow(a["sj_min"], b["sj_min"]),
                     arrow(a["sj_mean"], b["sj_mean"])])
    return md_table(["geometry", "built by", "quads", "inverted", "min scaled J", "mean scaled J"], rows)


SCRIPTS = ("_static/smoother_viewer.js", "_static/smoother_surface_viewer.js", "_static/smoother_page.js")


def viewer_block(page: str, geometries: list[Geometry], static: str) -> str:
    """The raw-HTML block that mounts the page viewer on the geometries whose frames file exists."""
    cases = [{"key": g.key, "label": g.label, "url": "../../_static/smoother/quad/" + g.frames, "kind": g.kind, "note": g.note}
             for g in geometries if os.path.isfile(os.path.join(static, g.frames))]
    div = "sm-" + page.replace("_", "-")
    return ("```{raw} html\n" + "".join('<script src="../../%s"></script>\n' % p for p in SCRIPTS)
            + '<div id="%s"></div>\n<script>window.initSmootherPage("%s", %s);</script>\n```\n'
            % (div, div, json.dumps(cases, indent=1)))


def write_pages(generated: str, static: str) -> None:
    """Write the viewer block and the table of each page that has at least one frames file under ``static``."""
    os.makedirs(generated, exist_ok=True)
    for page, geometries in PAGES.items():
        if not any(os.path.isfile(os.path.join(static, g.frames)) for g in geometries):
            continue
        with open(os.path.join(generated, page + ".viewer.inc"), "w") as fid:
            fid.write(viewer_block(page, geometries, static))
        if page in OPTIONAL:
            continue                                          # table, movement lines and output come from the run
        if page == "simple_geometries":
            with open(os.path.join(generated, page + ".table.inc"), "w") as fid:
                fid.write(frames_table(geometries, static))


# -- 3-D frames: recorded through the public API on the script's own inputs ----------------------------------
def surface_inputs(stem: str, ns: dict[str, Any]) -> list[tuple[str, str, Any, Any, Any, int]]:
    """``[(file stem, title, mesh, resolved, normals, max frames)]`` of the curved-surface runs of ``stem``."""
    from nekmeshpy.quadmesh.smoother import constrain as cn

    def surf_normals(m: Any, res: Any) -> Any:
        return cn.surface_normals(m.points, m.corners, cn.claimed_surfaces(res))[0]

    if stem == "03_constraints":      # `res` in the namespace is the last (strip) one: resolve the box again
        res = cn.resolve(ns["box"], ns["constraints"])
        return [("03_b_box", "03b box surface: corners frozen, creases slide on their lines", ns["box"], res,
                 ns["nrm_box"], 7)]
    if stem == "04_curved_surfaces":
        return [("04_%s" % k, "04 %s: jittered closed surface, every vertex on the exact surface" % k, r["input"],
                 r["res"], r["nrm"], 7) for k, r in ns["rows"].items()]
    if stem == "cs_hemisphere":
        return [("cs_hemisphere_%s" % k, "flow-past-hemisphere surface, %s" % k, r["before"], ns["res"],
                 surf_normals(r["before"], ns["res"]), 6) for k, r in ns["runs"].items() if k == "stress"]
    for _page, (_g, _stem, prefix, _s, what) in OPTIONAL.items():
        if stem == _stem:
            rows = [("%s_%s" % (prefix, k), "%s, %s (hexes not updated)" % (what, k), r["before"], ns["res"],
                     surf_normals(r["before"], ns["res"]), 6) for k, r in ns["runs"].items()]
            if "box_runs" in ns:    # two frames (input, final): 5000 quads leave no room for a history
                rows += [("%s_box_%s" % (prefix, k), "the outer box, %s (hexes not updated)" % k, r["before"],
                          ns["box_res"], surf_normals(r["before"], ns["box_res"]), 2) for k, r in ns["box_runs"].items()]
            return rows
    if stem == "cs_wall_surface":
        return [("cs_wall_%s" % k, "carotid wall surface, %s (hexes not updated)" % k, r["before"], ns["res"],
                 surf_normals(r["before"], ns["res"]), 3) for k, r in ns["runs"].items()]
    return []


def write_surface_frames(path: str, title: str, mesh: Any, res: Any, nrm: Any, schedule: Any, max_frames: int) -> int:
    """Record the schedule on ``mesh``, keep at most ``max_frames`` frames (the input, evenly spaced stages, the last
    stage), write the playback JSON under its budget (``json_budget``) and return its size.  The labels keep the stage numbers
    of the full run."""
    from nekmeshpy.quadmesh.smoother import frames, smooth

    _, logs = smooth.run_schedule(mesh, res, schedule, normals=nrm, record=True)
    extent = float(np.ptp(np.asarray(mesh.points), axis=0).max())
    decimals = int(np.clip(np.ceil(-np.log10(extent * 1e-3)), 2, 6))      # 1e-3 of the model: under a pixel
    n = len(logs)
    for keep in range(min(max_frames - 1, n), 0, -1):
        pick = sorted({int(round(i)) for i in np.linspace(n - 1, 0, keep)})
        doc = frames.write_frames(path, mesh, [logs[i] for i in pick], nrm, title=title, decimals=decimals)
        for fr, i in zip(doc["frames"][1:], pick, strict=True):
            fr["label"] = "stage %d of %d. %s" % (i + 1, n, logs[i].method)
        for fr in doc["frames"]:
            fr["sj"] = [round(v, 3) for v in fr["sj"]]            # colours only; the summary keeps its own digits
        with open(path, "w") as fid:
            json.dump(doc, fid, separators=(",", ":"))
        if os.path.getsize(path) <= json_budget(path):
            return os.path.getsize(path)
    raise RuntimeError("%s: no frame subset fits in %d bytes" % (path, json_budget(path)))


def run_script(stem: str, scratch: str) -> tuple[dict[str, Any], str]:
    """Execute ``examples/smoother/quad/<stem>.py`` in ``scratch``; return its namespace and captured stdout."""
    cwd = os.getcwd()
    os.chdir(scratch)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ns = runpy.run_path(os.path.join(_EXAMPLES, stem + ".py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    return ns, buf.getvalue()


def build(stems: list[str] | None = None, *, generated: str = GENERATED, static: str = STATIC) -> dict[str, int]:
    """Regenerate the assets of ``stems`` (default: all); returns ``{frames file: bytes}``."""
    stems = list(stems or STAGES)
    os.makedirs(generated, exist_ok=True)
    os.makedirs(static, exist_ok=True)
    sizes: dict[str, int] = {}
    for stem in stems:
        with tempfile.TemporaryDirectory(prefix="nekmeshpy-smoother-") as scratch:
            print("running %s ..." % stem, flush=True)
            ns, _out = run_script(stem, scratch)
            if stem == "cs_wall_surface":
                with open(os.path.join(generated, "carotid_wall.table.inc"), "w") as fid:
                    fid.write(wall_table(ns))
            for page, (_g, ostem, _p, _s, what) in OPTIONAL.items():
                if stem == ostem:
                    with open(os.path.join(generated, page + ".table.inc"), "w") as fid:
                        fid.write(wall_table(ns) + "\n" + moved_lines(ns, what))
                        if "box_runs" in ns:        # the U-tube's second surface, the outer box
                            fid.write("\n**Outer box** (tag `outer`; planes, creases and corners held):\n\n"
                                      + wall_table(ns, "box_runs") + "\n"
                                      + moved_lines(ns, "the outer box", "box_runs"))
            for name in PLANAR_FRAMES.get(stem, []):
                shutil.copy(os.path.join(scratch, name), os.path.join(static, name))
                sizes[name] = os.path.getsize(os.path.join(static, name))
            for fname, title, mesh, res, nrm, maxf in surface_inputs(stem, ns):
                name = fname + "_frames.json"
                sizes[name] = write_surface_frames(os.path.join(static, name), title, mesh, res, nrm,
                                                   ns["SCHEDULE"], maxf)
    for name, size in sizes.items():
        if size > json_budget(name):
            raise RuntimeError("%s is %d bytes, over the %d budget" % (name, size, json_budget(name)))
    write_pages(generated, static)
    return sizes


def main() -> None:
    t0 = time.perf_counter()
    sizes = build(sys.argv[1:] or None)
    for name, size in sorted(sizes.items()):
        print("  %-36s %7.1f KB" % (name, size / 1e3))
    print("total %.1f KB of frames in %.1f s" % (sum(sizes.values()) / 1e3, time.perf_counter() - t0))


if __name__ == "__main__":
    main()
