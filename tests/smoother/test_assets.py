"""Smoke tests for the smoothing gallery pages (``docs/smoother/quad/``).

The full asset script runs every script that feeds the pages (about half a minute), so this builds one cheap
geometry (``01_basics``) into a temp directory and checks the contracts the docs build relies on: the frames
JSON is what the page viewer and its charts read and stays small, the page fragments are written from it, and
every fragment and asset a committed page names is committed.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "docs", "_ext"))

import gen_smoother_quad_assets as gen  # noqa: E402

DOCS = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
PAGES = os.path.join(DOCS, "smoother", "quad")


def test_build_one_script_writes_small_frames_and_page_fragments(tmp_path):
    gen_dir, static = tmp_path / "gen", tmp_path / "static"
    sizes = gen.build(["01_basics"], generated=str(gen_dir), static=str(static))
    assert set(sizes) == {"01_basics_frames.json"} and sizes["01_basics_frames.json"] <= gen.MAX_JSON_BYTES
    doc = json.load(open(static / "01_basics_frames.json"))
    assert len(doc["frames"]) >= 2 and len(doc["quads"]) > 0
    # the charts read the per-quad scaled J and the per-frame summary of every frame
    assert {"label", "points", "sj", "n_inverted", "sj_min", "sj_mean"} <= set(doc["frames"][0])
    assert all(len(f["sj"]) == len(doc["quads"]) for f in doc["frames"])
    table = (gen_dir / "simple_geometries.table.inc").read_text()
    assert "| annulus |" in table and re.search(r"\d\.\d{4} → \d\.\d{4}", table)
    assert "disk" not in table                                   # only geometries whose frames exist are listed
    viewer = (gen_dir / "simple_geometries.viewer.inc").read_text()
    assert "01_basics_frames.json" in viewer and "initSmootherPage" in viewer
    assert not (gen_dir / "carotid_wall.viewer.inc").exists()


def test_committed_pages_name_committed_assets():
    """Every ``_generated`` include a page names exists, and every frames file its viewer loads is committed and small."""
    for page, geometries in gen.PAGES.items():
        text = open(os.path.join(PAGES, page + ".md")).read()
        assert "literalinclude" not in text
        incs = re.findall(r"\{include\} (_generated/\S+)", text)
        assert incs, page
        for inc in incs:
            assert os.path.isfile(os.path.join(PAGES, inc)), inc
        viewer = open(os.path.join(PAGES, "_generated", page + ".viewer.inc")).read()
        for g in geometries:
            assert "_static/smoother/quad/" + g.frames in viewer, g.frames
            path = os.path.join(DOCS, "_static", "smoother", "quad", g.frames)
            assert os.path.isfile(path) and os.path.getsize(path) <= gen.json_budget(g.frames), g.frames
        for name in re.findall(r"_static/(\S+?\.js)\"", viewer):
            assert os.path.isfile(os.path.join(DOCS, "_static", name)), name
    committed = {f for f in os.listdir(os.path.join(DOCS, "_static", "smoother", "quad")) if f.endswith(".json") and "manifest" not in f
                 and not f.startswith("methods_")}      # the Methods pages' frames are checked below
    assert committed == {g.frames for gs in gen.PAGES.values() for g in gs}      # nothing unused is left behind


def _check_hex_manifest(prefix, page, fields, groups, n_variants):
    """The hex view (``gen_smoother_quad_hexviews_assets.py``): manifest well formed, every .vtp it names committed and small, the
    exterior carries ``bc_id`` and the groups the manifest names, the page includes every generated fragment."""
    static = os.path.join(DOCS, "_static", "smoother", "quad")
    manifest = json.load(open(os.path.join(static, prefix + "_manifest.json")))
    assert manifest["title"] and len(manifest["variants"]) == n_variants
    total = 0
    for variant in manifest["variants"]:
        assert variant["label"] and {l["role"] for l in variant["layers"]} == {"groups", "surface", "section"}
        assert {f["name"] for f in variant["fields"]} == fields
        for layer in variant["layers"]:
            path = os.path.join(static, layer["vtp"])
            assert os.path.isfile(path) and os.path.getsize(path) <= 400_000, layer["vtp"]
            text = open(path).read()
            assert 'NumberOfPolys="' in text and 'header_type="UInt64"' in text
            if layer["role"] == "section":
                assert all('Name="%s"' % f["name"] in text for f in variant["fields"]), layer["vtp"]
            if layer["role"] == "groups":
                assert 'Name="bc_id"' in text and groups <= set(layer["groups"].values()), layer["vtp"]
                assert set(layer.get("hidden", [])) <= set(layer["groups"].values())
            total += os.path.getsize(path)
    assert total < 2_000_000
    text = open(os.path.join(PAGES, page + ".md")).read()
    assert "## The hexes" in text
    for part in ("viewer", "text", "table") if prefix != "carotid_hex" else ("viewer", "quality", "layers", "text"):
        assert "_generated/%s.%s.inc" % (prefix.replace("_manifest", ""), part) in text
        assert os.path.isfile(os.path.join(PAGES, "_generated", "%s.%s.inc" % (prefix, part)))
    return manifest


def test_carotid_hex_assets_committed_small_and_manifest_valid():
    manifest = _check_hex_manifest("carotid_hex", "carotid_wall", {"sj", "bl"}, {"wall", "trunk_outlet"}, 2)
    assert [v["label"] for v in manifest["variants"]] == ["original", "smoothed wall, rebuilt (same 4 layers)"]
    assert os.path.isfile(os.path.join(DOCS, "_static", "smoother_hex_viewer.js"))
    page = open(os.path.join(PAGES, "carotid_wall.md")).read()
    assert "the hexes are regenerated from it (the carotid recipe re-run), not smoothed themselves" in page
    assert "thicker" not in page


def test_u_tube_and_wire_coil_hex_assets_committed_and_states_what_was_measured():
    ut = _check_hex_manifest("u_tube_hex", "u_tube_interface", {"sj", "region"}, {"interface", "outer", "inlet", "outlet"}, 1)
    wc = _check_hex_manifest("wire_coil_hex", "wire_coil_surface", {"sj", "region"}, {"coil", "wall", "inlet", "outlet"}, 1)
    assert ut["variants"][0]["layers"][0]["hidden"] == ["outer"] and wc["variants"][0]["layers"][0]["hidden"] == ["outer", "wall"]
    for name in ("u_tube_hex", "wire_coil_hex"):
        text = open(os.path.join(PAGES, "_generated", name + ".text.inc")).read()
        assert re.search(r"\d+ of \d+ hexes inverted", text) and "order 7" in text      # numbers, with the sampling order
        assert "cannot be carried" in text or "has no inverted hex" in text
    viewer = open(os.path.join(PAGES, "_generated", "u_tube_interface.viewer.inc")).read()
    assert "outer box" in viewer and "cs_utube_box_asis_frames.json" in viewer
    for fn in os.listdir(os.path.join(PAGES, "_generated")):
        assert "kno" + "ck" not in open(os.path.join(PAGES, "_generated", fn)).read()


def test_carotid_cost_table_has_the_stages_and_shares_sum_to_about_100():
    text = open(os.path.join(PAGES, "_generated", "carotid_cost.table.inc")).read()
    for stage in ("read the tessellation", "seam fields", "seam rings", "original hexes", "extract the wall quads",
                  "smoother set-up", "smoothing: laplace", "smoothing: optimize", "smoothing: local lifts",
                  "rebuild the hexes", "quality readings", "export"):
        assert stage in text, stage
    shares = [float(x) for x in re.findall(r"\| ([\d.]+) %", text.split("**total")[0])]
    assert len(shares) == 12 and abs(sum(shares) - 100) < 1.0
    page = open(os.path.join(PAGES, "carotid_wall.md")).read()
    assert "python examples/smoother/quad/cs_carotid_hex.py" in page and "single run" in page
    machine = open(os.path.join(PAGES, "_generated", "carotid_cost.machine.inc")).read()
    assert "laptop, running serial" in machine


#: local paths and private working folders that must never reach the docs (spelled split, so this file holds none)
PRIVATE = ("wk_" + "claude", "/Disk" + "_K", "/ho" + "me/", "rep" + "orts/", "pl" + "ans/", "no" + "tes/")


def test_methods_pages_assets_committed_small_and_tables_present():
    """The Methods pages (``gen_smoother_quad_methods_assets.py``): every include a page names is committed and every committed
    fragment is used, the figures are inline theme-aware SVG (no viewer, no frames files), and the generated tables carry
    the four methods, their budgets and the parameter defaults read from the code."""
    import inspect

    import gen_smoother_quad_methods_assets as gm

    from nekmeshpy.quadmesh.smoother import smooth

    methods, gen_dir = os.path.join(PAGES, "methods"), os.path.join(PAGES, "_generated")
    static = os.path.join(DOCS, "_static", "smoother", "quad")
    pages = ("index", "laplace", "winslow", "optimizer", "constraints", "grading")
    assert {p + ".md" for p in pages} <= set(os.listdir(methods))
    used = set()
    for page in pages:
        text = open(os.path.join(methods, page + ".md")).read()
        incs = re.findall(r"\{include\} \.\./_generated/(\S+)", text)
        assert incs, page
        for inc in incs:
            assert os.path.isfile(os.path.join(gen_dir, inc)), inc
        used |= set(incs)
        assert "$" not in text, page                                        # math via {math} only
        for name in PRIVATE + ("initSmootherPage",):
            assert name not in text, (page, name)
        if page != "grading":
            assert "## References" in text, page
    committed = {f for f in os.listdir(gen_dir) if f.startswith("methods_")}
    assert committed == used == {"methods_%s.inc" % f for f in gm.FRAGMENTS}      # nothing stale is left behind
    for fn in committed:
        body = open(os.path.join(gen_dir, fn)).read()
        for name in PRIVATE + ("frames.json", "initSmootherPage", "kno" + "ck"):
            assert name not in body, (fn, name)
    assert not [f for f in os.listdir(static) if f.startswith("methods_")]          # the Methods pages load no frames
    assert not os.path.exists(os.path.join(DOCS, "_static", "smoother_methods.js"))
    figure = open(os.path.join(gen_dir, "methods_compare.figure.inc")).read()
    assert figure.count('class="mm-row"') == 6 and figure.count('class="mm-cell"') == 30 and "disk, wall" not in figure
    assert figure.count('style="--mm-cols:5"') == 6 and "L-shape" in figure
    assert "var(--color" not in figure                                    # theme colours live in custom.css
    for name, rows_, cells in (("laplace.boundary", 1, 4), ("winslow.figure", 1, 3), ("constraints.slide", 2, 8)):
        body = open(os.path.join(gen_dir, "methods_%s.inc" % name)).read()
        assert body.count('class="mm-row"') == rows_ and body.count('class="mm-cell"') == cells, name
        assert "var(--color" not in body, name
    assert 'class="fzd"' in open(os.path.join(gen_dir, "methods_constraints.slide.inc")).read()
    steps = open(os.path.join(gen_dir, "methods_laplace.steps.inc")).read()
    assert steps.count('class="lp-fig"') == 3 and "identical" in steps
    assert open(os.path.join(gen_dir, "methods_optimizer.history.inc")).read().count('class="hc-fig"') == 2
    css = open(os.path.join(DOCS, "_static", "custom.css")).read()
    assert ".mm-mesh" in css and ".lp-fig" in css and ".hc-fig" in css and "--mm-cols" in css and ".fzd" in css
    three = open(os.path.join(gen_dir, "methods_compare.methods.inc")).read()
    assert "`laplace(passes=%d, w=%g)`" % (gm.LAPLACE_PASSES, gm.LAPLACE_W) in three
    assert gm.LAPLACE_PASSES == 100 and gm.LAPLACE_W == 0.95 and "`optimize(iters=200)`" in three
    assert gm.WINSLOW_SWEEPS == inspect.signature(smooth.winslow).parameters["sweeps"].default
    assert "`winslow(sweeps=%d)`" % gm.WINSLOW_SWEEPS in three
    assert "`delta=%g`" % inspect.signature(smooth.optimize).parameters["delta"].default in three
    assert gm.compress(smooth.user_schedule()) in three and gm.compress(smooth.untangle_schedule()) in three
    score = open(os.path.join(gen_dir, "methods_compare.score.inc")).read()
    for word in ("Laplace only", "Winslow only", "Optimizer only", "Combined schedule", "min scaled J", "mean phi",
                 "tangled squares", "L-shape", "free disk", "seconds"):
        assert word in score, word
    assert "laptop, running serial" in open(os.path.join(gen_dir, "methods_compare.shows.inc")).read()
    assert "`passes` | `3`" in open(os.path.join(gen_dir, "methods_laplace.params.inc")).read()
    assert "`delta` | `0.01`" in open(os.path.join(gen_dir, "methods_optimizer.params.inc")).read()
    assert "free_boundary" not in open(os.path.join(gen_dir, "methods_laplace.params.inc")).read()
    assert "`sweeps` | `200`" in open(os.path.join(gen_dir, "methods_winslow.params.inc")).read()
    assert "boundary='frozen', corner_deg=30.0" in open(os.path.join(gen_dir, "methods_constraints.params.inc")).read()
    assert "initPhiPlot" in open(os.path.join(gen_dir, "methods_optimizer.plot.inc")).read()
    boundary = open(os.path.join(gen_dir, "methods_constraints.boundary.inc")).read()
    assert boundary.count('class="bm-cell"') == 6 and boundary.count("<svg") >= 6 and "corner" in boundary
    tree = open(os.path.join(gen_dir, "methods_constraints.tree.inc")).read()
    assert "Frozen &gt; OnCurve" in tree and "#boundary-settings" in tree
    index = open(os.path.join(PAGES, "index.md")).read()                   # the Smoothing index page carries the workflow chart
    assert "_generated/" + gm.WORKFLOW in index and os.path.isfile(os.path.join(gen_dir, gm.WORKFLOW))
    for name in PRIVATE:
        assert name not in open(os.path.join(gen_dir, gm.WORKFLOW)).read(), name


def test_methods_schedule_compression_reads_back():
    """``compress`` writes a schedule down without losing a stage: the repeated block and the delta run."""
    import gen_smoother_quad_methods_assets as gm

    from nekmeshpy.quadmesh.smoother import smooth

    assert gm.compress(smooth.user_schedule(outer=2, n_lap=4, n_opt=7, final_local=False)) == \
        "2 × [laplace(passes=4, w=0.97) + optimize(iters=7)]"
    assert gm.compress(smooth.untangle_schedule(deltas=(1.0, 0.1), iters=5)) == "optimize(iters=5, delta=v) for v in 1, 0.1"


def test_every_committed_smoother_asset_is_referenced():
    """A file under ``docs/_static/smoother/`` is only worth committing if a page, a viewer manifest or a generated
    fragment names it, so a leftover from a renamed or dropped example cannot linger."""
    static = os.path.join(DOCS, "_static", "smoother")
    readers = []
    for root, dirs, files in os.walk(DOCS):
        dirs[:] = [d for d in dirs if d not in ("_build", "__pycache__")]
        for f in files:
            full = os.path.join(root, f)
            if f.endswith((".md", ".inc")) or (f.endswith(".json") and os.path.abspath(root).startswith(os.path.abspath(static))
                                                and "manifest" in f):
                readers.append(open(full, encoding="utf-8").read())
    text = "\n".join(readers)
    unused = []
    for root, _, files in os.walk(static):
        for f in files:
            if f.endswith("manifest.json"):
                continue                                  # a manifest is a reader, named by the page that loads it
            if f not in text:
                unused.append(os.path.relpath(os.path.join(root, f), static))
    assert not unused, "committed but referenced nowhere: %s" % sorted(unused)
