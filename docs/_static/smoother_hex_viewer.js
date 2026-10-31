// Generic 3-D hex viewer with toggles (vtk.js, loaded site-wide and deferred by conf.py; this file waits for window.vtk).
//
// Usage on a page (the generated include does this):
//   <script src="../_static/smoother_hex_viewer.js"></script>
//   <div id="sm-hex"></div>
//   <script>window.initSmootherHex("sm-hex", "../_static/smoother/some_manifest.json");</script>
//
// MANIFEST (a small JSON next to the .vtp files; every "vtp" is relative to the manifest):
//   {
//     "title": "text shown above the view",
//     "camera": {"direction": [x, y, z], "viewUp": [x, y, z]},        // optional: unit-ish vector focal point -> eye
//     "variants": [                                                  // one button each; one variant = no switch shown
//       { "label": "original hexes",                                  // the before/after (or any N-way) switch
//         "note": "optional one-line caption under the view",
//         "layers": [
//           { "name": "exterior", "role": "groups",  "vtp": "a_exterior.vtp",        // the gallery-style exterior view
//             "groups": {"1": "wall", "2": "inlet"},                              // bc_id -> name (cell data "bc_id")
//             "hidden": ["outer"] },                                              // groups unticked at first (optional)
//           { "name": "wall",    "role": "surface", "label": "Wall surface",         "vtp": "a_wall.vtp" },
//           { "name": "section", "role": "section", "label": "Hex cross-section",     "vtp": "a_section.vtp" }
//         ],
//         "fields": [                                                // colour-by choices for the "section" layer
//           { "name": "sj", "label": "Scaled Jacobian",    "kind": "sj",  "range": [0, 1] },   // continuous, red -> green
//           { "name": "bl", "label": "Boundary layer",     "kind": "cat", "range": [0, 1],    // 0 / 1 flag
//             "legend": "wall layer | other" }
//         ] },
//       ...
//     ] }
//
// Two views of one mesh: "Exterior" (role "groups": the boundary surface as the gallery draws it, one actor per named
// group with its own show/hide checkbox, coloured by group) and "Cross-section" (roles "surface" + "section").  The
// mode buttons appear only when a variant has a "groups" layer; the exterior is the first view shown.
//
// Layers: role "surface" (translucent, uncoloured shell: the wall) and role "section" (clip polydata: polygons carrying
// the fields as CELL data, drawn opaque and coloured by the chosen field).  Fields: name = the cell-data array; a field is
// offered only if the loaded section polydata really carries it (otherwise its option is not shown, and with fewer than
// two fields the colour-by menu is hidden altogether).  kind "sj" is the red-yellow-green scale of the other pages,
// "cat" a two-colour flag.
//
// Controls (built here): the mode switch, one checkbox per group (exterior); Wall surface, Hex cross-section, Colour by
// (cross-section); Edges (cell outlines of whichever view is showing), the variant switch, Reset camera.  Colours follow furo's CSS variables (--color-background-primary, --sv-edge) and are
// re-read when the theme changes.  No geometry-specific constant lives in this file.
//
// For tests: window.SmootherHexState[id]() returns {variant, colourBy, wall, section, edges, fields, ...}, the visibility
// the actors really have.
(function () {
  "use strict";

  window.SmootherHexState = window.SmootherHexState || {};

  function cssRGB(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(v);
    if (!m) return fallback;
    var h = m[1].length === 3 ? m[1].replace(/./g, "$&$&") : m[1];
    return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16) / 255; });
  }

  var PALETTE = [
    [0.86, 0.37, 0.34], [0.35, 0.62, 0.85], [0.47, 0.75, 0.42], [0.90, 0.68, 0.30],
    [0.62, 0.45, 0.80], [0.35, 0.78, 0.78], [0.85, 0.45, 0.70], [0.55, 0.55, 0.55],
  ];

  // the polys of `pd` whose cell-data bc_id equals `id`, on the same points (as the gallery viewer does)
  function groupPolyData(pd, bcId, id) {
    var conn = pd.getPolys().getData(), out = [], i = 0, cell = 0;
    while (i < conn.length) {
      var n = conn[i];
      if (bcId[cell] === id) for (var k = 0; k <= n; k++) out.push(conn[i + k]);
      i += n + 1; cell += 1;
    }
    var g = vtk.Common.DataModel.vtkPolyData.newInstance();
    g.setPoints(pd.getPoints());
    g.getPolys().setData(Uint32Array.from(out));
    return g;
  }

  function lutFor(kind) {
    var lut = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    var stops = kind === "cat"
      ? [[0.0, 150, 150, 150], [0.49, 150, 150, 150], [0.51, 62, 120, 200], [1.0, 62, 120, 200]]
      : [[0.0, 214, 64, 52], [0.2, 240, 150, 60], [0.45, 244, 213, 94], [1.0, 92, 170, 98]];
    stops.forEach(function (s) { lut.addRGBPoint(s[0], s[1] / 255, s[2] / 255, s[3] / 255); });
    lut.setMappingRange(0, 1);
    return lut;
  }

  function gradientCss(kind) {
    return kind === "cat"
      ? "linear-gradient(90deg, rgb(62,120,200) 0 50%, rgb(150,150,150) 50% 100%)"
      : "linear-gradient(90deg, rgb(214,64,52), rgb(240,150,60) 20%, rgb(244,213,94) 45%, rgb(92,170,98))";
  }

  var CSS = ".sm-hex-title{font-weight:600;margin-bottom:.3rem}" +
    ".sm-hex-controls{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem .9rem;margin:.3rem 0;font-size:.85em}" +
    ".sm-hex-controls label{display:inline-flex;align-items:center;gap:.3rem;cursor:pointer}" +
    ".sm-hex-controls select,.sm-hex-controls button{font:inherit;font-size:.95em;color:var(--color-foreground-primary);" +
    "background:var(--color-background-primary);border:1px solid var(--color-background-border);border-radius:4px;padding:.1rem .5rem;cursor:pointer}" +
    ".sm-hex-host{height:460px}.sm-hex .sm-hex-legend i{width:9rem}";
  function injectCss() {
    if (document.getElementById("sm-hex-css")) return;
    var st = document.createElement("style");
    st.id = "sm-hex-css"; st.textContent = CSS;
    document.head.appendChild(st);
  }

  function resolve(base, url) { return new URL(url, base).href; }

  function loadVtp(url, cache) {
    if (!cache[url]) {
      cache[url] = fetch(url).then(function (r) {
        if (!r.ok) throw new Error(url + ": " + r.status);
        return r.arrayBuffer();
      }).then(function (buf) {
        var rd = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();
        rd.parseAsArrayBuffer(buf);
        return rd.getOutputData(0);
      });
    }
    return cache[url];
  }

  function build(root, manifest, base) {
    var id = root.id;
    injectCss();
    root.classList.add("smoother-viewer", "sm-hex");
    root.innerHTML =
      '<div class="sm-hex-title"></div>' +
      '<div class="sm-tabs sm-hex-variants" role="tablist"></div>' +
      '<div class="sm-tabs sm-hex-modes" role="tablist"></div>' +
      '<div class="sm-hex-controls">' +
      '<span class="sm-hex-ext sm-hex-controls"></span>' +
      '<span class="sm-hex-sec sm-hex-controls">' +
      '<label><input type="checkbox" data-t="surface" checked> Wall surface</label>' +
      '<label><input type="checkbox" data-t="section" checked> Hex cross-section</label>' +
      '<label class="sm-hex-colour">Colour by <select data-t="colour"></select></label></span>' +
      '<label><input type="checkbox" data-t="edges" checked> Edges</label>' +
      '<button type="button" data-t="reset">Reset camera</button></div>' +
      '<div class="sv-host sm-hex-host"></div>' +
      '<div class="sv-legend sm-hex-legend"><span class="sm-hex-lname"></span><i></i><span class="sm-hex-lrange"></span></div>' +
      '<p class="sm-note sm-hex-note"></p>';
    var q = function (s) { return root.querySelector(s); };
    var host = q(".sm-hex-host"), tabs = q(".sm-hex-variants"), note = q(".sm-hex-note");
    var modes = q(".sm-hex-modes"), extBox = q(".sm-hex-ext"), secBox = q(".sm-hex-sec");
    var cbSurface = q('[data-t="surface"]'), cbSection = q('[data-t="section"]'), cbEdges = q('[data-t="edges"]');
    var selColour = q('[data-t="colour"]'), legend = q(".sm-hex-legend");
    q(".sm-hex-title").textContent = manifest.title || "";

    var win = vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({ container: host, background: [1, 1, 1] });
    var ren = win.getRenderer(), rw = win.getRenderWindow();
    var cache = {}, luts = { sj: lutFor("sj"), cat: lutFor("cat") };
    var state = { vi: 0, field: null, fields: [], loaded: null, token: 0, mode: "exterior", hasGroups: false };
    var actors = { surface: [], section: [], groups: [] };            // actors currently in the scene
    var groupOff = {};                                    // group name -> unticked (kept across variants)

    function newActor(pd, opts) {
      var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
      mapper.setInputData(pd);
      var actor = vtk.Rendering.Core.vtkActor.newInstance();
      actor.setMapper(mapper);
      actor.getProperty().setLighting(false);
      ren.addActor(actor);
      return { actor: actor, mapper: mapper, pd: pd, opts: opts };
    }

    function clear() {
      actors.surface.concat(actors.section, actors.groups).forEach(function (a) { ren.removeActor(a.actor); });
      actors = { surface: [], section: [], groups: [] };
      extBox.innerHTML = "";
    }

    function present(pd, f) { return !!pd.getCellData().getArrayByName(f.name); }

    function restyle() {
      var bg = cssRGB(host, "--color-background-primary", [1, 1, 1]);
      var edge = cssRGB(host, "--sv-edge", [0.2, 0.2, 0.2]);
      var shell = cssRGB(host, "--color-foreground-muted", [0.55, 0.55, 0.55]);
      ren.setBackground(bg[0], bg[1], bg[2]);
      actors.section.concat(actors.groups).forEach(function (a) { a.actor.getProperty().setEdgeColor(edge[0], edge[1], edge[2]); });
      actors.surface.forEach(function (a) {
        a.actor.getProperty().setColor(shell[0], shell[1], shell[2]);
        a.actor.getProperty().setEdgeColor(shell[0], shell[1], shell[2]);
      });
      rw.render();
    }

    function applyField() {
      var f = state.field;
      actors.section.forEach(function (a) {
        var m = a.mapper;
        if (!f) { m.setScalarVisibility(false); return; }
        var arr = a.pd.getCellData().getArrayByName(f.name);
        a.pd.getCellData().setActiveScalars(f.name);
        m.setScalarVisibility(!!arr);
        m.setScalarModeToUseCellData();
        m.setColorModeToMapScalars();
        m.setLookupTable(luts[f.kind === "cat" ? "cat" : "sj"]);
        var r = f.range || [0, 1];
        m.setScalarRange(r[0], r[1]);
        m.setUseLookupTableScalarRange(false);
      });
      legend.style.display = f ? "" : "none";
      if (f) {
        q(".sm-hex-lname").textContent = f.label || f.name;
        legend.querySelector("i").style.background = gradientCss(f.kind);
        var r = f.range || [0, 1];
        q(".sm-hex-lrange").textContent = f.kind === "cat" ? (f.legend || "1 | 0") : r[0] + " … " + r[1];
      }
    }

    function applyVisibility() {
      var ext = state.mode === "exterior" && state.hasGroups;
      actors.surface.forEach(function (a) { a.actor.setVisibility(!ext && cbSurface.checked); });
      actors.section.forEach(function (a) {
        a.actor.setVisibility(!ext && cbSection.checked);
        a.actor.getProperty().setEdgeVisibility(cbEdges.checked);
      });
      actors.groups.forEach(function (a) {
        a.actor.setVisibility(ext && !groupOff[a.name]);
        a.actor.getProperty().setEdgeVisibility(cbEdges.checked);
      });
      legend.style.display = !ext && state.field ? "" : "none";
      extBox.style.display = ext ? "" : "none";
      secBox.style.display = ext ? "none" : "";
      Array.prototype.forEach.call(modes.children, function (b) {
        b.setAttribute("aria-selected", String((b.dataset.mode === "exterior") === ext));
      });
      rw.render();
    }

    function addGroups(pd, l) {
      var bc = pd.getCellData().getArrayByName("bc_id");
      if (!bc) return;
      var data = bc.getData();
      Object.keys(l.groups).map(Number).sort(function (a, b) { return a - b; }).forEach(function (id, idx) {
        var name = l.groups[String(id)], g = groupPolyData(pd, data, id);
        if (g.getPolys().getNumberOfValues() === 0) return;
        var a = newActor(g, l);
        var c = PALETTE[idx % PALETTE.length];
        a.name = name; a.color = c;
        a.actor.getProperty().setColor(c[0], c[1], c[2]);
        a.mapper.setScalarVisibility(false);
        a.actor.getProperty().setEdgeVisibility(true);
        if (groupOff[name] === undefined) groupOff[name] = (l.hidden || []).indexOf(name) >= 0;
        actors.groups.push(a);
        var lab = document.createElement("label"), box = document.createElement("input"), sw = document.createElement("i");
        box.type = "checkbox"; box.checked = !groupOff[name]; box.dataset.group = name;
        box.addEventListener("change", function () { groupOff[name] = !box.checked; applyVisibility(); });
        sw.style.cssText = "display:inline-block;width:.7rem;height:.7rem;border-radius:2px;background:rgb(" +
          c.map(function (x) { return Math.round(x * 255); }).join(",") + ")";
        lab.appendChild(box); lab.appendChild(sw); lab.appendChild(document.createTextNode(" " + name));
        extBox.appendChild(lab);
      });
    }

    function fillColourMenu() {
      selColour.innerHTML = "";
      state.fields.forEach(function (f) {
        var o = document.createElement("option");
        o.value = f.name; o.textContent = f.label || f.name;
        selColour.appendChild(o);
      });
      if (state.fields.length) {
        var keep = state.fields.filter(function (f) { return state.field && f.name === state.field.name; })[0];
        state.field = keep || state.fields[0];
        selColour.value = state.field.name;
      } else {
        state.field = null;
      }
      q(".sm-hex-colour").style.display = state.fields.length > 1 ? "" : "none";
    }

    function resetCamera() {
      var cam = ren.getActiveCamera(), c = manifest.camera;
      ren.resetCamera();
      if (c && c.direction) {
        var fp = cam.getFocalPoint(), d = cam.getDistance(), v = c.direction;
        var n = Math.hypot(v[0], v[1], v[2]) || 1;
        cam.setPosition(fp[0] + d * v[0] / n, fp[1] + d * v[1] / n, fp[2] + d * v[2] / n);
        if (c.viewUp) cam.setViewUp(c.viewUp[0], c.viewUp[1], c.viewUp[2]);
        cam.orthogonalizeViewUp();
        ren.resetCamera();
      }
      cam.zoom(1.15);
      rw.render();
    }

    function show(vi, keepCamera) {
      var variant = manifest.variants[vi], token = ++state.token;
      note.textContent = "loading…";
      Promise.all(variant.layers.map(function (l) {
        return loadVtp(resolve(base, l.vtp), cache).then(function (pd) { return { l: l, pd: pd }; });
      })).then(function (loaded) {
        if (token !== state.token) return;
        clear();
        state.vi = vi;
        state.hasGroups = variant.layers.some(function (l) { return l.role === "groups"; });
        modes.style.display = state.hasGroups ? "" : "none";
        Array.prototype.forEach.call(tabs.children, function (b, i) { b.setAttribute("aria-selected", String(i === vi)); });
        var sec = [];
        loaded.forEach(function (x) {
          var role = x.l.role || "section";
          if (role === "groups") { addGroups(x.pd, x.l); return; }
          var a = newActor(x.pd, x.l);
          if (role === "surface") {
            a.actor.getProperty().setOpacity(0.22);
            a.actor.getProperty().setEdgeVisibility(false);
            actors.surface.push(a);
          } else {
            a.actor.getProperty().setEdgeVisibility(true);
            actors.section.push(a);
            sec.push(x.pd);
          }
        });
        state.fields = (variant.fields || []).filter(function (f) {
          return sec.length > 0 && sec.every(function (pd) { return present(pd, f); });
        });
        fillColourMenu();
        applyField();
        applyVisibility();
        restyle();
        note.textContent = variant.note || "";
        if (!keepCamera) resetCamera();
      }).catch(function (e) { note.textContent = "could not load the view: " + e.message; });
    }

    manifest.variants.forEach(function (v, i) {
      var b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab"); b.textContent = v.label;
      b.addEventListener("click", function () { show(i, true); });
      tabs.appendChild(b);
    });
    tabs.classList.toggle("single", manifest.variants.length < 2);
    [["exterior", "Exterior, by boundary group"], ["section", "Cross-section"]].forEach(function (m) {
      var b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab"); b.dataset.mode = m[0]; b.textContent = m[1];
      b.addEventListener("click", function () { state.mode = m[0]; applyVisibility(); });
      modes.appendChild(b);
    });
    [cbSurface, cbSection, cbEdges].forEach(function (c) { c.addEventListener("change", applyVisibility); });
    selColour.addEventListener("change", function () {
      state.field = state.fields.filter(function (f) { return f.name === selColour.value; })[0] || null;
      applyField(); rw.render();
    });
    q('[data-t="reset"]').addEventListener("click", resetCamera);
    new MutationObserver(restyle).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", restyle);

    window.SmootherHexState[id] = function () {
      var cam = ren.getActiveCamera();
      return {
        variant: manifest.variants[state.vi].label, variantIndex: state.vi,
        colourBy: state.field ? state.field.name : null, fields: state.fields.map(function (f) { return f.name; }),
        colourMenuShown: q(".sm-hex-colour").style.display !== "none",
        mode: state.mode, hasGroups: state.hasGroups,
        groups: actors.groups.map(function (a) { return { name: a.name, visible: a.actor.getVisibility() }; }),
        groupEdges: actors.groups.map(function (a) { return a.actor.getProperty().getEdgeVisibility(); }),
        groupCells: actors.groups.map(function (a) { return a.pd.getPolys().getNumberOfValues(); }),
        wall: actors.surface.map(function (a) { return a.actor.getVisibility(); }),
        section: actors.section.map(function (a) { return a.actor.getVisibility(); }),
        edges: actors.section.map(function (a) { return a.actor.getProperty().getEdgeVisibility(); }),
        scalars: actors.section.map(function (a) { return a.mapper.getScalarVisibility(); }),
        cells: actors.section.map(function (a) { return a.pd.getNumberOfCells ? a.pd.getNumberOfCells() : a.pd.getPolys().getNumberOfCells(); }),
        background: ren.getBackground(), cameraPosition: cam.getPosition(),
        legendShown: legend.style.display !== "none",
      };
    };
    show(0, false);
  }

  // initSmootherHex(id, manifestUrl): fetch the manifest, wait for vtk.js (conf.py loads it deferred), build the view.
  window.initSmootherHex = function (id, manifestUrl) {
    var root = document.getElementById(id);
    if (!root) return;
    var base = new URL(manifestUrl, document.baseURI).href;
    root.textContent = "loading…";
    fetch(base).then(function (r) { return r.json(); }).then(function (manifest) {
      var tries = 0;
      (function wait() {
        if (window.vtk) { root.textContent = ""; build(root, manifest, base); return; }
        if (++tries > 200) { root.textContent = "vtk.js failed to load."; return; }
        window.setTimeout(wait, 50);
      })();
    }).catch(function (e) { root.textContent = "could not load the view: " + e.message; });
  };
})();
