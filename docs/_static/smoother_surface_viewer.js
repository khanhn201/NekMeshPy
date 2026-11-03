// vtk.js renderer for a smoothing run on a curved 3-D quad surface (vtk.js is loaded site-wide from the CDN by conf.py);
// the page controller (smoother_page.js) owns the selector, slider, play button and charts and drives this through mount().
//
// Input: the JSON written by nekmeshpy.quadmesh.smoother.frames.write_frames
//   {title, quads: [[i0..i3]], frames: [{label, points: [[x,y,z]], sj: [per quad], n_inverted, sj_min, sj_mean}]}
// Two panels share one camera, so orbiting either turns both.  Quads are coloured by their worst-corner signed scaled
// Jacobian (red <= 0, through yellow, to green = 1).  Background and edge colour follow the furo theme
// (--color-background-primary, --sv-edge) and are re-read by restyle() when the theme changes.
(function () {
  "use strict";

  function cssColour(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(v);
    if (!m) return fallback;
    var h = m[1].length === 3 ? m[1].replace(/./g, "$&$&") : m[1];
    return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16) / 255; });
  }

  function makeLut() {
    var lut = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
    [[0.0, 214, 64, 52], [0.2, 240, 150, 60], [0.45, 244, 213, 94], [1.0, 92, 170, 98]].forEach(function (s) {
      lut.addRGBPoint(s[0], s[1] / 255, s[2] / 255, s[3] / 255);
    });
    lut.setMappingRange(0, 1);
    return lut;
  }

  function makePanel(host, doc, lut) {
    var win = vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({
      container: host, background: [1, 1, 1],
    });
    var polys = new Uint32Array(doc.quads.length * 5);
    doc.quads.forEach(function (q, i) { polys[5 * i] = 4; polys.set(q, 5 * i + 1); });
    var pd = vtk.Common.DataModel.vtkPolyData.newInstance();
    var np = doc.frames[0].points.length;
    pd.getPoints().setData(new Float32Array(3 * np), 3);
    pd.getPolys().setData(polys);
    var sj = vtk.Common.Core.vtkDataArray.newInstance({ name: "sj", numberOfComponents: 1, values: new Float32Array(doc.quads.length) });
    pd.getCellData().setScalars(sj);
    var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
    mapper.setInputData(pd);
    mapper.setLookupTable(lut);
    mapper.setScalarModeToUseCellData();
    mapper.setColorModeToMapScalars();
    mapper.setScalarRange(0, 1);
    mapper.setUseLookupTableScalarRange(true);
    var actor = vtk.Rendering.Core.vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setEdgeVisibility(true);
    actor.getProperty().setLighting(false);          // flat colours: the colour is the data, not a highlight
    win.getRenderer().addActor(actor);
    return {
      win: win, actor: actor, pd: pd,
      set: function (frame) {
        var flat = new Float32Array(3 * np);
        for (var i = 0; i < np; i++) { flat[3 * i] = frame.points[i][0]; flat[3 * i + 1] = frame.points[i][1]; flat[3 * i + 2] = frame.points[i][2]; }
        pd.getPoints().setData(flat, 3);
        pd.getPoints().modified();
        sj.setData(Float32Array.from(frame.sj));
        sj.modified();
        pd.modified();
      },
    };
  }

  // mount(hosts, doc): one vtk.js panel in each of the two host elements (they must be laid out, i.e. visible).
  window.SmootherSurface = {
    mount: function (hosts, doc) {
      var lut = makeLut(), panels = hosts.map(function (h) { return makePanel(h, doc, lut); });
      var cam = panels[0].win.getRenderer().getActiveCamera();
      panels[1].win.getRenderer().setActiveCamera(cam);       // one camera: orbit either panel
      var last = [0, 0], lastCam = null;
      cam.onModified(function () {
        var key = cam.getPosition().concat(cam.getViewUp()).join();
        if (key === lastCam) return;
        lastCam = key;
        panels.forEach(function (p) { p.win.getRenderWindow().render(); });
      });
      function restyle() {
        var bg = cssColour(hosts[0], "--color-background-primary", [1, 1, 1]);
        var edge = cssColour(hosts[0], "--sv-edge", [0.2, 0.2, 0.2]);
        panels.forEach(function (p) {
          p.win.getRenderer().setBackground(bg[0], bg[1], bg[2]);
          p.actor.getProperty().setEdgeColor(edge[0], edge[1], edge[2]);
          p.win.getRenderWindow().render();
        });
      }
      function show(l, r) {
        last = [l, r];
        panels[0].set(doc.frames[l]);
        panels[1].set(doc.frames[r]);
        panels.forEach(function (p) { p.win.getRenderWindow().render(); });
      }
      panels[0].set(doc.frames[0]);
      panels[0].win.getRenderer().resetCamera();
      cam.azimuth(35); cam.elevation(25); cam.orthogonalizeViewUp();
      panels[0].win.getRenderer().resetCamera();
      cam.zoom(1.35);
      restyle();
      return {
        show: show,
        restyle: restyle,
        destroy: function () { panels.forEach(function (p) { p.win.delete(); }); },
      };
    },
  };
})();
