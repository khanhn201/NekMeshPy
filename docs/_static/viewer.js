// Live mesh viewer, backed by vtk.js (loaded from jsdelivr -- see docs/conf.py's
// html_js_files). docs/_ext/mesh_viewer.py emits the <div>+<script> calling initMeshViewer()
// below, and docs/_ext/gen_viewer_assets.py, which generates the .vtp files it loads.
//
// When the surface carries named boundary groups, io.export.boundary_to_vtp also writes
// a sidecar "<stem>.groups.json" ({bc_id: name}) next to the .vtp. When that sidecar is
// present, the polydata is split into one actor per group (sharing the same points, each
// with its own filtered cell set) so a group can be hidden independently -- e.g. a
// far-field box hiding the body it encloses. Without a sidecar (untagged surface), a
// single flat-coloured actor is used and no toggle UI is shown.
(function () {
  "use strict";

  var PALETTE = [
    [0.86, 0.37, 0.34], [0.35, 0.62, 0.85], [0.47, 0.75, 0.42],
    [0.90, 0.68, 0.30], [0.62, 0.45, 0.80], [0.35, 0.78, 0.78],
    [0.85, 0.45, 0.70], [0.55, 0.55, 0.55],
  ];

  function groupsUrl(vtpUrl) {
    return vtpUrl.replace(/\.vtp$/, ".groups.json");
  }

  function fetchGroups(url) {
    return fetch(url).then(function (resp) {
      if (!resp.ok) {
        throw new Error("no sidecar");
      }
      return resp.json();
    });
  }

  function extractGroupPolyData(sourcePD, bcIdData, targetId) {
    var conn = sourcePD.getPolys().getData();
    var newConn = [];
    var i = 0;
    var cellIndex = 0;
    while (i < conn.length) {
      var n = conn[i];
      if (bcIdData[cellIndex] === targetId) {
        newConn.push(n);
        for (var k = 0; k < n; k++) {
          newConn.push(conn[i + 1 + k]);
        }
      }
      i += n + 1;
      cellIndex += 1;
    }
    var pd = vtk.Common.DataModel.vtkPolyData.newInstance();
    pd.setPoints(sourcePD.getPoints());
    pd.getPolys().setData(Uint32Array.from(newConn));
    return pd;
  }

  function addToggleUI(container, entries) {
    var panel = document.createElement("div");
    panel.style.position = "absolute";
    panel.style.top = "8px";
    panel.style.left = "8px";
    panel.style.zIndex = "10";
    panel.style.background = "rgba(255, 255, 255, 0.85)";
    panel.style.border = "1px solid #ccc";
    panel.style.borderRadius = "4px";
    panel.style.padding = "6px 10px";
    panel.style.font = "13px sans-serif";
    panel.style.color = "#222";
    panel.style.maxHeight = "calc(100% - 16px)";
    panel.style.overflowY = "auto";

    entries.forEach(function (entry) {
      var row = document.createElement("label");
      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.gap = "6px";
      row.style.whiteSpace = "nowrap";
      row.style.cursor = "pointer";

      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = true;
      box.addEventListener("change", function () {
        entry.actor.setVisibility(box.checked);
        entry.renderWindow.render();
      });

      var swatch = document.createElement("span");
      swatch.style.display = "inline-block";
      swatch.style.width = "10px";
      swatch.style.height = "10px";
      swatch.style.background =
        "rgb(" + entry.color.map(function (c) { return Math.round(c * 255); }).join(",") + ")";

      var label = document.createElement("span");
      label.textContent = entry.name;

      row.appendChild(box);
      row.appendChild(swatch);
      row.appendChild(label);
      panel.appendChild(row);
    });

    // vtk.js's render-window interactor listens for pointer events on the whole
    // container (to orbit/pan/zoom the camera) -- without stopping propagation here,
    // it swallows clicks on the checkboxes before their own "change" handler runs,
    // so the box never visibly toggles and nothing else happens either.
    ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "wheel", "touchstart"]
      .forEach(function (evt) {
        panel.addEventListener(evt, function (e) { e.stopPropagation(); });
      });

    container.style.position = "relative";
    container.appendChild(panel);
  }

  function styleActor(actor) {
    actor.getProperty().setEdgeVisibility(true);
    actor.getProperty().setEdgeColor(0.2, 0.2, 0.2);
    // Flat, unlit shading: a mesh viewer cares about geometry/topology, not specular
    // highlights -- lighting only makes faces of one colour look like several
    // depending on the angle they happen to face the camera.
    actor.getProperty().setLighting(false);
  }

  function initMeshViewer(container, vtpUrl) {
    if (!window.vtk) {
      container.textContent = "vtk.js failed to load.";
      return;
    }

    var fullScreenRenderer = vtk.Rendering.Misc.vtkFullScreenRenderWindow.newInstance({
      container: container,
      background: [0.95, 0.95, 0.97],
    });
    var renderer = fullScreenRenderer.getRenderer();
    var renderWindow = fullScreenRenderer.getRenderWindow();

    var reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();

    reader
      .setUrl(vtpUrl, { loadData: true })
      .then(function () {
        var polydata = reader.getOutputData(0);
        var cellData = polydata.getCellData();
        var bcId = cellData ? cellData.getArrayByName("bc_id") : null;

        if (!bcId || bcId.getRange()[1] <= 0) {
          var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
          var actor = vtk.Rendering.Core.vtkActor.newInstance();
          actor.setMapper(mapper);
          mapper.setInputData(polydata);
          mapper.setScalarVisibility(false);
          actor.getProperty().setColor(0.75, 0.75, 0.85);
          styleActor(actor);
          renderer.addActor(actor);
          renderer.resetCamera();
          renderWindow.render();
          return;
        }

        fetchGroups(groupsUrl(vtpUrl))
          .then(function (idToName) {
            var bcIdData = bcId.getData();
            var entries = [];
            Object.keys(idToName)
              .map(Number)
              .sort(function (a, b) { return a - b; })
              .forEach(function (id, idx) {
                var groupPD = extractGroupPolyData(polydata, bcIdData, id);
                if (groupPD.getPolys().getNumberOfValues() === 0) {
                  return;
                }
                var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
                var actor = vtk.Rendering.Core.vtkActor.newInstance();
                actor.setMapper(mapper);
                mapper.setInputData(groupPD);
                mapper.setScalarVisibility(false);
                var color = PALETTE[idx % PALETTE.length];
                actor.getProperty().setColor(color[0], color[1], color[2]);
                styleActor(actor);
                renderer.addActor(actor);
                entries.push({
                  actor: actor,
                  color: color,
                  name: idToName[id],
                  renderWindow: renderWindow,
                });
              });
            addToggleUI(container, entries);
            renderer.resetCamera();
            renderWindow.render();
          })
          .catch(function () {
            // No sidecar (or malformed) -- fall back to single actor coloured by bc_id.
            var mapper = vtk.Rendering.Core.vtkMapper.newInstance();
            var actor = vtk.Rendering.Core.vtkActor.newInstance();
            actor.setMapper(mapper);
            mapper.setInputData(polydata);
            mapper.setScalarModeToUseCellData();
            mapper.setColorModeToMapScalars();
            mapper.setScalarVisibility(true);
            var range = bcId.getRange();
            mapper.setScalarRange(range[0], range[1]);
            styleActor(actor);
            renderer.addActor(actor);
            renderer.resetCamera();
            renderWindow.render();
          });
      })
      .catch(function (err) {
        container.textContent = "Could not load " + vtpUrl + ": " + err;
      });
  }

  window.initMeshViewer = initMeshViewer;
})();
