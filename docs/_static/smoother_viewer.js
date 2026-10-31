// Canvas renderer for a smoothing run on a planar (or nearly planar) quad mesh; the page controller
// (smoother_page.js) owns the selector, slider, play button and charts and drives this through mount().
//
// Input: the JSON written by nekmeshpy.quadmesh.smoother.frames.write_frames
//   {title, quads: [[i0..i3]], frames: [{label, points: [[x,y,z]], sj: [per quad], n_inverted, sj_min, sj_mean}]}
// Quads are coloured by their worst-corner signed scaled Jacobian (red <= 0, through yellow, to green = 1);
// an inverted quad is also outlined in red.  The 2-D projection drops the axis of smallest extent.
// The edge colour comes from the CSS variable --sv-edge, so restyle() redraws after a theme change.
// Curved 3-D surfaces use smoother_surface_viewer.js (vtk.js) instead.
(function () {
  "use strict";

  function colour(sj) {
    // red (<=0) -> orange -> yellow (0.4) -> green (>=1)
    var t = Math.max(0, Math.min(1, sj));
    var stops = [[0.0, [214, 64, 52]], [0.2, [240, 150, 60]], [0.45, [244, 213, 94]], [1.0, [92, 170, 98]]];
    for (var i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        var a = stops[i - 1], b = stops[i], u = (t - a[0]) / (b[0] - a[0]);
        return "rgb(" + [0, 1, 2].map(function (k) { return Math.round(a[1][k] + u * (b[1][k] - a[1][k])); }).join(",") + ")";
      }
    }
    return "rgb(92,170,98)";
  }

  function axes(doc) {
    var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    doc.frames.forEach(function (f) {
      f.points.forEach(function (p) {
        for (var k = 0; k < 3; k++) { if (p[k] < lo[k]) lo[k] = p[k]; if (p[k] > hi[k]) hi[k] = p[k]; }
      });
    });
    var ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    var drop = ext.indexOf(Math.min.apply(null, ext));
    var keep = [0, 1, 2].filter(function (k) { return k !== drop; });
    return { keep: keep, lo: lo, hi: hi };
  }

  function draw(canvas, doc, frame, ax) {
    var ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, pad = 14;
    var a = ax.keep[0], b = ax.keep[1];
    var sx = ax.hi[a] - ax.lo[a], sy = ax.hi[b] - ax.lo[b];
    var s = Math.min((W - 2 * pad) / sx, (H - 2 * pad) / sy);
    var ox = (W - s * sx) / 2, oy = (H - s * sy) / 2;
    function X(p) { return ox + (p[a] - ax.lo[a]) * s; }
    function Y(p) { return H - (oy + (p[b] - ax.lo[b]) * s); }
    ctx.clearRect(0, 0, W, H);
    var dark = getComputedStyle(canvas).getPropertyValue("--sv-edge") || "#333";
    ctx.lineJoin = "round";
    doc.quads.forEach(function (q, i) {
      var p = q.map(function (v) { return frame.points[v]; });
      ctx.beginPath();
      ctx.moveTo(X(p[0]), Y(p[0]));
      for (var k = 1; k < 4; k++) ctx.lineTo(X(p[k]), Y(p[k]));
      ctx.closePath();
      ctx.fillStyle = colour(frame.sj[i]);
      ctx.fill();
      ctx.lineWidth = frame.sj[i] <= 0 ? 2 : 0.7;
      ctx.strokeStyle = frame.sj[i] <= 0 ? "#b00020" : dark.trim() || "#333";
      ctx.stroke();
    });
  }

  // mount(hosts, doc): one canvas in each of the two host elements; show(left, right) draws frame indices.
  window.SmootherPlanar = {
    mount: function (hosts, doc) {
      var ax = axes(doc), cv = hosts.map(function (h) {
        var c = document.createElement("canvas");
        c.width = 520; c.height = 520;
        h.appendChild(c);
        return c;
      }), last = [0, 0];
      function show(l, r) {
        last = [l, r];
        draw(cv[0], doc, doc.frames[l], ax);
        draw(cv[1], doc, doc.frames[r], ax);
      }
      return {
        show: show,
        restyle: function () { show(last[0], last[1]); },
        destroy: function () { cv.forEach(function (c) { c.remove(); }); },
      };
    },
  };
})();
