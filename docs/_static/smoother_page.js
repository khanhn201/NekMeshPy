// Page controller of the smoothing galleries: a geometry selector, a before | after pair of panels, one slider / play
// button over the smoothing stages, and two compact charts drawn here on canvas from the same frames JSON
// (written by nekmeshpy.quadmesh.smoother.frames.write_frames):
//   {title, quads: [[i0..i3]], frames: [{label, points, sj: [scaled J per quad], n_inverted, sj_min, sj_mean}]}
//
//   window.initSmootherPage(id, cases)   cases = [{key, label, url, kind: "planar" | "surface", note}]
//
// The panels come from smoother_viewer.js (kind "planar", canvas) or smoother_surface_viewer.js (kind "surface",
// vtk.js); the charts are (1) the histogram of the per-quad scaled Jacobian, input against the picked frame, and
// (2) the history of the min and the mean scaled Jacobian over the stages, the picked stage marked.  Colours are
// read from furo's CSS variables, and everything is redrawn when the theme changes.  `?case=KEY&frame=N` in the
// page URL opens on that geometry and stage.
(function () {
  "use strict";

  var W = 400, H = 250;                       // logical size of a chart; the canvas backing store is 2x

  function css(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function summary(f) {
    return f.label + " — " + f.n_inverted + " inverted, min J " + f.sj_min.toFixed(3) + ", mean J " + f.sj_mean.toFixed(3);
  }

  // x of a frame in the history chart: the stage number its label carries ("3. optimize", "stage 8 of 12. ..."), else its index
  function stageOf(f, i) {
    var m = /^(?:stage )?(\d+)[.\s]/.exec(f.label);
    return i === 0 ? 0 : (m ? parseInt(m[1], 10) : i);
  }

  function lowBound(doc) {
    var lo = 1;
    doc.frames.forEach(function (f) { lo = Math.min(lo, f.sj_min); });
    return Math.floor(lo * 10) / 10;
  }

  function colours(root) {
    return {
      fg: css(root, "--color-foreground-primary", "#222"),
      muted: css(root, "--color-foreground-muted", "#777"),
      grid: css(root, "--color-background-border", "#ccc"),
      bg: css(root, "--color-background-primary", "#fff"),
      sel: css(root, "--color-brand-primary", "#2962ff"),
      bad: "#d64034",
    };
  }

  function prep(canvas) {
    canvas.width = 2 * W; canvas.height = 2 * H;
    var ctx = canvas.getContext("2d");
    ctx.setTransform(2, 0, 0, 2, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = "12px sans-serif";
    return ctx;
  }

  function histogram(doc, i) {
    var NB = 20, lo = lowBound(doc);
    function counts(sj) {
      var c = new Array(NB).fill(0);
      sj.forEach(function (v) {
        var b = Math.floor((Math.max(lo, Math.min(1, v)) - lo) / (1 - lo) * NB);
        c[Math.min(NB - 1, b)] += 1;
      });
      return c;
    }
    if (!doc._hmax) {                                       // a fixed y scale over every frame, so bars do not rescale
      doc._hmax = 1;
      doc.frames.forEach(function (f) { doc._hmax = Math.max(doc._hmax, Math.max.apply(null, counts(f.sj))); });
    }
    return { lo: lo, NB: NB, a: counts(doc.frames[0].sj), b: counts(doc.frames[i].sj), ymax: doc._hmax };
  }

  function drawHistogram(canvas, doc, i, c) {
    var ctx = prep(canvas), h = histogram(doc, i);
    var L = 40, R = 8, T = 22, B = 34, pw = W - L - R, ph = H - T - B;
    function X(v) { return L + (v - h.lo) / (1 - h.lo) * pw; }
    function Y(n) { return T + ph - n / h.ymax * ph; }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";            // legend, top left
    ctx.fillStyle = c.muted; ctx.globalAlpha = 0.45; ctx.fillRect(L, 5, 9, 9); ctx.globalAlpha = 1;
    ctx.fillText("input", L + 13, 13);
    ctx.fillStyle = c.sel; ctx.fillRect(L + 58, 5, 9, 9); ctx.fillStyle = c.muted; ctx.fillText("picked stage", L + 71, 13);
    ctx.strokeStyle = c.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, T + ph); ctx.lineTo(L + pw, T + ph); ctx.stroke();
    var bw = pw / h.NB;
    for (var k = 0; k < h.NB; k++) {
      ctx.fillStyle = c.muted; ctx.globalAlpha = 0.45;
      ctx.fillRect(L + k * bw + 1, Y(h.a[k]), bw / 2 - 1, T + ph - Y(h.a[k]));
      ctx.globalAlpha = 1; ctx.fillStyle = c.sel;
      ctx.fillRect(L + k * bw + bw / 2, Y(h.b[k]), bw / 2 - 1, T + ph - Y(h.b[k]));
    }
    if (h.lo < 0) {                                         // the inverted side
      ctx.strokeStyle = c.bad; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), T + ph); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.fillStyle = c.muted; ctx.textBaseline = "top";
    ctx.textAlign = "left"; ctx.fillText(h.lo.toFixed(1), L - 6, T + ph + 4);
    ctx.textAlign = "right"; ctx.fillText("1", L + pw + 4, T + ph + 4);
    if (h.lo < 0) { ctx.textAlign = "center"; ctx.fillStyle = c.bad; ctx.fillText("0", X(0), T + ph + 4); }
    ctx.fillStyle = c.muted; ctx.textAlign = "center"; ctx.fillText("worst-corner scaled J", L + pw / 2, H - 14);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(String(h.ymax), L - 4, T + 3); ctx.fillText("0", L - 4, T + ph);
    ctx.save(); ctx.translate(11, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
    ctx.fillText("quads", 0, 0); ctx.restore();
  }

  function drawHistory(canvas, doc, i, c) {
    var ctx = prep(canvas), lo = lowBound(doc);
    var xs = doc.frames.map(stageOf), xmax = Math.max(1, xs[xs.length - 1]);
    var L = 40, R = 40, T = 22, B = 34, pw = W - L - R, ph = H - T - B;
    function X(s) { return L + s / xmax * pw; }
    function Y(v) { return T + (1 - (v - lo) / (1 - lo)) * ph; }
    ctx.fillStyle = c.muted; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
    ctx.fillText("scaled Jacobian over the stages", L, 13);
    ctx.strokeStyle = c.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, T + ph); ctx.lineTo(L + pw, T + ph); ctx.stroke();
    if (lo < 0) {
      ctx.strokeStyle = c.bad; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(L + pw, Y(0)); ctx.stroke(); ctx.setLineDash([]);
    }
    var series = [["min", "sj_min", c.sel, []], ["mean", "sj_mean", c.fg, [5, 3]]];
    series.forEach(function (s) {
      ctx.strokeStyle = s[2]; ctx.lineWidth = 2; ctx.setLineDash(s[3]);
      ctx.beginPath();
      doc.frames.forEach(function (f, k) { if (k) ctx.lineTo(X(xs[k]), Y(f[s[1]])); else ctx.moveTo(X(xs[k]), Y(f[s[1]])); });
      ctx.stroke(); ctx.setLineDash([]);
      var last = doc.frames[doc.frames.length - 1][s[1]];
      ctx.fillStyle = s[2]; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(s[0], L + pw + 5, Y(last) + (s[0] === "min" ? 6 : -6));
    });
    ctx.strokeStyle = c.muted; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(xs[i]), T); ctx.lineTo(X(xs[i]), T + ph); ctx.stroke();    // the picked stage
    series.forEach(function (s) {
      ctx.fillStyle = c.bg; ctx.strokeStyle = s[2]; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(xs[i]), Y(doc.frames[i][s[1]]), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    });
    ctx.fillStyle = c.muted; ctx.textBaseline = "top";
    ctx.textAlign = "left"; ctx.fillText("input", L, T + ph + 4);
    ctx.textAlign = "right"; ctx.fillText("stage " + xmax, L + pw, T + ph + 4);
    ctx.textAlign = "center"; ctx.fillText("stage", L + pw / 2, H - 14);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText("1", L - 4, T + 3); ctx.fillText(lo.toFixed(1), L - 4, T + ph);
    if (lo < 0) { ctx.fillStyle = c.bad; ctx.fillText("0", L - 4, Y(0)); }
  }

  // A run whose frames file was thinned to fewer than this many frames has no real history to show: the page then
  // shows only the input and the final state side by side (and their histogram), without slider, play or history chart.
  var MIN_HISTORY_FRAMES = 4;

  window.initSmootherPage = function (id, cases) {
    var root = document.getElementById(id);
    if (!root) return;
    root.classList.add("smoother-viewer", "sm-page");
    root.innerHTML =
      '<div class="sm-tabs" role="tablist"></div><p class="sm-note"></p>' +
      '<div class="sv-panels"><figure><div class="sv-host"></div><figcaption class="sv-l"></figcaption></figure>' +
      '<figure><div class="sv-host"></div><figcaption class="sv-r"></figcaption></figure></div>' +
      '<div class="sv-controls"><button type="button" class="sv-play" aria-label="play">▶</button>' +
      '<input type="range" min="0" max="0" value="0" aria-label="stage"><span class="sv-stage"></span></div>' +
      '<div class="sv-legend"><span>scaled Jacobian</span><i></i><span>≤ 0 &nbsp;&nbsp; 0.4 &nbsp;&nbsp; 1</span></div>' +
      '<div class="sm-charts"><figure><canvas></canvas></figure><figure><canvas></canvas></figure></div>';
    var tabs = root.querySelector(".sm-tabs"), note = root.querySelector(".sm-note");
    var hosts = Array.prototype.slice.call(root.querySelectorAll(".sv-host"));
    var slider = root.querySelector("input"), play = root.querySelector(".sv-play");
    var charts = root.querySelectorAll(".sm-charts canvas");
    var docs = {}, state = { doc: null, view: null, cur: 0, timer: null, token: 0, kase: null };
    var qs = window.location.search, qCase = /[?&]case=([\w-]+)/.exec(qs), qFrame = /[?&]frame=(\d+)/.exec(qs);

    function stop() {
      if (state.timer) { clearInterval(state.timer); state.timer = null; }
      play.textContent = "▶";
    }
    function show(i) {
      var doc = state.doc;
      state.cur = i; slider.value = i;
      state.view.show(0, i);
      var c = colours(root);
      drawHistogram(charts[0], doc, i, c);
      if (doc.frames.length >= MIN_HISTORY_FRAMES) drawHistory(charts[1], doc, i, c);
      root.querySelector(".sv-l").textContent = summary(doc.frames[0]);
      root.querySelector(".sv-r").textContent = summary(doc.frames[i]);
      root.querySelector(".sv-stage").textContent = "stage " + i + " / " + (doc.frames.length - 1);
    }
    function load(url) {
      if (!docs[url]) docs[url] = fetch(url).then(function (r) { return r.json(); });
      return docs[url];
    }
    function select(k, first) {
      var token = ++state.token;
      stop();
      if (state.view) { state.view.destroy(); state.view = null; }
      state.kase = k;
      Array.prototype.forEach.call(tabs.children, function (b) {
        b.setAttribute("aria-selected", b.dataset.key === k.key ? "true" : "false");
      });
      note.textContent = k.note || "";
      hosts.forEach(function (h) { h.classList.toggle("surface", k.kind === "surface"); });
      load(k.url).then(function (doc) {
        if (token !== state.token) return;
        if (k.kind === "surface" && !window.vtk) {          // conf.py loads vtk.js deferred: wait for it
          var tries = 0;
          (function wait() {
            if (token !== state.token) return;
            if (window.vtk) { select(k, first); return; }
            if (++tries > 200) { note.textContent = "vtk.js failed to load."; return; }
            window.setTimeout(wait, 50);
          })();
          return;
        }
        state.doc = doc;
        state.view = (k.kind === "surface" ? window.SmootherSurface : window.SmootherPlanar).mount(hosts, doc);
        slider.max = doc.frames.length - 1;
        var n = doc.frames.length;
        root.classList.toggle("sm-compare", n < MIN_HISTORY_FRAMES);
        show(first && qFrame && n >= MIN_HISTORY_FRAMES ? Math.max(0, Math.min(n - 1, parseInt(qFrame[1], 10))) : n - 1);
      }).catch(function (e) { note.textContent = "could not load " + k.url + ": " + e; });
    }

    cases.forEach(function (k) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = k.label; b.dataset.key = k.key; b.setAttribute("role", "tab");
      b.addEventListener("click", function () { select(k, false); });
      tabs.appendChild(b);
    });
    tabs.classList.toggle("single", cases.length < 2);
    slider.addEventListener("input", function () { stop(); show(parseInt(slider.value, 10)); });
    play.addEventListener("click", function () {
      if (state.timer) { stop(); return; }
      var n = state.doc.frames.length;
      if (state.cur >= n - 1) show(0);
      play.textContent = "❚❚";
      state.timer = setInterval(function () {
        if (state.cur >= n - 1) { stop(); return; }
        show(state.cur + 1);
      }, state.kase.kind === "surface" ? 700 : 450);
    });
    var restyle = function () { if (state.view) { state.view.restyle(); show(state.cur); } };
    new MutationObserver(restyle).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", restyle);
    var start = cases.filter(function (k) { return qCase && k.key === qCase[1]; })[0] || cases[0];
    select(start, true);
  };
})();
