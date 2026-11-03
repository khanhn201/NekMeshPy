// The shape metric of one quad corner against its signed scaled Jacobian s, and the barrier-free version the optimizer
// descends on, for a corner with two equal edges (so the edge-ratio factor K is 1 and phi = 1 / s^2):
//
//   exact       phi    = 1 / s^2                                      (even in s: blind to orientation, infinite at s = 0)
//   barrier-free phi_d = c / s_d^2,  s_d = (s + sqrt(s^2 + 4 d^2)) / 2,  c = s_d(1)^2   (finite for every s, 1 at s = 1)
//   with penalty (optional)  (phi_d - 1) + kappa * max(0, eps - s)       (one corner's term of the objective)
//
//   window.initPhiPlot(id, {delta, kappa, eps, deltas})   deltas: the values the slider steps through (smoother.untangle_schedule)
//
// Canvas, log y axis; colours are read from furo's CSS variables and everything is redrawn when the theme changes.
(function () {
  "use strict";

  var W = 560, H = 340;
  var XMIN = -0.6, XMAX = 1.0, YMIN = 0.5, YMAX = 1e5;

  function css(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function sd(s, d) { return 0.5 * (s + Math.sqrt(s * s + 4 * d * d)); }
  function phiD(s, d) { var one = sd(1, d); return one * one / Math.pow(sd(s, d), 2); }
  function phiExact(s) { return 1 / (s * s); }

  window.initPhiPlot = function (id, cfg) {
    var root = document.getElementById(id);
    if (!root) return;
    root.classList.add("smoother-viewer", "phi-plot");
    var stops = cfg.deltas.slice();
    if (stops.indexOf(cfg.delta) < 0) stops.push(cfg.delta);
    stops.sort(function (a, b) { return b - a; });
    root.innerHTML =
      '<div class="pp-controls"><label>delta <input type="range" min="0" max="' + (stops.length - 1) + '" step="1" value="' +
      stops.indexOf(cfg.delta) + '" aria-label="delta"></label><output></output>' +
      '<label class="pp-check"><input type="checkbox" class="pp-pen"> add the inversion penalty</label></div>' +
      '<canvas aria-label="phi and phi_delta against the signed scaled Jacobian"></canvas>';
    var slider = root.querySelector('input[type="range"]'), out = root.querySelector("output");
    var pen = root.querySelector(".pp-pen"), canvas = root.querySelector("canvas");

    function draw() {
      var d = stops[parseInt(slider.value, 10)];
      out.textContent = "delta = " + d + (d === cfg.delta ? " (the optimizer's default)" : "");
      var c = {
        fg: css(root, "--color-foreground-primary", "#222"), muted: css(root, "--color-foreground-muted", "#777"),
        grid: css(root, "--color-background-border", "#ccc"), sel: css(root, "--color-brand-primary", "#2962ff"),
        bad: "#d64034",
      };
      canvas.width = 2 * W; canvas.height = 2 * H;
      var ctx = canvas.getContext("2d");
      ctx.setTransform(2, 0, 0, 2, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.font = "12px sans-serif";
      var L = 52, R = 14, T = 16, B = 40, pw = W - L - R, ph = H - T - B;
      var lmin = Math.log10(YMIN), lmax = Math.log10(YMAX);
      function X(s) { return L + (s - XMIN) / (XMAX - XMIN) * pw; }
      function Y(v) { return T + (1 - (Math.log10(Math.max(v, YMIN)) - lmin) / (lmax - lmin)) * ph; }
      ctx.strokeStyle = c.grid; ctx.lineWidth = 1; ctx.fillStyle = c.muted;
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (var e = 0; e <= 5; e++) {                                   // decade grid
        var yy = Y(Math.pow(10, e));
        ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(L + pw, yy); ctx.stroke();
        ctx.fillText("1e" + e, L - 6, yy);
      }
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      [-0.5, 0, 0.5, 1].forEach(function (s) { ctx.fillText(String(s), X(s), T + ph + 5); });
      ctx.fillText("signed scaled Jacobian s", L + pw / 2, H - 16);
      ctx.save(); ctx.translate(13, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText("per-corner term (log)", 0, 0); ctx.restore();
      ctx.strokeStyle = c.bad; ctx.setLineDash([3, 3]);               // the s = 0 wall
      ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), T + ph); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(X(cfg.eps), T); ctx.lineTo(X(cfg.eps), T + ph); ctx.strokeStyle = c.grid; ctx.stroke();
      ctx.fillStyle = c.muted; ctx.textAlign = "right"; ctx.textBaseline = "top";
      ctx.fillText("s = 0: folded", X(0) - 4, T + ph - 16);
      ctx.textAlign = "left"; ctx.fillText("ε", X(cfg.eps) + 3, T + ph - 16);

      function curve(f, color, width, dash) {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
        ctx.beginPath();
        var started = false, N = 400;
        for (var i = 0; i <= N; i++) {
          var s = XMIN + (XMAX - XMIN) * i / N, v = f(s);
          if (!isFinite(v) || v <= YMIN) { started = false; continue; }
          if (started) ctx.lineTo(X(s), Y(v)); else { ctx.moveTo(X(s), Y(v)); started = true; }
        }
        ctx.stroke(); ctx.setLineDash([]);
      }
      curve(phiExact, c.muted, 1.5, [5, 3]);
      curve(function (s) { return phiD(s, d); }, c.sel, 2.5);
      if (pen.checked) {
        curve(function (s) { return phiD(s, d) - 1 + cfg.kappa * Math.max(0, cfg.eps - s); }, c.bad, 2);
      }
      ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
      var lx = L + pw - 215, ly = T + 14;
      [["exact phi = 1/s²  (infinite at 0)", c.muted], ["barrier-free phi_delta", c.sel]]
        .concat(pen.checked ? [["phi_delta − 1 + penalty", c.bad]] : []).forEach(function (it, k) {
          ctx.fillStyle = it[1]; ctx.fillRect(lx, ly + 18 * k - 8, 14, 3); ctx.fillStyle = c.fg;
          ctx.fillText(it[0], lx + 20, ly + 18 * k);
        });
    }
    slider.addEventListener("input", draw);
    pen.addEventListener("change", draw);
    new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    draw();
  };
})();
