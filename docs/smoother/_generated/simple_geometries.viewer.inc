```{raw} html
<script src="../_static/smoother_viewer.js"></script>
<script src="../_static/smoother_surface_viewer.js"></script>
<script src="../_static/smoother_page.js"></script>
<div id="sm-simple-geometries"></div>
<script>window.initSmootherPage("sm-simple-geometries", [
 {
  "key": "annulus",
  "label": "annulus",
  "url": "../_static/smoother/01_basics_frames.json",
  "kind": "planar",
  "note": "A jittered annular patch, smoothed with the user schedule (Laplace and optimizer stages alternating)."
 },
 {
  "key": "disk",
  "label": "disk",
  "url": "../_static/smoother/03_a_disk_frames.json",
  "kind": "planar",
  "note": "A disk with its input folded; the wall slides along the circle."
 },
 {
  "key": "pipe",
  "label": "pipe section",
  "url": "../_static/smoother/cs_pipe_stress_frames.json",
  "kind": "planar",
  "note": "The O-grid section of a circular pipe, vertices perturbed; the wall slides on its circle, the layers next to it are kept."
 },
 {
  "key": "box",
  "label": "box",
  "url": "../_static/smoother/03_b_box_frames.json",
  "kind": "surface",
  "note": "The surface of a box: the 8 corners stay, the creases slide on their lines, faces stay on their planes."
 },
 {
  "key": "sphere",
  "label": "sphere",
  "url": "../_static/smoother/04_sphere_frames.json",
  "kind": "surface",
  "note": "A jittered cubed sphere; every vertex is pulled back onto the exact sphere."
 },
 {
  "key": "ellipsoid",
  "label": "ellipsoid",
  "url": "../_static/smoother/04_ellipsoid_frames.json",
  "kind": "surface",
  "note": "A jittered ellipsoid; the elements do not shrink where the surface curves more."
 },
 {
  "key": "torus",
  "label": "torus",
  "url": "../_static/smoother/04_torus_frames.json",
  "kind": "surface",
  "note": "A jittered torus (a donut) with a few inverted quads in the input."
 },
 {
  "key": "hemisphere",
  "label": "hemisphere",
  "url": "../_static/smoother/cs_hemisphere_stress_frames.json",
  "kind": "surface",
  "note": "The hemisphere surface of the flow-past-hemisphere mesh, perturbed; the rim is frozen."
 }
]);</script>
```
