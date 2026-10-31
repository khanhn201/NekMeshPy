```{raw} html
<script src="../../_static/smoother_viewer.js"></script>
<script src="../../_static/smoother_surface_viewer.js"></script>
<script src="../../_static/smoother_page.js"></script>
<div id="sm-u-tube-interface"></div>
<script>window.initSmootherPage("sm-u-tube-interface", [
 {
  "key": "asis",
  "label": "interface",
  "url": "../../_static/smoother/quad/cs_utube_asis_frames.json",
  "kind": "surface",
  "note": "The U-tube wall of the hex mesh as the example builds it; the hexes are not updated."
 },
 {
  "key": "stress",
  "label": "interface, perturbed input",
  "url": "../../_static/smoother/quad/cs_utube_stress_frames.json",
  "kind": "surface",
  "note": "The same wall with its vertices perturbed; the hexes are not updated."
 },
 {
  "key": "box_asis",
  "label": "outer box",
  "url": "../../_static/smoother/quad/cs_utube_box_asis_frames.json",
  "kind": "surface",
  "note": "The six outer faces of the box as the example builds them: planar, the creases and corners held on their lines; the hexes are not updated."
 },
 {
  "key": "box_stress",
  "label": "outer box, perturbed input",
  "url": "../../_static/smoother/quad/cs_utube_box_stress_frames.json",
  "kind": "surface",
  "note": "The same box with its vertices perturbed (put back on its planes and lines); the hexes are not updated."
 }
]);</script>
```
