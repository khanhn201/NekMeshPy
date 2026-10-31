```{raw} html
<script src="../_static/smoother_viewer.js"></script>
<script src="../_static/smoother_surface_viewer.js"></script>
<script src="../_static/smoother_page.js"></script>
<div id="sm-carotid-wall"></div>
<script>window.initSmootherPage("sm-carotid-wall", [
 {
  "key": "asis",
  "label": "as built",
  "url": "../_static/smoother/cs_wall_asis_frames.json",
  "kind": "surface",
  "note": "The wall of the carotid hex mesh as the example builds it; the hexes behind it are not updated."
 },
 {
  "key": "stress",
  "label": "perturbed input",
  "url": "../_static/smoother/cs_wall_stress_frames.json",
  "kind": "surface",
  "note": "The same wall with its vertices perturbed; the hexes behind it are not updated."
 }
]);</script>
```
