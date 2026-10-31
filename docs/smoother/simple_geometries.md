# Simple geometries

Eight small meshes, flat and curved, each damaged in its own way and smoothed under the constraints that suit it.

```{include} _generated/simple_geometries.viewer.inc
```

| view | shows |
|---|---|
| panels | left the input, right the stage picked with the slider (or play); quads coloured by worst-corner scaled Jacobian, red at or below zero (inverted) through yellow to green at one |
| histogram | every quad's scaled Jacobian, input (grey) against the picked stage (blue); quads left of the dashed red zero line are inverted |
| history | minimum and mean scaled Jacobian after each stage; the vertical line marks the picked stage |

## Key points

- **Inverted input recovers**: a mesh that starts with inverted quads has a negative minimum and ends with none.
- **Good input mostly moves its mean**, not its minimum.
- **Size does not follow curvature** (the ellipsoid): the shape metric knows angles, not curvature, so elements do not
  get smaller where the surface curves more.

## The numbers

```{include} _generated/simple_geometries.table.inc
```

Run `python examples/smoother/<script>` for the script named in a row to reproduce; it prints these numbers.

<details>
<summary>Notes and caveats</summary>

- The geometries: an annulus, a disk, a pipe section, the surface of a box, a sphere, an ellipsoid, a torus and a
  hemisphere.
- On the 3-D surfaces, drag either panel to orbit both; they need JavaScript and load vtk.js from a CDN.
- The 3-D surfaces keep only a few evenly spaced stages in their data, so their history has fewer points.
- The table compares the input with the last stage, read from the same data the viewer plays.

</details>
