# Sliding on a box

With `boundary="slide"` (a fitted `OnSurface` per face, a fitted `OnCurve` per edge) a boundary vertex moves within the boundary as meshed: on its face plane, along its edge line, or not at all at a corner.

```{include} _generated/slide_box.viewer.inc
```

```{include} _generated/slide_box.svg.inc
```

## Key points

```{include} _generated/slide_box.keypoints.inc
```

## The numbers

Distances are each boundary vertex's distance to its input face plane or edge line; corner move is the largest
corner displacement.

```{include} _generated/slide_box.table.inc
```

Run `python examples/smoother/hex/02_slide_box.py` to reproduce; it prints these numbers.
