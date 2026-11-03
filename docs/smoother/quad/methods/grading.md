# Boundary layers and grading

The shape metric seeks squares, so a designed thin layer is the first thing smoothing relaxes. The protections, on a
graded strip and on a pipe section's O-grid whose wall slides on its circle:

```{include} ../_generated/methods_grading.figure.inc
```

## Key points

- **A designed layer is not ideal to the metric**: a rectangle of edge ratio {math}`r` scores
  {math}`\varphi=((r+1/r)/2)^2` and the objective has no target aspect, so the optimizer thickens it; Laplace less;
  Winslow most once its guard lets it run ({doc}`winslow`).
- **Protecting a layer moves the problem to its edge**: the size jump between kept and relaxed cells grows.

## The protections

- **`KeepLayer(seeds, rings)`** freezes every vertex within `rings` quad-rings of the seeds: the layer is kept exactly,
  but cannot slide; with a sliding wall, the frozen claim wins.
- **`SlideColumns(columns, direction)`** ties rigid columns that slide together along one direction, so a straight wall
  can slide while the cells across the layer keep their shape; `constrain.grid_columns` finds the columns.
- **`weights`** scale each quad's shape term in `optimize`. Alone they do not protect a layer: Laplace and Winslow
  stages ignore them, and a zero weight lets the layer degrade.

## The numbers

```{include} ../_generated/methods_grading.table.inc
```

```{include} ../_generated/methods_grading.text.inc
```

Check a layer by its thickness and `on_target_report`'s `max_frozen_move`: the scaled Jacobian reads 1 for a rectangle
of any edge ratio.
