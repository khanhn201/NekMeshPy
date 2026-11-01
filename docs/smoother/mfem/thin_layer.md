# Demo: a thin designed layer

The block of the hex page {doc}`../hex/thin_layer` (a designed 0.02-thick first layer under a perturbed interior,
boundary frozen), smoothed three ways: the native optimizer, TMOP with `target="ideal"`, and TMOP with
`target="input"` and the layer named `designed`. The stage and its knobs: {doc}`backend`.

```{include} _generated/thin_layer.viewer.inc
```

```{include} _generated/thin_layer.svg.inc
```

## Key points

```{include} _generated/thin_layer.keypoints.inc
```

## The numbers

Check a layer by its thickness: the scaled Jacobian reads 1 for a box-shaped hex of any aspect ratio.

```{include} _generated/thin_layer.table.inc
```

Run `PYTHONPATH=. python examples/smoother/mfem/01_thin_layer.py` (needs `pip install nekmeshpy[smooth]`); it prints
these numbers.
