# A free curved wall

A cylinder whose wall and caps may move, under the same Laplace passes: `boundary="free"`, `"tangent"`, `"slide"`, and `"slide"` with an exact cylinder target.

```{include} _generated/free_boundary.viewer.inc
```

```{include} _generated/free_boundary.svg.inc
```

## Key points

```{include} _generated/free_boundary.keypoints.inc
```

## The numbers

`boundary` applies to every method; the runs here are Laplace only, where `free` and `tangent` differ most. Distances are the largest over the wall vertices to
the exact cylinder and (rims excluded) to the input's faceted wall, and over the cap vertices to the input cap planes.

```{include} _generated/free_boundary.table.inc
```

Run `python examples/hexsmoother/03_free_boundary.py` to reproduce; it prints these numbers.
