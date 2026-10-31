# Which method when

The smoother has three methods, {doc}`Laplace smoothing <laplace>`, {doc}`Winslow smoothing <winslow>` and an
{doc}`optimizer <optimizer>`, and a schedule that combines Laplace and the optimizer. Each runs here on the same inputs
with the same constraints, so the differences are the method's own.

## The four runs

```{include} ../_generated/methods_compare.methods.inc
```

## Side by side

One row per input, every panel on the row at one scale. The L-shape is a concave outline meshed as one block; its
reflex corner is the point (1, 1).

```{include} ../_generated/methods_compare.figure.inc
```

## Scoreboard

```{include} ../_generated/methods_compare.score.inc
```

## What this shows

```{include} ../_generated/methods_compare.shows.inc
```

<details>
<summary>Every reading, per input</summary>

```{include} ../_generated/methods_compare.tables.inc
```

</details>

## Trade-offs

```{include} ../_generated/methods_compare.trade.inc
```

## In practice

- **Attach geometry first** ({doc}`constraints`), or use `boundary="slide"` when there is none: a free boundary
  slides under every method, and the optimizer pushes it out.
- **Start from `s.suggest()`**: when a quad starts inverted it leads with optimizer passes over a wide-to-narrow `delta`.
- **Add a Winslow stage by hand** where Laplace fails: a concave outline, a thin curved block ({doc}`winslow`).
- **Protect designed grading** with `KeepLayer` ({doc}`grading`); Winslow needs it most.
- **Judge by more than one reading**: the scaled Jacobian is blind to edge ratio, size and grading.

```{toctree}
:hidden:
:maxdepth: 1

laplace
winslow
optimizer
constraints
grading
```

## References

- K. Mittal, P. Fischer, "Mesh smoothing for the spectral element method", *J. Sci. Comput.* 78(2), 1152-1173 (2019), [doi:10.1007/s10915-018-0812-9](https://doi.org/10.1007/s10915-018-0812-9). *Used for:* The approach this smoother follows for spectral-element meshes. Cited as a record; its method is not summarised here.
- K. Mittal, "Spectral element mesh generation and improvement methods", M.S. thesis, University of Illinois at Urbana-Champaign (2016), [hdl:2142/95509](http://hdl.handle.net/2142/95509). *Used for:* With the paper above, the mesh-improvement approach this smoother follows for spectral-element meshes.
- S. A. Canann, J. R. Tristano, M. L. Staten, "An approach to combined Laplacian and optimization-based smoothing for triangular, quadrilateral, and quad-dominant meshes", *7th International Meshing Roundtable*, Sandia National Laboratories (1998); no DOI. *Used for:* A combination of constrained Laplacian smoothing with an optimization-based smoother for triangle, quad and mixed meshes: the idea of the combined schedule.
