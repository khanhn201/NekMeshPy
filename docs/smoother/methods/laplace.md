# Laplacian smoothing

`smooth.laplace` evens a mesh by many cheap passes. One pass, step by step, computed by the smoother itself:

```{include} ../_generated/methods_laplace.steps.inc
```

## Key points

- **A pass is shrink, average, snap**, with no neighbour list: a vertex sees only the shrunk corners of its own quads.
- **Boundary vertices slide along the boundary**, in three layers ({ref}`boundary-settings`).
- **A pass that hurts is refused**: one that adds an inverted quad, or lowers the minimum scaled Jacobian by more than
  `max_sj_drop`, is rolled back and the stage stops.

## The pass

The bilinear map of quad {math}`q` is {math}`x_q(r,s)=\sum_j N_j(r,s)\,x_j` with {math}`N_j=\tfrac14(1+r_jr)(1+s_js)`.
Step (a) re-reads the four corners at {math}`\pm w` instead of {math}`\pm1`:

```{math}
\tilde x_{q,k} = x_q(w\,r_k,\,w\,s_k) = \alpha\,x_k + \beta\,(x_{k+1}+x_{k-1}) + \gamma\,x_{k+2},

\alpha=\tfrac{(1+w)^2}{4},\ \ \beta=\tfrac{1-w^2}{4},\ \ \gamma=\tfrac{(1-w)^2}{4}.
```

Steps (b) and (c), with {math}`m_v` the number of quad corners at {math}`v`, {math}`P` the projection onto its target,
{math}`i` the pass and {math}`N` the number of passes:

```{math}
x_v \leftarrow \frac{1}{m_v}\sum_{(q,k)\to v}\tilde x_{q,k},
\qquad
x_v \leftarrow x_v + \omega_i\,\bigl(P(x_v)-x_v\bigr),\quad \omega_i=\frac{i}{N}.
```

The last pass is a full snap; a frozen vertex returns to its anchor every pass.

## Boundary vertices

A boundary vertex has quads on one side only, so a plain average pulls it inward. Three layers keep it on the boundary: **source** (a boundary side is not shrunk across its normal), **movement** (under `Tangent`, the move loses its component normal to the boundary) and **constraint** (the snap). One vertex under each setting: {ref}`boundary-settings`. The same rules over a whole disk:

```{include} ../_generated/methods_laplace.boundary.inc
```

## Parameters

```{include} ../_generated/methods_laplace.params.inc
```

`laplace_local` is the same call with `elements` set to the failing quads.

## Limitations

- **No shape objective**: it never reads {math}`\varphi`.
- **`boundary="tangent"` holds a boundary only approximately**: it keeps the area, but a tangent step can leave a
  curved boundary slightly (figure above); `"slide"` or an attached curve holds the shape.
- **`boundary="free"` shrinks a convex boundary**: repeated averaging is diffusion.
- **No repair of folds by design**: nothing in a pass aims at validity, and a fold can stay, as on a concave outline
  ({doc}`winslow`).
- **It relaxes designed grading** ({doc}`grading`).

<details>
<summary>The update on a regular interior vertex</summary>

With four quads around {math}`v`, the pass is the weighted average

```{math}
x_v \leftarrow \alpha\,x_v + \tfrac{\beta}{2}\sum_{4\ \text{edge nbrs}} x_u + \tfrac{\gamma}{4}\sum_{4\ \text{diagonal nbrs}} x_u,
\qquad \alpha+2\beta+\gamma=1,
```

a Laplacian update with self-weight {math}`\alpha`: repeated passes are repeated diffusion.

</details>

## References

- K. Mittal, P. Fischer, "Mesh smoothing for the spectral element method", *J. Sci. Comput.* 78(2), 1152-1173 (2019), [doi:10.1007/s10915-018-0812-9](https://doi.org/10.1007/s10915-018-0812-9). *Used for:* The approach this smoother follows for spectral-element meshes. Cited as a record; its method is not summarised here.
- K. Mittal, "Spectral element mesh generation and improvement methods", M.S. thesis, University of Illinois at Urbana-Champaign (2016), [hdl:2142/95509](http://hdl.handle.net/2142/95509). *Used for:* With the paper above, the mesh-improvement approach this smoother follows for spectral-element meshes.
- D. A. Field, "Laplacian smoothing and Delaunay triangulations", *Communications in Applied Numerical Methods* 4(6), 709-712 (1988), [doi:10.1002/cnm.1630040603](https://doi.org/10.1002/cnm.1630040603). *Used for:* Laplacian smoothing relocates vertices to improve an existing mesh; Field studies it on Delaunay triangulations.
- P. M. Knupp, "A method for hexahedral mesh shape optimization", *Int. J. Numer. Meth. Engng* 58(2), 319-332 (2003), [doi:10.1002/nme.768](https://doi.org/10.1002/nme.768). *Used for:* Abstract: Laplacian smoothing "fails to guarantee improvement of mesh quality" and can create inverted elements; the paper proposes a shape-improvement method that guarantees untangled elements, with element quality based on the condition number of Jacobian matrices (hexahedra).
