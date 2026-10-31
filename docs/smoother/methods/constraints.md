# Constraints and projection

A constraint says what a vertex may do. Every vertex gets one, resolved once; two shared operations then hold it at every
step: remove the forbidden part of the move, and project back.

## Choosing constraints

```{include} ../_generated/methods_constraints.tree.inc
```

## What a constraint is

A constraint applies to **vertices**, not to the mesh: connectivity never changes. Read it as a boundary condition on
the vertex's motion:

| constraint | like | removes | directions left |
|---|---|---|---|
| `Frozen` | Dirichlet: position fixed | all of it | 0 |
| `OnCurve` | slip: stay on a curve | the part off the tangent | 1 |
| `OnSurface` | no penetration: stay on a surface | the normal part | 2 |
| free (nothing named) | natural: no condition | nothing | all |

With {math}`P` the projection onto the target and {math}`n`, {math}`t` its unit normal and tangent at {math}`P(x)`, the
two operations are

```{math}
\mathrm{snap}:\ x\leftarrow x+\omega\,\bigl(P(x)-x\bigr),
\qquad
\mathrm{constrain}(d)=
\begin{cases}
d-(d\cdot n)\,n & \text{on a surface}\\
(d\cdot t)\,t & \text{on a curve}\\
0 & \text{frozen}
\end{cases}
```

{math}`\omega=1` is the full projection; Laplace ramps it {math}`i/N\to1`.

```{include} ../_generated/methods_constraints.figure.inc
```

## Targets

| target | what it is | for |
|---|---|---|
| `plane(point, normal)`, `sphere(center, radius)` | exact surfaces | a flat face, a ball |
| `surface_from_trimesh(points, tris)` | a surface read off a tessellation: an approximation | a CAD tessellation, a scanned wall |
| `Surface(project, normal)` | your own closest-point and unit-normal functions | an implicit surface, any other wall |
| `line(point, direction)`, `circle(center, radius, axis)` | exact curves | a straight edge, a round rim |
| `spline_curve(X, closed)` | a cubic spline through ordered points | a rim known only by its points; what `boundary="slide"` builds |
| `Curve(project, tangent)` | your own | any other edge |

Normals come from the target, so one folded quad cannot poison them; a tessellation is only as exact as its triangles.

## Which vertices get which

One dimension below the element: a vertex sits on the boundary of whatever it belongs to.

| mesh | interior | boundary | corners |
|---|---|---|---|
| planar quads | free | `OnCurve`, along each boundary chain | `Frozen` |
| quads on a curved 3-D surface | `OnSurface` | the open rim: `OnCurve` | `Frozen` |
| hexes | free | faces: `OnSurface`; edges where two patches meet: `OnCurve` | `Frozen` |


(boundary-settings)=
## Boundary settings

`boundary` is a convenience: it assigns the constraints above to every boundary vertex that no constraint names.
Below, one vertex, proposed move to final position, under each setting.

```{include} ../_generated/methods_constraints.boundary.inc
```

| boundary setting | what holds the vertex | how it moves | when to use |
|---|---|---|---|
| `"frozen"` (default) | its anchor, restored every step | not at all | a boundary placed as meshed |
| `"free"` + `free_boundary="chain"` | nothing | as proposed, toward the chord: a convex boundary shrinks | the earlier behaviour |
| `"free"` + `free_boundary="tangent"` (default for free) | the tangent rule | only the tangent part of the move; a corner is held | a closed surface, or a boundary that may move |
| `"slide"` | a cubic spline through the input boundary; corners frozen | tangent move, then projected onto the spline | no known geometry, a shape to keep |
| `OnCurve(points, curve)` | your own curve | tangent move, then projected onto it | known geometry: always preferred |

- **Three layers**: source (the boundary side is not shrunk across its normal), movement (the tangent rule), constraint
  (the projection). Each setting adds layers, as the chips show.
- **`free_boundary` belongs to Laplace and Winslow.** The optimizer moves a free boundary vertex anywhere, so it pushes
  a free boundary out.
- **A corner** (turn above `corner_deg`, default 30 degrees, or not two boundary neighbours) is held under `"free"`
  and frozen under `"slide"`.
- **On a curve, Laplace projects the proposed point directly**; the optimizer and Winslow take the tangent step first.
  Both end on the curve.

Under `"slide"`, each unclaimed boundary chain is cut at its corners, at vertices without two boundary neighbours and
at vertices another constraint claims; each piece gets a spline, periodic for a corner-free loop, 3-D on a surface mesh.
The curve is the input boundary's own interpolant: a badly sampled boundary stays badly sampled.

```{include} ../_generated/methods_constraints.slide.inc
```

## Precedence and conflicts

**Frozen > OnCurve > OnSurface > Free.** A vertex on two surfaces or on two curves is frozen and listed in the report;
on one surface and one curve, the curve wins. A box mesh: face interiors `OnSurface(plane)`, the 12 edges
`OnCurve(line)`, the 8 corners `Frozen`. Name an edge vertex from both its faces and it freezes; a corner claimed by
two edges freezes.

## Layers and columns

`KeepLayer(seeds, rings)` freezes every vertex within `rings` quad-rings of the seeds. `SlideColumns(columns, direction)`
ties a rigid column to one slide parameter; it is frozen whole if anything else claims a member. Details and
measurements: {doc}`grading`.

## Calls

```{include} ../_generated/methods_constraints.params.inc
```

## References

- K. Mittal, P. Fischer, "Mesh smoothing for the spectral element method", *J. Sci. Comput.* 78(2), 1152-1173 (2019), [doi:10.1007/s10915-018-0812-9](https://doi.org/10.1007/s10915-018-0812-9). *Used for:* The approach this smoother follows for spectral-element meshes. Cited as a record; its method is not summarised here.
- K. Mittal, "Spectral element mesh generation and improvement methods", M.S. thesis, University of Illinois at Urbana-Champaign (2016), [hdl:2142/95509](http://hdl.handle.net/2142/95509). *Used for:* With the paper above, the mesh-improvement approach this smoother follows for spectral-element meshes.
