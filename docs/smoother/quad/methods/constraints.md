# Constraints and projection

A constraint says what a vertex may do. Every vertex gets one, resolved once; two shared operations then hold it at every
step: remove the forbidden part of the move, and project back.

## Choosing constraints

```{include} ../_generated/methods_constraints.tree.inc
```

## What a constraint is

A constraint applies to **vertices**, not to the mesh: connectivity never changes. Read it as a boundary condition on
the vertex's motion:

| constraint | like | removes from a move | known geometry | no geometry: fitted to the input |
|---|---|---|---|---|
| `Frozen(ids)` | Dirichlet: fixed | all of it | | |
| `OnCurve(ids, curve=None)` | slip on a curve | all but the curve's tangent, then projects back | `curve=circle(...)`, `line(...)` | a spline through the input boundary |
| `OnSurface(ids, surface=None)` | no penetration | the normal part, then projects back | `surface=sphere(...)`, `plane(...)` | the input faces, as a triangle surface |
| `Tangent(ids)` | slip, no target | the part normal to the boundary; nothing projects back | | the boundary's own tangent, every step |
| free (nothing named) | natural | nothing | | |

With {math}`P` the projection onto the target and {math}`n`, {math}`t` its unit normal and tangent at {math}`P(x)`, the
two operations are

```{math}
\mathrm{snap}:\ x\leftarrow x+\omega\,\bigl(P(x)-x\bigr),
\qquad
\mathrm{constrain}(d)=
\begin{cases}
d-(d\cdot n)\,n & \text{surface}\\
(d\cdot t)\,t & \text{curve, Tangent}\\
0 & \text{Frozen}
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
| planar quads | free | `OnCurve` | `Frozen` |
| quads on a curved 3-D surface | `OnSurface` | the open rim: `OnCurve` | `Frozen` |
| hexes | free | faces: `OnSurface`; edges where two patches meet: `OnCurve` | `Frozen` |

The hex smoother reuses the same objects: {doc}`../../../user/hex_smoother`.

(boundary-settings)=
## Boundary settings

`resolve(mesh, constraints, boundary=...)` is only a shorthand: it states one constraint for every boundary vertex that
no constraint names.

| `boundary` | states | the vertex | when |
|---|---|---|---|
| `"frozen"` (default) | `Frozen` | stays as meshed | the safest |
| `"free"` | nothing | moves as proposed: Laplace pulls a convex boundary in, the optimizer pushes it out | its position does not matter |
| `"tangent"` | `Tangent` | slides along the boundary, never inward | Laplace, Winslow; no geometry |
| `"slide"` | `OnCurve` fitted to the input; hexes: `OnSurface` per patch, `OnCurve` on patch edges | slides along, then is projected onto the fitted curve | no geometry, any method |

Corners are frozen in every mode but `"free"`. One vertex, proposed move to final position, under each:

```{include} ../_generated/methods_constraints.boundary.inc
```

**`Tangent`.** Under it, Laplace and Winslow move a boundary vertex only tangentially: their shrink-and-average step would pull it
toward the chord between its neighbours, inside a convex boundary; the move (Laplace, Winslow) or the gradient
(optimizer) loses its component normal to the boundary, so the vertex slides along the boundary and never moves inward.
The tangent is that of the circle through the vertex and its two boundary neighbours, exact on a circle:

```{math}
t\propto|a|^2\,c-|c|^2\,a
```

with {math}`a=x_{\text{prev}}-x` and {math}`c=x_{\text{next}}-x`.

On a hex boundary a face vertex moves in its patch's tangent plane and a patch-edge vertex along the edge's tangent.
Nothing pulls the vertex back, so a curved boundary is kept to second order, not exactly.

**Slide and OnCurve are one behaviour**, a tangent move then a projection onto a curve; only the curve's origin differs:

| | `OnCurve(ids, curve)` | `boundary="slide"` |
|---|---|---|
| the curve | your own, exact | a cubic spline through the input boundary vertices: periodic on a corner-free loop, 3-D on a surface mesh |
| cut at | your choice of `ids` | corners, and vertices another constraint claims |
| holds | the true shape | the input's shape: a badly sampled boundary stays badly sampled |
| hexes | `OnSurface(ids, surface)` + `OnCurve(ids, curve)` | per patch a plane or its input faces; per patch edge a line or spline |

On a curve Laplace projects its proposed point directly; Winslow and the optimizer take the tangent step first. Both end
on the curve.

**Known geometry goes on the boundary as `OnCurve`**; without it, `"slide"` fits one to the input and `"tangent"` only
slides along the input boundary. On the free disk below the optimizer reaches the same shape under both:

```{include} ../_generated/methods_constraints.slide.inc
```

### Corners

A boundary vertex is a **corner** when its turn is a jump: above `corner_deg` (30 degrees) **and** above three times
the mean turn of its two boundary neighbours. A vertex without exactly two boundary neighbours is one too.

| boundary | turn at a vertex | its neighbours' | corner |
|---|---|---|---|
| a square's corner | 90° | 0° | **yes** |
| a circle sampled with 8 points | 45° | 45° | no |
| a 3:1 ellipse sampled with 12 points, at its tip | 78° | 33° | no: ratio 2.4 |

On a surface mesh the turn is measured in the surface's tangent plane, so the surface's own bending is not a corner.
On hexes the patches split where faces meet at more than `corner_deg`; a vertex where three or more patches meet, or
where a patch edge turns by a jump, is a corner.

## Precedence and conflicts

**Frozen > OnCurve > OnSurface > Tangent > Free.** A vertex on two surfaces or on two curves is frozen and listed in the report;
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
