# Optimizer

`smooth.optimize` lowers a shape objective plus an inversion penalty by nonlinear conjugate gradients, every step
projected onto the vertex's target. The objective per accepted iteration, from two real runs:

```{include} ../_generated/methods_optimizer.history.inc
```

## Key points

- **It minimises {math}`\varphi`**, which is 1 only for a square: angles and edge ratio, not size, so it undoes designed
  grading ({doc}`grading`).
- **It untangles only with a wide {math}`\delta`**: the exact {math}`\varphi` is a wall at a folded corner (plot below).

## The metric

On a quad corner let {math}`a`, {math}`b` be the edges to the next and previous corner (counter-clockwise),
{math}`J=[a\ b]`, and {math}`n` the reference normal (the surface's, see {doc}`constraints`):

```{math}
s=\frac{(a\times b)\cdot n}{|a|\,|b|},
\qquad
\varphi=\frac{\|J\|_F^2\,\|J^{+}\|_F^2}{d^2}=\frac{(|a|^2+|b|^2)^2}{4\,|a\times b|^2}\ \ge 1,\quad d=2 .
```

{math}`s` is the signed sine of the corner angle; {math}`\varphi` the squared condition number of {math}`J` over
{math}`d^2`. Both are dimensionless; {math}`\varphi` is blind to orientation, hence a separate penalty:

```{math}
F(x)=\sum_{q}\frac14\sum_{c\in q}\Bigl[\,w_q\,(\varphi_c-1)+\kappa\,\max(0,\ \varepsilon-s_c)\Bigr].
```

## The barrier and the width delta

With {math}`K=(|a|^2+|b|^2)^2/(4|a|^2|b|^2)`, {math}`\varphi=K/s^2`. The descent uses a regularised sine:

```{math}
s_\delta=\tfrac12\bigl(s+\sqrt{s^2+4\delta^2}\bigr)>0,
\qquad
\varphi_\delta=\frac{c\,K}{s_\delta^{2}},\quad c=s_\delta(1)^2 .
```

```{include} ../_generated/methods_optimizer.plot.inc
```

| in the plot | meaning |
|---|---|
| x axis | {math}`s` of one corner: 1 at a right angle, {math}`\le 0` when folded |
| y axis | that corner's term of {math}`F`, log scale, for equal edges ({math}`K=1`) |
| dashed grey, exact {math}`\varphi=1/s^2` | the shape term; infinite at {math}`s=0`, so a folded corner can never be pulled across |
| blue, {math}`\varphi_\delta` | what the descent minimises: finite everywhere and still falling through {math}`s=0`, so a fold can be undone; 1 at {math}`s=1`, like {math}`\varphi` |
| red (box ticked), {math}`\varphi_\delta-1+\kappa\max(0,\varepsilon-s)` | adds the inversion penalty: a steep slope {math}`\kappa` for any corner below {math}`\varepsilon` |
| slider | {math}`\delta`: wide flattens {math}`\varphi_\delta` below zero (untangles, but loose shape); narrow approaches {math}`\varphi` (sharp shape, a wall again) |

```{include} ../_generated/methods_optimizer.phi.inc
```

## The step and its scale

- **Direction**: Fletcher-Reeves conjugate gradients on the projected analytic gradient.
- **Line search**: doubling, then golden section on {math}`F(\mathrm{snap}(x+\alpha d))`; a trial adding an inverted
  quad is refused.
- **Scale**: the unit is the local edge length {math}`\ell_v`: the largest move per unit {math}`\alpha` is one
  {math}`\ell_v`, and the tolerances count edge lengths. So units do not matter, and a constant factor on {math}`F`
  cancels:

```{include} ../_generated/methods_optimizer.rescale.inc
```

## Parameters

```{include} ../_generated/methods_optimizer.params.inc
```

<details>
<summary>The gradient</summary>

With {math}`A=|a|^2`, {math}`B=|b|^2`, {math}`\mathrm{d}=a\cdot b`, {math}`C=|a\times b|^2=AB-\mathrm d^2`,
{math}`g=(a\times b)\cdot n` and the reference normals held fixed:

```{math}
\frac{\partial\varphi}{\partial a}=\frac{(A+B)\,a}{C}-\frac{(A+B)^2\,(B\,a-\mathrm d\,b)}{2\,C^2},
\qquad
\frac{\partial s}{\partial a}=\frac{b\times n}{\sqrt{AB}}-\frac{g\,a}{A\sqrt{AB}},
\quad
\frac{\partial s}{\partial b}=\frac{n\times a}{\sqrt{AB}}-\frac{g\,b}{B\sqrt{AB}},
```

({math}`b`: swap {math}`a\leftrightarrow b`, {math}`A\leftrightarrow B`); a corner's vertex receives
{math}`-(\partial_a+\partial_b)`, its neighbours {math}`\partial_a`, {math}`\partial_b`. With {math}`\delta>0`: {math}`\partial\varphi_\delta/\partial a=c\,[\,\partial_aK/s_\delta^2-2K\,s_\delta'\,\partial_a s/s_\delta^3\,]`.

</details>

## References

- P. M. Knupp, "Algebraic mesh quality metrics", *SIAM J. Sci. Comput.* 23(1), 193-218 (2001), [doi:10.1137/S1064827500371499](https://doi.org/10.1137/S1064827500371499). *Used for:* An algebraic theory of mesh quality metrics built on the Jacobian; the element condition number measures the distance of an element to the set of degenerate elements. Here {math}`\varphi` is the square of the Frobenius condition number of {math}`J`, scaled by {math}`d^2`.
- L. A. Freitag, P. M. Knupp, "Tetrahedral mesh improvement via optimization of the element condition number", *Int. J. Numer. Meth. Engng* 53(6), 1377-1391 (2002), [doi:10.1002/nme.341](https://doi.org/10.1002/nme.341). *Used for:* A condition-number shape measure for tetrahedra that gives the distance of a tetrahedron from the set of inverted elements, optimised with a mean-quality and a worst-element objective.
- J. M. Escobar, E. Rodriguez, R. Montenegro, G. Montero, J. M. Gonzalez-Yuste, "Simultaneous untangling and smoothing of tetrahedral meshes", *Comput. Methods Appl. Mech. Engrg.* 192(25), 2775-2787 (2003), [doi:10.1016/S0045-7825(03)00299-8](https://doi.org/10.1016/S0045-7825(03)00299-8). *Used for:* Untangling and smoothing treated together, for tetrahedra (as its title says). The barrier-free {math}`\varphi_\delta` above is this implementation's own formulation; no formula of that paper is claimed here.
- K. Mittal, P. Fischer, "Mesh smoothing for the spectral element method", *J. Sci. Comput.* 78(2), 1152-1173 (2019), [doi:10.1007/s10915-018-0812-9](https://doi.org/10.1007/s10915-018-0812-9). *Used for:* The approach this smoother follows for spectral-element meshes. Cited as a record; its method is not summarised here.
- K. Mittal, "Spectral element mesh generation and improvement methods", M.S. thesis, University of Illinois at Urbana-Champaign (2016), [hdl:2142/95509](http://hdl.handle.net/2142/95509). *Used for:* With the paper above, the mesh-improvement approach this smoother follows for spectral-element meshes.
- R. Fletcher, C. M. Reeves, "Function minimization by conjugate gradients", *The Computer Journal* 7(2), 149-154 (1964), [doi:10.1093/comjnl/7.2.149](https://doi.org/10.1093/comjnl/7.2.149). *Used for:* A conjugate-gradient method for unconstrained minimisation.
- J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006), [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5). *Used for:* A textbook on the optimization methods named here (line search, conjugate gradients).
