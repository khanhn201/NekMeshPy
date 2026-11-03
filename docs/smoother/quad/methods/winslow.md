# Winslow smoothing

`smooth.winslow` is an alternative to Laplace for the cases where Laplace fails: **concave outlines** and **thin,
curved, high-aspect blocks**. The L-shape of {doc}`index`, one block, its reflex corner at (1, 1):

```{include} ../_generated/methods_winslow.figure.inc
```

## Key points

- **Inverse Laplace**: the *logical* coordinates are harmonic in *physical* space, so the neighbour weights are
  physical lengths.
- **No fold away from a reflex corner**, where two quads must share a 270-degree turn.
- **Close to the optimizer's shape** on a valid, ungraded input, for less time.
- **Same boundary handling as Laplace**: every constraint and every `boundary` setting ({doc}`constraints`).

## The equation

For each Cartesian component of the map {math}`x(\xi,\eta)`:

```{math}
\alpha\,x_{\xi\xi}-2\beta\,x_{\xi\eta}+\gamma\,x_{\eta\eta}=0,
\qquad
\alpha=|x_\eta|^2,\ \ \beta=x_\xi\!\cdot x_\eta,\ \ \gamma=|x_\xi|^2 .
```

A {math}`\xi` neighbour weighs {math}`\alpha`, the square of the *other* spacing: on a thin cell the long direction
hardly acts. On a surface the constraints drop the normal part of each move, which gives the surface equation.

## A sweep

One Gauss-Seidel pass over coloured vertices, {math}`\alpha,\beta,\gamma` from the current positions, over the ring
of {math}`2n` neighbours {math}`x_k` (edge neighbours and opposite corners), with
{math}`w_k=\alpha W_{\xi\xi,k}-2\beta W_{\xi\eta,k}+\gamma W_{\eta\eta,k}`:

```{math}
x_v \leftarrow x_v+\omega\,\frac{\sum_k w_k\,(x_k-x_v)}{\sum_k w_k}.
```

| vertex | stencil {math}`W` |
|---|---|
| valence 4 | the 9-point stencil, {math}`x_{\xi\xi}=E-2x+W`, {math}`x_{\xi\eta}=\tfrac14(NE-NW+SW-SE)` |
| valence {math}`n\ne4` | least-squares quadratic fit on {math}`n` congruent rhombi of angle {math}`2\pi/n`; keeps {math}`\sum_k w_k>0` |
| boundary | {math}`x\leftarrow\tfrac12(x_{\text{prev}}+x_{\text{next}})`, tangential under `Tangent`; a corner is held |

**The guard**: a sweep that adds an inverted quad, or drops the minimum scaled Jacobian more than `max_sj_drop` below the
best reached, is retried with {math}`\omega` halved (`retries`), then refused: the stage stops. It also stops below
`tol`.

## What it does here

```{include} ../_generated/methods_winslow.table.inc
```

## When to use it, and when not

- **Use it** where Laplace leaves folds on a concave outline or squeezes a thin curved block, or as a cheaper
  optimizer on a valid, ungraded mesh. Add it by hand: `("winslow", {"sweeps": 100})`.
- **Not on a graded layer without `KeepLayer`**: the guard stops it early here, but relaxed it erases the grading.
- **Not for long runs on a closed surface**: with no boundary the mesh can slide over it. Keep `sweeps` modest; the
  guard stops a slow slide.
- **A reflex-corner quad can stay inverted**: no method here repairs it on a one-block layout; give the corner more
  quads.

## Parameters

```{include} ../_generated/methods_winslow.params.inc
```

`winslow_local` is the same call with `elements` set to the failing quads.

## References

- A. M. Winslow, "Numerical solution of the quasilinear Poisson equation in a nonuniform triangle mesh", *Journal of
  Computational Physics* 1(2), 149-172 (1966), [doi:10.1016/0021-9991(66)90001-5](https://doi.org/10.1016/0021-9991(66)90001-5).
  *Used for:* the elliptic smoothing equation, harmonic logical coordinates in the physical domain.
- P. M. Knupp, S. Steinberg, *Fundamentals of Grid Generation*, CRC Press (1993). *Used for:* the inverse form of the
  equations and the discrete 9-point stencil.
