# Demo: a bent pipe at order 3

A pipe bent through 90 degrees at order 3: an O-grid section of 20 quads swept through 8 layers, 160 hexes, every
wall node on the exact torus. The interior nodes, vertices and curved nodes alike, are perturbed; the wall is held and the caps
slide in their planes (`boundary="slide"`). Three repairs: TMOP on the straight-sided shadow (the order-1 route), the
native optimizer on the curved mesh, and TMOP on the curved mesh, both TMOP runs with `target="input"`. The stage and
its knobs: {doc}`backend`.

```{include} _generated/bent_pipe.viewer.inc
```

```{include} _generated/bent_pipe.svg.inc
```

## Key points

```{include} _generated/bent_pipe.keypoints.inc
```

## How it is set up

The code below is the demo script's own; every name is the landed API.

**1. Build a curved mesh.** `order=` is a factory argument: the wall circle takes it, the O-grid carries the curvature
inward, and `sweep` evaluates the bend's path at every node, so every wall node lies on the torus.

```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
:language: python
:start-after: "[doc:build]"
:end-before: "[/doc:build]"
```

A mesh built from explicit points (`linemesh.loft`, `from_grid`, a plain `loft` between profiles) is high order in
storage and straight in geometry: smoothing it as curved gains nothing.

**2. Constrain the boundary.** `boundary="slide"` puts the caps on their axis-aligned planes and the wall on a curved
target. Under TMOP each edge and face node takes the hold of its vertices: cap nodes slide in their planes, wall nodes
are held, interior nodes are free. Pass `Frozen` constraints to fix more.

```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
:language: python
:start-after: "[doc:constraints]"
:end-before: "[/doc:constraints]"
```

**3. Run TMOP on the curved mesh**, as a schedule stage (`TMOP = {"target": "input"}` in the script). The knobs that matter at order N are `quad_order` (default
`2N + 2`) and `sample_order` (default `max(N, 7)`), both in the stage's dict.

```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
:language: python
:start-after: "[doc:tmop]"
:end-before: "[/doc:tmop]"
:dedent: 4
```

Or as a reviewed round. The session keeps a curved mesh curved; `keep_curved=False` smooths its straight-sided shadow
and returns a linear mesh.

```python
s = hs.SmoothSession(mesh, [], boundary="slide")          # keep_curved=True by default
p = s.propose([("tmop", {"target": "input"})])
print(p.report)               # adds the curved map at sample_order and the .re2 corners, before -> after
s.accept(p)
```

**4. Read the result at the solver's order, and at the corners.** A reading at the stored nodes is silent between them,
and a `.re2` file keeps the 8 corners of each hex only.

```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
:language: python
:start-after: "[doc:quality]"
:end-before: "[/doc:quality]"
:dedent: 4
```

**5. Export.** `.vtu` keeps the curved geometry for viewing; `.re2` writes the corners.

```{literalinclude} ../../../examples/smoother/mfem/02_bent_pipe.py
:language: python
:start-after: "[doc:export]"
:end-before: "[/doc:export]"
```

## The numbers

One scaled Jacobian, read three ways: at the stored nodes, on the order-7 lattice (the order the solver runs at), and
through the 8 corners alone (the straight-sided hex a `.re2` file keeps). For the order-1 route the stored nodes are the
corners. The wall is read between its nodes too, on a finer lattice, against the exact torus.

```{include} _generated/bent_pipe.table.inc
```

Run `PYTHONPATH=. python examples/smoother/mfem/02_bent_pipe.py` (needs `pip install nekmeshpy[smooth]`); it prints
these numbers.
