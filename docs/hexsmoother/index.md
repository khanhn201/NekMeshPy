# Hex smoothing

`nekmeshpy.hexmesh.smoother` moves the vertices of a linear hex mesh, never its connectivity, under the same
constraints, Laplace smoothing and shape optimizer as the {doc}`quad smoother <../user/smoother>`
({doc}`user guide <../user/hex_smoother>`).

```{warning}
Experimental. The hex smoother is native (pure NumPy), works on **linear meshes only**, and has been tested on small
meshes (a few hundred hexes). High-order support and an MFEM TMOP backend are planned.
```

Each page below is one small example that shows one behaviour. Every page has a 3-D viewer: the input and each result
as variants, a translucent boundary shell and cross-sections coloured by each hex's worst-corner scaled Jacobian.

| page | shows |
|---|---|
| {doc}`methods` | Laplace, the optimizer and the suggested schedule on one perturbed input, boundary frozen |
| {doc}`slide_box` | `boundary="slide"`: face vertices stay on their planes, edge vertices on their lines, corners fixed |
| {doc}`free_boundary` | a free curved wall: `boundary="free"`, `"tangent"`, `"slide"` and an exact cylinder target |
| {doc}`thin_layer` | a designed thin layer, unprotected and kept with `KeepLayer` |
| {doc}`untangle` | inverted hexes repaired by the optimizer's `delta` continuation |

```{toctree}
:maxdepth: 1
:hidden:

methods
slide_box
free_boundary
thin_layer
untangle
```

Every number on these pages is computed by the example script named on the page and written into the page by
`python docs/_ext/gen_hexsmoother_assets.py`; none is typed.
