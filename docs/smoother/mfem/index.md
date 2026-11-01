# MFEM smoother

TMOP, the Target-Matrix Optimization Paradigm of [MFEM](https://mfem.org), runs as an optional stage of the NekMeshPy
smoothers: `("tmop", {...})` in a schedule or a session round. The native optimizer asks every element to become a
square or a cube; TMOP asks each element to match a **target** element instead, so a graded mesh or a designed thin
layer can keep its sizes while its shapes are repaired. It works on planar quad meshes and on hex meshes, at order 1
for now.

```bash
pip install nekmeshpy[smooth]      # PyMFEM
```

```{toctree}
:maxdepth: 1

TMOP via the MFEM backend <backend>
Demo: a thin designed layer <thin_layer>
```

## Workflow

Each box links to the section that explains it.

```{include} _generated/workflow.inc
```

## Key points

- **Optional.** MFEM is imported only when a `tmop` stage runs; the native smoothers never need it.
- **The target is the choice that matters.** `target="input"` keeps each element's own size and aspect, so a graded
  mesh stays graded; `"ideal"` seeks squares and cubes, as the native optimizer does, and thickens a thin layer unless
  that layer is `designed`.
- **Same constraints, partly mapped.** Frozen vertices and axis-aligned planes and lines carry over exactly; a curved
  or tilted target is held fixed for now, and the stage's `reason` says how many vertices were held.
- **Order 1, planar quads.** A curved mesh is smoothed as its straight-sided shadow, and a quad surface in 3-D is
  refused.

<details>
<summary>Regenerating the pages</summary>

The demo page is a run of `examples/smoother/mfem/01_thin_layer.py`; its numbers, viewer and chart are fragments in
`docs/smoother/mfem/_generated/` and files in `docs/_static/smoother/mfem/`, and no number is typed. After the bridge
or the example changes, regenerate and commit (the `thin_layer` key needs PyMFEM; the workflow chart does not):

```bash
python docs/_ext/gen_smoother_mfem_assets.py workflow thin_layer
```

</details>
