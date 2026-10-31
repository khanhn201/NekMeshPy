# Smoothing

Deep dives are the in-depth companion to the {doc}`gallery <../user/gallery>`: the gallery shows meshes, a deep dive
explains one topic. More topics (topology, mesh metrics) will follow.

`nekmeshpy.quadmesh.smoother` moves a quad mesh's vertices, never its connectivity, under constraints, and reports what
it did ({doc}`user guide <../user/smoother>`). Each page below has a viewer (input left, the stage you pick right,
coloured by scaled Jacobian) with a histogram and history of the per-quad scaled Jacobian.

```{toctree}
:maxdepth: 1

simple_geometries
carotid_wall
wire_coil_surface
u_tube_interface
```

## Key points

- **Smoothing repairs a damaged mesh; it does not improve a good one.** Inverted or perturbed inputs recover; a good
  mesh changes little, and some of its numbers can get worse.
- **A smoothed surface is not propagated into a hex mesh.** Each case page's hex section measures what carrying it in
  does.
- **Folded or overlapping input is a known limitation**: a fold can persist.
- **High-order nodes are rebuilt** (carried and projected) after the linear smoothing, not smoothed.

<details>
<summary>Regenerating the pages</summary>

- Every page is a run of a script in `examples/smoother/`. The pages are static Markdown that include fragments from
  `docs/smoother/_generated/` and load `docs/_static/smoother/*_frames.json`; the docs build only reads them, and no
  number is typed.
- After the smoother or an example changes, regenerate and commit:

```bash
python docs/_ext/gen_smoother_assets.py                    # every page, about a minute
python docs/_ext/gen_smoother_assets.py 01_basics          # the frames of one script
python docs/_ext/gen_smoother_hex_assets.py                # the hex views (carotid, U-tube, wire coil), about two minutes
```

</details>
