# Getting started

This tutorial installs NekMeshPy and walks through building, inspecting, and
exporting your first all-hex mesh — an O-grid ("butterfly") pipe — from the
Python toolkit. It assumes only a working Python 3.9+ environment.

## Install

NekMeshPy is driven entirely from Python; there is no config file or
command-line tool.

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, pytest
```

The extras are optional and split by role: `plot` (matplotlib, for
{mod}`nekmeshpy.io.viz`), `io` (meshio, for the non-Nek writers), `test`
(pytest), `dev` (ruff, mypy, pytest), `docs` (Sphinx), and `all`.

## Build a mesh

The build pattern is always the same: describe the **boundary** as a
{class}`~nekmeshpy.linemesh.LineMesh`, fill it into a **section**
({class}`~nekmeshpy.quadmesh.QuadMesh`), then sweep the section into a **volume**
({class}`~nekmeshpy.hexmesh.HexMesh`). Boundaries are named *as you build* — the
tag rides up from the line onto the swept faces — so no post-hoc face detection is
needed.

```python
from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model.fields import uniform_spacing

# 1. Boundary: a closed circular loop, tagged "wall" at the lowest level.
n = 4 * 6                                   # 4 * n_side points around the ring
boundary = LineMesh.circle(radius=0.5, n=n, element_tags=["wall"] * n)

# 2. Section: fill the loop with a butterfly O-grid, 4 radial layers.
section = QuadMesh.ogrid(boundary, n_side=6, radial=uniform_spacing(4),
                         smoothing_method="bilinear")

# 3. Volume: sweep 40 layers along +z; name the two end caps.
mesh = HexMesh.extrude(section, axis=(0, 0, 1), length=5.0,
                       layers=uniform_spacing(40),
                       first_tag="inlet", last_tag="outlet")
```

## Inspect it

The containers are pure data; the checks are free functions or thin methods that
take the mesh as their first argument.

```python
print(mesh.report())                 # element / point / boundary counts
print(mesh.quality_summary())        # scaled-Jacobian min / mean / n_inverted

assert mesh.is_watertight()          # closed, leak-tight boundary, single body
assert mesh.is_conforming()          # no hanging-point / T-junction interfaces
print(mesh.boundary_group_tags)      # ['inlet', 'outlet', 'wall']
```

## Export it

Boundary **names** are mapped to Nek BC codes (or integer ids) only at export —
the mesh itself carries plain names.

```python
codes = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
export.to_re2(mesh, "pipe", groups=codes)     # native Nek5000/NekRS (.re2 + .rea)
export.to_vtk(mesh, "pipe.vtk", groups=codes) # ParaView / VisIt
export.write(mesh, "pipe.vtu", groups=codes)  # anything meshio supports
```

## Visualize it (optional)

With the `plot` extra installed, render the named boundary faces:

```python
from nekmeshpy.io import viz
viz.plot(mesh, names=["inlet", "outlet", "wall"])   # matplotlib
```

## Next steps

- {doc}`concepts` — how the line→quad→hex ladder, the tag systems, factories, and
  smoothing fit together.
- {doc}`howto` — recipes for pipes, external-flow domains, spheres, and merged
  multi-block meshes, each linked to a runnable `examples/` script.
- {doc}`../reference/index` — the full API.
