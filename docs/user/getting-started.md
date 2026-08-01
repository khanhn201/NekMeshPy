# Getting started

Build, inspect, and export your first all-hex mesh — an O-grid pipe. Assumes
Python 3.9+.

## Install

NekMeshPy is driven from Python; there is no config file or CLI.

```bash
pip install -e .              # core (numpy, scipy)
pip install -e ".[all]"       # + matplotlib, meshio, pytest
```

Extras: `plot` (matplotlib, for {mod}`nekmeshpy.io.viz`), `io` (meshio), `test`
(pytest), `dev` (ruff, mypy, pytest), `docs` (Sphinx), and `all`.

## Build a mesh

The pattern is always: describe the **boundary** as a
{class}`~nekmeshpy.linemesh.LineMesh`, fill it into a **section**
({class}`~nekmeshpy.quadmesh.QuadMesh`), then sweep into a **volume**
({class}`~nekmeshpy.hexmesh.HexMesh`). Boundaries are named as you build — the tag
rides up onto the swept faces — so no post-hoc face detection is needed.

```python
from nekmeshpy import HexMesh, LineMesh, QuadMesh, export

# 1. Boundary: a closed circular loop, tagged "wall" at the lowest level.
n = 4 * 6                                   # 4 * n_side points around the ring
boundary = LineMesh.circle(radius=0.5, n=n, element_tags=["wall"] * n)

# 2. Section: fill the loop with an O-grid, 4 radial layers.
section = QuadMesh.ogrid(boundary, n_side=6, radial=4,
                         smoothing_method="bilinear")

# 3. Volume: sweep 40 layers along +z; name the two end caps.
mesh = HexMesh.extrude(section, axis=(0, 0, 1), length=5.0, layers=40,
                       first_tag="inlet", last_tag="outlet")
```

`radial=4` / `layers=40` is the plain-`int` spelling: *that many uniform layers*,
i.e. exactly `uniform_spacing(n)` from {mod}`nekmeshpy.model.fields`. Pass an
explicit position array there instead (`geometric_spacing(4, 1.2)`) when you want
the layers graded — see
[the layer convention](concepts.md#the-explicit-initial-layer-convention).

## Inspect it

Containers are pure data; the checks take the mesh as their first argument.

```python
print(mesh)                          # <HexMesh 5945 points, 5280 hexes, order 1, …>
print(mesh.report())                 # element / point / boundary counts

stats = mesh.quality_summary()       # a QualitySummary NamedTuple, not a dict
print(stats.min, stats.mean, stats.n_inverted)

assert mesh.is_watertight()          # closed, leak-tight boundary, single body
assert mesh.is_conforming()          # no hanging-point / T-junction interfaces
print(mesh.boundary_group_tags)      # ['inlet', 'outlet', 'wall']
```

Every container has a `__repr__`, so `print(mesh)` gives the one-line inventory —
counts, `order`, and the two tag vocabularies — without reaching for `report()`.
The report-returning reads are **NamedTuples**, reached by attribute:
{class}`~nekmeshpy.model.quality.QualitySummary` from `quality_summary()`,
{class}`~nekmeshpy.model.topology.TopologyReport` from `topology_report()`, and
`WeldResult` from `weld()`.

## Export it

Boundary **names** map to Nek BC codes (or integer ids) only at export.

```python
codes = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
export.to_re2(mesh, "pipe.re2", groups=codes) # native Nek5000/NekRS binary mesh
export.to_vtu(mesh, "pipe.vtu", groups=codes) # ParaView / VisIt (XML VTK)
export.write(mesh, "pipe.msh", groups=codes)  # anything meshio supports
```

## Visualize it (optional)

With the `plot` extra, render named boundary faces:

```python
from nekmeshpy.io import viz
viz.plot(mesh, names=["inlet", "outlet", "wall"])   # matplotlib
```

## Next steps

- {doc}`concepts` — how the ladder, tag systems, factories, and smoothing fit
  together.
- {doc}`howto` — recipes for pipes, external-flow domains, spheres, and merged
  multi-block meshes.
- {doc}`../reference/index` — the full API.
