# NekMeshPy

An object-oriented, all-hex meshing **toolkit** with Nek5000/NekRS export.

The library is a set of composable primitives: a shared-point mesh model, named
physical groups, `HexMesh` factories, smoothing / surface operations, sizing
fields, quality + topology checks, and meshio I/O. Concrete meshers (bifurcation
vessel, straight pipes, external-flow domains) are built on the toolkit and live
in [`examples/`](https://github.com/nekmeshpy/nekmeshpy/tree/main/examples), not
in the library.

```python
from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model.fields import uniform_spacing

boundary = LineMesh.circle(0.5, 24, element_tags=["wall"] * 24)
section  = QuadMesh.ogrid(boundary, n_side=6, radial=uniform_spacing(4))
mesh     = HexMesh.extrude(section, axis=(0, 0, 1), length=5.0,
                           layers=uniform_spacing(40),
                           first_tag="inlet", last_tag="outlet")
export.to_re2(mesh, "pipe", groups={"wall": "W  ", "inlet": "v  ", "outlet": "O  "})
```

## Where to go next

- **{doc}`user/getting-started`** — install, build, and export your first mesh.
- **{doc}`user/concepts`** — the line→quad→hex ladder, tag systems, factories,
  smoothing, export.
- **{doc}`user/howto`** — recipes distilled from `examples/`.
- **{doc}`reference/index`** — auto-generated API reference.
- **{doc}`developer/architecture`** — architecture, extension points, conventions,
  contributing.

```{toctree}
:hidden:
:caption: User guide

user/getting-started
user/concepts
user/howto
```

```{toctree}
:hidden:
:caption: Developer guide

developer/architecture
developer/extending
developer/conventions
developer/contributing
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```
