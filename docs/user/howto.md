# How-to
## Point->Line->Quad->Hex levels
## Tagging

During meshing, taggings helps with naming, extracting, attaching sections and much more.
Tagging tends to propagate from the lower level upward.

When output to `.re2` file, in HexMesh,
tags of Hex elements can be used to distinguish between solid and fluid mesh
in a conjugate heat transfer mesh; tags of Quad elements can be used to specify boundary conditions
and periodic conditions.

## Boundary Condition
`BoundaryConditions` class specify a mapping from `NekMeshPy` `Tags`
to boundary condition in Nek5000/NekRS,
either through `cbc` field or `boundaryID` field or both.

Tags refer to a tagged **face** of the mesh. HexMesh will have a mapping from tagged Quad to
BC.

```python
from nekmeshpy import BoundaryConditions, writer

VEL_BC  = {"wall": ("W  ", 1), "inlet": ("v  ", 2), "outlet": ("O  ", 3)}  # (cbc, boundaryID)
TEMP_BC = {"wall": ("I  ", 1), "inlet": ("t  ", 2), "outlet": ("O  ", 3)}
BC = BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC)

writer.to_re2(mesh, "case.re2", bc=BC)
writer.to_vtu(mesh, "case.vtu", bc=BC)
```
The code above output both `cbc` and `boundaryID` to `.re2` file. If you prefer to only output
one:
```python
VEL_BC  = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}  # cbc only
# or
VEL_BC  = {"wall": 1    , "inlet": 2    , "outlet": 3    }  # boundaryID only
```
- If a tag is not specified, BC are set to default value `cbc='E  '` and `boundaryID=0`.
- If `cbc` or `boundaryID` explicitly set to `None`, it also default to the default value.
- If `velocity` and `temperature` is specified in `BoundaryConditions`,
  the written `.re2` contains 1 set of BC
  for velocity and 1 set of BC for temperature.
  If only `velocity` is specified, only 1 set of BC for velocity is written to `.re2`.
- `writer.to_vtu` will write `boundaryID` if provided. Else, `to_vtu` will use the ordering of
  given `cbc` as id.

## Conjugate heat transfer

A conjugate mesh has fluid and solid regions. Tag the region on the elements
(`element_tags="fluid"` / `"solid"`), name the fluid/solid interface once, and say which
region is the fluid at export:

```python
VEL_BC  = {"interface": ("W  ", 1),                              # wall, as the fluid sees it
           "inlet": ("v  ", 2),
           "outlet": ("O  ", 3)}
TEMP_BC = {"inlet": ("t  ", 1),
           "outlet":("O  ", 2),
           "outer": ("f  ", 3)}                                  # interface stays conformal
BC = BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC)

writer.to_re2(mesh, "case.re2", bc=BC, fluid="fluid")
```

- **The velocity block only sees the fluid.** The interface is one named face with a hex of
  each region on its sides, but `to_re2` writes velocity rows for fluid elements only, so
  `"interface": ("W  ", 1)` is all it takes: the solid side is left out for you. A velocity
  name that has no fluid-side face at all raises, since it is in the wrong table.
- **`fluid=`** names the velocity-mesh region. Nek expects those elements listed first, so
  `to_re2` reorders the written file (the mesh object is untouched). The header gets
  `nelgv < nelgt`.
- **`temperature=` is required** once `fluid=` leaves a solid region: Nek reads a second
  boundary block whenever `nelgt > nelgv`. It covers every element, and a name left out
  stays conformal `E`, which is what an interface needs for temperature.
- A velocity entry that lands on a solid element raises. It is the wrong field's table.
- If you also write a restart file, pass the same `fluid=` to `writer.to_fld`, so the two
  files number their elements alike.

Examples: `u_tube_box.py`, `wire_coil.py`, `rod_bundle.py`, `chimera.py`.


## Asymmetric BC (flux measuring)

A flux-measuring plane is a surface *inside* one region. It is one named face with a hex
of that region on each side, so it can't be told apart by region, and it should be written
once, from the upstream side. That is the case for a per-region entry: a `{region: ...}`
mapping read against the region of the hex that owns each row, where `None` writes no row.

```python
FLUX_UPSTREAM, FLUX_DOWNSTREAM = "flux_upstream", "flux_downstream"   # element_tags on the two sides

VEL_BC = {"wall": "W  ",
          "inlet": "v  ",
          "outlet": "O  ",
          "flux_1": {FLUX_UPSTREAM: "f1 ", FLUX_DOWNSTREAM: None}}
```

Tag the elements on either side of the plane with two different region names, and give the
plane's entry a code for the upstream one and `None` for the other. Name every region the
boundary borders: a missing one raises rather than being dropped silently.

Examples: `carotid.py`, `femoral.py`.

## Periodic boundary

A periodic face needs a code *and* a correspondence, so they are stated in two places that
must agree.

1. **Name the two ends differently.** One tag over both ends can't say which end is which.
2. **Code both `'P  '`** in the table of each field that is periodic there.
3. **State the pairing and the map** with `hexmesh.Periodic`:

```python
from nekmeshpy import hexmesh
from nekmeshpy.core import affine

PERIODIC = [hexmesh.Periodic("inlet", "outlet", affine.translation([0.0, 0.0, LENGTH]))]
VEL_BC = {"wall": ("W  ", 1), "inlet": ("P  ", 2), "outlet": ("P  ", 2)}

writer.to_re2(mesh, "case.re2", bc=VEL_BC, periodic=PERIODIC)
```

- `transform` carries the first group **onto** the second. It is checked, not trusted: a
  wrong pitch raises with the residual instead of writing a mesh the solver mis-solves.
- The two sides stay two boundary faces; `Periodic` pairs them without welding. The pairing
  fills `bc(1)` / `bc(2)` (partner element and face) of each `'P  '` row.
- **A name is coded `'P  '` if and only if `periodic=` pairs it**, checked per field within
  that field's own region. Either mismatch raises.
- In a conjugate mesh a pair may be periodic for temperature only (a solid cut): code it
  `'P  '` in `TEMP_BC` and leave it out of `VEL_BC`.
- A rotational pair is periodic for a scalar, or for a velocity field that is invariant in
  the global frame: Nek does not rotate vectors across a periodic face.
- A pair already resolved with `hexmesh.periodic_pairs(mesh, PERIODIC)` can be passed as
  `periodic=` instead.

Example: `wire_coil.py`.
