# How-to recipes

Recipes distilled from runnable scripts under
[`examples/`](https://github.com/khanhn201/NekMeshPy/tree/main/examples) — flat
gmsh-style scripts (constants at top, assign to a `mesh` global, export). Edit the
constants and re-run:

```bash
PYTHONPATH=. python examples/circular_pipe.py    # writes .re2/.vtu in cwd
```

The test suite runs these scripts and inspects the `mesh` global. If a recipe and
its script disagree, the **script is the source of truth**.

## Build an O-grid pipe

Fill a tagged circular loop with an O-grid section and sweep it along an axis.

```python
boundary = linemesh.circle(radius, 4 * n_side, element_tag="wall")
section  = quadmesh.ogrid(boundary, n_side=n_side, radial=4)
mesh     = hexmesh.extrude(section, axis=(0, 0, 1), length=L, layers=n_axial,
                           first_tag="inlet", last_tag="outlet")
```

→ `examples/circular_pipe.py`. For a branching pipe with an internal tagged
interface, see `examples/circular_pipe_tjunction.py`.

## Build a structured duct

Give `quadmesh.structured` four open `LineMesh` edges, each tagged with its own
uniform side name; resolution comes from the edges' own points. Prefer the
**mapping** spelling — `structured({"bottom": …, "right": …, "top": …, "left": …})`
— over the CCW 4-sequence: a misspelt key raises, whereas two transposed positions
give a plausible-looking twisted patch. `side_tags=` (the same four keys) overrides
the edges' own tags.

→ `examples/flow_past_half_cylinder.py`.

## Sweep a section along a curved path

Author **one** cross-section and let `hexmesh.sweep` carry it along the path by a
moving frame — each station is a rigid placement, so the walls come out at the true
offsets through a bend rather than each point following its own copy of the curve.

```python
mesh = hexmesh.sweep(section, centerline, fractions,
                     origin=START,              # required: the section's reference
                     tangent=dcenterline,       # analytic derivative; O(h²) without
                     orientation="fixed", up=PLANE_NORMAL,   # exact for a planar path
                     first_tag="inlet", last_tag="outlet")
```

`origin=` has no default (a disc's centroid is not its centre), and there is no
`order=` — the block inherits the section's. For a path built from pieces of
differing curvature, derive `fractions` with
`linemesh.sweep_fractions(breaks, total_length, target)`: it subdivides each piece
at roughly `target` *on its own*, so every junction carries a station instead of
being straddled by an element fitted across two geometries.

Rather than hand-writing the centerline, walk it — `paths.walk` turtles a table of
moves through space and hands back all of that at once, junction table included:

```python
from nekmeshpy.core import paths

path = paths.walk([paths.line(4.0), paths.arc(2.0, 90.0),
                   paths.arc(2.0, 90.0, tilt=90.0),   # the second bend leaves the plane
                   paths.helix(1.5, 720.0, rise=3.0),
                   paths.line(4.0, roll=90.0)],       # ...twisting as it runs
                  start=START, heading=(0, 0, 1), up=(1, 0, 0))
mesh = hexmesh.sweep_path(section, path, target_length=0.5, origin=START,
                          first_tag="inlet", last_tag="outlet")
```

No `orientation`/`up` at the sweep: a walked path carries its own frame, so `tilt`
and `roll` arrive as authored. `target_length`/`layers` replace `fractions` — the
path knows its own junctions and lands a station on each.

→ `examples/serpentine_pipe.py` — one O-grid disc swept along an 8-pass coil of
straights and 180° U-bends.

## External flow past a body (2-D section, swept)

Wrap a body in an annular O-grid or structured patches, tag the wall and each
far-field side at the line level, and sweep along the span with `loft`, naming the
end caps.

- `examples/flow_past_cylinder.py` — `quadmesh.annulus` between a circle and a
  tagged far-field box (the box's per-line tags split the outer ring into sides).
- `examples/flow_past_half_cylinder.py` — a single `structured` section whose
  bottom edge is a composite curve (ground → semicircular bump → ground).
- `examples/backward_facing_step.py` — three `quadmesh.rectangle` grids joined
  with `quadmesh.attach`, then swept by `hexmesh.extrude`.

## External flow past a sphere (3-D shell)

Build a closed far-field cube surface with `quadmesh.box` (each face tagged with
the far-field side it forms), take the body surface from `quadmesh.sphere` at the
**same** `n` — it is the cube's connectivity with the points projected, so the two
pair by index — and fill the shell with `hexmesh.annulus`:

```python
cube   = quadmesh.box(S, n, patch_tags={"x_min": "inlet", "x_max": "outlet",
                                       "y_min": "bottom", "y_max": "top",
                                       "z_min": "front", "z_max": "back"})
sphere = quadmesh.sphere(R, n)                  # same quads, index-paired
mesh   = hexmesh.annulus(sphere, cube, radial=geometric_spacing(n_radial, ratio))
```

→ `examples/flow_past_sphere.py`, and `examples/flow_past_hemisphere.py` for the
half cubed-sphere shell.

## Stitch multiple blocks

Build each block independently, then weld coincident seam points with
`hexmesh.merge`. Leave a welded-away face **untagged** (`NO_TAG` / omitted
side) so merge stays a plain concatenate with no stale interior tag.

```python
mesh = hexmesh.merge([block_a, block_b])   # welds coincident boundary points only
```

`tol` is a **fraction** of `conform.bbox_scale` -- the largest of the x/y/z ranges over
every point handed in -- not a distance, so the default means the same thing at any
model size. Divide if you know a real distance. Prefer `attach` (below) whenever you
can name the two groups: it needs no tolerance at all.

→ `merge` now survives mainly in `examples/chimera.py` / `chimera_full.py`; the
`flow_past_*` examples build their shells with `annulus` / `extrude`, and the carotid
pipeline has moved to `attach`.

## Join two blocks along an interface you can name

When you already know which face group meets which, say so instead. `attach` confines
the pairing to those two groups and needs no tolerance at all — bijectivity proves it —
and it clears the welded-shut faces for you, which `merge` leaves to the caller.

```python
cap  = hexmesh.boundary_mesh(block, "outlet")        # the block's own nodes, bit-exact
stub = hexmesh.extrude(cap, 0.5, 2, first_tag="join", last_tag="outlet")
mesh = hexmesh.attach([block, stub], [Seam(0, "outlet", 1, "join")])
```

Growing a solid jacket onto a fluid core is the same call, with the interface named so
the exporter can give the two sides different codes:

```python
wall  = hexmesh.boundary_mesh(core, "wall")
shell = hexmesh.loft([wall, *offsets], element_tags="solid",
                     first_tag="inner", last_tag="wall")
mesh  = hexmesh.attach([core, shell],
                       [Seam(0, "wall", 1, "inner", attach_tag="interface")])
```

`attach` is **n-ary**: every seam is stated up front and the assembly is welded and
stitched once.  Chaining two-block joins instead would rebuild the whole accumulated
mesh per link — 32 blocks cost 63240 hex-passes chained against 3840 in one pass.

```python
mesh = hexmesh.attach([core, leg_p, leg_m],
                      [Seam(core, "port_p", leg_p, "join"),
                       Seam(core, "port_m", leg_m, "join")])
```

`hexmesh.tagged_faces(mesh, tag)` hands back the ids a tag resolves to, if you want to
reorder or subset the group before passing it; `quadmesh.attach` / `quadmesh.tagged_edges`
are the same pair one rung down, joining two sections along an edge group.

## The carotid vessel pipeline

A `TriMesh` surface (`data/car.{vtx,tri}`) is cut into legs via seam fields, each
leg filled with `spined_ogrid`, lofted and `attach`ed (its smoother is off:
`SMOOTH_ITERS = 0`). →
`examples/carotid.py` — the golden-regression case: its `.re2` / `.vtu`
output is frozen in `tests/golden/` (coordinates to `1e-12`, connectivity and
boundary tags byte-for-byte), so any change that moves it is a bug unless
deliberately re-based.

## The femoral T-junction: build the surface, then mesh it like a scan

`examples/femoral.py` runs the carotid pipeline on a surface it **builds** rather
than loads — a main vessel and a branch teeing off at an angle, joined on a flat
elliptical seam and ringed by a trough. Everything downstream is the carotid's:
conduction seam fields, a sign-based cut into three legs, conformal seam rings
from Fourier-refit arcs, an O-grid per station, `loft`, `attach`.

Two things make it worth reading on its own.

**The conduction solve is volumetric, not on the wall.** `examples/femoral_vol.py`
tet-meshes the interior with gmsh and solves P1 Laplace there. A harmonic field
with a no-flux wall has `grad(u)` tangent to the wall, so every level set meets the
wall at a right angle — which a field solved on the wall alone cannot tell you
anything about. The Dirichlet condition is imposed on the *cut itself*: the tets are
clipped along each leg's bounding level sets and `u` is pinned on the nodes the clip
creates, because pinning it on a band of nearby nodes instead quantizes the boundary
and shreds the level sets.

**Stations are placed by distance, not by level.** The field is not linear in
distance, so "two diameters from the junction" cannot be read off a level value.
`leg_levels` walks the gradient from seam to cap to build the level-to-distance map,
then lays uniform layers out from the junction and geometric ones to the cap, solving
for the growth ratio so the two meet without a jump.

It needs gmsh (`pip install -e ".[all]"`, or just `".[mesh]"`). The gmsh wheel's
bundled library links against libGLU even when it is driven headlessly, so on a bare
Linux box you may also need `libglu1-mesa`. It caches the built surfaces and the tet
solve under `examples/data/` — a cold run is about five minutes, a warm one about two. Change a knob and the cache key changes with it, so a stale result
is never handed back.
