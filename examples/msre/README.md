# msre: conformal hex mesh of the MSRE fuel-salt volume

`./run.sh` rebuilds a conformal, Nek5000-valid hex mesh of the Molten Salt Reactor Experiment
(MSRE) fuel-salt fluid domain, `core_fluid_cleanup_v5.STEP` (one OCC solid, 14,113 faces, mm,
axis = z, z 25.4 to 2708). The STEP is not shipped. The pipeline starts from four shipped inputs: the
CAD-derived `p1/lattice.pkl`, `p4/bore_axes.npy` and `p4/pipe_axis.npy`, and the frozen
bottom-zone topology `p4/topo.pkl` (see below).

Output: `p4/msre.re2`, `p4/msre.vtu` (order 2) and `p4/msre.pkl` (a `HexMesh`).

- **Size and validity:** about 2.45M hexes at order 2 with core `NZ=32` (2,447,348 on the reference
  run; the count can change by a lower-head layer between machines, see *Reproducibility*). Zero
  inverted corners and zero twisted faces.
- **Connectivity:** one component, conformal (no hanging points, no improper element pairs). All boundary faces
  are tagged. The written `.re2` passes nekRS/parRSB's `parCon` checks (`p4/re2check.py`); it stays
  unambiguous for `connectivityTol` below about 0.31 (the default is 0.2).
- **Boundary rows** in the `.re2`, as `cbc` / `boundaryID`:

  | name | cbc | boundaryID | faces |
  |---|---|---|---|
  | inlet | `v  ` | 1 | 308 |
  | outlet | `O  ` | 2 | 140 |
  | wall | `W  ` | 3 | all others (649,362) |
  | channel | `W  ` | 4 | 431,664 (fuel-channel side walls in the core, incl. the tall cells' channels up to the post tops) |

This is a multi-script pipeline, not a flat example. `tests/test_examples.py` runs only the
top-level `examples/*.py`, so it does not run this. The code was developed in `scratch/kirk`
(2026-11-21 to 2026-11-22) and copied here unchanged apart from paths and defaults.

No gmsh meshing is used anywhere. gmsh only *reads* the STEP in `cad/` to measure it.

## Running

```bash
./run.sh                      # full chain: about 20 min on a 72-core machine
FROM=volute ./run.sh          # restart at a stage
PAR=0 ./run.sh                # everything in sequence (about 1 h)
MEM=40G CORE_NZ=64 ./run.sh   # memory cap per step, finer core (dz ~25 mm)
NOCAP=1 ./run.sh              # no systemd memory scope (e.g. on a server without user systemd)
```

Each step runs inside `systemd-run --user --scope -p MemoryMax=$MEM` when user systemd is
available. An uncapped run of the larger steps can exhaust a small machine. Peak memory is about
20 GB, in assembly.

With `PAR=1` (the default), independent work runs concurrently:
- **Bottom levels:** the top level runs first (level AB copies it), then the other 13 levels at once.
  These take about 2–3 min each, mostly single-threaded smoothing and untangling.
- **Bands:** all 13 at once.
- **Core and top head:** these run in the background beside the bottom → lower head → downcomer →
  volute → inlet chain.

Each step writes `logs/<stage>.log`.

Stage names (for `FROM`): `section core top bottom levels bands head vanes L2 downcomer volute inlet pocket assemble`.
Each stage writes a checkpoint `.pkl` next to its script, and the next stage reads it. Any stage can
therefore be rerun alone with its script, for example `cd p4 && python inlet2.py`.

Re-measuring the CAD (optional; needs the STEP and gmsh):

```bash
cd cad
MSRE_STEP=/path/core_fluid_cleanup_v5.STEP python analyze.py    # -> cad/faces.pkl, curves.pkl
MSRE_STEP=/path/core_fluid_cleanup_v5.STEP python fit_bores.py  # -> p4/bore_axes.npy
MSRE_STEP=/path/core_fluid_cleanup_v5.STEP python fit_pipe.py   # -> p4/pipe_axis.npy
```

`p1/lattice.pkl` is the per-cell classification of the 50.8 mm graphite lattice: full or cut cell,
and the bays at the edge. It was produced in an early slicing pass of the STEP that is not part of
this package.

## Reproducibility

Floating-point results differ slightly between machines (numpy and BLAS builds). The iterative
smoothers and untanglers amplify those last-digit differences, so the checkpoints are not
bit-identical across machines. They already differ at the first bottom level. The fragile steps
therefore correct themselves:
- **Vane straightening** (`vanepos2.py`): the local untangle retries with wider zones and stricter
  targets until the 2D section is valid.
- **Lower head** (`head3d.py`): adds L2b layers (2 → 3 → 4) until the morph from the vane layout back
  to the band layout has no inverted or twisted hex.

The reference run (2,447,348 hexes) needed 3 L2b layers; the development run needed 2.

`p4/topo.pkl` is shipped frozen. `p4/topo.py` built it, but `p4/levels.py` was changed afterwards
(rim-post clamping), and the current pair no longer regenerates that topology. `topo.py` is kept as
the record of how it was made.

## Domain and decomposition

| region | z (mm) | made by |
|---|---|---|
| top head: dome, outlet pipe, side branch | 2070 to 2708 | `p5/head_top.py`, `p5/skin.py` |
| side region under the dome rim | 1947.7 to 2070 | `p5/pocket.py` |
| core: channels and stringer-top pyramids | 411.3 to 2070 | `p1/build_section.py`, `p1/upper.py`, `p1/bores.py` |
| bottom lattice bands A..KN (pins, slots, plates, posts, grid, rim) | 206.5 to 411.3 | `p4/topo.py`, `p4/levels.py`, `p4/positions.py`, `p4/bands.py` |
| lower head with 48 anti-swirl vanes | 25.4 to ~299 | `p4/head.py`, `p4/vanes2.py`, `p4/vanepos2.py`, `p4/head3d.py` |
| downcomer and 84 inlet bores | 299 to 1889 | `p4/dc3.py`, `p4/tilt.py` |
| inlet volute and tangential inlet pipe | 1682 to 1886, R 761 to 1068 | `p4/volute.py`, `p4/inlet2.py` |
| assembly, tags, `.re2` | | `p4/assemble.py`, `p4/hexfix.py` |

Every block is built by lofting the same 2D section between levels. Where neighbouring blocks share
a face, both sides evaluate the same function at the same points. That is how the weld in
`hexmesh.merge` stays conformal at order 2: it refuses high-order edges that differ by more than
about 1.5e-5.

## Stages

### p1: core

- `build_section.py`, then `smooth_bores.py`: the global 2D section of the core (about 22.6k quads). Each 50.8 mm stringer
  cell uses one template: 4 pyramid-face triangles meeting at an apex node, ridge nodes on the
  diagonals, and fuel-channel stadia centred on the joints. Cut cells, bays, thimble rings and the
  graphite-to-can gap are included. `smooth_bores.py` then smooths the thimble and basket bore rings
  with the toolkit smoother. Built with `q2d.Sec`, a small 2D quad builder with exact
  curved edges: `arc`, composite curves, and an `"fn"` edge that evaluates a callable.
- `upper.py ORDER NZ`: lofts the core from z 411.29 with `NZ` layers through the channel zone. The
  side walls of the `chan` quads are tagged `channel` in every core zone (the tall cells keep their channels above 1986); the graphite-to-can gap stays `wall`. It
  then lofts the stringer-top zones (45° chamfers, pyramids, post tips to 2070) between per-node
  height fields. `bores.py` builds the 4 thimble and basket bores.

### p3 + p4: bottom lattice zone (bands)

- `p3/*`: the band-A section library: one tile per stringer cell type (`bsec`, `rules`, `secA`,
  `footprint`, `cells_A`, `edgeA`, `selA`, `wallA`). `p4/topo.py` assembles the single global
  bottom section (`topo.pkl`).
- `p4/levels.py` / `positions.py`: node positions at each of 14 levels (A/B ... KN). Each family
  of nodes moves to its band's exact features: slot edges, plate and bar edges, pin, post and plug
  rings. Validity is checked by **corner** scaled Jacobian, because a concave quad passes an area
  check. Bad patches are repaired by the greedy untangler (`q2.untangle`, `ab_fix`).
- `p4/bands.py`: each band lofts between two levels with its own fluid mask on the one section, so
  bands are conformal with each other by construction. The **rim** (`RIM=1`) adds two radial rows
  outside the R692.15 wall (to R701.675 and R704.85). Per band:
  - **B:** the y-slot ends plus the 3.175 mm gap against the can.
  - **C:** the open rim annulus.
  - **F1–F3:** the groove R692.15–704.85.
  - **C/D/E:** the R685.8–692.15 annulus, approximated by the elements touching the wall.

### p4: lower head and vanes

- `head.py`: the bottom section continued by an annulus R692 → 711 → 725 → 738. The annulus
  changes from 1088 to 364 columns through a ring of 3:1 templates.
- `vanes2.py`: each of the 48 vanes is selected as the section quads whose centroid lies in the
  exact 12.7 mm strip. Single-quad toggles remove quads with 3 outline edges or two opposite ones.
  Single-quad turns of the outline are pillowed.
- `vanepos2.py`: straightens the vane outlines onto the exact vane lines, in four steps.
  1. Outline nodes are projected with monotone spacing.
  2. One length-weighted harmonic step follows.
  3. Pillow inner nodes are placed in each degenerate turn quad.
  4. A local untangle runs with outline nodes sliding on their line.

  The outlines end exact to 1e-13.
- `head3d.py`:
  - **L2a**: head shell to vane top, with the vanes removed (the fins are welded to the shell).
  - **L2b**: 2 layers morphing back to the band positions. Invalid 2D blends between them are
    untangled before lofting (`valid_blend`).
  - **Pillow cap**: transition to the unpillowed KN band. The exact dip field (`DZ=exact`) is
    required for fold-free order-2 cap hexes.

### p4: downcomer, bores, volute, inlet

- The 84 bores are **tilted 30.6° tangentially**, all with horizontal axes. `tilt.py` maps the bore
  cross-section (u, v) to either wall by sliding along the axis.
- `dc3.py`: a (θ, z) structured downcomer on the lower head's 364 columns. Each hole block is an
  O-grid on the *exact* wall opening. The bores slide along their axes, and their walls lie on the
  CAD cylinders to 1e-12.
- `volute.py`: a 24 mm transition band carries the hole O-grids into a plain grid through two
  pillow-cap stages. It is built in the downcomer's unrolled coordinates. The map to the volute
  wall is the exact slide inside each opening and a uniform 1.152° rotation away from the openings.
  Then the D section (core block plus C-ring) is revolved.
- `inlet2.py`: the inlet pipe (r 64.135) enters tangentially through a long oval window in the
  torus. The D section's outer 2 layers are rebuilt as a shell over (θ, tube angle) with one
  O-grid hole whose ring is the pipe/torus trim curve. The same two-stage transition is used.
  The window disk slides along the pipe axis to a cap plane and is extruded to the end plane.

### p5: top head

- `head_top.py`: an inner volume that does not touch the CAD.
  1. The z = 2070 cap section is lofted to an inner dome.
  2. An outlet column is extruded up to the top plate, from section quads selected and smoothed in
     polar coordinates (`top_shift`).
  3. A branch duct runs along +x to the outlet plane.

  Quads made curved-invalid by the warp are straightened (`straighten_bad`).
- `skin.py`: one layer from every free face onto the CAD surfaces. Each stored node (corner, edge
  and face GLL node) gets a target from the classes of the faces it touches: dome, rim, pipe, duct
  and top. Shared edges therefore land on intersection curves. `hexfix.straighten_bad` is the last
  resort for any element still folded.
- `pocket.py`: the region between the core can (R711.2) and the vessel (R735.42) from the floor
  z 1947.74, plus the ring above the can top, up to the dome rim. It is a revolved (r, z) section on
  the core outline's 424 θ nodes. Near the knuckle it uses a 3-quad triangle split, so no quad has
  two edges on one arc.

### p4: assembly

- `assemble.py` merges all parts with `clear_seam_tags=True` and runs `hexfix`.
- Face tags: inlet and outlet are set geometrically on their end planes; `channel` is kept from the
  core; everything else is wall.
- It writes `.pkl`, `.re2` (with `boundaryID`) and `.vtu` (`--vtu`).
- It prints `hexmesh.report` before writing the `.re2`: quality (curved and corner), validity, tags
  and topology, including improper element pairs (two hexes meeting in anything but one shared
  point, line or face; e.g. two faces sharing two lines, which nekRS reports as `face_check failed`).
- `re2check.py FILE.re2` reruns parRSB's own checks on a written `.re2` (from any source) and prints
  the largest safe `connectivityTol`.
- `plot_folds.py MESH` plots any curved-folded elements: their curved edges against the corner hex,
  and the Jacobian field on slices.

## Settings

The values marked *as run* are what produced the mesh described above.

| variable | default | meaning |
|---|---|---|
| `ORDER` | 2 | polynomial order of every block. Keep ≤ 2: at order 3 the curved folds return. |
| `CORE_NZ` | 32 | layers in the core channel zone (dz ≈ 49 mm) |
| `RIM` (bands) | 1 | mesh the bottom rim region |
| `NB` (head3d) | 2 | L2b layers. 1 gives 121 twisted faces. |
| `HC`, `DIP`, `DZ` (head3d) | 3, 0.5, exact | pillow-cap thickness, dip fraction, dip field |
| `CLIP` (inlet2) | 0.8 | fraction of the window half-length kept; the grazing tips are clipped |
| `DIPA`, `LAMA` (inlet2) | 0.05, 0.9 | stage-A dip and per-axis square scale; both needed for a valid stage A |
| `T_CAP` (inlet2) | -490 | pipe-axis position of the slide plane |
| `POCKET_DZ`, `PHI_W` (pocket) | 14, 22 | annulus row height; wall node below the cap corner (deg) |
| `NOFIX` (assemble) | unset | skip the `hexfix` pass |

## Compromises (geometry not modelled exactly)

- **Bottom zone:**
  - Rods in band F2 dropped.
  - Rim posts below band A virtual, with their wall-cut lens made fluid.
  - Band-A T-cell pins at r16.51.
  - The 8 sliver bay tips filled as graphite.
  - The D/E rim annulus has variable width.
  - Central cross and plugs simplified (see `scratch/kirk/p2/lower_zone.md` for the CAD map).
- **Edge contacts:** about 350 boundary edges are shared by four faces. These are true edge contacts
  of fluid regions, mostly where the E rim annulus meets the F groove only along R692.15 at the
  plates. They are not holes.
- **Vanes:** exact in plan. Vane tips inside the downcomer dropped.
- **Inlet pipe:** the window's two tangential tips are clipped (`CLIP`); the pipe radius there
  drops to 45 mm and is blended back by the cap plane.
- **Side region:** the can top is at the core level 2014.48 (CAD 2016.25). The bracket frame and
  the rim screw holes are not modelled.
- **Top head:** the thimbles use the core radii (27.05 / 30.14) instead of 30.2.
- **`.re2`:** the file is corner-only, so curved boundaries are faceted in the solver file. The
  `.vtu` keeps order 2.
