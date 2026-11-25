# msre: conformal hex mesh of the MSRE fuel-salt volume

`./run.sh` rebuilds a conformal, Nek5000-valid hex mesh of the Molten Salt Reactor Experiment
(MSRE) fuel-salt fluid domain, `core_fluid_cleanup_v5.STEP` (one OCC solid, 14,113 faces, mm,
axis = z, z 25.4 to 2708). The STEP is not shipped. The pipeline starts from four shipped inputs: the
CAD-derived `p1/lattice.pkl`, `p4/bore_axes.npy` and `p4/pipe_axis.npy`, and `p4/rim_exclude.npy` (the
8 lattice quads the bottom rim leaves out, see *Reproducibility*).

Output: `finish/msre.re2`, `finish/msre.vtu` and `finish/msre.pkl` (a linear `HexMesh`).

No pillows anywhere: the bottom rim is a strip lofted from the lattice onto the wall, and the 48
lower-head vanes are a skin lofted onto the vane surfaces.

- **Size and validity:** 2,294,549 hexes on the reference run, core `NZ=32`. Every check in
  `hexmesh.report` passes: zero inverted corners, zero twisted faces, **zero overlapping element
  pairs** (the exact trilinear test), and clean when sampled at the solver's order N=7 (min
  +0.012; nekRS's Jacobian ratio min 0.0034). Min corner scaled Jacobian 0.025, shortest edge 0.133 mm.
- **Connectivity:** one component, watertight (no pinch edges), conformal (no hanging points, no
  improper element pairs). All boundary faces are tagged. The written `.re2` passes nekRS/parRSB's
  `parCon` checks (`p4/re2check.py`); it stays unambiguous for `connectivityTol` below about 0.27
  (the default is 0.2).
- **Boundary rows** in the `.re2`, as `cbc` / `boundaryID`:

  | name | cbc | boundaryID | faces |
  |---|---|---|---|
  | inlet | `v  ` | 1 | 332 |
  | outlet | `O  ` | 2 | 140 |
  | wall | `W  ` | 3 | all others (649,246) |
  | channel | `W  ` | 4 | 431,664 (fuel-channel side walls in the core, incl. the tall cells' channels up to the post tops) |

This is a multi-script pipeline, not a flat example. `tests/test_examples.py` runs only the
top-level `examples/*.py`, so it does not run this. The code was developed in `scratch/kirk`
(2026-11-21 to 2026-11-25) and copied here unchanged apart from paths and defaults.

No gmsh meshing is used anywhere. gmsh only *reads* the STEP in `cad/` to measure it.

## Running

```bash
./run.sh                      # full chain (see below for timing) on a 72-core machine
FROM=polish ./run.sh          # restart at a stage
PAR=0 ./run.sh                # everything in sequence
MEM=96G CORE_NZ=64 ./run.sh   # memory cap per step, finer core (dz ~25 mm)
NOCAP=1 ./run.sh              # no systemd memory scope (e.g. on a server without user systemd)
```

Each step runs inside `systemd-run --user --scope -p MemoryMax=$MEM` when user systemd is
available. An uncapped run of the larger steps can exhaust a small machine. Peak memory is about
25 GB (the finishing steps and the full report, which fork worker processes; the default `MEM` is 64G).

With `PAR=1` (the default), independent work runs concurrently:
- **Bottom levels:** the top level runs first (level AB copies it), then the other 13 levels at once.
  These take about 2–3 min each, mostly single-threaded smoothing and untangling.
- **Bands:** all 13 at once.
- **Core and top head:** these run in the background beside the bottom → lower head chain; the
  downcomer → volute → inlet chain runs in the background beside the lower head.

Each step writes `logs/<stage>.log`.

Stage names (for `FROM`): `section core top topology bottom levels bands head downcomer L2 untangle
l2overlap pocket assemble pinch smooth polish overlap revert export`.
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
therefore correct themselves, and every one of them is guarded rather than trusted:
- **Lower head** (`vloft.py`): adds L2b layers (2 → 3 → 4) until the morph from the vane layout back
  to the band layout has no inverted or twisted hex. The few skin hexes the loft leaves folded (3–6
  on the reference run) are untangled locally (`vskin_twist.py`), and any overlap is repaired with the
  exact test (`vskin_ovl_fix.py`).
- **Finishing**: the polish keeps a move only if no face twists further, no edge shrinks below 70%
  and nothing starts to overlap; `revert_sampled.py` then puts back any node whose move folds an
  element between its corners at the solver's order.

`p4/rim_exclude.npy` lists the 8 lattice quads (by their top-level centroids) the bottom rim leaves
out: each one came out folded at some level when it was kept. It was found by iterating `topo2.py` →
all 14 levels → unselecting the quads that fold, until clean (one pass with this list).

## Domain and decomposition

| region | z (mm) | made by |
|---|---|---|
| top head: dome, outlet pipe, side branch | 2070 to 2708 | `p5/head_top.py`, `p5/skin.py` |
| side region under the dome rim | 1947.7 to 2070 | `p5/pocket.py` |
| core: channels and stringer-top pyramids | 411.3 to 2070 | `p1/build_section.py`, `p1/upper.py`, `p1/bores.py` |
| bottom lattice bands A..KN (pins, slots, plates, posts, grid, rim) | 206.5 to 411.3 | `p4/topo2.py`, `p4/levels.py`, `p4/positions.py`, `p4/bands.py` |
| lower head with 48 anti-swirl vanes | 25.4 to ~299 | `p4/head.py`, `p4/vloft.py`, `p4/vskin_twist.py`, `p4/vskin_ovl_fix.py` |
| downcomer and 84 inlet bores | 299 to 1889 | `p4/dc3.py`, `p4/tilt.py` |
| inlet volute and tangential inlet pipe | 1682 to 1886, R 761 to 1068 | `p4/volute.py`, `p4/inlet2.py` |
| assembly, tags | | `p4/assemble.py` |
| finishing: watertight, smoothing, polish, checks, `.re2` | | `finish/*` |

Every block is built by lofting the same 2D section between levels. Where neighbouring blocks share
a face, both sides evaluate the same function at the same points, so the weld in `hexmesh.merge`
is conformal.

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
  `footprint`, `cells_A`, `edgeA`, `selA`, `wallA`).
- `p4/topo2.py` assembles the single global bottom section (`topo2.pkl`) and its **lofted rim**. It keeps
  every lattice quad that stays 2 mm inside the R692.15 wall at all 14 levels, cleans the outline
  (no holes or pinches, star-shaped), and lofts one radial strip from each outline node onto the
  wall. No clipping, no pillows. Parts of a pin's O-grid may be cut: near the rim the template rings
  are not physical, so they are held on their circle only for real pins and posts (`levels.ring_held`,
  within 80 mm of the wall); the interior keeps its old geometry exactly.
- `p4/levels.py` / `positions.py`: node positions at each of 14 levels (A/B ... KN). Each family
  of nodes moves to its band's exact features: slot edges, plate and bar edges, pin, post and plug
  rings. Validity is checked by **corner** scaled Jacobian, because a concave quad passes an area
  check. Bad patches are repaired by the greedy untangler (`q2.untangle`, `ab_fix`). Only a 15 mm
  band at the wall is smoothed; the lofted wall nodes keep monotone angles (`loft_angles`).
- `p4/bands.py`: each band lofts between two levels with its own fluid mask on the one section, so
  bands are conformal with each other by construction. The **rim** (`RIM=1`) adds two radial rows
  outside the R692.15 wall (to R701.675 and R704.85). Per band:
  - **B:** the y-slot ends plus the 3.175 mm gap against the can.
  - **C:** the open rim annulus.
  - **F1–F3:** the groove R692.15–704.85.
  - **C/D/E:** the R685.8–692.15 annulus, approximated by the elements touching the wall.

### p4: lower head and vanes

- `head.py`: the bottom section continued by an annulus R692 → 711 → 725 → 738. The annulus
  changes from the rim's 1136 wall nodes to 380 columns through a ring of 3:1 templates; the template
  nodes in that ring are free (the head's top face between the bands and the downcomer is a wall) and
  are relaxed so no template quad is a sliver.
- `vloft.py`: the vanes, as a lofted skin.
  1. **Select:** per vane, the section cells within `DELTA` (2.5 mm) of the vane footprint.
  2. **Smooth the stair:** guarded, vectorised Laplacian sweeps on the corridor outline and the
     section within 40 mm; a move is shortened until no quad (nor its halfway blend back to the
     band positions) drops below 0.1, no edge below 0.5 mm, and no outline segment comes within
     `DELTA` of the vane.
  3. **Map:** every outline node onto the vane footprint in order (closest point, isotonic).
  4. **Loft:** L2a (shell to `HSK` = 5 mm above the vane top, corridors removed) and L2b (back to the
     band positions), then one skin layer from the carved surface onto the rounded vane, relaxed
     along the vane (`SKSM` guarded sweeps).
- `vskin_twist.py`: local, twist-aware untangle of the few skin hexes the loft folds.
- `vskin_ovl_fix.py`: overlap repair with the exact trilinear test (vane nodes slide on the vane,
  shell nodes on the shell).
- `head3d.py`, `vanes.py`: section helpers used by `vloft.py`.

### p4: downcomer, bores, volute, inlet

- The 84 bores are **tilted 30.6° tangentially**, all with horizontal axes. `tilt.py` maps the bore
  cross-section (u, v) to either wall by sliding along the axis.
- `dc3.py`: a (θ, z) structured downcomer on the lower head's columns. Each hole block is an
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

- `assemble.py` merges all parts (linear, `LINEAR=1`) with `clear_seam_tags=True`.
- Face tags: inlet and outlet are set geometrically on their end planes; `channel` is kept from the
  core; everything else is wall.
- `re2check.py FILE.re2` reruns parRSB's own checks on a written `.re2` (from any source) and prints
  the largest safe `connectivityTol`.

### finish: watertight, quality, checks

- `pinch_fix.py`: the bands' fluid masks leave about 340 **pinch edges** -- two fluid cells meeting
  only along an edge, mostly on the R692.15 wall between bands. Each is removed by dropping the more
  exposed cell (336 hexes) and tagging what it uncovers `wall`, so the domain is watertight.
- `region_fix.py`: toolkit smoother (untangle, then shape) on local zones around cells below corner
  sj 0.15; a zone is kept only if its worst element improves and nothing twists or overlaps.
- `greedy_fix.py`: twist-aware greedy polish, interior nodes and nodes on flat walls; big zones are
  split into blocks so they run in parallel; two passes with shifted blocks. Guards: no face twists
  further, no edge shrinks below 70%, nothing starts to overlap.
- `p4/vskin_ovl_fix.py` again on the whole model, then `revert_sampled.py`: any node whose move folds
  an element between its corners at the solver's order N=7 goes back to its pre-polish position.
- `export_final.py`: prints `hexmesh.report` (every check, including overlap and the N=7 sampling)
  and writes `.re2` (with `boundaryID`), `.pkl` and `.vtu`. `bc_check.py` checks the `.re2`
  boundary rows (one per open face, none on an interior face, no coincident boundary faces);
  `qreg.py` prints corner quality and edge length per region.

## Settings

The values marked *as run* are what produced the mesh described above.

| variable | default | meaning |
|---|---|---|
| `ORDER` | 2 | order of the core, top head, bands and downcomer blocks; the assembly is linear (`LINEAR=1`) |
| `L2_ORDER` (vloft) | 1 | order the lower head is built at |
| `CORE_NZ` | 32 | layers in the core channel zone (dz ≈ 49 mm) |
| `RIM` (bands) | 1 | mesh the bottom rim region |
| `GMIN` (topo2) | 2.0 | lattice quads kept at least this far inside the wall, at every level |
| `RING_SITES_ONLY`, `RING_FREE_R` (levels) | 1, 80 | template rings freed only this close to the wall |
| `DELTA`, `HSK`, `ALPHA` (vloft) | 2.5, 5, 0.05 | corridor gap to the vane, room above the vane top, uniform-spacing blend |
| `SJMIN`, `EMIN`, `NSW` (vloft) | 0.1, 0.5, 60 | stair-smoothing guards and sweeps |
| `SKSM`, `SKF` (vloft) | 20, 0.15 | skin-surface smoothing sweeps and floor |
| `CLIP` (inlet2) | 0.8 | fraction of the window half-length kept; the grazing tips are clipped |
| `DIPA`, `LAMA` (inlet2) | 0.05, 0.9 | stage-A dip and per-axis square scale; both needed for a valid stage A |
| `T_CAP` (inlet2) | -490 | pipe-axis position of the slide plane |
| `POCKET_DZ`, `PHI_W` (pocket) | 14, 22 | annulus row height; wall node below the cap corner (deg) |

## Compromises (geometry not modelled exactly)

- **Bottom zone:**
  - Rods in band F2 dropped.
  - Rim posts below band A virtual, with their wall-cut lens made fluid.
  - Band-A T-cell pins at r16.51.
  - The 8 sliver bay tips filled as graphite.
  - The D/E rim annulus has variable width.
  - Central cross and plugs simplified (see `scratch/kirk/p2/lower_zone.md` for the CAD map).
- **Bottom rim:** the lattice stops 2 mm or more inside the R692.15 wall and one lofted strip fills
  the rest, so the strip is thin next to a pin and up to ~15 mm deep where cells were dropped.
  The 336 cells that only touched other fluid along an edge are dropped (`pinch_fix.py`).
- **Vanes:** 12.7 mm slabs with fully rounded top edges and ends (radius 6.35), ending at r ≈ 722
  where the vane becomes lower than its own thickness. Vane tips inside the downcomer dropped.
- **Inlet pipe:** the window's two tangential tips are clipped (`CLIP`); the pipe radius there
  drops to 45 mm and is blended back by the cap plane.
- **Side region:** the can top is at the core level 2014.48 (CAD 2016.25). The bracket frame and
  the rim screw holes are not modelled.
- **Top head:** the thimbles use the core radii (27.05 / 30.14) instead of 30.2.
- **`.re2`:** the file is corner-only, so curved boundaries are faceted in the solver file.
