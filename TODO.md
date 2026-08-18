Local, immediate simple tasks that can be done right away, not to be commited.
- [x] 1) Need a quadmesh and linemesh .morph.offset to create an offseted surface based on
surface tangent. The offset vector
at each point should be average of that point's tangent across
all elements sharing that point. Ofcourse high order nodes follow the same rule.
Tangent and derivative should be calculated based on the underlying GLL nodes and
functions.
The point of this is for skinning a mesh, which generates a thin, perpendicular 
boundary layers near a surface. These thin skin layers can then be loft to generate a hexmesh
or quadmesh.

- [ ] 2) API for attaching to known surface without needing to merge (continue loft, solid-fluid in chimera, bifurcation interfaces)
- [ ] 3) New Tjunction topology
- [ ] 4) Plottings utils or simply vtk export the full scene
- [x] 5) Rendered, live or simplified mesh showcase in docs
- [ ] 6) Update all staled readme and docs
- [ ] 7) Mapping of grid on parametric curves
- [ ] 8) TriMesh.isosurface should return LineMesh
- [ ] 9) TetMesh.isosurface should return TriMesh
- [ ] 10) Offload marching algorithm to skimage or vtk?
- [ ] 11) Generalize Path in Sweeping to be fully 3D
- [ ] 12) Option to select backends for marching?
- [ ] 13) TJunctions should return only the interface then user can loft on their own
    - [ ]  For quadrant Tjunction, additionally the hexmesh on the sides
- [ ] 14) APIs to handle conduction solves
    - [ ] Inputs: Dirichlets faces, Dirichlets values
- [x] 15) Names variables to plurals: quads, hexes
- [ ] 16) Check element overlap
- [x] 17) Get rid of all apply_smoothing in quad
- [ ] 18) p-refine/h-refine
