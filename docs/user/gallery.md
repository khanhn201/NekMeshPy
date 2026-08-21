# Gallery

Rendered on client-side with [vtk.js](https://kitware.github.io/vtk-js/).
This live viewer only render elements linearly.

## carotid

Vessel surface pipeline: seams cut into legs, O-grid legs, lofted and merged.

```{mesh-viewer} carotid
```

[`examples/carotid.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/carotid.py)

## circular_pipe

O-grid circular pipe.

```{mesh-viewer} circular_pipe
```

[`examples/circular_pipe.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/circular_pipe.py)

## circular_pipe_tjunction

Analytic pipe T-junction.

```{mesh-viewer} circular_pipe_tjunction
```

[`examples/circular_pipe_tjunction.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/circular_pipe_tjunction.py)

## quadrant_pipe_tjunction

Welded small-branch T-junction, built from quadrant blocks.

```{mesh-viewer} quadrant_pipe_tjunction
```

[`examples/quadrant_pipe_tjunction.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/quadrant_pipe_tjunction.py)

## cob_tjunction

Unequal-radius T-junction with the branch cut straight through the main pipe, so
there is no hub to degenerate at a small radius ratio, and a boundary layer grown
outward over the wall.

```{mesh-viewer} cob_tjunction
```

[`examples/cob_tjunction.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/cob_tjunction.py)

## serpentine_pipe

One O-grid disc swept along a path.

```{mesh-viewer} serpentine_pipe
:height: 480px
```

[`examples/serpentine_pipe.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/serpentine_pipe.py)

## chimera

Several two-manifold units chained along one axis, alternating connector pipe;
cob T-junctions throughout, a boundary layer over the whole fluid wall, and a
solid jacket attached to the finished tube.

```{mesh-viewer} chimera
:height: 560px
```

[`examples/chimera.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/chimera.py)

## chimera_full

The full manifold: risers, T1/T2 junction chains, and a serpentine coil feeding
`chimera.py`'s two ports.

```{mesh-viewer} chimera_full
:height: 560px
```

[`examples/chimera_full.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/chimera_full.py)


## backward_facing_step

Backward-facing step channel.

```{mesh-viewer} backward_facing_step
```

[`examples/backward_facing_step.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/backward_facing_step.py)

## flow_past_cylinder

External flow around a circular cylinder.

```{mesh-viewer} flow_past_cylinder
```

[`examples/flow_past_cylinder.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/flow_past_cylinder.py)

## flow_past_half_cylinder

External flow over a half-cylinder bump.

```{mesh-viewer} flow_past_half_cylinder
```

[`examples/flow_past_half_cylinder.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/flow_past_half_cylinder.py)

## flow_past_sphere

External flow around a sphere (cubed-sphere far field).

```{mesh-viewer} flow_past_sphere
```

[`examples/flow_past_sphere.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/flow_past_sphere.py)

## flow_past_hemisphere

External flow around a hemisphere on the ground.

```{mesh-viewer} flow_past_hemisphere
```

[`examples/flow_past_hemisphere.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/flow_past_hemisphere.py)

## rod_bundle

61-rod hexagonal fuel bundle with a helical wire wrap: solid rods, coolant, and the
duct wall as three conjugate regions. The wire is a bulge in the rod's own wall profile
rather than a body of its own, and the shared cell walls slide out of its way.

```{mesh-viewer} rod_bundle
```

[`examples/rod_bundle.py`](https://github.com/khanhn201/NekMeshPy/tree/main/examples/rod_bundle.py)
