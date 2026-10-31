"""Hex smoothing of a cylinder whose wall is free: ``free`` vs ``tangent`` vs ``slide``.

The mesh is an O-grid disc (radius 1, 16 sectors) extruded 1.0 along z in 4 layers.  The
perturbed input keeps it exactly a cylinder: interior vertices move in 3-D, wall vertices
along the circle (angle) and in z, cap vertices within their cap plane, rim vertices along
the rim (all by up to 0.25 of a cell).  Three runs, all with the *same Laplace-only*
schedule, ``laplace(passes=20)``, which is where ``free`` and ``tangent`` differ most:

    (a) boundary="free"     -- the raw averaged move, wall vertices included
    (b) boundary="tangent"  -- the move projected onto the local tangent plane (corners frozen)
    (c) boundary="slide"    -- wall vertices held on the input wall's faces
    (d) "exact wall": boundary="slide" plus OnSurface(wall, cylinder) and OnCurve(rims, circle),
        where the targets are the true cylinder and its two rim circles, defined here

Per run: volume change against the input (``hexmesh.volume``, the signed trilinear hex
volumes summed; the input's own volume is the faceted polyhedron, a little under
pi R^2 L), the largest wall distance |hypot(x, y) - R| over the input wall vertices, the
largest distance of the non-rim wall vertices to the input's *faceted* wall (``facet``; the sagitta of
one facet is printed for scale), the largest cap off-plane distance, scaled Jacobian min / mean and the inverted count.

    PYTHONPATH=. python examples/smoother/hex/03_free_boundary.py

Writes ``03_free_boundary_{input,free,tangent,slide,exact}.vtu`` to the current directory and
leaves ``mesh`` (the slide result), ``rows``, ``meshes`` and ``sj`` in the namespace.
"""

import numpy as np

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.hexmesh import smoother as hs

# -- parameters --------------------------------------------------------------
R, L = 1.0, 1.0
SECTORS, LAYERS = 16, 4
PERTURB = 0.25               # fraction of a cell
SEED = 0
LAPLACE = [("laplace", {"passes": 20})]

# -- the cylinder and its perturbed input --------------------------------------
cyl = hexmesh.extrude(quadmesh.ogrid(linemesh.circle(R, SECTORS), 4, 2), L, LAYERS)
exact = cyl.points.copy()
rho = np.hypot(exact[:, 0], exact[:, 1])
wall = np.isclose(rho, R)
cap = np.isclose(exact[:, 2], 0.0) | np.isclose(exact[:, 2], L)
rim = wall & cap
wall_only, cap_only = wall & ~cap, cap & ~wall
inner = ~wall & ~cap
print("cylinder: %d hexes; vertices: %d interior, %d wall, %d cap (non-rim), %d rim"
      % (cyl.n_hexes, inner.sum(), wall_only.sum(), cap_only.sum(), rim.sum()))

h_arc, h_z, h_xy = 2 * np.pi * R / SECTORS, L / LAYERS, 0.2
rng = np.random.default_rng(SEED)
u = rng.uniform(-PERTURB, PERTURB, exact.shape)
P = exact.copy()
P[inner] += u[inner] * np.array([h_xy, h_xy, h_z])                       # interior: 3-D
ang = np.arctan2(exact[:, 1], exact[:, 0])
for m, dz in ((wall_only, True), (rim, False)):                          # wall: angle (+ z); rim: angle
    a = ang[m] + u[m, 0] * h_arc / R
    P[m, 0], P[m, 1] = R * np.cos(a), R * np.sin(a)
    if dz:
        P[m, 2] += u[m, 1] * h_z
P[cap_only, :2] += u[cap_only, :2] * h_xy                                # cap: within its plane
before = hexmesh.HexMesh.from_corners(P, cyl.corners, cyl.element_tags)


def per_hex_sj(m):
    return hs.corner_metrics(m.points, m.corners).sj.min(axis=1)


def off_wall(m):
    return float(np.abs(np.hypot(m.points[wall, 0], m.points[wall, 1]) - R).max())


# the input's faceted wall: its wall quads (all four corners on the wall) split into triangles
fq = hs.boundary_faces_outward(before)
fq = fq[wall[fq].all(axis=1)]
facets = hs.surface_from_trimesh(before.points, np.concatenate([fq[:, [0, 1, 2]], fq[:, [0, 2, 3]]]),
                                 name="input wall")


def off_facets(m, which=None):
    """Largest distance of the input wall vertices (rims excluded unless ``which`` says so) to the
    input's faceted wall.  A rim vertex slides on a spline through the input rim vertices, which bulges
    off the chords, so rims are read separately."""
    p = m.points[wall_only if which is None else which]
    return float(np.linalg.norm(facets.project(p) - p, axis=1).max())


def cylinder_surface(radius):
    """The exact cylinder about the z axis: closest point and outward normal."""
    def radial(p):
        q = np.array(p, float)
        q[:, 2] = 0.0
        n = np.linalg.norm(q, axis=1, keepdims=True)
        return q / np.where(n > 0, n, 1.0)

    return hs.Surface(lambda p: np.column_stack([radius * radial(p)[:, :2], p[:, 2]]), radial, "cylinder")


def off_cap(m):
    return float(np.abs(m.points[cap, 2] - before.points[cap, 2]).max())


v_in, v_exact = hexmesh.volume(before), np.pi * R**2 * L
q0 = hs.quality(before.points, before.corners)
print("input: volume %.5f (exact cylinder %.5f, faceted %.2f %% under) | inverted %d | sj min %.4f mean %.4f"
      % (v_in, v_exact, 100 * (v_exact - v_in) / v_exact, q0.n_inverted, q0.sj_min, q0.sj_mean))
print("       wall distance %.2e, cap off-plane %.2e (the input is exactly on the cylinder and its caps)"
      % (off_wall(before), off_cap(before)))

SAGITTA = R * (1 - np.cos(np.pi / SECTORS))
cylinder = cylinder_surface(R)
rim_lo = np.flatnonzero(rim & np.isclose(exact[:, 2], 0.0))
rim_hi = np.flatnonzero(rim & np.isclose(exact[:, 2], L))
exact_wall = [hs.OnSurface(np.flatnonzero(wall_only), cylinder),
              hs.OnCurve(rim_lo, hs.circle((0, 0, 0), R)), hs.OnCurve(rim_hi, hs.circle((0, 0, L), R))]
variants = (("free", "free", []), ("tangent", "tangent", []),
            ("slide", "slide", []), ("exact", "slide", exact_wall))
meshes = {"input": before}
rows = {"input": dict(q=q0, volume=v_in, volume_change_pct=0.0, wall_dist=off_wall(before), facet=off_facets(before),
                      cap_off_plane=off_cap(before))}
print()
for label, boundary, cons in variants:
    r = hs.resolve(before, cons, boundary=boundary)
    sched = [("laplace", {"passes": 20})]
    out, logs = hs.run_schedule(before, r, sched)
    q = logs[-1].after
    v = hexmesh.volume(out)
    meshes[label] = out
    rows[label] = dict(q=q, volume=v, volume_change_pct=100 * (v - v_in) / v_in,
                       wall_dist=off_wall(out), facet=off_facets(out), facet_rim=off_facets(out, rim), cap_off_plane=off_cap(out),
                       drift_vs_input_wall=float(np.abs(out.points[wall] - before.points[wall]).max()))
    d = rows[label]
    print("%-8s volume %.5f (%+.2f %% vs input, %+.2f %% vs exact) | wall dist max %.2e | facet dist %.2e | cap off-plane %.2e"
          % (label, v, d["volume_change_pct"], 100 * (v - v_exact) / v_exact, d["wall_dist"],
             d["facet"], d["cap_off_plane"]))
    print("         inverted %d | sj min %.4f mean %.4f" % (q.n_inverted, q.sj_min, q.sj_mean))

# the slide wall is the faceted input wall: its distance to the exact cylinder is the chord error
print("\n(facet distance is over the %d wall vertices off the rims; rims slide on a spline through the input rim "
      "vertices, which bulges off the chords: slide rim %.1e off the facets)" % (wall_only.sum(), rows["slide"]["facet_rim"]))
print("facet sagitta R(1 - cos(pi/%d)) = %.2e; slide wall vertices sit %.1e off the input facets and %.2e off the "
      "exact cylinder (chord error only)" % (SECTORS, SAGITTA, rows["slide"]["facet"], rows["slide"]["wall_dist"]))
print("exact wall: %.1e off the cylinder, %.1e off the input facets; rims are claimed by OnCurve and wall vertices "
      "by OnSurface, caps slide" % (rows["exact"]["wall_dist"], rows["exact"]["facet"]))
print("slide wall vertices moved along the wall by up to %.3f (max displacement from the input position)"
      % rows["slide"]["drift_vs_input_wall"])

sj = {k: per_hex_sj(m) for k, m in meshes.items()}
for label, m in meshes.items():
    writer.to_vtu(m, "03_free_boundary_%s.vtu" % label)

sagitta = SAGITTA
mesh = meshes["slide"]
