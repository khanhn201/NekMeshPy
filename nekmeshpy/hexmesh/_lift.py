"""Fixed-arity ``HexMesh`` operations that raise a rung (delta +1).

``extrude`` sweeps one section straight; ``annulus`` fills between two index-paired
closed surfaces; ``from_grid`` builds a block from one structured point grid.  All
three are thin: they position profiles and hand them to
:func:`~nekmeshpy.hexmesh._assemble.loft`, which owns the index space, so the numbering
they expose is the loft's carried up unchanged.

Free functions bound onto :class:`~nekmeshpy.HexMesh` by ``hexmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``HexMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal

import numpy as np

from .._typing import (
    FloatArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..linemesh._assemble import _sweep_lattice, _sweep_path
from ..model import frames
from ..model.fields import reject_loop_caps, validate_layers
from ..quadmesh import QuadMesh
from ..quadmesh._lift import from_grid as quad_from_grid
from ..quadmesh._morph import blend as quad_blend
from ..quadmesh._morph import transform as quad_transform
from ..quadmesh._morph import translate
from ._assemble import _loft_evaluated, loft
from .hexmesh import _GRID_SIDES, _ORIGIN, _Z_AXIS, HexMesh


def extrude(
    section: QuadMesh,
    length: float,
    layers: int | FloatArray,
    *,
    axis: Vec3 = _Z_AXIS,
    origin: Point = _ORIGIN,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> HexMesh:
    """Sweep a single quad ``section`` a distance ``length`` along ``axis`` into
    a hex block.

    The section is translated rigidly along ``axis`` (its placement is
    preserved); ``origin`` shifts the whole block. ``layers`` is either an ``int``
    count of uniform layers or the normalized copy-plane positions in ``[0, 1]``,
    strictly increasing, last ``1``
    (:func:`validate_layers <nekmeshpy.model.fields.validate_layers>`);
    ``layers[0]`` is the near cap and ``layers.size - 1`` hex layers span it to
    ``1``. ``first_tag`` / ``last_tag`` name the caps. The straight special case
    of ``loft``.

    ``length`` and ``layers`` are positional-or-keyword, like ``path`` /
    ``fractions`` on the sibling :func:`sweep`: they are required, so making them
    keyword-only bought nothing. Every existing
    ``extrude(section, axis=..., length=..., layers=...)`` call still binds."""
    axis_u: Vec3 = np.asarray(axis, dtype=float)
    axis_u = axis_u / np.linalg.norm(axis_u)
    offsets = validate_layers(layers, "extrude layers") * float(length)
    # the sweep is a rigid translation, so it *is* the rung-preserving ``translate``:
    # every node -- corners, shared edge-interior, private quad-interior -- rides the
    # same vector (and ``origin`` shifts all three alike), because that map is itself
    # composed down the ladder.  Each slice reuses the section's B-rep verbatim (same
    # edge LineMesh connectivity, same per-quad edge indices / flips); only
    # coordinates move.
    placed = translate(section, np.asarray(origin, dtype=float))
    slices = [translate(placed, d * axis_u) for d in offsets]
    return loft(slices, first_tag=first_tag, last_tag=last_tag)

def annulus(
    inner: QuadMesh,
    outer: QuadMesh,
    radial: int | FloatArray,
    *,
    inner_tag: str = "",
    outer_tag: str = "",
) -> HexMesh:
    """Shell O-grid filling the region between an inner and an outer closed quad
    surface (e.g. a sphere inside a cubic far-field box).

    The two surfaces are paired by index: equal point count ``P`` and identical
    ``quads`` connectivity, with point ``p`` of ``inner`` joined radially to point
    ``p`` of ``outer``. ``radial`` is either an ``int`` count of uniform shell layers
    or the shell positions in ``[0, 1]``, strictly increasing
    (:func:`validate_layers <nekmeshpy.model.fields.validate_layers>`);
    ``radial[0]`` is the inner shell and the last is ``1``, so
    ``radial.size - 1`` shell layers blend inner -> outer directly in 3-D.

    Wall faces are tagged from the surfaces' per-quad ``element_tags`` (a closed
    surface has no free boundary edges): inner caps face 5, outer caps face 6. A
    non-empty scalar ``inner_tag`` / ``outer_tag`` overrides and names the whole
    wall."""
    radial = validate_layers(radial, "annulus radial")
    A: PointArray = np.asarray(inner.points, dtype=float).reshape(-1, 3)
    B: PointArray = np.asarray(outer.points, dtype=float).reshape(-1, 3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "annulus: inner and outer surfaces must have equal point counts "
            "(got %d, %d); build one from the other's points so they pair by "
            "index" % (A.shape[0], B.shape[0]))
    if not np.array_equal(inner.quads, outer.quads):
        raise ValueError(
            "annulus: inner and outer surfaces must share identical quad "
            "connectivity (they are paired by index)")
    if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
        raise ValueError("annulus: inner and outer surfaces touch or cross")

    # shell t is the straight-chord blend inner -> outer sharing inner's quads;
    # consecutive shells loft into hex layers.
    shells = quad_blend(inner, outer, radial)
    # wall tags from the surfaces' per-quad element_tags; scalar arg overrides
    inner_caps: str | StrArray = (
        inner_tag if inner_tag
        else (inner.element_tags.dense(inner.n_quads) if inner.element_tags else ""))
    outer_caps: str | StrArray = (
        outer_tag if outer_tag
        else (outer.element_tags.dense(outer.n_quads) if outer.element_tags else ""))
    return loft(shells, first_tag=inner_caps, last_tag=outer_caps)

def from_grid(
    P: PointArray,
    *,
    side_tags: Mapping[str, str] | None = None,
    element_tag: str = "",
    order: int = 1,
) -> HexMesh:
    """Build hexes from a structured point grid ``P`` ``(ni+1,nj+1,nk+1,3)``.
    ``side_tags`` maps side names (``x_min``/``x_max``/``y_min``/``y_max``/
    ``z_min``/``z_max``) to boundary names on the six outer sides; a side left out
    or mapped to ``NO_TAG`` emits no tag row. ``element_tag`` is written
    to every hex's ``element_tags``.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each hex carries ``(order+1)**3`` straight-sided (trilinear) GLL nodes.

    Built as a :meth:`loft` of the grid's **``k``-sections**: section ``k`` is
    the :meth:`QuadMesh.from_grid <nekmeshpy.quadmesh.QuadMesh.from_grid>` of the
    slab ``P[:, :, k, :]`` (itself a ``LineMesh`` loft), and the sweep runs
    ``k = 0..nk``.  Every tagged side rides a channel the rung below already has:
    the section's four ``edge_tags`` become the ``x_min`` / ``x_max`` / ``y_min`` /
    ``y_max`` swept side faces (section side ``s`` -> hex face ``s``, which is
    exactly the ``_GRID_SIDES`` Nek face numbering) and the sweep's caps the
    ``z_min`` / ``z_max`` ones; ``element_tag`` rides the section's per-quad tags.
    So corners, shared edges and shared faces all come out of the layer-by-layer
    B-rep assembly instead of a ``unique_edges`` re-derivation.

    **Ordering is the loft's, carried up unchanged** -- composing the rung below
    means accepting its numbering, so nothing is relabelled here.  The grid is
    numbered ``i`` fastest, ``k`` slowest: grid node ``(i, j, k)`` is point
    ``(k*(nj+1) + j)*(ni+1) + i`` and grid cell ``(i, j, k)`` is hex
    ``(k*nj + j)*ni + i``, i.e. ``points`` equals
    ``P.transpose(2, 1, 0, 3).reshape(-1, 3)`` -- *not* the ``P.reshape(-1, 3)``
    (``k``-fastest) order this factory used historically.  ``face_tags`` stays
    lexsorted by ``(element, face)``, so its row order follows the hex ids; each
    tagged row still names the same physical side."""
    P = np.asarray(P, dtype=float)
    tags = {s: n for s, n in (side_tags or {}).items() if n}
    for side in tags:
        _GRID_SIDES[side]        # reject an unknown side name (KeyError)
    # x/y sides are the section's own edges; z sides are the sweep's end caps.
    side_map = {s: n for s, n in tags.items() if _GRID_SIDES[s][0] == "side"}
    slices = [quad_from_grid(P[:, :, k, :], side_tags=side_map,
                             element_tag=element_tag, order=order)
              for k in range(P.shape[2])]
    # the loft *is* the result: its sweep-major numbering is carried up unchanged.
    return loft(slices, first_tag=tags.get("z_min", ""),
                last_tag=tags.get("z_max", ""))


def sweep(
    section: QuadMesh,
    path: Callable[[FloatArray], PointArray],
    fractions: FloatArray,
    *,
    origin: Point | Sequence[float],
    tangent: Callable[[FloatArray], PointArray] | None = None,
    orientation: Literal["transport", "fixed", "frenet"] = "transport",
    up: Vec3 | Sequence[float] | PointArray | None = None,
    twist: float = 0.0,
    close_twist: bool = True,
    normal: Vec3 | Sequence[float] | None = None,
    loop: bool = False,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> HexMesh:
    """A block swept from one ``QuadMesh`` ``section`` along the curve ``path`` -- a
    round pipe bent through a 90-degree elbow or a U-turn, from one O-grid disc.

    Sweep one cross-section along a **curved** path: the section is carried by a moving
    orthonormal frame, so at every station it is placed by a *rigid* motion -- the
    curved generalization of :func:`extrude`, which is this with a straight path and a
    constant frame.

    **The section is moved rigidly, not point-by-point.**  Through a bend of radius
    ``Rb`` a node sitting ``d`` outboard of the centreline traverses radius ``Rb + d``
    and one ``d`` inboard traverses ``Rb - d``: they cover different distances and
    neither of them follows the path.  That is the correct behaviour of a swept solid,
    and it is what offsetting every section point along its own copy of the curve would
    get wrong -- that would shear the section, and on a bend tighter than the section
    is wide it would fold the inboard nodes through the axis and invert the elements.
    Nothing here prevents that fold -- a bend radius must exceed the section's own
    in-plane extent -- but it is not silently meshed either: the folded layer comes out
    mixed-winding and ``loft`` rejects it, naming the sweep as the likely cause.

    ``path`` is **vectorized** -- ``(K,) -> (K,3)`` -- unlike ``loft_fn``'s
    profile-at-a-time callable, and deliberately so: a rotation-minimizing frame is a
    *sequential* integration along the curve, so it cannot be evaluated at one isolated
    parameter.  (``loft_fn``'s ``f`` is scalar for its own reason: it returns a
    *mesh*, and a callable handing back one mesh can only take one parameter value.
    ``LineMesh.loft_fn``'s ``f`` returns coordinates, so it is vectorized again.
    The three shapes disagree because the three return types do.)  ``sweep`` samples
    the whole node lattice
    ``_refined_lattice(fractions, order)`` in one call, builds the frame field on it,
    places the section at every level -- corner levels *and* the intermediate GLL levels
    -- and delegates to :func:`loft` through ``sweep_nodes``.  So the sweep direction is
    exact at any order, not straight-subdivided between slices.

    ``fractions`` are the path parameter values themselves, in ``path``'s own units, and
    grade the sweep exactly as they do on ``loft_fn``.  ``loop=True`` takes the
    trailing wrap value; the closing profile is the *identical* placement as the first
    (not a re-evaluation), so a closed sweep welds exactly rather than to a tolerance.

    ``orientation`` picks the frame generator
    (:func:`nekmeshpy.model.frames.sweep_placements`): ``"transport"`` -- the default,
    rotation-minimizing, seeded from the section's own in-plane axis so it does not spin
    at the start and correct on a non-planar path; ``"fixed"`` with ``up=`` -- exact and
    zero-twist, the right choice for a planar path (an elbow, a U-turn), failing loudly
    if a tangent turns parallel to ``up``; ``"frenet"`` -- included but wrong for a
    sweep, being undefined on a straight run and sign-flipping through an inflection.
    It names a *mode* and nothing else; the per-station up vectors that used to be
    passed as ``orientation`` are now a ``(K,3)`` ``up=`` with ``orientation="fixed"``.
    ``up`` therefore takes either a single ``(3,)`` world direction or a ``(K,3)``
    per-station field (told apart by rank).  ``twist`` adds a total roll in radians
    about the tangent, spread over the stations.

    ``tangent`` is the path's **derivative**, ``(K,) -> (K,3)``, and is worth passing
    whenever the path has one in closed form.  Without it the tangent field is central
    differences of the *sampled* centreline: O(h^2), and worst exactly where the
    curvature jumps (a straight run meeting an arc), which tilts every frame there --
    so the centreline lands exactly and the section does not.  **This is the quiet
    failure mode, not a loud one**: measured on ``examples/serpentine_pipe.py`` the
    finite-differenced sweep pulls the wall 1.1e-4 *inside* ``R_PIPE`` -- 0.2% of the
    tube radius -- while passing every quality, watertightness and topology check the
    suite has; passing the analytic derivative takes the same measurement to 4.1e-11.
    It is normalized here, so any non-unit scaling of the true derivative will do.

    ``origin`` is **required**: it is the section's reference point, the one that rides
    the path.  It used to default to the section's centroid, which is defensible and
    frequently wrong -- an O-grid disc's centroid is *not* its centre (the grid is
    slightly asymmetric), so the obvious call produced a quietly off-axis block with no
    error anywhere.  There is no safe default, so there is no default; pass the centre
    the boundary loop was built about.  ``normal=`` overrides the section's own fitted
    plane, needed only when it is not planar (which is otherwise a ``ValueError``
    rather than a silent shear).  Tags behave exactly as on :func:`loft`:
    ``element_tags`` is per sweep layer and overrides the section's own where non-empty,
    and ``first_tag`` / ``last_tag`` cap the ends (rejected when ``loop=True``).

    The order is the **section's own** -- a rigid placement cannot change it, so there
    is nothing for a separate ``order=`` argument to say that argument one does not
    already say.
    """
    order = section.order
    _, t = _sweep_lattice(fractions, order, loop=loop, name="sweep")
    if loop:
        reject_loop_caps("HexMesh.sweep", first_tag, last_tag)
    tv: FloatArray = t[:-1] if loop else t
    P, T = _sweep_path(path, tangent, tv)
    places = frames.sweep_placements(
        section.points, P, orientation=orientation, up=up, twist=twist,
        close_twist=close_twist, loop=loop, origin=origin, normal=normal,
        path_tangents=T)
    profs: list[QuadMesh] = [quad_transform(section, M, o) for M, o in places]
    if loop:
        profs.append(profs[0])          # the seam profile *is* the first placement
    return _loft_evaluated(profs, t, order, loop=loop, element_tags=element_tags,
                           first_tag=first_tag, last_tag=last_tag, name="sweep")


#: Rung-raising combinators bound onto ``HexMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "extrude": extrude,
    "sweep": sweep,
    "annulus": annulus,
    "from_grid": from_grid,
}
