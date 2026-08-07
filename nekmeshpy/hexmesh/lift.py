"""Fixed-arity ``HexMesh`` operations that raise a rung (delta +1).

``extrude`` sweeps one section straight; ``annulus`` fills between two index-paired
closed surfaces; ``from_grid`` builds a block from one structured point grid.  All
three are thin: they position profiles and hand them to
:func:`~nekmeshpy.hexmesh.assemble.loft`, which owns the index space, so the numbering
they expose is the loft's carried up unchanged.

Free functions bound onto :class:`HexMesh <nekmeshpy.hexmesh.hexmesh.HexMesh>` by ``hexmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``HexMesh.<name>`` sugar.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Callable, Literal

import numpy as np
from scipy.spatial import cKDTree

from .._typing import (
    FloatArray,
    IntArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..linemesh.assemble import _sweep_lattice, _sweep_path
from ..linemesh.shape import path_fractions
from ..model import frames
from ..model.fields import reject_loop_caps, validate_layers
from ..model.paths import SpacePath
from ..quadmesh import QuadMesh
from ..quadmesh.lift import from_grid as quad_from_grid
from ..quadmesh.morph import blend as quad_blend
from ..quadmesh.morph import reindex as quad_reindex
from ..quadmesh.morph import rotate as quad_rotate
from ..quadmesh.morph import transform as quad_transform
from ..quadmesh.morph import translate
from ..quadmesh.ports import Port
from ..quadmesh.query import plane_normal as quad_plane_normal
from .assemble import _loft_evaluated, loft
from .hexmesh import _GRID_SIDES, _ORIGIN, _Z_AXIS, HexMesh

_log = logging.getLogger(__name__)


def extrude(
    section: QuadMesh,
    length: float,
    layers: int | FloatArray,
    *,
    axis: Vec3 = _Z_AXIS,
    origin: Point = _ORIGIN,
    first_tag: str | Sequence[str] | StrArray | None = None,
    last_tag: str | Sequence[str] | StrArray | None = None,
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
    ``fractions`` on the sibling :func:`sweep <nekmeshpy.hexmesh.lift.sweep>`: they are required, so making them
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
    inner_tag: str | None = None,
    outer_tag: str | None = None,
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
    surface has no free boundary edges): inner caps face 5, outer caps face 6.  A
    scalar ``inner_tag`` / ``outer_tag`` overrides and names the whole wall:
    ``None`` (the default) is "not asked for" and inherits the surface's tags, and
    ``NO_TAG`` is an explicit override *to* untagged, which suppresses them."""
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
    # wall tags from the surfaces' per-quad element_tags, which a scalar arg
    # overrides.  ``None`` is "not asked for" and inherits; ``NO_TAG`` is an
    # explicit override *to* untagged, so it suppresses the surface's own tags
    # rather than falling through to them.
    inner_caps: str | StrArray = (
        inner_tag if inner_tag is not None
        else (inner.element_tags.dense(inner.n_quads) if inner.element_tags else ""))
    outer_caps: str | StrArray = (
        outer_tag if outer_tag is not None
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

    Built as a :func:`loft <nekmeshpy.hexmesh.assemble.loft>` of the grid's **``k``-sections**: section ``k`` is
    the :func:`QuadMesh.from_grid <nekmeshpy.quadmesh.lift.from_grid>` of the
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
    return loft(slices, first_tag=tags.get("z_min"),
                last_tag=tags.get("z_max"))


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
    first_tag: str | Sequence[str] | StrArray | None = None,
    last_tag: str | Sequence[str] | StrArray | None = None,
) -> HexMesh:
    """A block swept from one ``QuadMesh`` ``section`` along the curve ``path`` -- a
    round pipe bent through a 90-degree elbow or a U-turn, from one O-grid disc.

    Sweep one cross-section along a **curved** path: the section is carried by a moving
    orthonormal frame, so at every station it is placed by a *rigid* motion -- the
    curved generalization of :func:`extrude <nekmeshpy.hexmesh.lift.extrude>`, which is this with a straight path and a
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
    -- and delegates to :func:`loft <nekmeshpy.hexmesh.assemble.loft>` through ``sweep_nodes``.  So the sweep direction is
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
    rather than a silent shear).  Tags behave exactly as on :func:`loft <nekmeshpy.hexmesh.assemble.loft>`:
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


def sweep_path(
    section: QuadMesh,
    path: SpacePath,
    *,
    origin: Point | Sequence[float],
    target_length: float | None = None,
    layers: int | None = None,
    fractions: FloatArray | Sequence[float] | None = None,
    orientation: Literal["transport", "fixed", "frenet"] = "transport",
    up: Vec3 | Sequence[float] | PointArray | None = None,
    twist: float = 0.0,
    close_twist: bool = True,
    normal: Vec3 | Sequence[float] | None = None,
    loop: bool = False,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray | None = None,
    last_tag: str | Sequence[str] | StrArray | None = None,
) -> HexMesh:
    """:func:`sweep <nekmeshpy.hexmesh.lift.sweep>` driven by a
    :class:`SpacePath <nekmeshpy.model.paths.SpacePath>` rather than by a loose
    ``(centerline, tangent, fractions)`` triple.

    The path object already carries its own analytic tangent and its own junction
    table, so this asks for an element length instead of a station array: give exactly
    one of ``target_length`` (the desired hex length along the sweep), ``layers`` (that
    many on average), or ``fractions`` (the stations verbatim, for a path graded piece
    by piece).  See
    :func:`path_fractions <nekmeshpy.linemesh.shape.path_fractions>` for the resolution.

    Everything else is ``sweep``'s, unchanged and with ``sweep``'s own defaults --
    including ``orientation="transport"``.  A planar walk out of
    :func:`paths.embed <nekmeshpy.model.paths.embed>` almost always wants
    ``orientation="fixed"`` with ``up=`` the plane normal, but that is not defaulted
    here: ``up``'s **sign** picks which of two frames rolled 180 degrees apart carries
    the section, and a normal derived from the embedding's axis order would be a guess
    at the caller's intent rather than a reading of it."""
    fr = path_fractions(path, target_length=target_length, layers=layers,
                        fractions=fractions)
    return sweep(section, path.centerline, fr, origin=origin, tangent=path.tangent,
                 orientation=orientation, up=up, twist=twist, close_twist=close_twist,
                 normal=normal, loop=loop, element_tags=element_tags,
                 first_tag=first_tag, last_tag=last_tag)


def _find_roll(a: QuadMesh, b: QuadMesh, axis: Vec3 | Sequence[float]) -> int:
    """The quarter turn ``k`` about ``axis`` that minimizes the index-wise deviation
    between ``a`` and ``b`` about their own centres.

    Two discs off the same quadrant recipe carry their seams on the same 45-degree
    family, but each arrives through its own chain of axis-permuting rotations, so the
    index pairing between them may be rolled by a multiple of 90 degrees.  Blending
    across a rolled pairing twists the result into inverted elements, so the roll is
    *measured* rather than assumed."""
    ca, cb = a.points.mean(axis=0), b.points.mean(axis=0)
    best_k, best_d = 0, np.inf
    for k in range(4):
        cand = quad_rotate(a, k * np.pi / 2.0, axis=axis, center=ca)
        d = float(np.linalg.norm((cand.points - ca) - (b.points - cb), axis=1).max())
        if d < best_d:
            best_k, best_d = k, d
    return best_k


def _self_map(a: QuadMesh, k: int, axis: Vec3 | Sequence[float]) -> IntArray:
    """``a``'s own near-4-fold self-map under ``k`` quarter turns: which of ``a``'s own
    points does point ``i`` land closest to, rotated.  Entirely about ``a``'s own
    geometry -- nothing to do with the section it is about to be paired against."""
    ca = a.points.mean(axis=0)
    cand = quad_rotate(a, k * np.pi / 2.0, axis=axis, center=ca)
    _, sigma = cKDTree(cand.points).query(a.points)
    return np.asarray(sigma, dtype=np.int64)


def adapter(a: QuadMesh | Port, b: QuadMesh | Port, *,
            axis: Vec3 | Sequence[float] | None = None, layers: int = 2,
            max_deviation: float = 0.2, radius_tol: float = 0.05) -> HexMesh:
    """A short block morphing between two same-connectivity sections whose *node
    patterns* differ slightly -- and whose **both** end faces are bit-exact.

    For a seam between two pieces built by nearly, but not quite, the same recipe: a
    couple of percent between one quadrant's wall spacing and another's, say.  A plain
    coordinate weld across such a seam can never be exact, and at ``order > 1``
    :func:`merge <nekmeshpy.hexmesh.assemble.merge>` verifies shared high-order edge
    and face nodes against ``conform.entity_tol`` (~1e-9 of the model extent), so
    "close" fails.  A blend is exact instead: its first slice **is** ``a``'s own
    points, and its last is ``b``'s own geometry reached through ``a``'s labelling
    (:func:`quadmesh.reindex <nekmeshpy.quadmesh.morph.reindex>`), so both end welds
    are bit-exact while the mismatch is absorbed smoothly inside.

    ``a`` is left completely untouched -- whatever it is already bit-identical to (a
    swept connector's own terminal section, say) stays so.  Relabelling ``b`` is what
    buys that: rotating ``a``'s *coordinates* into the pairing instead would make the
    blend's first slice a rotated copy of ``a``, close to but not identical with
    ``a``'s own literal end.

    The 90-degree roll between the two index patterns is measured, not assumed (see
    ``_find_roll``); ``axis`` is the axis it is measured about, normally the seam
    normal.  A residual deviation above ``max_deviation`` means no quarter turn aligns
    the two at all, which is a ``ValueError`` rather than a twisted block.

    Passing :class:`Ports <nekmeshpy.quadmesh.ports.Port>` rather than bare sections
    lets ``axis`` default to ``a``'s own stated normal -- which is what the seam normal
    always was -- and adds the two checks a fitted plane cannot make on its own: that
    the ports face each other, and that they are the same size."""
    if layers < 1:
        raise ValueError("adapter: layers must be >= 1, got %d" % layers)
    sec_a = a.section if isinstance(a, Port) else a
    sec_b = b.section if isinstance(b, Port) else b
    pa, a_stated = _as_port(a, sec_b.points.mean(axis=0), "adapter")
    pb, b_stated = _as_port(b, sec_a.points.mean(axis=0), "adapter")
    _check_facing(pa, pb, a_stated and b_stated, "adapter", radius_tol)
    if axis is None:
        if not a_stated:
            raise ValueError(
                "adapter: give axis=, the direction the 90-degree roll between the two "
                "index patterns is measured about (normally the seam normal). It "
                "defaults to a's own normal only when a is a Port, which states one.")
        axis = pa.normal
    a, b = sec_a, sec_b
    ca, cb = a.points.mean(axis=0), b.points.mean(axis=0)
    k = _find_roll(a, b, axis)
    b_aligned = quad_reindex(a, b, _self_map(a, k, axis))
    dev = float(np.linalg.norm((b_aligned.points - cb) - (a.points - ca), axis=1).max())
    _log.debug("adapter: roll k=%d, residual index-wise deviation %.3e", k, dev)
    if dev > max_deviation:
        raise ValueError(
            "adapter: no 90-degree roll aligns these two node patterns -- the best of "
            "the four leaves an index-wise deviation of %.3g about each section's own "
            "centre (max_deviation %.3g). Blending across it would twist the block "
            "into inverted elements; use bridge() for patterns this far apart."
            % (dev, max_deviation))
    return loft(quad_blend(a, b_aligned, np.linspace(0.0, 1.0, layers + 1)))


def _as_port(x: QuadMesh | Port, toward: Point, who: str) -> tuple[Port, bool]:
    """``(port, was_stated)`` -- promote a bare ``QuadMesh`` by *guessing* its outward
    direction as the one pointing at ``toward``, or take a ``Port``'s stated one.

    The guess is what these joins have always done, and it is right whenever the two
    sections really do face each other.  It cannot tell that they do: handed two ports
    facing the same way it flips one and folds the connector, with nothing to catch it.
    Passing a :class:`Port <nekmeshpy.quadmesh.ports.Port>` states the direction instead,
    which is what lets the caller below check rather than assume."""
    if isinstance(x, Port):
        return x, True
    n = quad_plane_normal(x, check=False)
    c = x.points.mean(axis=0)
    d = np.asarray(toward, dtype=float) - c
    n = n if float(n @ d) > 0.0 else -n
    r = float(np.linalg.norm(np.asarray(x.points, dtype=float) - c, axis=1).max())
    return Port(x, n, c, r), False


def _check_facing(pa: Port, pb: Port, both_stated: bool, who: str,
                  radius_tol: float) -> None:
    """The two checks a guessed direction cannot make.  Skipped unless *both* sides
    stated theirs -- a guess is derived from the very geometry being checked, so
    checking it would only ever confirm itself."""
    if not both_stated:
        return
    if not pa.faces(pb):
        raise ValueError(
            "%s: the two ports do not face each other (normals %s and %s, cosine "
            "%+.3f). A connector between them would have to fold back through one of "
            "the components; reverse whichever port is stated the wrong way round."
            % (who, np.array2string(pa.normal, precision=3),
               np.array2string(pb.normal, precision=3),
               float(pa.normal @ pb.normal)))
    big = max(pa.radius, pb.radius)
    if abs(pa.radius - pb.radius) > radius_tol * big:
        raise ValueError(
            "%s: the two ports are different sizes (radius %.6g and %.6g, %.1f%% "
            "apart). This joins same-radius sections; blend or loft between them "
            "instead if a taper is what you want."
            % (who, pa.radius, pb.radius,
               100.0 * abs(pa.radius - pb.radius) / big))


def _stub_sections(disc: QuadMesh, direction: Vec3, distance: float,
                   count: int) -> list[QuadMesh]:
    """``count`` copies of ``disc``'s own exact pattern, rigidly carried a total
    ``distance`` along ``direction`` from its own centroid.

    ``direction`` is the disc's **own** true normal at both call sites, not the raw
    centroid-to-centroid direction: the two differ by a small angle whenever a disc is
    not perfectly centred on its nominal position, and a rigid placement is exactly
    perpendicular to whatever tangent it is handed -- even at station 0 -- so a tangent
    a hair off the disc's own normal makes the first section a hair off the disc
    itself."""
    c = disc.points.mean(axis=0)
    if count < 2 or distance <= 0.0:
        return [disc]
    up = (0.0, 0.0, 1.0) if abs(direction[2]) < 0.9 else (1.0, 0.0, 0.0)
    s = np.linspace(0.0, 1.0, count)
    P: PointArray = c + s[:, None] * distance * direction
    T: PointArray = np.tile(direction, (count, 1))
    places = frames.sweep_placements(disc.points, P, orientation="fixed", up=up,
                                     origin=c, path_tangents=T)
    return [quad_transform(disc, M, o) for M, o in places]


def bridge(a: QuadMesh | Port, b: QuadMesh | Port, *, layers: int = 4,
           stub_fraction: float = 0.3, stub_max: float = 1.5, blend_layers: int = 6,
           radius_tol: float = 0.05) -> HexMesh:
    """A connector between two same-radius sections whose node patterns are too far
    apart for :func:`adapter <nekmeshpy.hexmesh.lift.adapter>` -- two legs of different
    T-junctions, built by different algorithms.

    A short rigid stub is carried off **each** side along its own true normal, so both
    near ends stay bit-exact to whatever ``a`` and ``b`` are themselves bonded to, and
    the remaining gap is spanned by a straight blend -- with the stubs and the blend
    lofted together as **one** block.  That single loft is what makes this exact at
    ``order > 1``: leaving the far seam to :func:`merge
    <nekmeshpy.hexmesh.assemble.merge>`'s tolerance weld is fine at order 1, but order
    > 1 also verifies shared high-order edge nodes against ``conform.entity_tol``,
    which an approximate weld cannot meet.  One loft has no internal seam to verify.

    The blend needs an honest point correspondence, and here the two patterns are
    genuinely far apart -- stations spaced by arc length against uniform angular ones
    can differ by a *median* comparable to the section radius, and no rotation improves
    it, because the mismatch is a difference in station *distribution* rather than
    orientation.  Nearest-neighbour matching of the two centred point clouds fixes
    that, and **every** section on ``b``'s side is then relabelled through it, not just
    the one touching the blend: relabelling only the tip leaves the blend's last slice
    and ``b``'s own naturally-labelled stub disagreeing, which twists that seam into
    inverted elements."""
    if blend_layers < 1:
        raise ValueError("bridge: blend_layers must be >= 1, got %d" % blend_layers)
    sec_a = a.section if isinstance(a, Port) else a
    sec_b = b.section if isinstance(b, Port) else b
    pa, a_stated = _as_port(a, sec_b.points.mean(axis=0), "bridge")
    pb, b_stated = _as_port(b, sec_a.points.mean(axis=0), "bridge")
    _check_facing(pa, pb, a_stated and b_stated, "bridge", radius_tol)
    ca, cb = sec_a.points.mean(axis=0), sec_b.points.mean(axis=0)
    length = float(np.linalg.norm(cb - ca))
    stub = min(stub_max, stub_fraction * length)
    n_stub = max(2, layers // 2)
    a_secs = _stub_sections(sec_a, pa.normal, stub, n_stub)
    b_secs_raw = _stub_sections(sec_b, pb.normal, stub, n_stub)[::-1]

    a_end, b_end = a_secs[-1], b_secs_raw[0]
    _, sigma = cKDTree(b_end.points - b_end.points.mean(axis=0)).query(
        a_end.points - a_end.points.mean(axis=0))
    if len(set(sigma.tolist())) != sigma.size:
        raise ValueError(
            "bridge: nearest-neighbour matching of the two node patterns is not a "
            "permutation -- they are too dissimilar to pair one-for-one, so the blend "
            "would collapse several of one section's nodes onto one of the other's")
    b_secs = [quad_reindex(a_end, s, sigma) for s in b_secs_raw]
    blend_secs = quad_blend(a_end, b_secs[0], np.linspace(0.0, 1.0, blend_layers + 1))
    return loft(a_secs[:-1] + blend_secs + b_secs[1:])


__all__ = [
    "adapter",
    "annulus",
    "bridge",
    "extrude",
    "from_grid",
    "sweep",
    "sweep_path",
]
