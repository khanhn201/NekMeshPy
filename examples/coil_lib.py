"""The serpentine coil's own move table -- the one physical part, defined once.

``serpentine_pipe.py`` builds this coil standing alone; ``chimera_full.py`` sweeps a
copy of it between the branches of each pair of T2 junctions.  They are the same part,
so the numbers live here rather than in both scripts: they used to be duplicated
verbatim, under a comment instructing the reader to "change them here and there
together", which is the defect rather than the mitigation.

The shape is **traced from a reference photo and is fixed**.  It is only ever placed,
never reshaped or rescaled.

The walk is in the coil's own 2-D ``(u, v)``: ``u`` runs along a pass, ``v`` stacks pass
to pass.  A caller lifts it onto its own plane with
:func:`paths.embed <nekmeshpy.core.paths.embed>` -- ``serpentine_pipe.py`` maps
``u -> +z`` and ``v -> -x``, which is also where ``chimera_full.py`` puts it.

Measured facts of the path as shipped: total length 1238.823001646924; 23 segments
(12 straights + 11 arcs); turns ``[+90, -90, -180, +180, -180, +180, -180, +180, -180,
-90, +90]``; min bend radius 2.5 = ``5 * R_PIPE``; min non-adjacent self-distance 4.806
against a tube diameter of ``2 * R_PIPE = 1.0`` -- so the coil never touches itself.
"""

R_PIPE = 0.5      # tube radius -- the swept cross-section, not the path
PASS_LEN = 136.0  # length of a full vertical pass
U_R = 2.5         # tight U-turn radius: bottom turns + top hairpins
U_R_MID = 4.0     # wider radius of the raised middle bridge
R_HOOK = U_R_MID  # the two end hooks turn at the same radius
HOOK_JOG = 5.0    # the hook's short sideways step
HOOK_DROP = 20.0  # the hook's straight run out to the inlet / outlet
RAISE = 4.0       # extra length on passes 1/4/5/8 -- what lifts the middle bridge
                  # above the two flanking hairpins

#: The two end hooks, each the other's time reversal (reverse the order, negate every
#: turn) -- which is what lands both openings on the same ``v`` facing the same way.
HOOK_IN = [("line", HOOK_DROP, 0.0), ("arc", R_HOOK, 90.0),
           ("line", HOOK_JOG, 0.0), ("arc", R_HOOK, -90.0)]
HOOK_OUT = [("arc", R_HOOK, -90.0), ("line", HOOK_JOG, 0.0),
            ("arc", R_HOOK, 90.0), ("line", HOOK_DROP, 0.0)]

#: ``("line", length, 0.0)`` or ``("arc", radius, signed turn in degrees)``; a positive
#: turn is counter-clockwise in the ``(u, v)`` plane.  8 vertical passes joined by 7
#: semicircular 180-degree U-bends alternating bottom / top, hooked at both ends.
#:
#: The coil is deliberately **not** symmetric top to bottom: passes 4 and 5 are ``RAISE``
#: longer than their neighbours, so the wide middle bridge (``U_R_MID``) that joins the
#: two half-coils sits *above* the two flanking hairpins (``U_R``) rather than level with
#: them, and passes 1 and 8 are lengthened to match.
MOVES = (HOOK_IN
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R, -180.0)]      # pass 1 -> bottom
    + [("line", PASS_LEN, 0.0), ("arc", U_R, 180.0)]               # pass 2 -> top hairpin
    + [("line", PASS_LEN, 0.0), ("arc", U_R, -180.0)]              # pass 3 -> bottom
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R_MID, 180.0)]   # 4 -> RAISED bridge
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R, -180.0)]      # pass 5 -> bottom
    + [("line", PASS_LEN, 0.0), ("arc", U_R, 180.0)]               # pass 6 -> top hairpin
    + [("line", PASS_LEN, 0.0), ("arc", U_R, -180.0)]              # pass 7 -> bottom
    + [("line", PASS_LEN + RAISE, 0.0)]                            # pass 8
    + HOOK_OUT)

#: Target hex length along the sweep.  NOT cubic: the coil is slender (a pass is 272
#: tube radii long), so ~1.6*R_PIPE would cost ~100k hexes.  The real floor is the
#: tightest turn -- ``sweep_fractions`` rounds a segment's length/target to the NEAREST
#: station count, so a target near a U-turn's own arc length (``pi * U_R`` = 7.85) rounds
#: down to ONE station spanning the whole 180 degrees: two opposed sections lerped into
#: a near-zero-volume hex.  2.0 puts 4 stations in that turn; 6.0 or 8.0 put 1.
TARGET_LEN = 2.0
