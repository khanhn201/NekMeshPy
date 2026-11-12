import os
"""Complete EVG sweep mesh of the box: regular faces, rim cores, hub core.
-> out_full.vtu / out_full.npz"""
import numpy as np, meshio, time, pickle
import mesher
from mesher import Nodes, map_gov
from faces import face_pairs
from corners import loop_sides
from structured import assign, tfi_face, unstructured_face
from columns import sweep, scaled_jacobian, orient
import items as IT
from core import Core, NARC
from hubcore import HubCore, trace_triple
from loops2 import build_loop

mesher.GRAPH = pickle.load(open("maps/graph2.pkl", "rb"))
edges, verts = mesher.GRAPH

# ---- cores: (hoops, ends, xi_start), anchor faces in order
IT.CORES[0] = Core(0, [72, 73], (-2, 81), eps=2.5)
IT.CORES[1] = Core(1, [79, 80], (-1, 82), eps=2.5)
IT.CORES[2] = Core(2, [81], (83, 73), xi_start=5.64, eps=3.5)
IT.CORES[3] = Core(3, [82], (80, 83), xi_start=-5.6, eps=3.5)
RIM = {}
for cid, c in IT.CORES.items():
    govs = [c.ends[0]] + c.hoops + [c.ends[1]]
    for an in range(len(govs) - 1):
        f = tuple(sorted((govs[an], govs[an + 1])))
        RIM.setdefault(f, []).append((cid, an))

IT.HUB["core"] = HubCore()
HUBF = {}
for f, pcs in IT.HUB["core"].pieces().items():
    HUBF[tuple(sorted(f))] = pcs
g = IT.hub_geom()
IT.HUB["trace"] = {"T+": trace_triple((83, 81, 102), g[("line", +1)][-1], verts[90]["X"]),
                   "T-": trace_triple((83, 82, 102), g[("line", -1)][-1], verts[99]["X"])}
IT.HUB["trace_ends"] = {"T+": (("h", "T", +1), ("v", 90)), "T-": (("h", "T", -1), ("v", 99))}
for _tid, _vid in (("T+", 90), ("T-", 99)):            # end the traces exactly on their (possibly moved) vertex
    _P = IT.HUB["trace"][_tid]
    _L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(_P, axis=0), axis=1))]
    _w = np.clip(1.0 - (_L[-1] - _L) / 8.0, 0.0, 1.0)
    IT.HUB["trace"][_tid] = _P + _w[:, None] * (verts[_vid]["X"] - _P[-1])[None, :]
TRF = {}
for tid, fs in (("T+", [(81, 83), (81, 102), (83, 102)]), ("T-", [(82, 83), (82, 102), (83, 102)])):
    for f in fs:
        TRF.setdefault(f, []).append(tid)


FORCE_MAP = {(81, 92): 81, (81, 102): 81}
for _kv in os.environ.get("EVG_FORCEMAP", "").split(","):
    if _kv:
        _f, _m = _kv.split(":"); FORCE_MAP[tuple(int(g) for g in _f.split("-"))] = int(_m)
FORCE_U = {(92, 102), (93, 102), (81, 92), (82, 93), (83, 92), (83, 93), (81, 102), (82, 102)}


SPLIT = {}     # (face cut between two cores: tried, the cut crosses the fans -- disabled)


def split_face(f, loop):
    """cut a face carrying two cores' tombstones between them (plate -> vertex c);
    each half is meshed in its own pipe's map so neither pipe's converging rays fold"""
    from mesher import to_map, lift, TUBES
    c, TA, TB = SPLIT[f]
    ends = [IT.start_end(it) for it in loop]
    n = len(loop)
    iq = [i for i, it in enumerate(loop) if it[0] == "q" and loop[i - 1][0] == "o" and loop[(i + 1) % n][0] == "o"
          and loop[i - 1][1] != loop[(i + 1) % n][1]][0]
    loop = loop[iq:] + loop[:iq]; ends = ends[iq:] + ends[:iq]
    jc = [i for i, e in enumerate(ends) if e[0] == ("v", c)][0]
    ea, eb = loop[0][1], loop[0][2]
    sa, fa = to_map(TA, IT.endpoint_X(ea)[None]); sb, fb = to_map(TA, IT.endpoint_X(eb)[None])
    fb = fa + (fb - fa + np.pi) % (2 * np.pi) - np.pi
    srow = 0.0 if sa[0] < TUBES[TA].L / 2 else TUBES[TA].L
    Pm, _ = lift(TA, np.array([srow]), np.array([0.5 * (fa[0] + fb[0])]), TB)
    Pm[0, 1] = mesher.PLATE if hasattr(mesher, "PLATE") else Pm[0, 1]
    from geom import PLATE
    Pm[0, 1] = PLATE
    m = ("m", f)
    IT.MPTS[m] = Pm[0]
    sc, fc = to_map(TA, verts[c]["X"][None])
    fc = fa + (fc - fa + np.pi) % (2 * np.pi) - np.pi
    t = np.linspace(0, 1, 41)
    P, _ = lift(TA, srow + t * (sc[0] - srow), 0.5 * (fa[0] + fb[0]) + t * (fc[0] - 0.5 * (fa[0] + fb[0])), TB)
    P[0] = Pm[0]; P[-1] = verts[c]["X"]
    name = ("cut",) + f
    IT.CUTS[name] = dict(pts=P, a=m, b=("v", c))
    L1 = [("q", m, eb)] + loop[1:jc] + [("k", name, True)]
    L2 = loop[jc:] + [("q", ea, m), ("k", name, False)]
    has = lambda L, T: any(it[0] == "o" and T in IT.CORES[it[1]].hoops and
                           IT.CORES[it[1]].hoops[it[2] - 1 if it[2] > 0 else 0] == T for it in L)
    owner = lambda L: TA if any(it[0] == "o" and TA in IT.CORES[it[1]].hoops for it in L) else TB
    return [(L1, owner(L1)), (L2, owner(L2))]


def special(f):
    return f in FORCE_U or f in RIM or f in HUBF or f in TRF or any(
        e.get("faces") and f in [tuple(sorted(x)) for x in e["faces"]] for e in edges if e.get("alive", True))


import os
HFACE = {tuple(int(g) for g in k.split("-")): float(v) for k, v in
         (kv.split(":") for kv in os.environ.get("EVG_HFACE", "").split(",") if kv)}


def main():
    import patches as PT
    t0 = time.time()
    FS, PATCHED, fallback, fails = {}, {}, [], []
    for A, X in face_pairs():
        try:
            loop = build_loop(A, X, RIM, HUBF, TRF)
        except AssertionError as e:
            fails.append((A, X, str(e)[:120])); continue
        M0 = FORCE_MAP.get((A, X), map_gov(A, X))
        maps = [M0] + [g for g in (A, X) if 0 <= g < 1000 and g != M0]
        err = None
        for M in maps:
            try:
                F = PT.Face((A, X), loop, M); F.plan()
                PATCHED[(A, X)] = F; mesher.MAPGOV[(A, X)] = M
                break
            except AssertionError as e:
                err = (M, str(e)[:120])
        else:
            fallback.append(((A, X), err))
    print("faces", len(PATCHED) + len(fallback), "kinds", {k: sum(F.kind == k for F in PATCHED.values()) for k in ("quad", "tri", "cut")},
          "loop fails", fails, flush=True)
    assert not fallback, ("faces without a structured decomposition", fallback)
    PT.align_all(list(PATCHED.values()), tol=float(os.environ.get("EVG_SNAP", 0.4)) * mesher.H)
    fixed = {("oa", c): NARC for c in IT.CORES}
    fixed[("hz",)] = IT.HUB_COUNTS["nz1"] + IT.HUB_COUNTS["nz2"]
    fixed[("ha",)] = IT.HUB_COUNTS["ma"]
    Hk = {}
    for f, (sd, loop) in FS.items():
        if f[:2] in HFACE:
            for it in loop:
                k = IT.key(it, f); Hk[k] = min(Hk.get(k, mesher.H), HFACE[f[:2]])
    for f, F in PATCHED.items():
        if f[:2] in HFACE:
            for it in F.loop:
                k = IT.key(it, f); Hk[k] = min(Hk.get(k, mesher.H), HFACE[f[:2]])
            for sides in F.patches:
                for sd in sides:
                    for pc in sd:
                        if pc[0] == "c":
                            k = ("cut", F.f, F.M, pc[1]); Hk[k] = min(Hk.get(k, mesher.H), HFACE[f[:2]])
    counts, dev, ctx = PT.assign2(FS, PATCHED, mesher.H, fixed=fixed, Hk_fn=lambda k: Hk.get(k, mesher.H), narc=NARC)
    for k in sorted(dev, key=dev.get)[-4:]:
        print("   most refined:", k, round(dev[k], 2), counts[k])
    dv = np.array(list(dev.values()))
    print(f"ILP {len(counts)} items, size ratio {dv.min():.2f}..{dv.max():.2f} ({time.time()-t0:.0f}s)", flush=True)
    print("   transitions:", len(ctx.trans), "patches,", sum(v[2] for v in ctx.trans.values()), "templates", flush=True)
    nodes = Nodes(); hexes = []; owner = []
    for cid, c in IT.CORES.items():
        nz = counts.get(("oz", cid))
        if nz is None:
            print("core", cid, "unused"); continue
        out = c.build(nodes, nz, h=mesher.H)
        out["nz"] = nz; out["narc"] = NARC
        IT.CORE_OUT[cid] = out
        hexes.extend(out["hex"]); owner.extend([(("core", cid), -99)] * len(out["hex"]))
        by = {}
        for q, gv in out["lateral"] + out["endq"]:
            by.setdefault(gv, []).append(q)
        for gv, qs in by.items():
            sweep(nodes, np.array(qs), (gv,), hexes, owner, ("core", cid))
    hc = IT.HUB_COUNTS
    out = IT.HUB["core"].build(nodes, hc["ma"], hc["nz1"], hc["nz2"])
    IT.HUB["out"] = out
    hexes.extend(out["hex"]); owner.extend([(("hub",), -99)] * len(out["hex"]))
    by = {}
    for q, gv in out["lateral"]:
        by.setdefault(gv, []).append(q)
    for gv, qs in by.items():
        sweep(nodes, np.array(qs), (gv,), hexes, owner, ("hub",))
    mfail, switched = [], []
    FACEQ = {}
    for fk, F in PATCHED.items():
        A, X = fk
        other = X if F.M == A else A
        try:
            Q = PT.mesh_face(nodes, F, counts, other, ctx)
        except AssertionError as e:
            mfail.append((A, X, F.M, str(e)[:160])); continue
        FACEQ[fk] = (F.M, Q, F.kind, len(F.patches))
        hx, ow = [], []
        sweep(nodes, Q, (A, X), hx, ow, (A, X))
        hexes.extend(hx); owner.extend(ow)
    pickle.dump((FACEQ, nodes.arr()), open("out_full_faces.pkl", "wb"))
    print("faces meshed in the other map:", switched)
    print("mesh fails", mfail)
    XYZ = nodes.arr()
    H = orient(hexes, XYZ)
    sj = scaled_jacobian(XYZ[H])
    print(f"{time.time()-t0:.0f}s", len(H), "hexes; scaled J: min %.3f p1 %.3f median %.3f <0: %d <0.2: %d" %
          (sj.min(), np.percentile(sj, 1), np.median(sj), (sj < 0).sum(), (sj < 0.2).sum()))
    tags = sorted(set(o[0] for o in owner), key=str)
    tid = {t: i for i, t in enumerate(tags)}
    own = np.array([tid[o[0]] for o in owner]); gov = np.array([o[1] for o in owner])
    meshio.write("out_full.vtu", meshio.Mesh(XYZ, [("hexahedron", H)], cell_data={"sj": [sj], "part": [own], "gov": [gov]}))
    np.savez("out_full.npz", X=XYZ, H=H, sj=sj, part=own, gov=gov)
    pickle.dump(tags, open("out_full_tags.pkl", "wb"))
    pickle.dump(nodes.key, open("out_full_keys.pkl", "wb"))


if __name__ == "__main__":
    main()
