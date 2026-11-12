"""General face loop builder: graph edges + rim-core outlines + hub-core pieces + traced
edges are joined into chains at shared endpoint labels; chains are closed along the
plate (the face's trace in its map row), trying both pairings per plate row."""
import numpy as np, itertools
from collections import defaultdict
import items as IT
from mesher import graph, map_gov, to_map, TUBES


def pieces_of(A, X, rim_faces, hub_faces, trace_faces):
    edges, verts = graph()
    F = (A, X)
    P = []
    for i, e in enumerate(edges):
        if not e.get("alive", True) or any(g >= 1000 for g in e["gov"]):
            continue
        fl = e.get("faces")
        ok = (F in [tuple(sorted(f)) for f in fl]) if fl else (A in e["gov"] and X in e["gov"])
        if ok:
            P.append((("v", e["v"][0]), ("v", e["v"][1]), [("e", i, False)]))
    for cid, an in rim_faces.get(F, []):
        P.append((("c", cid, an, +1), ("c", cid, an, -1),
                  [("o", cid, an, "up", False), ("o", cid, an, "arc", False), ("o", cid, an, "down", False)]))
    for p in hub_faces.get(F, []):
        a, b = IT.hub_endpoints(p)
        P.append((a, b, [("h", p, False)]))
    for tid in trace_faces.get(F, []):
        a, b = IT.HUB["trace_ends"][tid]
        P.append((a, b, [("t", tid, False)]))
    return P


def rev_items(its):
    out = []
    for it in reversed(its):
        if it[0] == "e": out.append(("e", it[1], not it[2]))
        elif it[0] == "o": out.append(it[:4] + (not it[4],))
        elif it[0] in "ht": out.append((it[0], it[1], not it[2]))
        else: out.append((it[0], it[2], it[1]))
    return out


def chains(P):
    """join pieces at shared labels into chains (start, end, items)"""
    deg = defaultdict(list)
    for k, (a, b, its) in enumerate(P):
        deg[a].append(k); deg[b].append(k)
    used = set(); out = []
    starts = [l for l, ks in deg.items() if len(ks) == 1] + list(deg)
    for s in starts:
        for k in deg[s]:
            if k in used:
                continue
            cur, its = s, []
            kk = k
            while True:
                used.add(kk)
                a, b, pi = P[kk]
                if a == cur:
                    its += pi; cur = b
                else:
                    its += rev_items(pi); cur = a
                nxt = [j for j in deg[cur] if j not in used]
                if len(deg[cur]) != 2 or not nxt:
                    break
                kk = nxt[0]
            out.append((s, cur, its))
    return out


def is_plate(label):
    from geom import PLATE
    return abs(IT.endpoint_X(label)[1] - PLATE) < 0.05


def build_loop(A, X, rim_faces, hub_faces, trace_faces):
    M = map_gov(A, X)
    ch = chains(pieces_of(A, X, rim_faces, hub_faces, trace_faces))
    closed = [c for c in ch if c[0] == c[1]]
    if closed and len(ch) == 1:
        return closed[0][2]
    ends = [l for c in ch for l in (c[0], c[1])]
    bad = [l for l in ends if not is_plate(l)]
    assert not bad, ("chain ends off the plate", A, X, bad)
    # plate points: side and order along the row
    rows = defaultdict(list)
    for l in ends:
        x = IT.endpoint_X(l)
        s, ph = to_map(M, x[None])
        rows[int(s[0] > TUBES[M].L / 2)].append((ph[0], l))
    orders = {}
    for side, pts in rows.items():
        ang = np.array([p[0] for p in pts])
        o = np.argsort(ang); a = ang[o]
        gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
        cut = int(np.argmax(gaps)) + 1
        orders[side] = [pts[i][1] for i in np.roll(o, -cut)]
    sides = sorted(orders)
    for par in itertools.product((0, 1), repeat=len(sides)):
        pair = {}
        okp = True
        for side, p in zip(sides, par):
            L = orders[side]
            if len(L) % 2:
                okp = False; break
            seq = L if p == 0 else L[1:] + L[:1]
            for i in range(0, len(seq), 2):
                pair[seq[i]] = seq[i + 1]; pair[seq[i + 1]] = seq[i]
        if not okp:
            continue
        # walk
        by_end = defaultdict(list)
        for k, c in enumerate(ch):
            by_end[c[0]].append(k); by_end[c[1]].append(k)
        used, loop = set(), []
        k = 0; cur = ch[0][0]
        while k not in used:
            used.add(k)
            s0, e0, its = ch[k]
            if s0 == cur:
                loop += its; cur = e0
            else:
                loop += rev_items(its); cur = s0
            nxt = pair[cur]
            loop.append(("q", cur, nxt))
            cur = nxt
            cand = [j for j in by_end[cur] if j not in used]
            if not cand:
                break
            k = cand[0]
        if len(used) == len(ch) and cur == ch[0][0]:
            return loop
    raise AssertionError(("no closing pairing", A, X, len(ch)))
