"""Loop items of a face boundary, uniform interface for loops / ILP / meshing.

('e', eid, rev)            graph edge
('p', va, vb)              plate segment between graph vertices
('q', ea, eb)              plate segment between endpoints ('v', vid) | ('c', cid, anchor, sgn)
('o', cid, anchor, part, rev)  core tombstone outline part: 'up' | 'arc' | 'down'
"""
import numpy as np
import mesher
from mesher import graph, to_map, lift, resample_poly, R, TUBES

RHO = R + 4.0
CORES = {}            # cid -> Core
CORE_OUT = {}         # cid -> built core dict (after build)


MPTS = {}            # extra plate points ('m', name) -> X
CUTS = {}            # cut name -> dict(pts, a, b)


def endpoint_X(ep):
    edges, verts = graph()
    if ep[0] == "v":
        return verts[ep[1]]["X"]
    if ep[0] == "m":
        return MPTS[ep]
    if ep[0] == "h":
        if ep[1] == "plate":
            return hub_geom()[ep[2]][0]
        if ep[1] == "T":
            return hub_geom()[("line", ep[2])][-1]
        return hub_geom()[("corner", ep[2], ep[3])][-1]
    c = CORES[ep[1]]
    g = c.outline_geom(ep[2], n=4)
    return g["up"][0] if ep[3] > 0 else g["down"][-1]


def pts(it):
    edges, verts = graph()
    t = it[0]
    if t == "e":
        P = edges[it[1]]["pts"]
        return P[::-1] if it[2] else P
    if t == "p":
        Xa, Xb = verts[it[1]]["X"], verts[it[2]]["X"]
    elif t == "q":
        Xa, Xb = endpoint_X(it[1]), endpoint_X(it[2])
    elif t == "h":
        g = hub_geom()[it[1]]
        return g[::-1] if it[2] else g
    elif t == "t":
        g = HUB["trace"][it[1]]
        return g[::-1] if it[2] else g
    elif t == "k":
        g = CUTS[it[1]]["pts"]
        return g[::-1] if it[2] else g
    else:
        g = CORES[it[1]].outline_geom(it[2], n=24)[it[3]]
        return g[::-1] if it[4] else g
    return np.array([Xa + s * (Xb - Xa) for s in np.linspace(0, 1, 8)])


def key(it, face):
    t = it[0]
    if t == "e":
        return ("e", it[1])
    if t == "p":
        return ("p", face, min(it[1], it[2]), max(it[1], it[2]))
    if t == "q":
        return ("q", face) + tuple(sorted([it[1], it[2]], key=str))
    if t == "h":
        return ("ha",) if it[1][0] == "top" else ("hz",)
    if t == "t":
        return ("t", it[1])
    if t == "k":
        return ("k", it[1])
    return ("oz", it[1]) if it[3] in ("up", "down") else ("oa", it[1])


def length(it):
    P = pts(it)
    L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    return L * (1.1 if it[0] in "pq" else 1.0)


def start_end(it):
    """endpoint labels (for loop walking)"""
    edges, verts = graph()
    t = it[0]
    if t == "e":
        a, b = edges[it[1]]["v"]
        a, b = ("v", a), ("v", b)
        return (b, a) if it[2] else (a, b)
    if t == "p":
        return ("v", it[1]), ("v", it[2])
    if t == "q":
        return it[1], it[2]
    cid, an, part, rev = it[1], it[2], it[3], it[4]
    ends = dict(up=(("c", cid, an, +1), ("c", cid, an, "A")), arc=(("c", cid, an, "A"), ("c", cid, an, "D")),
                down=(("c", cid, an, "D"), ("c", cid, an, -1)))[part]
    return (ends[1], ends[0]) if rev else ends


def ep_node(nodes, ep):
    edges, verts = graph()
    if ep[0] == "v":
        return nodes.get(("v", ep[1]), verts[ep[1]]["X"])
    if ep[0] == "m":
        return nodes.get(ep, MPTS[ep])
    if ep[0] == "h":
        pc = HUB["out"]["pieces"]
        if ep[1] == "plate":
            return pc[ep[2]][0]
        if ep[1] == "T":
            return pc[("line", ep[2])][-1]
        return pc[("corner", ep[2], ep[3])][-1]
    ol = CORE_OUT[ep[1]]["outline"][ep[2]]
    return ol[0] if ep[3] > 0 else ol[-1]


WARP = {}           # canonical item id -> (canonical split fractions, cumulative node index at each)


def canon(it, face):
    """global identity of an item and whether the loop runs against its canonical direction"""
    t = it[0]
    if t in "etk":
        return (t, it[1]), bool(it[2])
    if t == "h":
        return ("h", it[1]), bool(it[2])
    if t == "o":
        return ("o", it[1], it[2], it[3]), bool(it[4])
    ea, eb = (("v", it[1]), ("v", it[2])) if t == "p" else (it[1], it[2])
    lo_, hi_ = sorted([ea, eb], key=str)
    return ("pq", face, lo_, hi_), ea != lo_


def warp_fracs(cid, nf):
    """canonical arclength fraction of every node 0..nf (piecewise uniform between splits)"""
    w = WARP.get(cid)
    k = np.arange(nf + 1)
    if w is None:
        return k / nf
    us, cs = w
    assert cs[-1] == nf, ("warp count", cid, cs, nf)
    return np.interp(k, cs, us)


def resample_at(P, fr):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return np.stack([np.interp(fr * L[-1], L, P[:, k]) for k in range(3)], 1)


def node_ids(nodes, it, nf, M, other, face):
    """fine node ids along an item with nf segments"""
    edges, verts = graph()
    t = it[0]
    if t == "e":
        e = edges[it[1]]
        P = resample_at(e["pts"], warp_fracs(("e", it[1]), nf))
        ids = [nodes.get(("v", e["v"][0]), verts[e["v"][0]]["X"])]
        ids += [nodes.get(("e", it[1], nf, k), P[k]) for k in range(1, nf)]
        ids.append(nodes.get(("v", e["v"][1]), verts[e["v"][1]]["X"]))
        return ids[::-1] if it[2] else ids
    if t == "h":
        seg = HUB["out"]["pieces"][it[1]]
        assert len(seg) - 1 == nf, ("hub count", it, len(seg) - 1, nf)
        return seg[::-1] if it[2] else seg
    if t == "t":
        P = resample_at(HUB["trace"][it[1]], warp_fracs(("t", it[1]), nf))
        a, b = HUB["trace_ends"][it[1]]
        ids = [ep_node(nodes, a)] + [nodes.get(("t", it[1], nf, k), P[k]) for k in range(1, nf)] + [ep_node(nodes, b)]
        return ids[::-1] if it[2] else ids
    if t == "k":
        c = CUTS[it[1]]
        P = resample_at(c["pts"], warp_fracs(("k", it[1]), nf))
        ids = [ep_node(nodes, c["a"])] + [nodes.get(("k", it[1], nf, k), P[k]) for k in range(1, nf)] + [ep_node(nodes, c["b"])]
        return ids[::-1] if it[2] else ids
    if t == "o":
        cid, an, part, rev = it[1], it[2], it[3], it[4]
        ol = CORE_OUT[cid]["outline"][an]
        nz = CORE_OUT[cid]["nz"]
        seg = dict(up=ol[:nz + 1], arc=ol[nz:nz + 1 + CORE_OUT[cid]["narc"]], down=ol[nz + CORE_OUT[cid]["narc"]:])[part]
        assert len(seg) - 1 == nf, ("outline count", it, len(seg) - 1, nf)
        return seg[::-1] if rev else seg
    # plate segment (p or q)
    if t == "p":
        ea, eb = ("v", it[1]), ("v", it[2])
    else:
        ea, eb = it[1], it[2]
    ia, ib = ep_node(nodes, ea), ep_node(nodes, eb)
    Xa, Xb = nodes.X[ia], nodes.X[ib]
    sa, fa = to_map(M, np.array([Xa])); sb, fb = to_map(M, np.array([Xb]))
    fb = fa + (fb - fa + np.pi) % (2 * np.pi) - np.pi
    s_row = 0.0 if sa[0] < TUBES[M].L / 2 else TUBES[M].L
    lo_, hi_ = sorted([ea, eb], key=str)
    fr = warp_fracs(canon(it, face)[0], nf)
    lf = fr if ea == lo_ else 1.0 - fr[::-1]
    ph = fa[0] + lf * (fb[0] - fa[0])
    P, _ = lift(M, np.full(len(ph), s_row), ph, other)
    from geom import PLATE
    P[:, 1] = PLATE
    ids = [ia]
    for k in range(1, nf):
        kk = k if ea == lo_ else nf - k
        ids.append(nodes.get(("p", face, M, lo_, hi_, kk), P[k]))
    ids.append(ib)
    return ids


# ---------------------------------------------------------------- hub core items
HUB = {}          # 'core': HubCore, 'geom': {piece: polyline}, 'out': built dict, 'trace': {tid: polyline}
import os
HUB_COUNTS = dict(zip(("ma", "nz1", "nz2"), (int(v) for v in os.environ.get("EVG_HUB", "4,10,4").split(","))))


def hub_geom():
    """polylines of the hub pieces (independent of counts)"""
    if "geom" in HUB:
        return HUB["geom"]
    h = HUB["core"]
    G, Z = h.grid(8, 20, 6)
    ma, NN = 8, G.shape[1] - 1
    geom = {}
    for f, pcs in h.pieces().items():
        for p in pcs:
            if p[0] == "line":
                j = NN if p[1] > 0 else 0
                geom[p] = G[ma, j, :]
            elif p[0] == "corner":
                i = 2 * ma if p[1] > 0 else 0
                j = NN if p[2] > 0 else 0
                geom[p] = G[i, j, :]
            else:
                i0 = 2 * ma if p[1] > 0 else 0
                j = NN if p[2] > 0 else 0
                rng = range(i0, ma - 1, -1) if p[1] > 0 else range(0, ma + 1)
                geom[p] = np.array([G[i, j, -1] for i in rng])
    HUB["geom"] = geom
    return geom


def hub_endpoints(p):
    """labels of the two ends of hub piece p (as listed: plate/corner first)"""
    if p[0] == "line":
        return ("h", "plate", p), ("h", "T", p[1])
    if p[0] == "corner":
        return ("h", "plate", p), ("h", "C", p[1], p[2])
    return ("h", "C", p[1], p[2]), ("h", "T", p[2])
