"""Lower head zone (L2) in 3D:
  L2a  head shell S0 -> vane tops S1, vane quads removed (fins welded to the shell)
  L2b  S1 -> S2 - HC, morphing the vane configuration back to the KN positions
  cap  pillow-cap transition S2 - HC -> S2 onto the unpillowed KN band / downcomer
-> head_L2.pkl (HexMesh), head_L2.vtu"""
import os, sys, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bands as B
import head as HD
from nekmeshpy import quadmesh, hexmesh
from nekmeshpy.core.fields import gll_nodes
from scipy.spatial import cKDTree

ORDER = int(os.environ.get("ORDER", "2"))
NA, NB = 2, int(os.environ.get("NB", "2"))
HC = float(os.environ.get("HC", "3.0"))      # max cap thickness
DIP = float(os.environ.get("DIP", "0.5"))    # pillow dip / cap thickness
PSIZE = float(os.environ.get("PSIZE", "0.3"))  # pillow dip / parent size


def hc(r):
    """local cap thickness: HC, but at most 40% of the S1 -> S2 gap."""
    return np.minimum(HC, 0.4 * (HD.s2(r) - HD.s1(r)))
VP = pickle.load(open(os.path.join(HERE, "vanepos.pkl"), "rb"))
HS = pickle.load(open(os.path.join(HERE, "head_sec.pkl"), "rb"))
QUADS, KEYS, Q = VP["quads"], VP["keys"], VP["Q"]
NG = len(B.KEYS)                           # global (bands) nodes come first, same order
NQG = len(B.QUADS)
PW = VP["pillow"]
VANE = VP["vane"]
ANN_R = {1: HD.R_CAN, 2: HD.R_MID, 3: HD.R_VES}


HQ = np.array([[HS["keys"][k] for k in q] for q, _, _ in HS["quads"]])     # unpillowed section
HLAB = [l for _, l, _ in HS["quads"]]


def section(X, quads=None, labs=None):
    """Sec for node positions X (pillowed numbering, head_sec nodes first): bands' arcs on the
    global part, annulus circles; pillow inner edges straight."""
    quads = Q if quads is None else quads
    labs = [l for _, l, _ in QUADS] if labs is None else labs
    S = B.section(X[:NG], top=False)
    S.X = [tuple(p) for p in X]
    S.quads = [list(q) for q in quads]
    S.lab = list(labs)
    on = lambda i, r: abs(np.hypot(*X[i]) - r) < 1e-7
    for q in quads:
        for a in range(4):
            ia, ib = q[a], q[(a + 1) % 4]
            for r in ANN_R.values():
                if on(ia, r) and on(ib, r):
                    S.arc(ia, ib, np.zeros(2), r)
    Xa = np.array(S.X)
    for kk in list(S.cur):
        a, b_, spec = S.cur[kk]
        e = S.eval_edge(a, b_, np.array([0.0, 1.0]))
        if np.abs(e[0] - Xa[a]).max() > 1e-6 or np.abs(e[1] - Xa[b_]).max() > 1e-6:
            del S.cur[kk]
    return S


def qmesh(X, zf, mask=None):
    qm = section(X).to_quadmesh(ORDER, 0.0)
    qm = quadmesh.transform_fn(qm, lambda P: np.c_[P[:, :2], zf(P[:, :2])])
    if mask is not None:
        qm = quadmesh.select(qm, mask)
    return quadmesh.tag_edges(qm, quadmesh.boundary_edges(qm), "wall")


def zmix(fa, fb, t):
    return lambda P: (1 - t) * fa(np.hypot(P[:, 0], P[:, 1])) + t * fb(np.hypot(P[:, 0], P[:, 1]))


_BND = None


def valid_blend(Xa, Xb, f, thr=0.03):
    """(1-f) Xa + f Xb, untangled (q2.untangle, wall nodes on the vessel circle, other boundary
    nodes fixed) when the straight blend is not a valid 2D section."""
    global _BND
    from q2 import corner_sj, untangle
    P = (1 - f) * Xa + f * Xb
    if f in (0.0, 1.0) or corner_sj(P, Q).min() >= thr:
        return P
    if _BND is None:
        import collections
        ec = collections.Counter()
        for q in Q:
            for a in range(4):
                ec[frozenset((q[a], q[(a + 1) % 4]))] += 1
        _BND = np.zeros(len(P), bool)
        _BND[[v for e, c in ec.items() if c == 1 for v in e]] = True
    r0 = np.hypot(*P.T)
    wall = _BND & (r0 > HD.R_VES - 1e-6)
    free = ~_BND | wall
    proj = lambda i, p: HD.R_VES * p / np.linalg.norm(p) if wall[i] else p
    s0 = corner_sj(P, Q).min()
    P = untangle(P, Q, free, proj, target=0.05, rounds=100, grow=2)
    print("   blend f=%.3f: min corner sj %.3f -> %.3f" % (f, s0, corner_sj(P, Q).min()), flush=True)
    return P


def block(Xa, Xb, za, zb, nl, mask=None, tags=("wall", "wall")):
    """loft nl layers between (Xa, za) and (Xb, zb); heights blend linearly, positions too unless
    the straight blend is an invalid section (then it is untangled)."""
    g = gll_nodes(ORDER)
    fr = np.linspace(0, 1, nl + 1)
    sl, sw = [], []
    for k, f in enumerate(fr):
        sl.append(qmesh(valid_blend(Xa, Xb, f), zmix(za, zb, f), mask))
        if k < nl:
            sw.append([qmesh(valid_blend(Xa, Xb, ft), zmix(za, zb, ft), mask)
                       for ft in (f + t * (fr[k + 1] - f) for t in g[1:-1])])
    return hexmesh.loft(sl, sweep_nodes=sw, element_tags="fluid", first_tag=tags[0], last_tag=tags[1])


def cap_zmap(X, Sp=None):
    """z offset field of the cap's mid surface: 0 outside pillowed quads, -HC/2 on the pillow
    inner quad, linear across the trapezoids (pillow distance in the parent's bilinear frame)."""
    par = np.array(PW)
    corners = np.array([[X[HS["keys"][k]] if k in HS["keys"] else None for k in HS["quads"][qi][0]] for qi in par], float)
    tree = cKDTree(corners.mean(1))
    edge_tree = None
    if Sp is not None:
        g = gll_nodes(ORDER)
        pts = []
        for qi in par:
            q = HQ[qi]
            for a_ in range(4):
                pts.append(Sp.eval_edge(q[a_], q[(a_ + 1) % 4], g))
        edge_tree = cKDTree(np.concatenate(pts))

    def inv_bilinear(C, p):
        uv = np.full((len(p), 2), 0.5)
        for _ in range(60):
            u, v = uv[:, 0:1], uv[:, 1:2]
            x = (1 - u) * (1 - v) * C[:, 0] + u * (1 - v) * C[:, 1] + u * v * C[:, 2] + (1 - u) * v * C[:, 3]
            du = (1 - v) * (C[:, 1] - C[:, 0]) + v * (C[:, 2] - C[:, 3])
            dv = (1 - u) * (C[:, 3] - C[:, 0]) + u * (C[:, 2] - C[:, 1])
            J = np.stack([du, dv], -1)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            rr = p - x
            uv[:, 0] += (rr[:, 0] * J[:, 1, 1] - rr[:, 1] * J[:, 0, 1]) / det
            uv[:, 1] += (J[:, 0, 0] * rr[:, 1] - J[:, 1, 0] * rr[:, 0]) / det
        return uv

    inner = np.array([[X[KEYS[("pv", qi, j)]] for j in range(4)] for qi in par], float)
    tc = [np.stack([corners[:, j], corners[:, (j + 1) % 4], inner[:, (j + 1) % 4], inner[:, j]], 1) for j in range(4)]

    def bil_res(C, uv, p):
        u, v = uv[:, 0:1], uv[:, 1:2]
        x = (1 - u) * (1 - v) * C[:, 0] + u * (1 - v) * C[:, 1] + u * v * C[:, 2] + (1 - u) * v * C[:, 3]
        return np.linalg.norm(x - p, axis=1)

    def dz_exact(P):
        """per point: the one pillow parent containing it; weight 1 on its inner quad, the
        trapezoid's own coordinate across each trapezoid (0 on the parent edge, 1 on the inner
        edge), 0 outside every pillowed parent."""
        w = np.full(len(P), -1.0)
        _, idx = tree.query(P, k=4)
        for j in range(4):
            todo = w < 0
            if not todo.any():
                break
            ids = idx[todo, j]; Pt = P[todo]
            uvp = inv_bilinear(corners[ids], Pt)
            inside = (uvp >= -1e-7).all(1) & (uvp <= 1 + 1e-7).all(1) & (bil_res(corners[ids], uvp, Pt) < 1e-7)
            wt = np.full(len(Pt), -1.0)
            uv = inv_bilinear(inner[ids], Pt)
            wt = np.where(inside & (uv >= -1e-9).all(1) & (uv <= 1 + 1e-9).all(1) & (bil_res(inner[ids], uv, Pt) < 1e-7), 1.0, wt)
            for t in range(4):
                uv = inv_bilinear(tc[t][ids], Pt)
                ok = inside & (wt < 0) & (bil_res(tc[t][ids], uv, Pt) < 1e-7) & (uv[:, 0] >= -1e-7) & (uv[:, 0] <= 1 + 1e-7) & (uv[:, 1] >= -1e-7) & (uv[:, 1] <= 1 + 1e-7)
                wt = np.where(ok, np.clip(uv[:, 1], 0.0, 1.0), wt)
            wt = np.where(inside & (wt < 0), 0.0, wt)
            w[np.where(todo)[0][inside]] = wt[inside]
        w[w < 0] = 0.0
        if edge_tree is not None:
            dd, _ = edge_tree.query(P)
            w[dd < 1e-7] = 0.0
        return -DIP * hc(np.hypot(P[:, 0], P[:, 1])) * w
    if os.environ.get("DZ", "exact") == "exact":
        return dz_exact

    def dz_inner(P):
        """1 on the pillow inner quads (exactly), 0 elsewhere."""
        w = np.zeros(len(P))
        _, idx = tree.query(P, k=4)
        for j in range(4):
            uv = inv_bilinear(inner[idx[:, j]], P)
            w = np.where((uv >= -1e-9).all(1) & (uv <= 1 + 1e-9).all(1), 1.0, w)
        return w

    def dz(P):
        wmin = np.full(len(P), np.inf)
        _, idx = tree.query(P, k=4)
        for j in range(4):
            C = corners[idx[:, j]]
            uv = inv_bilinear(C, P)
            u, v = uv[:, 0:1], uv[:, 1:2]
            x = (1 - u) * (1 - v) * C[:, 0] + u * (1 - v) * C[:, 1] + u * v * C[:, 2] + (1 - u) * v * C[:, 3]
            ok = (uv >= -1e-6).all(1) & (uv <= 1 + 1e-6).all(1) & (np.linalg.norm(x - P, axis=1) < 1e-6)
            d = np.max(np.abs(uv - 0.5), 1) / 0.25
            w = np.where(d > 1.9, 0.0, np.where(d < 1.1, 1.0, np.clip(2.0 - d, 0.0, 1.0)))
            wmin = np.where(ok, np.minimum(wmin, w), wmin)
        wmin[~np.isfinite(wmin)] = 0.0
        if edge_tree is not None:
            dd, _ = edge_tree.query(P)
            wmin[dd < 1e-7] = 0.0
        if os.environ.get("DZ", "hybrid") == "hybrid":
            wmin = np.maximum(wmin, dz_inner(P))
        return -DIP * hc(np.hypot(P[:, 0], P[:, 1])) * wmin
    return dz


if __name__ == "__main__":
    t0 = time.time()
    Xv, Xk = VP["X_v"], VP["X_kn"]
    s0, s1, s2 = HD.s0, HD.s1, HD.s2
    s2c = (lambda r: s2(r) - hc(r)) if len(PW) else s2
    CAPONLY = bool(os.environ.get("CAPONLY"))
    if not CAPONLY:
        L2a = block(Xv, Xv, s0, s1, NA, mask=~VANE)
        print("L2a: %d hexes (%.0fs)" % (len(L2a.hexes), time.time() - t0), flush=True)
        for nb_ in range(NB, NB + 3):            # more layers until the vane -> band morph is untwisted
            L2b = block(Xv, Xk, s1, s2c, nb_)
            vb = hexmesh.validity(L2b)
            print("L2b: %d layers, %d hexes, inverted %d twisted %d (%.0fs)" % (nb_, len(L2b.hexes), vb.inverted.size, vb.twisted.size, time.time() - t0), flush=True)
            if vb.ok:
                break
    c1, c2 = None, []
    if len(PW):
        # cap layer 1: pillowed section, bottom at S2-HC, top = mid surface (S2, pillow inner quads dipped)
        dz = cap_zmap(Xk, section(Xk, HQ, HLAB))
        top = lambda P: s2(np.hypot(P[:, 0], P[:, 1])) + dz(P[:, :2])
        bot = lambda P: s2c(np.hypot(P[:, 0], P[:, 1]))
        c1 = hexmesh.loft([qmesh(Xk, bot), qmesh(Xk, top)], element_tags="fluid", first_tag="wall", last_tag="wall")
        # cap layer 2: one hex per pillowed parent, inner quad (mid surface) -> parent quad (S2);
        # parents coloured so that no two in one loft share a node
        color = {}
        inc = {}
        for qi in PW:
            for v in HQ[qi]:
                inc.setdefault(v, []).append(qi)
        for qi in PW:
            used = {color[o] for v in HQ[qi] for o in inc[v] if o in color}
            color[qi] = min(c for c in range(64) if c not in used)
        full_in = quadmesh.transform_fn(section(Xk).to_quadmesh(ORDER, 0.0), lambda P: np.c_[P[:, :2], top(P)])
        Spar = section(Xk, HQ, HLAB)
        INQ = {qi: Xk[Q[qi]] for qi in PW}                         # inner quad corners (straight edges)
        cen_in = {qi: INQ[qi].mean(0) for qi in PW}

        def inv_bil(C, p):
            uv = np.array([0.5, 0.5])
            for _ in range(40):
                u, v = uv
                x = (1 - u) * (1 - v) * C[0] + u * (1 - v) * C[1] + u * v * C[2] + (1 - u) * v * C[3]
                J = np.c_[(1 - v) * (C[1] - C[0]) + v * (C[2] - C[3]), (1 - u) * (C[3] - C[0]) + u * (C[2] - C[1])]
                uv = uv + np.linalg.solve(J, p - x)
            return uv

        def coons(qi, u, v):
            q = HQ[qi]
            e = lambda a_, b_, t: Spar.eval_edge(q[a_], q[b_], np.array([t]))[0]
            c = [np.array(Spar.X[q[k]]) for k in range(4)]
            bot, rgt, topp, lft = e(0, 1, u), e(1, 2, v), e(3, 2, u), e(0, 3, v)
            return ((1 - v) * bot + v * topp + (1 - u) * lft + u * rgt
                    - ((1 - u) * (1 - v) * c[0] + u * (1 - v) * c[1] + u * v * c[2] + (1 - u) * v * c[3]))

        c2 = []
        for cc in sorted(set(color.values())):
            ids = np.array(sorted(qi for qi in PW if color[qi] == cc))
            m_in = np.zeros(len(Q), bool); m_in[ids] = True          # pillowed quads[qi] is parent qi's inner quad
            bot_q = quadmesh.select(full_in, m_in)
            ctree = cKDTree(np.array([cen_in[qi] for qi in ids]))

            def expand_t(tf, ids=ids, ctree=ctree):
                def f(P):
                    out = np.empty_like(P)
                    _, j = ctree.query(P[:, :2])
                    onb = np.zeros(len(P), bool)
                    for k, p in enumerate(P):
                        qi = ids[j[k]]
                        uv = np.clip(inv_bil(INQ[qi], p[:2]), 0, 1)
                        onb[k] = min(uv[0], uv[1], 1 - uv[0], 1 - uv[1]) < 1e-7
                        xy = (1 - tf) * p[:2] + tf * coons(qi, *uv)
                        out[k, :2] = xy
                    zz = s2(np.hypot(out[:, 0], out[:, 1]))
                    if tf >= 1.0:
                        out[:, 2] = zz
                    else:
                        out[:, 2] = np.where(onb, zz + dz(out[:, :2]), (1 - tf) * P[:, 2] + tf * zz)
                    return out
                return f
            g = gll_nodes(ORDER)
            c2.append(hexmesh.loft([bot_q, quadmesh.transform_fn(bot_q, expand_t(1.0))],
                                   sweep_nodes=[[quadmesh.transform_fn(bot_q, expand_t(t)) for t in g[1:-1]]],
                                   element_tags="fluid", first_tag="", last_tag="wall"))
        print("cap: c1 %d hexes, c2 %d hexes in %d colours (%.0fs)" % (len(c1.hexes), sum(len(c.hexes) for c in c2), len(c2), time.time() - t0), flush=True)
        if CAPONLY:
            for nm_, blk_ in [("c1", c1)] + [("c2", c) for c in c2]:
                print("  %s curved %s" % (nm_, hexmesh.quality_summary(blk_)), flush=True)
            sys.exit()
    def edge_diff(A, Bm):
        la, lb = A.quad_mesh.line_mesh, Bm.quad_mesh.line_mesh
        Pa, Pb = np.asarray(la.points), np.asarray(lb.points)
        ta = cKDTree(Pa)
        d, ia = ta.query(Pb)
        ea = {frozenset(e): i for i, e in enumerate(np.asarray(la.lines).tolist())}
        Ia, Ib = np.asarray(la.interior), np.asarray(lb.interior)
        worst = []
        for li, (u, v) in enumerate(np.asarray(lb.lines)):
            if d[u] < 1e-6 and d[v] < 1e-6:
                k = ea.get(frozenset((ia[u], ia[v])))
                if k is None:
                    continue
                A_ = Ia[k] if np.asarray(la.lines)[k][0] == ia[u] else Ia[k][::-1]
                worst.append((np.abs(A_ - Ib[li]).max(), li, Pb[u], Pb[v], A_, Ib[li]))
        worst.sort(key=lambda w: -w[0])
        return worst[:5]
    for w in (edge_diff(c1, c2[0]) if c2 else []):
        print("  c1/c2 edge diff %.3g at %s - %s\n     c1 %s\n     c2 %s" % (w[0], np.round(w[2], 2), np.round(w[3], 2), np.round(w[4], 3).tolist(), np.round(w[5], 3).tolist()), flush=True)
    if os.environ.get("STOPDBG"): sys.exit()
    for nm, pair in ((("L2a+L2b", [L2a, L2b]),) + ((("L2b+c1", [L2b, c1]), ("c1+c2", [c1] + c2)) if c1 is not None else ())):
        try:
            hexmesh.merge(pair, clear_seam_tags=True); print("  merge ok:", nm, flush=True)
        except ValueError as ex:
            print("  merge FAIL:", nm, str(ex)[:80], flush=True)
    mesh = hexmesh.merge([L2a, L2b] + ([c1] + c2 if c1 is not None else []), clear_seam_tags=True)
    print("L2 merged: %d hexes (%.0fs)" % (len(mesh.hexes), time.time() - t0), flush=True)
    print(hexmesh.topology_report(mesh))
    print(hexmesh.quality_summary(mesh))
    print("corner:", hexmesh.corner_summary(mesh))
    pickle.dump(mesh, open(os.path.join(HERE, os.environ.get("OUTL2", "head_L2.pkl")), "wb"))
    if os.environ.get("NOVTU"): sys.exit()
    from nekmeshpy.io.writer import to_vtu
    to_vtu(mesh, os.path.join(HERE, "head_L2.vtu"))
    print("done (%.0fs)" % (time.time() - t0))
