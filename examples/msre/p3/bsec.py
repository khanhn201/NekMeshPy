"""Bottom-zone section topology (regular lattice cells) with per-config node coordinates.

Global lines: for every x-joint m (x = 25.4 + 50.8 m) three lines along y -- 'J' (the joint),
'L' / 'R' (the merged slot/plate "jline" of the cell left / right of it); same for y-joints
with 'B' / 'T'.  Every crossing of an x-line and a y-line near a joint crossing is one node.
A cell = 4 corner squares + 4 side strips (footprint joint strip inside) + inner O-grid
(footprint D regions on its boundary -> ring circle -> disc)."""
import numpy as np
import footprint as fp
import rules
from geom import H, CENTRAL

FP_NODES, FP_QUADS, FP_HALVES = fp.build()
HALF = {cs: keys for cs, keys in FP_HALVES}
SITES = set(rules.sites())
D4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def xj(m):
    return 25.4 + rules.P * m


def side_lines(c, s):
    """(joint line id, jline id) of side s of cell c; ids are ('x'|'y', kind, m)."""
    i, j = c
    if s == 0:
        return ("x", "J", i), ("x", "L", i)
    if s == 2:
        return ("x", "J", i - 1), ("x", "R", i - 1)
    if s == 1:
        return ("y", "J", j), ("y", "B", j)
    return ("y", "J", j - 1), ("y", "T", j - 1)


def cross(a, b):
    xa, ya = (a, b) if a[0] == "x" else (b, a)
    return ("X", xa, ya)


def regular_cells():
    """full stringer cells handled by the plain template (no central / bore / cut neighbours)."""
    bad = set(CENTRAL) | {(1, 1), (-1, 1), (-1, -1), (1, -1)}
    out = []
    for c, t in fp.bs.CELLS.items():
        if t != "full" or c in bad:
            continue
        if any(fp.bs.CELLS.get((c[0] + d[0], c[1] + d[1])) in ("thimble", "basket") for d in D4):
            continue
        if any(fp.bs.CELLS.get((c[0] + d[0], c[1] + d[1])) != "full" for d in D4):
            continue
        out.append(c)
    return out


BORES = {(1, 1), (-1, 1), (-1, -1), (1, -1)}
T_PIN = 16.51          # pins next to a bore: 1.27 clear of the bore bulge (compromise, was 19.05)


def t_side(c, s):
    """True if side s of stringer cell c faces a bore (thimble / basket site)."""
    return (c[0] + D4[s][0], c[1] + D4[s][1]) in BORES and c not in BORES


def t_nodes(c, s):
    """the 7 bore-boundary nodes on a T side (E-, X1-, X2-, Xc, X2+, X1+, E+), keyed by their
    exact coordinates (= the core section's nodes)."""
    from geom import R_OPEN, XU1, XU2
    f = fp.bs.frame(c, s)
    e = np.sqrt(R_OPEN ** 2 - H ** 2)
    out = []
    for u in (-e, -XU1, -XU2, 0.0, XU2, XU1, e):
        p = f(u, 2 * H - np.sqrt(R_OPEN ** 2 - u ** 2))
        k = (round(p[0], 6) + 0.0, round(p[1], 6) + 0.0)
        out.append(k)
    return out


def pin_r(c):
    return T_PIN if any(t_side(c, s) for s in range(4)) else rules.R_PIN


def build(cells):
    quads, inner = [], {}
    for c in cells:
        bnd = []                      # inner O-grid boundary, CCW
        for s in range(4):
            jl, jn = side_lines(c, s)
            jl_prev, jn_prev = side_lines(c, (s - 1) % 4)
            jl_next, jn_next = side_lines(c, (s + 1) % 4)
            # corner square at the start corner of side s
            C, Cj = cross(jl, jl_prev), cross(jl, jn_prev)
            Cq, Cp = cross(jn, jn_prev), cross(jn, jl_prev)
            if t_side(c, s):
                Cj = t_nodes(c, s)[0]
            if t_side(c, (s - 1) % 4):
                Cp = t_nodes(c, (s - 1) % 4)[-1]
            quads.append(((C, Cj, Cq, Cp), "corner", c))
            Ej, Eq = cross(jl, jn_next), cross(jn, jn_next)
            keys = HALF.get((c, s))
            if t_side(c, s):
                bnd += [Cq] + t_nodes(c, s)
            elif keys is not None:
                quads.append(((Cj, keys["Jm"], keys["Am"], Cq), "strip", c))
                quads.append(((keys["Jp"], Ej, Eq, keys["Ap"]), "strip", c))
                bnd += [Cq, keys["Am"], keys["Bm"], keys["M"], keys["Bp"], keys["Ap"]]
            else:
                # plain side: 5 points along the joint and the jline, same count as a footprint side
                mj = [("mj", c, s, t) for t in range(5)]
                mq = [("mq", c, s, t) for t in range(5)]
                jj = [Cj] + mj + [Ej]
                qq = [Cq] + mq + [Eq]
                for t in range(6):
                    quads.append(((jj[t], jj[t + 1], qq[t + 1], qq[t]), "strip", c))
                bnd += [Cq] + mq
        # start the O-grid at a T side if there is one (its 8 nodes face a 7-segment square side)
        ts = [s for s in range(4) if t_side(c, s)]
        if ts:
            k0 = 6 * ts[0]
            bnd = bnd[k0:] + bnd[:k0]
        n = len(bnd)
        assert n in (24, 26, 28), n
        na, nb = {24: (6, 6), 26: (7, 6), 28: (8, 6)}[n]
        ring = [("ring", c, k) for k in range(n)]
        sqb = [("sqb", c, k) for k in range(n)]
        for k in range(n):
            l = (k + 1) % n
            quads.append(((bnd[k], bnd[l], ring[l], ring[k]), "oring", c))
            quads.append(((ring[k], ring[l], sqb[l], sqb[k]), "disc", c))
        g = {}
        for a in range(1, na):
            for b in range(1, nb):
                g[(a, b)] = ("sqi", c, a, b)
        for k, ab in enumerate(walk_of(na, nb)):
            g[ab] = sqb[k]
        for a in range(na):
            for b in range(nb):
                quads.append(((g[(a, b)], g[(a + 1, b)], g[(a + 1, b + 1)], g[(a, b + 1)]), "disc", c))
        inner[c] = bnd
    # footprint quads of these cells
    have = set(cells)
    for (c, s), keys in FP_HALVES:
        if c in have:
            for q in fp.QUADS:
                quads.append((tuple(keys[r] for r in q), "chan", c))
    return quads, inner


# ---------------------------------------------------------------- configs --
def walk_of(na, nb):
    return ([(t, 0) for t in range(na)] + [(na, t) for t in range(nb)] +
            [(na - t, nb) for t in range(na)] + [(0, nb - t) for t in range(nb)])


def ring_r(cfg, c):
    return T_PIN if (cfg["ring"] == rules.R_PIN and c in T_CELLS) else cfg["ring"]


T_CELLS = set()
def _frac(z, ax):
    z0 = rules.Z["AB"] if ax == "x" else rules.Z["BC"]
    return float(np.clip((z - z0) / (rules.Z["top"] - z0), 0.0, 1.0))


def line_pos(lid, cfg):
    ax, kind, m = lid
    j = xj(m)
    if kind == "J":
        return j
    if "z" in cfg:
        f = _frac(cfg["z"], ax)
        a = line_pos(lid, dict(jline="slot")); b = line_pos(lid, dict(jline="top"))
        return a + f * (b - a)
    lo, hi = (rules.y_slot(j) if ax == "x" else rules.x_slot(j))
    if cfg["jline"] == "slot":
        return lo if kind in ("L", "B") else hi
    if cfg["jline"] == "top":
        return j - fp.KB if kind in ("L", "B") else j + fp.KB
    raise ValueError(cfg)


def coords(key, cfg, inner, cache):
    if key in cache:
        return cache[key]
    if isinstance(key[0], (float, np.floating)) and key not in FP_NODES:
        p = np.array(key, float)                          # static node (bore boundary etc.)
        cache[key] = p
        return p
    if isinstance(key[0], (float, np.floating)):
        nd = FP_NODES[key]
        if "z" in cfg:
            zm = nd["zmove"]
            f = 1.0 if zm is None else float(np.clip((cfg["z"] - zm[0]) / (zm[1] - zm[0]), 0, 1))
            p = np.array(nd["ref"]) + f * (np.array(nd["top"]) - np.array(nd["ref"]))
        else:
            p = np.array(nd[cfg["fp"]])
    elif key[0] == "X":
        _, a, b = key
        p = np.array([line_pos(a, cfg), line_pos(b, cfg)])
    elif key[0] in ("mj", "mq"):
        _, c, s, t = key
        jl, jn = side_lines(c, s)
        jl_prev, jn_prev = side_lines(c, (s - 1) % 4)
        jl_next, jn_next = side_lines(c, (s + 1) % 4)
        lid = jl if key[0] == "mj" else jn
        a = coords(cross(lid, jn_prev), cfg, inner, cache)
        b = coords(cross(lid, jn_next), cfg, inner, cache)
        p = a + (t + 1) / 6.0 * (b - a)
    elif key[0] in ("sqb", "sqi"):
        c = key[1]
        cc = np.array(c, float) * rules.P
        n = len(inner[c])
        na, nb = {24: (6, 6), 26: (7, 6), 28: (8, 6)}[n]
        h = 0.55 * ring_r(cfg, c)
        walk = walk_of(na, nb)
        if key[0] == "sqb":
            # evenly spaced along the square, corner k=0 towards the ring's node 0
            k = key[2]
            q0 = coords(("ring", c, 0), cfg, inner, cache) - cc
            a0 = np.arctan2(q0[1], q0[0])
            corners = [cc + h * np.sqrt(2) * np.array([np.cos(a0 + j * np.pi / 2), np.sin(a0 + j * np.pi / 2)]) for j in range(5)]
            lens = [na, nb, na, nb]
            sd = 0; tt = k
            while tt >= lens[sd]:
                tt -= lens[sd]; sd += 1
            p = corners[sd] + (tt / lens[sd]) * (corners[sd + 1] - corners[sd])
        else:
            _, _, a, b = key
            u, v = a / na, b / nb
            widx = {ab: i for i, ab in enumerate(walk)}
            E = lambda ab: coords(("sqb", c, widx[ab]), cfg, inner, cache)
            bot, top = E((a, 0)), E((a, nb))
            lef, rig = E((0, b)), E((na, b))
            p = ((1 - v) * bot + v * top + (1 - u) * lef + u * rig
                 - ((1 - u) * (1 - v)) * E((0, 0)) - (u * (1 - v)) * E((na, 0))
                 - (u * v) * E((na, nb)) - ((1 - u) * v) * E((0, nb)))
    elif key[0] == "ring":
        _, c, k = key
        cc = np.array(c, float) * rules.P
        ov = cfg.get("ring_override", {}).get((c, k))
        if ov is not None:
            cache[key] = np.array(ov)
            return cache[key]
        q = coords(inner[c][k], cfg, inner, cache) - cc
        p = cc + ring_r(cfg, c) * q / np.linalg.norm(q)
    else:
        raise KeyError(key)
    cache[key] = p
    return p


def check(quads, inner, cfg, ref=None):
    cache = {}
    A = []
    for q, lab, c in quads:
        P = np.array([coords(k, cfg, inner, cache) for k in q])
        A.append(0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    return np.array(A)


if __name__ == "__main__":
    import collections
    cells = regular_cells()
    quads, inner = build(cells)
    print("cells", len(cells), "quads", len(quads), collections.Counter(l for _, l, _ in quads))
    REF = dict(jline="slot", fp="ref", ring=rules.R_PIN)
    TOP = dict(jline="top", fp="top", ring=rules.R_PIN)
    POST = dict(jline="slot", fp="ref", ring=rules.R_POST)
    a0 = check(quads, inner, TOP)
    sgn = np.sign(a0)
    for name, cfg in (("TOP", TOP), ("REF/pin", REF), ("REF/post", POST)):
        a = check(quads, inner, cfg)
        bad = (np.sign(a) != sgn) | (np.abs(a) < 1e-9)
        labs = collections.Counter(quads[i][1] for i in np.where(bad)[0])
        print(name, "flipped/degenerate:", bad.sum(), dict(labs), "min |area| %.3g" % np.abs(a).min())
    print("TOP orientation mix:", collections.Counter(sgn))


T_CELLS.update(c for c, t in fp.bs.CELLS.items()
               if t == "full" and c not in CENTRAL and any(t_side(c, s) for s in range(4)))
