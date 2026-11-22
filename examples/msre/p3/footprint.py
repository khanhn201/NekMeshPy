"""Channel footprints for the bottom section: the core's half-groove quads (10 per half)
with two coordinate configs per node -- TOP (= core section, z >= 411.286) and REF (the
exact slot/rod positions below the plenum).  Node ids are keyed by TOP coordinates, so
the TOP config reproduces the core section's channel quads bit for bit."""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "p1")
sys.path.insert(0, P1)
_cwd = os.getcwd(); os.chdir(P1)
import build_section as bs                       # noqa: E402  (lattice + template constants)
os.chdir(_cwd)
from geom import H, GL, GR, KB, IU, YU, YD, CENTRAL   # noqa: E402
import rules                                      # noqa: E402

UA = GL + np.sqrt(GR ** 2 - KB ** 2)
SITES = set(rules.sites())


def half_list():
    """(cell, side) of every half-groove of a FULL channel (half-channels excluded)."""
    out = []
    for c, t in bs.CELLS.items():
        if t not in ("full", "cut"):
            continue
        for s in range(4):
            if bs.GROOVE.get((c, s)) == "full":
                if t == "cut":
                    # cut cells keep only grooved triangles that survive the edge loft
                    outc = bs.corners_out(c)
                    if len(outc) == 1:
                        keep = [(outc[0] + 2) % 4, (outc[0] + 3) % 4]
                    else:
                        k = outc[0] if (outc[0] + 1) % 4 == outc[1] else outc[1]
                        keep = [(k + 3) % 4]
                    if s not in keep:
                        continue
                out.append((c, s))
    return out


def joint_coord(c, s):
    f = bs.frame(c, s)
    p = f(0, H)
    n = bs.D[s]
    return (p[0], "x") if abs(n[0]) > 0.5 else (p[1], "y")


def roles_local(c, s):
    """local (u, depth) of every role in the TOP and REF configs, and the move window."""
    jc, axis = joint_coord(c, s)
    central = c in CENTRAL or bs.CELLS.get((c[0] + int(bs.D[s][0]), c[1] + int(bs.D[s][1]))) is None
    top = dict(Jm=(-(GL + GR), 0), Jp=(GL + GR, 0), Km=(-IU, 0), K0=(0, 0), Kp=(IU, 0),
               Am=(-UA, KB), Ap=(UA, KB), Im=(-IU, KB), I0=(0, KB), Ip=(IU, KB),
               Ym=(-YU, YD), Y0=(0, YD), Yp=(YU, YD), Bm=(-GL, GR), M=(0, GR), Bp=(GL, GR))
    cc = np.array(c, float) * rules.P
    nrm = bs.D[s]
    if c in CENTRAL:
        return top, top, None
    if axis == "x":
        lo, hi = rules.y_slot(jc)
        zmove = (rules.Z["AB"], rules.Z["top"])
        nb = (c[0] + int(bs.D[s][0]), c[1] + int(bs.D[s][1]))
        both_posts = c in SITES and nb in SITES
        ut = rules.ROD_HW if (rules.is_rod_joint(jc) and both_posts) else IU
    else:
        lo, hi = rules.x_slot(jc)
        zmove = (rules.Z["BC"], rules.Z["top"])
        ut = IU
    cpos = cc[0] if axis == "x" else cc[1]
    d = jc - lo if cpos < jc else hi - jc            # slot-edge depth on this cell's side
    g = 6.35 - d                                      # room to the pin circle at u = 0
    ua = min(UA, ut + 2.5)
    ref = dict(Jm=(-(GL + GR), 0), Jp=(GL + GR, 0), Km=(-ut, 0), K0=(0, 0), Kp=(ut, 0),
               Am=(-ua, d), Ap=(ua, d), Im=(-ut, d), I0=(0, d), Ip=(ut, d),
               Ym=(-(ut - 1.0), d + 0.3 * g), Y0=(0, d + 0.3 * g), Yp=(ut - 1.0, d + 0.3 * g),
               Bm=(-ut, d + 0.6 * g), M=(0, d + 0.6 * g), Bp=(ut, d + 0.6 * g))
    return top, ref, zmove


QUADS = (("Jm", "Am", "Im", "Km"), ("Km", "Im", "I0", "K0"), ("K0", "I0", "Ip", "Kp"),
         ("Kp", "Ip", "Ap", "Jp"), ("Am", "Im", "Ym", "Bm"), ("Im", "I0", "Y0", "Ym"),
         ("Ym", "Y0", "M", "Bm"), ("Ip", "Ap", "Bp", "Yp"), ("I0", "Ip", "Yp", "Y0"),
         ("Y0", "Yp", "Bp", "M"))
BOUNDARY = ("Jm", "Am", "Bm", "M", "Bp", "Ap", "Jp")     # groove curve; joint closes it


def build():
    """nodes: {key: dict(top=(x,y), ref=(x,y), zmove=(z0,z1)|None)}, quads [(keys...)], halves."""
    nodes, quads, halves = {}, [], []
    for c, s in half_list():
        f = bs.frame(c, s)
        top, ref, zmove = roles_local(c, s)
        keys = {}
        for r in top:
            pt = f(top[r][0], H - top[r][1])
            pr = f(ref[r][0], H - ref[r][1])
            k = (round(pt[0], 6) + 0.0, round(pt[1], 6) + 0.0)
            if k in nodes:
                if np.abs(np.array(nodes[k]["ref"]) - pr).max() > 1e-9:
                    raise RuntimeError("REF mismatch at shared node %s (%s): %s vs %s" % (k, r, nodes[k]["ref"], pr))
            else:
                nodes[k] = dict(top=tuple(pt), ref=tuple(pr), zmove=zmove)
            keys[r] = k
        for q in QUADS:
            quads.append(tuple(keys[r] for r in q))
        halves.append(((c, s), keys))
    return nodes, quads, halves


if __name__ == "__main__":
    nodes, quads, halves = build()
    print("halves", len(halves), "nodes", len(nodes), "quads", len(quads))
    # validity of both configs (corner signed areas)
    for cfg in ("top", "ref"):
        X = {k: np.array(v[cfg]) for k, v in nodes.items()}
        areas = []
        for q in quads:
            p = np.array([X[k] for k in q])
            a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
            # orientation can be either way per half; compare to TOP orientation
            areas.append(a)
        areas = np.array(areas)
        print(cfg, "area range", areas.min().round(3), areas.max().round(3))
    Xt = {k: np.array(v["top"]) for k, v in nodes.items()}
    Xr = {k: np.array(v["ref"]) for k, v in nodes.items()}
    flips = 0
    for q in quads:
        pt = np.array([Xt[k] for k in q]); pr = np.array([Xr[k] for k in q])
        at = np.sum(pt[:, 0] * np.roll(pt[:, 1], -1) - np.roll(pt[:, 0], -1) * pt[:, 1])
        ar = np.sum(pr[:, 0] * np.roll(pr[:, 1], -1) - np.roll(pr[:, 0], -1) * pr[:, 1])
        flips += (at * ar <= 0)
    print("orientation flips TOP->REF:", flips)
