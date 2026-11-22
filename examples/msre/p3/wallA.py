"""Band-A wall R692.15 on the structured section: clean the fluid selection (max-2 rule, no pinches),
pillow single-quad turns of the outer outline, and give every outline node its wall constraint."""
import pickle, collections
import numpy as np
import bsec, rules, secA, selA

R_W = rules.R_RING
TOP = dict(z=rules.Z["top"], ring=rules.R_PIN)


def adjacency(Q):
    edge_q = collections.defaultdict(list)
    for qi, q in enumerate(Q):
        for a in range(4):
            edge_q[frozenset((q[a], q[(a + 1) % 4]))].append(qi)
    nb = -np.ones((len(Q), 4), int)
    for qi, q in enumerate(Q):
        for a in range(4):
            for qj in edge_q[frozenset((q[a], q[(a + 1) % 4]))]:
                if qj != qi:
                    nb[qi, a] = qj
    return edge_q, nb


def clean(quads, sel, cen, solid, locked_in):
    """max 2 outline edges per element (adjacent ones only), near the wall; toggles locked."""
    Q = [q for q, _, _ in quads]
    edge_q, nb = adjacency(Q)
    r = np.hypot(*cen.T)
    zone = np.where(r > R_W - 40)[0]
    locked = np.zeros(len(Q), bool)
    for it in range(200):
        changed = 0
        for i in zone:
            d = [a for a in range(4) if nb[i, a] >= 0 and sel[nb[i, a]] != sel[i]]
            d += [a for a in range(4) if nb[i, a] < 0 and sel[i]]
            d = sorted(set(d))
            bad = len(d) >= 3 or (len(d) == 2 and d[1] - d[0] == 2)
            if not bad or locked[i]:
                continue
            want = not sel[i]
            if (want and solid[i]) or (not want and locked_in[i]):
                continue
            sel[i] = want; locked[i] = True; changed += 1
        # pinch nodes: an outline node shared by two separate outline passes
        deg = collections.Counter()
        for e, qs in edge_q.items():
            s_ = [sel[q] for q in qs]
            if (len(qs) == 2 and s_[0] != s_[1]) or (len(qs) == 1 and s_[0]):
                for v in e:
                    deg[v] += 1
        pinch = {v for v, n in deg.items() if n > 2}
        for i in zone:
            if not sel[i] and not solid[i] and any(v in pinch for v in Q[i]) and r[i] < R_W + 3:
                sel[i] = True; changed += 1
        if not changed:
            break
    return sel


def pillow_turns(quads, sel, loop):
    """pillow (inner quad + 4 trapezoids) every selected element that is the only quad inside a
    turn of the outline loop."""
    out = list(quads)
    sel = list(sel)
    pos = {v: i for i, v in enumerate(loop)}
    n = len(loop)
    done = 0
    inc = collections.defaultdict(list)
    for qi, (q, _, _) in enumerate(quads):
        for v in q:
            inc[v].append(qi)
    for k in range(n):
        u, v, w = loop[k - 1], loop[k], loop[(k + 1) % n]
        for qi in inc[v]:
            q, lab, c = out[qi]
            if not sel[qi] or v not in q:
                continue
            a = q.index(v)
            if {q[(a - 1) % 4], q[(a + 1) % 4]} == {u, w}:
                inner_ = [("pw", qi, j) for j in range(4)]
                out[qi] = (tuple(inner_), lab, c)
                for j in range(4):
                    out.append(((q[j], q[(j + 1) % 4], inner_[(j + 1) % 4], inner_[j]), lab, c))
                    sel.append(True)
                done += 1
    return out, np.array(sel), done


def build():
    template, quads, inner, alias, sel = pickle.load(open("selA.pkl", "rb"))
    cache = {}
    cen = np.array([np.mean([bsec.coords(k, TOP, inner, cache) for k in q], 0) for q, _, _ in quads])
    labs = np.array([l for _, l, _ in quads])
    solid = np.array([not s and np.hypot(*c) < R_W for s, c in zip(sel, cen)])   # pins etc.
    SITES = set(rules.sites())
    inwall = lambda c: isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], int) and c in SITES and \
        np.hypot(*np.array(c, float)) * rules.P + rules.R_PIN < R_W
    locked_in = (labs == "chan") | np.array([l == "oring" and inwall(c) for _, l, c in quads])
    sel = clean(quads, sel.copy(), cen, solid, locked_in)
    loops = secA.boundary_loops([q for q, s in zip(quads, sel) if s])
    loops.sort(key=len, reverse=True)
    outer = loops[0]
    quads2, sel2, npil = pillow_turns(quads, sel, outer)
    print("selected %d, outer outline %d nodes, pillowed %d turn elements" % (sel.sum(), len(outer), npil))
    pickle.dump((template, quads2, inner, alias, sel2, outer), open("wallA.pkl", "wb"))


if __name__ == "__main__":
    build()
