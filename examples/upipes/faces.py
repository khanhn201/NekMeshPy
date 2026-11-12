"""faces of the Voronoi graph: governor pairs, their boundary loops (edges + plate)."""
import numpy as np, pickle
from collections import defaultdict
from graph import gname
from tubes import TUBES

def is_core(g): return g >= 1000

def face_pairs():
    pairs = set()
    for pid in TUBES:
        M = np.load(f"maps/tube_{pid}.npz")
        from comps import clean
        for v in np.unique(clean(M["lab"])):
            if v in (-10, -11): continue
            if pid == 83 and v == -11: continue
            pairs.add(tuple(sorted((pid, int(v)))))
    return sorted(pairs)

def loops(edges, verts, A, X):
    """chains of graph edges bounding face (A, X); plate vertices close the chains"""
    E = [i for i, e in enumerate(edges) if A in e["gov"] and X in e["gov"]]
    inc = defaultdict(list)
    for i in E:
        a, b = edges[i]["v"]
        inc[a].append(i); inc[b].append(i)
    # walk chains between vertices of odd usage / plate vertices
    used = set(); chains = []
    ends = [v for v, l in inc.items() if len(l) != 2 or verts[v]["plate"]]
    for v0 in ends + list(inc):
        for i in inc[v0]:
            if i in used: continue
            chain = [i]; used.add(i); v = v0
            while True:
                a, b = edges[chain[-1]]["v"]
                v = b if a == v else a
                nxt = [j for j in inc[v] if j not in used]
                if verts[v]["plate"] or len(inc[v]) != 2 or not nxt:
                    break
                chain.append(nxt[0]); used.add(nxt[0])
            chains.append(chain)
    return E, chains

if __name__ == "__main__":
    edges, verts = pickle.load(open("maps/graph.pkl", "rb"))
    P = face_pairs()
    print(len(P), "faces")
    bad = 0
    for A, X in P:
        E, ch = loops(edges, verts, A, X)
        nplate = len({v for i in E for v in edges[i]["v"] if verts[v]["plate"]})
        odd = [v for v in {v for i in E for v in edges[i]["v"]}
               if sum(v in edges[i]["v"] for i in E) % 2 and not verts[v]["plate"]]
        flag = "" if not odd else "  OPEN at " + str([(v, [gname(g) for g in verts[v]["gov"]]) for v in odd])
        if flag or len(E) > 4:
            bad += bool(flag)
            print(gname(A), gname(X), "edges", [("/".join(gname(g) for g in edges[i]["gov"]), round(edges[i]["L"], 1)) for i in E], "plate", nplate, flag)
    print(bad, "faces with open boundary")
