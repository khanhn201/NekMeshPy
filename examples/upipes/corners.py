"""corner structure of every face loop (in its map) -> structured candidates."""
import numpy as np
from mesher import graph, face_loop, map_gov, to_map, R
from faces import face_pairs

def loop_sides(A, X, thr=40.0, loop=None):
    """split the face loop into sides at corners (turn > thr deg in the map).
    returns list of sides, each a list of loop items, and the corner turns"""
    edges, verts = graph()
    if loop is None:
        loop = face_loop(A, X)
    M = map_gov(A, X)
    # tangent direction at the start and end of every item, in the map
    import items as IT
    item_pts = IT.pts
    dirs = []
    for it in loop:
        P = item_pts(it)
        s, ph = to_map(M, P); ph = np.unwrap(ph)
        uv = np.stack([s, ph * (R + 4)], 1)
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
        k0 = int(np.clip(np.searchsorted(L, min(6.0, L[-1] / 3)), 1, len(uv) - 1))
        k1 = int(np.clip(np.searchsorted(L, L[-1] - min(6.0, L[-1] / 3)) - 1, 0, len(uv) - 2))
        d0 = uv[k0] - uv[0]; d1 = uv[-1] - uv[k1]
        dirs.append((np.arctan2(d0[1], d0[0]), np.arctan2(d1[1], d1[0])))
    turns = []
    for i in range(len(loop)):
        a = dirs[i - 1][1]; b = dirs[i][0]
        turns.append(np.degrees((b - a + np.pi) % (2 * np.pi) - np.pi))
    turns = np.array(turns)
    corner = np.abs(turns) > thr
    idx = np.where(corner)[0]
    if len(idx) == 0:
        return [loop], turns
    sides = []
    for j in range(len(idx)):
        a, b = idx[j], idx[(j + 1) % len(idx)]
        rng = list(range(a, b)) if b > a else list(range(a, len(loop))) + list(range(0, b))
        sides.append([loop[k] for k in rng])
    return sides, turns

if __name__ == "__main__":
    from collections import Counter
    from graph import gname
    HUB = {81, 82, 83}
    edges, verts = graph()
    cnt = Counter(); odd = []
    for A, X in face_pairs():
        if A in HUB or X in HUB: continue
        if any(A in e["gov"] and X in e["gov"] and any(g >= 1000 for g in e["gov"]) for e in edges): continue
        try:
            sides, turns = loop_sides(A, X)
        except AssertionError as e:
            odd.append((A, X, "open")); continue
        cnt[len(sides)] += 1
        if len(sides) != 4:
            odd.append((gname(A), gname(X), len(sides), np.round(turns, 0).tolist()))
    print(cnt)
    for o in odd: print(o)
