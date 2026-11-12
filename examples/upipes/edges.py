"""Voronoi edges from the tube label maps.

Every pair of map-adjacent pixels with different labels X, Y on tube A straddles the
Voronoi edge governed by (A, X, Y).  The exact switch is found by bisection along the
pixel pair (rays are re-marched), and the edge point is the tie point there.
Governor codes: pipe id >= 0; walls -1..-5; self part of pipe p: 1000+p; focal: 2000+p.
"""
import sys, time, numpy as np
from multiprocessing import Pool
from rays import march, SELF, FOCAL
from tubes import TUBES
from comps import clean


def code(lab, pid):
    lab = np.asarray(lab)
    return np.where(lab == SELF, 1000 + pid, np.where(lab == FOCAL, 2000 + pid, lab))


def map_edges(pid, steps=9):
    M = np.load(f"maps/tube_{pid}.npz")
    lab = code(clean(M["lab"]), pid)
    raw_to_clean = {}
    S, Ph = M["S"], M["Ph"]
    ns, nphi = lab.shape
    dphi = Ph[1] - Ph[0]
    A1, A2 = [], []
    # phi neighbours (periodic) and s neighbours
    i, j = np.meshgrid(np.arange(ns), np.arange(nphi), indexing="ij")
    j2 = (j + 1) % nphi
    m = lab[i, j] != lab[i, j2]
    A1.append(np.stack([i[m], j[m]], 1)); A2.append(np.stack([i[m], j2[m]], 1))
    i, j = np.meshgrid(np.arange(ns - 1), np.arange(nphi), indexing="ij")
    m = lab[i, j] != lab[i + 1, j]
    A1.append(np.stack([i[m], j[m]], 1)); A2.append(np.stack([i[m] + 1, j[m]], 1))
    p1 = np.concatenate(A1); p2 = np.concatenate(A2)
    X = lab[p1[:, 0], p1[:, 1]]; Y = lab[p2[:, 0], p2[:, 1]]
    s1, f1 = S[p1[:, 0]], Ph[p1[:, 1]]
    s2, f2 = S[p2[:, 0]], Ph[p2[:, 1]]
    f2 = np.where(f2 < f1 - np.pi, f2 + 2 * np.pi, f2)       # seam
    T = TUBES[pid]
    cand = list(M["cand"])
    P1 = M["P"][p1[:, 0], p1[:, 1]].copy(); P2 = M["P"][p2[:, 0], p2[:, 1]].copy()
    for _ in range(steps):
        sm, fm = 0.5 * (s1 + s2), 0.5 * (f1 + f2)
        _, lm, pm = march(T, sm, fm, cand)
        lm = code(lm, pid)
        if pid == 83:                       # same absorption as the cleanup
            lm = np.where(lm == 2083, 102, lm)
        side1 = lm == X
        s1 = np.where(side1, sm, s1); f1 = np.where(side1, fm, f1); P1 = np.where(side1[:, None], pm, P1)
        s2 = np.where(side1, s2, sm); f2 = np.where(side1, f2, fm); P2 = np.where(side1[:, None], P2, pm)
    E = 0.5 * (P1 + P2)
    key = np.sort(np.stack([np.full(len(X), pid), X, Y], 1), 1)
    sm = 0.5 * (s1 + s2); fm = 0.5 * (f1 + f2)
    return pid, key, E, np.stack([sm, fm], 1), np.linalg.norm(P1 - P2, axis=1)


if __name__ == "__main__":
    pids = sorted(TUBES)
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.map(map_edges, pids)
    out = {}
    for pid, key, E, sf, gap in res:
        out[f"key_{pid}"] = key; out[f"E_{pid}"] = E; out[f"sf_{pid}"] = sf; out[f"gap_{pid}"] = gap
        print(pid, len(E), "max tie gap", gap.max().round(3), flush=True)
    np.savez_compressed("maps/edgepts.npz", **out)
    print(f"{time.time()-t0:.0f}s")
