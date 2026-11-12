"""quality of every boundary surface: pipes (in their (s,R phi) map), walls, plate (in-plane)"""
import numpy as np, sys
from tubes import TUBES, BOX
from geom import R, PLATE
D = np.load(sys.argv[1]); X, H = D["X"], D["H"]
F6 = np.concatenate([H[:, [0,3,2,1]], H[:, [4,5,6,7]], H[:, [0,1,5,4]], H[:, [1,2,6,5]], H[:, [2,3,7,6]], H[:, [3,0,4,7]]])
u, idx, inv, cnt = np.unique(np.sort(F6, 1), axis=0, return_index=True, return_inverse=True, return_counts=True)
B = F6[idx[cnt == 1]]
lo, hi = BOX
def stats(name, uv):
    L = np.linalg.norm(np.roll(uv, -1, 1) - uv, axis=2); ar = L.max(1) / np.maximum(L.min(1), 1e-12)
    e1 = np.roll(uv, -1, 1) - uv; e2 = np.roll(uv, 1, 1) - uv
    cr = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    sg = np.sign(np.median(cr))
    sj = sg * cr / (np.linalg.norm(e1, axis=2) * np.linalg.norm(e2, axis=2))
    return (name, len(uv), ar.max(), (ar > 8).sum(), sj.min(1).min(), (sj.min(1) < 0.3).sum())
rows = []
allpipe = []
for p, T in TUBES.items():
    d = np.abs(T.dist(X[B].reshape(-1, 3))[0]).reshape(-1, 4)
    Q = B[(d < 5e-3).all(1)]
    _, s = T.dist(X[Q].reshape(-1, 3)); Tg, N, Bn = T.frame(s); v = X[Q].reshape(-1, 3) - T.spine(s)
    ph = np.arctan2(np.einsum("ij,ij->i", v, Bn), np.einsum("ij,ij->i", v, N)).reshape(-1, 4)
    ph = ph[:, :1] + (ph - ph[:, :1] + np.pi) % (2 * np.pi) - np.pi
    allpipe.append(stats(p, np.stack([s.reshape(-1, 4), R * ph], -1)))
for nm, ax, v, (i, j) in (("x_min", 0, lo[0], (1, 2)), ("x_max", 0, hi[0], (1, 2)), ("z_min", 2, lo[2], (0, 1)), ("z_max", 2, hi[2], (0, 1)), ("top", 1, hi[1], (0, 2)), ("plate", 1, PLATE, (0, 2))):
    Q = B[np.all(np.abs(X[B][..., ax] - v) < 1e-6, 1)]
    rows.append(stats(nm, X[Q][..., [i, j]]))
pa = np.array([r[2] for r in allpipe]); ps = np.array([r[4] for r in allpipe])
print("pipes: quads %d, max aspect %.1f, aspect>8 %d, min corner sin %.2f, quads with corner sin<0.3: %d"
      % (sum(r[1] for r in allpipe), pa.max(), sum(r[3] for r in allpipe), ps.min(), sum(r[5] for r in allpipe)))
for r in rows:
    print("%-6s quads %5d, max aspect %6.1f, aspect>8 %4d, min corner sin %5.2f, sin<0.3: %d" % r)
