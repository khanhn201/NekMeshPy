"""connected components of each label on every tube map (phi periodic)."""
import numpy as np
from scipy import ndimage
from tubes import TUBES

def components(lab):
    """returns comp id array (same shape) and list of (label, npix)"""
    ns, nphi = lab.shape
    comp = -np.ones(lab.shape, int)
    info = []
    for v in np.unique(lab):
        m = lab == v
        cc, n = ndimage.label(m)
        # merge across phi seam
        parent = list(range(n + 1))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for i in range(ns):
            a, b = cc[i, 0], cc[i, -1]
            if a and b:
                ra, rb = find(a), find(b)
                if ra != rb: parent[ra] = rb
        roots = {}
        for k in range(1, n + 1):
            r = find(k)
            roots.setdefault(r, len(info) + len(roots))
        for k in range(1, n + 1):
            comp[cc == k] = roots[find(k)]
        for r, cid in sorted(roots.items(), key=lambda t: t[1]):
            info.append((int(v), int((comp == cid).sum())))
    return comp, info

def clean(lab, minpix=40, minwidth=2):
    """approximate-graph cleanup: absorb tiny or 1-2 px thin regions into neighbours"""
    lab = lab.copy()
    for it in range(10):
        comp, info = components(lab)
        changed = False
        for cid, (v, npx) in enumerate(info):
            m = comp == cid
            # thickness in phi: max run length per s row
            thin = m.sum(1).max() <= minwidth and m.sum(0).max() > 3 * minwidth
            if npx >= minpix and not thin:
                continue
            # neighbour labels
            d = ndimage.binary_dilation(m, structure=np.ones((3, 3)))
            d |= np.roll(m, 1, 1) | np.roll(m, -1, 1)
            ring = d & ~m
            if not ring.any():
                continue
            vals, cnt = np.unique(lab[ring], return_counts=True)
            lab[m] = vals[cnt.argmax()]
            changed = True
        if not changed:
            break
    return lab

if __name__ == "__main__":
    for pid in sorted(TUBES):
        M = np.load(f"maps/tube_{pid}.npz")
        lab0 = M["lab"]
        lab = clean(lab0)
        comp, info = components(lab)
        nchg = (lab != lab0).sum()
        print(pid, "changed", nchg, " comps:", [(v, n) for v, n in info])
