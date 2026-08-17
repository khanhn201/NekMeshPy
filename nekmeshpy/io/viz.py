"""Matplotlib visualisation of a ``HexMesh`` (free function)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from ..hexmesh import HexMesh
from ..hexmesh.query import face_tag_rows

_log = logging.getLogger("nekmeshpy")


def plot(
    mesh: HexMesh,
    names: Sequence[str] = ("wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"),
    out_name: str = "carotid",
    panes: bool = False,
    grid: bool = True,
    save_path: str | None = None,
) -> HexMesh:
    """Draw the named boundary faces in colour order; panes transparent unless ``panes=True``."""
    import matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as exc:                     # pragma: no cover
        _log.warning("plot skipped (matplotlib unavailable): %s", exc)
        return mesh
    elements = mesh.points[mesh.corners]            # (N,8,3) per-element coords
    names = list(names)
    colors = [(0.80, 0.80, 0.82), (0.85, 0.20, 0.20),
              (0.20, 0.70, 0.25), (0.20, 0.35, 0.90)]
    alphas = [0.5, 1, 1, 1]
    fig = plt.figure(figsize=(7.8, 8.8))
    ax = fig.add_subplot(111, projection="3d")
    all_rows, all_names = face_tag_rows(mesh)
    handles = []
    for ti, name in enumerate(names):
        rows = all_rows[all_names == name]
        if not rows.shape[0]:
            continue
        polys = [elements[e, mesh.FACE_POINTS[s - 1, :], :] for e, s in rows.tolist()]
        pc = Poly3DCollection(polys, facecolor=colors[ti % len(colors)],
                              edgecolor=(0.15, 0.15, 0.15),
                              linewidths=0.2, alpha=alphas[ti % len(alphas)])
        ax.add_collection3d(pc)
        handles.append((plt.Rectangle((0, 0), 1, 1, fc=colors[ti % len(colors)]), name))
    allP = elements.reshape(-1, 3)
    mn = allP.min(axis=0)
    mx = allP.max(axis=0)
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    try:
        ax.set_box_aspect(mx - mn)
    except Exception:
        pass
    ax.view_init(elev=12, azim=-35)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if not panes:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(grid)
    if handles:
        ax.legend([h for h, _ in handles], [n for _, n in handles], loc="upper left")
    ax.set_title("Carotid hex mesh: tagged boundary")
    if save_path is None and matplotlib.get_backend().lower() == "agg":
        save_path = "%s_mesh.png" % out_name
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
        _log.info("wrote %s", save_path)
    else:
        plt.show()
    return mesh
