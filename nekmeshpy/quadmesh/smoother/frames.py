"""Playback of a smoothing run: the vertex positions after every stage, as one small JSON file.

``smooth.run_schedule(..., record=True)`` keeps the positions after each stage; :func:`write_frames`
writes them with the connectivity (once) and, per frame, the worst-corner signed scaled Jacobian of every
quad, which is what the docs viewer (``docs/_static/smoother_viewer.js``) colours by.  Nothing here moves a
vertex or judges a mesh: the numbers are the ones :func:`smooth.quality` and :func:`smooth.corner_metrics`
already report, read at each frame.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..._typing import PointArray
from ..quadmesh import QuadMesh
from . import smooth

__all__ = ["frames_of", "write_frames"]


def frames_of(mesh: QuadMesh, logs: Sequence[smooth.StageLog]) -> list[tuple[str, PointArray]]:
    """``[(label, points)]``: the input, then the state after each recorded stage."""
    if any(lg.points is None for lg in logs):
        raise ValueError("the stage logs carry no positions; run the schedule with record=True")
    out = [("input", np.asarray(mesh.points, dtype=float))]
    for i, lg in enumerate(logs, 1):
        assert lg.points is not None
        out.append(("%d. %s" % (i, lg.method), lg.points))
    return out


def write_frames(path: str, mesh: QuadMesh, logs: Sequence[smooth.StageLog],
                 normals: PointArray | None = None, *, title: str = "", decimals: int = 6) -> dict[str, Any]:
    """Write the playback JSON of one run and return it.

    ``{"title", "quads": [[i0, i1, i2, i3]], "frames": [{"label", "points", "sj", "n_inverted",
    "sj_min", "sj_mean"}]}``: connectivity once, then per frame the ``(P, 3)`` points (rounded to
    ``decimals``), the per-quad worst-corner signed scaled Jacobian, and the summary the viewer prints.
    ``normals`` are the reference normals the run used (default: the input's own winding)."""
    nrm = smooth.reference_normals(mesh.points, mesh.corners) if normals is None else normals
    frames = []
    for label, pts in frames_of(mesh, logs):
        m = smooth.corner_metrics(pts, mesh.corners, nrm)
        sj = m.sj.min(axis=1)
        frames.append({"label": label, "points": np.round(pts, decimals).tolist(),
                       "sj": np.round(sj, 4).tolist(), "n_inverted": int((sj <= 0).sum()),
                       "sj_min": round(float(sj.min(initial=np.inf)), 4),
                       "sj_mean": round(float(sj.mean()) if sj.size else 0.0, 4)})
    doc = {"title": title, "quads": np.asarray(mesh.corners).tolist(), "frames": frames}
    with open(path, "w") as fid:
        json.dump(doc, fid, separators=(",", ":"))
    return doc
