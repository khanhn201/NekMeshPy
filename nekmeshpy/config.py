"""Parameters and options for the bifurcation hex-mesh generator (surface
pipeline).  ``Config`` is a plain dataclass -- instantiate it directly and pass
it to :class:`~nekmeshpy.algorithms.bifurcation.BifurcationMesher`::

    from nekmeshpy import Config, BifurcationMesher
    cfg = Config()
    cfg.interior_method = "winslow"
    BifurcationMesher(cfg).run()

Paths in ``vtx_file`` / ``tri_file`` are resolved to the packaged ``data/``
folder by default.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, List

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # ---- input surface (STL-equivalent node-list + triangle-index files) ---
    vtx_file: str = os.path.join(_HERE, "data", "car.vtx")   # vertices  (N x 3)
    tri_file: str = os.path.join(_HERE, "data", "car.tri")   # triangles (M x 3, 1-based)

    # ---- partition / seam --------------------------------------------------
    n_half: int = 8                 # half-ring resolution; MULTIPLE OF 4
                                    # (each cross-section ring has 2*n_half pts)
    seam_smooth_iters: int = 8      # cut-seam smoothing iterations
    seam_smooth_lambda: float = 0.6  # cut-seam smoothing relaxation

    # Selects the WHOLE pipeline.  Only 'surface' is ported here.
    method: str = "surface"         # 'surface'  (only surface is ported)
    seam_npb: int = 24              # [volumetric] tet resolution (unused here)
    iso_blend: float = 0            # [volumetric] (unused here)

    # ---- slicing along each leg --------------------------------------------
    n_slices: int = 20              # cross-sections per leg (hex layers = n_slices)
    min_loop_pts: int = 6           # ignore isocontour loops smaller than this

    # ---- cross-section O-grid ----------------------------------------------
    center_scale: float = 0.5       # inner square-core size (fraction of diameter)

    # Radial layer positions of the O-ring, from the core edge (0) to the wall
    # (last entry MUST be 1).  Number of entries = radial cells.
    radial: List[float] = field(default_factory=lambda: [0.4, 0.8, 1.0])

    # ---- options -----------------------------------------------------------
    resample_spline: bool = True    # spline-smooth + uniform resample along leg
    project_to_stl: bool = True     # snap wall/ring nodes back onto the STL
    plot: bool = True               # show/save the meshed result

    # How the cross-section INTERIOR is meshed (SURFACE mode):
    #   'bilinear' | 'harmonic' | 'harmonic3d' | 'winslow'
    interior_method: str = "harmonic"

    # ---- hex-mesh smoothing (post-assembly) --------------------------------
    smooth_iters: int = 8           # smoothing sweeps (0 = off)
    smooth_lambda: float = 0.5      # relaxation factor in (0,1]

    # ---- boundary tags -----------------------------------------------------
    tag_wall: int = 1               # vessel walls
    tag_trunk: int = 2              # trunk (leg A) outlet
    tag_top1: int = 3              # branch B outlet
    tag_top2: int = 4              # branch C outlet

    # ---- flux-measurement planes -------------------------------------------
    tag_f1: int = 5                 # flux plane near the tag_top2 ('O  ') outlet
    tag_f2: int = 6                 # flux plane near the tag_top1 ('int') outlet
    flux_offset: int = 2            # hex layers in from the outlet cap (0 = off)

    # ---- export ------------------------------------------------------------
    out_name: str = "bifurcation"   # base name for output files
    export_re2: bool = True         # write <out_name>.re2 / .rea (Nek5000/NekRS)
    export_vtk: bool = True         # write <out_name>.vtk (visualization)

    # -- serialization / validation -----------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (JSON/YAML-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Build a Config from a dict, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError("unknown config keys: %s" % ", ".join(sorted(unknown)))
        return cls(**d)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load a Config from a ``.yaml``/``.yml`` or ``.json`` file."""
        with open(path, "r") as fh:
            text = fh.read()
        if path.lower().endswith((".yaml", ".yml")):
            import yaml
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def save(self, path: str) -> str:
        """Write the config to ``.yaml``/``.yml`` or ``.json``."""
        d = self.to_dict()
        with open(path, "w") as fh:
            if path.lower().endswith((".yaml", ".yml")):
                import yaml
                yaml.safe_dump(d, fh, sort_keys=False)
            else:
                json.dump(d, fh, indent=2)
        return path

    def validate(self) -> "Config":
        """Raise ``ValueError`` on inconsistent parameters; returns ``self``."""
        errs = []
        if self.n_half % 4 != 0 or self.n_half <= 0:
            errs.append("n_half must be a positive multiple of 4 (got %r)" % self.n_half)
        if not self.radial or abs(self.radial[-1] - 1.0) > 1e-12:
            errs.append("radial must be non-empty and end at 1.0 (got %r)" % self.radial)
        if any(b <= a for a, b in zip(self.radial, self.radial[1:])):
            errs.append("radial must be strictly increasing (got %r)" % self.radial)
        if self.n_slices <= 0:
            errs.append("n_slices must be positive (got %r)" % self.n_slices)
        if not (0.0 < self.center_scale < 1.0):
            errs.append("center_scale must be in (0,1) (got %r)" % self.center_scale)
        if str(self.method).lower() not in ("surface", "volumetric"):
            errs.append("method must be 'surface' or 'volumetric' (got %r)" % self.method)
        if errs:
            raise ValueError("invalid Config:\n  - " + "\n  - ".join(errs))
        return self

    # -----------------------------------------------------------------------
    def flux_tag_for(self, outlet_tag: int) -> int:
        """Flux-measurement plane tag for a leg given that leg's outlet tag, or
        0 when the leg has none / flux planes are disabled."""
        if self.flux_offset is None or self.flux_offset <= 0:
            return 0
        if outlet_tag == self.tag_top2:
            return self.tag_f1                  # near the 'O  ' outlet
        if outlet_tag == self.tag_top1:
            return self.tag_f2                  # near the 'int' outlet
        return 0
