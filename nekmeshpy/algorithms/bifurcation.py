"""Bifurcation hex-mesh generator -- orchestrator (SURFACE pipeline).

``BifurcationMesher`` drives the whole surface pipeline through the mesh objects:
load the :class:`~nekmeshpy.geometry.trimesh.TriMesh`, solve the three intrinsic-Laplacian
seam fields, cut it into legs (:class:`~nekmeshpy.algorithms.cutsurface.CutSurface`), build
the conformal seam rings + spine, extrude each leg's O-grid slices
(:class:`~nekmeshpy.algorithms.ogrid.OGridLeg`) into a :class:`~nekmeshpy.geometry.hexmesh.HexMesh`,
reposition the interior, smooth, and export / plot.

    from nekmeshpy import Config, BifurcationMesher
    hexmesh = BifurcationMesher(Config()).run()
"""

import logging

import numpy as np

from ..geometry.hexmesh import HexMesh
from ..geometry.trimesh import TriMesh
from ..io import export, viz
from ..model.physical import PhysicalGroups
from ..ops import interior, smoothing, trisurf
from .cutsurface import CutSurface
from .ogrid import OGridLeg
from .registry import register_algorithm

_log = logging.getLogger("nekmeshpy")


@register_algorithm("bifurcation")
class BifurcationMesher:
    def __init__(self, config):
        self.cfg = config
        self.surface = None

    # -- seam / opening solvers (bifurcation-specific) ------------------
    @staticmethod
    def _order_openings(surf):
        """Order the three boundary loops A/B/C: A = trunk (lowest mean Z),
        B/C = branches by mean X.  Returns ``gloops`` (list of 3 vertex arrays)."""
        loops = trisurf.boundary_loops(surf)
        assert len(loops) == 3, (
            "expected exactly 3 boundary loops, got %d" % len(loops))
        Z = surf.xyz[:, 2]
        X = surf.xyz[:, 0]
        meanZ = np.array([Z[c].mean() for c in loops])
        iA = int(np.argmin(meanZ))
        rest = [i for i in range(3) if i != iA]
        meanX = np.array([X[loops[i]].mean() for i in rest])
        order = np.argsort(meanX, kind="stable")
        return [loops[iA], loops[rest[order[0]]], loops[rest[order[1]]]]

    @staticmethod
    def _harmonic_field(surf, gloops, neumann, dvals):
        """One harmonic bifurcation problem: Laplace with natural Neumann on
        loop ``neumann`` (0-based), Dirichlet ``dvals`` on the other two, then
        shifted to zero mean on the free loop."""
        dnodes = []
        dv = []
        for k in range(3):
            if k == neumann:
                continue
            g = np.asarray(gloops[k]).ravel()
            dnodes.append(g)
            dv.append(dvals[k] * np.ones(g.size))
        u = trisurf.solve_dirichlet(surf, np.concatenate(dnodes), np.concatenate(dv))
        return u - np.mean(u[gloops[neumann]])

    def _seam_fields(self, surf, gloops):
        """The three harmonic seam fields U (nv,3)."""
        nan = np.nan
        dvals = [[nan, 0, 1], [1, nan, 0], [0, 1, nan]]
        U = np.zeros((surf.n_vertices, 3))
        for k in range(3):
            U[:, k] = self._harmonic_field(surf, gloops, k, dvals[k])
        return U

    # -- pipeline --------------------------------------------------------
    def run(self):
        """Run the full surface pipeline; returns the assembled HexMesh."""
        cfg = self.cfg
        cfg.validate()
        if str(getattr(cfg, "method", "surface")).lower() == "volumetric":
            raise NotImplementedError(
                "Only the SURFACE pipeline is ported; set cfg.method = 'surface'.")

        surf = TriMesh.from_files(cfg.vtx_file, cfg.tri_file)
        self.surface = surf

        gloops = self._order_openings(surf)
        U = self._seam_fields(surf, gloops)

        cut = CutSurface.from_fields(surf, U)
        cut.smooth_seams(cfg.seam_smooth_iters, cfg.seam_smooth_lambda)
        rings, A1, A2, spine = cut.seam_rings(gloops, cfg.n_half)

        params = {"radial": cfg.radial, "center_scale": cfg.center_scale,
                  "resample_spline": cfg.resample_spline,
                  "project_to_stl": cfg.project_to_stl}
        outlet_tag = [cfg.tag_trunk, cfg.tag_top1, cfg.tag_top2]
        levels = np.linspace(0, 1, cfg.n_slices + 2)[1:-1]

        hexmesh = HexMesh(groups=PhysicalGroups.from_config(cfg))
        for leg in range(3):
            sub, us, _, _, _ = cut.leg_field(leg, gloops)
            fr, frlev = trisurf.extract_rings(sub, us, levels, cfg.min_loop_pts)
            half1, half2 = OGridLeg(fr, rings[leg], spine, surf, params).build(frlev)
            flux_tag = cfg.flux_tag_for(outlet_tag[leg])
            for half in (half1, half2):
                # extrude the half (opening cap = the leg outlet; the seam end
                # is interior, so no far cap), then tag the flux-measurement
                # plane on face 5 of the interior ring cfg.flux_offset in.
                eids = hexmesh.add_extruded_section(
                    half, first_cap_tag=outlet_tag[leg], wall_tag=cfg.tag_wall)
                if flux_tag > 0 and 0 <= cfg.flux_offset < eids.shape[0]:
                    for e in eids[cfg.flux_offset, :]:
                        hexmesh.tag_face(int(e), 5, flux_tag)
        hexmesh.finalize()

        interior.set_interior(hexmesh, cfg.interior_method, cfg.tag_wall)

        if cfg.smooth_iters and cfg.smooth_iters > 0:
            smoothing.smooth(hexmesh, surf, {"smooth_iters": cfg.smooth_iters,
                                             "smooth_lambda": cfg.smooth_lambda,
                                             "tag_wall": cfg.tag_wall,
                                             "project_to_stl": cfg.project_to_stl})

        export.summary(hexmesh, cfg)
        if cfg.export_vtk:
            export.to_vtk(hexmesh, cfg.out_name + ".vtk")
            _log.info("wrote %s.vtk", cfg.out_name)
        if cfg.export_re2:
            export.to_re2(hexmesh, cfg.out_name)
            _log.info("wrote %s.re2, %s.rea", cfg.out_name, cfg.out_name)
        if cfg.plot:
            viz.plot(hexmesh, cfg)
        return hexmesh
