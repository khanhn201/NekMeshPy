"""``nekmesh`` command-line interface.

Subcommands::

    nekmesh mesh    [--config c.yaml] [--interior M] [--out NAME] [--no-plot]
                    [--format re2,vtk,vtu ...]
    nekmesh quality FILE            # scaled-Jacobian report for a hex mesh
    nekmesh info    FILE            # points / cells / groups summary
    nekmesh convert IN OUT          # meshio-based format conversion

Run ``nekmesh <cmd> -h`` for the options of each subcommand.
"""

import argparse
import logging
import sys

from .config import Config
from .io import export
from .model import quality


# -- helpers ------------------------------------------------------------
def _load_hex(path):
    """Read a mesh file (any meshio format) and return (points, hexes)."""
    from .model.mesh import Mesh
    m = Mesh.read(path)
    if "hexahedron" not in m.cells:
        raise SystemExit("%s: no hexahedron cells found (types: %s)"
                         % (path, ", ".join(m.cell_types) or "none"))
    return m.points, m.cells["hexahedron"], m


# -- subcommands --------------------------------------------------------
def cmd_mesh(args):
    from .algorithms.bifurcation import BifurcationMesher

    cfg = Config.from_file(args.config) if args.config else Config()
    if args.interior:
        cfg.interior_method = args.interior
    if args.out:
        cfg.out_name = args.out
    if args.no_plot:
        cfg.plot = False
    formats = [f.strip().lower() for f in args.format.split(",")] if args.format else None
    if formats is not None:
        cfg.export_re2 = "re2" in formats
        cfg.export_vtk = "vtk" in formats
    cfg.validate()

    mesh = BifurcationMesher(cfg).run()

    # any extra meshio formats requested beyond the native re2/vtk
    for fmt in (formats or []):
        if fmt in ("re2", "vtk"):
            continue
        path = "%s.%s" % (cfg.out_name, fmt)
        export.write(mesh, path)
        print("wrote %s" % path)
    return 0


def cmd_pipe(args):
    from .algorithms.pipes import CircularPipe, RectangularPipe

    if args.shape == "circular":
        algo = CircularPipe(radius=args.radius, length=args.length,
                            n_axial=args.n_axial, n_side=args.n_side,
                            n_radial=args.n_radial, center_scale=args.center_scale,
                            radial_grading=args.radial_grading,
                            axial_grading=args.axial_grading)
    else:
        algo = RectangularPipe(width=args.width, height=args.height,
                               length=args.length, nx=args.nx, ny=args.ny,
                               n_axial=args.n_axial, axial_grading=args.axial_grading)
    mesh = algo.run()

    formats = ([f.strip().lower() for f in args.format.split(",")]
               if args.format else ["re2", "vtk"])
    for fmt in formats:
        if fmt == "re2":
            export.to_re2(mesh, args.out)
            print("wrote %s.re2, %s.rea" % (args.out, args.out))
        elif fmt == "vtk":
            export.to_vtk(mesh, args.out + ".vtk")
            print("wrote %s.vtk" % args.out)
        else:
            path = "%s.%s" % (args.out, fmt)
            export.write(mesh, path)
            print("wrote %s" % path)
    stats = quality.summary(*mesh.weld()[:2])
    print("%d hex elements, min scaled Jac=%.4f mean=%.4f"
          % (mesh.n_elements, stats["min"], stats["mean"]))
    return 0


def cmd_quality(args):
    points, hexes, _ = _load_hex(args.file)
    stats = quality.summary(points, hexes)
    hist = quality.histogram(points, hexes, bins=args.bins) if args.histogram else None
    print(quality.format_report(stats, hist))
    return 0


def cmd_info(args):
    from .model.mesh import Mesh
    m = Mesh.read(args.file)
    print("file        : %s" % args.file)
    print("points      : %d" % m.n_points)
    for t in m.cell_types:
        print("cells[%-10s]: %d" % (t, m.n_cells(t)))
    if m.cell_sets:
        print("groups      : %s" % ", ".join(m.cell_sets.keys()))
    return 0


def cmd_convert(args):
    from .model.mesh import Mesh
    m = Mesh.read(args.infile)
    m.write(args.outfile)
    print("converted %s -> %s" % (args.infile, args.outfile))
    return 0


# -- parser -------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="nekmesh",
                                description="NekMeshPy hex-mesh toolkit")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mesh", help="generate the bifurcation hex mesh")
    m.add_argument("--config", help="YAML/JSON config file")
    m.add_argument("--interior", help="interior method (bilinear|harmonic|harmonic3d|winslow)")
    m.add_argument("--out", help="output base name")
    m.add_argument("--no-plot", action="store_true", help="disable plotting")
    m.add_argument("--format", help="comma list, e.g. re2,vtk,vtu,msh")
    m.set_defaults(func=cmd_mesh)

    pp = sub.add_parser("pipe", help="generate a straight circular/rectangular pipe")
    pp.add_argument("--shape", choices=("circular", "rectangular"), default="circular")
    pp.add_argument("--out", default="pipe", help="output base name")
    pp.add_argument("--format", help="comma list, e.g. re2,vtk,vtu (default re2,vtk)")
    pp.add_argument("--length", type=float, default=1.0)
    pp.add_argument("--n-axial", type=int, default=10, dest="n_axial")
    pp.add_argument("--axial-grading", type=float, default=1.0, dest="axial_grading")
    # circular
    pp.add_argument("--radius", type=float, default=0.5)
    pp.add_argument("--n-side", type=int, default=4, dest="n_side")
    pp.add_argument("--n-radial", type=int, default=3, dest="n_radial")
    pp.add_argument("--center-scale", type=float, default=0.5, dest="center_scale")
    pp.add_argument("--radial-grading", type=float, default=1.0, dest="radial_grading")
    # rectangular
    pp.add_argument("--width", type=float, default=1.0)
    pp.add_argument("--height", type=float, default=1.0)
    pp.add_argument("--nx", type=int, default=8)
    pp.add_argument("--ny", type=int, default=8)
    pp.set_defaults(func=cmd_pipe)

    q = sub.add_parser("quality", help="report scaled-Jacobian quality")
    q.add_argument("file", help="hex mesh file (any meshio format)")
    q.add_argument("--histogram", action="store_true", help="show distribution")
    q.add_argument("--bins", type=int, default=10)
    q.set_defaults(func=cmd_quality)

    i = sub.add_parser("info", help="summarize a mesh file")
    i.add_argument("file")
    i.set_defaults(func=cmd_info)

    c = sub.add_parser("convert", help="convert between mesh formats (meshio)")
    c.add_argument("infile")
    c.add_argument("outfile")
    c.set_defaults(func=cmd_convert)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
