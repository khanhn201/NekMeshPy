"""MSRE core geometry constants (mm), read off core_fluid_cleanup_v5.STEP."""
import numpy as np

PITCH = 50.8            # stringer lattice pitch; stringer centres at PITCH*(i,j)
H = PITCH / 2           # half cell; joints at H + PITCH*k
GR = 5.08               # groove end radius == groove depth (half channel width)
GL = 10.16              # half length of the straight part of a channel
RG = 701.675            # graphite outer radius
RC = 704.85             # core-can inner radius
RCO = 711.2             # core-can outer radius (downcomer inner)
RV = 738.505            # vessel inner radius (downcomer outer)
R_OPEN = 33.02          # thimble-site opening radius
R_THIM = 27.051         # control-rod thimble OD/2
R_BASKET = 30.137       # sample basket OD/2
THIMBLES = {(1, 1): R_BASKET, (-1, 1): R_THIM, (-1, -1): R_THIM, (1, -1): R_THIM}

Z_CORE0 = 411.29        # bottom of the straight channels
Z_CORE1 = 1986.09       # top of the straight channels (start of 45 deg chamfer)

# template knobs (not geometry)
POST = 11.43            # half side of the inner graphite square (stringer top post)
KC = 12.5               # half length of the half-groove's inner quad
KB = 2.54                # depth of the half-groove's inner quad from the joint

TIP = 2.0               # half side of the apex quad of a pyramid-topped stringer
CENTRAL = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}   # stringers with a round post
R_POST = 9.525          # central post radius (collars r11.049 / base r14.605 dropped)

# upper-zone levels
Z_PYR = 1986.09         # pyramid base (normal stringers), apex at +H
Z_PYR_TALL = 2036.89    # pyramid base of the 44 tall stringers on the two axes
Z_SHOULDER = 1988.63    # flat top of the 5 central stringers
Z_POST = 2014.48        # central post top (cone/dome tip dropped)
RF = 0.65                # ridge node at this fraction of the apex -> corner distance
XU = 12.0               # T side: arc nodes at u = +-XU
XF = 0.8                # flat side: interior node at v = XF*H
Z_EDGE = 1916.24          # half-channels and bays are capped here
IU = 10.16                # half-groove: chain / joint nodes at u = +-IU
YU, YD = 9.0, 3.64      # half-groove D region: middle row at u = +-YU, depth YD
RF2 = 0.4               # inner ridge node (fraction apex -> corner)
XU1, XU2 = 15.0, 7.5    # T side: arc nodes
YF = 0.45               # flat side: second interior node at v = YF*H
