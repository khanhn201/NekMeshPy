"""Exact bottom-zone geometry rules (from p2/lower_zone.md), mm."""
import numpy as np

P = 50.8
R_PIN, R_POST = 19.05, 12.70
ROD_HW = 4.7625
PLATE_HW = 6.35
PLATE_DX = 0.2315                     # y-plates sit +x of nominal
Z = dict(top=411.29,                 # = geom.Z_CORE0 (core built at 411.29; CAD 411.286)
         _cad_top=411.286, AB=398.586, BC=373.186, CD=362.0735, DE=350.326, EF=347.786,
         F3=346.1985, F2=336.6735, F1=335.086, F0=333.4985, HI=322.386, IJ=318.9984, JK=313.236)
R_C_E_BARS, R_RING, R_B_BARS, R_CAN, R_CANO, R_TIP, R_VES = 685.8, 692.15, 701.675, 704.85, 711.2, 716.59, 738.5


def y_slot(xj):
    """[lo, hi] of the band-B slot (walls along y) on the x-joint xj = s*(25.4+50.8k)."""
    k = int(round((abs(xj) - 25.4) / P))
    if xj > 0:
        return 20.869 + P * k, 30.394 + P * k
    if k == 0:
        return -30.394, -20.407
    return -30.394 - P * k, -20.869 - P * k


def x_slot(yj):
    """[lo, hi] of the band C-E slot (walls along x) on the y-joint yj."""
    k = int(round((abs(yj) - 25.4) / P))
    if yj > 0:
        return (20.407, 30.394) if k == 0 else (20.869 + P * k, 30.394 + P * k)
    return -30.394 - P * k, -20.869 - P * k


def is_plate_joint(xj):
    """y-plates stand on x = +-(25.4 + 101.6 k)."""
    k = (abs(xj) - 25.4) / P
    return int(round(k)) % 2 == 0


def plate_edges(xj):
    c = xj + PLATE_DX
    return c - PLATE_HW, c + PLATE_HW


def is_rod_joint(xj):
    """rods join posts 2m+1 and 2m+2 (both sides), i.e. they cross the odd x-joints."""
    return not is_plate_joint(xj)


def sites():
    """the 560 pin/post sites S."""
    out = []
    key = {(3, 13), (13, 3), (6, 12), (12, 6)}
    for i in range(-14, 15):
        for j in range(-14, 15):
            if max(abs(i), abs(j)) <= 1:
                continue
            r = P * np.hypot(i, j)
            if r <= 673.1 + 1e-6 or (abs(i), abs(j)) in key:
                out.append((i, j))
    return out
