| stage | seconds | share of the total |
| --- | --- | --- |
| 1  read the tessellation (TriMesh) | 0.00 | 0.0 % |
| 2  seam fields, cut into legs | 0.02 | 0.4 % |
| 3  seam rings, leg rings, spine | 0.19 | 3.5 % |
| 4  O-grid sections, extrusion, welds, tags (original hexes) | 0.53 | 9.6 % |
| 5  extract the wall quads | 0.00 | 0.1 % |
| 6  smoother set-up (gate, overlaps, surface, normals) | 0.03 | 0.6 % |
| 7a smoothing: laplace | 0.03 | 0.5 % |
| 7b smoothing: optimize | 2.50 | 45.0 % |
| 7c smoothing: local lifts | 0.00 | 0.0 % |
| 8  rebuild the hexes on the smoothed wall | 0.64 | 11.4 % |
| 9  quality readings (original and rebuilt) | 1.58 | 28.5 % |
| 10 export (original and rebuilt, .vtu) | 0.02 | 0.3 % |
| **total of the stages above** | **5.56** | **100 %** |

Smoothing (7) vs building the original hexes (4) and rebuilding them (8): **2.54 s** vs **0.53 s** and **0.64 s**.
