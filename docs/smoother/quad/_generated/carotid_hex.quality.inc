| mesh | hexes | min scaled J | mean | worst 5 % | inverted | order-7 sampling: min / inverted |
| --- | --- | --- | --- | --- | --- | --- |
| original (carotid recipe) | 7200 | 0.0960 | 0.9167 | 0.7951 | 0 | 0.0960 / 0 |
| smoothed wall, seam ring free | 7200 | -0.1253 | 0.9169 | 0.7903 | 7 | -0.1253 / 7 |
| **rebuilt on the smoothed wall** (seam ring held) | 7200 | 0.0216 | 0.9184 | 0.7963 | 0 | 0.0216 / 0 |

Scaled Jacobian of the order-1 hexes, read at their 8 corners (the straight-sided hex a `.re2` file holds). The last column re-reads the same trilinear map on an 8^3 lattice, the solver's order: it agrees with the corner reading to 3 decimals in every row.
