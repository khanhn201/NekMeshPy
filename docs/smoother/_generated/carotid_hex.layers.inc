`RADIAL` (constants at the top of `examples/smoother/cs_carotid_hex.py`) is the original's: `[0.0, 0.4, 0.7, 0.9, 1.0]`, 4 layers between the core square and the wall, the wall layer 0.10 of the gap. The rebuild keeps exactly this layering.

| mesh | layers | first layer, fraction of the core-to-wall gap | first layer / local radius: min | median | max |
| --- | --- | --- | --- | --- |  --- |
| original wall | 4 | 0.100 | 0.0319 | 0.0548 | 0.0726 |
| smoothed wall, rebuilt | 4 | 0.100 | 0.0304 | 0.0550 | 0.0726 |

The first layer is measured as each wall hex's edge into the volume, against the mean radius of its station's wall ring (1440 wall hexes in both). Its thickness over the local radius is essentially unchanged, so the smoothed wall leaves the original boundary layer intact: the layering is a property of the construction, not of the surface.
