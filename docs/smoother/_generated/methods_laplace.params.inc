| parameter | default | meaning |
| --- | --- | --- |
| `passes` | `3` | number of passes |
| `w` | `0.97` | shrink factor per pass, in (0, 1]; 1 is a no-op, smaller moves vertices more |
| `normals` | `None` | reference normals for the readings (default: the input's winding) |
| `elements` | `None` | boolean mask of quads: only their vertices move (the local variant) |
| `max_sj_drop` | `0.05` | a pass that adds an inverted quad or lowers the minimum scaled J by more than this is refused and the stage stops on the last accepted pass |
| `kappa` | `1000.0` | inversion-penalty weight, used only for the F/quad reading in the log |
| `eps` | `0.05` | inversion-penalty threshold on the scaled J, likewise only for the reading |
| `free_boundary` | `'tangent'` | rule for a free boundary vertex (no target, not frozen): `'tangent'` keeps only the move along the local boundary tangent, so the boundary does not shrink; `'chain'` applies the move as is (a convex boundary is pulled in onto its chords) |
