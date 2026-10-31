| run | quads | inverted | overlapping | min scaled J | mean scaled J | worst 5 % scaled J |
| --- | --- | --- | --- | --- | --- | --- |
| as built | 2040 | 0 → 0 | 0 → 0 | 0.5148 → 0.5755 | 0.9581 → 0.9703 | 0.7914 → 0.8730 |
| perturbed input | 2040 | 0 → 0 | 0 → 0 | 0.2624 → 0.5763 | 0.9164 → 0.9703 | 0.6684 → 0.8731 |

- as built: min scaled J **rose** (0.5148 → 0.5755), mean scaled J **rose** (0.9581 → 0.9703), worst 5 % **rose** (0.7914 → 0.8730); no quad is inverted or overlapping at the end.
- perturbed input: min scaled J **rose** (0.2624 → 0.5763), mean scaled J **rose** (0.9164 → 0.9703), worst 5 % **rose** (0.6684 → 0.8731); no quad is inverted or overlapping at the end.

- as built: vertices of the U-tube wall moved **0.41 local edges** on average (largest 2.63); the size ratio between neighbours went 1.90 → 1.36.
- perturbed input: vertices of the U-tube wall moved **0.48 local edges** on average (largest 2.88); the size ratio between neighbours went 2.41 → 1.37.

**Outer box** (tag `outer`; planes, creases and corners held):

| run | quads | inverted | overlapping | min scaled J | mean scaled J | worst 5 % scaled J |
| --- | --- | --- | --- | --- | --- | --- |
| as built | 5000 | 0 → 0 | 0 → 0 | 0.7071 → 0.7993 | 0.9980 → 0.9983 | 1.0000 → 0.9964 |
| perturbed input | 5000 | 0 → 0 | 0 → 0 | 0.5304 → 0.7991 | 0.9386 → 0.9983 | 0.8292 → 0.9964 |

- as built: min scaled J **rose** (0.7071 → 0.7993), mean scaled J **rose** (0.9980 → 0.9983), worst 5 % **fell** (1.0000 → 0.9964); no quad is inverted or overlapping at the end.
- perturbed input: min scaled J **rose** (0.5304 → 0.7991), mean scaled J **rose** (0.9386 → 0.9983), worst 5 % **rose** (0.8292 → 0.9964); no quad is inverted or overlapping at the end.

- as built: vertices of the outer box moved **0.04 local edges** on average (largest 1.43); the size ratio between neighbours went 1.17 → 1.37.
- perturbed input: vertices of the outer box moved **0.21 local edges** on average (largest 1.93); the size ratio between neighbours went 1.60 → 1.37.
