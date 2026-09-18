# Original pressure-wall change

`original_pressure_walls.patch` is the byte-identical patch saved before the
validation work began. It belongs to the repository owner and was not introduced
by the reliability/tensor implementation.

Commit `7489f432e17b6648b23980981724505dbe3b4553` records that original patch separately: the `walls_y` option adds
outer solid rows, while superficial averaging and exported velocity fields remain
restricted to the original sample. The following implementation commit retains
these semantics while adding validation and diagnostics. The original patch is
relative to baseline c1a195b62a9e519dbfeb23f2c764da1f2d14fad8; do not apply it
again to the updated solver.

The local original-patch snapshot is `c02f39cede5598eb8c8d3c0a4c350c32a650ce22`.
The connected GitHub app uploaded an identical tree as the commit above because
terminal Git had no credentials. Commit metadata differs; file content does not.
