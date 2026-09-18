# Reference-data verification ledger

The exact dataset is DRP-29, DOI 10.17612/P7BC78, by Rodolfo Victor and Masa
Prodanovic, with Petrobras acknowledged for imaging. Public metadata is retrieved
from the portal's own documented download links and archived locally. The metadata
archive and API response state ODC-BY 1.0. No authenticated gate was bypassed.

Exact requested records are used: CT `38f6b4fe-4a55-4188-ac28-06cf03c0d509`,
segmentation `f4d09c8b-9067-4378-868f-f1f6c09394c3`, velocity
`4ecb6b63-db08-4e09-8d9c-f21c702e4b9f`, and pressure
`b23e5db4-a8bb-4a59-aaa1-8e91bbf21d53`. The segmentation description explicitly
states 0 pore, 255 grain despite the generic `isSegmented=no` field. Do not use
similarly named `segmented*.raw` files in the CT folder.

Metadata provides 500×500×500, 5 micrometres isotropic CT spacing, no file offset
or slice gap, 8-bit segmentation, little-endian uint16 CT, and little-endian
float32 pressure/velocities. Load as C-order `(stack,height,width)=(z,y,x)`;
physical handedness is not established. File counts/checksums and slice previews
are retained. CT/segmentation intensity correlation is a consistency check,
not proof of orientation.

The analysis entries share sample, origin label and enclosing CT path. Their
migrated `digitalDataset` UUID differs from the enclosing CT node identifier;
this discrepancy is retained instead of silently asserting exact correspondence.
Reference-field solid/pore occupancy checks provide additional evidence when
all files are available.

The linked article is [Victor et al., DOI 10.1016/j.advwatres.2016.02.002](https://doi.org/10.1016/j.advwatres.2016.02.002).
Full methods were not recovered through accessible primary sources. Neither the
public entry descriptions nor the retrieved metadata supply complete forcing,
viscosity/density, pressure/velocity units, boundary conditions, reservoir extent
or convergence controls. Three component files do not imply three independent
loads. Therefore the rock is an **application example only**, with no claimed
independent permeability error or fitted unit scale. Its periodic body-force
calculation cannot validate an unspecified finite pressure-driven reference.

The 128³ and 256³ crops are predetermined `[0:n,0:n,0:n]` in the declared storage
ordering, with no permeability-based selection. The morphology ROI is `[0:64]^3`.
They are different domains, not grid refinement or proof of an REV, and are never
compared numerically to the full-volume reference solution.

## Existing cylinder and sphere formulas

The original scripts contain truncated Sangani–Acrivos correlations. Primary
publisher records identify [square/hexagonal cylinders](https://doi.org/10.1016/0301-9322(82)90029-5)
and [periodic sphere arrays](https://doi.org/10.1016/0301-9322(82)90047-7), including
series expansions and numerical drag solutions. Full-text coefficient definitions
and the average-velocity/drag normalization were not recovered from an accessible
primary copy in this session. These formulas are retained as **legacy correlation
comparisons**, not certified independent exact solutions at arbitrary solid fraction.
No discrepancy is automatically attributed to reference truncation or to LBM.
The campaign uses fixed target solid fraction .1 at three lattice resolutions and
records actual voxel fraction; discretization and formula/averaging questions remain
separate. Channel and laminate continuum formulas have explicit averaging definitions.

Source access attempts returned publisher access restrictions/403 for full text.
An owner-supplied lawful copy of the methods would allow the unresolved conventions
to be checked. This does not block a transparently labelled periodic application.

All four reference fields were downloaded and checksummed. A deterministic
one-in-five sample is finite. About 9.46% of sampled solid-labelled voxels have
nonzero velocity; for an interior 128³ check, every active velocity voxel outside
the pore mask lies within its one-voxel face-neighbour dilation. Pressure is almost
entirely zero in solid-labelled voxels. These observations are consistent with a
near-interface support convention but do not identify it. No support masking,
axis fitting, pressure scaling or force calibration is used to manufacture agreement.
The raw checks and unscaled pressure profiles are retained.
