# Executed evidence ledger — 18 September 2026

Baseline HEAD: `c1a195b62a9e519dbfeb23f2c764da1f2d14fad8`.
The existing uncommitted pressure `walls_y` addition was saved as a patch and its
behavior retained. No reset, push, release, license change, driver change or
interference with `porous_flow-opt` jobs was performed.

This ledger supersedes the former blanket correctness and performance claims.
The final case inventory and remaining work are in `CODE_AND_BENCHMARK_HANDOFF.md`.
All paths below are relative to `results/2026-09-18-pilot/`.

| Item | Observed evidence | Action/status |
|---|---|---|
| Existing CPU channel tests | `baseline/pytest.log`: 2 passed, 8.48 s | Retained; later source tests also executed |
| Actual remote hardware and CUDA | `baseline/environment.json` | Host b-11uxx6xvs954, real allocation and compiled CUDA kernel passed |
| Invalid tau/zero steps/fully fluid | `baseline/probes.json` contains misleading finite outputs from original source | Shared validation, special geometry states and guarded extraction |
| Shared statuses and precision | `reliability_second.log`: 91 passed, one inapplicable dimensional skip | NumPy, CuPy array, CUDA and pressure tested |
| First regression failures | `reliability_first.log`: 16 failed, 70 passed, one skipped | Retained; NVRTC stdint header unavailable, changed to checked 64-bit type; straight high-force channel hit quality rejection, so instability fixture changed to porous mask |
| Legacy fine channels after shared core | `after_core.log`: 2 passed | Original collision and reflection numerics preserved |
| CPU wheel outside repository | `installed_cpu_tests.log`: initial wheel 40 passed, 7 skipped | CPU-only environment, no source-tree imports |
| Channels, tau, tensors, backend states, pressure | `verification/`, definitions plus raw results, CSV histories | Executed; failures and sensitivity retained |
| Weak-force float32 storage | `verification/precision32_*/` | All three cases unconverged; no accepted float32 claim for these settings |
| Oblique 3D laminate | `verification/laminate3_*/` | Signed cross terms and reciprocity pass; continuum tensor errors about 3.32%, 2.88%, 2.41% |
| Repeated throughput | `performance/` | Five measured repetitions for completed configurations; setup/loop/public-call timing distinctions retained |
| Bentheimer exact segmentation | `dataset/dataset_manifest.json`, `inspection.json` | Byte count, labels, checksums, fixed crops and morphology ROI verified |
| Reference fields | `dataset/reference_field_inspection.json`, `reference_support_and_profiles.json` | Units/BVP missing; velocity support includes one-voxel solid shell; no matched benchmark claim |
| Bentheimer 128/256 pilots | `rocks/` | Original periodic boundary problem on specified crops; timeout is not permeability |
| Full 500³ | `rocks/full_500_decision.json` | Memory/prequalification gate; no automatic allocation |

The common result monitor evaluates all superficial velocity components, full-field
RMS change, global periodic population mass, finite states, positive fluid density
and Mach. Pressure additionally checks signed gradient and cross-section mass flux.
These checks do not certify continuum, segmentation or representative-volume error.

The initial SciPy morphology implementation is an exhaustive covering-ball local
thickness for bounded finite ROIs; it is not nearest-wall-distance binning.
PoreSpy dependency resolution failed on Python 3.14 (`porespy_dependencies.log`).
Periodic morphology is explicitly unsupported. Connectivity diagnostics distinguish
finite spanning from periodic winding and never modify the input mask.

Primary full-text reference formulas and the DRP-29 simulation methods remain
unrecovered. See `docs/reference_scope.md`; legacy sphere/cylinder discrepancies are
reported without declaring an exact validation pass or blaming the reference.

Final follow-up: source tests passed 110 with one dimensional skip; the final built
wheel passed 46 with nine GPU-dependent skips in an external CPU-only environment,
and 110 with one skip using the actual GPU outside the checkout. Installed package
hashes match the source. See `release_tests.xml`, `installed_cpu_release_tests.xml`
and `installed_gpu_release_tests.xml`; earlier failures above remain archived.

The 256³ periodic application now has two accepted full tensors: standard force
and half force, totaling 31.20 minutes across all directional solves. Halving force
changes the tensor by 2.787e-6 relative; tightening convergence tolerances 100-fold
changes the x response by 8.644e-6 relative. See `long_rock/` and
`convergence_review/`. An accepted 128³ velocity export exactly reproduces its
earlier history. This does not resolve the external reference's missing boundary
conditions or justify a 500³ allocation. Only the measured 2.46% diagnostic-overhead
optimization was adopted; two additional kernel probes were retained without
changing production kernels.
