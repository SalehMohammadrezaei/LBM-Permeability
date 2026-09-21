# Code and benchmark handoff

Technical evidence only; no manuscript or submission material was edited.
Repository: `/home/impres/Saleh/lbm-permeability` on `b-11uxx6xvs954`.
Historical baseline: `c1a195b62a9e519dbfeb23f2c764da1f2d14fad8` (pre-campaign main).
[PR #1](https://github.com/SalehMohammadrezaei/LBM-Permeability/pull/1) was merged
using a merge commit, preserving all three development commits:
`7b5908a0cf34c3e78e5790c8142caa6894bdf1b9` on main. Its reviewed second parent is
`84649088294e6b29dd186b98de8e60b141bdd81a`; the merge tree is identical to that
reviewed head. This handoff is updated by a subsequent documentation-only commit.
The owner's initial pressure-wall patch is recorded
separately in commit `7489f432e17b6648b23980981724505dbe3b4553` and `docs/provenance/original_pressure_walls.patch`.
The original local copy remains in `results/2026-09-18-pilot/baseline/user_changes.patch`.
No release, license/driver change or intervention in `porous_flow-opt` processes
was performed. Selected small evidence records are tracked under
`benchmarks/evidence/2026-09-18/`; large local outputs remain excluded from Git.

Evidence root below means `results/2026-09-18-pilot/`. Commands are in
`docs/benchmark_reproduction.md`; equations and limitations are in
`docs/numerical_limits.md`. Raw JSON is authoritative over rounded values here.

### Final software versus historical snapshots

The final software includes the packaging-portability fixes in reviewed commit
`84649088294e6b29dd186b98de8e60b141bdd81a`, now reachable through the main merge
above. The package version remains 0.1.0; no release or version bump was made.
Its 18 package-file hashes are recorded in
`benchmarks/evidence/packaging-followup/summary.json` and `installed_source.json`.

| Evidence scope | Source identity and interpretation |
|---|---|
| Initial baseline and owner's wall patch | Baseline commit plus the saved original patch; owner patch subsequently recorded as `7489f432e17b6648b23980981724505dbe3b4553` |
| Verification, initial rock pilots and accepted tensor campaign | Historical working-tree snapshots, identified by each result's package hashes and the saved source archives; their recorded baseline commit alone does not identify the modified source |
| Accepted 256³ tensor campaign | `accepted_campaign_source.tar.gz` and its manifest, plus per-result hashes; predates the CLI missing-file fix and optional Git/RSS packaging follow-up |
| Archived pre-portability wheel and compact bundle | Package files correspond to `63182cef1f9509ab9e77cc6d4bcf830048221929`; 110-pass GPU and 46-pass CPU wheel records belong to this snapshot |
| Final merged software | `84649088294e6b29dd186b98de8e60b141bdd81a`, incorporated unchanged by merge `7b5908a0cf34c3e78e5790c8142caa6894bdf1b9`; separate portability test records below |

Historical results, source manifests, source archives, wheels and the compact
bundle retain their original bytes and hashes. They were not regenerated or
relabeled as results from the merged software. The bundle's embedded handoff is
also historical; this repository file is the updated handoff. Packaging changes
affect host metadata/error handling only; no equations, convergence settings or
archived numerical values changed, and no large simulations were rerun for merge.
The merge-time preservation check recomputed SHA256 for all 535 existing files
under `results/` and `benchmarks/evidence/`; every hash matched its pre-merge value.

## 1. Changes and reasons

* Shared strict mask/parameter validation, actual GPU runtime detection, explicit
  backend/precision selection, consistent lifecycle and termination reasons.
* Invalid density and nonfinite states are rejected; no clipping or NaN repair.
  Default permeability extraction rejects unconverged states. Zero forcing,
  all-solid and fully fluid periodic forced geometries have explicit semantics.
* Full-vector and field convergence, periodic mass conservation, Mach/Reynolds
  diagnostics and histories. Invalid CLI input or unsuccessful simulation exits 2.
* Sequential independent-load 2×2/3×3 tensors with raw signed columns, per-load
  validity, reciprocity and clearly labelled symmetric-part eigensystems.
* Strengthened existing porosity, explicit binary decoding, optional finite/periodic
  connectivity diagnostics without pore removal, optional volume-weighted
  covering-ball local thickness with tested finite-image conventions.
* Preserved original collision/reflection and 2D Zou-He algebra; investigated
  pressure flux, reservoir padding and the owner's `walls_y` feature.
* Checked 64-bit CUDA indexing ABI, no fast math, honest mixed precision metadata,
  strict JSON, reproducible result/configuration export, tests and bounded runners.
* Removed redundant diagnostic work, preserving all check intervals and criteria.
  On 256³, three warmed timing pairs gave median 9.006 → 8.785 seconds per 500
  updates (2.46% reduction); complete histories were exactly equal. This is a
  modest workload-specific gain, not a general speedup guarantee.

## 2. Executed tests and environment

Measured workstation: Threadripper PRO 7995WX, 192 logical CPUs, approximately
393 GiB RAM; RTX 6000 Ada, 46068 MiB reported VRAM. Initially about 330 GiB RAM
and 46.3 GB device allocation memory were free. A compiled CUDA allocation/kernel
test executed successfully. The environment is WSL2, Python 3.14.7, isolated
`.venv`, NumPy 2.5.3, CuPy 14.2.0, SciPy 1.18.1, Matplotlib 3.11.2 and pytest
9.1.1. Driver 595.71; runtime API 12090, driver API 13020. Exact measured
availability/version records are in `baseline/environment.json`.

| Check | Actual result | Evidence |
|---|---|---|
| Original tests before modifications | 2 passed, 8.48 s | `baseline/tests.xml` |
| First reliability suite | 16 failed, 70 passed, 1 skipped | `reliability_first.xml` |
| Reliability fixes retested | 91 passed, 1 skipped | `reliability_second.xml` |
| Full suite including tiny-force guard | 107 passed, 1 skipped, 58.61 s | `final_guard_tests.xml` |
| Post-overhead-change suite | 107 passed, 1 skipped, 58.15 s | `review_tests.xml` |
| Pre-portability source, including pressure failure and missing-file regressions | 110 passed, 1 skipped, 40.55 s | `release_tests.xml` |
| Pre-portability built wheel, outside checkout, CPU-only env | 46 passed, 9 skipped, 25.70 s | `installed_cpu_release_tests.xml` |
| Pre-portability built wheel, outside checkout, actual GPU | 110 passed, 1 skipped, 39.34 s | `installed_gpu_release_tests.xml` |

The one real-GPU suite skip is an inapplicable z direction in 2D. CPU-only wheel
skips do not qualify GPU behavior. Tests import the installed package from
`/tmp/lbm-installed-cpu/lib/python3.14/site-packages`, not the checkout.
GPU wheel tests import `/tmp/lbm-installed-gpu-release/lbm_permeability` with
dependencies from the isolated workstation environment. Both installed copies'
package source hashes match the archived pre-portability snapshot, not the newer
merged package; see `installed_*_release_source.json`.
Between the accepted tensor snapshot and that archived wheel, the only package
change was catching file-system
`OSError` in the CLI so a missing input produces saved error metadata and exit 2.
Numerical code is unchanged. `accepted_campaign_source.tar.gz` preserves the
exact earlier package source for strict campaign-resume checks.
The first regression failures are retained: NVRTC could not include `stdint.h`
(resolved with a checked 64-bit typedef); a high-force straight-channel fixture
reached the Mach rejection rather than invalid density, so a porous instability
fixture was used to exercise density failure. Tolerances were not relaxed.

The final merged software has separate evidence in the repository directory
`benchmarks/evidence/packaging-followup/` (not the historical evidence root):

| Check | Actual result | Evidence in that directory |
|---|---|---|
| Focused missing-Git, memory and CLI regressions | 17 passed, 1.76 s | `focused_fixed.xml` |
| Full source suite with actual GPU | 124 passed, 1 skipped, 43.25 s | `source_fixed.xml` |
| Fresh installed wheel outside checkout, CPU-only env | 59 passed, 10 skipped, 25.36 s | `installed_cpu.xml`, `installed_source.json` |

The final package's GPU coverage is from the source suite; a new GPU installed-wheel
suite was not run after the portability changes. The installed CPU check verifies
all 18 package files against the source and tests export/error handling with Git
missing and `resource` unavailable. Linux/macOS RSS units were tested with simulated
reports; native Windows/macOS execution was not performed. The initial portability
collection failure and corrected reruns remain recorded in that directory.

Before merging, the PR head was verified as exactly
`84649088294e6b29dd186b98de8e60b141bdd81a`. Both hosted CPU `tests` check runs
completed successfully: [PR run 35355909677](https://github.com/SalehMohammadrezaei/LBM-Permeability/actions/runs/35355909677)
and [push run 35355904441](https://github.com/SalehMohammadrezaei/LBM-Permeability/actions/runs/35355904441).
GitHub reported main unprotected, required-check contexts empty and no rulesets.
The merge used method `merge` with the expected head SHA supplied to GitHub.

## 3. Completed verification and application evidence

Acceptance criteria were saved in `benchmarks/configs/acceptance.json` before the
main verification. `verification/case_definitions.json` records exact masks,
loads and settings. All runs, including failures, remain available.

| Cases | Observed result | Location under evidence root |
|---|---|---|
| Axis-aligned channels, apertures 12/24/48 cells | Relative continuum errors 0.6944%, 0.1736%, 0.04327% | `verification/channel_*` |
| Tau 0.7/0.85/1/1.2/1.6, coarse/fine channels | Finite-grid tau dependence retained; coarse high-tau error 6.03% | `verification/tau_*` |
| Non-cubic 3D and axis permutations | NumPy/CuPy-array/CUDA accepted dominant k agreement ~4.9e-13 relative or better; fixed fields <=1.7e-16 absolute | `verification/parity_*`, `extruded_axis_*` |
| Force halving/reversal and tighter tolerance, porous tensor | Relative tensor changes 2.90e-6 / 1.16e-5 / 1.03e-9 | `verification/sensitivity_*` |
| Oblique laminates, nonzero signed cross terms | 2D near analytical response; 3D errors 3.32%, 2.88%, 2.41%; reciprocity ~1e-12 | `verification/laminate*` |
| Float32 population storage, weak forcing | Three-axis cases at 1e-6 and x at 1e-5 did not converge: excluded | `verification/precision32_*`, `precision_followup/` |
| Float32 at x-force 1e-4, small porous geometry | Accepted vs float64, relative k difference 1.63e-6; Mach .0142, Re(length 6) .0946 | `precision_followup/` |
| 2D pressure channels, padding 0/4/8/16 | k 6.04176 / 5.94224 / 5.93312 / 5.93369; continuum channel value 6 | `verification/pressure_*` |
| Pressure porous padding | k 1.63822 / 1.62198 / 1.61712 / 1.61580; flux span <1e-6 | `verification/pressure_porous_*` |
| Sphere/cylinder arrays | Executed; legacy correlation discrepancies retained, not declared independent validation | `verification/sphere_*`, `cylinder_*`, `legacy_comparisons.log` |
| Connectivity/blocked flow | Closed cavity, blocked direction, conducting channel, face-touch without winding; diagonal-only case unconverged and excluded | `geometry/` |
| Fixed-step throughput | Supported CPU/GPU paths, matched settings, warmup and five measured repetitions | `performance/`, `plots/performance_table.csv` |
| Bentheimer 128³ loading/smoke | 200-step run executed; deliberately not an accepted permeability | `rocks/smoke_128/` |
| Bentheimer 256³ initial tensor | All three loads timed out at ~180 s; accepted tensor null | `rocks/tensor_256/` |
| Bentheimer 128³ tolerance sensitivity | Both accepted; 100-fold tighter relative/absolute tolerances change full x-response column 2.239e-6 relative | `convergence_review/tolerance_128.json` |
| Bentheimer 256³ tolerance sensitivity | Both accepted; full x-response column changes 8.644e-6 relative; standard stops at 20,700 updates in 356.97 s | `convergence_review/tolerance_256.json` |
| Bentheimer 256³ accepted standard tensor | All x/y/z loads accepted; 921.14 s including the reused x solve; reciprocity 6.392e-7 | `long_rock/tensor_force_0/` |
| Bentheimer 256³ half-force tensor | All loads accepted; 950.92 s; tensor change 2.787e-6 relative; reciprocity 1.516e-6 | `long_rock/tensor_force_1/`, `long_rock/force_sensitivity.json` |
| Accepted 128³ velocity-field export | 7,600 updates; history exactly matches earlier standard run; solve 19.22 s, export 0.94 s | `accepted_128_fields/fields.npz`, `export_evidence.json`, `velocity_slice.png` within that directory |
| Repeated time to accepted permeability | Matching non-cubic cases, warmup plus three repetitions per backend; all accepted | `end_to_end/summary.json` |
| Whole-process CLI timing | Includes new interpreter, imports, input read, solve and JSON/CSV export; all 900-step runs accepted | `cli_timing/summary.json` |

The 3D oblique laminate discrepancy is a recorded discretization limitation,
not hidden by enforcing symmetry. Channel aperture refinement fixes physical
aperture via dx=1/h; the uniform streamwise-period length is immaterial to this
analytical solution. Other synthetic tensors use declared default dx=1; compare
their lattice values or explicitly rescale, not arbitrary physical rock units.

Accepted 256³ raw tensor in units of **10⁻¹² m²**, rows=response and columns=load
(x,y,z), voxel size 5 µm:

```
 3.6559365305  0.1548872389  0.2212277691
 0.1548887178  4.3113093336  0.0438484028
 0.2212275371  0.0438455265  4.4011895524
```

Symmetric-part principal values are 3.5693648045, 4.2964536163 and 4.5026169956
in the same units; eigenvectors are stored as columns in the result JSON.
Directional updates were 20,700 / 14,600 / 15,500. Maximum Mach across loads
was 0.0002929; maximum pore-mean Reynolds number (declared length 20 cells)
was 0.0005059; maximum relative population mass drift was 2.83e-13.

Median synchronized public-call times for the small non-cubic 3D case were
2.403 s (NumPy), 22.948 s (CuPy array) and 0.836 s (CUDA), all 1,100 updates.
Small cases can be dominated by dispatch and diagnostic costs. A separate
16×24 channel CLI case took median 1.121 / 8.673 / 1.108 s respectively,
including process startup and export. These use different geometries; their
timing difference does not measure import overhead. Raw repetitions, variability,
phase costs and concurrent load are retained in the corresponding JSON files.

### Bentheimer provenance and scope

Public source: [DRP-29](https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-29),
DOI 10.17612/P7BC78. The portal metadata declares ODC-BY 1.0; dataset attribution
is Rodolfo Victor, Masa Prodanovic and Petrobras. Exact metadata and all six
downloaded entry checksums are in `dataset/dataset_manifest.json`.

The segmentation is headerless 500³ uint8, 0=pore and 255=solid, 5 µm isotropic
voxels. Segmentation SHA256:
`42bf2c7b771333d1a640afb7b5967b25cd04c78504f9648e457b75749ebe1b2a`.
Arrays use C-order (z,y,x), vector components (x,y,z); physical handedness has
not been independently established. Full porosity is 0.254740648. Deterministic
origin crops `[0:n,0:n,0:n]` have porosity 0.294813632965 at 128³ and
0.250908732414 at 256³. Crop-size variation is NOT grid refinement and a crop's
permeability is not compared directly to a full-volume reference.

CT is little-endian uint16, reference fields little-endian float32. Sizes and
finite values were checked. Reference velocity includes nonzero values in a
one-face-neighbour solid shell; do not silently zero or shift these fields.
An analysis-object dataset UUID differs from the enclosing CT-node UUID; this
metadata inconsistency is retained. Forcing, viscosity, units, boundary conditions
and reference convergence could not be recovered. Consequently this dataset is
an **application example**, not an independent numerical benchmark. No 3D
pressure solver was added to manufacture a matching problem.

Local-thickness example: origin 64³ ROI, original-volume mask, finite solid
exterior, pore-volume weighting, diameters in metres. Pore-volume diameter
percentiles 10/50/90 are approximately 22.36/58.31/82.46 µm; see actual labels
in `dataset/morphology_64/result.json`. These are crop-conditioned measurements,
not full-rock periodic PSD estimates. Plot: `plots/psd.png`.

## 4. Equations and conventions

`nu=(tau-0.5)/3`, rho_ref=1, force input F is force per volume. The Guo velocity
is `u=(sum_q(f_q*c_q)+F/2)/rho` in fluid and zero in solid. Superficial mean U
uses ALL original sample voxels. `K_ij=nu*U_i(load j)/F_j`, from independent
single-component loads; `K_m2=K_lu*dx_m²`. Tensor rows are response directions,
columns load directions, both x/y/z. Reciprocity is Frobenius
`||K-K.T||/||K||`. Principal axes/values are explicitly from `(K+K.T)/2`;
raw K is retained, including negative off-diagonal entries and numerical
near-zero eigenvalues. No forced positive definiteness or symmetrization.

BGK plus quadratic equilibrium is a weakly compressible Navier–Stokes LBM whose
creeping-flow regime must be verified; it is not an inertia-free Stokes solver.
Periodic reflection retains solid-node storage and a one-step return delay.
Pressure mode uses original sample face centres, signed
`G=(mean_fluid(p_in)-mean_fluid(p_out))/(Nx_sample-1)`, and `k=nu*U_sample/G`.
Padding changes its boundary-value problem. Pressure and periodic porous flows
are not interchangeable solver-error comparisons.

The optional PSD is the volume-weighted largest covering open-ball diameter:
EDT supplies candidate radii, then each voxel receives the largest ball covering
it. It is not a nearest-wall histogram. Radius is measured between voxel centres,
diameter is twice radius, exterior is solid, and actual SciPy version is recorded.

## 5. Supported and tested scope

NumPy float64, CuPy-array float64, CUDA float64/float32 STORAGE work in periodic
2D/3D. All collision arithmetic is currently float64, including float32 storage.
GPU-only 2D pressure mode is experimental; float64 pressure is verified here.
Float32 pressure has no comparable qualification campaign. NumPy is a reference
implementation, not an optimized 96-core CPU solver. Python 3.14 was tested;
the existing package declaration Python >=3.8 was retained but older versions
were not tested. Hosted CPU CI passed for the reviewed final head as recorded
above; it does not certify GPU execution.

The qualified float32-storage trial used its predeclared rtol=1e-5 and atol=1e-10;
float64 used rtol=1e-6 and atol=1e-12. It is a storage-accuracy comparison against
the tighter float64 answer, not a matching-tolerance speed comparison.

Convergence requires all velocity means plus RMS field change and conservation
checks for three consecutive samples. This is not an error theorem. See
`docs/gpu_convergence_review.md` for stopping equations, tolerance studies and
read-only inspection of LBPM, Sailfish and FluidX3D. External solvers were not
installed or timed; their published throughput is not a measured speedup here.

## 6. Unresolved issues and blocked cases

* Reference DRP-29 simulation metadata and primary analytical correlation
  assumptions remain insufficient for independent sphere/cylinder/rock claims.
* Full 500³ float64 was not attempted: extrapolated sampled allocation demand
  ~54.1 GB exceeds current free VRAM, before the required 20% margin. Two
  population buffers alone require 38 GB. Float32 rock is not qualified.
* 3D sloping voxel surfaces retain percent-level discretization error; weak-force
  float32 convergence is not reliable. No universal tau or force range is claimed.
* PSD is bounded to small finite ROIs (default <=2.1 million voxels), not periodic
  or a demonstrated full-volume algorithm. Periodic connectivity defaults to
  <=250,000 voxels. No silent pore pruning occurs.
* Memory is sampled allocator reservation plus process-lifetime RSS high-water,
  not an exact per-case physical peak. Concurrent load is recorded where available.
* WSL CUDA-event durations were ~11.1% larger than synchronized host wall time;
  reported comparable throughput consistently uses host wall time.

## 7. Follow-up campaign and remaining costs

The owner authorized long runs, even beyond 60 minutes, after convergence and
easy-optimization review. `benchmarks/convergence_review.py` checks actual rock
tolerance sensitivity; `benchmarks/configs/long_rock.json` specifies sequential
256³ x/y/z loads at F=1e-6 and F/2. Initial pilot history suggested roughly
15 minutes per accepted tensor, with a conservative 600-second cap per load.
Both tensors completed: 15.35 minutes at F and 15.85 minutes at F/2, 31.20 minutes
summed across all six directional solves. The x load at F was reused from the
tolerance study, counted in tensor runtime but not charged twice in the budget.
The independent tighter x run added 9.14 minutes. No 500³ allocation was attempted.
Large performance gains would require separately
validated sparse storage, fused/in-place updates or precision reformulation.

Two additional kernel probes did not justify implementation changes: explicit
solid-node early return gave no gain; conditional integer wrapping gave about
0.23% on the large probe, without a consistent small-case gain. The adopted
2.46% diagnostic-overhead reduction is the only demonstrated code speed gain
from this review. Looser convergence may save updates for a chosen error target,
but the main tensor thresholds were not relaxed.

No further simulation is running for this campaign. Remaining optional numerical
work is qualification of other crops, a matched external reference if its missing
methods become available, or a separately validated memory/performance redesign.
Another same-size tensor can be budgeted at approximately 15–16 minutes under
the measured conditions; a 500³ runtime is not responsibly estimated before its
memory and precision requirements are resolved.

The historical compact review artifact is `final/technical_evidence.tar.gz`: handoff, source
snapshot, configurations, machine-readable results, histories, logs and plots.
Large raw volumes and NPZ fields remain in the evidence directory and are excluded
from this compact bundle; their checksums are in `final/excluded_binary_artifacts.json`.
The tested pre-portability wheel is included. `final/source_sha256.json` and
`final/artifact_checksums.json` identify source and artifact contents; commit,
working-tree status and tracked changes are also saved in `final/`.
`followup_summary.json` records authorized follow-up costs without double charging
the reused x solve; `pilot_budget_ledger.json` retains the initial allowance ledger.
Recorded initial pilot/test time was 1,283.33 seconds (21.39 minutes), below the
1,800-second initial allowance. Recorded authorized follow-up simulation/test
calls total 2,983.80 seconds (49.73 minutes). This sums calls, including warmups
and failures; some CPU checks overlapped GPU work. It excludes ordinary code,
installation and download time and is not total elapsed project time.

## New local overnight campaign (separate evidence)

A subsequent, unpushed local branch `overnight-rock-evidence-20260918` adds a
configurable bounded campaign runner, streaming NPY/VTI export, conservative
streamline plotting and explicit public-call performance metrics. It does not
change numerical equations, convergence rules or any historical result file.
The working campaign is `results/overnight-20260918T154529Z`; its `frozen.json`
and source copy identify the new calculations independently of the historical
snapshots described above. Execution instructions and GPU architecture are in
`docs/overnight_campaign.md`.

Before production: focused tests **10 passed (2.20 s)**; installed-wheel tests
outside the checkout **7 passed (2.15 s)**, with all 19 package Python files
verified against the wheel installation. A real CUDA allocation/kernel test
passed. An actual non-cubic 32×34×36 rock campaign accepted all three directional
loads, exported NPY/VTI fields, verified exact-case reuse and generated its
report. The first renderer attempt exposed a VTK API incompatibility; its failure
was retained and the corrected render was executed successfully. The later
physical-axis render check also completed. These are workflow checks, not large
rock results.

The eight-hour production campaign is launched separately after these checks.
Consult its `status.json`, `campaign.log`, `case_inventory.csv` and eventual
`finished.json` for actual progress. Its final technical handoff will be
`report/OVERNIGHT_HANDOFF.md`, with small evidence in
`report/technical_evidence.tar.gz`; full fields remain separate. No pending
large calculation is claimed complete here. The independent comparison remains
blocked for the specific reasons recorded in `independent_comparison.md` inside
the new campaign, including unqualified local LBPM source/binary and output
conventions. The existing external installations were left untouched.
