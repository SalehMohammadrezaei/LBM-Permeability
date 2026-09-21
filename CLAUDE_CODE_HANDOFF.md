# Handoff to Claude Code — independent review requested

Work in `/home/impres/Saleh/lbm-permeability` on the owner's workstation.
Repository: https://github.com/SalehMohammadrezaei/LBM-Permeability
This document was prepared on 2026-09-20. Review the actual files and evidence;
do not assume the previous assistant's conclusions are a substitute for review.

## Scope and preservation

The owner requests code, testing, numerical evidence and software documentation.
Do not edit a manuscript or submission material. Preserve existing changes,
original pressure-wall work, all raw results and original source hashes. Do not
reset, overwrite historical outputs, stop the owner's `porous_flow-opt` processes,
change system CUDA/drivers, push, merge, publish or change licensing without new
authorization. The latest campaign authorization was eight hours, not unlimited
permission for another campaign. Start with read-only review; propose measured
costs before another large campaign. Use the isolated `.venv`.

One GPU simulation process at a time; limit additional CPU work to 16 physical
cores where controllable. Recheck resources: reserve 20% of currently free VRAM,
50 GiB available RAM and adequate disk space. These are a shared workstation's
available resources, not guaranteed capacity. No external credentials are needed
for local review; do not copy credentials from conversation history.

## Git and historical work

Current local branch: `overnight-rock-evidence-20260918`.
Current local commit: `3057489472a4c1a1cb9f470ed3cf5d54a268ffb6`.
The new branch has NOT been pushed. Working tree was clean immediately before
creating this handoff; this handoff itself is a new uncommitted documentation file.

Previous main documentation commit: `1633a95863d7dba1628cbfc58417983d72a42970`.
PR #1 was merged with merge commit `7b5908a0cf34c3e78e5790c8142caa6894bdf1b9`.
Its reviewed head was `84649088294e6b29dd186b98de8e60b141bdd81a`.
Original owner pressure-wall commit: `7489f432e17b6648b23980981724505dbe3b4553`;
patch: `docs/provenance/original_pressure_walls.patch`.

Previously merged work includes strict validation/backend selection, explicit
failure statuses, field/conservation convergence checks, sequential 2D/3D tensors,
porosity and exact local thickness, pressure-mode tests, CLI/error handling,
optional Git provenance and platform-safe RSS reporting. See the existing handoff
for the complete earlier changes and qualifications; these were not newly added
by the overnight commit.

Start with:

```bash
cd /home/impres/Saleh/lbm-permeability
git status --short
git log -5 --oneline
git diff 1633a95863d7dba1628cbfc58417983d72a42970..3057489472a4c1a1cb9f470ed3cf5d54a268ffb6
```

Read applicable AGENTS.md instructions, then these files completely:

- `CODE_AND_BENCHMARK_HANDOFF.md` (its final overnight section was written before production and is now stale about pending status).
- `docs/benchmark_reproduction.md` (mostly the earlier campaign).
- `docs/numerical_limits.md`, `docs/gpu_convergence_review.md`.
- `docs/overnight_campaign.md`.
- `results/overnight-20260918T154529Z/report/OVERNIGHT_HANDOFF.md`.
- The new code listed below and the existing solver, CUDA kernels, diagnostics,
  tensor, morphology, validation and memory helpers.

## Exact new/modified files in the overnight commit

| File | Change |
|---|---|
| `benchmarks/campaign.py` | New configurable parent runner: sequential task processes, source/config freezing, completed-case reuse, resource sampling, pilot-based crop selection, execution budget, case inventory. |
| `benchmarks/campaign_worker.py` | New solver/geometry/morphology/report worker, immediate per-load results, NPY/VTI export, repeated timings. |
| `benchmarks/configs/overnight_rock.json` | New explicit dataset, ordered candidate crops, force and float64 convergence configuration. Dataset path is workstation-specific and must be edited in a copied config on another machine. |
| `lbm_permeability/field_export.py` | New streaming appended-raw VTK ImageData writer; cell data, physical spacing/origin, x/y/z vector ordering. |
| `benchmarks/rock_streamlines.py` | New qualitative streamline integration with conservative all-pore interpolation stencils and checked short segments. |
| `benchmarks/campaign_report.py` | New tensor/sensitivity summaries, charts, cutaway rendering, artifact manifests and compact archive. |
| `benchmarks/performance.py` | Added explicitly named public-call and iteration-loop MLUPS fields; retained legacy fields and historical records. |
| `tests/test_campaign.py` | New identity/fingerprint regression. |
| `tests/test_field_export.py` | New non-cubic VTI roundtrip through an independent VTK reader, plus invalid-shape check. |
| `tests/test_rock_streamlines.py` | New boundary/solid stopping and interpolation-stencil tests. |
| `docs/overnight_campaign.md` | New execution/export conventions and actual GPU architecture. |
| `CODE_AND_BENCHMARK_HANDOFF.md` | Added pre-production workflow/test handoff section. |

The overnight commit did not change solver equations, convergence logic, CUDA
collision/streaming kernels, or the exact morphology algorithm. VTK 9.7.0 was
installed into `.venv` for testing/rendering, not globally. The new solver package
module does not require VTK to write VTI files.

## Campaign and result locations

All paths below are relative to the repository. Let
`C=results/overnight-20260918T154529Z`.

The unattended campaign finished, with `finished.json` reporting 17,187.70 s
(about 4 h 46 min), report exit 0, and unchanged source. All 26 task execution
records reported completed; fixed-step states intentionally have `max_steps`.
Ten accepted solve cases are baseline x/y/z, half-force x/y/z, tight-tolerance x,
and tau 0.8/1.0/1.2. No independent external-solver calculation was completed.

| Location under C | Contents |
|---|---|
| `request.txt` | Full original overnight requirements. |
| `launch.json`, `campaign.log`, `finished.json`, `status.json`, `case_inventory.csv` | Launch/source commit, progress, completion and case inventory. |
| `frozen.json`, `source/`, `session_*/` | Exact config, package/benchmark/test hashes, source copies and session records. |
| `environment.json`, `selection_plan.json` | Hardware/dependencies/actual CUDA kernel test and predeclared crop selection. |
| `pilot_64/`, `pilot_128/`, `pilot_256/`, `selected_preflight/` | Fixed 300-update diagnostics/export pilots, not accepted permeability. |
| `geometry/attempt_01/` | Unchanged Boolean mask, checksum provenance, porosity and x/y/z profiles. |
| `baseline_x/attempt_01/`, `baseline_y/attempt_01/`, `baseline_z/attempt_01/` | Accepted baseline results, histories, execution/resource records, full fields. |
| `half_x/`, `half_y/`, `half_z/`, `tight_x/` | Accepted force and tolerance sensitivity loads; actual files in `attempt_01/`. |
| `morphology_profile_32/`, `morphology_profile_64/`, `morphology_profile_96/`, `morphology_plan.json` | Exact morphology timing profiles and explicit large-size planning. |
| `morphology_main/attempt_01/` | Full-support exact local-thickness map, histogram/CDF, percentiles and boundary-interior exclusions. |
| `performance_cuda_64/`, `performance_cuda_128/`, `performance_cuda_256/`, `performance_cuda_384/` | Warmup plus three 1,000-update float64 repetitions at each size. |
| `matched_numpy/`, `matched_cupy-array/`, `matched_cuda/` | Matching 32-cubed fixed-step backend timings; not an independent numerical comparison. |
| `tau_0.8/`, `tau_1.0/`, `tau_1.2/` | Accepted centered 128-cubed rock relaxation-time study. |
| `independent_comparison.md`, `independent_source_inspection.json` | Read-only MPLBM-UT/Palabos/LBPM investigation and explicit blocker. |
| `historical_preservation_check.json` | All 535 original evidence-file hashes unchanged at the recorded check. |
| `report/all_attempts.csv`, `report/timings.csv` | Consolidated attempt and timing tables. |
| `report/baseline_tensor.json`, `report/half_tensor.json`, `report/sensitivity.json` | Raw tensors, symmetric-part principal systems, reciprocity and sensitivity outcomes. |
| `report/OVERNIGHT_HANDOFF.md` | Generated completed-campaign handoff; generic wording does not substitute for numerical review. |
| `report/technical_evidence.tar.gz`, `report/technical_evidence.sha256` | Approximately 7 MB compact evidence bundle and checksum; large fields excluded. |
| `report/artifact_checksums.json`, `report/large_artifacts.json` | Small-artifact hashes and separate large-file paths/hashes. |
| `accepted_384_vti.tar.gz`, `accepted_384_vti.tar.gz.sha256` | Later requested archive: three accepted VTI loads plus metadata/results/provenance; about 1.44 GB. All archived VTI hashes were checked against originals. |

Each baseline folder contains `ux.npy`, `uy.npy`, `uz.npy`, `rho.npy`,
`fields.vti`, `field_metadata.json`, `result.json`, `convergence.csv`,
`provenance.json`, `job.json` and execution/resource records. No full physical
velocity scale was supplied: velocity is in lattice units, not m/s.

Figures in `C/report/`: `rock_cutaway.png`, `speed_slice.png/.pdf`,
`speed_distribution.png/.pdf`, `tensor.png/.pdf`,
`principal_directions.png/.pdf`, `local_thickness.png/.pdf`.
Plotting inputs include `streamlines.json`, `streamlines.vtp`,
`render_metadata.json` and `speed_distribution.json`.
`plot_errors.json` is empty for the final report.

Earlier campaign: `results/2026-09-18-pilot/`, with selected tracked evidence in
`benchmarks/evidence/2026-09-18/` and `benchmarks/evidence/packaging-followup/`.
Do not confuse its 256-cubed origin crop with the new centered 384-cubed crop.

## Numerical settings and observed results

Dataset: DRP-29 Bentheimer, DOI 10.17612/P7BC78; raw file
`results/2026-09-18-pilot/dataset/Seg_Oxyz_0001_0001_0001.raw`.
SHA256: `42bf2c7b771333d1a640afb7b5967b25cd04c78504f9648e457b75749ebe1b2a`.
Headerless 500-cubed uint8, 0=pore, 255=solid, C-order (z,y,x), dx=5e-6 m.
Dataset attribution/license and unresolved reference metadata are in the older
manifest/handoff. The original portal simulation is not a matched benchmark.

Selected crop: `[58:442,58:442,58:442]`, 384 cubed. Candidate 400 cubed fit the
memory estimate but exceeded the conservative runtime allocation. Selection was
made before inspecting candidate permeability. Do not relabel support-size
variation as grid refinement.

CUDA, float64 storage and arithmetic, BGK/Guo, periodic all axes; independent
rest initialization. Baseline tau=1, F=1e-6; rtol=1e-6, atol=1e-12; check every
100, stability every 50, three consecutive passes, min_steps=300; mass_tol=1e-8,
max_mach=.05; characteristic_length=20 cells is only a diagnostic convention.
Read each job/result for actual maximum-step and wall caps.

nu=(tau-0.5)/3. Force is force per volume, rho_ref=1. Superficial U averages the
entire original sample. K_ij=nu*U_i(load j)/F_j; K_m2=K_lu*dx². Rows=response,
columns=load, x/y/z. Raw signed tensor retained; principal values/directions are
from (K+K.T)/2, not a replacement for raw K.

384-cubed baseline K in 10^-12 m²:

```
4.1760619006   0.2099906886   0.1208977325
0.2100260906   5.0972477878  -0.0270400789
0.1209184617  -0.0270350742   4.8661490206
```

Reciprocity ||K-K.T||/||K|| = 7.1288e-6. Baseline all-load public-call time is
5543.48 s; this is not the total campaign time or full disk-export cost.
Half-force tensor relative Frobenius change = 2.0560e-6 (0.0002056%).
Half-force x-column change = 2.5258e-6 (0.0002526%).
100x tighter x tolerance column change = 6.8432e-6 (0.0006843%).
All passed the predeclared relative target 0.001 (0.1%).

Porosity = 0.2541225221421983. Exact local-thickness p10/p50/p90 diameters are
22.3607/50.9902/99.4987 micrometres. Full morphology swept 315 distinct EDT radii;
algorithm elapsed approximately 1726 s. It computes largest covering open-ball
diameters, not a nearest-wall histogram. Equal pore-voxel weighting includes
isolated pores. Finite solid exterior differs from periodic flow. Check boundary
influence records before interpreting the PSD.

## Tests actually executed for the new work

- `C/focused_final_tests.xml`: 10 passed, 2.20 s.
- `C/installed_wheel_tests.xml`: 7 passed, 2.15 s, outside checkout.
- `C/installed_source.json`: all 19 installed package Python files matched source.
- `C/environment.json`: actual CUDA allocation/kernel test passed.
- `C/runner_validation_final/`: actual non-cubic 32x34x36 x/y/z solves accepted,
  full NPY/VTI export, exact-case reuse, report generation; about 25 s overall.
- `C/runner_validation_executed/report/plot_errors.json`: retained first VTK
  AddActor2D API failure. Corrected render subsequently ran successfully.
- `C/render_check/`: later physical-axis rendering check.
- `C/runner_validation/`: initial short-budget harness skipped solves; do NOT
  count it as executed numerical validation.

Focused reproduction (inspect resources first):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_campaign.py tests/test_field_export.py \
  tests/test_rock_streamlines.py tests/test_outputs.py
```

These tests are not a new full release suite. Earlier full CPU/GPU tests belong
to their documented source snapshots. VTK-dependent tests may skip if VTK is
missing; a skip does not certify VTI roundtrip behavior.

## Review priorities — do not assume correctness from convergence

1. **Relaxation-time sensitivity is material.** In the centered 128-cubed rock,
   accepted k_lu values were 0.088973048852 (tau .8), 0.099900894019 (tau 1),
   and 0.109675001145 (tau 1.2). Relative changes vs tau 1 are -10.9387% and
   +9.7838%. Investigate scaling, forcing, convergence adequacy and voxel-wall
   reflection/discretization. Do not claim the cause is proven or that this
   measures the error of the 384-cubed result. Tight tolerance/force agreement
   alone does not establish physical accuracy. Preserve the findings.
2. **Overall runner failure propagation needs work.** `Campaign.task` can return
   None on a child failure without raising; mandatory failed baseline loads can
   leave the script's final exit code zero. `finalize` records report return code
   but main does not propagate it. Add meaningful child/required-load/report
   failure regressions if fixing this. Saved successful overnight task records
   are not invalidated by this unexercised failure-path weakness.
3. **Audit budgets and resume thoroughly.** Review deadline use, final-report
   allowance, system-clock dependence, resource-sampling exceptions, per-case vs
   aggregate completion, all measured repetitions (not just the final one), and
   whether re-selection/adaptive caps reproduce intended resume behavior. The
   fingerprint unit test and successful single-case reuse are limited coverage.
   There are no population checkpoints; interrupted loads restart from rest.
4. **Audit output integrity and report selection.** Review partial exports,
   completion markers, result/field hashes, attempt selection and source/config
   matching during standalone report regeneration. Do not overwrite original
   reports while experimenting: use a separate review output/copy. The existing
   report command writes into CAMPAIGN/report and is not non-destructive.
5. **Check numerical/morphology evidence independently.** Recompute tensors and
   sensitivity from saved means/forces, check histories, mass/density/Mach/Re,
   histogram normalization, diameter conventions, boundary exclusions, and
   indexing. Review performance denominators, repeated variability and measured
   concurrent load; no generic GPU speedup claim follows from one geometry.
6. **Inspect visualizations.** Original-resolution fields/mask are used for paths;
   solid rendering is display-subsampled. Check interpolation/segment safety,
   boundary termination, physical axes, VTI cell association and seed metadata.
   Unweighted streamline density is qualitative, not flux. Review figure quality
   rather than assuming error-free rendering means publication quality.
7. **Independent comparison remains blocked.** Local LBPM source lacked Git
   identity and verified source-to-binary correspondence; inspected scalar output
   used an extra porosity factor and momentum conventions differed. The existing
   Palabos driver was 2D. Do not claim a matched external result or compare with
   unspecified portal conditions. Read the recorded inspection before new setup.
8. **Update stale documentation only after review.** Main handoff's overnight
   section predates execution; generated handoff lacks detailed tau analysis.
   Keep historical source identities separate and preserve original result bytes.

Requested deliverable from Claude: a severity-ranked independent review with
file/line references, explicit confirmed vs suspected issues, recomputed checks
from existing results, and a bounded remediation/test plan. Explain what evidence
supports the software and which physical-accuracy claims remain unsupported.
Do not begin another large simulation or publish changes merely to complete the
review. If implementing authorized fixes, use fresh evidence directories and
identify the new source; never silently relabel the existing campaign.
