# Local reproduction

Selected executed records are tracked in `benchmarks/evidence/2026-09-18/`, with
a source-path/checksum manifest. The full `results/` directory is local and ignored
by Git; paths below refer to that complete workstation campaign. Download raw
data and use fresh output directories when reproducing from a new clone.

No remote publication, license changes or modifications to other users' processes
are required. The initial environment is `.venv`, installed with NumPy, pytest,
SciPy, matplotlib, build and cupy-cuda12x. Exact versions are in
`results/2026-09-18-pilot/baseline/environment.json`; baseline user changes are
preserved in `baseline/user_changes.patch`. Full source hashes accompany every run.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python benchmarks/inspect_environment.py results/environment.json
.venv/bin/python -m pytest -q
.venv/bin/python benchmarks/run_cases.py --suite verification \
  --campaign results/new-verification --budget 850
.venv/bin/python benchmarks/fetch_bentheimer.py --output results/dataset
```

`run_cases.py --case SUBSTRING` selects a case, and `--config FILE.json` loads
explicit definitions. The generated `case_definitions.json` is sufficient to
repeat exact masks and parameters. Resume only accepts matching case definitions
AND package source hashes; use a new campaign directory after code changes.
Timeout, maximum-step and exception cases remain in result directories. Each
result has strict JSON (unavailable numbers are null), parameter settings,
source hashes, original-volume porosity, convergence history and memory/timing
semantics. Fixed-step states intentionally end at `max_steps`: these are throughput
or implementation checks, not valid permeability measurements.

Record resource availability again before a large case. Use one GPU process and
at most two small CPU workers. Retain 20% of currently free VRAM as a planning
margin and at least tens of GiB host RAM. Initial on-disk budget is 10 GiB.
The 30-minute simulation allowance includes validation and timing pilots; longer
simulation campaigns need owner approval. Ordinary code work is not stopped by
that simulation checkpoint.

Real-GPU release checks must execute, not just skip: CPU/CuPy-array/CUDA fixed-step
states, separately accepted permeabilities, x/y/z directions on non-cubic grids,
float32 storage versus float64, invalid inputs, failure statuses and pressure
mode. GPU-less CI checks only the CPU scope and does not certify CUDA.

Performance memory fields are explicit: process RSS is the lifetime high-water
mark, and GPU memory is sampled CuPy pool reservation including cached blocks.
These are not claimed to be exact per-case physical-device peaks. Clear unused
pool blocks only for this process between cases. Timings synchronize execution;
setup, solve with diagnostics, export, and full tensor costs are distinguished.

## Additional executed tools

```bash
.venv/bin/python benchmarks/performance.py --budget 280 --output results/new-performance
.venv/bin/python benchmarks/inspect_bentheimer.py results/2026-09-18-pilot/dataset
.venv/bin/python benchmarks/rock_application.py
.venv/bin/python benchmarks/summarize.py --campaign results/2026-09-18-pilot/verification
.venv/bin/python benchmarks/legacy_correlations.py
.venv/bin/python benchmarks/plot_evidence.py results/2026-09-18-pilot
```

The rock pilot script uses fixed paths to the audited dataset, 128³ smoke (200
steps), and sequential 256³ loads capped at 180 seconds each. Run it only within
an available simulation budget. It does not continue a population checkpoint.
The owner subsequently authorized longer runs, including beyond 60 minutes if
needed, after a convergence/performance review. The reviewed configuration is
`benchmarks/configs/long_rock.json`; its default runner cap is 60 additional
simulation minutes. The executed follow-up commands are:

```bash
.venv/bin/python benchmarks/convergence_review.py --mode performance --size 256
.venv/bin/python benchmarks/convergence_review.py --mode convergence --size 128
.venv/bin/python benchmarks/convergence_review.py --mode convergence --size 256
.venv/bin/python benchmarks/long_rock.py \
  --output results/2026-09-18-pilot/long_rock \
  --reuse-x results/2026-09-18-pilot/convergence_review/rock_256_standard/result.json
.venv/bin/python benchmarks/evidence_index.py results/2026-09-18-pilot
.venv/bin/python benchmarks/end_to_end.py --output results/new-end-to-end
.venv/bin/python benchmarks/cli_timing.py --output results/new-cli-timing
.venv/bin/python benchmarks/export_rock_fields.py --output results/new-accepted-fields
.venv/bin/python benchmarks/probe_solid_shortcut.py --kind solid --output results/new-probes
.venv/bin/python benchmarks/probe_solid_shortcut.py --kind wrap --output results/new-probes
.venv/bin/python benchmarks/replay_convergence.py
.venv/bin/python benchmarks/plot_rock.py
.venv/bin/python benchmarks/summarize_followup.py
.venv/bin/python benchmarks/export_handoff.py
```

For performance reproduction, `solver_before.py` must be present in the review
output directory: it is the saved pre-optimization package solver. Run into a new
directory with `--output` to preserve existing evidence. Long-run reuse checks
mask/source hashes, accepted status and numerical settings. It retains the
original load's actual caps, provenance and cost, includes its elapsed time in
tensor time, and does not count its simulation budget twice. Omitting `--reuse-x`
repeats all six loads from rest. Never overwrite archived outputs during a repeat.
The final CLI missing-file fix changed the package source hash after the accepted
tensor campaign. To reuse that campaign's accepted x load, restore the package
from `results/2026-09-18-pilot/accepted_campaign_source.tar.gz` into an isolated
copy, or run fresh loads in a new directory with the final source. Do not bypass
the resume hash check. The numerical solver files are identical between these
two snapshots. The field-export tool uses the saved 128³ settings and dataset;
it also produces its labelled velocity-slice PNG/PDF after exporting the NPZ.

The performance plot uses **synchronized entire public-call wall time** for its
MLUPS denominator. The older `solve_and_diagnostics_s` in archived raw pilots
means the iteration loop and scheduled checks, excluding mandatory initial/final
checks. Do not quote this as end-to-end throughput. Updated result timing reports
those checks separately. Setup includes cold compilation when incurred. Exports
are separately timed and are off for throughput runs. Tensor elapsed time includes
all loads; failed end-to-end runs are not timing successes.

For a clean CPU wheel check:

```bash
.venv/bin/python -m build
python3 -m venv /tmp/lbm-installed-cpu
/tmp/lbm-installed-cpu/bin/python -m pip install dist/*.whl pytest scipy
mkdir -p /tmp/lbm-installed-check/tests
cp tests/test*.py /tmp/lbm-installed-check/tests/
cd /tmp/lbm-installed-check
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/lbm-installed-cpu/bin/python -m pytest -q tests
/tmp/lbm-installed-cpu/bin/python -c 'import lbm_permeability; print(lbm_permeability.__file__)'
```

The installed module path must be inside `/tmp/lbm-installed-cpu`, not the source
checkout. GPU skips here are expected and do not replace the recorded workstation
suite. Wheel checks should be repeated after subsequent source changes.

The executed GPU wheel check used the same isolated dependencies but imported
the built package from a separate directory. From the repository, then `/tmp`:

```bash
.venv/bin/python -m pip install --no-deps --target /tmp/lbm-installed-gpu-release dist/*.whl
cd /tmp/lbm-installed-check
PYTHONPATH=/tmp/lbm-installed-gpu-release OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/impres/Saleh/lbm-permeability/.venv/bin/python -m pytest -q tests
PYTHONPATH=/tmp/lbm-installed-gpu-release \
  /home/impres/Saleh/lbm-permeability/.venv/bin/python -c \
  'import lbm_permeability; print(lbm_permeability.__file__)'
```

Use fresh target directories on repeat to avoid stale installations. The recorded
`installed_cpu_release_source.json` and `installed_gpu_release_source.json` verify
every installed package Python file against the checkout. The final compact bundle
is `results/2026-09-18-pilot/final/technical_evidence.tar.gz`; checksums are beside
it. Raw dataset volumes and NPZ fields are deliberately kept outside the compact
archive. Fetch commands and dataset checksums permit their reconstruction.
