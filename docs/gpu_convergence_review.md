# GPU cost and convergence review

The owner authorized simulation beyond the initial 30-minute pilot, including
more than 60 minutes if necessary, conditional on examining convergence and
straightforward performance improvements first. No further budget permission
is needed for the documented follow-up. One GPU process is used at a time.

## What stopping means

For periodic cases, checks use post-update populations and velocity with the
Guo half-force correction. Every Cartesian superficial mean is checked,
including transverse responses. At successive checks separated by 100 updates:

```
abs(U_i[n] - U_i[n-1]) <= atol + rtol * max(abs(U_i[n]), abs(U_i[n-1]))
sqrt(mean(sum_i((u_i[n]-u_i[n-1])**2))) <= atol + rtol * sqrt(mean(sum_i(u_i[n]**2)))
```

All components and the field check must pass three consecutive times, with
relative total population mass drift <= 1e-8 for float64. Density must be finite
and positive in fluid, populations and velocities finite, and final maximum
Mach <= 0.05. Periodic total mass includes populations stored in solid nodes by
the existing reflection algorithm; superficial velocity assigns zero velocity
to solids and averages over the entire original volume. Timeouts and step caps
are failures, even when the instantaneous permeability looks plausible.

This is a steady-state heuristic, not an a priori bound on permeability error.
A slowly decaying mode can have a small change per monitoring interval while
retaining a larger total remaining error. The check interval is part of the
definition: the same numeric tolerance at a different interval is not equivalent.
Absolute tolerance protects zero components but can dominate tiny flows. The
rock follow-up therefore tightens BOTH tolerances by 100 (1e-6/1e-12 to
1e-8/1e-14) and compares the full response column. Acceptance is relative
column difference < 0.001, fixed before execution. Force halving separately
tests the linear response regime; it cannot replace tolerance sensitivity.

## Bounded optimization

`solver_before.py` in the convergence-review evidence preserves the old loop.
Intermediate stability checks formerly computed unused velocity fields. At full
monitor checks the loop also repeated population/density checks performed by the
monitor. The revised loop computes only density for stability-only checks and
leaves full validity checks to the full monitor. Check intervals, tolerances,
collision, streaming, force, boundary treatment and final mandatory checks are
unchanged. `benchmarks/convergence_review.py --mode performance --size 256`
compares interleaved before/after 500-update calls, excludes the first pair as
warmup, and checks exact equality of entire diagnostic histories.

The earlier rock profile found full diagnostics cost about 0.066–0.085 seconds
per warmed check, versus roughly 1.75 seconds per 100 complete updates. Removing
all convergence checks could not explain an order-of-magnitude speedup. Host
wall clocks agreed, but CUDA event elapsed time was about 11.1% larger under
this WSL environment. Use synchronized host wall time consistently; event and
wall measurements must not be mixed when computing throughput or speedups.

## Comparable public implementations

Sources were inspected read-only, with exact commits, URLs and checksums in
`results/2026-09-18-pilot/convergence_review/external/sources.json`. None was
installed or used to generate a reference permeability. No external code was
copied into this package and its license was not changed.

* [LBPM](https://github.com/OPM/LBPM) is directly relevant: its permeability
  executable invokes `MRTModel`, whose inspected implementation uses a pore-node
  layout and alternating even/odd updates. It checks relative change in the
  flow component parallel to forcing every 1000 steps. Its
  [steady-state tutorial](https://github.com/OPM/LBPM/wiki/LBPM-Tutorial,-Step-5.-Assessing-Steady-State-Permeability)
  explicitly shows that a permeability history can still drift and require
  more updates and a tighter tolerance. Our full-vector/field checks give more
  information, but neither rule alone proves a specified permeability error.
  Sparse pore-node storage is worth a separate implementation study for this
  approximately 25% porosity rock, with its neighbour-list and boundary costs.
* [Sailfish](https://github.com/sailfish-team/sailfish) provides an LGPLv3 GPU
  LBM implementation for CUDA/OpenCL. It is a useful independent implementation
  candidate, but a matching rock case and numerical conventions would need to
  be established before a reference or performance comparison.
* [FluidX3D](https://github.com/ProjectPhysX/FluidX3D) documents in-place streaming,
  FP32 arithmetic, separately selectable population storage, and fused updates.
  Its performance table uses an empty cube and different precision and boundary
  machinery. Those timings are not a measured speedup over this code. It is
  publicly readable with noncommercial licensing, not unrestricted open source.
  Its design motivates independent work on memory traffic and precision, but
  directly adopting its implementation would raise licensing considerations.

Changing this solver's float32 *storage* option does not select FP32 arithmetic:
current CUDA collision arithmetic remains double. Weak forcing already failed
float32 acceptance in the recorded porous tests. In-place streaming, sparse
layout, shifted populations, FP32 arithmetic and a new collision model are
substantial changes needing separate parity, precision and boundary validation.
They are not unverified switches to enable for the accepted rock campaign.

## Tolerance versus requested accuracy

`benchmarks/replay_convergence.py` replays other stopping tolerances on a saved
tight x-load trajectory. This is labelled post-hoc sensitivity, not a new
accepted simulation or a changed main-campaign rule. For the 128³ crop, rtol
1e-4 with atol 1e-12 would first pass three checks at step 4000; the full response
column there differs from the tight final column by 0.0181%, within the previously
declared 0.1% comparison target. Rtol 1e-3 would stop at step 2500 and misses that
target (0.163%). The main tensor campaign retains rtol 1e-6. These observations
show why tolerance must be calibrated against permeability accuracy; they do
not establish a universally safe faster setting for other rocks or directions.

The independent 256³ x-load check also passed: standard stopping was at 20,700
updates (356.971 seconds); 100-fold tighter stopping gave a full response-column
change of 8.6439e-6 relative (0.0008644%). Both runs together cost 905.653 seconds.
At rtol 1e-4 the saved tight trajectory would pass at step 11,300, with 0.0774%
column error; rtol 1e-3 would pass at 6600 but have 0.777% error. This is further
evidence against choosing a tolerance by runtime alone. Main runs remain at 1e-6.

An additional generated-kernel experiment explicitly copied solid populations
and returned before collision intermediates. On the 256³ rock, three warmed
200-step loop timings had median 3.464 seconds for the original and 3.474 seconds
for the shortcut: no observed gain. Float64 populations differed by at most
4.44e-16 after 200 steps; small float32 states were identical. This experiment
was not incorporated into the solver. Its sources are reproducible through
`benchmarks/probe_solid_shortcut.py`; raw timings and generated-source hashes
are in `convergence_review/solid_shortcut_probe.json`.

A second probe replaced periodic remainder calculations in the streaming kernel
with conditional wrapping for one-cell offsets. All tested populations were
bitwise identical on the non-cubic small geometries (float64/float32) and the
256³ float64 rock. The large-case warmed median changed only 3.627 → 3.619
seconds per 200 updates (0.23%); small-case timings did not show a consistent
benefit. This change was also not incorporated. These probes do not establish
an easy large speedup. Their failed performance expectations are retained in
`convergence_review/stream_wrap_probe.json`.
