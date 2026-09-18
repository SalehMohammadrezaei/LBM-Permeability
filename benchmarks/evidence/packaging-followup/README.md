# Packaging portability follow-up

This follow-up makes Git metadata and process RSS optional. It does not change
the numerical equations, convergence thresholds or archived benchmark results.

Executed on the existing Linux/WSL workstation:

| Check | Result |
|---|---|
| Focused packaging and CLI regressions | 17 passed, 1.76 s |
| Full source suite, real GPU available | 124 passed, 1 skipped, 43.25 s |
| Fresh built-wheel installation outside the repository, CPU-only environment | 59 passed, 10 skipped, 25.36 s |

The GPU suite's skip is the inapplicable 2D z direction; the installed CPU run
also skips GPU-dependent cases. All 18 installed package source files match
the tested checkout. The fresh-process regression removes Git from PATH and
blocks `resource` imports before package import, then verifies an accepted CPU
solve, NPZ/JSON/CSV export and CLI missing-file error reporting. Other regressions
cover failed/timed-out Git, RSS unavailability, Linux/macOS units and pressure
export without RSS. Native Windows/macOS execution was not performed.

`summary.json` records commands, hashes and exact JUnit counters. Initial focused
and source attempts failed collection because the new pressure metadata had an
extra closing parenthesis; those logs/XML are retained. The corrected suites
passed as above. The original campaign under `2026-09-18/` was not edited or
rerun. Local build outputs and additional logs remain in
`results/packaging-followup/` and are excluded from Git.
