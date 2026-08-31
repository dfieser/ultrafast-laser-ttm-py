# Validation against the MATLAB reference implementation

This directory holds the machinery that makes the port's central claim
checkable: *the Python solvers reproduce the MATLAB reference implementation
within stated tolerances.*

## Contents

- `generate_fixtures.m` — run once in MATLAB (`matlab -batch generate_fixtures`
  from this folder, with the MATLAB repo as a sibling directory named
  `ultrafast-laser-ttm-toolbox`). Runs every solver on a fixed set of
  configurations, including the exact configs of the MATLAB repo's baseline
  examples.
- `fixtures/<case>.mat` — authoritative full-precision result structs
  (MATLAB `-v7`, readable via `scipy.io.loadmat`).
- `fixtures/<case>.json` — human-readable companions (NaN/Inf serialize
  as null).
- `fixtures/manifest.json` — MATLAB version, toolbox version, and per-case
  wall times (these double as the performance baseline for the port).

## Why tolerances, not exact equality

MATLAB's `ode15s` (variable-order NDF) and SciPy's `BDF` are different
integrators of the same family; agreement is expected at the level of the
solver tolerances (`RelTol`/`AbsTol`), not bit-for-bit. The test suite in
`tests/` states an explicit tolerance per compared quantity.

## Provenance

Fixtures were generated with MATLAB R2026a against
[ultrafast-laser-ttm-toolbox](https://github.com/dfieser/ultrafast-laser-ttm-toolbox)
at the version recorded in `fixtures/manifest.json`.
