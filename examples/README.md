# Examples

Runnable baseline scripts, one per solver, mirroring the MATLAB repo's
`examples/` one-to-one (same configurations, same outputs). Run from the
repository root, e.g.:

```bash
python examples/run_surface_point.py
```

| Script | Solver | Runtime |
| --- | --- | --- |
| `run_surface_point.py` | 0D surface point, 50 pulses | < 1 s |
| `run_single_pulse.py` | single-pulse visualizer | < 1 s |
| `run_radial_profile.py` | radial profile, 100 pulses | ~1 s |
| `run_depth_profile.py` | 1D depth profile, 100 pulses | ~15 s |
| `run_inversion_quantifier.py` | inversion statistics, 20 pulses | ~10 s |
| `run_scanning_beam.py` | 2 mm scan at 1 m/s (36,000 pulses) | ~10 min |

Each script is a thin, editable wrapper: change the `cfg`/`params` dict to
explore other materials, powers, repetition rates, pulse widths, or spot
sizes. Text output (and figures, for the scanning solver) land under
`outputs/`; the other solvers show figures interactively via
`matplotlib.pyplot.show()` and save PNGs when `saveFigures` is true.
