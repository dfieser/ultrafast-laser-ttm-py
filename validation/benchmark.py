"""Compare laserttm solver wall times against the MATLAB reference.

MATLAB baselines come from validation/fixtures/manifest.json (recorded when
the golden fixtures were generated). Run the Python side with:

    python validation/benchmark.py [case ...]

With no arguments, benchmarks every fixture case whose solver is ported.
The first Numba call includes JIT compilation; each case is run twice and
the second (warm) time is reported.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from tests.conftest import load_fixture

# fixture-case prefix -> ported entry point (grows as solvers land)
PORTED = {}


def _register():
    from laserttm import (
        depth_profile_solver,
        inversion_quantifier,
        radial_profile_solver,
        scanning_beam_solver,
        single_pulse_visualizer,
        surface_point_solver,
    )

    PORTED["surface_point"] = surface_point_solver
    PORTED["single_pulse"] = single_pulse_visualizer
    PORTED["depth_profile"] = depth_profile_solver
    PORTED["radial_profile"] = radial_profile_solver
    PORTED["inversion"] = inversion_quantifier

    def _scanning(cfg):
        params = {k: v for k, v in cfg.items()
                  if k not in ("makePlots", "saveFigures", "outputDir")}
        return scanning_beam_solver(params, cfg.get("outputDir"), save_plots=False)

    PORTED["scanning"] = _scanning


def bench_case(case: str, matlab_s: float) -> None:
    fx = load_fixture(case)
    solver = None
    for prefix, fn in PORTED.items():
        if case.startswith(prefix):
            solver = fn
            break
    if solver is None:
        print(f"  {case:28s}  (solver not ported yet, skipping)")
        return

    cfg = dict(fx["cfg"])
    cfg["makePlots"] = False
    cfg["saveFigures"] = False
    cfg["outputDir"] = os.path.join(HERE, "..", "outputs", "bench")

    import contextlib
    import io

    for attempt in ("cold", "warm"):
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            solver(cfg)
        elapsed = time.perf_counter() - t0
        if attempt == "warm":
            ratio = matlab_s / elapsed if elapsed > 0 else float("inf")
            print(f"  {case:28s}  MATLAB {matlab_s:8.2f} s   "
                  f"Python {elapsed:8.2f} s   ({ratio:5.1f}x)")


def main() -> None:
    _register()
    with open(os.path.join(HERE, "fixtures", "manifest.json")) as f:
        manifest = json.load(f)
    wanted = set(sys.argv[1:])
    print("Wall-time comparison (Python warm-run vs MATLAB fixture generation):")
    for entry in manifest["cases"]:
        if entry["status"] != "ok":
            continue
        if wanted and entry["name"] not in wanted:
            continue
        bench_case(entry["name"], entry["wallTime_s"])


if __name__ == "__main__":
    main()
