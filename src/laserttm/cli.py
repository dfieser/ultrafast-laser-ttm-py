"""Command-line interface: run any laserttm solver from a JSON config file.

Examples
--------
List the available solvers::

    laserttm list

Run a config (the file supplies the solver id and the cfg fields)::

    laserttm run study.json --out results.json

where ``study.json`` looks like::

    {
      "solver": "radial_profile",
      "material": "W",
      "Pavg": 70,
      "f_rep": 40e6,
      "spotRadius": 150e-6,
      "simDuration": 2.5e-5,
      "makePlots": false
    }

Every key except ``solver`` is passed straight through as the solver's cfg
dict, so the field names match the library and the MATLAB reference
one-to-one. Plots are off by default in CLI runs (batch-friendly); pass
``--plots`` or set ``makePlots``/``saveFigures`` in the config to override.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_list(_args: argparse.Namespace) -> int:
    from .runtools import SOLVER_DESCRIPTIONS

    width = max(len(s) for s in SOLVER_DESCRIPTIONS)
    for sid, desc in SOLVER_DESCRIPTIONS.items():
        print(f"  {sid:<{width}}  {desc}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .runtools import get_solver, save_results_npz, summarize_results

    with open(args.config) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        print("error: config file must contain a JSON object", file=sys.stderr)
        return 2

    cfg_solver = cfg.pop("solver", None)
    solver_id = args.solver or cfg_solver
    if solver_id is None:
        print("error: no solver given (use --solver or a 'solver' key in "
              "the config)", file=sys.stderr)
        return 2

    # Batch-friendly default: no figure windows unless asked for.
    if args.plots:
        cfg["makePlots"] = True
    elif "makePlots" not in cfg and "saveFigures" not in cfg:
        cfg["makePlots"] = False

    solver = get_solver(solver_id)
    results = solver(cfg)

    if args.out:
        if args.out.endswith(".npz"):
            save_results_npz(results, args.out)
        else:
            with open(args.out, "w") as f:
                json.dump(summarize_results(results, max_array=args.max_array),
                          f, indent=2)
        print(f"Results written to: {args.out}")
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from . import __version__

    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laserttm",
        description="Two-temperature model solvers for ultrafast pulsed-laser "
                    "heating of metals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available solvers")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="run a solver from a JSON config file")
    p_run.add_argument("config", help="path to the JSON config file")
    p_run.add_argument("--solver", help="solver id (overrides the config's "
                                        "'solver' key)")
    p_run.add_argument("--out", help="write results to this path "
                                     "(.json summary or .npz arrays)")
    p_run.add_argument("--max-array", type=int, default=10000,
                       help="largest array length written in full to JSON "
                            "output (default: 10000)")
    p_run.add_argument("--plots", action="store_true",
                       help="show matplotlib figures (off by default)")
    p_run.set_defaults(func=_cmd_run)

    p_version = sub.add_parser("version", help="print the package version")
    p_version.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
