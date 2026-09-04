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


def _cmd_list(args: argparse.Namespace) -> int:
    from . import schema

    if getattr(args, "json", False):
        print(json.dumps(schema.list_solvers(), indent=2))
        return 0
    for row in schema.list_solvers():
        print(f"\n{row['id']}  ({row['nParams']} config keys)")
        print(f"  {row['summary']}")
        print(f"  Use when:     {row['whenToUse']}")
        print(f"  Not when:     {row['whenNotToUse']}")
    print("\nRun 'laserttm describe <solver>' for every key, unit and default.")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    from . import schema

    try:
        described = schema.describe_solver(args.solver, args.section)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    if args.json:
        print(json.dumps(described, indent=2, default=str))
        return 0

    print(f"{described['id']}: {described['title']}")
    print(f"\n{described['summary']}")
    print(f"\nUse when:  {described['whenToUse']}")
    print(f"Not when:  {described['whenNotToUse']}")

    params = described.get("params")
    if params:
        print(f"\nConfig keys ({len(params)}):")
        for name in sorted(params):
            spec = params[name]
            unit = f" [{spec['unit']}]" if spec.get("unit") else ""
            print(f"\n  {name}{unit}  default={spec['default']!r}")
            print(f"      {spec['summary']}")
            if spec.get("choices"):
                print(f"      choices: {', '.join(spec['choices'])}")
            if spec.get("range"):
                low, high = spec["range"]
                print(f"      valid range: {low:g} to {high:g}")
            if spec.get("notes"):
                print(f"      note: {spec['notes']}")

    result_fields = described.get("results")
    if result_fields:
        print(f"\nResults keys ({len(result_fields)}):")
        for spec in result_fields:
            unit = f" [{spec['unit']}]" if spec.get("unit") else ""
            flags = ""
            if spec.get("gatedBy"):
                flags += f"  (needs {spec['gatedBy']})"
            if spec.get("prefer"):
                flags += f"  (prefer {spec['prefer']})"
            dims = spec.get("dims")
            shape = f"  [{' x '.join(dims)}]" if dims else ""
            print(f"\n  {spec['name']}{unit}  {spec['kind']}{shape}{flags}")
            print(f"      {spec['summary']}")

    if described.get("files"):
        print("\nWrites: " + ", ".join(described["files"]))
    for label, example in (described.get("examples") or {}).items():
        print(f"\nExample ({label}): {json.dumps(example)}")
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    from . import schema

    try:
        print(json.dumps(schema.json_schema(args.solver), indent=2))
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    return 0


def _cmd_materials(args: argparse.Namespace) -> int:
    from . import schema

    rows = schema.materials_table()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        print(f"\n{row['key']}   melting point {row['T_melt_C']:g} degC")
        print(f"  gamma {row['gamma']:g} J m^-3 K^-2,  Cl {row['Cl']:.3g} "
              f"J m^-3 K^-1,  G {row['G']:.3g} W m^-3 K^-1")
        print(f"  conductivity {row['kTotal']:g} W m^-1 K^-1"
              + (f" (ke0 {row['ke0']:g} + kl {row['kl']:g})"
                 if row["ke0"] is not None else ""))
        if row["delta_opt"] is not None:
            print(f"  optical penetration depth {row['delta_opt'] * 1e9:.1f} nm")
        print(f"  solvers: {', '.join(row['solvers'])}")
    return 0


def _print_problems(report: dict) -> None:
    """Render a validate_config report the way a person reads it."""
    for kind, items in (("error", report["errors"]),
                        ("warning", report["warnings"])):
        for problem in items:
            print(f"  [{kind}] {problem['message']}", file=sys.stderr)
            print(f"          {problem['suggestion']}", file=sys.stderr)


def _print_derived(report: dict) -> None:
    """The pulse-train numbers a config implies, so a wrong Pavg and f_rep
    pair is visible before a run is spent on it."""
    derived = report.get("derived")
    est = report.get("estimate")
    if derived:
        print(f"  Pulse energy {derived['pulseEnergy_J']:.4g} J, peak fluence "
              f"{derived['peakFluence_J_cm2']:.4g} J/cm^2, absorbed "
              f"{derived['absorbedFluence_J_m2'] / 1e4:.4g} J/cm^2.")
    if est:
        print(f"  {est['nPulses']} pulses, roughly {est['estRuntime_s']:g} s "
              f"plus about {est['warmup_s']:g} s of kernel loading on the "
              "first call.")


def _cmd_validate(args: argparse.Namespace) -> int:
    from . import schema

    # utf-8-sig transparently strips a byte-order mark, which PowerShell
    # writes by default when a config is redirected to a file.
    with open(args.config, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        print("error: config file must contain a JSON object", file=sys.stderr)
        return 2
    solver_id = args.solver or cfg.pop("solver", None)
    cfg.pop("solver", None)
    if solver_id is None:
        print("error: no solver given (use --solver or a 'solver' key in "
              "the config)", file=sys.stderr)
        return 2

    try:
        report = schema.validate_config(solver_id, cfg)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["ok"] else 2

    if report["ok"]:
        print(f"Config is valid for {report['solver']}.")
    else:
        print(f"{len(report['errors'])} problem(s) with the config for "
              f"'{report['solver']}':", file=sys.stderr)
    _print_derived(report)
    _print_problems(report)
    return 0 if report["ok"] else 2


def _cmd_run(args: argparse.Namespace) -> int:
    from . import schema
    from .runtools import get_solver, save_results_npz, summarize_results

    # utf-8-sig transparently strips a byte-order mark, which PowerShell
    # writes by default when a config is redirected to a file.
    with open(args.config, encoding="utf-8-sig") as f:
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

    # Check the config before spending minutes on it. A typo would otherwise
    # be ignored and the run would silently proceed on defaults.
    try:
        report = schema.validate_config(solver_id, cfg)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    if not report["ok"]:
        print(f"{len(report['errors'])} problem(s) with the config for "
              f"'{report['solver']}':", file=sys.stderr)
        _print_problems(report)
        return 2
    for problem in report["warnings"]:
        print(f"  [warning] {problem['message']}", file=sys.stderr)

    if args.dry_run:
        print(f"Config is valid for {report['solver']}. Nothing was run.")
        _print_derived(report)
        return 0

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
            with open(args.out, "w", encoding="utf-8") as f:
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
    p_list.add_argument("--json", action="store_true", help="emit JSON")
    p_list.set_defaults(func=_cmd_list)

    p_desc = sub.add_parser(
        "describe",
        help="show a solver's config keys, units, defaults and ranges")
    p_desc.add_argument("solver", help="solver id (see 'laserttm list')")
    p_desc.add_argument("--section", default="all",
                        choices=["all", "inputs", "results", "files",
                                 "examples"],
                        help="limit the output to one section")
    p_desc.add_argument("--json", action="store_true", help="emit JSON")
    p_desc.set_defaults(func=_cmd_describe)

    p_schema = sub.add_parser(
        "schema", help="emit a JSON Schema for a solver's config")
    p_schema.add_argument("solver", help="solver id")
    p_schema.set_defaults(func=_cmd_schema)

    p_mat = sub.add_parser("materials",
                           help="list material presets and their properties")
    p_mat.add_argument("--json", action="store_true", help="emit JSON")
    p_mat.set_defaults(func=_cmd_materials)

    p_val = sub.add_parser(
        "validate", help="check a config without running anything")
    p_val.add_argument("config", help="path to the JSON config file")
    p_val.add_argument("--solver", help="solver id (overrides the config's "
                                        "'solver' key)")
    p_val.add_argument("--json", action="store_true", help="emit JSON")
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="run a solver from a JSON config file")
    p_run.add_argument("config", help="path to the JSON config file")
    p_run.add_argument("--solver", help="solver id (overrides the config's "
                                        "'solver' key)")
    p_run.add_argument("--dry-run", action="store_true",
                       help="validate and estimate the run, then stop")
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


def _tolerant_console() -> None:
    """Never let a narrow console encoding crash a listing.

    Windows consoles often run cp1252, which cannot encode every character
    in the schema text. Replacing the odd character beats a traceback.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _tolerant_console()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
