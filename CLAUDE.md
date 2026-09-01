# laserttm

Python port of the Ultrafast Laser TTM Toolbox (MATLAB), backing
doi:10.1007/s11665-026-14738-6. Six two-temperature-model solvers for
ultrafast pulsed-laser heating of metals. Published to PyPI as `laserttm`.
The MATLAB repository is the naming ancestor: config field names carry over
one to one, and that correspondence is a documented feature.

## Three invariants

**Numerics are frozen.** `tests/` assert solver output against MATLAB golden
fixtures in `validation/fixtures/` down to rtol 1e-6. Do not change
arithmetic, arithmetic order, loop structure, or accumulation order in
`kernels.py` or inside any solver's pulse loop. Refactors here must be
provable code moves. Run the full suite before and after:
`pytest -m "slow or not slow"`.

**The public API is additive only.** The six entry points, the config-dict
convention, and every existing results key are consumed by PyPI users, the
CLI, the MCP server, `docs/` and `examples/`. Add keys and deprecation
aliases. Never rename or remove.

**`main` auto-publishes to PyPI on every push.** Every commit must be
individually releasable: tests green, no half-finished migration on `main`.

## Output basenames asserted by tests

Do not change these strings: `TTM_` (surface_point), `TTM_Radial_Result_`
(radial_profile), `TTMmov_` (scanning_beam), `Inversion_Analysis_`
(inversion_quantifier). `TTM_1D_Result_` and `TTM1D_` are not asserted but
are user visible.

## Where things live

| Module | Owns |
| --- | --- |
| `schema.py` | The input contract. Every config key with unit, default, range and meaning. Standard library only, so discovery never pays the numba import. |
| `materials.py` | One `Material` record per metal, plus per-field overrides and the k(T) table choice. |
| `config.py` | MATLAB `getCfgField` semantics: missing, None and empty fall back to the default. |
| `kernels.py` | Numba-jitted numerics, line-faithful to MATLAB. Treat as frozen. |
| `runtools.py` | Solver registry and result serialization. |
| `cli.py`, `mcp_server.py` | The two machine-facing front ends. |
| `plotting.py`, `progress.py`, `units.py` | Figures, the progress window, display-unit selection. |
| six solver modules | One solver each. |

## Rules that keep getting broken

**Defaults live in the schema.** Never hardcode a default at a
`get_cfg_field` call site once a key is in `schema.py`. A new config key
needs a `ParamSpec` in the same commit. `tests/test_schema.py` pins every
default against a snapshot of the original source, which is what makes
schema changes provable rather than hopeful.

**Key spellings follow MATLAB even when the style is inconsistent.** `Pavg`,
`tau_FWHM`, `T0_C`, `f_rep` and `spotRadius` coexist by design. Units belong
in the schema, not in renamed keys. Any rename ships with an alias and a
justification in the commit message.

**Physics does not belong in `plotting.py`.** Anything computed there is
silently lost whenever `makePlots` is false, which is the default for both
the CLI and the MCP server.

**All physical inputs are SI.** Display conversion happens only in
`units.py`. Open report files with `encoding="utf-8"`; the headers contain
an em dash. Use `os.path.join`, never a literal `/`.

## Agent self-service

Before writing a config, run `laserttm describe <solver>`. Before running
anything long, `laserttm run cfg.json --dry-run`, or `validate_config`
through MCP. Do not read solver source to learn key names: if `describe`
does not show a key, that is a schema bug to fix, not a lookup to work
around.

## Commands

```
.venv\Scripts\python.exe -m pytest -q                       # fast suite
.venv\Scripts\python.exe -m pytest -q -m "slow or not slow"  # with fixtures
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\laserttm.exe describe depth_profile
.venv\Scripts\laserttm.exe validate cfg.json
```

Windows and PowerShell is the development environment. The tkinter progress
window disables itself when there is no usable display.

## Before touching a solver

1. Does this change a computed number? If yes, stop and reconsider.
2. Is it additive for every published name?
3. Does a new or changed config key need a `ParamSpec`?
4. Does a test-asserted output basename move?
5. Is the commit releasable on its own?

## Attribution

Never add Co-Authored-By trailers or any AI credit to commits, pull
requests, code comments, documentation or release notes.
