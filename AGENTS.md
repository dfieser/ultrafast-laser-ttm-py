# AGENTS.md

Guidance for AI coding agents working on this repository. Written to be tool
agnostic. Human contributors should read [CONTRIBUTING.md](CONTRIBUTING.md)
first; this file covers the constraints an agent is most likely to violate.

laserttm is a Python port of the Ultrafast Laser TTM Toolbox (MATLAB),
backing doi:10.1007/s11665-026-14738-6. Six two-temperature-model solvers for
ultrafast pulsed-laser heating of metals, published to PyPI as `laserttm`.
The MATLAB repository is the naming ancestor: config field names carry over
one to one, and that correspondence is a documented feature.

## Three invariants

**Numerics are frozen.** `tests/` assert solver output against MATLAB golden
fixtures in `validation/fixtures/` down to rtol 1e-6. Do not change
arithmetic, arithmetic order, loop structure, or accumulation order in
`kernels.py` or inside any solver's pulse loop. Refactors there must be
provable code moves. Run the full suite before and after.

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
| `schema.py` | The contract, both directions: every config key and every results key with unit, default, range and meaning. Standard library only, so discovery never pays the numba import. |
| `materials.py` | One `Material` record per metal, plus per-field overrides and the k(T) table choice. |
| `config.py` | MATLAB `getCfgField` semantics: missing, None and empty fall back to the default. |
| `physics.py` | The shared closed-form identities: two-bath energetics, the pulse deposit, derived pulse-train quantities. One home each, so a correction lands everywhere at once. |
| `kernels.py` | Numba-jitted numerics, line-faithful to MATLAB. Treat as frozen. |
| `reporting.py` | The report-file conventions: output directory, filename slug, caseTag prefix, banner header, XY table. The basenames tests assert are built here. |
| `runtools.py` | Solver registry and result serialization. |
| `cli.py`, `mcp_server.py` | The two machine-facing front ends. |
| `plotting.py`, `progress.py`, `units.py` | Figures, the progress window, display-unit selection. |
| six solver modules | One solver each. |

## Rules that keep getting broken

**Defaults live in the schema.** Never hardcode a default at a
`get_cfg_field` call site. A new config key needs a `ParamSpec` in
`schema.py` in the same commit. `tests/test_schema.py` pins every default
against a snapshot of the original source, which is what makes schema
changes provable rather than hopeful.

**Key spellings follow MATLAB even when the style is inconsistent.** `Pavg`,
`tau_FWHM`, `T0_C`, `f_rep` and `spotRadius` coexist by design. Units belong
in the schema, not in renamed keys. Any rename ships with an alias and a
justification in the commit message.

**Physics does not belong in `plotting.py`.** Anything computed there is
silently lost whenever `makePlots` is false, which is the default for both
the CLI and the MCP server. This has already caused one real bug.

**Shared identities live in `physics.py`, once.** The two-bath
equilibration, the depth deposit, and the derived laser quantities used to
exist as up to five inline copies, which is how a defect in one of them
cost four extra fixes. Never re-inline one; call the helper, and put any
new closed-form identity there in the same commit.

**All physical inputs are SI.** Display conversion happens only in
`units.py`. Open report files with an explicit `encoding`; the headers
contain an em dash. Use `os.path.join`, never a literal `/`.

## Discovering the API instead of reading source

Every solver describes itself. Prefer these over grepping for key names:

```bash
laserttm list                      # solvers, and which one answers which question
laserttm describe depth_profile    # every config and results key, with units
laserttm describe depth_profile --section results   # results keys alone
laserttm materials                 # material presets
laserttm validate cfg.json         # check a config without running it
laserttm run cfg.json --dry-run    # validate and report the cost
laserttm schema depth_profile      # JSON Schema, for typed consumers
```

The same surface exists in Python as `describe_solver`, `describe_results`,
`validate_config`, `estimate_run`, `json_schema` and `list_solvers`, and
over MCP as tools of those names. `docs/results-contract.md` is generated
from the same registry. If `describe` does not show a key, that is a schema
bug to fix, not a lookup to work around; a solver key without a
`ResultField` row fails the contract test.

## Commands

```bash
python -m pytest -q                        # fast suite
python -m pytest -q -m "slow or not slow"  # including the MATLAB fixtures
ruff check src tests examples validation
```

On Windows the development environment is PowerShell, and the interpreter is
`.venv\Scripts\python.exe`. The progress window disables itself when there is
no usable display, so batch and CI runs stay headless.

## Before touching a solver

1. Does this change a computed number? If yes, stop and reconsider.
2. Is it additive for every published name?
3. Does a new or changed config key need a `ParamSpec`?
4. Does a test-asserted output basename move?
5. Is the commit releasable on its own?

## Attribution

Do not add AI attribution of any kind to this repository: no co-author
trailers, no generated-by notes in commits, pull requests, code comments,
documentation or release notes.
