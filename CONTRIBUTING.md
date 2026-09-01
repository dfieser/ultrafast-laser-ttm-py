# Contributing

Thanks for your interest in improving this repository.

## Good contributions

- bug fixes tied to a clear issue
- portability improvements
- example and documentation clarification
- small solver-interface improvements that preserve current behavior
- validation additions that stay generic rather than manuscript-specific

## Before opening a larger change

- prefer starting from the canonical examples in `examples/`
- keep reusable solver logic in `src/laserttm/`
- keep generated outputs out of version control
- preserve the shared result-contract fields where practical
- solver behavior is pinned by the MATLAB golden fixtures in
  `validation/fixtures/`; a change that shifts results outside the stated
  test tolerances needs a strong physical justification

## Suggested workflow

1. make a focused change
2. run `python -m pytest` (add `-m slow` for the long scanning baseline)
   and `ruff check src tests examples validation`
3. update `README.md` or `docs/` if user-facing behavior changed
4. keep pull requests small enough that solver behavior changes are easy to review

## Using an AI coding agent

[AGENTS.md](AGENTS.md) holds the constraints an agent needs before changing
anything here, chiefly that solver numerics are pinned to the MATLAB golden
fixtures and that the published interface is additive only. Most agent tools
read that file automatically. Please do not add AI attribution to commits or
pull requests.

## Scope guidance

This repository stays centered on reusable pulsed-laser thermal modeling workflows. Changes that mainly serve one historical study, or one manuscript-specific validation campaign, generally belong outside the public-facing copy.
