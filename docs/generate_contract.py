"""Generate docs/results-contract.md from the schema's results registry.

Run from the repository root:

    python docs/generate_contract.py

tests/test_results_envelope.py asserts the file matches the registry, so
regenerate and commit it in the same change as any registry edit.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from laserttm import schema

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results-contract.md")


def _meaning(field: dict) -> str:
    text = field["summary"]
    if field.get("gatedBy"):
        text += f" Present only when `{field['gatedBy']}` is enabled."
    if field.get("prefer"):
        text += f" Kept for compatibility, prefer `{field['prefer']}`."
    return text


def _table(fields) -> list[str]:
    lines = ["| Key | Type | Unit | Meaning |", "| --- | --- | --- | --- |"]
    for f in fields:
        d = f.as_dict()
        unit = d["unit"] if d["unit"] else ""
        lines.append(f"| `{d['name']}` | {d['kind']} | {unit} "
                     f"| {_meaning(d)} |")
    return lines


def render() -> str:
    lines = [
        "# Results contract",
        "",
        "Every key a solver returns, in every release. Generated from the",
        "registry in `src/laserttm/schema.py` by",
        "`python docs/generate_contract.py`, and checked against the live",
        "solvers by the test suite, so it cannot drift from the code. Do",
        "not edit by hand.",
        "",
        "The contract is additive: within contract version v1 no key is",
        "renamed, removed, or changed in meaning, and new keys may appear",
        "in any release. The same information is available at runtime:",
        "",
        "```python",
        "import laserttm",
        "laserttm.describe_results(\"depth_profile\")",
        "```",
        "",
        "```bash",
        "laserttm describe depth_profile --section results",
        "```",
        "",
        "and over MCP through the `describe_solver` tool.",
        "",
        "All temperatures are Celsius except temperature differences,",
        "which are Kelvin. A NaN marks a quantity that did not occur in",
        "the run, such as an inversion that never happened.",
        "",
        "## Shared envelope",
        "",
        "These keys appear in every solver's results.",
        "",
    ]
    lines.extend(_table(schema.RESULT_ENVELOPE))
    for sid in schema.SOLVER_IDS:
        sch = schema.solver_schema(sid)
        lines.extend(["", f"## {sid}", "", f"{sch.title}.", ""])
        lines.extend(_table(schema.RESULTS[sid]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(render())
    print(f"Wrote {OUT_PATH}")
