"""Reports must be UTF-8 regardless of the machine's locale.

Every solver's report header contains an em dash. Written with the default
locale encoding, that raises UnicodeEncodeError on a cp1252 Windows box and
takes the whole run with it after all the compute is done.
"""

import io
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "laserttm"


def test_no_text_write_relies_on_the_locale_encoding():
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'open\([^)]*"w"', line) and "encoding=" not in line:
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, (
        "these writes would use the machine locale: " + ", ".join(offenders))


def test_sources_are_clean_utf8_without_a_bom():
    """A BOM or mojibake here means an editor or shell re-encoded the file."""
    for path in sorted(SRC.glob("*.py")):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name} has a UTF-8 BOM"
        text = raw.decode("utf-8")
        assert "â€" not in text, f"{path.name} contains mojibake"


@pytest.mark.parametrize("solver_id", ["surface_point", "radial_profile"])
def test_report_round_trips_as_utf8(tmp_path, solver_id, monkeypatch):
    from laserttm.runtools import get_solver

    monkeypatch.setattr("matplotlib.pyplot.show", lambda *a, **k: None,
                        raising=False)
    cfg = {"f_rep": 5e6, "simDuration": 2 / 5e6, "makePlots": False,
           "outputDir": str(tmp_path)}
    if solver_id == "radial_profile":
        cfg["Nr"] = 8

    buf = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buf):
        results = get_solver(solver_id)(cfg)

    raw = pathlib.Path(results["outputFile"]).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")          # must not raise
    assert "—" in text             # the em dash survived
    assert "â€" not in text   # and was not double-encoded
