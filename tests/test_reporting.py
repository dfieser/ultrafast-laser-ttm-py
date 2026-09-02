"""reporting.py must reproduce the exact text the inline copies wrote.

The end-to-end guarantee lives in the solver tests, which assert the
output basenames, and in the report round-trip test. These pin the
helpers directly so a formatting change fails here first, with a
readable diff.
"""

import io

from laserttm import reporting


def test_filename_slug_matches_the_inline_form():
    # 5 MHz, 500 fs, 10 W, 100 um: the display-unit slug with '.' as 'p'.
    slug = reporting.filename_slug(5e6, 500e-15, 10.0, 100e-6)
    assert slug == "5_MHz_500_fs_10_W_100_um"


def test_filename_slug_spells_decimal_points_as_p():
    slug = reporting.filename_slug(9.25e6, 512e-15, 7.5, 82.5e-6)
    assert "." not in slug
    assert slug == "9p25_MHz_512_fs_7p5_W_82p5_um"


def test_apply_case_tag():
    assert reporting.apply_case_tag({}, "TTM_x.txt") == "TTM_x.txt"
    assert (reporting.apply_case_tag({"caseTag": "run 1"}, "TTM_x.txt")
            == "run_1__TTM_x.txt")


def test_write_header_layout():
    buf = io.StringIO()
    reporting.write_header(buf, "Title — Output", "  Mode: scale")
    lines = buf.getvalue().split("\n")
    assert lines[0] == "=" * 60
    assert lines[1] == "  Title — Output"
    assert lines[2].startswith("  Generated: ")
    assert lines[3] == "  Mode: scale"
    assert lines[4] == "=" * 60
    assert lines[5] == ""  # the blank line after the banner


def test_write_xy_table_formats():
    buf = io.StringIO()
    reporting.write_xy_table(
        buf, "  XY Data: Time (s) | Te (deg C) | Tl (deg C)",
        ("Time_s", "Te_degC", "Tl_degC"),
        [(1.5e-9, 25.0, 24.5)])
    lines = buf.getvalue().split("\n")
    assert lines[0] == "=" * 60
    assert lines[1] == "  XY Data: Time (s) | Te (deg C) | Tl (deg C)"
    assert lines[2] == "=" * 60
    assert lines[3] == f"{'Time_s':>20}  {'Te_degC':>16}  {'Tl_degC':>16}"
    assert lines[4] == f"{1.5e-9:20.12e}  {25.0:16.6f}  {24.5:16.6f}"


def test_resolve_output_dir_creates_and_defaults(tmp_path):
    target = tmp_path / "nested" / "out"
    assert reporting.resolve_output_dir({"outputDir": str(target)}) == str(target)
    assert target.is_dir()
