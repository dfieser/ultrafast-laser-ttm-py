"""Report-file conventions shared by every solver.

Each solver writes one human-readable text report. The pieces that are
the same convention in all of them live here: resolving the output
directory, the filename slug the tests assert basenames against, the
``caseTag`` prefix, the banner header with its wall-clock line, and the
three-column XY time-series table. The body of each report is genuinely
solver-specific and stays with its solver.

Every helper reproduces byte-for-byte the text the inline copies wrote,
because report files are user-visible output that people diff and parse.
"""

import os
from datetime import datetime

from .config import get_cfg_field, safe_tag
from .units import smart_freq, smart_length, smart_time

_RULE = "=" * 60 + "\n"

NO_HISTORY_NOTE = ("  storeHistory=False: the per-sample time series was "
                   "not retained,\n  so no XY table is written.\n")


def resolve_output_dir(cfg: dict) -> str:
    """``outputDir`` from the config, defaulting to ``./outputs``, created."""
    default_out = os.path.join(os.getcwd(), "outputs")
    output_dir = get_cfg_field(cfg, "outputDir", default_out)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def filename_slug(f_rep: float, tau_fwhm: float, pavg: float,
                  spot_radius: float) -> str:
    """The ``{rate}_{width}_{power}_{spot}`` middle of every report
    filename, in display units with ``.`` spelled ``p`` so the name
    survives any filesystem."""
    frep_v, frep_u = smart_freq(f_rep)
    tau_v, tau_u = smart_time(tau_fwhm)
    spot_v, spot_u = smart_length(spot_radius)
    freq_str = (f"{frep_v:.4g}_{frep_u}").replace(".", "p")
    pulse_str = (f"{tau_v:.4g}_{tau_u}").replace(".", "p")
    power_str = (f"{pavg:.4g}_W").replace(".", "p")
    spot_str = (f"{spot_v:.4g}_{spot_u}").replace(".", "p")
    return f"{freq_str}_{pulse_str}_{power_str}_{spot_str}"


def case_tag(cfg: dict) -> str:
    """The sanitized ``caseTag``, or an empty string when unset."""
    return safe_tag(get_cfg_field(cfg, "caseTag", ""))


def apply_case_tag(cfg: dict, filename: str) -> str:
    """Prefix ``caseTag__`` when the config carries a case tag."""
    tag = case_tag(cfg)
    if tag:
        return f"{tag}__{filename}"
    return filename


def write_header(fid, title: str, *extra_lines: str) -> None:
    """The report banner: rule, title, timestamp, extras, rule."""
    fid.write(_RULE)
    fid.write(f"  {title}\n")
    # Local wall-clock on purpose, matching the MATLAB reference output
    fid.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")  # noqa: DTZ005
    for line in extra_lines:
        fid.write(line + "\n")
    fid.write(_RULE + "\n")


def write_xy_table(fid, heading: str, columns: tuple[str, str, str],
                   rows) -> None:
    """The banner-framed time-series table: one ``(t, a, b)`` row per
    sample, time in ``%20.12e`` and both temperatures in ``%16.6f``."""
    fid.write(_RULE)
    fid.write(heading + "\n")
    fid.write(_RULE)
    fid.write(f"{columns[0]:>20}  {columns[1]:>16}  {columns[2]:>16}\n")
    fid.writelines(f"{t:20.12e}  {a:16.6f}  {b:16.6f}\n" for t, a, b in rows)
