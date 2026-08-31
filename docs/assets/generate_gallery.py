"""Regenerate the README gallery figures.

Python port of the MATLAB repo's docs/assets/generate_gallery.m: runs the
four baseline solver configurations from examples/ (plots off), caches the
results under outputs/gallery_cache.npz, and renders the gallery images in
docs/assets/ in light and dark variants:

    fig_single_pulse[-dark].png       single-pulse electron-lattice dynamics
    fig_heat_accumulation[-dark].png  multi-pulse heat accumulation
    fig_scanning_map[-dark].png       scanning-beam peak-temperature map
    fig_radial_profile[-dark].png     residual radial temperature profile

Delete outputs/gallery_cache.npz to force the solver runs to repeat.

Note: the scanning-map colorbar here is in true degC (the MATLAB gallery
plotted the Kelvin-valued map under a degC label).
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(REPO_ROOT, "outputs", "gallery_cache.npz")


# ========================================================================
#  Solver runs (mirror the baseline examples, plots off)
# ========================================================================
def run_baseline_cases() -> dict:
    from laserttm import (
        depth_profile_solver,
        radial_profile_solver,
        scanning_beam_solver,
        surface_point_solver,
    )

    runs_dir = os.path.join(REPO_ROOT, "outputs", "gallery_cache_runs")

    def quiet(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kwargs)

    print("=== [1/5] Surface-point baseline (50 pulses) ===")
    cfg = {"material": "W", "Pavg": 10, "spotRadius": 100e-6,
           "f_rep": 5e6, "tau_FWHM": 500e-15, "absorbance": 0.55,
           "makePlots": False, "saveFigures": False}
    cfg["simDuration"] = 50 / cfg["f_rep"]
    cfg["outputDir"] = os.path.join(runs_dir, "surface_point")
    sp = quiet(surface_point_solver, cfg)

    print("=== [2/5] Surface-point long run (600 pulses) ===")
    cfg2 = dict(cfg)
    cfg2["simDuration"] = 600 / cfg["f_rep"]
    cfg2["outputDir"] = os.path.join(runs_dir, "surface_point_long")
    sp_l = quiet(surface_point_solver, cfg2)

    print("=== [3/5] Depth-profile baseline (100 pulses) ===")
    cfg_d = {"material": "W", "Pavg": 40, "spotRadius": 100e-6,
             "f_rep": 18e6, "tau_FWHM": 500e-15, "absorbance": 0.55,
             "makePlots": False, "saveFigures": False,
             "enableRadialProfile": False}
    cfg_d["simDuration"] = 100 / cfg_d["f_rep"]
    cfg_d["outputDir"] = os.path.join(runs_dir, "depth")
    dp = quiet(depth_profile_solver, cfg_d)

    print("=== [4/5] Radial-profile baseline (100 pulses) ===")
    cfg_r = {"material": "W", "Pavg": 40, "spotRadius": 100e-6,
             "f_rep": 18e6, "tau_FWHM": 500e-15, "absorbance": 0.55,
             "Nr": 80, "rMax_factor": 4, "radialSolveMode": "scale",
             "makePlots": False, "saveFigures": False}
    cfg_r["simDuration"] = 100 / cfg_r["f_rep"]
    cfg_r["outputDir"] = os.path.join(runs_dir, "radial")
    rp = quiet(radial_profile_solver, cfg_r)

    print("=== [5/5] Scanning-beam baseline (36000 pulses, several minutes) ===")
    params = {"material": "W", "gamma": 137.3, "Cl": 2.54e6,
              "G": 1.65e17, "kl": 174, "Pavg": 40, "spotRadius": 100e-6,
              "f_rep": 18e6, "tau_FWHM": 100e-15, "pulseProfile": "gaussian",
              "v_scan": 1.0, "scanLength": 2e-3, "absorbance": 0.55,
              "Leff": 100e-9, "T0_C": 25, "Nx": 120, "Ny": 60, "xPad": 3,
              "yExtent": 5, "depthProfile": "exponential", "dzTarget": 500e-9,
              "Ndiff": 100, "NadiPerGap": 10}
    sc = quiet(scanning_beam_solver, params, os.path.join(runs_dir, "scan"), False)

    return {
        "sp_time_s": np.asarray(sp["time_s"], float),
        "sp_Te_C": np.asarray(sp["Te_C"], float),
        "sp_Tl_C": np.asarray(sp["Tl_C"], float),
        "sp_Teq_C": np.asarray(sp["TeqVals_C"], float),
        "sp_Tresid_C": np.asarray(sp["TresidVals_C"], float),
        "sp_nPulses": sp["nPulses"],
        "sp_f_rep": cfg["f_rep"], "sp_Pavg": cfg["Pavg"],
        "spL_Teq_C": np.asarray(sp_l["TeqVals_C"], float),
        "spL_Tresid_C": np.asarray(sp_l["TresidVals_C"], float),
        "spL_nPulses": sp_l["nPulses"],
        "dp_TePeak_C": np.asarray(dp["TePeakPerPulse_C"], float),
        "dp_TlPeak_C": np.asarray(dp["TlPeakPerPulse_C"], float),
        "dp_base_C": np.asarray(dp["baseTempPerPulse_C"], float),
        "dp_invMax_K": np.asarray(dp["invMaxPerPulse_K"], float),
        "dp_nPulses": dp["nPulses"],
        "rp_r_um": np.asarray(rp["rGrid_um"], float),
        "rp_final_C": np.asarray(rp["finalRadialProfile_C"], float),
        "rp_spot_um": rp["spotRadius_um"], "rp_nPulses": rp["nPulses"],
        "sc_Tpeak_map": np.asarray(sc["Tpeak_map"], float),
        "sc_xGrid": np.asarray(sc["xGrid"], float),
        "sc_yGrid": np.asarray(sc["yGrid"], float),
        "sc_nPulses": sc["nPulses"],
    }


# ========================================================================
#  Styles
# ========================================================================
@dataclass
class Style:
    suffix: str
    surface: str
    ink_main: str
    ink_second: str
    ink_muted: str
    grid: str
    electron: str
    lattice: str
    ramp_stops: tuple


LIGHT = Style(
    suffix="", surface="#fcfcfb", ink_main="#0b0b0b", ink_second="#52514e",
    ink_muted="#898781", grid="#e1e0d9", electron="#eb6834", lattice="#2a78d6",
    ramp_stops=("#fdece1", "#fbd9c6", "#f8c3a6", "#f3ab85", "#ed9264",
                "#e57746", "#d95926", "#b84a1e", "#963a16", "#732c10"))

DARK = Style(
    suffix="-dark", surface="#1a1a19", ink_main="#ffffff", ink_second="#c3c2b7",
    ink_muted="#898781", grid="#2c2c2a", electron="#d95926", lattice="#3987e5",
    ramp_stops=("#241811", "#3b2315", "#54301a", "#6f3d1e", "#8c4a21",
                "#aa5723", "#c76627", "#e07a3c", "#f0965f", "#fcb488"))

_FONT = ["Segoe UI", "DejaVu Sans"]


def new_fig(st: Style, w_px: int, h_px: int):
    fig = plt.figure(figsize=(w_px / 100, h_px / 100), facecolor=st.surface)
    ax = fig.add_axes((0.10, 0.16, 0.86, 0.66))
    ax.set_facecolor(st.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st.ink_muted)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(direction="out", colors=st.ink_muted, labelsize=15)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(_FONT)
    ax.grid(True, color=st.grid, linewidth=0.75)
    ax.set_axisbelow(True)
    return fig, ax


def style_title(ax, st: Style, title: str, subtitle: str):
    ax.set_title(title, fontsize=19, fontweight="bold", color=st.ink_main,
                 fontfamily=_FONT, pad=34)
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=13.5, color=st.ink_second, fontfamily=_FONT)


def ramp_colormap(st: Style):
    return LinearSegmentedColormap.from_list("ramp" + st.suffix, st.ramp_stops, N=256)


def style_legend(ax, st: Style, loc: str):
    lg = ax.legend(loc=loc, fontsize=15, labelcolor=st.ink_main,
                   facecolor=st.surface, edgecolor=st.grid, framealpha=1.0)
    for t in lg.get_texts():
        t.set_fontfamily(_FONT)
    return lg


def _label(ax, which: str, text: str, st: Style, size: int):
    fn = ax.set_xlabel if which == "x" else ax.set_ylabel
    fn(text, fontsize=size, color=st.ink_second, fontfamily=_FONT)


def save(fig, out_path: str, st: Style):
    fig.savefig(out_path, dpi=200, facecolor=st.surface)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ========================================================================
#  Figure 1: single-pulse electron-lattice dynamics
# ========================================================================
def render_single_pulse(s: dict, st: Style, out_path: str):
    trep = 1.0 / float(s["sp_f_rep"])
    t = s["sp_time_s"]
    in_win = (t > 0) & (t <= trep)
    t_ps = t[in_win] * 1e12
    te_k = s["sp_Te_C"][in_win] + 273.15
    tl_k = s["sp_Tl_C"][in_win] + 273.15

    fig, ax = new_fig(st, 980, 470)
    ax.plot(t_ps, te_k, "-", color=st.electron, lw=2.75,
            label="Electron temperature  $T_e$")
    ax.plot(t_ps, tl_k, "-", color=st.lattice, lw=2.75,
            label="Lattice temperature  $T_l$")
    ax.set_xscale("log")
    ax.set_xlim(0.05, trep * 1e12)
    ax.set_ylim(0, 3050)
    ax.set_xticks([0.1, 1, 10, 100, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["0.1 ps", "1 ps", "10 ps", "0.1 ns",
                        "1 ns", "10 ns", "100 ns"])
    ax.minorticks_off()
    _label(ax, "x", "Time within one pulse period", st, 17)
    _label(ax, "y", "Surface temperature (K)", st, 17)
    style_title(ax, st, "Single-pulse electron-lattice dynamics in tungsten",
                "surface_point_solver  ·  10 W  ·  5 MHz  ·  500 fs  ·  100 μm spot")
    style_legend(ax, st, "upper right")
    save(fig, out_path, st)


# ========================================================================
#  Figure 2: multi-pulse heat accumulation
# ========================================================================
def render_accumulation(s: dict, st: Style, out_path: str):
    n = int(s["spL_nPulses"])
    p = np.arange(1, n + 1)
    fig, ax = new_fig(st, 980, 470)
    ax.plot(p, s["spL_Teq_C"], "-", color=st.electron, lw=2.75,
            label="Equilibrated after each pulse  $T_{eq}$")
    ax.plot(p, s["spL_Tresid_C"], "-", color=st.lattice, lw=2.75,
            label="Residual before next pulse  $T_{resid}$")
    ax.set_xlim(0, n)
    ax.set_ylim(0, float(np.max(s["spL_Teq_C"])) * 1.18)
    _label(ax, "x", "Pulse number", st, 17)
    _label(ax, "y", "Surface temperature (°C)", st, 17)
    style_title(ax, st, "Multi-pulse heat accumulation in tungsten",
                "surface_point_solver  ·  600 pulses  ·  10 W  ·  5 MHz  ·  "
                "500 fs  ·  100 μm spot")
    style_legend(ax, st, "upper left")
    save(fig, out_path, st)


# ========================================================================
#  Figure 3: scanning-beam peak-temperature map
# ========================================================================
def render_scanning_map(s: dict, st: Style, out_path: str):
    x_mm = s["sc_xGrid"] * 1e3
    y_mm = s["sc_yGrid"] * 1e3
    t_c = s["sc_Tpeak_map"] - 273.15

    fig = plt.figure(figsize=(7.0, 4.3), facecolor=st.surface)
    ax = fig.add_axes((0.13, 0.30, 0.84, 0.52))
    ax.set_facecolor(st.surface)
    im = ax.imshow(t_c, extent=(x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]),
                   origin="lower", aspect="auto", cmap=ramp_colormap(st),
                   interpolation="bilinear")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_color(st.ink_muted)
        ax.spines[side].set_linewidth(0.75)
    ax.tick_params(direction="out", colors=st.ink_muted, labelsize=13)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(_FONT)
    _label(ax, "x", "Scan direction x (mm)", st, 16)
    _label(ax, "y", "y (mm)", st, 16)
    style_title(ax, st, "Scanning-beam peak temperature",
                "scanning_beam_solver  ·  40 W  ·  18 MHz  ·  1 m/s")

    cax = fig.add_axes((0.13, 0.115, 0.84, 0.045))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Peak surface temperature (°C)", fontsize=15,
                 color=st.ink_second, fontfamily=_FONT)
    cb.ax.tick_params(direction="out", colors=st.ink_muted, labelsize=13)
    for lab in cb.ax.get_xticklabels():
        lab.set_fontfamily(_FONT)
    cb.outline.set_edgecolor(st.grid)
    save(fig, out_path, st)


# ========================================================================
#  Figure 4: residual radial temperature profile
# ========================================================================
def render_radial_profile(s: dict, st: Style, out_path: str):
    r = s["rp_r_um"]
    t_c = s["rp_final_C"]
    spot = float(s["rp_spot_um"])

    fig, ax = new_fig(st, 700, 430)
    ax.fill_between(r, t_c, 0, color=st.electron, alpha=0.10, lw=0)
    ax.plot(r, t_c, "-", color=st.electron, lw=3)
    ax.set_xlim(0, float(np.max(r)))
    t_max = float(np.max(t_c))
    ax.set_ylim(0, t_max * 1.22)
    ax.axvline(spot, ls="--", color=st.ink_muted, lw=1.5)
    ax.text(spot + float(np.max(r)) * 0.02, t_max * 1.1,
            f"spot radius $w_0$ = {spot:.0f} μm",
            fontsize=14, color=st.ink_second, fontfamily=_FONT)
    _label(ax, "x", "Radial distance r (μm)", st, 16)
    _label(ax, "y", "Residual temperature (°C)", st, 16)
    style_title(ax, st, "Residual radial temperature profile",
                "radial_profile_solver  ·  100 pulses  ·  40 W  ·  18 MHz")
    ax.tick_params(labelsize=14)
    save(fig, out_path, st)


def main():
    if os.path.exists(CACHE_PATH):
        print(f"Using cached solver results: {CACHE_PATH}")
        with np.load(CACHE_PATH) as z:
            s = {k: z[k] for k in z.files}
    else:
        s = run_baseline_cases()
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH, **s)
        print(f"Cached solver results: {CACHE_PATH}")

    for st in (LIGHT, DARK):
        render_single_pulse(s, st, os.path.join(ASSETS_DIR, f"fig_single_pulse{st.suffix}.png"))
        render_accumulation(s, st, os.path.join(ASSETS_DIR, f"fig_heat_accumulation{st.suffix}.png"))
        render_scanning_map(s, st, os.path.join(ASSETS_DIR, f"fig_scanning_map{st.suffix}.png"))
        render_radial_profile(s, st, os.path.join(ASSETS_DIR, f"fig_radial_profile{st.suffix}.png"))
    print(f"Gallery figures written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
