"""Temperature-inversion quantifier.

Python port of ``src/Inversion_Quantifier.m``: runs (or reuses) the depth
solver, then quantifies the electron–lattice surface temperature inversion
(Tl > Te) across pulses — per-pulse magnitude, timing, and duration, plus
summary statistics and their trend with heat accumulation. Config fields and
the v1 result contract match the MATLAB solver field-for-field.
"""

from __future__ import annotations

import os

import numpy as np

from .config import get_cfg_field
from .depth_profile import depth_profile_solver
from .reporting import (
    apply_case_tag,
    filename_slug,
    resolve_output_dir,
    write_header,
)
from .units import smart_energy, smart_freq, smart_length, smart_time


def _resolve_out_path(cfg, f_rep, tau_fwhm, pavg, spot_radius, n_pulses):
    """Output directory and report path, shared by both result branches."""
    output_dir = resolve_output_dir(cfg)
    out_filename = apply_case_tag(cfg, (
        f"Inversion_Analysis_{filename_slug(f_rep, tau_fwhm, pavg, spot_radius)}_"
        f"{n_pulses}p.txt"))
    return output_dir, os.path.join(output_dir, out_filename)


def inversion_quantifier(cfg: dict | None = None) -> dict:
    """Run the inversion quantifier. Returns the v1 results dict."""
    if cfg is None:
        cfg = {}

    make_plots = get_cfg_field(cfg, "makePlots", True)
    save_figures = get_cfg_field(cfg, "saveFigures", False)
    if save_figures:
        make_plots = True

    print("=== Temperature Inversion Quantifier ===")

    # ==================  Run or load depth solver  ==========================
    depth_results = get_cfg_field(cfg, "depthResults", None)
    if depth_results is not None:
        print("  Using pre-computed Depth Profile Solver results.")
        dr = depth_results
    else:
        print("  Running Depth Profile Solver...")
        depth_cfg = dict(cfg)
        depth_cfg["makePlots"] = False
        depth_cfg["saveFigures"] = False
        dr = depth_profile_solver(depth_cfg)
        print("  Depth solver complete.")

    # ==================  Extract data  ======================================
    n_pulses = int(dr["nPulses"])
    material = dr["material"]

    te_peak = np.asarray(dr["TePeakPerPulse_C"], dtype=float)
    tl_peak = np.asarray(dr["TlPeakPerPulse_C"], dtype=float)
    teq = np.asarray(dr["TeqVals_C"], dtype=float)
    tresid = np.asarray(dr["TresidVals_C"], dtype=float)
    tbase = np.asarray(dr["baseTempPerPulse_C"], dtype=float)

    inv_max = np.asarray(dr["invMaxPerPulse_K"], dtype=float)
    t_max_inv = np.asarray(dr["tMaxInvPerPulse_s"], dtype=float)
    t_onset = np.asarray(dr["tInvOnsetPerPulse_s"], dtype=float)
    inv_dur = np.asarray(dr["invDurationPerPulse_s"], dtype=float)
    te_at_inv = np.asarray(dr["Te_atMaxInvPerPulse_C"], dtype=float)
    tl_at_inv = np.asarray(dr["Tl_atMaxInvPerPulse_C"], dtype=float)

    f_rep = dr["f_rep"]
    pavg = dr["Pavg"]
    tau_fwhm = dr["tau_FWHM"]
    spot_radius = dr["spotRadius"]
    absorbance = dr["absorbance"]
    f_peak = dr["F_peak"]
    gamma_mat = dr["gamma"]
    cl = dr["Cl"]
    g_ep = dr["G"]
    ke0 = dr["ke0"]
    kl = dr["kl"]
    alpha_opt = dr["alpha_opt"]

    # ==================  Inversion analysis  ================================
    print("\n--- Inversion Analysis ---")

    inv_threshold = 0.5  # [K]
    has_inversion = inv_max > inv_threshold
    n_inv_pulses = int(np.sum(has_inversion))
    inv_pulse_idx = np.flatnonzero(has_inversion) + 1  # 1-based, as in MATLAB

    print(f"  Pulses with inversion (Tl-Te > {inv_threshold:.1f} K): "
          f"{n_inv_pulses} / {n_pulses}")

    if n_inv_pulses == 0:
        print("  No significant inversion detected in any pulse.")
        print("  This may indicate:")
        print("    - Pulse energy too low")
        print("    - Spatial grid too coarse to resolve inversion")
        print("    - Pulse width too long (inversion is a femtosecond phenomenon)")
        # An absent inversion is still a result: write the report and return
        # real paths, so the contract's outputFile always exists.
        output_dir, out_path = _resolve_out_path(
            cfg, f_rep, tau_fwhm, pavg, spot_radius, n_pulses)
        with open(out_path, "w", encoding="utf-8") as fid:
            write_header(fid, "Temperature Inversion Quantifier — Output")
            fid.write(f"--- Material: {str(material).upper()} ---\n\n")
            fid.write(f"  Pulses analyzed:  {n_pulses}\n")
            fid.write(f"  Pulses with inversion (Tl-Te > "
                      f"{inv_threshold:.1f} K): 0\n\n")
            fid.write("  No significant inversion detected in any pulse.\n")
            fid.write("  Possible causes: pulse energy too low, spatial grid\n")
            fid.write("  too coarse to resolve it, or pulse width too long\n")
            fid.write("  for this femtosecond-scale phenomenon.\n")
        print(f"  Output written to: {out_path}")
        return _build_results(
            cfg, dr, n_pulses, material, inv_max, te_peak, tl_peak,
            tbase, teq, tresid, t_max_inv, t_onset, inv_dur, te_at_inv, tl_at_inv,
            0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            out_path, output_dir)

    inv_max_valid = inv_max[has_inversion]
    tbase_valid = tbase[has_inversion]
    t_max_inv_valid = t_max_inv[has_inversion]
    inv_dur_valid = inv_dur[has_inversion]

    mean_inv = float(np.mean(inv_max_valid))
    max_inv_all = float(np.max(inv_max_valid))
    min_inv = float(np.min(inv_max_valid))
    std_inv = float(np.std(inv_max_valid, ddof=1)) if n_inv_pulses > 1 else 0.0

    first_inv = inv_max_valid[0]
    last_inv = inv_max_valid[-1]

    te_excursion = te_peak[has_inversion] - tbase[has_inversion]
    inv_fraction = inv_max_valid / np.maximum(te_excursion, 1.0)

    if n_inv_pulses > 2:
        p_coeff = np.polyfit(inv_pulse_idx.astype(float), inv_max_valid, 1)
        inv_slope = float(p_coeff[0])
        inv_trend = inv_slope * n_pulses
        r_corr = float(np.corrcoef(tbase_valid, inv_max_valid)[0, 1])
    else:
        inv_slope = 0.0
        inv_trend = 0.0
        r_corr = np.nan

    print("\n  === Inversion Summary ===")
    print(f"  Mean inversion:    {mean_inv:.2f} K")
    max_pulse = int(inv_pulse_idx[int(np.argmax(inv_max_valid))])
    print(f"  Max inversion:     {max_inv_all:.2f} K  (pulse {max_pulse})")
    print(f"  Min inversion:     {min_inv:.2f} K")
    print(f"  Std dev:           {std_inv:.2f} K")
    print(f"  First pulse inv:   {first_inv:.2f} K  (Tbase = {tbase_valid[0]:.1f} C)")
    print(f"  Last pulse inv:    {last_inv:.2f} K  (Tbase = {tbase_valid[-1]:.1f} C)")

    mx_tim_v, mx_tim_u = smart_time(
        float(np.mean(t_max_inv_valid[~np.isnan(t_max_inv_valid)])))
    mx_dur_v, mx_dur_u = smart_time(
        float(np.mean(inv_dur_valid[inv_dur_valid > 0])))
    print(f"  Mean inv timing:   {mx_tim_v:.3g} {mx_tim_u} after pulse center")
    print(f"  Mean inv duration: {mx_dur_v:.3g} {mx_dur_u}")

    if n_inv_pulses > 2:
        print(f"  Trend slope:       {inv_slope:.4g} K/pulse")
        if abs(inv_trend) > 0.1:
            direction = "INCREASING" if inv_trend > 0 else "DECREASING"
            print(f"  Trend:             {direction} "
                  f"({inv_trend:.1f} K over {n_pulses} pulses)")
        else:
            print("  Trend:             STABLE (< 0.1 K change)")
        print(f"  Corr(Tbase, inv):  {r_corr:.3f}")
    print(f"  Mean inv fraction: {100 * np.mean(inv_fraction):.1f}% of Te excursion")

    # ==================  Output file  =======================================
    output_dir, out_path = _resolve_out_path(
        cfg, f_rep, tau_fwhm, pavg, spot_radius, n_pulses)
    frep_v, frep_u = smart_freq(f_rep)
    tau_v, tau_u = smart_time(tau_fwhm)
    ep_v, ep_u = smart_energy(pavg / f_rep)
    spot_v, spot_u = smart_length(spot_radius)

    with open(out_path, "w", encoding="utf-8") as fid:
        write_header(fid, "Temperature Inversion Quantifier — Output")
        fid.write(f"--- Material: {str(material).upper()} ---\n")
        fid.write(f"  gamma  = {gamma_mat:.2f}  J m^-3 K^-2\n")
        fid.write(f"  Cl     = {cl:.4e}  J m^-3 K^-1\n")
        fid.write(f"  G      = {g_ep:.4e}  W m^-3 K^-1\n")
        fid.write(f"  ke0    = {ke0:.1f}  W m^-1 K^-1\n")
        fid.write(f"  kl     = {kl:.1f}  W m^-1 K^-1\n")
        fid.write(f"  alpha  = {alpha_opt:.4e}  m^-1  "
                  f"(skin depth {1e9 / alpha_opt:.1f} nm)\n")
        fid.write("\n--- Laser ---\n")
        fid.write(f"  Average Power:    {pavg:.4g} W\n")
        fid.write(f"  Rep Rate:         {frep_v:.4g} {frep_u}\n")
        fid.write(f"  Pulse Energy:     {ep_v:.4g} {ep_u}\n")
        fid.write(f"  Pulse Width:      {tau_v:.4g} {tau_u}\n")
        fid.write(f"  Spot Radius:      {spot_v:.4g} {spot_u}\n")
        fid.write(f"  Fluence (peak):   {f_peak / 1e4:.5g} J/cm^2\n")
        fid.write(f"  Absorbance:       {absorbance:.2f}\n")
        fid.write("\n--- Inversion Summary ---\n")
        fid.write(f"  Pulses simulated:      {n_pulses}\n")
        fid.write(f"  Pulses with inversion: {n_inv_pulses}\n")
        fid.write(f"  Mean inversion:        {mean_inv:.3f} K\n")
        fid.write(f"  Max inversion:         {max_inv_all:.3f} K\n")
        fid.write(f"  Min inversion:         {min_inv:.3f} K\n")
        fid.write(f"  Std deviation:         {std_inv:.3f} K\n")
        fid.write(f"  First pulse inv:       {first_inv:.3f} K  "
                  f"(Tbase = {tbase_valid[0]:.1f} C)\n")
        fid.write(f"  Last pulse inv:        {last_inv:.3f} K  "
                  f"(Tbase = {tbase_valid[-1]:.1f} C)\n")
        fid.write(f"  Mean inv timing:       {mx_tim_v:.4g} {mx_tim_u} "
                  "after pulse center\n")
        fid.write(f"  Mean inv duration:     {mx_dur_v:.4g} {mx_dur_u}\n")
        if n_inv_pulses > 2:
            fid.write(f"  Trend slope:           {inv_slope:.4g} K/pulse\n")
            fid.write(f"  Corr(Tbase, inv):      {r_corr:.4f}\n")
        fid.write(f"  Mean inv fraction:     "
                  f"{100 * np.mean(inv_fraction):.2f}% of Te excursion\n")

        fid.write("\n--- Per-Pulse Inversion Data ---\n")
        fid.write(f"{'Pulse':>6}  {'Tbase(C)':>12}  {'PeakTe(C)':>12}  "
                  f"{'PeakTl(C)':>12}  {'Teq(C)':>12}  {'Tresid(C)':>12}  "
                  f"{'MaxInv(K)':>14}  {'Te@Inv(C)':>14}  {'Tl@Inv(C)':>14}  "
                  f"{'InvDur(ps)':>12}\n")
        fid.write("-" * 140 + "\n")
        for p in range(n_pulses):
            dur_str = "N/A"
            if inv_dur[p] > 0:
                dur_str = f"{inv_dur[p] * 1e12:.3g}"
            te_inv_str = "N/A"
            tl_inv_str = "N/A"
            if not np.isnan(te_at_inv[p]):
                te_inv_str = f"{te_at_inv[p]:.2f}"
                tl_inv_str = f"{tl_at_inv[p]:.2f}"
            fid.write(f"{p + 1:6d}  {tbase[p]:12.2f}  {te_peak[p]:12.1f}  "
                      f"{tl_peak[p]:12.2f}  {teq[p]:12.2f}  {tresid[p]:12.2f}  "
                      f"{inv_max[p]:14.3f}  {te_inv_str:>14}  {tl_inv_str:>14}  "
                      f"{dur_str:>12}\n")
    print(f"\n  Output written to: {out_path}")

    # ==================  Plots  =============================================
    if make_plots:
        from .plotting import plot_inversion_quantifier

        plot_inversion_quantifier(
            n_pulses=n_pulses, inv_max=inv_max, tbase=tbase,
            te_peak=te_peak, tl_peak=tl_peak,
            te_at_inv=te_at_inv, tl_at_inv=tl_at_inv,
            t_max_inv=t_max_inv, t_onset=t_onset, inv_dur=inv_dur,
            has_inversion=has_inversion, n_inv_pulses=n_inv_pulses,
            save_dir=(output_dir if save_figures else None))

    return _build_results(
        cfg, dr, n_pulses, material, inv_max, te_peak, tl_peak,
        tbase, teq, tresid, t_max_inv, t_onset, inv_dur, te_at_inv, tl_at_inv,
        n_inv_pulses, mean_inv, max_inv_all, min_inv, std_inv, inv_slope, r_corr,
        float(np.mean(inv_fraction)), out_path, output_dir)


def _build_results(cfg, dr, n_pulses, material, inv_max, te_peak, tl_peak,
                   tbase, teq, tresid, t_max_inv, t_onset, inv_dur,
                   te_at_inv, tl_at_inv,
                   n_inv_pulses, mean_inv, max_inv_all, min_inv, std_inv,
                   inv_slope, r_corr, mean_inv_fraction, out_path, output_dir):
    input_cfg = {k: v for k, v in cfg.items() if k != "depthResults"}
    return {
        "solver": "Inversion",
        "solverId": "inversion_quantifier",
        "contractVersion": "v1",
        "material": material,
        "nPulses": n_pulses,
        "nInvPulses": n_inv_pulses,
        # Summary statistics
        "meanInv_K": mean_inv,
        "maxInv_K": max_inv_all,
        "minInv_K": min_inv,
        "stdInv_K": std_inv,
        "invSlope_KperPulse": inv_slope,
        "corrBaseTempInv": r_corr,
        "meanInvFraction": mean_inv_fraction,
        # Per-pulse arrays
        "invMaxPerPulse_K": inv_max,
        "TePeak_C": te_peak,
        "TlPeak_C": tl_peak,
        "Tbase_C": tbase,
        "Teq_C": teq,
        "Tresid_C": tresid,
        "tMaxInv_s": t_max_inv,
        "tOnset_s": t_onset,
        "invDuration_s": inv_dur,
        "Te_atMaxInv_C": te_at_inv,
        "Tl_atMaxInv_C": tl_at_inv,
        # Scalar summaries from the depth solver
        "peakTe_C": dr["peakTe_C"],
        "peakTl_C": dr["peakTl_C"],
        "finalResid_C": dr["finalResid_C"],
        "wallTime_s": dr["wallTime_s"],
        "outputFile": out_path,
        "outputDir": output_dir,
        "inputConfig": input_cfg,
        "depthResults": dr,
        "depthOutputFile": dr["outputFile"],
    }
