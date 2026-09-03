"""Surface-point (0D) pulsed-laser TTM solver.

Python port of ``src/Surface_Point_Solver.m`` from the MATLAB reference
implementation (ultrafast-laser-ttm-toolbox). The numerical pipeline — the
two-stage per-pulse strategy with adaptive RK4 electron–lattice dynamics and
Crank–Nicolson inter-pulse diffusion — is ported line-faithfully; agreement
with the MATLAB golden fixtures is asserted by ``tests/test_surface_point.py``.

Config fields, defaults, and the returned result contract (v1) match the
MATLAB solver field-for-field.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .config import get_cfg_field
from .kernels import cn_coast_const, profile_code, rk4_pulse_phase
from .materials import k_model_name, resolve_material
from .physics import (
    deposit_amplitude,
    deposit_pulse,
    deposit_shape_weight,
    depth_deposit_shape,
    derive_laser,
    energy_mismatch_pct,
    equilibrate,
    validity_warnings,
)
from .progress import ProgressReporter
from .reporting import (
    NO_HISTORY_NOTE,
    apply_case_tag,
    case_tag,
    filename_slug,
    resolve_output_dir,
    write_header,
    write_xy_table,
)
from .schema import defaults as schema_defaults
from .schema import effective_config, require_pulses
from .units import smart_energy, smart_freq, smart_length, smart_time

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz




def surface_point_solver(cfg: dict | None = None) -> dict:
    """Run the 0D surface-point TTM solver. Returns the v1 results dict."""
    if cfg is None:
        cfg = {}

    # Defaults come from the schema so there is one place to read them and
    # one place they can change. See schema.describe_solver('surface_point').
    d = schema_defaults("surface_point")

    make_plots = get_cfg_field(cfg, "makePlots", d["makePlots"])
    save_figures = get_cfg_field(cfg, "saveFigures", d["saveFigures"])
    if save_figures:
        make_plots = True

    # storeHistory=False drops the per-sample time histories, which grow
    # without bound (roughly 550 samples per pulse) and feed only the
    # timeline figures and the report's XY table. Every per-pulse and
    # summary number is tracked incrementally instead, so the physics and
    # the reported scalars are unchanged.
    store_history = bool(get_cfg_field(cfg, "storeHistory", d["storeHistory"]))
    if not store_history and make_plots:
        print("  storeHistory=False: time-history figures disabled.")
        make_plots = False
        save_figures = False

    print("Starting Surface TTM Pulsed Laser Calculator...")

    # ========================  USER INPUTS  =================================
    material = get_cfg_field(cfg, "material", d["material"])

    pavg = get_cfg_field(cfg, "Pavg", d["Pavg"])
    spot_radius = get_cfg_field(cfg, "spotRadius", d["spotRadius"])

    absorbance = get_cfg_field(cfg, "absorbance", d["absorbance"])
    leff = get_cfg_field(cfg, "Leff", d["Leff"])
    t0_c = get_cfg_field(cfg, "T0_C", d["T0_C"])

    pulse_profile_name = get_cfg_field(cfg, "pulseProfile", d["pulseProfile"])
    tau_fwhm = get_cfg_field(cfg, "tau_FWHM", d["tau_FWHM"])
    f_rep = get_cfg_field(cfg, "f_rep", d["f_rep"])
    sim_duration = get_cfg_field(cfg, "simDuration", d["simDuration"])

    depth_profile = get_cfg_field(cfg, "depthProfile", d["depthProfile"])
    dz_target = get_cfg_field(cfg, "dzTarget", d["dzTarget"])
    n_diff = int(get_cfg_field(cfg, "Ndiff", d["Ndiff"]))
    show_progress = get_cfg_field(cfg, "showProgress", d["showProgress"])

    # ==================  Material properties  ===============================
    mat = resolve_material(cfg, needs_optical=False)
    gamma, cl, g_ep, kl = mat.gamma, mat.cl, mat.g_ep, mat.k_total

    # ==================  Incident Fluence  ==================================
    dl = derive_laser(pavg=pavg, f_rep=f_rep, spot_radius=spot_radius,
                      absorbance=absorbance, t0_c=t0_c, gamma=gamma,
                      g_ep=g_ep, sim_duration=sim_duration)
    ep_calc = dl.pulse_energy
    f_si = dl.peak_fluence
    t0 = dl.t0_k
    eabs_areal = dl.absorbed_fluence
    eabs_vol = eabs_areal / leff                            # [J/m^3]
    trep = dl.period
    n_pulses = dl.n_pulses
    require_pulses("surface_point", n_pulses)
    tau_eph = dl.tau_eph

    # Diffusion parameters
    alpha_l = kl / cl
    ldiff = max(5.0 * np.sqrt(alpha_l * sim_duration), 50e-6)
    dz = dz_target
    nz = int(np.ceil(ldiff / dz)) + 1
    ldiff = (nz - 1) * dz
    z_grid = np.arange(nz) * dz
    print(f"  Diffusion: kl={kl:.1f} W/mK, alpha={alpha_l:.3e} m^2/s, "
          f"L={ldiff * 1e6:.1f} um, dz={dz * 1e9:.2f} nm")

    # ==================  Source term  =======================================
    pulse_offset = 5.0 * tau_fwhm
    prof_code = profile_code(pulse_profile_name)

    # ==================  Pulse-by-pulse simulation  =========================
    print(f"  Running pulse-by-pulse simulation ({n_pulses} pulses)...")

    dt_floor_abs = 1e-17
    pulse_fine_win = 6.0 * tau_fwhm
    relax_tol = 1e-6
    relax_max_t = min(trep / 2.0, 50.0 * tau_fwhm)

    # Loop-invariant deposit shape and coast sampling stride
    depth_is_exp = str(depth_profile).lower() == "exponential"
    exp_decay_z, box_mask_z = depth_deposit_shape(z_grid, leff)
    legacy_deposit = bool(get_cfg_field(cfg, "legacyDeposit",
                                        d["legacyDeposit"]))
    w_shape = deposit_shape_weight(z_grid, leff, depth_is_exp)
    n_coast_sample = min(n_diff, 50)
    sample_interval = max(1, n_diff // n_coast_sample)
    progress_interval = max(1, n_pulses // 20)

    cell_times: list[np.ndarray] = []
    cell_te: list[np.ndarray] = []
    cell_tl: list[np.ndarray] = []
    cell_coast_t: list[np.ndarray] = []
    cell_coast_tl: list[np.ndarray] = []
    pulse_center_t = pulse_offset + np.arange(n_pulses) * trep
    teq_vals = np.zeros(n_pulses)
    tresid_vals = np.zeros(n_pulses)
    absorbed = 0.0

    te_now = t0
    tl_now = t0
    tz = t0 * np.ones(nz)

    # Incremental summaries for storeHistory=False. They replicate exactly
    # what the concatenated arrays would yield: the electron trace is the
    # fine Te samples followed by the coast surface samples, in pulse order,
    # with argmax taking the first occurrence.
    track_te_peak = -np.inf
    track_tl_peak = -np.inf
    track_peak_pulse = 1
    track_peak_time = 0.0
    track_nt = 0
    track_t_end = 0.0

    tic_all = time.perf_counter()
    progress = ProgressReporter(n_pulses, title="laserttm: surface point",
                                enabled=show_progress)

    for np_i in range(n_pulses):
        t_pulse = pulse_offset + np_i * trep
        t_start = t_pulse - 5.0 * tau_fwhm

        # --- Phase 1: fine adaptive RK4 around the pulse ---
        loc_t, loc_te, loc_tl, absorbed_phase = rk4_pulse_phase(
            t_pulse, t_start, te_now, tl_now,
            gamma, g_ep, cl,
            n_pulses, trep, pulse_offset, prof_code, tau_fwhm, eabs_vol,
            pulse_fine_win, relax_tol, relax_max_t, dt_floor_abs,
        )
        absorbed += absorbed_phase
        if store_history:
            cell_times.append(loc_t)
            cell_te.append(loc_te)
            cell_tl.append(loc_tl)
        else:
            i_max = int(np.argmax(loc_te))
            if loc_te[i_max] > track_te_peak:
                track_te_peak = float(loc_te[i_max])
                track_peak_pulse = np_i + 1
                track_peak_time = float(loc_t[i_max])
            track_tl_peak = max(track_tl_peak, float(loc_tl.max()))
            track_nt += loc_t.size
            track_t_end = float(loc_t[-1])

        # Equilibrium temperature (energy-conserving)
        teq = equilibrate(loc_te[-1], loc_tl[-1], gamma, cl)
        teq_vals[np_i] = teq

        # --- Phase 2: 1D Crank-Nicolson thermal diffusion ---
        t_fine_end = loc_t[-1]
        if np_i < n_pulses - 1:
            t_next_pulse_start = pulse_offset + (np_i + 1) * trep - 5.0 * tau_fwhm
        else:
            t_next_pulse_start = sim_duration  # last pulse: coast to end
        coast_gap = t_next_pulse_start - t_fine_end

        if coast_gap > 0:
            # Set initial depth profile from the post-pulse layer energy
            amp = (None if legacy_deposit else deposit_amplitude(
                teq, tz[0], gamma, cl, leff, w_shape))
            tz = deposit_pulse(tz, teq, exp_decay_z, box_mask_z, depth_is_exp,
                               amplitude=amp)

            tz, c_t, c_tl = cn_coast_const(
                tz, coast_gap, n_diff, alpha_l, dz, t0, t_fine_end, sample_interval
            )
            if store_history:
                cell_coast_t.append(c_t)
                cell_coast_tl.append(c_tl)
            elif c_tl.size:
                # The stitched Te trace carries the coast surface samples too.
                c_imax = int(np.argmax(c_tl))
                if c_tl[c_imax] > track_te_peak:
                    track_te_peak = float(c_tl[c_imax])
                    track_peak_pulse = np_i + 1
                    track_peak_time = float(c_t[c_imax])
                track_tl_peak = max(track_tl_peak, float(c_tl[c_imax]))
                track_nt += c_t.size
                track_t_end = float(c_t[-1])
            n_coast_steps = c_t.size
            tresidual = tz[0]
        else:
            if store_history:
                cell_coast_t.append(np.empty(0))
                cell_coast_tl.append(np.empty(0))
            n_coast_steps = 0
            tresidual = teq

        tresid_vals[np_i] = tresidual
        te_now = tresidual
        tl_now = tresidual

        if (np_i + 1) % progress_interval == 0 or np_i + 1 == n_pulses:
            print(f"    Pulse {np_i + 1}/{n_pulses}: Te_peak={loc_te.max():.1f} K, "
                  f"Teq={teq:.1f} K, Tresid={tresidual:.1f} K  "
                  f"({loc_t.size} fine + {n_coast_steps} diff steps)")
        progress.update(np_i + 1)

    progress.close()
    wall_time_s = time.perf_counter() - tic_all
    print(f"  Simulation wall time: {wall_time_s:.2f} s")

    if store_history:
        # --- Stitch per-pulse arrays into global vectors (0-based indices) ---
        times = np.concatenate([
            arr for pair in zip(cell_times, cell_coast_t) for arr in pair
        ])
        te = np.concatenate([
            arr for pair in zip(cell_te, cell_coast_tl) for arr in pair
        ])
        tl = np.concatenate([
            arr for pair in zip(cell_tl, cell_coast_tl) for arr in pair
        ])
        pulse_start_idx = np.zeros(n_pulses, dtype=np.int64)
        ptr = 0
        for p in range(n_pulses):
            pulse_start_idx[p] = ptr
            ptr += cell_times[p].size
            ptr += cell_coast_t[p].size
        nt = ptr
        t_end = times[nt - 1]
    else:
        times = np.empty(0)
        te = np.empty(0)
        tl = np.empty(0)
        nt = track_nt
        t_end = track_t_end

    # ==================  Post-processing  ===================================
    te_c = te - 273.15
    tl_c = tl - 273.15

    if store_history:
        idx_te_peak = int(np.argmax(te))
        te_peak = te[idx_te_peak]
        tl_peak = tl.max()
        te_final = te[-1]
        tl_final = tl[-1]

        peak_pulse0 = int(np.searchsorted(pulse_start_idx, idx_te_peak,
                                          side="right") - 1)
        peak_pulse = peak_pulse0 + 1  # 1-based, as in the MATLAB contract
        t_peak_local_val, t_peak_local_unit = smart_time(
            times[idx_te_peak] - pulse_center_t[peak_pulse0]
        )
    else:
        te_peak = track_te_peak
        tl_peak = track_tl_peak
        te_final = tresid_vals[-1]
        tl_final = tresid_vals[-1]
        peak_pulse = track_peak_pulse
        t_peak_local_val, t_peak_local_unit = smart_time(
            track_peak_time - pulse_center_t[peak_pulse - 1]
        )

    # Energy conservation check (approximate in the hybrid 0D+1D model)
    du_depth = cl * _trapezoid(tz - t0, z_grid)     # [J/m^2]
    absorbed_areal = absorbed * leff                # [J/m^2]
    err_rel = energy_mismatch_pct(absorbed_areal, du_depth)

    # ==================  Print results  =====================================
    print("\n============================================================")
    print("  Surface TTM Pulsed Laser Calculator Results")
    print("============================================================")
    print(f"  Material:                 {str(material).upper()}")
    print(f"  gamma (J m^-3 K^-2):     {gamma:.2f}")
    print(f"  Cl    (J m^-3 K^-1):     {cl:.4e}")
    print(f"  G     (W m^-3 K^-1):     {g_ep:.4e}")
    print(f"  kl    (W m^-1 K^-1):     {kl:.1f}")
    print(f"  alpha (m^2/s):           {alpha_l:.4e}")
    print("------------------------------------------------------------")
    tau_eph_val, tau_eph_unit = smart_time(tau_eph)
    ldiff_val, ldiff_unit = smart_length(ldiff)
    dz_val, dz_unit = smart_length(dz)
    t_end_val, t_end_unit = smart_time(t_end)
    print(f"  Incident Fluence F:      {f_si / 1e4:.5g} J/cm^2")
    print(f"  Absorbed Areal Energy:   {eabs_areal:.5g} J/m^2")
    print(f"  Absorbed Vol. Energy:    {eabs_vol:.5g} J/m^3")
    print(f"  tau_e-ph (at T0):        {tau_eph_val:.4g} {tau_eph_unit}")
    print("------------------------------------------------------------")
    print(f"  Diffusion domain:        {ldiff_val:.1f} {ldiff_unit}  "
          f"({nz} nodes, dz={dz_val:.1f} {dz_unit})")
    print(f"  Depth profile:           {depth_profile}")
    print(f"  Diff steps/period:       {n_diff}")
    print("------------------------------------------------------------")
    print(f"  Number of Pulses:        {n_pulses}")
    print(f"  Time Steps:              {nt}")
    print(f"  Simulation Duration:     {t_end_val:.4g} {t_end_unit}")
    print("------------------------------------------------------------")
    print(f"  Peak Electron Temp:      {te_peak - 273.15:.2f} deg C  "
          f"(pulse {peak_pulse}, {t_peak_local_val:.4g} {t_peak_local_unit} from center)")
    print(f"  Peak Lattice Temp:       {tl_peak - 273.15:.2f} deg C")
    print(f"  Final Electron Temp:     {te_final - 273.15:.2f} deg C")
    print(f"  Final Lattice Temp:      {tl_final - 273.15:.2f} deg C")
    print(f"  Final Residual (surf):   {tresid_vals[-1] - 273.15:.2f} deg C  (after diffusion)")
    print("------------------------------------------------------------")
    print(f"  E_absorbed (areal):     {absorbed_areal:.4g} J/m^2")
    print(f"  E_depth (areal):        {du_depth:.4g} J/m^2")
    print("  (Note: mismatch is expected in hybrid 0D+1D model)")
    print("============================================================\n")

    # ==================  Export results to file  ============================
    output_dir = resolve_output_dir(cfg)

    frep_val, frep_unit = smart_freq(f_rep)
    tau_val, tau_unit = smart_time(tau_fwhm)
    ep_val, ep_unit = smart_energy(ep_calc)
    spot_val, spot_unit = smart_length(spot_radius)
    out_filename = apply_case_tag(cfg, (
        f"TTM_{filename_slug(f_rep, tau_fwhm, pavg, spot_radius)}_"
        f"{n_pulses}p_{pulse_profile_name}.txt"))
    out_path = os.path.join(output_dir, out_filename)

    leff_val, leff_unit = smart_length(leff)
    with open(out_path, "w", encoding="utf-8") as fid:
        write_header(fid, "Surface TTM Pulsed Laser Calculator — Output")

        fid.write("--- Laser Parameters ---\n")
        fid.write(f"  Average Power:         {pavg:.4g} W\n")
        fid.write(f"  Repetition Rate:       {frep_val:.4g} {frep_unit}\n")
        fid.write(f"  Pulse Energy:          {ep_val:.4g} {ep_unit}\n")
        fid.write(f"  Pulse Width (FWHM):    {tau_val:.4g} {tau_unit}\n")
        fid.write(f"  Pulse Profile:         {pulse_profile_name}\n")
        fid.write(f"  Spot Radius:           {spot_val:.4g} {spot_unit}\n")
        fid.write(f"  Fluence (peak):        {f_si / 1e4:.5g} J/cm^2\n")
        fid.write(f"  Absorbance:            {absorbance:.2f}\n")
        fid.write(f"  Effective Thickness:   {leff_val:.4g} {leff_unit}\n")
        fid.write(f"  Initial Temperature:   {t0_c:.2f} deg C\n\n")

        fid.write("--- Simulation Settings ---\n")
        fid.write(f"  Material:              {str(material).upper()}\n")
        fid.write(f"  Number of Pulses:      {n_pulses}\n")
        fid.write(f"  Time Steps:            {nt}\n")
        fid.write(f"  Simulation Duration:   {t_end_val:.4g} {t_end_unit}\n")
        fid.write(f"  Depth Profile:         {depth_profile}\n")
        fid.write(f"  Depth Nodes:           {nz}\n")
        fid.write(f"  Diff Steps/Period:     {n_diff}\n\n")

        fid.write("--- Results ---\n")
        fid.write(f"  Peak Electron Temp:    {te_peak - 273.15:.2f} deg C  "
                  f"(pulse {peak_pulse}, {t_peak_local_val:.4g} {t_peak_local_unit} from center)\n")
        fid.write(f"  Peak Lattice Temp:     {tl_peak - 273.15:.2f} deg C\n")
        fid.write(f"  Final Electron Temp:   {te_final - 273.15:.2f} deg C\n")
        fid.write(f"  Final Lattice Temp:    {tl_final - 273.15:.2f} deg C\n")
        fid.write(f"  Final Residual (surf): {tresid_vals[-1] - 273.15:.2f} deg C\n")
        fid.write("\n--- Energy and Pulse Accumulation ---\n")
        fid.write(f"  E_absorbed (areal):    {absorbed_areal:.4g} J/m^2\n")
        fid.write(f"  E_depth (areal):       {du_depth:.4g} J/m^2\n")
        for p in range(n_pulses):
            fid.write(f"  Pulse {p + 1}: Teq={teq_vals[p] - 273.15:.2f} deg C, "
                      f"Tresid={tresid_vals[p] - 273.15:.2f} deg C\n")
        fid.write("\n")

        if store_history:
            write_xy_table(
                fid, "  XY Data: Time (s) | Te (deg C) | Tl (deg C)",
                ("Time_s", "Te_degC", "Tl_degC"),
                ((times[i], te_c[i], tl_c[i]) for i in range(nt)))
        else:
            fid.write(NO_HISTORY_NOTE)
    print(f"  Output written to: {out_path}")

    results = {
        "solver": "SurfacePoint",
        "solverId": "surface_point",
        "contractVersion": "v1",
        "material": material,
        "caseTag": case_tag(cfg),
        "resolvedConfig": effective_config("surface_point", cfg),
        "materialProps": mat.props(k_model_name(mat, constant_only=True)),
        "warnings": validity_warnings(tl_peak - 273.15, mat.t_melt_c, material),
        "outputFile": out_path,
        "outputDir": output_dir,
        "inputConfig": cfg,
        "time_s": times,
        "Te_K": te,
        "Tl_K": tl,
        "Te_C": te_c,
        "Tl_C": tl_c,
        "nPulses": n_pulses,
        "peakPulse": peak_pulse,
        "peakTe_C": te_peak - 273.15,
        "peakTl_C": tl_peak - 273.15,
        "finalTe_C": te_final - 273.15,
        "finalTl_C": tl_final - 273.15,
        "finalResid_C": tresid_vals[-1] - 273.15,
        "wallTime_s": wall_time_s,
        "TeqVals_C": teq_vals - 273.15,
        "TresidVals_C": tresid_vals - 273.15,
        "absorbedAreal_J_m2": absorbed_areal,
        "depthEnergy_J_m2": du_depth,
        "energyMismatch_pct": err_rel,
        "makePlots": make_plots,
        "saveFigures": save_figures,
    }

    # ==================  Plot temperature evolution  ========================
    if make_plots:
        from .plotting import plot_surface_point

        fig_path = plot_surface_point(
            times=times, tl=tl, t_end=t_end,
            teq_vals=teq_vals, tresid_vals=tresid_vals,
            material=material, gamma=gamma, cl=cl, g_ep=g_ep, kl=kl,
            pavg=pavg, frep_val=frep_val, frep_unit=frep_unit,
            ep_val=ep_val, ep_unit=ep_unit, tau_val=tau_val, tau_unit=tau_unit,
            spot_val=spot_val, spot_unit=spot_unit, f_si=f_si,
            absorbance=absorbance, te_peak=te_peak, tl_peak=tl_peak,
            peak_pulse=peak_pulse, absorbed_areal=absorbed_areal,
            du_depth=du_depth, nt=nt,
            save_path=(os.path.join(
                output_dir, os.path.splitext(os.path.basename(out_path))[0] + ".png")
                if save_figures else None),
        )
        if fig_path is not None:
            results["figureFile"] = fig_path

    return results
