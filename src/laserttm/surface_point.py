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
from datetime import datetime

import numpy as np

from .config import get_cfg_field
from .kernels import cn_coast_const, profile_code, rk4_pulse_phase
from .materials import resolve_material
from .progress import ProgressReporter
from .units import smart_energy, smart_freq, smart_length, smart_time

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _matlab_round(x: float) -> int:
    """MATLAB round(): half away from zero (positive args here)."""
    return int(np.floor(x + 0.5))


def surface_point_solver(cfg: dict | None = None) -> dict:
    """Run the 0D surface-point TTM solver. Returns the v1 results dict."""
    if cfg is None:
        cfg = {}

    make_plots = get_cfg_field(cfg, "makePlots", True)
    save_figures = get_cfg_field(cfg, "saveFigures", False)
    if save_figures:
        make_plots = True

    print("Starting Surface TTM Pulsed Laser Calculator...")

    # ========================  USER INPUTS  =================================
    material = get_cfg_field(cfg, "material", "W")

    pavg = get_cfg_field(cfg, "Pavg", 1.0)                 # Average power [W]
    spot_radius = get_cfg_field(cfg, "spotRadius", 80e-6)  # Spot radius [m]

    absorbance = get_cfg_field(cfg, "absorbance", 0.55)    # Absorbance A (0..1)
    leff = get_cfg_field(cfg, "Leff", 100e-9)              # Effective heated thickness [m]
    t0_c = get_cfg_field(cfg, "T0_C", 25.0)                # Initial temperature [deg C]

    pulse_profile_name = get_cfg_field(cfg, "pulseProfile", "gaussian")
    tau_fwhm = get_cfg_field(cfg, "tau_FWHM", 500e-15)     # Pulse width FWHM [s]
    f_rep = get_cfg_field(cfg, "f_rep", 1e6)               # Repetition rate [Hz]
    sim_duration = get_cfg_field(cfg, "simDuration", 100e-6)

    depth_profile = get_cfg_field(cfg, "depthProfile", "exponential")  # 'box'|'exponential'
    dz_target = get_cfg_field(cfg, "dzTarget", 500e-9)     # Depth grid spacing [m]
    n_diff = int(get_cfg_field(cfg, "Ndiff", 100))         # CN steps per inter-pulse period
    show_progress = get_cfg_field(cfg, "showProgress", None)  # waitbar popup

    # ==================  Material properties  ===============================
    mat = resolve_material(cfg, needs_optical=False)
    gamma, cl, g_ep, kl = mat.gamma, mat.cl, mat.g_ep, mat.k_total

    # ==================  Incident Fluence  ==================================
    ep_calc = pavg / f_rep                                  # Pulse energy [J]
    f_si = 2.0 * ep_calc / (np.pi * spot_radius**2)         # Peak Gaussian fluence [J/m^2]

    t0 = t0_c + 273.15
    eabs_areal = absorbance * f_si                          # [J/m^2]
    eabs_vol = eabs_areal / leff                            # [J/m^3]
    trep = 1.0 / f_rep
    n_pulses = _matlab_round(sim_duration * f_rep)
    tau_eph = gamma * t0 / g_ep

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
    exp_decay_z = np.exp(-z_grid / leff)
    box_mask_z = z_grid <= leff
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
        cell_times.append(loc_t)
        cell_te.append(loc_te)
        cell_tl.append(loc_tl)

        # Equilibrium temperature (energy-conserving)
        utot = 0.5 * gamma * loc_te[-1] ** 2 + cl * loc_tl[-1]
        teq = (-cl + np.sqrt(cl**2 + 2.0 * gamma * utot)) / gamma
        teq_vals[np_i] = teq

        # --- Phase 2: 1D Crank-Nicolson thermal diffusion ---
        t_fine_end = loc_t[-1]
        if np_i < n_pulses - 1:
            t_next_pulse_start = pulse_offset + (np_i + 1) * trep - 5.0 * tau_fwhm
        else:
            t_next_pulse_start = sim_duration  # last pulse: coast to end
        coast_gap = t_next_pulse_start - t_fine_end

        if coast_gap > 0:
            # Set initial depth profile from post-pulse Teq
            if depth_is_exp:
                tz = tz + (teq - tz[0]) * exp_decay_z
            else:  # 'box' and the MATLAB otherwise-branch
                tz[box_mask_z] = teq

            tz, c_t, c_tl = cn_coast_const(
                tz, coast_gap, n_diff, alpha_l, dz, t0, t_fine_end, sample_interval
            )
            cell_coast_t.append(c_t)
            cell_coast_tl.append(c_tl)
            tresidual = tz[0]
        else:
            cell_coast_t.append(np.empty(0))
            cell_coast_tl.append(np.empty(0))
            tresidual = teq

        tresid_vals[np_i] = tresidual
        te_now = tresidual
        tl_now = tresidual

        if (np_i + 1) % progress_interval == 0 or np_i + 1 == n_pulses:
            print(f"    Pulse {np_i + 1}/{n_pulses}: Te_peak={loc_te.max():.1f} K, "
                  f"Teq={teq:.1f} K, Tresid={tresidual:.1f} K  "
                  f"({loc_t.size} fine + {cell_coast_t[-1].size} diff steps)")
        progress.update(np_i + 1)

    progress.close()
    wall_time_s = time.perf_counter() - tic_all
    print(f"  Simulation wall time: {wall_time_s:.2f} s")

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
    pulse_end_idx = np.zeros(n_pulses, dtype=np.int64)
    coast_end_idx = np.zeros(n_pulses, dtype=np.int64)
    ptr = 0
    for p in range(n_pulses):
        pulse_start_idx[p] = ptr
        ptr += cell_times[p].size
        pulse_end_idx[p] = ptr - 1
        ptr += cell_coast_t[p].size
        coast_end_idx[p] = ptr - 1
    nt = ptr
    t_end = times[nt - 1]

    # ==================  Post-processing  ===================================
    te_c = te - 273.15
    tl_c = tl - 273.15

    idx_te_peak = int(np.argmax(te))
    te_peak = te[idx_te_peak]
    tl_peak = tl.max()
    te_final = te[-1]
    tl_final = tl[-1]

    peak_pulse0 = int(np.searchsorted(pulse_start_idx, idx_te_peak, side="right") - 1)
    peak_pulse = peak_pulse0 + 1  # 1-based pulse number, as in the MATLAB contract
    t_peak_local_val, t_peak_local_unit = smart_time(
        times[idx_te_peak] - pulse_center_t[peak_pulse0]
    )

    # ==================  Baseline envelope fit  =============================
    baseline_pulse_nums = np.arange(1, n_pulses + 1, dtype=float)
    baseline_temps = tresid_vals.copy()

    baseline_times_s = np.zeros(n_pulses)
    for bp in range(n_pulses):
        if 0 <= coast_end_idx[bp] < nt:
            baseline_times_s[bp] = times[coast_end_idx[bp]]
        else:
            baseline_times_s[bp] = times[pulse_end_idx[bp]]

    baseline_fit_y = None
    extrap_times_s = None
    if n_pulses >= 3:
        from scipy.optimize import minimize

        def exp_sat(p, n):
            return p[0] - (p[0] - t0) * np.exp(-n / max(p[1], 0.1))

        def cost(p):
            return np.sum((exp_sat(p, baseline_pulse_nums) - baseline_temps) ** 2)

        p0 = np.array([baseline_temps[-1] * 1.2, n_pulses / 3.0])
        # fminsearch analogue: Nelder-Mead with TolX=1e-4 (default), TolFun=1e-12
        res = minimize(cost, p0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-12,
                                "maxfev": 10000, "maxiter": 400})
        p_fit = res.x
        t_ss_k = p_fit[0]
        n_char = p_fit[1]
        baseline_fit_y = exp_sat(p_fit, baseline_pulse_nums)
        fit_residual = float(np.sqrt(np.mean((baseline_fit_y - baseline_temps) ** 2)))
        baseline_fit_ok = True

        n_extrap = int(np.ceil(n_pulses * 1.5))
        extrap_nums = np.arange(1, n_extrap + 1, dtype=float)
        extrap_times_s = baseline_times_s[0] + (extrap_nums - 1) * trep
    else:
        t_ss_k = baseline_temps[-1]
        n_char = np.nan
        baseline_fit_ok = False
        fit_residual = np.nan
    t_ss_c = t_ss_k - 273.15

    # Energy conservation check (approximate in the hybrid 0D+1D model)
    du_depth = cl * _trapezoid(tz - t0, z_grid)     # [J/m^2]
    absorbed_areal = absorbed * leff                # [J/m^2]
    err_rel = abs(absorbed_areal - du_depth) / max(abs(absorbed_areal),
                                                   np.finfo(float).eps) * 100.0

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
    print("  Baseline Envelope (material temperature buildup):")
    print(f"  Projected Steady-State:  {t_ss_c:.2f} deg C")
    if baseline_fit_ok:
        print(f"  Characteristic Pulses:   {n_char:.1f}  "
              f"({(1 - np.exp(-n_pulses / n_char)) * 100:.1f}% of steady-state reached)")
        print(f"  Baseline Fit RMSE:       {fit_residual:.4g} K")
    print("------------------------------------------------------------")
    print(f"  E_absorbed (areal):     {absorbed_areal:.4g} J/m^2")
    print(f"  E_depth (areal):        {du_depth:.4g} J/m^2")
    print("  (Note: mismatch is expected in hybrid 0D+1D model)")
    print("============================================================\n")

    # ==================  Export results to file  ============================
    default_out = os.path.join(os.getcwd(), "outputs")
    output_dir = get_cfg_field(cfg, "outputDir", default_out)
    os.makedirs(output_dir, exist_ok=True)

    frep_val, frep_unit = smart_freq(f_rep)
    tau_val, tau_unit = smart_time(tau_fwhm)
    ep_val, ep_unit = smart_energy(ep_calc)
    spot_val, spot_unit = smart_length(spot_radius)
    freq_str = (f"{frep_val:.4g}_{frep_unit}").replace(".", "p")
    pulse_str = (f"{tau_val:.4g}_{tau_unit}").replace(".", "p")
    power_str = (f"{pavg:.4g}_W").replace(".", "p")
    spot_str = (f"{spot_val:.4g}_{spot_unit}").replace(".", "p")
    out_filename = (f"TTM_{freq_str}_{pulse_str}_{power_str}_{spot_str}_"
                    f"{n_pulses}p_{pulse_profile_name}.txt")
    out_path = os.path.join(output_dir, out_filename)

    leff_val, leff_unit = smart_length(leff)
    with open(out_path, "w") as fid:
        fid.write("============================================================\n")
        fid.write("  Surface TTM Pulsed Laser Calculator — Output\n")
        # Local wall-clock on purpose, matching the MATLAB reference output
        fid.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")  # noqa: DTZ005
        fid.write("============================================================\n\n")

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
        fid.write("\n--- Baseline Envelope (Material Temperature) ---\n")
        fid.write(f"  Projected Steady-State: {t_ss_c:.2f} deg C\n")
        if baseline_fit_ok:
            fid.write(f"  Char. Pulses (n_char):  {n_char:.1f}\n")
            fid.write(f"  Steady-State Reached:   "
                      f"{(1 - np.exp(-n_pulses / n_char)) * 100:.1f}%\n")
            fid.write(f"  Baseline Fit RMSE:      {fit_residual:.4g} K\n")
        fid.write(f"  E_absorbed (areal):    {absorbed_areal:.4g} J/m^2\n")
        fid.write(f"  E_depth (areal):       {du_depth:.4g} J/m^2\n")
        for p in range(n_pulses):
            fid.write(f"  Pulse {p + 1}: Teq={teq_vals[p] - 273.15:.2f} deg C, "
                      f"Tresid={tresid_vals[p] - 273.15:.2f} deg C\n")
        fid.write("\n")

        fid.write("============================================================\n")
        fid.write("  XY Data: Time (s) | Te (deg C) | Tl (deg C)\n")
        fid.write("============================================================\n")
        fid.write(f"{'Time_s':>20}  {'Te_degC':>16}  {'Tl_degC':>16}\n")
        fid.writelines(
            f"{times[i]:20.12e}  {te_c[i]:16.6f}  {tl_c[i]:16.6f}\n" for i in range(nt)
        )
    print(f"  Output written to: {out_path}")

    results = {
        "solver": "SurfacePoint",
        "solverId": "surface_point",
        "contractVersion": "v1",
        "material": material,
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
        "finalResid_C": tresid_vals[-1] - 273.15,
        "projectedSteadyState_C": t_ss_c,
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
            baseline_fit_ok=baseline_fit_ok, baseline_fit_y=baseline_fit_y,
            extrap_times_s=extrap_times_s, t_ss_c=t_ss_c,
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
