"""Radial surface pulsed-laser TTM solver.

Python port of ``src/Radial_Profile_Solver.m``: 0D electron–lattice dynamics
mapped radially under the Gaussian beam profile, with per-pulse depth
Crank–Nicolson cooling and cylindrical-coordinate radial diffusion, both with
temperature-dependent k(T).

Two modes, as in MATLAB (``radialSolveMode``): ``'scale'`` (default) solves
one 0D TTM at beam centre and scales the rise by the Gaussian fluence ratio;
``'independent'`` solves a 0D TTM per radial node (captures the nonlinear
Ce(Te) response at each local fluence). Config fields, defaults, and the v1
result contract match the MATLAB solver field-for-field.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np

from .config import get_cfg_field
from .kernels import (
    cn_coast_kt,
    cn_depth_multi_kt,
    k_hybrid,
    profile_code,
    radial_coast_kt,
    rk4_pulse_phase,
)
from .materials import k_model_name, k_table, resolve_material
from .progress import ProgressReporter
from .units import smart_energy, smart_freq, smart_length, smart_time


def _matlab_round(x: float) -> int:
    return int(np.floor(x + 0.5))


def _melt_radius_um(t_surf_c: np.ndarray, r_grid: np.ndarray,
                    t_melt_c: float) -> float:
    """Interpolated radius [um] where the surface profile crosses t_melt_c."""
    below = np.flatnonzero(t_surf_c < t_melt_c)
    if below.size > 0 and below[0] > 0:
        ib = below[0]
        return float(np.interp(
            t_melt_c,
            t_surf_c[ib - 1: ib + 1][::-1],
            (r_grid[ib - 1: ib + 1] * 1e6)[::-1]))
    return 0.0


def _early_stop_hit(t_surf_k: np.ndarray, r_grid: np.ndarray, t_melt_c: float,
                    target_um: float, pulse_no: int) -> bool:
    """True when the melt radius has reached the early-stop target."""
    t_surf_c = t_surf_k - 273.15
    if t_surf_c[0] < t_melt_c:
        return False
    cur_melt_r = _melt_radius_um(t_surf_c, r_grid, t_melt_c)
    if cur_melt_r >= target_um:
        print(f"    >>> Early stop at pulse {pulse_no}: melt radius "
              f"{cur_melt_r:.2f} um >= target {target_um:.2f} um")
        return True
    return False


def radial_profile_solver(cfg: dict | None = None) -> dict:
    """Run the radial-profile TTM solver. Returns the v1 results dict."""
    if cfg is None:
        cfg = {}

    make_plots = get_cfg_field(cfg, "makePlots", True)
    save_figures = get_cfg_field(cfg, "saveFigures", False)
    if save_figures:
        make_plots = True

    # storeHistory=False drops the per-pulse time histories (kept only for
    # the timeline plots), bounding memory for very long runs: ~100k pulses
    # would otherwise accumulate gigabytes of RK4/coast samples. All physics,
    # per-pulse scalars, and radial profiles are unaffected.
    store_history = bool(get_cfg_field(cfg, "storeHistory", True))
    if not store_history and make_plots:
        print("  storeHistory=False: time-history figures disabled.")
        make_plots = False
        save_figures = False

    print("=== Radial Surface TTM Pulsed Laser Calculator ===")

    # ========================  USER INPUTS  =================================
    material = get_cfg_field(cfg, "material", "W")
    mat = resolve_material(cfg, needs_optical=False)
    gamma, cl, g_ep, kl = mat.gamma, mat.cl, mat.g_ep, mat.k_total

    pavg = get_cfg_field(cfg, "Pavg", 40.0)
    spot_radius = get_cfg_field(cfg, "spotRadius", 100e-6)
    f_rep = get_cfg_field(cfg, "f_rep", 18e6)
    tau_fwhm = get_cfg_field(cfg, "tau_FWHM", 100e-15)
    pulse_profile_name = get_cfg_field(cfg, "pulseProfile", "gaussian")

    absorbance = get_cfg_field(cfg, "absorbance", 0.55)
    leff = get_cfg_field(cfg, "Leff", 100e-9)
    t0_c = get_cfg_field(cfg, "T0_C", 25.0)

    nr = int(get_cfg_field(cfg, "Nr", 80))
    r_max_factor = get_cfg_field(cfg, "rMax_factor", 5)

    radial_solve_mode = str(get_cfg_field(cfg, "radialSolveMode", "scale")).lower()
    if radial_solve_mode not in ("scale", "independent"):
        raise ValueError(
            f'Unknown radialSolveMode "{radial_solve_mode}". '
            "Use 'scale' or 'independent'.")

    sim_duration = get_cfg_field(cfg, "simDuration", 1e-3)

    early_stop_melt_radius_um = get_cfg_field(cfg, "earlyStopMeltRadius_um", 0)
    # Defaults to this material's melting point, not tungsten's.
    early_stop_t_melt_c = get_cfg_field(cfg, "earlyStopT_melt_C", mat.t_melt_c)
    early_stop_check_interval = int(get_cfg_field(cfg, "earlyStopCheckInterval", 100))
    early_stop_enabled = early_stop_melt_radius_um > 0

    depth_profile = str(get_cfg_field(cfg, "depthProfile", "exponential")).lower()
    dz_target = get_cfg_field(cfg, "dzTarget", 500e-9)
    n_diff_min = int(get_cfg_field(cfg, "Ndiff", 100))
    show_progress = get_cfg_field(cfg, "showProgress", None)

    # Hybrid k(T): tungsten table, constant kl otherwise (this solver has no
    # separate electron conductivity, unlike the depth solver's ke0+kl table)
    k_tab_t, k_tab_k = k_table(mat)

    # ==================  Derived quantities  ================================
    t0 = t0_c + 273.15
    ep = pavg / f_rep
    f_peak = 2.0 * ep / (np.pi * spot_radius**2)
    eabs_areal = absorbance * f_peak
    eabs_vol = eabs_areal / leff
    trep = 1.0 / f_rep
    n_pulses = _matlab_round(sim_duration * f_rep)

    # Diffusion parameters (hybrid k at T0 for grid sizing)
    alpha_l = float(k_hybrid(np.array([t0]), k_tab_t, k_tab_k)[0]) / cl
    ldiff = max(5.0 * np.sqrt(alpha_l * sim_duration), 50e-6)
    dz = dz_target
    nz = int(np.ceil(ldiff / dz)) + 1
    z_grid = np.arange(nz) * dz

    # Radial grid
    r_max = r_max_factor * spot_radius
    r_grid = np.linspace(0.0, r_max, nr)
    r_plot_um = r_grid * 1e6
    fluence_ratio = np.exp(-2.0 * r_grid**2 / spot_radius**2)
    dr = r_max / (nr - 1)

    # Adaptive CN substep counts (target Fourier number <= 0.5)
    f_target = 0.5
    n_diff = max(n_diff_min, int(np.ceil(alpha_l * trep / (f_target * dz**2))))
    n_diff_rad = max(10, int(np.ceil(alpha_l * trep / (f_target * dr**2))))

    print(f"  Material:       {str(material).upper()}")
    print(f"  Conductivity:   {k_model_name(mat)}")
    print(f"  Mode:           {radial_solve_mode}")
    print(f"  Radial points:  {nr}  (0 to {r_max * 1e6:.0f} um)")
    print(f"  Pulses:         {n_pulses}  (simDuration={sim_duration:.4g} s)")
    print(f"  Spot Radius:    {spot_radius * 1e6:.0f} um")
    print(f"  Depth CN steps: {n_diff}  (f={alpha_l * trep / n_diff / dz**2:.3f})")
    print(f"  Radial CN steps:{n_diff_rad}  (f={alpha_l * trep / n_diff_rad / dr**2:.3f})")
    l_lat = np.sqrt(alpha_l * sim_duration)
    print(f"  Lateral diff:   {l_lat * 1e6:.2f} um  (spot: {spot_radius * 1e6:.0f} um)")

    # ==================  Simulation  ========================================
    pulse_offset = 5.0 * tau_fwhm
    dt_floor_abs = 1e-17
    pulse_fine_win = 6.0 * tau_fwhm
    relax_tol = 1e-6
    relax_max_t = min(trep / 2.0, 50.0 * tau_fwhm)
    prof_code = profile_code(pulse_profile_name)

    # Loop-invariant deposit shape (same precompute as scanning_beam)
    depth_is_exp = depth_profile == "exponential"
    exp_decay_z = np.exp(-z_grid / leff)
    box_mask_z = z_grid <= leff

    cell_times: list[np.ndarray] = []
    cell_tl: list[np.ndarray] = []
    cell_coast_t: list[np.ndarray] = []
    cell_coast_tl: list[np.ndarray] = []
    teq_vals = np.zeros(n_pulses)
    tresid_vals = np.zeros(n_pulses)
    tresid_radial = np.zeros((n_pulses, nr))

    n_profile_snaps = min(n_pulses, 12)
    if n_pulses > 1:
        profile_snap_pulses = np.unique(np.round(
            np.logspace(0, np.log10(n_pulses), n_profile_snaps)).astype(int))
    else:
        profile_snap_pulses = np.array([1])

    progress_interval = max(1, n_pulses // 20)
    tic_all = time.perf_counter()
    n_pulses_run = n_pulses
    progress = ProgressReporter(n_pulses, title="laserttm: radial profile",
                                enabled=show_progress)

    if radial_solve_mode == "scale":
        print("  Running single center-point simulation...")

        te_now = t0
        tl_now = t0
        tz = t0 * np.ones(nz)
        tr_surf = t0 * np.ones(nr)

        for np_i in range(n_pulses):
            t_pulse = pulse_offset + np_i * trep
            t_start = t_pulse - 5.0 * tau_fwhm

            # --- Phase 1: RK4 around the pulse (beam centre, full fluence) ---
            loc_t, loc_te, loc_tl, _ = rk4_pulse_phase(
                t_pulse, t_start, te_now, tl_now,
                gamma, g_ep, cl,
                n_pulses, trep, pulse_offset, prof_code, tau_fwhm, eabs_vol,
                pulse_fine_win, relax_tol, relax_max_t, dt_floor_abs,
            )
            if store_history:
                cell_times.append(loc_t)
                cell_tl.append(loc_tl)

            utot = 0.5 * gamma * loc_te[-1] ** 2 + cl * loc_tl[-1]
            teq = (-cl + np.sqrt(cl**2 + 2.0 * gamma * utot)) / gamma
            teq_vals[np_i] = teq

            # Deposit pulse heat into the radial surface array
            dt_pulse = teq - te_now
            tr_surf = tr_surf + dt_pulse * fluence_ratio

            # --- Phase 2: depth CN cooling + radial CN diffusion ---
            t_fine_end = loc_t[-1]
            if np_i < n_pulses - 1:
                t_next_start = pulse_offset + (np_i + 1) * trep - 5.0 * tau_fwhm
            else:
                t_next_start = sim_duration
            coast_gap = t_next_start - t_fine_end

            if coast_gap > 0:
                if depth_is_exp:
                    tz = tz + (teq - tz[0]) * exp_decay_z
                else:  # 'box' and the MATLAB otherwise-branch
                    tz[box_mask_z] = teq

                n_diff_local = max(n_diff, int(np.ceil(
                    alpha_l * coast_gap / (f_target * dz**2))))
                n_sample = min(n_diff_local, 50)
                sample_int = max(1, n_diff_local // n_sample)
                tz, c_t, c_tl = cn_coast_kt(
                    tz, coast_gap, n_diff_local, dz, t0, cl,
                    k_tab_t, k_tab_k, t_fine_end, sample_int)
                if store_history:
                    cell_coast_t.append(c_t)
                    cell_coast_tl.append(c_tl)
                tresidual = tz[0]

                # Apply depth cooling to the radial surface array
                if (teq - t0) > 1e-10:
                    survival = (tresidual - t0) / (teq - t0)
                else:
                    survival = 1.0
                tr_surf = t0 + (tr_surf - t0) * survival

                # Radial CN diffusion (cylindrical coordinates)
                n_rad_steps = max(n_diff_rad, int(np.ceil(
                    alpha_l * coast_gap / (f_target * dr**2))))
                dt_rad = coast_gap / n_rad_steps
                tr_surf = radial_coast_kt(
                    tr_surf, n_rad_steps, dt_rad, dr, t0, cl, k_tab_t, k_tab_k)
            elif store_history:
                cell_coast_t.append(np.empty(0))
                cell_coast_tl.append(np.empty(0))

            tresid_vals[np_i] = tr_surf[0]
            tresid_radial[np_i, :] = tr_surf
            te_now = tr_surf[0]
            tl_now = tr_surf[0]

            if (np_i + 1) % progress_interval == 0 or np_i + 1 == n_pulses:
                print(f"    Pulse {np_i + 1}/{n_pulses}: "
                      f"Teq={teq - 273.15:.1f} C, Tresid={tr_surf[0] - 273.15:.1f} C")
            progress.update(np_i + 1)

            # --- Early stop check ---
            if (early_stop_enabled
                    and (np_i + 1) % early_stop_check_interval == 0
                    and _early_stop_hit(tr_surf, r_grid, early_stop_t_melt_c,
                                        early_stop_melt_radius_um, np_i + 1)):
                n_pulses_run = np_i + 1
                break

    else:
        print(f"  Running independent 0D solves at {nr} radial nodes (pulse-major)...")

        te_all = t0 * np.ones(nr)
        tl_all = t0 * np.ones(nr)
        tz_all = t0 * np.ones((nz, nr))
        eabs_vol_all = eabs_vol * fluence_ratio
        active = fluence_ratio >= 1e-12

        for np_i in range(n_pulses):
            t_pulse = pulse_offset + np_i * trep
            t_start = t_pulse - 5.0 * tau_fwhm
            t_fine_end_center = t_start

            # --- Phase 1: RK4 TTM at each radial node ---
            for ri in range(nr):
                if not active[ri]:
                    continue

                loc_t, loc_te, loc_tl, _ = rk4_pulse_phase(
                    t_pulse, t_start, te_all[ri], tl_all[ri],
                    gamma, g_ep, cl,
                    n_pulses, trep, pulse_offset, prof_code, tau_fwhm,
                    eabs_vol_all[ri],
                    pulse_fine_win, relax_tol, relax_max_t, dt_floor_abs,
                )
                if ri == 0:
                    if store_history:
                        cell_times.append(loc_t)
                        cell_tl.append(loc_tl)
                    t_fine_end_center = loc_t[-1]

                utot = 0.5 * gamma * loc_te[-1] ** 2 + cl * loc_tl[-1]
                teq = (-cl + np.sqrt(cl**2 + 2.0 * gamma * utot)) / gamma
                if ri == 0:
                    teq_vals[np_i] = teq

                # Deposit pulse energy into this node's depth profile
                if depth_is_exp:
                    tz_all[:, ri] = tz_all[:, ri] \
                        + (teq - tz_all[0, ri]) * exp_decay_z
                else:
                    tz_all[box_mask_z, ri] = teq
                te_all[ri] = teq
                tl_all[ri] = teq

            # --- Phase 2: depth CN diffusion at each node ---
            if np_i < n_pulses - 1:
                t_next_start = pulse_offset + (np_i + 1) * trep - 5.0 * tau_fwhm
            else:
                t_next_start = sim_duration
            coast_gap = t_next_start - t_fine_end_center

            if coast_gap > 0:
                n_diff_local = max(n_diff, int(np.ceil(
                    alpha_l * coast_gap / (f_target * dz**2))))
                dt_diff = coast_gap / n_diff_local
                n_sample = min(n_diff_local, 50)
                sample_int = max(1, n_diff_local // n_sample)

                c_t, c_tl = cn_depth_multi_kt(
                    tz_all, active, n_diff_local, dt_diff, dz, t0, cl,
                    k_tab_t, k_tab_k, t_fine_end_center, sample_int)
                if store_history:
                    cell_coast_t.append(c_t)
                    cell_coast_tl.append(c_tl)
                te_all[active] = tz_all[0, active]
                tl_all[active] = tz_all[0, active]

                # --- Phase 3: radial CN diffusion ---
                tr_pre = te_all.copy()
                n_rad_steps = max(n_diff_rad, int(np.ceil(
                    alpha_l * coast_gap / (f_target * dr**2))))
                dt_rad = coast_gap / n_rad_steps
                tr_surf = radial_coast_kt(
                    te_all, n_rad_steps, dt_rad, dr, t0, cl, k_tab_t, k_tab_k)

                # Adjust depth profiles to match post-radial surface temps
                for ri in range(nr):
                    dt_pre = tr_pre[ri] - t0
                    if dt_pre > 1e-10:
                        survival = max((tr_surf[ri] - t0) / dt_pre, 0.0)
                        tz_all[:, ri] = t0 + (tz_all[:, ri] - t0) * survival
                    elif tr_surf[ri] > t0 + 1e-10:
                        # Heat arrived via radial diffusion into a cold node
                        tz_all[0, ri] = tr_surf[ri]
                te_all = tr_surf.copy()
                tl_all = tr_surf.copy()
            elif store_history:
                cell_coast_t.append(np.empty(0))
                cell_coast_tl.append(np.empty(0))

            tresid_radial[np_i, :] = te_all
            tresid_vals[np_i] = te_all[0]

            if (np_i + 1) % progress_interval == 0 or np_i + 1 == n_pulses:
                print(f"    Pulse {np_i + 1}/{n_pulses}: "
                      f"Teq={teq_vals[np_i] - 273.15:.1f} C, "
                      f"Tresid={te_all[0] - 273.15:.1f} C")
            progress.update(np_i + 1)

            # --- Early stop check ---
            if (early_stop_enabled
                    and (np_i + 1) % early_stop_check_interval == 0
                    and _early_stop_hit(te_all, r_grid, early_stop_t_melt_c,
                                        early_stop_melt_radius_um, np_i + 1)):
                n_pulses_run = np_i + 1
                break

    # ==================  Shared epilogue (both modes)  ======================
    n_pulses = n_pulses_run
    cell_times = cell_times[:n_pulses]
    cell_tl = cell_tl[:n_pulses]
    cell_coast_t = cell_coast_t[:n_pulses]
    cell_coast_tl = cell_coast_tl[:n_pulses]
    teq_vals = teq_vals[:n_pulses]
    tresid_vals = tresid_vals[:n_pulses]
    tresid_radial = tresid_radial[:n_pulses, :]

    progress.close()
    wall_time = time.perf_counter() - tic_all
    print(f"  Wall time: {wall_time:.2f} s")

    profile_snap_pulses = profile_snap_pulses[profile_snap_pulses <= n_pulses]
    snap_radial_profiles = tresid_radial[profile_snap_pulses - 1, :]
    profile_snaps_label = []
    for p in profile_snap_pulses:
        tv, tu = smart_time(p * trep)
        profile_snaps_label.append(f"Pulse {p} ({tv:.3g} {tu})")

    # ==================  Stitch centre-point time history  ==================
    if store_history:
        all_times = np.concatenate([
            arr for pair in zip(cell_times, cell_coast_t) for arr in pair])
        all_tl_surf = np.concatenate([
            arr for pair in zip(cell_tl, cell_coast_tl) for arr in pair])
    else:
        all_times = np.empty(0)
        all_tl_surf = np.empty(0)

    # ==================  Print results  =====================================
    frep_v, frep_u = smart_freq(f_rep)
    tau_v, tau_u = smart_time(tau_fwhm)
    ep_v, ep_u = smart_energy(ep)
    spot_v, spot_u = smart_length(spot_radius)
    sim_dur_v, sim_dur_u = smart_time(sim_duration)

    print("\n============================================================")
    print("  Radial Surface TTM Calculator — Results")
    print("============================================================")
    print(f"  Material:              {str(material).upper()}")
    print(f"  Mode:                  {radial_solve_mode}")
    print(f"  gamma [J m^-3 K^-2]:  {gamma:.2f}")
    print(f"  Cl    [J m^-3 K^-1]:  {cl:.4e}")
    print(f"  G     [W m^-3 K^-1]:  {g_ep:.4e}")
    print(f"  kl    [W m^-1 K^-1]:  {kl:.1f}")
    print("------------------------------------------------------------")
    print(f"  Avg Power:             {pavg:.3g} W")
    print(f"  Rep Rate:              {frep_v:.4g} {frep_u}")
    print(f"  Pulse Energy:          {ep_v:.4g} {ep_u}")
    print(f"  Pulse Width (FWHM):    {tau_v:.4g} {tau_u}")
    print(f"  Spot Radius:           {spot_v:.4g} {spot_u}")
    print(f"  Fluence (peak):        {f_peak / 1e4:.5g} J/cm^2")
    print(f"  Absorbance:            {absorbance:.2f}")
    print("------------------------------------------------------------")
    print(f"  Pulses simulated:      {n_pulses}")
    print(f"  Simulation duration:   {sim_dur_v:.4g} {sim_dur_u}")
    print(f"  Radial nodes:          {nr}")
    print(f"  Wall time:             {wall_time:.2f} s")
    print("------------------------------------------------------------")
    print(f"  Peak Teq (center):     {teq_vals.max() - 273.15:.1f} C")
    print(f"  Final Tresid (center): {tresid_vals[-1] - 273.15:.1f} C")
    print("============================================================\n")

    # ==================  Export results  ====================================
    default_out = os.path.join(os.getcwd(), "outputs")
    output_dir = get_cfg_field(cfg, "outputDir", default_out)
    os.makedirs(output_dir, exist_ok=True)

    freq_str = (f"{frep_v:.4g}_{frep_u}").replace(".", "p")
    pulse_str = (f"{tau_v:.4g}_{tau_u}").replace(".", "p")
    power_str = (f"{pavg:.4g}_W").replace(".", "p")
    spot_str = (f"{spot_v:.4g}_{spot_u}").replace(".", "p")
    out_filename = (f"TTM_Radial_Result_{freq_str}_{pulse_str}_{power_str}_"
                    f"{spot_str}_{n_pulses}p_{pulse_profile_name}.txt")
    case_tag = get_cfg_field(cfg, "caseTag", "")
    if case_tag:
        out_filename = f"{case_tag}__{out_filename}"
    out_path = os.path.join(output_dir, out_filename)

    final_radial_t = tresid_radial[-1, :]
    with open(out_path, "w") as fid:
        fid.write("============================================================\n")
        fid.write("  Radial Surface TTM Calculator — Output\n")
        # Local wall-clock on purpose, matching the MATLAB reference output
        fid.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")  # noqa: DTZ005
        fid.write(f"  Mode: {radial_solve_mode}\n")
        fid.write("============================================================\n\n")
        fid.write(f"--- Material: {str(material).upper()} ---\n")
        fid.write(f"  gamma = {gamma:.2f}  J m^-3 K^-2\n")
        fid.write(f"  Cl    = {cl:.4e}  J m^-3 K^-1\n")
        fid.write(f"  G     = {g_ep:.4e}  W m^-3 K^-1\n")
        fid.write(f"  kl    = {kl:.1f}  W m^-1 K^-1\n")
        fid.write("\n--- Laser ---\n")
        fid.write(f"  Average Power:   {pavg:.4g} W\n")
        fid.write(f"  Rep Rate:        {frep_v:.4g} {frep_u}\n")
        fid.write(f"  Pulse Energy:    {ep_v:.4g} {ep_u}\n")
        fid.write(f"  Pulse Width:     {tau_v:.4g} {tau_u}\n")
        fid.write(f"  Spot Radius:     {spot_v:.4g} {spot_u}\n")
        fid.write(f"  Fluence (peak):  {f_peak / 1e4:.5g} J/cm^2\n")
        fid.write(f"  Absorbance:      {absorbance:.2f}\n")
        fid.write("\n--- Results ---\n")
        fid.write(f"  Pulses:          {n_pulses}\n")
        fid.write(f"  Sim duration:    {sim_dur_v:.4g} {sim_dur_u}\n")
        fid.write(f"  Peak Teq:        {teq_vals.max() - 273.15:.1f} C\n")
        fid.write(f"  Final Tresid:    {tresid_vals[-1] - 273.15:.1f} C\n")
        fid.write(f"  Wall time:       {wall_time:.2f} s\n")
        fid.write("\n--- Per-Pulse Data ---\n")
        for p in range(n_pulses):
            fid.write(f"  Pulse {p + 1}: Teq={teq_vals[p] - 273.15:.2f} C, "
                      f"Tresid={tresid_vals[p] - 273.15:.2f} C\n")
        fid.write("\n--- Final Radial Profile (r [um] | T [C]) ---\n")
        for ri in range(nr):
            fid.write(f"  r = {r_grid[ri] * 1e6:8.2f} um :  "
                      f"T = {final_radial_t[ri] - 273.15:.2f} C\n")
    print(f"  Output written to: {out_path}\n")

    # ==================  Plots  =============================================
    if make_plots:
        from .plotting import plot_radial_profile

        plot_radial_profile(
            all_times=all_times, all_tl_surf=all_tl_surf,
            r_plot_um=r_plot_um, r_grid=r_grid,
            snap_radial_profiles=snap_radial_profiles,
            profile_snaps_label=profile_snaps_label,
            tresid_radial=tresid_radial, teq_vals=teq_vals,
            tresid_vals=tresid_vals, spot_radius=spot_radius, r_max=r_max,
            material=material, mode=radial_solve_mode,
            gamma=gamma, cl=cl, g_ep=g_ep, kl=kl,
            pavg=pavg, frep_v=frep_v, frep_u=frep_u, ep_v=ep_v, ep_u=ep_u,
            tau_v=tau_v, tau_u=tau_u, spot_v=spot_v, spot_u=spot_u,
            f_peak=f_peak, absorbance=absorbance,
            n_pulses=n_pulses, wall_time=wall_time, t0_c=t0_c,
            save_dir=(output_dir if save_figures else None))
        print("Done.")

    return {
        "solver": "Radial",
        "solverId": "radial_profile",
        "contractVersion": "v1",
        "material": material,
        "mode": radial_solve_mode,
        "nPulses": n_pulses,
        "peakTeq_C": teq_vals.max() - 273.15,
        "finalResid_C": tresid_vals[-1] - 273.15,
        "wallTime_s": wall_time,
        "outputFile": out_path,
        "outputDir": output_dir,
        "inputConfig": cfg,
        "rGrid_um": r_grid * 1e6,
        "finalRadialProfile_C": tresid_radial[-1, :] - 273.15,
        "spotRadius_um": spot_radius * 1e6,
    }
