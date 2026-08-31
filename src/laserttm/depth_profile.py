"""Depth-resolved (1D) pulsed-laser TTM solver.

Python port of ``src/Depth_Profile_Solver.m`` from the MATLAB reference
implementation: coupled electron–lattice PDEs along depth (method of lines),
solved per pulse with a stiff BDF integrator (MATLAB used ode15s; SciPy's
BDF is the same solver family, so agreement with the golden fixtures is at
integrator-tolerance level rather than the bit-level of the RK4-based 0D
solver), then Crank–Nicolson diffusion with temperature-dependent k(T) on a
coarser, deeper grid between pulses.

Config fields, defaults, and the v1 result contract match the MATLAB solver
field-for-field. The MATLAB figure set (plots 1–8) is not ported yet; the
compute path, console report, and text export are complete.
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, diags

from .config import get_cfg_field
from .kernels import (
    cn_coast_kt,
    k_hybrid,
    k_table_for,
    profile_code,
    ttm_1d_rhs,
)
from .progress import ProgressReporter
from .units import smart_energy, smart_freq, smart_length, smart_time

# Material presets: gamma [J m^-3 K^-2], Cl [J m^-3 K^-1], G [W m^-3 K^-1],
#                   ke0 [W m^-1 K^-1],   kl [W m^-1 K^-1], alpha_opt [m^-1]
_PRESETS = {
    "w":  (137.3, 2.54e6, 1.65e17, 150.0, 24.0, 5.88e7),
    "cu": (98.0,  3.45e6, 0.90e17, 390.0, 11.0, 7.09e7),
    "al": (136.0, 2.42e6, 2.40e17, 220.0, 17.0, 1.22e8),
}

_DEFAULT_SNAPSHOT_DELAYS = (0.0, 0.5e-12, 1e-12, 2e-12, 5e-12, 10e-12, 50e-12, 200e-12)

# Surface inversion threshold [K], shared by the per-pulse metrics and the
# whole-run post-processing so their counts agree.
_INV_THRESHOLD_K = 0.5

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _matlab_round(x: float) -> int:
    return int(np.floor(x + 0.5))


def depth_profile_solver(cfg: dict | None = None) -> dict:
    """Run the 1D depth-resolved TTM solver. Returns the v1 results dict."""
    if cfg is None:
        cfg = {}

    make_plots = get_cfg_field(cfg, "makePlots", True)
    save_figures = get_cfg_field(cfg, "saveFigures", False)
    if save_figures:
        make_plots = True

    print("=== 1D Two-Temperature Model Pulsed Laser Calculator ===")

    # ========================  USER INPUTS  =================================
    material = get_cfg_field(cfg, "material", "W")

    gamma_manual = get_cfg_field(cfg, "gamma_manual", 137.3)
    cl_manual = get_cfg_field(cfg, "Cl_manual", 2.54e6)
    g_manual = get_cfg_field(cfg, "G_manual", 1.65e17)
    ke0_manual = get_cfg_field(cfg, "ke0_manual", 150.0)
    kl_manual = get_cfg_field(cfg, "kl_manual", 24.0)
    alpha_opt_manual = get_cfg_field(cfg, "alpha_opt_manual", 5.88e7)

    pavg = get_cfg_field(cfg, "Pavg", 40.0)
    spot_radius = get_cfg_field(cfg, "spotRadius", 100e-6)
    f_rep = get_cfg_field(cfg, "f_rep", 18e6)
    tau_fwhm = get_cfg_field(cfg, "tau_FWHM", 100e-15)
    pulse_profile_name = get_cfg_field(cfg, "pulseProfile", "gaussian")

    absorbance = get_cfg_field(cfg, "absorbance", 0.55)
    t0_c = get_cfg_field(cfg, "T0_C", 25.0)

    lz = get_cfg_field(cfg, "Lz", 1000e-9)
    nz = int(get_cfg_field(cfg, "Nz", 200))

    sim_duration = get_cfg_field(cfg, "simDuration", 100e-6)
    snapshot_delays = np.asarray(
        get_cfg_field(cfg, "snapshotDelays", _DEFAULT_SNAPSHOT_DELAYS), dtype=float
    )

    enable_radial = get_cfg_field(cfg, "enableRadialProfile", True)
    nr_radial = int(get_cfg_field(cfg, "Nr_radial", 20))
    r_max_factor = get_cfg_field(cfg, "rMax_factor", 3)
    dz_target_diff = get_cfg_field(cfg, "dzTarget_diff", 500e-9)
    n_diff = int(get_cfg_field(cfg, "Ndiff", 100))

    rel_tol = get_cfg_field(cfg, "relTol", 1e-6)
    abs_tol = get_cfg_field(cfg, "absTol", 1e-1)

    show_progress = get_cfg_field(cfg, "showProgress", None)

    # ==================  Material presets  ==================================
    key = str(material).lower()
    if key in _PRESETS:
        gamma, cl, g_ep, ke0, kl, alpha_opt = _PRESETS[key]
    elif key == "custom":
        gamma, cl, g_ep = gamma_manual, cl_manual, g_manual
        ke0, kl, alpha_opt = ke0_manual, kl_manual, alpha_opt_manual
    else:
        raise ValueError(f'Unknown material "{material}". Use W, Cu, Al, or custom.')

    # Hybrid k(T): tungsten table, constant ke0+kl otherwise
    k_tab_t, k_tab_k = k_table_for(key, ke0, kl)

    # ==================  Derived quantities  ================================
    t0 = t0_c + 273.15
    ep = pavg / f_rep
    f_peak = 2.0 * ep / (np.pi * spot_radius**2)
    eabs_areal = absorbance * f_peak
    trep = 1.0 / f_rep
    tau_eph = gamma * t0 / g_ep
    delta_opt = 1.0 / alpha_opt
    n_pulses = _matlab_round(sim_duration * f_rep)

    # Phase 2 coarse diffusion grid
    k_eff = float(k_hybrid(np.array([t0]), k_tab_t, k_tab_k)[0])
    alpha_diff = k_eff / cl
    ldiff = max(5.0 * np.sqrt(alpha_diff * sim_duration), 50e-6)
    dz_diff = dz_target_diff
    nz_diff = int(np.ceil(ldiff / dz_diff)) + 1
    ldiff = (nz_diff - 1) * dz_diff
    z_grid_diff = np.arange(nz_diff) * dz_diff

    # Fine spatial grid
    assert nz >= 3, "Need at least 3 spatial nodes."
    dz = lz / (nz - 1)
    z_grid = np.arange(nz) * dz
    depth_abs_profile = alpha_opt * np.exp(-alpha_opt * z_grid)

    if dz > delta_opt / 3:
        warnings.warn(
            f"dz = {dz * 1e9:.1f} nm > delta_opt/3 = {delta_opt / 3 * 1e9:.1f} nm. "
            "Consider increasing Nz or decreasing Lz.")

    print(f"  Material:  {str(material).upper()}")
    print(f"  Grid:      {nz} nodes, dz = {dz * 1e9:.2f} nm, depth = {lz * 1e9:.1f} nm")
    print(f"  Skin depth (1/alpha_opt): {delta_opt * 1e9:.1f} nm  "
          f"({delta_opt / dz:.1f} grid cells)")
    print(f"  ke0 = {ke0:.0f} W/mK,  kl = {kl:.0f} W/mK,  k_hybrid(T0) = {k_eff:.1f} W/mK")
    print(f"  Fluence:   {f_peak / 1e4:.4g} J/cm^2,  Absorbed: {eabs_areal / 1e4:.4g} J/cm^2")
    print(f"  tau_e-ph (at T0): {tau_eph * 1e12:.4g} ps")
    print(f"  Diffusion: k(T0)={k_eff:.1f} W/mK, alpha={alpha_diff:.3e} m^2/s, "
          f"Ldiff={ldiff * 1e6:.1f} um, dz={dz_diff * 1e9:.0f} nm")
    print(f"  Pulses:    {n_pulses}  (simDuration={sim_duration:.4g} s)")

    # Jacobian sparsity pattern: 2x2 blocks of tridiagonals
    tri = diags([np.ones(nz - 1), np.ones(nz), np.ones(nz - 1)], (-1, 0, 1))
    j_pat = bmat([[tri, tri], [tri, tri]], format="csc")

    prof_code = profile_code(pulse_profile_name)
    pulse_offset = 5.0 * tau_fwhm
    relax_max_t = min(trep * 0.9, max(200.0 * tau_fwhm, 500e-12))

    def rhs(t, y):
        return ttm_1d_rhs(t, y, nz, dz, gamma, cl, g_ep, kl,
                          k_tab_t, k_tab_k, depth_abs_profile,
                          n_pulses, trep, pulse_offset, prof_code,
                          tau_fwhm, eabs_areal)

    # ==================  Pulse-by-pulse two-phase simulation  ===============
    print(f"  Running two-phase simulation ({n_pulses} pulses, {2 * nz} fine ODEs)...")

    cell_times_fine: list[np.ndarray] = []
    cell_te_fine: list[np.ndarray] = []
    cell_tl_fine: list[np.ndarray] = []
    cell_coast_t: list[np.ndarray] = []
    cell_coast_tl: list[np.ndarray] = []
    teq_vals = np.zeros(n_pulses)
    tresid_vals = np.zeros(n_pulses)
    te_peak_per_pulse = np.zeros(n_pulses)
    tl_peak_per_pulse = np.zeros(n_pulses)

    inv_threshold_pp = _INV_THRESHOLD_K
    inv_max_per_pulse = np.zeros(n_pulses)
    t_max_inv_per_pulse = np.full(n_pulses, np.nan)
    t_inv_onset_per_pulse = np.full(n_pulses, np.nan)
    inv_duration_per_pulse = np.zeros(n_pulses)
    te_at_max_inv_per_pulse = np.full(n_pulses, np.nan)
    tl_at_max_inv_per_pulse = np.full(n_pulses, np.nan)
    base_temp_per_pulse = np.zeros(n_pulses)

    # Spatial snapshots (first pulse only; used by the pending plot port)
    snap_te: list[np.ndarray] = []
    snap_tl: list[np.ndarray] = []
    snap_labels: list[str] = []

    tz_diff = t0 * np.ones(nz_diff)

    n_profile_snaps = min(n_pulses, 12)
    if n_pulses > 1:
        profile_snap_pulses = np.unique(np.round(
            np.logspace(0, np.log10(n_pulses), n_profile_snaps)).astype(int))
    else:
        profile_snap_pulses = np.array([1])
    profile_snaps_tz: list[np.ndarray] = []
    profile_snaps_label: list[str] = []

    # Loop-invariant fine-to-coarse grid overlap (depends only on the grids)
    overlap_mask = z_grid_diff <= lz
    overlap_any = bool(overlap_mask.any())
    z_diff_overlap = z_grid_diff[overlap_mask]
    beyond = np.flatnonzero(~overlap_mask)
    first_beyond = int(beyond[0]) if beyond.size > 0 and beyond[0] > 0 else -1

    n_coast_sample = min(n_diff, 50)
    sample_interval = max(1, n_diff // n_coast_sample)
    progress_interval = max(1, n_pulses // 20)

    tic_all = time.perf_counter()
    progress = ProgressReporter(n_pulses, title="laserttm: depth profile",
                                enabled=show_progress)

    for np_i in range(1, n_pulses + 1):
        t_pulse_center = pulse_offset + (np_i - 1) * trep

        # ---------------- PHASE 1: full 1D TTM on the fine grid ------------
        t_p1_start = t_pulse_center - 5.0 * tau_fwhm
        if np_i == 1:
            t_p1_start = 0.0
        t_p1_end = t_pulse_center + relax_max_t

        t_fine_init = np.interp(z_grid, z_grid_diff, tz_diff)
        y_current = np.concatenate([t_fine_init, t_fine_init])

        if np_i == 1:
            n_dense, n_log_pts = 150, 200
        else:
            n_dense, n_log_pts = 40, 50
        t_during = np.linspace(t_pulse_center - 5.0 * tau_fwhm,
                               t_pulse_center + 20.0 * tau_fwhm, n_dense)
        post_delay = 20.0 * tau_fwhm
        coast_len = t_p1_end - t_pulse_center
        if coast_len > post_delay:
            t_log = t_pulse_center + np.logspace(
                np.log10(post_delay), np.log10(coast_len), n_log_pts)
        else:
            t_log = np.empty(0)
        t_out = np.unique(np.concatenate(
            [[t_p1_start], t_during, t_log, [t_p1_end]]))
        t_out = t_out[(t_out >= t_p1_start) & (t_out <= t_p1_end)]

        sol = solve_ivp(rhs, (t_out[0], t_out[-1]), y_current,
                        method="BDF", t_eval=t_out,
                        rtol=rel_tol, atol=abs_tol,
                        max_step=tau_fwhm, jac_sparsity=j_pat)
        if not sol.success:
            raise RuntimeError(f"BDF integration failed on pulse {np_i}: {sol.message}")
        t_sol = sol.t
        y_sol = sol.y.T  # (nt, 2*nz)

        te_s = y_sol[:, 0]
        tl_s = y_sol[:, nz]

        cell_times_fine.append(t_sol)
        cell_te_fine.append(te_s)
        cell_tl_fine.append(tl_s)
        te_peak_per_pulse[np_i - 1] = te_s.max()
        tl_peak_per_pulse[np_i - 1] = tl_s.max()
        base_temp_per_pulse[np_i - 1] = te_s[0]

        # --- Per-pulse inversion metrics (with sub-step interpolation) ---
        d_t_inv = tl_s - te_s
        max_inv_idx = int(np.argmax(d_t_inv))
        max_inv = d_t_inv[max_inv_idx]
        inv_max_per_pulse[np_i - 1] = max_inv
        if max_inv > inv_threshold_pp:
            inv_indices = np.flatnonzero(d_t_inv > inv_threshold_pp)

            # Refine peak time via parabolic interpolation
            t_max_raw = t_sol[max_inv_idx]
            if 0 < max_inv_idx < d_t_inv.size - 1:
                t3 = t_sol[max_inv_idx - 1: max_inv_idx + 2]
                d3 = d_t_inv[max_inv_idx - 1: max_inv_idx + 2]
                denom = (t3[0] - t3[1]) * (t3[0] - t3[2]) * (t3[1] - t3[2])
                if abs(denom) > 0:
                    a_coef = (t3[2] * (d3[1] - d3[0]) + t3[1] * (d3[0] - d3[2])
                              + t3[0] * (d3[2] - d3[1])) / denom
                    b_coef = (t3[2]**2 * (d3[0] - d3[1]) + t3[1]**2 * (d3[2] - d3[0])
                              + t3[0]**2 * (d3[1] - d3[2])) / denom
                    if abs(a_coef) > 0:
                        t_peak_interp = -b_coef / (2.0 * a_coef)
                        if t3[0] <= t_peak_interp <= t3[2]:
                            t_max_raw = t_peak_interp
                            max_inv = (a_coef * t_max_raw**2 + b_coef * t_max_raw
                                       + d3[0] - a_coef * t3[0]**2 - b_coef * t3[0])
                            inv_max_per_pulse[np_i - 1] = max(
                                max_inv, inv_max_per_pulse[np_i - 1])
            t_max_inv_per_pulse[np_i - 1] = t_max_raw - t_pulse_center

            # Refine onset time via linear interpolation at threshold
            i_first = inv_indices[0]
            if i_first > 0:
                t1, t2 = t_sol[i_first - 1], t_sol[i_first]
                d1, d2 = d_t_inv[i_first - 1], d_t_inv[i_first]
                frac = np.clip((inv_threshold_pp - d1) / (d2 - d1), 0.0, 1.0)
                t_onset_interp = t1 + frac * (t2 - t1)
            else:
                t_onset_interp = t_sol[i_first]
            t_inv_onset_per_pulse[np_i - 1] = t_onset_interp - t_pulse_center

            # Refine end time via linear interpolation at threshold
            i_last = inv_indices[-1]
            if i_last < d_t_inv.size - 1:
                t1, t2 = t_sol[i_last], t_sol[i_last + 1]
                d1, d2 = d_t_inv[i_last], d_t_inv[i_last + 1]
                frac = np.clip((inv_threshold_pp - d1) / (d2 - d1), 0.0, 1.0)
                t_end_interp = t1 + frac * (t2 - t1)
            else:
                t_end_interp = t_sol[i_last]
            inv_duration_per_pulse[np_i - 1] = t_end_interp - t_onset_interp

            # Te/Tl at the (refined) max-inversion instant
            te_at_max_inv_per_pulse[np_i - 1] = np.interp(t_max_raw, t_sol, te_s)
            tl_at_max_inv_per_pulse[np_i - 1] = np.interp(t_max_raw, t_sol, tl_s)

        # Spatial snapshots (first pulse only)
        if np_i == 1:
            for delay in snapshot_delays:
                t_snap = t_pulse_center + delay
                if t_p1_start <= t_snap <= t_sol[-1]:
                    idx = int(np.argmin(np.abs(t_sol - t_snap)))
                    snap_te.append(y_sol[idx, :nz].copy())
                    snap_tl.append(y_sol[idx, nz:].copy())
                    dv, du = smart_time(t_sol[idx] - t_pulse_center)
                    snap_labels.append(f"{dv:.3g} {du}")

        # --- Map fine-grid end state back onto the coarse diffusion grid ---
        te_end_fine = y_sol[-1, :nz]
        tl_end_fine = y_sol[-1, nz:]
        t_equil_fine = 0.5 * (te_end_fine + tl_end_fine)

        utot = _trapezoid(0.5 * gamma * te_end_fine**2 + cl * tl_end_fine, z_grid) / lz
        teq = (-cl + np.sqrt(cl**2 + 2.0 * gamma * utot)) / gamma
        teq_vals[np_i - 1] = teq

        if overlap_any:
            # np.interp clamps to the end value, matching the MATLAB
            # interp1(..., 'linear', T_equil_fine(end)) extrapolation here.
            tz_diff[overlap_mask] = np.interp(
                z_diff_overlap, z_grid, t_equil_fine)
        if first_beyond > 0:
            tz_diff[first_beyond] = 0.5 * (tz_diff[first_beyond - 1]
                                           + tz_diff[first_beyond])

        # ---------------- PHASE 2: CN diffusion on the coarse grid ----------
        t_fine_end = t_sol[-1]
        if np_i < n_pulses:
            t_next_pulse_start = pulse_offset + np_i * trep - 5.0 * tau_fwhm
        else:
            t_next_pulse_start = sim_duration
        coast_gap = t_next_pulse_start - t_fine_end

        if coast_gap > 0:
            tz_diff, c_t, c_tl = cn_coast_kt(
                tz_diff, coast_gap, n_diff, dz_diff, t0, cl,
                k_tab_t, k_tab_k, t_fine_end, sample_interval)
            cell_coast_t.append(c_t)
            cell_coast_tl.append(c_tl)
            tresidual = tz_diff[0]
        else:
            cell_coast_t.append(np.empty(0))
            cell_coast_tl.append(np.empty(0))
            tresidual = teq

        tresid_vals[np_i - 1] = tresidual

        if np_i in profile_snap_pulses:
            profile_snaps_tz.append(tz_diff.copy())
            tv, tu = smart_time(np_i * trep)
            profile_snaps_label.append(f"Pulse {np_i} ({tv:.3g} {tu})")

        if np_i % progress_interval == 0 or np_i == n_pulses:
            print(f"    Pulse {np_i}/{n_pulses}: "
                  f"Te_peak={te_peak_per_pulse[np_i - 1] - 273.15:.0f} degC, "
                  f"Teq={teq - 273.15:.1f} degC, Tresid={tresidual - 273.15:.1f} degC  "
                  f"({t_sol.size} fine + {cell_coast_t[-1].size} coast)")
        progress.update(np_i)

    progress.close()
    wall_time = time.perf_counter() - tic_all
    print(f"  Simulation wall time: {wall_time:.2f} s")

    # --- Stitch per-pulse arrays into global vectors ---
    all_times = np.concatenate([
        arr for pair in zip(cell_times_fine, cell_coast_t) for arr in pair])
    all_te_surf = np.concatenate([
        arr for pair in zip(cell_te_fine, cell_coast_tl) for arr in pair])
    all_tl_surf = np.concatenate([
        arr for pair in zip(cell_tl_fine, cell_coast_tl) for arr in pair])
    pulse_start_idx = np.zeros(n_pulses, dtype=np.int64)
    pulse_end_idx = np.zeros(n_pulses, dtype=np.int64)
    ptr = 0
    for p in range(n_pulses):
        pulse_start_idx[p] = ptr
        ptr += cell_times_fine[p].size
        pulse_end_idx[p] = ptr - 1
        ptr += cell_coast_t[p].size

    # ==================  Post-processing  ===================================
    d_t_surf = all_tl_surf - all_te_surf
    inversion_mask = d_t_surf > _INV_THRESHOLD_K
    first_pulse_center = pulse_offset

    if inversion_mask.any():
        inv_idx = np.flatnonzero(inversion_mask)
        max_inv_rel_idx = int(np.argmax(d_t_surf[inv_idx]))
        max_inv_all = d_t_surf[inv_idx][max_inv_rel_idx]
        max_inv_idx_all = inv_idx[max_inv_rel_idx]
        t_inv_onset = all_times[inv_idx[0]] - first_pulse_center
        t_inv_max = all_times[max_inv_idx_all] - first_pulse_center
        te_at_max_inv = all_te_surf[max_inv_idx_all]
        tl_at_max_inv = all_tl_surf[max_inv_idx_all]
        inv_detected = True

        on_v, on_u = smart_time(t_inv_onset)
        mx_v, mx_u = smart_time(t_inv_max)
        print("\n  ** SURFACE TEMPERATURE INVERSION DETECTED (Tl > Te) **")
        print(f"  Onset:          {on_v:.4g} {on_u} after pulse center")
        print(f"  Max (Tl-Te):    {max_inv_all:.1f} degC  at  {mx_v:.4g} {mx_u}")
        print(f"  At max inv:     Te = {te_at_max_inv - 273.15:.0f} degC,  "
              f"Tl = {tl_at_max_inv - 273.15:.0f} degC")
    else:
        inv_detected = False
        print("\n  No significant temperature inversion detected.")

    idx_te_peak = int(np.argmax(all_te_surf))
    te_peak_all = all_te_surf[idx_te_peak]
    tl_peak_all = all_tl_surf.max()
    t_peak_te_rel = all_times[idx_te_peak] - first_pulse_center
    pk_te_v, pk_te_u = smart_time(t_peak_te_rel)

    peak_pulse = 1
    for p in range(n_pulses):
        if pulse_start_idx[p] <= idx_te_peak <= pulse_end_idx[p]:
            peak_pulse = p + 1
            break

    e_input = n_pulses * eabs_areal
    du_depth = cl * _trapezoid(tz_diff - t0, z_grid_diff)

    # ==================  Print results  =====================================
    print("\n============================================================")
    print("  1D TTM Pulsed Laser Calculator — Results")
    print("============================================================")
    print(f"  Material:                 {str(material).upper()}")
    print(f"  gamma  [J m^-3 K^-2]:    {gamma:.2f}")
    print(f"  Cl     [J m^-3 K^-1]:    {cl:.4e}")
    print(f"  G      [W m^-3 K^-1]:    {g_ep:.4e}")
    print(f"  ke0    [W m^-1 K^-1]:    {ke0:.1f}")
    print(f"  kl     [W m^-1 K^-1]:    {kl:.1f}")
    print(f"  alpha_opt [m^-1]:        {alpha_opt:.4e}  (skin depth {delta_opt * 1e9:.1f} nm)")
    print("------------------------------------------------------------")
    frep_v, frep_u = smart_freq(f_rep)
    tau_v, tau_u = smart_time(tau_fwhm)
    ep_v, ep_u = smart_energy(ep)
    spot_v, spot_u = smart_length(spot_radius)
    lz_v, lz_u = smart_length(lz)
    dz_v, dz_u = smart_length(dz)
    ldiff_v, ldiff_u = smart_length(ldiff)
    dz_diff_v, dz_diff_u = smart_length(dz_diff)
    sim_dur_v, sim_dur_u = smart_time(sim_duration)
    print(f"  Avg Power:               {pavg:.3g} W")
    print(f"  Rep Rate:                {frep_v:.4g} {frep_u}")
    print(f"  Pulse Energy:            {ep_v:.4g} {ep_u}")
    print(f"  Pulse Width (FWHM):      {tau_v:.4g} {tau_u}")
    print(f"  Spot Radius:             {spot_v:.4g} {spot_u}")
    print(f"  Fluence (peak):          {f_peak / 1e4:.5g} J/cm^2")
    print(f"  Absorbance:              {absorbance:.2f}")
    print("------------------------------------------------------------")
    print(f"  Fine TTM grid:           {lz_v:.1f} {lz_u}  ({nz} nodes, dz={dz_v:.1f} {dz_u})")
    print(f"  Diffusion grid:          {ldiff_v:.1f} {ldiff_u}  "
          f"({nz_diff} nodes, dz={dz_diff_v:.1f} {dz_diff_u})")
    print(f"  Diff steps/period:       {n_diff}")
    print(f"  Pulses simulated:        {n_pulses}")
    print(f"  Simulation duration:     {sim_dur_v:.4g} {sim_dur_u}")
    print(f"  Total time points:       {all_times.size}")
    print(f"  Wall time:               {wall_time:.2f} s")
    print("------------------------------------------------------------")
    print(f"  Peak surface Te:         {te_peak_all - 273.15:.1f} degC  "
          f"(pulse {peak_pulse}, at {pk_te_v:.4g} {pk_te_u})")
    print(f"  Peak surface Tl:         {tl_peak_all - 273.15:.1f} degC")
    print(f"  Final surface Te:        {all_te_surf[-1] - 273.15:.2f} degC")
    print(f"  Final surface Tl:        {all_tl_surf[-1] - 273.15:.2f} degC")
    print(f"  Final residual (surf):   {tresid_vals[-1] - 273.15:.2f} degC  (after diffusion)")
    print("------------------------------------------------------------")
    if inv_detected:
        print(f"  Inversion onset:         {on_v:.4g} {on_u}")
        print(f"  Max inversion (Tl-Te):   {max_inv_all:.1f} degC  at {mx_v:.4g} {mx_u}")
    print(f"  E_absorbed (areal):      {e_input:.4g} J/m^2")
    print(f"  E_depth    (areal):      {du_depth:.4g} J/m^2")
    print("  (Note: mismatch expected in hybrid fine+coarse model)")
    print("============================================================\n")

    # ==================  Export results to file  ============================
    default_out = os.path.join(os.getcwd(), "outputs")
    output_dir = get_cfg_field(cfg, "outputDir", default_out)
    os.makedirs(output_dir, exist_ok=True)

    freq_str = (f"{frep_v:.4g}_{frep_u}").replace(".", "p")
    pulse_str = (f"{tau_v:.4g}_{tau_u}").replace(".", "p")
    power_str = (f"{pavg:.4g}_W").replace(".", "p")
    spot_str = (f"{spot_v:.4g}_{spot_u}").replace(".", "p")
    out_filename = (f"TTM_1D_Result_{freq_str}_{pulse_str}_{power_str}_"
                    f"{spot_str}_{n_pulses}p_{pulse_profile_name}.txt")
    case_tag = get_cfg_field(cfg, "caseTag", "")
    if case_tag:
        out_filename = f"{case_tag}__{out_filename}"
    out_path = os.path.join(output_dir, out_filename)

    with open(out_path, "w") as fid:
        fid.write("============================================================\n")
        fid.write("  1D TTM Pulsed Laser Calculator — Output\n")
        # Local wall-clock on purpose, matching the MATLAB reference output
        fid.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")  # noqa: DTZ005
        fid.write("============================================================\n\n")
        fid.write(f"--- Material:  {str(material).upper()} ---\n")
        fid.write(f"  gamma  = {gamma:.2f}  J m^-3 K^-2\n")
        fid.write(f"  Cl     = {cl:.4e}  J m^-3 K^-1\n")
        fid.write(f"  G      = {g_ep:.4e}  W m^-3 K^-1\n")
        fid.write(f"  ke0    = {ke0:.1f}  W m^-1 K^-1\n")
        fid.write(f"  kl     = {kl:.1f}  W m^-1 K^-1\n")
        fid.write(f"  alpha  = {alpha_opt:.4e}  m^-1  (skin depth {delta_opt * 1e9:.1f} nm)\n")
        fid.write("\n--- Laser ---\n")
        fid.write(f"  Average Power:    {pavg:.4g} W\n")
        fid.write(f"  Rep Rate:         {frep_v:.4g} {frep_u}\n")
        fid.write(f"  Pulse Energy:     {ep_v:.4g} {ep_u}\n")
        fid.write(f"  Pulse Width:      {tau_v:.4g} {tau_u}\n")
        fid.write(f"  Spot Radius:      {spot_v:.4g} {spot_u}\n")
        fid.write(f"  Fluence (peak):   {f_peak / 1e4:.5g} J/cm^2\n")
        fid.write(f"  Absorbance:       {absorbance:.2f}\n")
        fid.write(f"  Profile:          {pulse_profile_name}\n")
        fid.write("\n--- Grid & Solver ---\n")
        fid.write(f"  Fine TTM grid:    {lz_v:.4g} {lz_u}  ({nz} nodes, dz={dz_v:.2f} {dz_u})\n")
        fid.write(f"  Diffusion grid:   {ldiff_v:.4g} {ldiff_u}  "
                  f"({nz_diff} nodes, dz={dz_diff_v:.1f} {dz_diff_u})\n")
        fid.write(f"  Diff steps/period: {n_diff}\n")
        fid.write(f"  Pulses:           {n_pulses}\n")
        fid.write(f"  Sim duration:     {sim_dur_v:.4g} {sim_dur_u}\n")
        fid.write(f"  Time points:      {all_times.size}\n")
        fid.write(f"  Wall time:        {wall_time:.2f} s\n")
        fid.write("\n--- Results ---\n")
        fid.write(f"  Peak surface Te:  {te_peak_all - 273.15:.1f} degC  (pulse {peak_pulse})\n")
        fid.write(f"  Peak surface Tl:  {tl_peak_all - 273.15:.1f} degC\n")
        fid.write(f"  Final surface Te: {all_te_surf[-1] - 273.15:.2f} degC\n")
        fid.write(f"  Final surface Tl: {all_tl_surf[-1] - 273.15:.2f} degC\n")
        fid.write(f"  Final residual:   {tresid_vals[-1] - 273.15:.2f} degC  (after diffusion)\n")
        if inv_detected:
            fid.write(f"  Inversion onset:  {on_v:.4g} {on_u}\n")
            fid.write(f"  Max Tl-Te:        {max_inv_all:.1f} degC at {mx_v:.4g} {mx_u}\n")
        fid.write(f"  E_absorbed:       {e_input:.4g} J/m^2\n")
        fid.write(f"  E_depth:          {du_depth:.4g} J/m^2\n")
        fid.write("  (Note: mismatch expected in hybrid fine+coarse model)\n\n")
        for p in range(n_pulses):
            fid.write(f"  Pulse {p + 1}: Teq={teq_vals[p] - 273.15:.2f} degC, "
                      f"Tresid={tresid_vals[p] - 273.15:.2f} degC\n")
        fid.write("\n============================================================\n")
        fid.write("  Surface XY Data: Time (s) | Te_surf (degC) | Tl_surf (degC)\n")
        fid.write("============================================================\n")
        fid.write(f"{'Time_s':>20}  {'Te_surf_degC':>16}  {'Tl_surf_degC':>16}\n")
        fid.writelines(
            f"{all_times[i]:20.12e}  {all_te_surf[i] - 273.15:16.6f}  "
            f"{all_tl_surf[i] - 273.15:16.6f}\n" for i in range(all_times.size))
    print(f"  Output written to: {out_path}\n")

    if make_plots:
        from .plotting import plot_depth_profile

        plot_depth_profile(
            all_times=all_times, all_tl_surf=all_tl_surf,
            teq_vals=teq_vals, tresid_vals=tresid_vals,
            snap_te=snap_te, snap_tl=snap_tl, snap_labels=snap_labels,
            z_grid=z_grid, lz=lz,
            profile_snaps_tz=profile_snaps_tz,
            profile_snaps_label=profile_snaps_label,
            z_grid_diff=z_grid_diff, ldiff=ldiff, t0=t0,
            enable_radial=enable_radial, nr_radial=nr_radial,
            r_max_factor=r_max_factor, spot_radius=spot_radius,
            alpha_diff=alpha_diff, sim_duration=sim_duration,
            material=material, gamma=gamma, cl=cl, g_ep=g_ep,
            ke0=ke0, kl=kl, alpha_opt=alpha_opt, delta_opt=delta_opt,
            pavg=pavg, frep_v=frep_v, frep_u=frep_u, ep_v=ep_v, ep_u=ep_u,
            tau_v=tau_v, tau_u=tau_u, spot_v=spot_v, spot_u=spot_u,
            f_peak=f_peak, absorbance=absorbance,
            n_pulses=n_pulses, te_peak_all=te_peak_all,
            tl_peak_all=tl_peak_all, peak_pulse=peak_pulse,
            tresid_last=tresid_vals[-1], e_input=e_input, du_depth=du_depth,
            inv_detected=inv_detected,
            max_inv=(max_inv_all if inv_detected else 0.0),
            mx_v=(mx_v if inv_detected else 0.0),
            mx_u=(mx_u if inv_detected else ""),
            save_dir=(output_dir if save_figures else None),
            case_tag=case_tag)

    print("Done.")

    return {
        "solver": "1D",
        "solverId": "depth_profile",
        "contractVersion": "v1",
        "material": material,
        "nPulses": n_pulses,
        "peakTe_C": te_peak_all - 273.15,
        "peakTl_C": tl_peak_all - 273.15,
        "finalResid_C": tresid_vals[-1] - 273.15,
        "wallTime_s": wall_time,
        "outputFile": out_path,
        "outputDir": output_dir,
        "inputConfig": cfg,
        # Per-pulse arrays (degC or seconds)
        "TePeakPerPulse_C": te_peak_per_pulse - 273.15,
        "TlPeakPerPulse_C": tl_peak_per_pulse - 273.15,
        "TeqVals_C": teq_vals - 273.15,
        "TresidVals_C": tresid_vals - 273.15,
        "baseTempPerPulse_C": base_temp_per_pulse - 273.15,
        # Per-pulse inversion data
        "invMaxPerPulse_K": inv_max_per_pulse,
        "tMaxInvPerPulse_s": t_max_inv_per_pulse,
        "tInvOnsetPerPulse_s": t_inv_onset_per_pulse,
        "invDurationPerPulse_s": inv_duration_per_pulse,
        "Te_atMaxInvPerPulse_C": te_at_max_inv_per_pulse - 273.15,
        "Tl_atMaxInvPerPulse_C": tl_at_max_inv_per_pulse - 273.15,
        # Input parameters (for downstream analysis)
        "f_rep": f_rep,
        "Pavg": pavg,
        "tau_FWHM": tau_fwhm,
        "spotRadius": spot_radius,
        "absorbance": absorbance,
        "T0_C": t0_c,
        "F_peak": f_peak,
        "gamma": gamma,
        "Cl": cl,
        "G": g_ep,
        "ke0": ke0,
        "kl": kl,
        "alpha_opt": alpha_opt,
        "Trep": trep,
        "simDuration": sim_duration,
    }
