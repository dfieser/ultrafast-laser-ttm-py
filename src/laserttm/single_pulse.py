"""Single-pulse 1D TTM solver with spatial snapshots.

Python port of ``src/Single_Pulse_Visualizer.m``: one pulse on the fine 1D
grid (method of lines + stiff BDF, matching ode15s at tolerance level), with
depth-profile snapshots at chosen delays and surface-inversion metrics.

Unlike the depth solver, the electron conductivity here uses a constant
``ke0`` (``ke = ke0*Te/Tl``, no hybrid k(T) table) — reproduced exactly by
handing the shared RHS kernel a flat conductivity table of ke0 + kl.
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
from .kernels import profile_code, ttm_1d_rhs
from .units import smart_energy, smart_freq, smart_length, smart_time

# gamma, Cl, G, ke0, kl, alpha_opt — same table as the depth solver
_PRESETS = {
    "w":  (137.3, 2.54e6, 1.65e17, 150.0, 24.0, 5.88e7),
    "cu": (98.0,  3.45e6, 0.90e17, 390.0, 11.0, 7.09e7),
    "al": (136.0, 2.42e6, 2.40e17, 220.0, 17.0, 1.22e8),
}

_DEFAULT_SNAPSHOT_DELAYS = (0.0, 0.5e-12, 1e-12, 2e-12, 5e-12, 10e-12, 50e-12, 200e-12)

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def single_pulse_visualizer(cfg: dict | None = None) -> dict:
    """Run the single-pulse 1D TTM solver. Returns the v1 results dict."""
    if cfg is None:
        cfg = {}

    make_plots = get_cfg_field(cfg, "makePlots", True)
    save_figures = get_cfg_field(cfg, "saveFigures", False)
    if save_figures:
        make_plots = True

    print("=== 1D Two-Temperature Model — Single Pulse Visualizer ===")

    material = get_cfg_field(cfg, "material", "W")
    gamma_manual = get_cfg_field(cfg, "gamma_manual", 137.3)
    cl_manual = get_cfg_field(cfg, "Cl_manual", 2.54e6)
    g_manual = get_cfg_field(cfg, "G_manual", 1.65e17)
    ke0_manual = get_cfg_field(cfg, "ke0_manual", 150.0)
    kl_manual = get_cfg_field(cfg, "kl_manual", 24.0)
    alpha_opt_manual = get_cfg_field(cfg, "alpha_opt_manual", 5.88e7)

    pavg = get_cfg_field(cfg, "Pavg", 1.0)
    spot_radius = get_cfg_field(cfg, "spotRadius", 80e-6)
    f_rep = get_cfg_field(cfg, "f_rep", 1e6)
    tau_fwhm = get_cfg_field(cfg, "tau_FWHM", 100e-15)
    pulse_profile_name = get_cfg_field(cfg, "pulseProfile", "gaussian")

    absorbance = get_cfg_field(cfg, "absorbance", 0.55)
    t0_c = get_cfg_field(cfg, "T0_C", 25.0)

    lz = get_cfg_field(cfg, "Lz", 1000e-9)
    nz = int(get_cfg_field(cfg, "Nz", 200))

    n_pulses = 1  # always 1 for the single-pulse visualizer

    snapshot_delays = np.asarray(
        get_cfg_field(cfg, "snapshotDelays", _DEFAULT_SNAPSHOT_DELAYS), dtype=float)

    rel_tol = get_cfg_field(cfg, "relTol", 1e-6)
    abs_tol = get_cfg_field(cfg, "absTol", 1e-1)

    key = str(material).lower()
    if key in _PRESETS:
        gamma, cl, g_ep, ke0, kl, alpha_opt = _PRESETS[key]
    elif key == "custom":
        gamma, cl, g_ep = gamma_manual, cl_manual, g_manual
        ke0, kl, alpha_opt = ke0_manual, kl_manual, alpha_opt_manual
    else:
        raise ValueError(f'Unknown material "{material}". Use W, Cu, Al, or custom.')

    # Constant-ke0 electron conductivity: flat table makes the shared RHS
    # kernel's ke0_local = (ke0+kl) - kl = ke0 at every temperature.
    k_tab_t = np.array([1.0, 1e12])
    k_tab_k = np.array([ke0 + kl, ke0 + kl])

    t0 = t0_c + 273.15
    ep = pavg / f_rep
    f_peak = 2.0 * ep / (np.pi * spot_radius**2)
    eabs_areal = absorbance * f_peak
    trep = 1.0 / f_rep
    tau_eph = gamma * t0 / g_ep
    delta_opt = 1.0 / alpha_opt

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
    print(f"  ke0 = {ke0:.0f} W/mK,  kl = {kl:.0f} W/mK")
    print(f"  Fluence:   {f_peak / 1e4:.4g} J/cm^2,  Absorbed: {eabs_areal / 1e4:.4g} J/cm^2")
    print(f"  tau_e-ph (at T0): {tau_eph * 1e12:.4g} ps")

    tri = diags([np.ones(nz - 1), np.ones(nz), np.ones(nz - 1)], (-1, 0, 1))
    j_pat = bmat([[tri, tri], [tri, tri]], format="csc")
    prof_code = profile_code(pulse_profile_name)
    pulse_offset = 5.0 * tau_fwhm

    def rhs(t, y):
        return ttm_1d_rhs(t, y, nz, dz, gamma, cl, g_ep, kl,
                          k_tab_t, k_tab_k, depth_abs_profile,
                          n_pulses, trep, pulse_offset, prof_code,
                          tau_fwhm, eabs_areal)

    print(f"  Simulating {n_pulses} pulse(s)  ({2 * nz} ODEs)...")

    sim_end_time = pulse_offset + (n_pulses - 1) * trep + max(trep, 500e-12)

    all_times_l: list[np.ndarray] = []
    all_te_l: list[np.ndarray] = []
    all_tl_l: list[np.ndarray] = []
    snap_te: list[np.ndarray] = []
    snap_tl: list[np.ndarray] = []
    snap_labels: list[str] = []
    te_peak_per_pulse = np.zeros(n_pulses)
    tl_peak_per_pulse = np.zeros(n_pulses)
    tresid_per_pulse = np.zeros(n_pulses)

    y_current = np.concatenate([t0 * np.ones(nz), t0 * np.ones(nz)])
    tic_all = time.perf_counter()

    for np_i in range(1, n_pulses + 1):
        t_pulse_center = pulse_offset + (np_i - 1) * trep

        t_period_start = t_pulse_center - 5.0 * tau_fwhm
        if np_i == 1:
            t_period_start = 0.0
        if np_i < n_pulses:
            t_period_end = pulse_offset + np_i * trep - 5.0 * tau_fwhm
        else:
            t_period_end = sim_end_time

        t_during = np.linspace(t_pulse_center - 5.0 * tau_fwhm,
                               t_pulse_center + 20.0 * tau_fwhm, 150)
        post_delay = 20.0 * tau_fwhm
        coast_len = t_period_end - t_pulse_center
        if coast_len > post_delay:
            t_log = t_pulse_center + np.logspace(
                np.log10(post_delay), np.log10(coast_len), 400)
        else:
            t_log = np.empty(0)
        t_out = np.unique(np.concatenate(
            [[t_period_start], t_during, t_log, [t_period_end]]))
        t_out = t_out[(t_out >= t_period_start) & (t_out <= t_period_end)]

        # The source is identically zero beyond 10*tau from the pulse center
        # (kernel cutoff), so the step cap that keeps the integrator from
        # missing the femtosecond pulse is only needed through the source
        # window.  scipy's BDF picks its initial step from the RHS at the
        # start of the span, which is ~zero before the pulse, and with no cap
        # it would leap over the pulse entirely (MATLAB's ode15s avoids this
        # by bounding the initial step with the output-time spacing).  The
        # relaxation tail is integrated uncapped: there the RHS is genuinely
        # active, so adaptive step selection is safe — and capping the full
        # inter-pulse span at tau would take >1e5 steps.
        t_cap_end = t_pulse_center + 20.0 * tau_fwhm
        if t_cap_end >= t_out[-1]:
            segments = [(t_out, tau_fwhm)]
        else:
            mask_a = t_out <= t_cap_end
            t_out_a = t_out[mask_a]
            if t_out_a[-1] < t_cap_end:
                t_out_a = np.append(t_out_a, t_cap_end)
            t_out_b = np.concatenate([[t_cap_end], t_out[~mask_a]])
            segments = [(t_out_a, tau_fwhm), (t_out_b, np.inf)]

        seg_t: list[np.ndarray] = []
        seg_y: list[np.ndarray] = []
        y_seg = y_current
        for t_eval_seg, max_step_seg in segments:
            sol = solve_ivp(rhs, (t_eval_seg[0], t_eval_seg[-1]), y_seg,
                            method="BDF", t_eval=t_eval_seg,
                            rtol=rel_tol, atol=abs_tol,
                            max_step=max_step_seg, jac_sparsity=j_pat)
            if not sol.success:
                raise RuntimeError(f"BDF integration failed: {sol.message}")
            keep = slice(1, None) if seg_t else slice(None)
            seg_t.append(sol.t[keep])
            seg_y.append(sol.y.T[keep])
            y_seg = sol.y[:, -1]
        t_sol = np.concatenate(seg_t)
        y_sol = np.vstack(seg_y)

        te_s = y_sol[:, 0]
        tl_s = y_sol[:, nz]
        all_times_l.append(t_sol)
        all_te_l.append(te_s)
        all_tl_l.append(tl_s)

        if np_i == 1:
            for delay in snapshot_delays:
                t_snap = t_pulse_center + delay
                if t_period_start <= t_snap <= t_period_end:
                    idx = int(np.argmin(np.abs(t_sol - t_snap)))
                    snap_te.append(y_sol[idx, :nz].copy())
                    snap_tl.append(y_sol[idx, nz:].copy())
                    dv, du = smart_time(t_sol[idx] - t_pulse_center)
                    snap_labels.append(f"{dv:.3g} {du}")

        te_peak_per_pulse[np_i - 1] = te_s.max()
        tl_peak_per_pulse[np_i - 1] = tl_s.max()
        tresid_per_pulse[np_i - 1] = tl_s[-1]
        y_current = y_sol[-1].copy()

        print(f"    Pulse {np_i}/{n_pulses}:  "
              f"Te_peak = {te_peak_per_pulse[np_i - 1] - 273.15:.0f} degC,  "
              f"Tl_surf_end = {tresid_per_pulse[np_i - 1] - 273.15:.1f} degC")

    wall_time = time.perf_counter() - tic_all
    print(f"  Simulation wall time: {wall_time:.2f} s")

    all_times = np.concatenate(all_times_l)
    all_te_surf = np.concatenate(all_te_l)
    all_tl_surf = np.concatenate(all_tl_l)

    # ==================  Post-processing  ===================================
    d_t_surf = all_tl_surf - all_te_surf
    inv_threshold = 0.5
    inversion_mask = d_t_surf > inv_threshold
    first_pulse_center = pulse_offset

    if inversion_mask.any():
        inv_idx = np.flatnonzero(inversion_mask)
        max_inv_rel = int(np.argmax(d_t_surf[inv_idx]))
        max_inv = d_t_surf[inv_idx][max_inv_rel]
        max_inv_idx = inv_idx[max_inv_rel]
        t_inv_onset = all_times[inv_idx[0]] - first_pulse_center
        t_inv_max = all_times[max_inv_idx] - first_pulse_center
        te_at_max_inv = all_te_surf[max_inv_idx]
        tl_at_max_inv = all_tl_surf[max_inv_idx]
        inv_detected = True

        on_v, on_u = smart_time(t_inv_onset)
        mx_v, mx_u = smart_time(t_inv_max)
        print("\n  ** SURFACE TEMPERATURE INVERSION DETECTED (Tl > Te) **")
        print(f"  Onset:          {on_v:.4g} {on_u} after pulse center")
        print(f"  Max (Tl-Te):    {max_inv:.1f} degC  at  {mx_v:.4g} {mx_u}")
        print(f"  At max inv:     Te = {te_at_max_inv - 273.15:.0f} degC,  "
              f"Tl = {tl_at_max_inv - 273.15:.0f} degC")
    else:
        inv_detected = False
        max_inv = 0.0
        print("\n  No significant temperature inversion detected.")

    idx_te_peak = int(np.argmax(all_te_surf))
    te_peak_all = all_te_surf[idx_te_peak]
    tl_peak_all = all_tl_surf.max()
    t_peak_te_rel = all_times[idx_te_peak] - first_pulse_center
    pk_te_v, pk_te_u = smart_time(t_peak_te_rel)

    te_end = y_current[:nz]
    tl_end = y_current[nz:]
    du_e = _trapezoid(0.5 * gamma * (te_end**2 - t0**2), z_grid)
    du_l = _trapezoid(cl * (tl_end - t0), z_grid)
    du_total = du_e + du_l
    e_input = n_pulses * eabs_areal

    # ==================  Print results  =====================================
    frep_v, frep_u = smart_freq(f_rep)
    tau_v, tau_u = smart_time(tau_fwhm)
    ep_v, ep_u = smart_energy(ep)
    spot_v, spot_u = smart_length(spot_radius)
    lz_v, lz_u = smart_length(lz)
    dz_v, dz_u = smart_length(dz)
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
    print(f"  Avg Power:               {pavg:.3g} W")
    print(f"  Rep Rate:                {frep_v:.4g} {frep_u}")
    print(f"  Pulse Energy:            {ep_v:.4g} {ep_u}")
    print(f"  Pulse Width (FWHM):      {tau_v:.4g} {tau_u}")
    print(f"  Spot Radius:             {spot_v:.4g} {spot_u}")
    print(f"  Fluence (peak):          {f_peak / 1e4:.5g} J/cm^2")
    print(f"  Absorbance:              {absorbance:.2f}")
    print("------------------------------------------------------------")
    print(f"  Depth domain:            {lz_v:.1f} {lz_u}  ({nz} nodes, dz={dz_v:.1f} {dz_u})")
    print(f"  Pulses simulated:        {n_pulses}")
    print(f"  Total time points:       {all_times.size}")
    print(f"  Wall time:               {wall_time:.2f} s")
    print("------------------------------------------------------------")
    print(f"  Peak surface Te:         {te_peak_all - 273.15:.1f} degC  "
          f"(at {pk_te_v:.4g} {pk_te_u})")
    print(f"  Peak surface Tl:         {tl_peak_all - 273.15:.1f} degC")
    print(f"  Final surface Te:        {all_te_surf[-1] - 273.15:.2f} degC")
    print(f"  Final surface Tl:        {all_tl_surf[-1] - 273.15:.2f} degC")
    if inv_detected:
        print(f"  Inversion onset:         {on_v:.4g} {on_u}")
        print(f"  Max inversion (Tl-Te):   {max_inv:.1f} degC  at {mx_v:.4g} {mx_u}")
    print(f"  E_absorbed (areal):      {e_input:.4g} J/m^2")
    print(f"  E_stored   (areal):      {du_total:.4g} J/m^2  "
          f"(electrons {du_e:.4g} + lattice {du_l:.4g})")
    print("============================================================\n")

    # ==================  Export results to file  ============================
    default_out = os.path.join(os.getcwd(), "outputs")
    output_dir = get_cfg_field(cfg, "outputDir", default_out)
    os.makedirs(output_dir, exist_ok=True)

    case_tag = get_cfg_field(cfg, "caseTag", "")
    freq_str = (f"{frep_v:.4g}_{frep_u}").replace(".", "p")
    pulse_str = (f"{tau_v:.4g}_{tau_u}").replace(".", "p")
    power_str = (f"{pavg:.4g}_W").replace(".", "p")
    spot_str = (f"{spot_v:.4g}_{spot_u}").replace(".", "p")
    out_filename = (f"TTM1D_{freq_str}_{pulse_str}_{power_str}_{spot_str}_"
                    f"{n_pulses}p_{pulse_profile_name}.txt")
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
        fid.write(f"  Depth:            {lz_v:.4g} {lz_u}  ({nz} nodes, dz={dz_v:.2f} {dz_u})\n")
        fid.write(f"  Pulses:           {n_pulses}\n")
        fid.write(f"  Time points:      {all_times.size}\n")
        fid.write(f"  Wall time:        {wall_time:.2f} s\n")
        fid.write("\n--- Results ---\n")
        fid.write(f"  Peak surface Te:  {te_peak_all - 273.15:.1f} degC\n")
        fid.write(f"  Peak surface Tl:  {tl_peak_all - 273.15:.1f} degC\n")
        fid.write(f"  Final surface Te: {all_te_surf[-1] - 273.15:.2f} degC\n")
        fid.write(f"  Final surface Tl: {all_tl_surf[-1] - 273.15:.2f} degC\n")
        if inv_detected:
            fid.write(f"  Inversion onset:  {on_v:.4g} {on_u}\n")
            fid.write(f"  Max Tl-Te:        {max_inv:.1f} degC at {mx_v:.4g} {mx_u}\n")
        fid.write(f"  E_absorbed:       {e_input:.4g} J/m^2\n")
        fid.write(f"  E_stored:         {du_total:.4g} J/m^2\n\n")
        fid.writelines(f"  Pulse {p + 1}:  Te_peak = {te_peak_per_pulse[p] - 273.15:.0f} degC,  "
                      f"Tl_peak = {tl_peak_per_pulse[p] - 273.15:.0f} degC,  "
                      f"Tresid = {tresid_per_pulse[p] - 273.15:.1f} degC\n" for p in range(n_pulses))
        fid.write("\n============================================================\n")
        fid.write("  Surface XY Data: Time (s) | Te_surf (degC) | Tl_surf (degC)\n")
        fid.write("============================================================\n")
        fid.write(f"{'Time_s':>20}  {'Te_surf_degC':>16}  {'Tl_surf_degC':>16}\n")
        fid.writelines(
            f"{all_times[i]:20.12e}  {all_te_surf[i] - 273.15:16.6f}  "
            f"{all_tl_surf[i] - 273.15:16.6f}\n" for i in range(all_times.size))
    print(f"  Output written to: {out_path}\n")

    if make_plots:
        from .plotting import plot_single_pulse

        plot_single_pulse(
            all_times=all_times, all_te_surf=all_te_surf, all_tl_surf=all_tl_surf,
            first_pulse_center=first_pulse_center, inversion_mask=inversion_mask,
            inv_detected=inv_detected, snap_te=snap_te, snap_tl=snap_tl,
            snap_labels=snap_labels, z_grid=z_grid, lz=lz,
            material=material, gamma=gamma, cl=cl, g_ep=g_ep, ke0=ke0, kl=kl,
            alpha_opt=alpha_opt, delta_opt=delta_opt, pavg=pavg,
            frep_v=frep_v, frep_u=frep_u, ep_v=ep_v, ep_u=ep_u,
            tau_v=tau_v, tau_u=tau_u, spot_v=spot_v, spot_u=spot_u,
            f_peak=f_peak, absorbance=absorbance, n_pulses=n_pulses,
            te_peak_all=te_peak_all, tl_peak_all=tl_peak_all, max_inv=max_inv,
            save_dir=(output_dir if save_figures else None), case_tag=case_tag)

    print("Done.")

    return {
        "solver": "SinglePulse",
        "solverId": "single_pulse",
        "contractVersion": "v1",
        "material": material,
        "peakTe_C": te_peak_all - 273.15,
        "peakTl_C": tl_peak_all - 273.15,
        "finalTe_C": all_te_surf[-1] - 273.15,
        "finalTl_C": all_tl_surf[-1] - 273.15,
        "wallTime_s": wall_time,
        "outputFile": out_path,
        "outputDir": output_dir,
        "inputConfig": cfg,
        "invDetected": inv_detected,
        "maxInv_C": max_inv,
    }
