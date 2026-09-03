"""2D scanning-laser TTM surface solver.

Python port of ``src/Scanning_Beam_Solver.m``: a pulsed laser moving along a
scan line. Each pulse deposits the single-pulse equilibrated temperature rise
as a Gaussian footprint at the current beam position; between pulses the
surface loses heat into the depth (Crank–Nicolson with survival scaling) and
laterally (ADI 2D diffusion with Dirichlet far-field borders).

As in MATLAB, the interface is ``scanning_beam_solver(params, output_dir,
save_plots)`` with a ``params`` dict rather than the ``cfg`` convention of
the other solvers; missing fields take the documented defaults. The v1
result contract matches the MATLAB solver field-for-field.
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.io import savemat

from .config import get_cfg_field
from .kernels import profile_code, rk4_single_pulse_response, scanning_chunk
from .materials import k_model_name, resolve_material
from .physics import deposit_shape_weight, derive_laser, equilibrate, validity_warnings
from .progress import ProgressReporter
from .reporting import apply_case_tag, case_tag, filename_slug, write_header
from .schema import defaults as schema_defaults
from .schema import effective_config, require_pulses
from .units import smart_energy, smart_freq, smart_length, smart_time

# Defaults come from the schema, which is the single place they are declared
# for every solver. The material property keys are dropped: they are
# overrides resolved by materials.resolve_material, not solver defaults.
_DEFAULTS = {k: v for k, v in schema_defaults("scanning_beam").items()
             if v is not None}



def scanning_beam_solver(params: dict | None = None,
                         output_dir: str | None = None,
                         save_plots: bool = True) -> dict:
    """Run the 2D scanning-beam TTM solver. Returns the v1 results dict."""
    if params is None:
        params = {}
    # Keep the caller's own keys: the merged defaults below would otherwise
    # look like explicit per-field material overrides.
    user_params = dict(params)

    # Merge defaults with user params (MATLAB echoes the merged struct)
    params = {**_DEFAULTS, **user_params}

    # The dict is authoritative when the positional arguments are unset, so
    # that outputDir/makePlots/saveFigures work here as in every other solver
    # (the CLI and the MCP server dispatch all solvers as solver(cfg)).
    if output_dir is None:
        output_dir = get_cfg_field(params, "outputDir",
                                   os.path.join(os.getcwd(), "outputs"))
    if save_plots:
        save_plots = bool(get_cfg_field(params, "makePlots", True)) or \
            bool(get_cfg_field(params, "saveFigures", False))

    mat = resolve_material(params, needs_optical=False, overrides=user_params)
    gamma, cl, g_ep, kl = mat.gamma, mat.cl, mat.g_ep, mat.k_total

    pavg = params["Pavg"]
    spot_radius = params["spotRadius"]
    f_rep = params["f_rep"]
    tau_fwhm = params["tau_FWHM"]
    pulse_profile_name = params["pulseProfile"]
    v_scan = params["v_scan"]
    scan_length = params["scanLength"]
    absorbance = params["absorbance"]
    leff = params["Leff"]
    t0_c = params["T0_C"]
    nx = int(params["Nx"])
    ny = int(params["Ny"])
    x_pad = params["xPad"]
    y_extent = params["yExtent"]
    n_diff = int(params["Ndiff"])
    n_adi_per_gap = int(params["NadiPerGap"])
    depth_profile = str(params["depthProfile"]).lower()
    dz_target = params["dzTarget"]

    # ==================  Derived quantities  ================================
    sim_duration = scan_length / v_scan
    dl = derive_laser(pavg=pavg, f_rep=f_rep, spot_radius=spot_radius,
                      absorbance=absorbance, t0_c=t0_c, gamma=gamma,
                      g_ep=g_ep, sim_duration=sim_duration)
    t0 = dl.t0_k
    ep = dl.pulse_energy
    f_peak = dl.peak_fluence
    eabs_vol = dl.absorbed_fluence / leff
    trep = dl.period
    alpha_l = kl / cl
    n_pulses = dl.n_pulses
    require_pulses("scanning_beam", n_pulses)
    pulse_spacing = v_scan / f_rep

    # Depth grid — dz must resolve the heated layer (Leff)
    ldiff = max(5.0 * np.sqrt(alpha_l * sim_duration), 50e-6)
    dz = min(dz_target, leff)
    nz = int(np.ceil(ldiff / dz)) + 1
    z_grid = np.arange(nz) * dz

    # 2D surface grid
    x_min = -x_pad * spot_radius
    x_max = scan_length + x_pad * spot_radius
    y_min = -y_extent * spot_radius
    y_max = y_extent * spot_radius
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)

    print(f"  [{str(params['material']).upper()}] v={v_scan:.3g} m/s, "
          f"P={pavg:.3g}W, f={f_rep:.4g} Hz, spot={spot_radius * 1e6:.0f}um, "
          f"{n_pulses} pulses")

    # ==================  Single-pulse TTM response  =========================
    pulse_offset = 5.0 * tau_fwhm
    prof_code = profile_code(pulse_profile_name)
    te_end, tl_end = rk4_single_pulse_response(
        t0, gamma, g_ep, cl, tau_fwhm, pulse_offset, prof_code, eabs_vol)
    teq_single = equilibrate(te_end, tl_end, gamma, cl)
    dteq_single = teq_single - t0

    # ==================  Precompute for the vectorized loop  ================
    gy_gauss = np.exp(-2.0 * y_grid**2 / spot_radius**2)
    inv2w2 = 2.0 / spot_radius**2

    depth_is_exp = depth_profile == "exponential"
    if depth_is_exp:
        exp_decay_z = np.exp(-z_grid / leff)
        box_mask_z = np.zeros(nz, dtype=np.bool_)
    else:
        exp_decay_z = np.zeros(nz)
        box_mask_z = z_grid <= leff
    legacy_deposit = bool(get_cfg_field(params, "legacyDeposit",
                                        _DEFAULTS["legacyDeposit"]))
    w_shape = deposit_shape_weight(z_grid, leff, depth_is_exp)

    coast_gap = trep
    dt_diff = coast_gap / n_diff
    r_diff = alpha_l * dt_diff / dz**2
    n_cn = nz - 1

    dt_adi = trep / n_adi_per_gap
    fxdt = (alpha_l / dx**2) * dt_adi
    fydt = (alpha_l / dy**2) * dt_adi

    # ==================  Main simulation loop  ==============================
    tsurf = t0 * np.ones((ny, nx))
    tpeak_map = t0 * np.ones((ny, nx))
    tz = t0 * np.ones(nz)
    peak_t_history = np.zeros(n_pulses)
    iy_center = int(np.argmin(np.abs(y_grid)))
    progress_interval = min(max(1, n_pulses // 10), 5000)

    tic_all = time.perf_counter()
    progress = ProgressReporter(n_pulses, title="laserttm: scanning beam",
                                enabled=get_cfg_field(params, "showProgress",
                                                      None))
    np_done = 0
    while np_done < n_pulses:
        np_next = min(np_done + progress_interval, n_pulses)
        scanning_chunk(np_done, np_next,
                       tsurf, tpeak_map, tz, peak_t_history,
                       x_grid, gy_gauss, inv2w2, v_scan, trep, dteq_single, t0,
                       depth_is_exp, exp_decay_z, box_mask_z,
                       legacy_deposit, gamma, cl, leff, w_shape,
                       n_cn, n_diff, r_diff,
                       n_adi_per_gap, fxdt, fydt)
        np_done = np_next
        elapsed = time.perf_counter() - tic_all
        pct_done = 100.0 * np_done / n_pulses
        eta_s = elapsed / np_done * (n_pulses - np_done)
        print(f"    Pulse {np_done}/{n_pulses} ({pct_done:.0f}%) | "
              f"Peak T={peak_t_history[np_done - 1] - 273.15:.1f} C | "
              f"Elapsed {elapsed:.1f}s | ETA {eta_s:.1f}s")
        progress.update(np_done)
    progress.close()
    wall_time = time.perf_counter() - tic_all
    print(f"  Wall time: {wall_time:.2f} s")

    # ==================  Build output struct  ===============================
    results = {
        "solver": "ScanningBeam",
        "solverId": "scanning_beam",
        "contractVersion": "v1",
        "material": params["material"],
        "caseTag": case_tag(params),
        "resolvedConfig": effective_config("scanning_beam", user_params),
        "materialProps": mat.props(k_model_name(mat, constant_only=True)),
        "warnings": validity_warnings(
            float(tpeak_map.max()) - 273.15, mat.t_melt_c, params["material"]),
        "Tpeak_map": tpeak_map,
        "Tsurf": tsurf,
        "peakT_history": peak_t_history,
        "xGrid": x_grid,
        "yGrid": y_grid,
        "nPulses": n_pulses,
        "peakT_C": float(tpeak_map.max()) - 273.15,
        "pulseSpacing": pulse_spacing,
        "simDuration_s": sim_duration,
        "wallTime": wall_time,
        "wallTime_s": wall_time,
        "dTeq_single": dteq_single,
        "params": params,
        "inputConfig": params,
    }

    frep_v, frep_u = smart_freq(f_rep)
    tau_v, tau_u = smart_time(tau_fwhm)
    ep_v, ep_u = smart_energy(ep)
    spot_v, spot_u = smart_length(spot_radius)
    sim_dur_v, sim_dur_u = smart_time(sim_duration)

    # ==================  Save text output  ==================================
    os.makedirs(output_dir, exist_ok=True)

    scan_str = (f"{v_scan:.3g}_mps").replace(".", "p")
    base_name = apply_case_tag(params, (
        f"TTMmov_{filename_slug(f_rep, tau_fwhm, pavg, spot_radius)}_"
        f"{scan_str}_{n_pulses}p"))

    out_path = os.path.join(output_dir, base_name + ".txt")
    with open(out_path, "w", encoding="utf-8") as fid:
        write_header(fid, "Moving Laser TTM — Output")
        fid.write(f"--- Material: {str(params['material']).upper()} ---\n")
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
        fid.write("\n--- Scan ---\n")
        fid.write(f"  Scan Speed:      {v_scan:.4g} m/s\n")
        fid.write(f"  Scan Length:     {scan_length * 1e3:.4g} mm\n")
        fid.write(f"  Pulse Spacing:   {pulse_spacing * 1e6:.4g} um\n")
        fid.write("\n--- Results ---\n")
        fid.write(f"  Pulses:          {n_pulses}\n")
        fid.write(f"  Sim Duration:    {sim_dur_v:.4g} {sim_dur_u}\n")
        fid.write(f"  Peak Temp:       {tpeak_map.max() - 273.15:.1f} C\n")
        fid.write(f"  Wall time:       {wall_time:.2f} s\n")
    results["outPath"] = out_path
    results["outputFile"] = out_path
    results["outputDir"] = output_dir

    # Save raw surface data to .mat for later replotting (MATLAB-compatible)
    mat_path = os.path.join(output_dir, base_name + "_surface.mat")
    savemat(mat_path, {
        "Tpeak_map_C": tpeak_map - 273.15,
        "Tsurf_C": tsurf - 273.15,
        "xGrid_um": x_grid * 1e6,
        "yGrid_um": y_grid * 1e6,
        "peakT_history": peak_t_history,
        "nPulses": float(n_pulses),
        "pulseSpacing": pulse_spacing,
    })
    print(f"  Surface data saved to: {mat_path}")
    results["matPath"] = mat_path

    # ==================  Plots (saved directly, as in MATLAB)  ==============
    if save_plots:
        from .plotting import plot_scanning_beam

        plot_scanning_beam(
            x_grid=x_grid, y_grid=y_grid, tpeak_map=tpeak_map,
            peak_t_history=peak_t_history, iy_center=iy_center,
            scan_length=scan_length, spot_radius=spot_radius,
            pulse_spacing=pulse_spacing, n_pulses=n_pulses,
            material=params["material"], v_scan=v_scan, pavg=pavg,
            frep_v=frep_v, frep_u=frep_u,
            output_dir=output_dir, base_name=base_name)
        print(f"  Saved to: {output_dir}")

    return results
