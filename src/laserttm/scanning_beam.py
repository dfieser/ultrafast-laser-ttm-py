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
from datetime import datetime

import numpy as np
from scipy.io import savemat

from .kernels import profile_code, rk4_single_pulse_response, scanning_chunk
from .progress import ProgressReporter
from .units import smart_energy, smart_freq, smart_length, smart_time

_DEFAULTS = {
    "material": "W",
    "gamma": 137.3,
    "Cl": 2.54e6,
    "G": 1.65e17,
    "kl": 174.0,
    "Pavg": 40.0,
    "spotRadius": 100e-6,
    "f_rep": 18e6,
    "tau_FWHM": 100e-15,
    "pulseProfile": "gaussian",
    "v_scan": 1.0,
    "scanLength": 2e-3,
    "absorbance": 0.55,
    "Leff": 100e-9,
    "T0_C": 25.0,
    "Nx": 120,
    "Ny": 60,
    "xPad": 3,
    "yExtent": 5,
    "depthProfile": "exponential",
    "dzTarget": 500e-9,
    "Ndiff": 100,
    "NadiPerGap": 10,
}

_PRESETS = {
    "w":  (137.3, 2.54e6, 1.65e17, 174.0),
    "cu": (98.0,  3.45e6, 0.90e17, 401.0),
    "al": (136.0, 2.42e6, 2.40e17, 237.0),
}


def _matlab_round(x: float) -> int:
    return int(np.floor(x + 0.5))


def scanning_beam_solver(params: dict | None = None,
                         output_dir: str | None = None,
                         save_plots: bool = True) -> dict:
    """Run the 2D scanning-beam TTM solver. Returns the v1 results dict."""
    if params is None:
        params = {}
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "outputs")

    # Merge defaults with user params (MATLAB echoes the merged struct)
    params = {**_DEFAULTS, **dict(params)}

    key = str(params["material"]).lower()
    if key in _PRESETS:
        gamma, cl, g_ep, kl = _PRESETS[key]
    elif key == "custom":
        gamma = params["gamma"]
        cl = params["Cl"]
        g_ep = params["G"]
        kl = params["kl"]
    else:
        raise ValueError(f'Unknown material "{params["material"]}".')

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
    t0 = t0_c + 273.15
    ep = pavg / f_rep
    f_peak = 2.0 * ep / (np.pi * spot_radius**2)
    eabs_vol = absorbance * f_peak / leff
    trep = 1.0 / f_rep
    alpha_l = kl / cl
    sim_duration = scan_length / v_scan
    n_pulses = _matlab_round(sim_duration * f_rep)
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
    utot = 0.5 * gamma * te_end**2 + cl * tl_end
    teq_single = (-cl + np.sqrt(cl**2 + 2.0 * gamma * utot)) / gamma
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
                                enabled=params.get("showProgress"))
    np_done = 0
    while np_done < n_pulses:
        np_next = min(np_done + progress_interval, n_pulses)
        scanning_chunk(np_done, np_next, n_pulses,
                       tsurf, tpeak_map, tz, peak_t_history,
                       x_grid, gy_gauss, inv2w2, v_scan, trep, dteq_single, t0,
                       depth_is_exp, exp_decay_z, box_mask_z,
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
        "Tpeak_map": tpeak_map,
        "Tsurf": tsurf,
        "peakT_history": peak_t_history,
        "xGrid": x_grid,
        "yGrid": y_grid,
        "nPulses": n_pulses,
        "pulseSpacing": pulse_spacing,
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

    freq_str = (f"{frep_v:.4g}_{frep_u}").replace(".", "p")
    pulse_str = (f"{tau_v:.4g}_{tau_u}").replace(".", "p")
    power_str = (f"{pavg:.4g}_W").replace(".", "p")
    spot_str = (f"{spot_v:.4g}_{spot_u}").replace(".", "p")
    scan_str = (f"{v_scan:.3g}_mps").replace(".", "p")
    base_name = (f"TTMmov_{freq_str}_{pulse_str}_{power_str}_{spot_str}_"
                 f"{scan_str}_{n_pulses}p")

    out_path = os.path.join(output_dir, base_name + ".txt")
    with open(out_path, "w") as fid:
        fid.write("============================================================\n")
        fid.write("  Moving Laser TTM — Output\n")
        # Local wall-clock on purpose, matching the MATLAB reference output
        fid.write(f"  Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")  # noqa: DTZ005
        fid.write("============================================================\n\n")
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
