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
from dataclasses import dataclass

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
from .physics import (
    deposit_pulse,
    depth_deposit_shape,
    derive_laser,
    equilibrate,
)
from .progress import ProgressReporter
from .reporting import (
    apply_case_tag,
    case_tag,
    filename_slug,
    resolve_output_dir,
    write_header,
)
from .schema import defaults as schema_defaults
from .schema import effective_config, require_pulses
from .units import smart_energy, smart_freq, smart_length, smart_time


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


@dataclass(frozen=True)
class _Setup:
    """Everything the pulse loop needs, resolved once before it starts.

    The two solve modes are genuinely different algorithms, so each gets its
    own function below; this record is what they share.
    """

    # Material and derived diffusion
    gamma: float
    cl: float
    g_ep: float
    k_tab_t: np.ndarray
    k_tab_k: np.ndarray
    alpha_l: float
    # Pulse train
    eabs_vol: float
    tau_fwhm: float
    trep: float
    sim_duration: float
    n_pulses: int
    pulse_offset: float
    prof_code: int
    # Grids
    t0: float
    nz: int
    dz: float
    nr: int
    r_grid: np.ndarray
    dr: float
    fluence_ratio: np.ndarray
    # Depth deposit shape (loop invariant)
    depth_is_exp: bool
    exp_decay_z: np.ndarray
    box_mask_z: np.ndarray
    # Time stepping
    f_target: float
    n_diff: int
    n_diff_rad: int
    dt_floor_abs: float
    pulse_fine_win: float
    relax_tol: float
    relax_max_t: float
    # Run controls
    store_history: bool
    progress_interval: int
    early_stop_enabled: bool
    early_stop_check_interval: int
    early_stop_t_melt_c: float
    early_stop_melt_radius_um: float


@dataclass
class _State:
    """Per-pulse results accumulated by whichever solve mode ran."""

    teq_vals: np.ndarray
    tresid_vals: np.ndarray
    tresid_radial: np.ndarray
    cell_times: list
    cell_tl: list
    cell_coast_t: list
    cell_coast_tl: list
    n_pulses_run: int

    @classmethod
    def empty(cls, n_pulses: int, nr: int) -> _State:
        return cls(teq_vals=np.zeros(n_pulses),
                   tresid_vals=np.zeros(n_pulses),
                   tresid_radial=np.zeros((n_pulses, nr)),
                   cell_times=[], cell_tl=[],
                   cell_coast_t=[], cell_coast_tl=[],
                   n_pulses_run=n_pulses)


def _solve_scale(s: _Setup, st: _State, progress: ProgressReporter) -> None:
    """One 0D TTM at beam centre, scaled radially by the Gaussian fluence.

    The centre-point rise is applied across the radial array in proportion to
    the local fluence, then a single survival factor carries the depth cooling
    outward. Cheap and accurate while Ce(Te) stays near-linear over the
    fluence range.
    """
    t0, gamma, cl, g_ep = s.t0, s.gamma, s.cl, s.g_ep
    k_tab_t, k_tab_k, alpha_l, f_target = s.k_tab_t, s.k_tab_k, s.alpha_l, s.f_target
    n_pulses, trep, tau_fwhm, dz, dr = s.n_pulses, s.trep, s.tau_fwhm, s.dz, s.dr
    pulse_offset, store_history = s.pulse_offset, s.store_history

    te_now = t0
    tl_now = t0
    tz = t0 * np.ones(s.nz)
    tr_surf = t0 * np.ones(s.nr)

    for np_i in range(n_pulses):
        t_pulse = pulse_offset + np_i * trep
        t_start = t_pulse - 5.0 * tau_fwhm

        # --- Phase 1: RK4 around the pulse (beam centre, full fluence) ---
        loc_t, loc_te, loc_tl, _ = rk4_pulse_phase(
            t_pulse, t_start, te_now, tl_now,
            gamma, g_ep, cl,
            n_pulses, trep, pulse_offset, s.prof_code, tau_fwhm, s.eabs_vol,
            s.pulse_fine_win, s.relax_tol, s.relax_max_t, s.dt_floor_abs,
        )
        if store_history:
            st.cell_times.append(loc_t)
            st.cell_tl.append(loc_tl)

        teq = equilibrate(loc_te[-1], loc_tl[-1], gamma, cl)
        st.teq_vals[np_i] = teq

        # Deposit pulse heat into the radial surface array
        dt_pulse = teq - te_now
        tr_surf = tr_surf + dt_pulse * s.fluence_ratio

        # --- Phase 2: depth CN cooling + radial CN diffusion ---
        t_fine_end = loc_t[-1]
        if np_i < n_pulses - 1:
            t_next_start = pulse_offset + (np_i + 1) * trep - 5.0 * tau_fwhm
        else:
            t_next_start = s.sim_duration
        coast_gap = t_next_start - t_fine_end

        if coast_gap > 0:
            tz = deposit_pulse(tz, teq, s.exp_decay_z, s.box_mask_z,
                               s.depth_is_exp)

            n_diff_local = max(s.n_diff, int(np.ceil(
                alpha_l * coast_gap / (f_target * dz**2))))
            n_sample = min(n_diff_local, 50)
            sample_int = max(1, n_diff_local // n_sample)
            tz, c_t, c_tl = cn_coast_kt(
                tz, coast_gap, n_diff_local, dz, t0, cl,
                k_tab_t, k_tab_k, t_fine_end, sample_int)
            if store_history:
                st.cell_coast_t.append(c_t)
                st.cell_coast_tl.append(c_tl)
            tresidual = tz[0]

            # Apply depth cooling to the radial surface array
            if (teq - t0) > 1e-10:
                survival = (tresidual - t0) / (teq - t0)
            else:
                survival = 1.0
            tr_surf = t0 + (tr_surf - t0) * survival

            # Radial CN diffusion (cylindrical coordinates)
            n_rad_steps = max(s.n_diff_rad, int(np.ceil(
                alpha_l * coast_gap / (f_target * dr**2))))
            dt_rad = coast_gap / n_rad_steps
            tr_surf = radial_coast_kt(
                tr_surf, n_rad_steps, dt_rad, dr, t0, cl, k_tab_t, k_tab_k)
        elif store_history:
            st.cell_coast_t.append(np.empty(0))
            st.cell_coast_tl.append(np.empty(0))

        st.tresid_vals[np_i] = tr_surf[0]
        st.tresid_radial[np_i, :] = tr_surf
        te_now = tr_surf[0]
        tl_now = tr_surf[0]

        if (np_i + 1) % s.progress_interval == 0 or np_i + 1 == n_pulses:
            print(f"    Pulse {np_i + 1}/{n_pulses}: "
                  f"Teq={teq - 273.15:.1f} C, Tresid={tr_surf[0] - 273.15:.1f} C")
        progress.update(np_i + 1)

        # --- Early stop check ---
        if (s.early_stop_enabled
                and (np_i + 1) % s.early_stop_check_interval == 0
                and _early_stop_hit(tr_surf, s.r_grid, s.early_stop_t_melt_c,
                                    s.early_stop_melt_radius_um, np_i + 1)):
            st.n_pulses_run = np_i + 1
            break


def _solve_independent(s: _Setup, st: _State, progress: ProgressReporter) -> None:
    """An independent 0D TTM at every radial node, pulse-major.

    Each node sees its own local fluence, so the nonlinear Ce(Te) response is
    captured per node rather than scaled from the centre. Costs one RK4 solve
    per node per pulse, and adds a depth-profile rescale after radial
    diffusion that the scaled mode does not need.
    """
    t0, gamma, cl, g_ep = s.t0, s.gamma, s.cl, s.g_ep
    k_tab_t, k_tab_k, alpha_l, f_target = s.k_tab_t, s.k_tab_k, s.alpha_l, s.f_target
    n_pulses, trep, tau_fwhm, dz, dr = s.n_pulses, s.trep, s.tau_fwhm, s.dz, s.dr
    pulse_offset, store_history, nr = s.pulse_offset, s.store_history, s.nr

    te_all = t0 * np.ones(nr)
    tl_all = t0 * np.ones(nr)
    tz_all = t0 * np.ones((s.nz, nr))
    eabs_vol_all = s.eabs_vol * s.fluence_ratio
    active = s.fluence_ratio >= 1e-12

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
                n_pulses, trep, pulse_offset, s.prof_code, tau_fwhm,
                eabs_vol_all[ri],
                s.pulse_fine_win, s.relax_tol, s.relax_max_t, s.dt_floor_abs,
            )
            if ri == 0:
                if store_history:
                    st.cell_times.append(loc_t)
                    st.cell_tl.append(loc_tl)
                t_fine_end_center = loc_t[-1]

            teq = equilibrate(loc_te[-1], loc_tl[-1], gamma, cl)
            if ri == 0:
                st.teq_vals[np_i] = teq

            # Deposit pulse energy into this node's depth profile
            tz_all[:, ri] = deposit_pulse(tz_all[:, ri], teq, s.exp_decay_z,
                                          s.box_mask_z, s.depth_is_exp)
            te_all[ri] = teq
            tl_all[ri] = teq

        # --- Phase 2: depth CN diffusion at each node ---
        if np_i < n_pulses - 1:
            t_next_start = pulse_offset + (np_i + 1) * trep - 5.0 * tau_fwhm
        else:
            t_next_start = s.sim_duration
        coast_gap = t_next_start - t_fine_end_center

        if coast_gap > 0:
            n_diff_local = max(s.n_diff, int(np.ceil(
                alpha_l * coast_gap / (f_target * dz**2))))
            dt_diff = coast_gap / n_diff_local
            n_sample = min(n_diff_local, 50)
            sample_int = max(1, n_diff_local // n_sample)

            c_t, c_tl = cn_depth_multi_kt(
                tz_all, active, n_diff_local, dt_diff, dz, t0, cl,
                k_tab_t, k_tab_k, t_fine_end_center, sample_int)
            if store_history:
                st.cell_coast_t.append(c_t)
                st.cell_coast_tl.append(c_tl)
            te_all[active] = tz_all[0, active]
            tl_all[active] = tz_all[0, active]

            # --- Phase 3: radial CN diffusion ---
            tr_pre = te_all.copy()
            n_rad_steps = max(s.n_diff_rad, int(np.ceil(
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
            st.cell_coast_t.append(np.empty(0))
            st.cell_coast_tl.append(np.empty(0))

        st.tresid_radial[np_i, :] = te_all
        st.tresid_vals[np_i] = te_all[0]

        if (np_i + 1) % s.progress_interval == 0 or np_i + 1 == n_pulses:
            print(f"    Pulse {np_i + 1}/{n_pulses}: "
                  f"Teq={st.teq_vals[np_i] - 273.15:.1f} C, "
                  f"Tresid={te_all[0] - 273.15:.1f} C")
        progress.update(np_i + 1)

        # --- Early stop check ---
        if (s.early_stop_enabled
                and (np_i + 1) % s.early_stop_check_interval == 0
                and _early_stop_hit(te_all, s.r_grid, s.early_stop_t_melt_c,
                                    s.early_stop_melt_radius_um, np_i + 1)):
            st.n_pulses_run = np_i + 1
            break


_SOLVE_MODES = {"scale": _solve_scale, "independent": _solve_independent}


def radial_profile_solver(cfg: dict | None = None) -> dict:
    """Run the radial-profile TTM solver. Returns the v1 results dict."""
    if cfg is None:
        cfg = {}

    # Defaults come from the schema so there is one place to read them and
    # one place they can change. See schema.describe_solver('radial_profile').
    d = schema_defaults("radial_profile")

    make_plots = get_cfg_field(cfg, "makePlots", d["makePlots"])
    save_figures = get_cfg_field(cfg, "saveFigures", d["saveFigures"])
    if save_figures:
        make_plots = True

    # storeHistory=False drops the per-pulse time histories (kept only for
    # the timeline plots), bounding memory for very long runs: ~100k pulses
    # would otherwise accumulate gigabytes of RK4/coast samples. All physics,
    # per-pulse scalars, and radial profiles are unaffected.
    store_history = bool(get_cfg_field(cfg, "storeHistory", d["storeHistory"]))
    if not store_history and make_plots:
        print("  storeHistory=False: time-history figures disabled.")
        make_plots = False
        save_figures = False

    print("=== Radial Surface TTM Pulsed Laser Calculator ===")

    # ========================  USER INPUTS  =================================
    material = get_cfg_field(cfg, "material", d["material"])
    mat = resolve_material(cfg, needs_optical=False)
    gamma, cl, g_ep, kl = mat.gamma, mat.cl, mat.g_ep, mat.k_total

    pavg = get_cfg_field(cfg, "Pavg", d["Pavg"])
    spot_radius = get_cfg_field(cfg, "spotRadius", d["spotRadius"])
    f_rep = get_cfg_field(cfg, "f_rep", d["f_rep"])
    tau_fwhm = get_cfg_field(cfg, "tau_FWHM", d["tau_FWHM"])
    pulse_profile_name = get_cfg_field(cfg, "pulseProfile", d["pulseProfile"])

    absorbance = get_cfg_field(cfg, "absorbance", d["absorbance"])
    leff = get_cfg_field(cfg, "Leff", d["Leff"])
    t0_c = get_cfg_field(cfg, "T0_C", d["T0_C"])

    nr = int(get_cfg_field(cfg, "Nr", d["Nr"]))
    r_max_factor = get_cfg_field(cfg, "rMax_factor", d["rMax_factor"])

    radial_solve_mode = str(
        get_cfg_field(cfg, "radialSolveMode", d["radialSolveMode"])).lower()
    if radial_solve_mode not in _SOLVE_MODES:
        raise ValueError(
            f'Unknown radialSolveMode "{radial_solve_mode}". '
            f"Use {' or '.join(repr(m) for m in _SOLVE_MODES)}.")

    sim_duration = get_cfg_field(cfg, "simDuration", d["simDuration"])

    early_stop_melt_radius_um = get_cfg_field(
        cfg, "earlyStopMeltRadius_um", d["earlyStopMeltRadius_um"])
    # Defaults to this material's melting point, not tungsten's.
    early_stop_t_melt_c = get_cfg_field(cfg, "earlyStopT_melt_C", mat.t_melt_c)
    early_stop_check_interval = int(get_cfg_field(
        cfg, "earlyStopCheckInterval", d["earlyStopCheckInterval"]))
    early_stop_enabled = early_stop_melt_radius_um > 0

    depth_profile = str(
        get_cfg_field(cfg, "depthProfile", d["depthProfile"])).lower()
    dz_target = get_cfg_field(cfg, "dzTarget", d["dzTarget"])
    n_diff_min = int(get_cfg_field(cfg, "Ndiff", d["Ndiff"]))
    show_progress = get_cfg_field(cfg, "showProgress", d["showProgress"])

    # Hybrid k(T): tungsten table, constant kl otherwise (this solver has no
    # separate electron conductivity, unlike the depth solver's ke0+kl table)
    k_tab_t, k_tab_k = k_table(mat)

    # ==================  Derived quantities  ================================
    dl = derive_laser(pavg=pavg, f_rep=f_rep, spot_radius=spot_radius,
                      absorbance=absorbance, t0_c=t0_c, gamma=gamma,
                      g_ep=g_ep, sim_duration=sim_duration)
    t0 = dl.t0_k
    ep = dl.pulse_energy
    f_peak = dl.peak_fluence
    eabs_areal = dl.absorbed_fluence
    eabs_vol = eabs_areal / leff
    trep = dl.period
    n_pulses = dl.n_pulses
    require_pulses("radial_profile", n_pulses)

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
    exp_decay_z, box_mask_z = depth_deposit_shape(z_grid, leff)

    n_profile_snaps = min(n_pulses, 12)
    if n_pulses > 1:
        profile_snap_pulses = np.unique(np.round(
            np.logspace(0, np.log10(n_pulses), n_profile_snaps)).astype(int))
    else:
        profile_snap_pulses = np.array([1])

    progress_interval = max(1, n_pulses // 20)
    tic_all = time.perf_counter()
    progress = ProgressReporter(n_pulses, title="laserttm: radial profile",
                                enabled=show_progress)

    setup = _Setup(
        gamma=gamma, cl=cl, g_ep=g_ep, k_tab_t=k_tab_t, k_tab_k=k_tab_k,
        alpha_l=alpha_l,
        eabs_vol=eabs_vol, tau_fwhm=tau_fwhm, trep=trep,
        sim_duration=sim_duration, n_pulses=n_pulses,
        pulse_offset=pulse_offset, prof_code=prof_code,
        t0=t0, nz=nz, dz=dz, nr=nr, r_grid=r_grid, dr=dr,
        fluence_ratio=fluence_ratio,
        depth_is_exp=depth_is_exp, exp_decay_z=exp_decay_z,
        box_mask_z=box_mask_z,
        f_target=f_target, n_diff=n_diff, n_diff_rad=n_diff_rad,
        dt_floor_abs=dt_floor_abs, pulse_fine_win=pulse_fine_win,
        relax_tol=relax_tol, relax_max_t=relax_max_t,
        store_history=store_history, progress_interval=progress_interval,
        early_stop_enabled=early_stop_enabled,
        early_stop_check_interval=early_stop_check_interval,
        early_stop_t_melt_c=early_stop_t_melt_c,
        early_stop_melt_radius_um=early_stop_melt_radius_um,
    )
    state = _State.empty(n_pulses, nr)

    if radial_solve_mode == "scale":
        print("  Running single center-point simulation...")
    else:
        print(f"  Running independent 0D solves at {nr} radial nodes (pulse-major)...")
    _SOLVE_MODES[radial_solve_mode](setup, state, progress)

    teq_vals = state.teq_vals
    tresid_vals = state.tresid_vals
    tresid_radial = state.tresid_radial
    cell_times = state.cell_times
    cell_tl = state.cell_tl
    cell_coast_t = state.cell_coast_t
    cell_coast_tl = state.cell_coast_tl
    n_pulses_run = state.n_pulses_run

    # ==================  Shared epilogue (both modes)  ======================
    n_pulses_requested = n_pulses
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
    output_dir = resolve_output_dir(cfg)
    out_filename = apply_case_tag(cfg, (
        f"TTM_Radial_Result_{filename_slug(f_rep, tau_fwhm, pavg, spot_radius)}_"
        f"{n_pulses}p_{pulse_profile_name}.txt"))
    out_path = os.path.join(output_dir, out_filename)

    final_radial_t = tresid_radial[-1, :]
    with open(out_path, "w", encoding="utf-8") as fid:
        write_header(fid, "Radial Surface TTM Calculator — Output",
                     f"  Mode: {radial_solve_mode}")
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
        fid.writelines(f"  r = {r_grid[ri] * 1e6:8.2f} um :  "
                       f"T = {final_radial_t[ri] - 273.15:.2f} C\n"
                       for ri in range(nr))
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
        "caseTag": case_tag(cfg),
        "resolvedConfig": effective_config("radial_profile", cfg),
        "materialProps": mat.props(k_model_name(mat)),
        "warnings": [],
        "mode": radial_solve_mode,
        "nPulses": n_pulses,
        "nPulsesRequested": n_pulses_requested,
        "earlyStopped": n_pulses < n_pulses_requested,
        "peakTeq_C": teq_vals.max() - 273.15,
        "finalResid_C": tresid_vals[-1] - 273.15,
        "TeqVals_C": teq_vals - 273.15,
        "TresidVals_C": tresid_vals - 273.15,
        "wallTime_s": wall_time,
        "outputFile": out_path,
        "outputDir": output_dir,
        "inputConfig": cfg,
        "rGrid_um": r_grid * 1e6,
        "finalRadialProfile_C": tresid_radial[-1, :] - 273.15,
        "spotRadius_um": spot_radius * 1e6,
    }
