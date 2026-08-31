"""Matplotlib figures mirroring the MATLAB solver plots.

Figures are created but not shown; call ``matplotlib.pyplot.show()`` from
your script to display them (the examples do). Pass ``save_path`` to also
write a PNG, mirroring the MATLAB ``saveFigures`` behavior.
"""

from __future__ import annotations

import numpy as np


def _time_scale(t_end: float) -> tuple[float, str]:
    """Pick plot time units the way the MATLAB solvers do."""
    if t_end < 1e-9:
        return 1e12, "ps"
    if t_end < 1e-6:
        return 1e9, "ns"
    if t_end < 1e-3:
        return 1e6, "μs"
    return 1e3, "ms"


def _save_fig(fig, save_dir, name, case_tag=""):
    """Save a figure the way the MATLAB solvers do (sanitized figure name)."""
    import re

    if save_dir is None:
        return None
    tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if case_tag:
        tag = f"{case_tag}_{tag}"
    path = f"{save_dir}/{tag}.png"
    fig.savefig(path, dpi=150)
    return path


def plot_single_pulse(*, all_times, all_te_surf, all_tl_surf, first_pulse_center,
                      inversion_mask, inv_detected, snap_te, snap_tl, snap_labels,
                      z_grid, lz, material, gamma, cl, g_ep, ke0, kl, alpha_opt,
                      delta_opt, pavg, frep_v, frep_u, ep_v, ep_u, tau_v, tau_u,
                      spot_v, spot_u, f_peak, absorbance, n_pulses,
                      te_peak_all, tl_peak_all, max_inv,
                      save_dir=None, case_tag=""):
    """Port of the Single_Pulse_Visualizer figures (plots 1-3 + parameters)."""
    import matplotlib.pyplot as plt

    # --- Plot 1: surface temperatures vs time (log axis) -------------------
    fig1 = plt.figure("Surface_Temperatures_vs_Time", figsize=(8, 5), facecolor="w")
    ax = fig1.add_subplot(111)
    t_rel = all_times - first_pulse_center
    pos = t_rel > 0
    t_plot = t_rel[pos] * 1e12
    ax.semilogx(t_plot, all_te_surf[pos] - 273.15, "b-", lw=1.8,
                label="$T_e$ (electron)")
    ax.semilogx(t_plot, all_tl_surf[pos] - 273.15, "r-", lw=1.8,
                label="$T_l$ (lattice)")
    if inv_detected:
        inv_rel = t_rel[inversion_mask] * 1e12
        if inv_rel.size:
            ax.axvspan(inv_rel[0], inv_rel[-1], color=(1, 0.85, 0.85), alpha=0.4,
                       label="$T_l > T_e$ (inversion)", zorder=0)
    ax.set_xlabel("Time after pulse center (ps)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True)
    ax.set_xlim(max(t_plot[0], 1e-2), t_plot[-1])
    _save_fig(fig1, save_dir, "Surface_Temperatures_vs_Time", case_tag)

    # --- Parameter panel ----------------------------------------------------
    fig_p = plt.figure("Simulation_Parameters", figsize=(4.8, 6), facecolor="w")
    lines = [
        "1D TTM Parameters", "",
        f"Material:  {str(material).upper()}",
        f"gamma:  {gamma:.1f}  J m⁻³ K⁻²",
        f"C_l:  {cl:.3e}  J m⁻³ K⁻¹",
        f"G:  {g_ep:.3e}  W m⁻³ K⁻¹",
        f"k_e0:  {ke0:.0f}  W m⁻¹ K⁻¹",
        f"k_l:  {kl:.0f}  W m⁻¹ K⁻¹",
        f"alpha_opt:  {alpha_opt:.2e} m⁻¹  (δ = {delta_opt * 1e9:.0f} nm)",
        "", "Laser",
        f"Avg Power:  {pavg:.3g} W",
        f"Rep Rate:  {frep_v:.4g} {frep_u}",
        f"Pulse Energy:  {ep_v:.4g} {ep_u}",
        f"Pulse Width:  {tau_v:.4g} {tau_u}",
        f"Spot Radius:  {spot_v:.4g} {spot_u}",
        f"Fluence:  {f_peak / 1e4:.4g} J/cm²",
        f"Absorbance:  {absorbance:.2f}",
        "", "Results",
        f"Pulses:  {n_pulses}",
        f"Peak T_e:  {te_peak_all - 273.15:.0f} °C",
        f"Peak T_l:  {tl_peak_all - 273.15:.0f} °C",
        f"Final T_surf:  {all_tl_surf[-1] - 273.15:.1f} °C",
    ]
    if inv_detected:
        lines += ["", "Inversion", f"Max (T_l - T_e):  {max_inv:.1f} °C"]
    fig_p.text(0.05, 0.95, "\n".join(lines), fontsize=9, family="monospace",
               va="top", bbox={"edgecolor": (0.4, 0.4, 0.4),
                               "facecolor": (0.97, 0.97, 0.97), "pad": 10})
    _save_fig(fig_p, save_dir, "Simulation_Parameters", case_tag)

    # --- Plot 2: snapshot subplot grid --------------------------------------
    n_snaps = len(snap_te)
    if n_snaps:
        n_cols = min(4, n_snaps)
        n_rows = int(np.ceil(n_snaps / n_cols))
        z_plot = z_grid * 1e9
        z_max = min(500.0, lz * 1e9)
        fig2, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6.5),
                                  facecolor="w", squeeze=False,
                                  num="Depth_Profiles_First_Pulse_Snapshots")
        for si in range(n_snaps):
            axs = axes[si // n_cols][si % n_cols]
            axs.plot(z_plot, snap_te[si] - 273.15, "b-", lw=1.6, label="$T_e$")
            axs.plot(z_plot, snap_tl[si] - 273.15, "r-", lw=1.6, label="$T_l$")
            axs.set_xlabel("Depth z (nm)", fontsize=9)
            axs.set_ylabel("T (°C)", fontsize=9)
            axs.set_title(f"t = {snap_labels[si]}", fontsize=10)
            axs.legend(loc="best", fontsize=7)
            axs.grid(True)
            axs.set_xlim(0, z_max)
        for si in range(n_snaps, n_rows * n_cols):
            axes[si // n_cols][si % n_cols].set_visible(False)
        fig2.tight_layout()
        _save_fig(fig2, save_dir, "Depth_Profiles_First_Pulse_Snapshots", case_tag)

        # --- Plot 3: overlaid snapshots -------------------------------------
        fig3 = plt.figure("Depth_Profiles_Overlaid", figsize=(8, 5), facecolor="w")
        ax3 = fig3.add_subplot(111)
        cmap = plt.get_cmap("tab10")
        for si in range(n_snaps):
            color = cmap(si % 10)
            ax3.plot(z_plot, snap_te[si] - 273.15, "-", color=color, lw=1.5,
                     label=f"$T_e$  t={snap_labels[si]}")
            ax3.plot(z_plot, snap_tl[si] - 273.15, "--", color=color, lw=1.2)
        ax3.plot([], [], "k--", lw=1.2, label="$T_l$ (dashed)")
        ax3.set_xlabel("Depth z (nm)", fontsize=12)
        ax3.set_ylabel("Temperature (°C)", fontsize=12)
        ax3.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax3.grid(True)
        ax3.set_xlim(0, z_max)
        fig3.tight_layout()
        _save_fig(fig3, save_dir, "Depth_Profiles_Overlaid", case_tag)


def plot_depth_profile(*, all_times, all_tl_surf, teq_vals, tresid_vals,
                       snap_te, snap_tl, snap_labels, z_grid, lz,
                       profile_snaps_tz, profile_snaps_label, z_grid_diff,
                       ldiff, t0, enable_radial, nr_radial, r_max_factor,
                       spot_radius, alpha_diff, sim_duration,
                       material, gamma, cl, g_ep, ke0, kl, alpha_opt, delta_opt,
                       pavg, frep_v, frep_u, ep_v, ep_u, tau_v, tau_u,
                       spot_v, spot_u, f_peak, absorbance,
                       n_pulses, te_peak_all, tl_peak_all, peak_pulse,
                       tresid_last, e_input, du_depth,
                       inv_detected, max_inv, mx_v, mx_u,
                       save_dir=None, case_tag=""):
    """Port of the Depth_Profile_Solver figure set (plots 1-8 + parameters)."""
    import matplotlib.pyplot as plt

    # --- Plot 1: lattice surface temperature, full timeline ----------------
    fig1 = plt.figure("Surface_Lattice_Temperature_vs_Time",
                      figsize=(8, 5), facecolor="w")
    ax = fig1.add_subplot(111)
    t_scale, t_unit = _time_scale(all_times[-1])
    ax.plot(all_times * t_scale, all_tl_surf - 273.15, "r-", lw=1.5,
            label="Lattice Temperature")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(f"Time ({t_unit})", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.grid(True)
    _save_fig(fig1, save_dir, "Surface_Lattice_Temperature_vs_Time", case_tag)

    # --- Parameter summary figure ------------------------------------------
    fig_p = plt.figure("Simulation_Parameters", figsize=(5, 6.5), facecolor="w")
    lines = [
        "1D TTM Parameters", "",
        f"Material:  {str(material).upper()}",
        f"gamma:  {gamma:.1f}  J m⁻³ K⁻²",
        f"C_l:  {cl:.3e}  J m⁻³ K⁻¹",
        f"G:  {g_ep:.3e}  W m⁻³ K⁻¹",
        f"k_e0:  {ke0:.0f}  W m⁻¹ K⁻¹",
        f"k_l:  {kl:.0f}  W m⁻¹ K⁻¹",
        f"alpha_opt:  {alpha_opt:.2e} m⁻¹  (δ = {delta_opt * 1e9:.0f} nm)",
        "", "Laser",
        f"Avg Power:  {pavg:.3g} W",
        f"Rep Rate:  {frep_v:.4g} {frep_u}",
        f"Pulse Energy:  {ep_v:.4g} {ep_u}",
        f"Pulse Width:  {tau_v:.4g} {tau_u}",
        f"Spot Radius:  {spot_v:.4g} {spot_u}",
        f"Fluence:  {f_peak / 1e4:.4g} J/cm²",
        f"Absorbance:  {absorbance:.2f}",
        "", "Results",
        f"Pulses:  {n_pulses}",
        f"Peak T_e:  {te_peak_all - 273.15:.0f} °C  (pulse {peak_pulse})",
        f"Peak T_l:  {tl_peak_all - 273.15:.0f} °C",
        f"Final T_resid:  {tresid_last - 273.15:.1f} °C",
        f"E_abs:  {e_input:.3g} J/m²",
        f"E_depth:  {du_depth:.3g} J/m²",
    ]
    if inv_detected:
        lines += ["", "Inversion",
                  f"Max (T_l - T_e):  {max_inv:.1f} °C",
                  f"At:  {mx_v:.3g} {mx_u}"]
    fig_p.text(0.05, 0.95, "\n".join(lines), fontsize=9.5, family="monospace",
               va="top", bbox={"edgecolor": (0.4, 0.4, 0.4),
                               "facecolor": (0.97, 0.97, 0.97), "pad": 10})
    _save_fig(fig_p, save_dir, "Simulation_Parameters", case_tag)

    # --- Plot 1b: heat accumulation per pulse (bars) ------------------------
    if n_pulses > 1:
        fig1b = plt.figure("Heat_Accumulation_Per_Pulse_Bar",
                           figsize=(8, 4.5), facecolor="w")
        axb = fig1b.add_subplot(111)
        x = np.arange(1, n_pulses + 1)
        width = 0.4
        axb.bar(x - width / 2, teq_vals - 273.15, width,
                color=(0.9, 0.3, 0.2), label="Right After Pulse")
        axb.bar(x + width / 2, tresid_vals - 273.15, width,
                color=(0.2, 0.6, 0.9), label="After Cooling")
        axb.set_xlabel("Pulse #", fontsize=12)
        axb.set_ylabel("Temperature (°C)", fontsize=12)
        axb.legend(loc="upper left", fontsize=9)
        axb.grid(True)
        axb.set_xlim(0.4, n_pulses + 0.6)
        _save_fig(fig1b, save_dir, "Heat_Accumulation_Per_Pulse_Bar", case_tag)

    # --- Plot 2: first-pulse snapshot grid + Plot 3: overlay ----------------
    n_snaps = len(snap_te)
    if n_snaps:
        n_cols = min(4, n_snaps)
        n_rows = int(np.ceil(n_snaps / n_cols))
        z_plot = z_grid * 1e9
        z_max = min(500.0, lz * 1e9)
        fig2, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6.5),
                                  facecolor="w", squeeze=False,
                                  num="Depth_Profiles_First_Pulse_Snapshots")
        for si in range(n_snaps):
            axs = axes[si // n_cols][si % n_cols]
            axs.plot(z_plot, snap_te[si] - 273.15, "b-", lw=1.6, label="$T_e$")
            axs.plot(z_plot, snap_tl[si] - 273.15, "r-", lw=1.6, label="$T_l$")
            axs.set_xlabel("Depth (nm)", fontsize=9)
            axs.set_ylabel("T (°C)", fontsize=9)
            axs.set_title(f"t = {snap_labels[si]}", fontsize=10)
            axs.legend(loc="best", fontsize=7)
            axs.grid(True)
            axs.set_xlim(0, z_max)
        for si in range(n_snaps, n_rows * n_cols):
            axes[si // n_cols][si % n_cols].set_visible(False)
        fig2.tight_layout()
        _save_fig(fig2, save_dir, "Depth_Profiles_First_Pulse_Snapshots", case_tag)

        fig3 = plt.figure("Depth_Profiles_Overlaid_First_Pulse",
                          figsize=(8.5, 5), facecolor="w")
        ax3 = fig3.add_subplot(111)
        cmap = plt.get_cmap("tab10")
        for si in range(n_snaps):
            color = cmap(si % 10)
            ax3.plot(z_plot, snap_te[si] - 273.15, "-", color=color, lw=1.5,
                     label=f"Electron  t={snap_labels[si]}")
            ax3.plot(z_plot, snap_tl[si] - 273.15, "--", color=color, lw=1.2)
        ax3.plot([], [], "k--", lw=1.2, label="Lattice (dashed)")
        ax3.set_xlabel("Depth (nm)", fontsize=12)
        ax3.set_ylabel("Temperature (°C)", fontsize=12)
        ax3.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax3.grid(True)
        ax3.set_xlim(0, z_max)
        fig3.tight_layout()
        _save_fig(fig3, save_dir, "Depth_Profiles_Overlaid_First_Pulse", case_tag)

    # --- Plot 4: multi-pulse heat-accumulation depth profiles ---------------
    n_p_snaps = len(profile_snaps_tz)
    nz_diff = z_grid_diff.size
    if n_p_snaps and n_pulses > 1:
        fig4 = plt.figure("Depth_Profiles_Heat_Accumulation_MultiPulse",
                          figsize=(9, 5), facecolor="w")
        ax4 = fig4.add_subplot(111)
        colors_acc = plt.get_cmap("viridis")(np.linspace(0, 1, n_p_snaps))
        z_diff_um = z_grid_diff * 1e6
        z_max_acc = ldiff * 1e6
        for si in range(n_p_snaps - 1, -1, -1):
            heated = np.flatnonzero(profile_snaps_tz[si] - t0 > 0.5)
            if heated.size:
                z_max_acc = min(z_max_acc,
                                z_grid_diff[min(heated[-1] + 5, nz_diff - 1)] * 1e6)
                break
        z_max_acc = max(z_max_acc, 5.0)
        for si in range(n_p_snaps):
            ax4.plot(z_diff_um, profile_snaps_tz[si] - 273.15, "-",
                     color=colors_acc[si], lw=1.6,
                     label=profile_snaps_label[si])
        ax4.set_xlabel("Depth (μm)", fontsize=12)
        ax4.set_ylabel("Temperature (°C)", fontsize=12)
        ax4.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax4.grid(True)
        ax4.set_xlim(0, z_max_acc)
        fig4.tight_layout()
        _save_fig(fig4, save_dir, "Depth_Profiles_Heat_Accumulation_MultiPulse",
                  case_tag)

    # --- Plots 5-8: radial profiles derived from the depth data -------------
    if enable_radial and n_p_snaps:
        print("\n=== Computing Radial Surface Temperature Profiles (Multi-Pulse) ===")
        r_max_rad = r_max_factor * spot_radius
        r_grid_rad = np.linspace(0.0, r_max_rad, nr_radial)
        r_rad_plot = r_grid_rad * 1e6
        fluence_ratio_rad = np.exp(-2.0 * r_grid_rad**2 / spot_radius**2)

        l_lat = np.sqrt(alpha_diff * sim_duration)
        print(f"  Lateral diffusion length: {l_lat * 1e6:.2f} um  "
              f"(spot radius: {spot_radius * 1e6:.0f} um)")
        if l_lat > 0.1 * spot_radius:
            import warnings

            warnings.warn(
                f"Lateral diffusion ({l_lat * 1e6:.1f} um) is >10% of spot "
                f"radius ({spot_radius * 1e6:.0f} um). Radial scaling "
                "approximation degrades.")

        # Plot 5: radial surface temperature buildup
        fig5 = plt.figure("Radial_Surface_Temperature_Buildup_MultiPulse",
                          figsize=(9, 5), facecolor="w")
        ax5 = fig5.add_subplot(111)
        colors_rad = plt.get_cmap("viridis")(np.linspace(0, 1, n_p_snaps))
        for si in range(n_p_snaps):
            d_t_center = profile_snaps_tz[si][0] - t0
            tsurf_r = t0 + d_t_center * fluence_ratio_rad
            ax5.plot(r_rad_plot, tsurf_r - 273.15, "-",
                     color=colors_rad[si], lw=1.6,
                     label=profile_snaps_label[si])
        ax5.axvline(spot_radius * 1e6, color="k", ls="--", lw=1.2)
        ax5.text(spot_radius * 1e6, ax5.get_ylim()[1], " 1/e² radius",
                 fontsize=9, va="top")
        ax5.set_xlabel("Radial Distance (μm)", fontsize=12)
        ax5.set_ylabel("Temperature (°C)", fontsize=12)
        ax5.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax5.grid(True)
        fig5.tight_layout()
        _save_fig(fig5, save_dir, "Radial_Surface_Temperature_Buildup_MultiPulse",
                  case_tag)

        # Plot 6: final depth x radius cross-section heatmap
        tz_final = profile_snaps_tz[-1]
        d_tz_final = tz_final - t0
        t_rz_final = t0 + np.outer(fluence_ratio_rad, d_tz_final)
        z_max_cs = ldiff * 1e6
        heated = np.flatnonzero(d_tz_final > 0.5)
        if heated.size:
            z_max_cs = min(z_max_cs,
                           z_grid_diff[min(heated[-1] + 5, nz_diff - 1)] * 1e6)
        z_max_cs = max(z_max_cs, 5.0)

        fig6 = plt.figure("Cross_Section_Depth_vs_Radius_Final",
                          figsize=(9, 5), facecolor="w")
        ax6 = fig6.add_subplot(111)
        z_diff_um = z_grid_diff * 1e6
        pm = ax6.pcolormesh(r_rad_plot, z_diff_um, t_rz_final.T - 273.15,
                            cmap="hot", shading="gouraud")
        cb = fig6.colorbar(pm, ax=ax6)
        cb.set_label("Temperature (°C)", fontsize=11)
        ax6.set_ylim(0, z_max_cs)
        ax6.invert_yaxis()
        ax6.axvline(spot_radius * 1e6, color="w", ls="--", lw=1.5)
        ax6.set_xlabel("Radial Distance (μm)", fontsize=12)
        ax6.set_ylabel("Depth (μm)", fontsize=12)
        fig6.tight_layout()
        _save_fig(fig6, save_dir, "Cross_Section_Depth_vs_Radius_Final", case_tag)

        # Plot 7: cross-section evolution over pulses
        if n_p_snaps > 1:
            n_cols_cs = min(4, n_p_snaps)
            n_rows_cs = int(np.ceil(n_p_snaps / n_cols_cs))
            fig7, axes7 = plt.subplots(
                n_rows_cs, n_cols_cs,
                figsize=(min(14, 3.5 * n_cols_cs), min(9, 2.8 * n_rows_cs)),
                facecolor="w", squeeze=False,
                num="Cross_Section_Evolution_Over_Pulses")
            c_lim_lo = t0 - 273.15
            c_lim_hi = float(np.max(profile_snaps_tz[-1])) - 273.15
            pm7 = None
            for si in range(n_p_snaps):
                ax7 = axes7[si // n_cols_cs][si % n_cols_cs]
                t_rz_si = t0 + np.outer(fluence_ratio_rad,
                                        profile_snaps_tz[si] - t0)
                pm7 = ax7.pcolormesh(r_rad_plot, z_diff_um, t_rz_si.T - 273.15,
                                     cmap="hot", shading="gouraud",
                                     vmin=c_lim_lo, vmax=c_lim_hi)
                ax7.set_ylim(0, z_max_cs)
                ax7.invert_yaxis()
                ax7.set_xlabel("r (μm)", fontsize=9)
                ax7.set_ylabel("z (μm)", fontsize=9)
                ax7.set_title(profile_snaps_label[si], fontsize=10)
            for si in range(n_p_snaps, n_rows_cs * n_cols_cs):
                axes7[si // n_cols_cs][si % n_cols_cs].set_visible(False)
            fig7.tight_layout()
            if pm7 is not None:
                cb7 = fig7.colorbar(pm7, ax=axes7, fraction=0.03, pad=0.02)
                cb7.set_label("Temperature (°C)", fontsize=10)
            _save_fig(fig7, save_dir, "Cross_Section_Evolution_Over_Pulses",
                      case_tag)

        # Plot 8: residual temperature vs pulse at multiple radii
        if n_pulses > 1:
            r_sample_factors = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
            r_sample_labels = ["r = 0 (center)", "r = 0.5w", "r = 1.0w",
                               "r = 1.5w", "r = 2.0w"]
            fluence_at_sample = np.exp(-2.0 * r_sample_factors**2)
            fig8 = plt.figure("Residual_Temperature_vs_Pulse_Multiple_Radii",
                              figsize=(9, 4.8), facecolor="w")
            ax8 = fig8.add_subplot(111)
            cmap8 = plt.get_cmap("tab10")
            pulses = np.arange(1, n_pulses + 1)
            for ri in range(r_sample_factors.size):
                tresid_r = t0 + (tresid_vals - t0) * fluence_at_sample[ri]
                ax8.plot(pulses, tresid_r - 273.15, "-",
                         color=cmap8(ri), lw=1.4,
                         label=f"{r_sample_labels[ri]}  "
                               f"({r_sample_factors[ri] * spot_radius * 1e6:.0f} um)")
            ax8.set_xlabel("Pulse Number", fontsize=12)
            ax8.set_ylabel("Residual Temperature (°C)", fontsize=12)
            ax8.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
            ax8.grid(True)
            fig8.tight_layout()
            _save_fig(fig8, save_dir,
                      "Residual_Temperature_vs_Pulse_Multiple_Radii", case_tag)

        print("  Radial profiles computed (no extra ODE solves needed).")


def plot_radial_profile(*, all_times, all_tl_surf, r_plot_um, r_grid,
                        snap_radial_profiles, profile_snaps_label,
                        tresid_radial, teq_vals, tresid_vals,
                        spot_radius, r_max, material, mode,
                        gamma, cl, g_ep, kl,
                        pavg, frep_v, frep_u, ep_v, ep_u, tau_v, tau_u,
                        spot_v, spot_u, f_peak, absorbance,
                        n_pulses, wall_time, t0_c,
                        save_dir=None, case_tag=""):
    """Port of the Radial_Profile_Solver figure set (plots 1-5)."""
    import matplotlib.pyplot as plt

    # --- Plot 1: centre timeline + parameter panel --------------------------
    fig1 = plt.figure("Radial_TTM_Center_Temperature_Timeline",
                      figsize=(12, 6), facecolor="w")
    ax1 = fig1.add_axes((0.07, 0.12, 0.42, 0.78))
    t_scale, t_unit = _time_scale(all_times[-1])
    ax1.plot(all_times * t_scale, all_tl_surf - 273.15, "r-", lw=1.5,
             label="Lattice (center)")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_xlabel(f"Time ({t_unit})", fontsize=12)
    ax1.set_ylabel("Temperature (°C)", fontsize=12)
    ax1.set_title("Surface Temperature at Beam Center", fontsize=13)
    ax1.grid(True)

    param_lines = [
        "Radial TTM Parameters", "",
        f"Material:  {str(material).upper()}",
        f"Mode:  {mode}",
        f"gamma:  {gamma:.1f}  J m⁻³ K⁻²",
        f"C_l:  {cl:.3e}  J m⁻³ K⁻¹",
        f"G:  {g_ep:.3e}  W m⁻³ K⁻¹",
        f"k_l:  {kl:.1f}  W m⁻¹ K⁻¹",
        "", "Laser",
        f"Avg Power:  {pavg:.3g} W",
        f"Rep Rate:  {frep_v:.4g} {frep_u}",
        f"Pulse Energy:  {ep_v:.4g} {ep_u}",
        f"Pulse Width:  {tau_v:.4g} {tau_u}",
        f"Spot Radius:  {spot_v:.4g} {spot_u}",
        f"Fluence:  {f_peak / 1e4:.4g} J/cm²",
        f"Absorbance:  {absorbance:.2f}",
        "", "Results",
        f"Pulses:  {n_pulses}",
        f"Peak T_eq:  {teq_vals.max() - 273.15:.0f} °C",
        f"Final T_resid:  {tresid_vals[-1] - 273.15:.1f} °C",
        f"Wall time:  {wall_time:.2f} s",
    ]
    fig1.text(0.56, 0.90, "\n".join(param_lines), fontsize=9,
              family="monospace", va="top",
              bbox={"edgecolor": (0.4, 0.4, 0.4),
                    "facecolor": (0.97, 0.97, 0.97), "pad": 8})
    _save_fig(fig1, save_dir, "Radial_TTM_Center_Temperature_Timeline", case_tag)

    # --- Plot 2: radial surface temperature buildup -------------------------
    n_snaps = snap_radial_profiles.shape[0]
    if n_snaps > 0:
        fig2 = plt.figure("Radial_Surface_Temperature_Buildup",
                          figsize=(9, 5), facecolor="w")
        ax2 = fig2.add_subplot(111)
        colors_rad = plt.get_cmap("viridis")(np.linspace(0, 1, n_snaps))
        for si in range(n_snaps):
            ax2.plot(r_plot_um, snap_radial_profiles[si] - 273.15, "-",
                     color=colors_rad[si], lw=1.6,
                     label=profile_snaps_label[si])
        ax2.plot(r_plot_um, tresid_radial[-1] - 273.15, "k-", lw=2.0,
                 label=f"Final (pulse {n_pulses})")
        ax2.axvline(spot_radius * 1e6, color="k", ls="--", lw=1.2)
        ax2.text(spot_radius * 1e6, ax2.get_ylim()[1], " 1/e² radius",
                 fontsize=9, va="top")
        ax2.set_xlabel("Radial Distance (μm)", fontsize=12)
        ax2.set_ylabel("Temperature (°C)", fontsize=12)
        ax2.set_title(f"Radial Surface Temperature Buildup - "
                      f"{str(material).upper()}  ({n_pulses} pulses, "
                      f"{frep_v:.4g} {frep_u})", fontsize=12)
        ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax2.grid(True)
        fig2.tight_layout()
        _save_fig(fig2, save_dir, "Radial_Surface_Temperature_Buildup", case_tag)

    # --- Plot 3: top-down 2D heatmap of the final state ---------------------
    fig3 = plt.figure("Surface_Temperature_Heatmap_Top_Down",
                      figsize=(7, 6.2), facecolor="w")
    ax3 = fig3.add_subplot(111)
    final_radial_c = tresid_radial[-1] - 273.15
    n_xy = 201
    xy_lim = r_max * 1e6
    x_lin = np.linspace(-xy_lim, xy_lim, n_xy)
    y_lin = np.linspace(-xy_lim, xy_lim, n_xy)
    xm, ym = np.meshgrid(x_lin, y_lin)
    rm = np.sqrt(xm**2 + ym**2)
    t2d = np.interp(rm.ravel(), r_plot_um, final_radial_c,
                    right=t0_c).reshape(n_xy, n_xy)
    im = ax3.imshow(t2d, extent=(-xy_lim, xy_lim, -xy_lim, xy_lim),
                    origin="lower", cmap="hot", aspect="equal")
    cb = fig3.colorbar(im, ax=ax3)
    cb.set_label("Temperature (°C)", fontsize=11)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax3.plot(spot_radius * 1e6 * np.cos(theta),
             spot_radius * 1e6 * np.sin(theta), "w--", lw=1.5)
    ax3.text(spot_radius * 1e6 * 0.7, spot_radius * 1e6 * 0.7, "1/e²",
             color="w", fontsize=9, fontweight="bold")
    ax3.set_xlabel("x (μm)", fontsize=12)
    ax3.set_ylabel("y (μm)", fontsize=12)
    ax3.set_title(f"Surface Temperature After {n_pulses} Pulses - "
                  f"{str(material).upper()} ({frep_v:.4g} {frep_u})", fontsize=12)
    fig3.tight_layout()
    _save_fig(fig3, save_dir, "Surface_Temperature_Heatmap_Top_Down", case_tag)

    # --- Plot 4: heat accumulation at multiple radii ------------------------
    if n_pulses > 1:
        r_sample_factors = [0.0, 0.5, 1.0, 1.5, 2.0]
        r_sample_labels = ["r = 0 (center)", "r = 0.5w", "r = 1.0w",
                           "r = 1.5w", "r = 2.0w"]
        fig4 = plt.figure("Heat_Accumulation_at_Multiple_Radii",
                          figsize=(9, 4.8), facecolor="w")
        ax4 = fig4.add_subplot(111)
        cmap4 = plt.get_cmap("tab10")
        pulses = np.arange(1, n_pulses + 1)
        for ri, factor in enumerate(r_sample_factors):
            i_nearest = int(np.argmin(np.abs(r_grid - factor * spot_radius)))
            ax4.plot(pulses, tresid_radial[:, i_nearest] - 273.15, "-",
                     color=cmap4(ri), lw=1.4,
                     label=f"{r_sample_labels[ri]}  "
                           f"({r_grid[i_nearest] * 1e6:.0f} um)")
        ax4.set_xlabel("Pulse Number", fontsize=12)
        ax4.set_ylabel("Residual Temperature (°C)", fontsize=12)
        ax4.set_title(f"Heat Accumulation at Different Radii - "
                      f"{str(material).upper()}  ({frep_v:.4g} {frep_u})",
                      fontsize=12)
        ax4.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax4.grid(True)
        fig4.tight_layout()
        _save_fig(fig4, save_dir, "Heat_Accumulation_at_Multiple_Radii", case_tag)

    # --- Plot 5: heat buildup per pulse (centre, bars) ----------------------
    if n_pulses > 1:
        fig5 = plt.figure("Heat_Buildup_Per_Pulse_Center",
                          figsize=(8, 4.5), facecolor="w")
        ax5 = fig5.add_subplot(111)
        x = np.arange(1, n_pulses + 1)
        width = 0.4
        ax5.bar(x - width / 2, teq_vals - 273.15, width,
                color=(0.9, 0.3, 0.2), label="Right After Pulse")
        ax5.bar(x + width / 2, tresid_vals - 273.15, width,
                color=(0.2, 0.6, 0.9), label="After Cooling")
        ax5.set_xlabel("Pulse #", fontsize=12)
        ax5.set_ylabel("Temperature (°C)", fontsize=12)
        ax5.set_title(f"Heat Buildup Per Pulse (Center) - "
                      f"{str(material).upper()}", fontsize=12)
        ax5.legend(loc="upper left", fontsize=9)
        ax5.grid(True)
        ax5.set_xlim(0.4, n_pulses + 0.6)
        _save_fig(fig5, save_dir, "Heat_Buildup_Per_Pulse_Center", case_tag)


def plot_inversion_quantifier(*, n_pulses, inv_max, tbase, te_peak, tl_peak,
                              te_at_inv, tl_at_inv, t_max_inv, t_onset, inv_dur,
                              has_inversion, n_inv_pulses,
                              save_dir=None, case_tag=""):
    """Port of the Inversion_Quantifier figure set (plots 1-4)."""
    import matplotlib.pyplot as plt

    pulses = np.arange(1, n_pulses + 1)

    # --- Plot 1: inversion magnitude vs pulse with base temperature ---------
    fig1 = plt.figure("Inversion_Magnitude_vs_Pulse", figsize=(9, 4.8),
                      facecolor="w")
    ax1 = fig1.add_subplot(111)
    ax1.plot(pulses, inv_max, "b-o", ms=3, lw=1.3, mfc="b",
             label="Inversion (Tl - Te)")
    ax1.set_ylabel("Inversion Magnitude (K)", fontsize=12, color="b")
    ax1.set_xlabel("Pulse Number", fontsize=12)
    ax1.grid(True)
    ax1r = ax1.twinx()
    ax1r.plot(pulses, tbase, "r-", lw=1.3, label="Base Temperature")
    ax1r.set_ylabel("Base Temperature (°C)", fontsize=12, color="r")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    fig1.tight_layout()
    _save_fig(fig1, save_dir, "Inversion_Magnitude_vs_Pulse", case_tag)

    # --- Plot 2: inversion vs base temperature (correlation) ----------------
    if n_inv_pulses > 2:
        fig2 = plt.figure("Inversion_vs_Base_Temperature", figsize=(7.5, 4.8),
                          facecolor="w")
        ax2 = fig2.add_subplot(111)
        inv_pulse_idx = np.flatnonzero(has_inversion) + 1
        sc = ax2.scatter(tbase[has_inversion], inv_max[has_inversion],
                         s=30, c=inv_pulse_idx, cmap="viridis")
        cb = fig2.colorbar(sc, ax=ax2)
        cb.set_label("Pulse Number", fontsize=11)
        ax2.set_xlabel("Base Temperature (°C)", fontsize=12)
        ax2.set_ylabel("Inversion Magnitude (K)", fontsize=12)
        ax2.grid(True)
        fig2.tight_layout()
        _save_fig(fig2, save_dir, "Inversion_vs_Base_Temperature", case_tag)

    # --- Plot 3: electron and lattice dynamics per pulse --------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 5), facecolor="w",
                                      num="Electron_Lattice_Dynamics_Per_Pulse")
    ax3a.plot(pulses, te_peak, "b-", lw=1.4, label="Peak $T_e$")
    ax3a.plot(pulses, tl_peak, "r-", lw=1.4, label="Peak $T_l$")
    ax3a.plot(pulses, tbase, "k--", lw=1.0, label="$T_{base}$")
    if n_inv_pulses > 0:
        valid_idx = np.flatnonzero(~np.isnan(te_at_inv)) + 1
        ax3a.plot(valid_idx, te_at_inv[~np.isnan(te_at_inv)], "m:", lw=1.2,
                  label="$T_e$ at max inv")
        ax3a.plot(valid_idx, tl_at_inv[~np.isnan(tl_at_inv)], "g:", lw=1.2,
                  label="$T_l$ at max inv")
    ax3a.set_xlabel("Pulse Number", fontsize=11)
    ax3a.set_ylabel("Temperature (°C)", fontsize=11)
    ax3a.legend(loc="best", fontsize=8)
    ax3a.grid(True)

    ax3b.plot(pulses, te_peak - tbase, "b-", lw=1.4,
              label=r"$\Delta T_e$ (peak - base)")
    ax3b.plot(pulses, tl_peak - tbase, "r-", lw=1.4,
              label=r"$\Delta T_l$ (peak - base)")
    ax3b.plot(pulses, inv_max, "m-", lw=1.4, label="Inversion ($T_l$ - $T_e$)")
    ax3b.set_xlabel("Pulse Number", fontsize=11)
    ax3b.set_ylabel("Temperature Change (K)", fontsize=11)
    ax3b.legend(loc="best", fontsize=8)
    ax3b.grid(True)
    fig3.tight_layout()
    _save_fig(fig3, save_dir, "Electron_Lattice_Dynamics_Per_Pulse", case_tag)

    # --- Plot 4: inversion timing and duration vs pulse ---------------------
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8.5, 5), facecolor="w",
                                      num="Inversion_Timing_vs_Pulse")
    valid_mask = ~np.isnan(t_max_inv)
    if valid_mask.any():
        t_arr = t_max_inv[valid_mask]
        scale, unit = _time_scale(float(np.max(np.abs(t_arr))) * 1.0001)
        ax4a.plot(np.flatnonzero(valid_mask) + 1, t_arr * scale, "b-o",
                  ms=3, lw=1.2, mfc="b", label="Time of max inversion")
        valid_onset = ~np.isnan(t_onset)
        if valid_onset.any():
            ax4a.plot(np.flatnonzero(valid_onset) + 1,
                      t_onset[valid_onset] * scale, "g-s", ms=3, lw=1.0,
                      mfc="g", label="Inversion onset")
        ax4a.set_ylabel(f"Time After Pulse ({unit})", fontsize=11)
        ax4a.legend(loc="best", fontsize=8)
    ax4a.grid(True)

    valid_dur = inv_dur > 0
    if valid_dur.any():
        d_arr = inv_dur[valid_dur]
        scale, unit = _time_scale(float(np.max(np.abs(d_arr))) * 1.0001)
        ax4b.plot(np.flatnonzero(valid_dur) + 1, d_arr * scale, "r-o",
                  ms=3, lw=1.2, mfc="r", label="Inversion duration")
        ax4b.set_ylabel(f"Duration ({unit})", fontsize=11)
        ax4b.legend(loc="best", fontsize=8)
    ax4b.set_xlabel("Pulse Number", fontsize=11)
    ax4b.grid(True)
    fig4.tight_layout()
    _save_fig(fig4, save_dir, "Inversion_Timing_vs_Pulse", case_tag)


def plot_scanning_beam(*, x_grid, y_grid, tpeak_map, peak_t_history, iy_center,
                       scan_length, spot_radius, pulse_spacing, n_pulses,
                       material, v_scan, pavg, frep_v, frep_u,
                       output_dir, base_name):
    """Port of the Scanning_Beam_Solver figures (4 PNGs saved directly)."""
    import matplotlib.pyplot as plt

    x_um = x_grid * 1e6
    y_um = y_grid * 1e6
    mat = str(material).upper()

    # --- Plot 1: peak temperature heatmap -----------------------------------
    fig1 = plt.figure(figsize=(12, 5), facecolor="w")
    ax1 = fig1.add_subplot(111)
    im = ax1.imshow(tpeak_map - 273.15,
                    extent=(x_um[0], x_um[-1], y_um[0], y_um[-1]),
                    origin="lower", cmap="hot", aspect="equal")
    cb = fig1.colorbar(im, ax=ax1)
    cb.set_label("Peak Temperature (°C)", fontsize=11)
    ax1.plot([0, scan_length * 1e6], [0, 0], "w--", lw=1.5)
    ax1.plot(0, 0, "go", ms=10, mew=2, mfc="none")
    ax1.plot(scan_length * 1e6, 0, "rs", ms=10, mew=2, mfc="none")
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(spot_radius * 1e6 * np.cos(theta),
             spot_radius * 1e6 * np.sin(theta), "g--", lw=1)
    ax1.plot(scan_length * 1e6 + spot_radius * 1e6 * np.cos(theta),
             spot_radius * 1e6 * np.sin(theta), "r--", lw=1)
    ax1.set_xlabel("x (μm) — scan direction", fontsize=12)
    ax1.set_ylabel("y (μm)", fontsize=12)
    ax1.set_title(f"Peak Temperature — {mat}  (v={v_scan:.3g} m/s, "
                  f"P={pavg:.3g}W, f={frep_v:.4g} {frep_u})", fontsize=13)
    fig1.tight_layout()
    fig1.savefig(f"{output_dir}/{base_name}_heatmap.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: peak centre-line profile -----------------------------------
    fig2 = plt.figure(figsize=(10, 5), facecolor="w")
    ax2 = fig2.add_subplot(111)
    ax2.plot(x_um, tpeak_map[iy_center, :] - 273.15, "r-", lw=2)
    ax2.axvline(0, color="g", ls="--", lw=1)
    ax2.axvline(scan_length * 1e6, color="r", ls="--", lw=1)
    ax2.text(0, ax2.get_ylim()[1], " Start", fontsize=9, va="top", color="g")
    ax2.text(scan_length * 1e6, ax2.get_ylim()[1], " End", fontsize=9,
             va="top", color="r")
    ax2.set_xlabel("x (μm) — scan direction", fontsize=12)
    ax2.set_ylabel("Peak Temperature (°C)", fontsize=12)
    ax2.set_title(f"Peak Temperature Along Center — {mat}", fontsize=13)
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(f"{output_dir}/{base_name}_centerline.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: peak cross-sections ----------------------------------------
    fig3 = plt.figure(figsize=(8, 5), facecolor="w")
    ax3 = fig3.add_subplot(111)
    ix_start = int(np.argmin(np.abs(x_grid)))
    ix_mid = int(np.argmin(np.abs(x_grid - scan_length / 2)))
    ix_end = int(np.argmin(np.abs(x_grid - scan_length)))
    ax3.plot(y_um, tpeak_map[:, ix_start] - 273.15, "b-", lw=1.5,
             label=f"x={x_grid[ix_start] * 1e6:.0f}um (start)")
    ax3.plot(y_um, tpeak_map[:, ix_mid] - 273.15, "g-", lw=1.5,
             label=f"x={x_grid[ix_mid] * 1e6:.0f}um (mid)")
    ax3.plot(y_um, tpeak_map[:, ix_end] - 273.15, "r-", lw=1.5,
             label=f"x={x_grid[ix_end] * 1e6:.0f}um (end)")
    ax3.axvline(spot_radius * 1e6, color="k", ls="--", lw=1)
    ax3.axvline(-spot_radius * 1e6, color="k", ls="--", lw=1)
    ax3.text(-spot_radius * 1e6, ax3.get_ylim()[1], " 1/e²", fontsize=9,
             va="top")
    ax3.set_xlabel("y (μm)", fontsize=12)
    ax3.set_ylabel("Peak Temperature (°C)", fontsize=12)
    ax3.set_title(f"Peak Cross-Sections — {mat}", fontsize=13)
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True)
    fig3.tight_layout()
    fig3.savefig(f"{output_dir}/{base_name}_crosssections.png", dpi=150)
    plt.close(fig3)

    # --- Plot 4: peak temperature vs laser position -------------------------
    fig4 = plt.figure(figsize=(8, 4), facecolor="w")
    ax4 = fig4.add_subplot(111)
    x_laser_um = np.arange(n_pulses) * pulse_spacing * 1e6
    ax4.plot(x_laser_um, peak_t_history - 273.15, "r-", lw=1.5)
    ax4.set_xlabel("Laser Position (μm)", fontsize=12)
    ax4.set_ylabel("Instantaneous Peak T (°C)", fontsize=12)
    ax4.set_title("Peak Temperature vs Laser Position", fontsize=13)
    ax4.grid(True)
    fig4.tight_layout()
    fig4.savefig(f"{output_dir}/{base_name}_peakVsPos.png", dpi=150)
    plt.close(fig4)


def plot_surface_point(*, times, tl, t_end, teq_vals, tresid_vals,
                       baseline_fit_ok, baseline_fit_y, extrap_times_s, t_ss_c,
                       material, gamma, cl, g_ep, kl,
                       pavg, frep_val, frep_unit, ep_val, ep_unit,
                       tau_val, tau_unit, spot_val, spot_unit, f_si,
                       absorbance, te_peak, tl_peak, peak_pulse,
                       absorbed_areal, du_depth, nt,
                       save_path=None):
    """Port of the Surface_Point_Solver temperature-evolution figure."""
    import matplotlib.pyplot as plt

    n_pulses = teq_vals.size
    fig = plt.figure("Surface TTM — Temperature Evolution",
                     figsize=(12, 6), facecolor="w")

    # --- Left panel: full continuous timeline -----------------------------
    ax1 = fig.add_axes((0.07, 0.12, 0.42, 0.78))
    t_scale, t_unit = _time_scale(t_end)
    t_plot = times[:nt] * t_scale

    ax1.plot(t_plot, tl[:nt] - 273.15, "r-", lw=1.5, label="$T_l$ (Lattice)")
    if baseline_fit_ok and baseline_fit_y is not None:
        ax1.plot(extrap_times_s[:n_pulses] * t_scale, baseline_fit_y - 273.15,
                 "--", color=(0.1, 0.5, 0.1), lw=2,
                 label=f"Baseline Fit → {t_ss_c:.1f} °C")
        ax1.axhline(t_ss_c, ls=":", color=(0.1, 0.5, 0.1), lw=1.2,
                    label="Projected $T_{ss}$")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_xlabel(f"Time ({t_unit})", fontsize=12)
    ax1.set_ylabel("Temperature (°C)", fontsize=12)
    ax1.set_title("Two-Temperature Model — Full Timeline", fontsize=13)
    ax1.grid(True)

    # --- Right-top: Teq vs Tresid per pulse -------------------------------
    ax2 = fig.add_axes((0.56, 0.55, 0.15, 0.35))
    x = np.arange(1, n_pulses + 1)
    width = 0.4
    ax2.bar(x - width / 2, teq_vals - 273.15, width,
            color=(0.9, 0.3, 0.2), label="$T_{eq}$ (post-pulse)")
    ax2.bar(x + width / 2, tresid_vals - 273.15, width,
            color=(0.2, 0.6, 0.9), label="$T_{resid}$ (after diff.)")
    ax2.set_xlabel("Pulse #", fontsize=10)
    ax2.set_ylabel("Temperature (°C)", fontsize=10)
    ax2.set_title("Accumulation & Cooling", fontsize=11)
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True)
    ax2.set_xlim(0.4, n_pulses + 0.6)

    # --- Parameter panel (right-bottom) -----------------------------------
    param_lines = [
        "Parameters",
        "",
        f"Material:  {str(material).upper()}",
        f"gamma:  {gamma:.1f}  J m$^{{-3}}$ K$^{{-2}}$",
        f"C$_l$:  {cl:.3e}  J m$^{{-3}}$ K$^{{-1}}$",
        f"G:  {g_ep:.3e}  W m$^{{-3}}$ K$^{{-1}}$",
        f"k$_l$:  {kl:.1f}  W m$^{{-1}}$ K$^{{-1}}$",
        "",
        "Laser",
        f"Avg Power:  {pavg:.3g} W",
        f"Rep Rate:  {frep_val:.4g} {frep_unit}",
        f"Pulse Energy:  {ep_val:.4g} {ep_unit}",
        f"Pulse Width:  {tau_val:.4g} {tau_unit}",
        f"Spot Radius:  {spot_val:.4g} {spot_unit}",
        f"Fluence:  {f_si / 1e4:.4g} J/cm²",
        f"Absorbance:  {absorbance:.2f}",
        "",
        "Results",
        f"Peak T$_e$:  {te_peak - 273.15:.1f} °C  (pulse {peak_pulse})",
        f"Peak T$_l$:  {tl_peak - 273.15:.1f} °C",
        f"Final T$_{{eq}}$:  {teq_vals[-1] - 273.15:.1f} °C",
        f"Final T$_{{resid}}$:  {tresid_vals[-1] - 273.15:.1f} °C",
        f"T$_{{ss}}$ (projected):  {t_ss_c:.1f} °C",
        f"E$_{{abs}}$:  {absorbed_areal:.3g} J/m²",
        f"E$_{{depth}}$:  {du_depth:.3g} J/m²",
        f"Time Steps:  {nt}",
    ]
    fig.text(0.57, 0.46, "\n".join(param_lines), fontsize=8,
             family="monospace", va="top",
             bbox={"edgecolor": (0.4, 0.4, 0.4),
                   "facecolor": (0.97, 0.97, 0.97), "pad": 8})

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        return save_path
    return None
