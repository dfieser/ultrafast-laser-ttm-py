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
