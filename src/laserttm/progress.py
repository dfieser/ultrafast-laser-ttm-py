"""Waitbar-style progress popup for the pulse loops.

The MATLAB reference toolbox shows a waitbar with a live time estimate
during long multi-pulse runs. This is the Python analogue, built on
tkinter so it adds no dependencies. Solvers create a ``ProgressReporter``
and call ``update`` once per pulse. Updates are throttled to a few per
second so the reporter costs nothing even at 100,000+ pulses.

Behavior is controlled by the ``showProgress`` config field:

- ``True``: show the popup.
- ``False``: never show it.
- unset: auto mode. The popup appears only when stdout is a terminal, the
  ``LASERTTM_NO_PROGRESS`` environment variable is unset, and a display is
  actually available. Test suites, piped batch jobs, and the MCP server's
  worker processes all fail those checks and stay headless.

Any windowing failure, such as a missing display on a cluster node,
silently disables the popup and the run continues. Closing the popup by
hand also just hides it. The run is never interrupted.
"""

from __future__ import annotations

import os
import sys
import time


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _auto_enabled() -> bool:
    if os.environ.get("LASERTTM_NO_PROGRESS"):
        return False
    try:
        return bool(sys.stdout is not None and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


class ProgressReporter:
    """Throttled tkinter waitbar with an updating time estimate.

    Parameters
    ----------
    total : int
        Total number of pulses.
    title : str
        Window title.
    enabled : bool | None
        ``True``/``False`` to force, ``None`` for auto mode (see module
        docstring).
    min_interval : float
        Minimum seconds between window refreshes.
    create_delay : float
        The window is only created once the run has been going this long,
        so short runs never flash a popup.
    """

    def __init__(self, total: int, title: str = "laserttm",
                 enabled: bool | None = None,
                 min_interval: float = 0.25, create_delay: float = 0.5):
        self.total = max(int(total), 1)
        self.title = title
        self.enabled = _auto_enabled() if enabled is None else bool(enabled)
        self.min_interval = min_interval
        self.create_delay = create_delay
        self._start = time.perf_counter()
        self._last_refresh = 0.0
        self._root = None
        self._bar = None
        self._label = None
        self._eta_label = None

    # ------------------------------------------------------------------ #

    def _create_window(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title(self.title)
        root.resizable(False, False)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", self._on_user_close)

        frame = ttk.Frame(root, padding=14)
        frame.grid(sticky="nsew")
        self._label = ttk.Label(frame, text="", anchor="w")
        self._label.grid(row=0, column=0, sticky="w")
        self._bar = ttk.Progressbar(frame, maximum=self.total,
                                    length=340, mode="determinate")
        self._bar.grid(row=1, column=0, pady=(8, 8), sticky="we")
        self._eta_label = ttk.Label(frame, text="estimating...", anchor="w")
        self._eta_label.grid(row=2, column=0, sticky="w")
        root.update()
        self._root = root

    def _on_user_close(self) -> None:
        self.close()
        self.enabled = False

    # ------------------------------------------------------------------ #

    def update(self, done: int) -> None:
        """Report that ``done`` of ``total`` pulses are complete."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if done < self.total and (now - self._last_refresh) < self.min_interval:
            return
        self._last_refresh = now
        elapsed = now - self._start

        try:
            if self._root is None:
                if elapsed < self.create_delay or done >= self.total:
                    return
                self._create_window()

            pct = 100.0 * done / self.total
            eta = elapsed / done * (self.total - done) if done else 0.0
            self._bar["value"] = done
            self._label.config(
                text=f"Pulse {done:,} of {self.total:,}  ({pct:.0f}%)")
            self._eta_label.config(
                text=f"Elapsed {_format_duration(elapsed)}    "
                     f"Remaining about {_format_duration(eta)}")
            self._root.update()
        except Exception:  # noqa: BLE001 - UI must never break the physics run
            # No display, closed window, interrupted Tcl: run on headless.
            self._root = None
            self.enabled = False

    def close(self) -> None:
        """Destroy the window. Safe to call multiple times."""
        if self._root is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._root.destroy()
            self._root = None

    # Context-manager sugar so solver loops can use ``with``.
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
