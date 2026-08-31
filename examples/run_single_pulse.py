"""Single-pulse visualizer run.

One femtosecond pulse on the fine 1D grid with depth-profile snapshots and
surface-inversion metrics (the conditions of the paper's tungsten case).
"""

import os

import matplotlib.pyplot as plt

from laserttm import single_pulse_visualizer

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = {
    "material": "W",
    "Pavg": 40,                    # average power [W]
    "spotRadius": 100e-6,          # 1/e^2 spot radius [m]
    "f_rep": 18e6,                 # repetition rate [Hz]
    "tau_FWHM": 500e-15,           # pulse width, FWHM [s]
    "absorbance": 0.55,
    "makePlots": True,
    "saveFigures": False,
    "outputDir": os.path.join(repo_root, "outputs", "Example_Single_Pulse"),
}

results = single_pulse_visualizer(cfg)
plt.show()
