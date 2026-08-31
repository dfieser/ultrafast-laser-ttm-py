"""Inversion-quantifier run.

Runs the depth solver for 20 pulses and quantifies the electron-lattice
surface temperature inversion (Tl > Te) pulse by pulse.
"""

import os

import matplotlib.pyplot as plt

from laserttm import inversion_quantifier

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
    "outputDir": os.path.join(repo_root, "outputs", "Example_Inversion"),
}
cfg["simDuration"] = 20 / cfg["f_rep"]   # simulate 20 pulses

results = inversion_quantifier(cfg)
plt.show()
