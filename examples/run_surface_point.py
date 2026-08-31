"""Surface-point baseline run.

Python analogue of examples/Example_Surface_Point_Baseline.m from the MATLAB
reference toolbox: same config, same solver defaults, same outputs.
"""

import os

import matplotlib.pyplot as plt

from laserttm import surface_point_solver

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = {
    "material": "W",
    "Pavg": 10,                    # average power [W]
    "spotRadius": 100e-6,          # 1/e^2 spot radius [m]
    "f_rep": 5e6,                  # repetition rate [Hz]
    "tau_FWHM": 500e-15,           # pulse width, FWHM [s]
    "absorbance": 0.55,
    "makePlots": True,
    "saveFigures": False,
    "outputDir": os.path.join(repo_root, "outputs", "Example_Surface_Point"),
}
cfg["simDuration"] = 50 / cfg["f_rep"]   # simulate 50 pulses

results = surface_point_solver(cfg)
plt.show()
