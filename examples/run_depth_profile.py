"""Depth-profile baseline run.

Python analogue of examples/Example_Depth_Profile_Baseline.m from the MATLAB
reference toolbox: same config, same solver defaults, same outputs.
"""

import os

import matplotlib.pyplot as plt

from laserttm import depth_profile_solver

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
    "enableRadialProfile": False,
    "outputDir": os.path.join(repo_root, "outputs", "Example_Depth_Profile"),
}
cfg["simDuration"] = 100 / cfg["f_rep"]   # simulate 100 pulses

results = depth_profile_solver(cfg)
plt.show()
