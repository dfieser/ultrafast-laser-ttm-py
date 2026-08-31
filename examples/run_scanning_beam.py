"""Scanning-beam baseline run.

Python analogue of examples/Example_Scanning_Beam_Baseline.m from the MATLAB
reference toolbox: a 2 mm scan at 1 m/s and 18 MHz (36000 pulses). Figures
are written straight to the output directory, as in MATLAB.
"""

import os

from laserttm import scanning_beam_solver

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

params = {
    # --- Material properties (gamma/Cl/G/kl used only when 'custom') ---
    "material": "W",
    "gamma": 137.3,
    "Cl": 2.54e6,
    "G": 1.65e17,
    "kl": 174,
    # --- Laser parameters ---
    "Pavg": 40,                    # average power [W]
    "spotRadius": 100e-6,          # 1/e^2 spot radius [m]
    "f_rep": 18e6,                 # repetition rate [Hz]
    "tau_FWHM": 100e-15,           # pulse width, FWHM [s]
    "pulseProfile": "gaussian",
    # --- Scan parameters ---
    "v_scan": 1.0,                 # scan speed [m/s]
    "scanLength": 2e-3,            # scan length [m]
    # --- Absorption & geometry ---
    "absorbance": 0.55,
    "Leff": 100e-9,                # heated thickness [m]
    "T0_C": 25,
    # --- 2D surface grid ---
    "Nx": 120,
    "Ny": 60,
    "xPad": 3,
    "yExtent": 5,
    # --- Depth / diffusion controls ---
    "depthProfile": "exponential",
    "dzTarget": 500e-9,
    "Ndiff": 100,
    "NadiPerGap": 10,
}

output_dir = os.path.join(repo_root, "outputs", "Scanning_Single_Run")
results = scanning_beam_solver(params, output_dir, save_plots=True)
