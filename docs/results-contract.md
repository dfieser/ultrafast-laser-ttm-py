# Results contract

Every key a solver returns, in every release. Generated from the
registry in `src/laserttm/schema.py` by
`python docs/generate_contract.py`, and checked against the live
solvers by the test suite, so it cannot drift from the code. Do
not edit by hand.

The contract is additive: within contract version v1 no key is
renamed, removed, or changed in meaning, and new keys may appear
in any release. The same information is available at runtime:

```python
import laserttm
laserttm.describe_results("depth_profile")
```

```bash
laserttm describe depth_profile --section results
```

and over MCP through the `describe_solver` tool.

All temperatures are Celsius except temperature differences,
which are Kelvin. A NaN marks a quantity that did not occur in
the run, such as an inversion that never happened.

## Shared envelope

These keys appear in every solver's results.

| Key | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `solver` | str |  | Human-readable solver name. |
| `solverId` | str |  | The id used by run(), the CLI and MCP. |
| `contractVersion` | str |  | Results-contract generation, 'v1' since the first release. |
| `material` | str |  | Material key as given in the config. |
| `caseTag` | str |  | Sanitized caseTag echoed back, empty when unset. |
| `resolvedConfig` | dict |  | The config actually in force: defaults overlaid with the caller's values under the same empty-means-default semantics the solver itself applies. |
| `materialProps` | dict |  | The resolved material record: gamma, Cl, G, the two conductivity terms spelled apart as kTotal_W_mK and kLattice_W_mK, optical properties, melting point, and the conductivity model in use. None only when a pre-0.1.22 depthResults dict was supplied. |
| `warnings` | list |  | Validity warnings raised during the run, for consumers that never see the console. Includes a peak lattice temperature above the melting point, which the model cannot represent. |
| `nPulses` | scalar |  | Number of pulses simulated. |
| `wallTime_s` | scalar | s | Wall-clock solve time. |
| `outputFile` | path |  | The text report written by this run. |
| `outputDir` | path |  | Directory holding every file written. |
| `inputConfig` | dict |  | The caller's config exactly as passed, unmerged. Kept for compatibility, prefer `resolvedConfig`. |

## surface_point

Surface point.

| Key | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `time_s` | array | s | Sample times over the whole train; empty when storeHistory is off. |
| `Te_K` | array | K | Electron temperature at each sample. |
| `Tl_K` | array | K | Lattice temperature at each sample. |
| `Te_C` | array | degC | Electron temperature, Celsius. |
| `Tl_C` | array | degC | Lattice temperature, Celsius. |
| `peakPulse` | scalar |  | 1-based pulse on which the electron peak occurred. |
| `peakTe_C` | scalar | degC | Peak electron temperature. |
| `peakTl_C` | scalar | degC | Peak lattice temperature. |
| `finalTe_C` | scalar | degC | Electron temperature at the last sample. |
| `finalTl_C` | scalar | degC | Lattice temperature at the last sample. |
| `finalResid_C` | scalar | degC | Residual surface temperature after the final inter-pulse diffusion. |
| `TeqVals_C` | array | degC | Post-pulse equilibrium temperature, one per pulse. |
| `TresidVals_C` | array | degC | Residual temperature after each inter-pulse diffusion. |
| `absorbedAreal_J_m2` | scalar | J/m^2 | Energy absorbed per unit area over the run. |
| `depthEnergy_J_m2` | scalar | J/m^2 | Energy stored in the depth grid at the end. |
| `energyMismatch_pct` | scalar | % | Bookkeeping mismatch between the two; expected to be nonzero in this hybrid 0D+1D model. |
| `makePlots` | bool |  | Echo of the plotting switch. Kept for compatibility, prefer `resolvedConfig`. |
| `saveFigures` | bool |  | Echo of the figure switch. Kept for compatibility, prefer `resolvedConfig`. |
| `figureFile` | path |  | The saved timeline figure. Present only when `makePlots` is enabled. |

## depth_profile

Depth profile.

| Key | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `peakPulse` | scalar |  | 1-based pulse on which the electron peak occurred. |
| `peakTe_C` | scalar | degC | Peak surface electron temperature. |
| `peakTl_C` | scalar | degC | Peak surface lattice temperature. |
| `finalResid_C` | scalar | degC | Residual surface temperature after the final diffusion. |
| `invDetected` | bool |  | Whether any surface inversion (Tl > Te) exceeded the threshold. |
| `maxInv_K` | scalar | K | Largest surface Tl - Te over the run. Zero when none. |
| `invThreshold_K` | scalar | K | Threshold above which Tl - Te counts as an inversion. |
| `absorbedAreal_J_m2` | scalar | J/m^2 | Energy absorbed per unit area over the run. |
| `depthEnergy_J_m2` | scalar | J/m^2 | Energy stored in the diffusion grid at the end. |
| `energyMismatch_pct` | scalar | % | Bookkeeping mismatch between the two; expected in the hybrid fine+coarse model. |
| `TePeakPerPulse_C` | array | degC | Peak surface electron temperature, one per pulse. |
| `TlPeakPerPulse_C` | array | degC | Peak surface lattice temperature, one per pulse. |
| `TeqVals_C` | array | degC | Post-pulse equilibrium temperature, one per pulse. |
| `TresidVals_C` | array | degC | Residual temperature after each inter-pulse diffusion. |
| `baseTempPerPulse_C` | array | degC | Baseline temperature each pulse started from. |
| `invMaxPerPulse_K` | array | K | Largest surface Tl - Te within each pulse. |
| `tMaxInvPerPulse_s` | array | s | Time of the largest inversion, relative to each pulse centre; NaN where none. |
| `tInvOnsetPerPulse_s` | array | s | Inversion onset time relative to each pulse centre; NaN where none. |
| `invDurationPerPulse_s` | array | s | How long the inversion lasted in each pulse. Zero where none. |
| `Te_atMaxInvPerPulse_C` | array | degC | Electron temperature at the moment of largest inversion; NaN where none. |
| `Tl_atMaxInvPerPulse_C` | array | degC | Lattice temperature at the moment of largest inversion; NaN where none. |
| `f_rep` | scalar | Hz | Echo of the repetition rate. Kept for compatibility, prefer `resolvedConfig`. |
| `Pavg` | scalar | W | Echo of the average power. Kept for compatibility, prefer `resolvedConfig`. |
| `tau_FWHM` | scalar | s | Echo of the pulse width. Kept for compatibility, prefer `resolvedConfig`. |
| `spotRadius` | scalar | m | Echo of the spot radius. Kept for compatibility, prefer `resolvedConfig`. |
| `absorbance` | scalar |  | Echo of the absorbance. Kept for compatibility, prefer `resolvedConfig`. |
| `T0_C` | scalar | degC | Echo of the initial temperature. Kept for compatibility, prefer `resolvedConfig`. |
| `F_peak` | scalar | J/m^2 | Peak fluence at beam centre. |
| `gamma` | scalar | J/(m^3 K^2) | Electron heat-capacity coefficient used. Kept for compatibility, prefer `materialProps`. |
| `Cl` | scalar | J/(m^3 K) | Lattice heat capacity used. Kept for compatibility, prefer `materialProps`. |
| `G` | scalar | W/(m^3 K) | Electron-phonon coupling used. Kept for compatibility, prefer `materialProps`. |
| `ke0` | scalar | W/(m K) | Electron conductivity coefficient used. Kept for compatibility, prefer `materialProps`. |
| `kl` | scalar | W/(m K) | Lattice conductivity term used; the lattice-only share here, unlike the total the 0D solvers call kl. Kept for compatibility, prefer `materialProps`. |
| `alpha_opt` | scalar | 1/m | Optical absorption coefficient used. Kept for compatibility, prefer `materialProps`. |
| `Trep` | scalar | s | Pulse period, 1/f_rep. |
| `simDuration` | scalar | s | Simulated duration. Kept for compatibility, prefer `resolvedConfig`. |
| `radialGrid_um` | array | um | Radial positions of the scaled view. Present only when `enableRadialProfile` is enabled. |
| `radialFluenceRatio` | array |  | Gaussian fluence ratio at each radius. Present only when `enableRadialProfile` is enabled. |
| `radialSurfaceProfiles_C` | array | degC | Surface temperature versus radius at each snapshot pulse. Present only when `enableRadialProfile` is enabled. |
| `crossSection_C` | array | degC | Depth-by-radius temperature map at the last snapshot. Present only when `enableRadialProfile` is enabled. |
| `lateralDiffusionLength_m` | scalar | m | Lateral diffusion length the radial scaling assumes small. Present only when `enableRadialProfile` is enabled. |

## radial_profile

Radial profile.

| Key | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `mode` | str |  | Which radial algorithm ran: 'scale' or 'independent'. |
| `nPulsesRequested` | scalar |  | Pulses the config asked for. nPulses is what actually ran. |
| `earlyStopped` | bool |  | True when the melt-radius early stop ended the run before nPulsesRequested. |
| `peakTeq_C` | scalar | degC | Largest post-pulse equilibrium temperature at beam centre. |
| `finalResid_C` | scalar | degC | Residual centre temperature after the final diffusion. |
| `TeqVals_C` | array | degC | Post-pulse equilibrium temperature at centre, one per pulse. |
| `TresidVals_C` | array | degC | Residual centre temperature after each inter-pulse diffusion. |
| `rGrid_um` | array | um | Radial grid positions. |
| `finalRadialProfile_C` | array | degC | Residual surface temperature versus radius after the last pulse. |
| `spotRadius_um` | scalar | um | Echo of the spot radius. Kept for compatibility, prefer `resolvedConfig`. |

## single_pulse

Single pulse.

| Key | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `peakTe_C` | scalar | degC | Peak surface electron temperature. |
| `peakTl_C` | scalar | degC | Peak surface lattice temperature. |
| `finalTe_C` | scalar | degC | Surface electron temperature at the end. |
| `finalTl_C` | scalar | degC | Surface lattice temperature at the end. |
| `finalResid_C` | scalar | degC | Same value as finalTl_C, under the cross-solver name. |
| `invDetected` | bool |  | Whether the surface inversion exceeded the threshold. |
| `maxInv_C` | scalar | K | Largest surface Tl - Te; a temperature difference despite the historical _C spelling. Kept for compatibility, prefer `maxInv_K`. |
| `maxInv_K` | scalar | K | Largest surface Tl - Te. Zero when none. |
| `invThreshold_K` | scalar | K | Threshold above which Tl - Te counts as an inversion. |
| `tInvOnset_s` | scalar | s | Inversion onset relative to the pulse centre. NaN when none. |
| `tMaxInv_s` | scalar | s | Time of the largest inversion relative to the pulse centre; NaN when none. |
| `absorbedAreal_J_m2` | scalar | J/m^2 | Energy absorbed per unit area. |
| `depthEnergy_J_m2` | scalar | J/m^2 | Energy stored in the depth grid at the end. |
| `energyMismatch_pct` | scalar | % | Bookkeeping mismatch between the two. |

## inversion_quantifier

Inversion analysis.

| Key | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `nInvPulses` | scalar |  | Pulses whose inversion exceeded the threshold. |
| `invThreshold_K` | scalar | K | Threshold above which Tl - Te counts as an inversion. |
| `meanInv_K` | scalar | K | Mean of the per-pulse maximum inversions. NaN when none. |
| `maxInv_K` | scalar | K | Largest inversion in any pulse. |
| `minInv_K` | scalar | K | Smallest inversion among inverted pulses. |
| `stdInv_K` | scalar | K | Standard deviation of the per-pulse maxima. |
| `invSlope_KperPulse` | scalar | K/pulse | Linear trend of inversion magnitude across the train. |
| `corrBaseTempInv` | scalar |  | Correlation between baseline temperature and inversion magnitude. NaN below 3 inverted pulses. |
| `meanInvFraction` | scalar |  | Mean inversion as a fraction of the electron excursion. |
| `invMaxPerPulse_K` | array | K | Largest surface Tl - Te within each pulse. |
| `TePeak_C` | array | degC | Peak surface electron temperature, one per pulse. |
| `TlPeak_C` | array | degC | Peak surface lattice temperature, one per pulse. |
| `Tbase_C` | array | degC | Baseline temperature each pulse started from. |
| `Teq_C` | array | degC | Post-pulse equilibrium temperature, one per pulse. |
| `Tresid_C` | array | degC | Residual temperature after each inter-pulse diffusion. |
| `tMaxInv_s` | array | s | Time of the largest inversion per pulse. NaN where none. |
| `tOnset_s` | array | s | Inversion onset per pulse. NaN where none. |
| `invDuration_s` | array | s | Inversion duration per pulse. Zero where none. |
| `Te_atMaxInv_C` | array | degC | Electron temperature at the largest inversion. NaN where none. |
| `Tl_atMaxInv_C` | array | degC | Lattice temperature at the largest inversion. NaN where none. |
| `peakTe_C` | scalar | degC | Peak surface electron temperature, from the depth run. |
| `peakTl_C` | scalar | degC | Peak surface lattice temperature, from the depth run. |
| `finalResid_C` | scalar | degC | Final residual temperature, from the depth run. |
| `depthResults` | dict |  | The full depth_profile results this analysis ran on. |
| `depthOutputFile` | path |  | The depth run's own text report. |

## scanning_beam

Scanning beam.

| Key | Type | Unit | Meaning |
| --- | --- | --- | --- |
| `Tpeak_map` | array | K | Peak temperature ever reached at each surface point. |
| `Tsurf` | array | K | Final surface temperature map. |
| `peakT_history` | array | K | Peak surface temperature after each pulse. |
| `xGrid` | array | m | Surface grid x positions. |
| `yGrid` | array | m | Surface grid y positions. |
| `peakT_C` | scalar | degC | Largest temperature anywhere on the map. |
| `pulseSpacing` | scalar | m | Distance the beam moves between pulses, v_scan/f_rep. |
| `simDuration_s` | scalar | s | Scan duration, scanLength/v_scan. |
| `wallTime` | scalar | s | Same value as wallTime_s. Kept for compatibility, prefer `wallTime_s`. |
| `dTeq_single` | scalar | K | Single-pulse equilibrium temperature rise used by the superposition. |
| `params` | dict |  | The defaults-merged parameter dict this solver ran from. Kept for compatibility, prefer `resolvedConfig`. |
| `outPath` | path |  | Same value as outputFile. Kept for compatibility, prefer `outputFile`. |
| `matPath` | path |  | MATLAB-compatible .mat with the surface maps. |
