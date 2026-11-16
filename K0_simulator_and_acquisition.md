# K0: Simulator and brain acquisition setup

*Written 2026-11-16, from branch `k0-kinetic-adv` (based on `main` at `e3fdcee`).*

This note records where the forward model and the human-brain acquisition settings live in the code, as asked for in `start.md`. The simulator is clear-cut. The brain acquisition setup has a few quirks to keep in mind before starting the checks.

## 1. Simulator

**`two_compartment_generator_physio.py`, class `TwoCompartmentHPDataGeneratorPhysio`**, method `_simulate_physio_model` (line ~150). The NLLS fitter has its own copy of the same equations: `simulate_physio_from_Pv` in `fit_two_compartment_physio_nlls.py:113`.

The model:

- dPe/dt = kVE·Pv − (kVE + kPL + R1P,app)·Pe
- dL/dt = kPL·Pe − R1L,app·L
- S_pyr = vB·Pv + (1−vB)·Pe
- S_lac = (1−vB)·L
- Pe and L both start at 0, and there is no direct lactate input. So setting kPL = 0 should give lactate of exactly zero.

**Input function:** the vascular input Pv is a randomly drawn gamma-variate: amplitude 0.7–1.3, α 2–4, β 0.6–1.2 s⁻¹, t0 0–2 s, rescaled so its peak equals the amplitude. To get a reproducible voxel these values must be fixed. `gamma_variate_input()` in the fitter accepts them directly.

**Fixed values the fitter assumes:** R1P = 1/30 s⁻¹ and R1L = 1/25 s⁻¹.

**Fitter to use for check 3:** `fit_nlls_known_Pv` (`fit_two_compartment_physio_nlls.py:180`). It fits kPL, kVE and vB together, so it will need a small wrapper to fit only kPL.

## 2. Brain acquisition setup

From `training_params_PYR11LAC80.md`, with the matching run record in `training_params_BrainEPI.md`:

| Item | Value |
|---|---|
| Time points | 12 |
| TR | 5.0 s |
| Sample times | 0, 5, 10, …, 55 s (built as `np.arange(0, 12*5, 5)` in `train_and_test_protocol.py:101`) |
| Flip angles | pyruvate 11° and lactate 80°, constant |
| T1 training ranges | pyruvate 20–60 s, lactate 15–55 s |
| kPL / kVE / vB ranges | 0.001–0.06 s⁻¹ / 0.01–0.25 s⁻¹ / 0.005–0.15 |

These match the nominal 5 s TR and 11°/80° flip angles in `start.md`.

## 3. How the code models the RF pulses (relevant to check 2)

1. **The pulses are smeared out over time.** Each pulse is treated as a steady extra signal decay of −ln(cos θ)/TR, spread across the whole TR. There is no instantaneous loss at the moment of the pulse. For lactate at 80° that's about 0.35 s⁻¹, roughly 9 times the T1 decay.
2. **There is no sin θ factor.** The signal is the remaining magnetization itself. "Lactate signal" here is not what an 80° pulse would read out from Mz just before the pulse.
3. **The equations are stepped only at the 12 sample times.** Pv and the RF loss are held at their value from the previous sample across each 5 s step.
4. **The input bolus is short compared with the 5 s spacing.** With typical values (α≈3, β≈0.9), the input peaks about 3 s after onset and is roughly 99% gone by the third sample at 10 s. Combined with point 3, this will probably cause a large discretization error when compared against a fine-step solver. Check 2 should show whether that matters.

## 4. Smaller points

- `training_params_BrainEPI.md` (the run record) doesn't list the T1 ranges. They are assumed to be the same as in `training_params_PYR11LAC80.md`.
- The `data/` folder has no brain data, only rat kidney and TRAMP. `INVIVO_DATA_PATH: data/*Brain*` in `training_params_PYR11LAC80.md` matches no files.

## 5. The "Measured" AIF in the configs

There is no measured AIF anywhere in the current code. `AIF_TYPE: Measured` in the configs is a leftover label, and the input actually used is a simulated gamma-variate.

What `AIF_TYPE: Measured` actually does:

1. `hybrid_model_utils.py:485` reads `AIF_TYPE` from the config file.
2. `train_and_test_protocol.py:107` accepts `"measured"`, `"physio"`, `"physiologic"` or `"physiological"`. All four lead to the same generator, `TwoCompartmentHPDataGeneratorPhysio`. Any other value stops with an error.
3. That generator makes its vascular input by drawing a random gamma-variate (`_draw_vascular_input`, amplitude 0.7–1.3, α 2–4, β 0.6–1.2 s⁻¹, t0 0–2 s). It never loads a measured curve, and nothing else in the code reads a vessel ROI or AIF from data.

Where the label comes from: the two earlier commits (`58e31fb`, `2172cbc`) had a separate `two_compartment_generator_measured.py` (`TwoCompartmentHPDataGeneratorMeasured`). In those versions:

- `train_hybrid_multihead.py` had `AIF_TYPE = 'Measured'` hard-coded. It chose between this generator and a `'GAMMA-Variate'` one.
- The "measured" generator used the voxel's pyruvate signal itself as the input, scaled by kVE/vB: Pe' = (kVE/vB)·S_pyr − [kVE(1−vB)/vB + kPL + R1P,app]·Pe.
- Even there, the pyruvate curve was still a randomly drawn gamma-variate with the same α/β/t0 ranges, so it was never real measured data.

The journal-submission commit `e3fdcee` replaced this with the physio generator but kept the `Measured` label in the configs.

What this means for K0: there's no measured brain AIF to reproduce. For check 1, the input will be a gamma-variate with fixed, documented values. A real measured brain input would have to come from outside this repo, since there's no brain data here either.

## 6. Next step

Check 1: simulate the kPL = 0.0175 s⁻¹ voxel and the kPL = 0 control, using fixed, documented input-function values. Save the time vector, pulse timing, signal scaling and fixed values to a JSON file alongside the run.
