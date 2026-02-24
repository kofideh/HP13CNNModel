The simple way to run the code is in Google Colab.
But if you want to run it locally, install the required packages using the following command:
pip install numpy scipy torch nibabel matplotlib scikit-learn pandas.

To train the model, run the following command:
python train_hybrid_multihead.py  

after specifying acquisition parameters in the code.

To test the model, run the following command:
python robust_clinical_inference.py 

after specifying the weight path in the code.



Project structure:
- fit_two_compartment.py — Traditional 2- and 3-state kinetic fits using `scipy.curve_fit`; exposes `fit_traditional_2state_model`, `fit_traditional_3state_model`, and `fit_traditional_2c_model`. Import these functions from other scripts; there’s no CLI entry point.
- hybrid_model_utils.py — Defines the `HybridMultiHead` PyTorch regressor plus helpers for NIfTI loading, preprocessing, evaluation, and plotting. Import its classes/functions; not meant to be run directly.
- two_compartment_generator.py — Gamma-variate AIF simulator with variable flip angles; `TwoCompartmentHPDataGenerator.generate_dataset()` returns synthetic (pyr, lac) timecourses and ground-truth kPL/kVE/vB. Import and call; no CLI.
- two_compartment_generator_measured.py — Measured-driver simulator (draws pyruvate, propagates lactate) with RF-loss handling; `TwoCompartmentHPDataGeneratorMeasured.generate_dataset()` mirrors the API above. Import and call; no CLI.
- measured_pyr_driver_curvefit.py — Curve-fit kPL/kVE/vB using measured pyruvate as the driver (`fit_measured_driver_curvefit`). Library only.
- measured_pyr_driver_kpl_kve_vb_gain.py — Expanded traditional fit helpers (2- and 3-state) plus measured-driver fitting that also estimates lactate gain (`fit_measured_pyr_driver_kve_vb`). Library only.
- train_hybrid_multihead.py — End-to-end training script: generates synthetic data (gamma or measured driver), trains `HybridMultiHead` with early stopping, writes weights and reports to output. Run from the project root after adjusting top-of-file settings (AIF_TYPE, FA schedules, n_samples, noise):  
  ```bash
  python train_hybrid_multihead.py
  ```
- robust_clinical_inference.py — Runs clinical inference on paired pyruvate/lactate NIfTI stacks, applies the trained hybrid model and traditional fitting, optional NAWM/auto-reference calibration, and writes param maps/logs to `output/…`. Set `weights_dir`, `pyr_files`, `lac_files`, and masks/VIF paths near the top, then run:  
  ```bash
  python robust_clinical_inference.py
  ```
- simulated_data_inference_SNR.py — Stress-test the trained model vs traditional fitting across noise/SNR levels; generates plots, CSVs, and a summary report under `output/…/simulatedData_*`. Ensure `weights_path` points to a trained model, then run:  
  ```bash
  python simulated_data_inference_SNR.py
  ```
- simulated_data_inference_R1variation.py — Varies T1 (R1) for pyr/lac to assess robustness; same outputs/requirements pattern as above. Run:  
  ```bash
  python simulated_data_inference_R1variation.py
  ```
- simulated_data_inference_FlipVariation.py — Varies flip-angle schedules to test model robustness; outputs plots/CSVs. Run:  
  ```bash
  python simulated_data_inference_FlipVariation.py
  ```
- simulated_data_inference_TimeCourses.py — Uses time-course generator with custom parameter ranges; evaluates the model and logs summary metrics. Run:  
  ```bash
  python simulated_data_inference_TimeCourses.py
  ```
- sim_export_enriched_nn.py — CLI tool for enriched simulations and analysis (AUROC, gain, TR sweeps, etc.) with optional NN estimator. Key options: `--experiment`, `--out`, `--estimate nn`, `--nn_weights PATH`. Example:  
  ```bash
  python LacPyrRatio_kPL/sim_export_enriched_nn.py \
    --experiment gain \
    --out results.json \
    --estimate nn \
    --nn_weights output/wts_GammaAIF/trained_hybrid_positive.pth
  ```