#Python environment
The simple way to run the code is in Google Colab.
But if you want to run it locally, install the required packages using the following command:
pip install numpy scipy torch nibabel matplotlib scikit-learn pandas.

To generate all figures shown in the paper for the 15 degree constant flip angle protocol and rat kidney data, run

python train_and_test_protocol.py

 
To generate all figures shown in the paper for the VFA protocol and TRAMP data, run


python train_and_test_protocol.py --training-params training_params_VFA.md


To generate the ROC and paired difference histogram figures for the AUC ratio matching experiment using the NLLS estimator, run

python AUCRatio_kPL.py --estimate nlls --n_pairs_requested 250 --match_tol 0.01 --match_max_tries 5000 --out AUCRatio_kPL_NLLS_pairs.csv  --delta_out AUCRatio_kPL_NLLS_paired_deltas.csv --summary_out AUCRatio_kPL_NLLS_summary.json --plot_out AUCRatio_kPL_NLLS.png



To generate the ROC and paired difference histogram figures for the AUC ratio matching experiment using the NN estimator at output/TrainingReport_TRAMP_VFA_20260521-170940/trained_hybrid_positive.pth, run


python AUCRatio_kPL.py --estimate nn --nn_weights output/<training_report_directory>/trained_hybrid_positive.pth --nn_use_robust_peak --nn_ptrain <P_train_from_training_report> --n_pairs_requested 250 --match_tol 0.01 --match_max_tries 5000 --out AUCRatio_kPL_TRAMP_VFA.csv --delta_out AUCRatio_kPL_TRAMP_VFA.csv --summary_out AUCRatio_kPL_TRAMP_VFA.json --plot_out AUCRatio_kPL_TRAMP_VFA.png


