# Hyperpolarized 13C MRI Analysis Demo Summary

**Generator Name:** TwoCompartmentHPDataGeneratorMeasured

**Date:** 2026-03-04 11:18:09

## Dataset
Total samples: 1000000
Training samples: 700000
Validation samples: 150000
Test samples: 150000
Noise level (std): 0.05
## Configuration
NUM_TIME_POINTS: 16
SCAN_TR=2.0
PYR_FA_SCHEDULE: [14.4775 14.9632 15.5014 16.1021 16.7787 17.5484 18.4349 19.4712 20.7048
 22.2077 24.0948 26.5651 30.     35.2644 45.     90.    ]
LAC_FA_SCHEDULE: [14.4775 14.9632 15.5014 16.1021 16.7787 17.5484 18.4349 19.4712 20.7048
 22.2077 24.0948 26.5651 30.     35.2644 45.     90.    ]
TRAINING_PEAK: 1.482615
## Parameter Ranges
kPL Range: 0.001 - 0.200 s^-1
kVE Range: 0.050 - 0.450 s^-1
vB Range: 0.030 - 0.180

## Calibration Meta
P_train: 1.3947864817380908
percentile: 99.9
pyr_channel: 0
min_peak: 1e-06
protocol_name: TRAMP_VFA
## Training
Max epochs: 1000
Actual epochs trained: 606
Early stopping patience: 50
Early stopping triggered: Yes
Best validation loss: 0.004803
Final training loss: 0.004800
Initial learning rate: 0.001
Training time: 1003.49 seconds
Test R² for kpl: 0.950Test R² for kve: 0.072Test R² for vb:  0.042

## SNR Analysis
Mean SNR for Pyruvate: 20.06
Mean SNR for Lactate: 7.70
Minimum SNR for Pyruvate: 10.25
Minimum SNR for Lactate: 0.70
Maximum SNR for Pyruvate: 29.65
Maximum SNR for Lactate: 26.66

=== X_raw Statistics ===
Maximum: 1.295481
Minimum: -0.342054
Mean: 0.165763
Shape: (1000000, 32)

=== X_norm Statistics ===
Maximum: 1.482615
Minimum: -0.253154
Mean: 0.165987
Shape: (1000000, 32)
