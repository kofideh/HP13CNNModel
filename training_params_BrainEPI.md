# Hyperpolarized 13C MRI Analysis Demo Summary

**Generator Name:** TwoCompartmentHPDataGeneratorPhysio

**Date:** 2026-05-06 16:18:20

## Dataset
Total samples: 1000000
Training samples: 700000
Validation samples: 150000
Test samples: 150000
Noise level (std): 0.05
## Configuration
NUM_TIME_POINTS: 12
SCAN_TR=5.0
PYR_FA_SCHEDULE: [11.0]
LAC_FA_SCHEDULE: [80.0]
TRAINING_PEAK: 1.067512
## Parameter Ranges
kPL Range: 0.001 - 0.060 s^-1
kVE Range: 0.010 - 0.250 s^-1
vB Range: 0.005 - 0.150

## Calibration Meta
P_train: 0.8508796210885081
percentile: 99.9
pyr_channel: 0
min_peak: 1e-06
protocol_name: Brain_EPI
AIF_TYPE: Measured
## Training
Max epochs: 1000
Actual epochs trained: 772
Early stopping patience: 50
Early stopping triggered: Yes
Best validation loss: 0.000822
Final training loss: 0.000838
Initial learning rate: 0.001
Training time: 1399.73 seconds
Test R² for kpl: 0.141Test R² for kve: 0.757Test R² for vb:  0.404

## SNR Analysis
Mean SNR for Pyruvate: 7.65
Mean SNR for Lactate: 2.00
Minimum SNR for Pyruvate: 0.57
Minimum SNR for Lactate: 0.30
Maximum SNR for Pyruvate: 21.35
Maximum SNR for Lactate: 6.37

=== X_raw Statistics ===
Maximum: 1.067512
Minimum: -0.251387
Mean: 0.034914
Shape: (1000000, 24)

=== X_norm Statistics ===
Maximum: 10.958412
Minimum: -11.246607
Mean: 0.097602
Shape: (1000000, 24)
