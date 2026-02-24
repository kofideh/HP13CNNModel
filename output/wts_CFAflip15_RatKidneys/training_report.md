# Hyperpolarized 13C MRI Analysis Demo Summary

**Generator Name:** TwoCompartmentHPDataGeneratorMeasured

**Date:** 2026-02-21 19:21:11

## Dataset
Total samples: 1000000
Training samples: 700000
Validation samples: 285000
Test samples: 15000
Noise level (std): 0.05
## Configuration
NUM_TIME_POINTS: 25
SCAN_TR=2.0
PYR_FA_SCHEDULE: [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
LAC_FA_SCHEDULE: [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
TRAINING_PEAK: 1.000000
## Parameter Ranges
kPL Range: 0.001 - 0.060 s^-1
kVE Range: 0.050 - 0.450 s^-1
vB Range: 0.030 - 0.180

## Training
Max epochs: 1000
Actual epochs trained: 493
Early stopping patience: 50
Early stopping triggered: Yes
Best validation loss: 0.004989
Final training loss: 0.004986
Initial learning rate: 0.001
Training time: 845.84 seconds
Test R² for kpl: 0.886Test R² for kve: 0.015Test R² for vb:  0.009

## SNR Analysis
Mean SNR for Pyruvate: 20.00
Mean SNR for Lactate: 3.59
Minimum SNR for Pyruvate: 20.00
Minimum SNR for Lactate: 0.77
Maximum SNR for Pyruvate: 20.00
Maximum SNR for Lactate: 10.98

=== X_raw Statistics ===
Maximum: 1.000000
Minimum: -0.409418
Mean: 0.077899
Shape: (1000000, 25, 2)

=== X_norm Statistics ===
Maximum: 1.000000
Minimum: -0.409418
Mean: 0.077899
Shape: (1000000, 25, 2)
