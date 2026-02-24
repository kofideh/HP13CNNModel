# Hyperpolarized 13C MRI Analysis Demo Summary

**Generator Name:** TwoCompartmentHPDataGeneratorMeasured

**Date:** 2026-02-16 11:13:43

## Dataset
Total samples: 1000000
Training samples: 700000
Validation samples: 285000
Test samples: 15000
Noise level (std): 0.05
## Configuration
NUM_TIME_POINTS: 12
SCAN_TR=5.0
PYR_FA_SCHEDULE: [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0]
LAC_FA_SCHEDULE: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
TRAINING_PEAK: 1.930879
## Parameter Ranges
kPL Range: 0.010 - 0.200 s^-1
kVE Range: 0.050 - 0.450 s^-1
vB Range: 0.030 - 0.180

## Training
Max epochs: 1000
Actual epochs trained: 418
Early stopping patience: 50
Early stopping triggered: Yes
Best validation loss: 0.004988
Final training loss: 0.005012
Initial learning rate: 0.001
Training time: 688.92 seconds
Test R² for kpl: 0.940Test R² for kve: 0.023Test R² for vb:  0.021

## SNR Analysis
Mean SNR for Pyruvate: 20.00
Mean SNR for Lactate: 9.25
Minimum SNR for Pyruvate: 9.76
Minimum SNR for Lactate: 0.63
Maximum SNR for Pyruvate: 30.26
Maximum SNR for Lactate: 38.62

=== X_raw Statistics ===
Maximum: 1.930879
Minimum: -0.278368
Mean: 0.135121
Shape: (1000000, 12, 2)

=== X_norm Statistics ===
Maximum: 1.930879
Minimum: -0.278368
Mean: 0.135121
Shape: (1000000, 12, 2)
