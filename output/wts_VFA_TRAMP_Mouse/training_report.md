# Hyperpolarized 13C MRI Analysis Demo Summary

**Generator Name:** TwoCompartmentHPDataGeneratorMeasured

**Date:** 2026-02-21 17:42:09

## Dataset
Total samples: 1000000
Training samples: 700000
Validation samples: 285000
Test samples: 15000
Noise level (std): 0.05
## Configuration
NUM_TIME_POINTS: 16
SCAN_TR=2.0
PYR_FA_SCHEDULE: [14.4775 14.9632 15.5014 16.1021 16.7787 17.5484 18.4349 19.4712 20.7048
 22.2077 24.0948 26.5651 30.     35.2644 45.     90.    ]
LAC_FA_SCHEDULE: [14.4775 14.9632 15.5014 16.1021 16.7787 17.5484 18.4349 19.4712 20.7048
 22.2077 24.0948 26.5651 30.     35.2644 45.     90.    ]
TRAINING_PEAK: 1.488831
## Parameter Ranges
kPL Range: 0.001 - 0.200 s^-1
kVE Range: 0.050 - 0.450 s^-1
vB Range: 0.030 - 0.180

## Training
Max epochs: 1000
Actual epochs trained: 746
Early stopping patience: 50
Early stopping triggered: Yes
Best validation loss: 0.004795
Final training loss: 0.004799
Initial learning rate: 0.001
Training time: 1232.46 seconds
Test R² for kpl: 0.951Test R² for kve: 0.074Test R² for vb:  0.038

## SNR Analysis
Mean SNR for Pyruvate: 20.05
Mean SNR for Lactate: 7.70
Minimum SNR for Pyruvate: 10.44
Minimum SNR for Lactate: 0.63
Maximum SNR for Pyruvate: 29.78
Maximum SNR for Lactate: 26.78

=== X_raw Statistics ===
Maximum: 1.488831
Minimum: -0.277439
Mean: 0.166079
Shape: (1000000, 16, 2)

=== X_norm Statistics ===
Maximum: 1.488831
Minimum: -0.277439
Mean: 0.166079
Shape: (1000000, 16, 2)
