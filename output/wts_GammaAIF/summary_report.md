# Hyperpolarized 13C MRI Analysis Demo Summary

**Generator Name:** TwoCompartmentHPDataGenerator

**Date:** 2026-02-06 19:08:20

## Dataset
- Total samples: 1000000
- Training samples: 700000
- Validation samples: 285000
- Test samples: 15000
- Time points: 25 (TR=2.0, 0-48.0s)

- Noise level (std): 0.05

- PYR FA SCHEDULE: [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]

- LAC FA SCHEDULE: [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]

## Parameter Ranges
- kPL Range: 0.010 - 0.200 s^-1
- kVE Range: 0.050 - 0.450 s^-1
- vB Range: 0.010 - 0.500

## Training
- Max epochs: 1000
- Actual epochs trained: 1000
- Early stopping patience: 50
- Early stopping triggered: No
- Best validation loss: 0.002319
- Final training loss: 0.002434
- Initial learning rate: 0.001
- Training time: 1693.29 seconds
Test R² for kpl: 0.971Test R² for kve: 0.894Test R² for vb:  0.732

## SNR Analysis
- Mean SNR for Pyruvate: 23.28
- Mean SNR for Lactate: 8.99
- Minimum SNR for Pyruvate: 14.83
- Minimum SNR for Lactate: 0.96
- Maximum SNR for Pyruvate: 43.49
- Maximum SNR for Lactate: 28.53

=== X_raw Statistics ===Maximum: 2.174419
Minimum: -0.272579
Mean: 0.162259
Shape: (1000000, 25, 2)

=== X_norm Statistics ===Maximum: 2.174419
Minimum: -0.272579
Mean: 0.162259
Shape: (1000000, 25, 2)
