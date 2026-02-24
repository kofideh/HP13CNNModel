# Hyperpolarized 13C MRI Analysis Demo Summary

**Date:** 2026-02-07 10:12:23

## Dataset
- Total samples: 50
- Test samples: 50
- Time points: 25 (TR=5s, 0-120s)

## Parameter Ranges
- kPL: 0.010 - 0.200 s^-1
## Training
- Initial learning rate: 0.001
Test R² for kpl: 0.955Test R² for kve: 0.843Test R² for vb:  0.622

## SNR Analysis
- Mean SNR for Pyruvate: 23.11
- Mean SNR for Lactate: 8.69
- Minimum SNR for Pyruvate: 17.61
- Minimum SNR for Lactate: 2.05
- Maximum SNR for Pyruvate: 34.65
- Maximum SNR for Lactate: 24.80

=== X_raw Statistics ===Maximum: 1.732745
Minimum: -0.166751
Mean: 0.156604
Shape: (50, 25, 2)

=== X_norm Statistics ===Maximum: 1.732745
Minimum: -0.166751
Mean: 0.156604
Shape: (50, 25, 2)
