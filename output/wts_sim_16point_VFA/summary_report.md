# Hyperpolarized 13C MRI Analysis Demo Summary

**Date:** 2025-10-20 02:40:27

## Dataset
- Total samples: 1000000
- Training samples: 700000
- Validation samples: 285000
- Test samples: 15000
- Time points: 16 (TR=5s, 0-75s)

## Parameter Ranges
- kPL: 0.010 - 0.200 s^-1
## Training
- Max epochs: 400
- Actual epochs trained: 400
- Early stopping patience: 50
- Early stopping triggered: No
- Best validation loss: 0.002442
- Final training loss: 0.002635
- Initial learning rate: 0.001
- Training time: 626.80 seconds
Test R² for kpl: 0.958Test R² for kve: 0.883Test R² for vb:  0.715

## SNR Analysis
- Mean SNR for Pyruvate: 4.59
- Mean SNR for Lactate: 4.58
- Minimum SNR for Pyruvate: 2.07
- Minimum SNR for Lactate: 0.35
- Maximum SNR for Pyruvate: 15.28
- Maximum SNR for Lactate: 14.87

=== X_raw Statistics ===Maximum: 2.130955
Minimum: -0.248615
Mean: 0.215864
Shape: (1000000, 16, 2)

=== X_norm Statistics ===Maximum: 2.130955
Minimum: -0.248615
Mean: 0.215864
Shape: (1000000, 16, 2)
