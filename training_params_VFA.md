# Hyperpolarized 13C MRI Analysis Demo Summary

## Dataset

TOTAL_SAMPLES: 1000000

## Configuration

NUM_TIME_POINTS: 16
SCAN_TR: 2.0
PYR_FA_SCHEDULE: [14.4775 14.9632 15.5014 16.1021 16.7787 17.5484 18.4349 19.4712 20.7048
22.2077 24.0948 26.5651 30.     35.2644 45.     90.    ]
LAC_FA_SCHEDULE: [14.4775 14.9632 15.5014 16.1021 16.7787 17.5484 18.4349 19.4712 20.7048
22.2077 24.0948 26.5651 30.     35.2644 45.     90.    ]
INVIVO_DATA_PATH: data/*TRAMP*

## Parameter Ranges

kPL Range: 0.001 - 0.200 s^-1
kVE Range: 0.010 - 0.300 s^-1
vB Range: 0.005 - 0.200
SNR Range: 2 - 100
T1P Range: 20 - 60
T1L Range: 15 - 55

## Calibration Meta

percentile: 99.9
pyr_channel: 0
min_peak: 1e-06
protocol_name: TRAMP_VFA
AIF_TYPE: Measured

## Training

Max epochs: 1000
Initial learning rate: 0.001



