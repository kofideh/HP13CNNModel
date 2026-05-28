## Dataset

TOTAL_SAMPLES: 1000000

## Configuration

NUM_TIME_POINTS: 12
SCAN_TR: 5.0
PYR_FA_SCHEDULE: [11]
LAC_FA_SCHEDULE: [80]
INVIVO_DATA_PATH: data/*Brain*
## Parameter Ranges

kPL Range: 0.001 - 0.06 s^-1
kVE Range: 0.010 - 0.250 s^-1
vB Range: 0.005 - 0.150
SNR Range: 2 - 100
T1P Range: 20 - 60
T1L Range: 15 - 55

## Calibration Meta

percentile: 99.9
pyr_channel: 0
min_peak: 1e-06
protocol_name: Brain_EPI
AIF_TYPE: Measured 

## Training

N_EPOCHS: 1000

