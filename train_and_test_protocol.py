import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from hybrid_model_utils import HybridMultiHead, compute_peak_percentile, load_training_config, prepare_hybrid_inputs
from fit_two_compartment_physio_nlls import fit_nlls_gamma_AIF, fit_nlls_known_Pv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train hybrid multi-head model with configurable training parameters file."
    )
    parser.add_argument(
        "--training-params",
        dest="training_params_path",
        default="training_params.md",
        help="Path to training parameters markdown file (default: training_params.md)",
    )
    # Ignore unknown args injected by debuggers/notebooks (e.g., -f <kernel.json>)
    args, _ = parser.parse_known_args()
    return args


def plot_true_vs_pred(y_true, y_pred, title_suffix):
    labels = ['kPL', 'kVE', 'vB']
    plt.figure(figsize=(15, 4))
    n_points_to_plot = 50
    for i in range(3):
        plt.subplot(1, 3, i+1)
        n_total = len(y_true)
        n_plot = min(n_points_to_plot, n_total)
        indices = np.random.choice(n_total, size=n_plot, replace=False)
        plt.scatter(y_true[indices, i], y_pred[indices, i], alpha=0.5)
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                [y_true[:, i].min(), y_true[:, i].max()], 'r--')

        ss_res = np.sum((y_pred[:, i] - y_true[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - ss_res / ss_tot

        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{labels[i]}: R²={r2:.3f}\n{title_suffix}")

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"true_vs_pred_{title_suffix}.png"))
    plt.close()
    df = pd.DataFrame({
        'kPL_true': y_true[:, 0],
        'kVE_true': y_true[:, 1],
        'vB_true': y_true[:, 2],
        'kPL_pred': y_pred[:, 0],
        'kVE_pred': y_pred[:, 1],
        'vB_pred': y_pred[:, 2],
    })
    df.to_csv(os.path.join(run_dir, f"predictions_vs_truth_{title_suffix}.csv"), index=False)


args = parse_args()
training_params_path = args.training_params_path
print(f"Using training params file: {training_params_path}")

_cfg = load_training_config(training_params_path)

NUM_TIMEPOINTS = _cfg["NUM_TIMEPOINTS"]
PYR_FA_SCHEDULE = _cfg["PYR_FA_SCHEDULE"]
LAC_FA_SCHEDULE = _cfg["LAC_FA_SCHEDULE"]
SCAN_TR = _cfg["SCAN_TR"]  # seconds
n_samples = _cfg["TOTAL_SAMPLES"]   
AIF_TYPE = _cfg["AIF_TYPE"]
protocol_name = _cfg["PROTOCOL_NAME"]
invivo_data_path = _cfg["INVIVO_DATA_PATH"]
n_epochs = _cfg["N_EPOCHS"]

KPL_MIN = _cfg["KPL_MIN"]
KPL_MAX = _cfg["KPL_MAX"]
KVE_MIN = _cfg["KVE_MIN"]
KVE_MAX = _cfg["KVE_MAX"]
VB_MIN = _cfg["VB_MIN"]
VB_MAX = _cfg["VB_MAX"]
SNR_MIN = _cfg["SNR_MIN"]
SNR_MAX = _cfg["SNR_MAX"]
R1P_MIN = 1/_cfg["T1P_MAX"]
R1P_MAX = 1/_cfg["T1P_MIN"]
R1L_MIN = 1/_cfg["T1L_MAX"]
R1L_MAX = 1/_cfg["T1L_MIN"]


# 1. Generate data
time_points = np.arange(0, NUM_TIMEPOINTS * SCAN_TR, SCAN_TR)
n_timepoints = len(time_points)

aif_key = str(AIF_TYPE).strip().lower().replace("-", "_").replace(" ", "_")


if aif_key in ["measured", "physio", "physiologic", "physiological"]:  
    from two_compartment_generator_physio import TwoCompartmentHPDataGeneratorPhysio
    generator = TwoCompartmentHPDataGeneratorPhysio(
    vb_range=(VB_MIN, VB_MAX),
    kpl_range=(KPL_MIN, KPL_MAX),
    kve_range=(KVE_MIN, KVE_MAX),
    r1p_range=(R1P_MIN, R1P_MAX),
    r1l_range=(R1L_MIN, R1L_MAX),
    time_points=time_points,                        # inferred from schedule length & TR
    flip_angle_pyr_deg=PYR_FA_SCHEDULE, # or scalar FAs
    flip_angle_lac_deg=LAC_FA_SCHEDULE,
    TR=SCAN_TR  # Example noise range for in vivo data
    )
else:
    raise ValueError(f"Unrecognized AIF_TYPE: {AIF_TYPE}")
# X returned from generator: shape (N, T, 2), RAW signals
# y labels unchanged
X, y, met_pools, _ = generator.generate_dataset(n_samples=n_samples, snr_range=(SNR_MIN, SNR_MAX))


# Build raw + normalized branches for training
X_norm, X_raw, prep_stats = prepare_hybrid_inputs(
    X,
    alpha=1.0,              # no global scaling during synthetic training
    pyr_channel=0,
    flatten=True  # MLP input
)


print("Training prep:", prep_stats)


output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(output_dir, f"TrainingReport_{protocol_name}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)



# Plot comparison of X_raw vs X_norm
# Select a few random samples to plot
n_samples_to_plot = 5
sample_indices = np.random.choice(X.shape[0], n_samples_to_plot, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Comparison of X_raw vs X_norm', fontsize=16)

for i, sample_idx in enumerate(sample_indices):
    if i >= 5:  # Only plot first 5 samples
        break
    
    row = i // 3
    col = i % 3
    
    # Plot pyruvate (channel 0)
    #reshape to (T,) for plotting
    X_raw_sample = X_raw[sample_idx].reshape(n_timepoints, 2)
    X_norm_sample = X_norm[sample_idx].reshape(n_timepoints, 2)
    axes[row, col].plot(time_points, X_raw_sample[:, 0], 'b-', label='Pyruvate (Raw)', linewidth=2)
    axes[row, col].plot(time_points, X_raw_sample[:, 1], 'r-', label='Lactate (Raw)', linewidth=2)
    
    # Plot normalized on secondary y-axis
    ax2 = axes[row, col].twinx()
    ax2.plot(time_points, X_norm_sample[:, 0], 'b--', label='Pyruvate (Norm)', alpha=0.7)
    ax2.plot(time_points, X_norm_sample[:, 1], 'r--', label='Lactate (Norm)', alpha=0.7)
    
    axes[row, col].set_xlabel('Time (s)')
    axes[row, col].set_ylabel('Raw Signal', color='black')
    ax2.set_ylabel('Normalized Signal', color='gray')
    axes[row, col].set_title(f'Sample {sample_idx + 1}')
    axes[row, col].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[row, col].grid(True, alpha=0.3)

# Hide the last subplot if we have fewer than 6 samples
if n_samples_to_plot < 6:
    axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "X_raw_vs_X_norm_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# Create histogram comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of all values in X_raw
axes[0].hist(X_raw.flatten(), bins=50, alpha=0.7, color='blue', label='X_raw')
axes[0].set_xlabel('Signal Value')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of X_raw Values')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogram of all values in X_norm
axes[1].hist(X_norm.flatten(), bins=50, alpha=0.7, color='red', label='X_norm')
axes[1].set_xlabel('Signal Value')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of X_norm Values')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "X_raw_vs_X_norm_histograms.png"), dpi=150, bbox_inches='tight')
plt.close()


# 3. Flatten inputs
X_raw_flat = X_raw.reshape(X.shape[0], -1)
X_norm_flat = X_norm.reshape(X.shape[0], -1)

data_peak = X_raw_flat.max()

# Safety checks
assert len(X_norm) == len(X_raw) == len(y), "X_norm, X_raw, and y must have same length"

N = len(y)
indices = np.arange(N)

# First split: 70% train, 30% temp
idx_train, idx_temp = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Second split: split temp equally into 15% val, 15% test
idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# Apply SAME indices to both branches and labels
Xn_train, Xn_val, Xn_test = X_norm[idx_train], X_norm[idx_val], X_norm[idx_test]
Xr_train, Xr_val, Xr_test = X_raw[idx_train], X_raw[idx_val], X_raw[idx_test]
y_train,  y_val,  y_test  = y[idx_train],    y[idx_val],    y[idx_test]

# Optional: print split sizes
print(f"Train/Val/Test sizes: {len(idx_train)} / {len(idx_val)} / {len(idx_test)}")
print(f"Fractions: {len(idx_train)/N:.3f} / {len(idx_val)/N:.3f} / {len(idx_test)/N:.3f}")

# Optional but recommended: save indices for exact reproducibility
split_indices = {
    "idx_train": idx_train,
    "idx_val": idx_val,
    "idx_test": idx_test,
}
# Example:
np.savez(f"train_val_test_indices_protocol_{protocol_name}.npz", **split_indices)



# 5. Convert to tensors
Xr_train_tensor = torch.tensor(Xr_train, dtype=torch.float32)
Xn_train_tensor = torch.tensor(Xn_train, dtype=torch.float32)
Xr_val_tensor = torch.tensor(Xr_val, dtype=torch.float32)
Xn_val_tensor = torch.tensor(Xn_val, dtype=torch.float32)
Xr_test_tensor = torch.tensor(Xr_test, dtype=torch.float32)
Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)




model = HybridMultiHead(input_dim_raw=X_raw_flat.shape[1], 
                        input_dim_norm=X_norm_flat.shape[1],
                        vb_range=(VB_MIN, VB_MAX),
                        kpl_range=(KPL_MIN, KPL_MAX),
                        kve_range=(KVE_MIN, KVE_MAX)
                        )


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)



# Early stopping parameters
patience = 50  # Number of epochs to wait for improvement
min_delta = 1e-6  # Minimum change to qualify as an improvement
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# 7. Training loop with early stopping
print("Training Neural Network with Early Stopping...")
training_start = time.perf_counter()
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    preds = model(Xn_train_tensor, Xr_train_tensor)
    loss = criterion(preds, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(Xn_val_tensor, Xr_val_tensor)
        val_loss = criterion(val_preds, y_val_tensor)
    
    # Store losses for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    
    # Early stopping logic
    if val_loss.item() < best_val_loss - min_delta:
        best_val_loss = val_loss.item()
        patience_counter = 0
        # Save best model state
        best_model_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, Best Val Loss = {best_val_loss:.4f}, Patience = {patience_counter}/{patience}")
    
    # Check if we should stop early
    if patience_counter >= patience:
        print(f"Early stopping triggered after epoch {epoch}! Best validation loss: {best_val_loss:.4f}")
        break

training_duration = time.perf_counter() - training_start
print(f"Training finished in {training_duration:.2f} seconds")

# Load best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', alpha=0.7)
plt.axvline(x=len(train_losses) - patience_counter - 1, color='red', linestyle='--', 
            label=f'Best Model (Epoch {len(train_losses) - patience_counter - 1})', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale for better visualization
plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

model_path = os.path.join(run_dir, "trained_hybrid_positive.pth")
if best_model_state is not None:
    model.load_state_dict(best_model_state)
torch.save(model.state_dict(), model_path)
# After training
print("Saved model with positivity constraints to ", model_path)
# 8. Evaluation
model.eval()


r2_kpl = 0
r2_kve = 0
r2_vb  = 0


with torch.no_grad():
    y_pred = model(Xn_test_tensor, Xr_test_tensor).numpy()
    y_true = y_test_tensor.numpy()

    r2_kpl = 1 - np.sum((y_pred[:, 0] - y_true[:, 0])**2) / np.sum((y_true[:, 0] - np.mean(y_true[:, 0]))**2)
    r2_kve = 1 - np.sum((y_pred[:, 1] - y_true[:, 1])**2) / np.sum((y_true[:, 1] - np.mean(y_true[:, 1]))**2)
    r2_vb  = 1 - np.sum((y_pred[:, 2] - y_true[:, 2])**2) / np.sum((y_true[:, 2] - np.mean(y_true[:, 2]))**2)

    plot_true_vs_pred(
        y_true,
        y_pred,
        f"{SNR_MIN}_Neural_Network"
    )




# Zero-noise NLLS sanity check
X0, y0, met0, _ = generator.generate_dataset(n_samples=20, snr_range=None)  # No noise for oracle test

bounds_same = (
    (KPL_MIN, KVE_MIN, VB_MIN),
    (KPL_MAX, KVE_MAX, VB_MAX)
)

p0_mid = (
    0.5 * (KPL_MIN + KPL_MAX),
    0.5 * (KVE_MIN + KVE_MAX),
    0.5 * (VB_MIN + VB_MAX),
)

pred0 = []

for j in range(len(y0)):
    res = fit_nlls_known_Pv(
        time_points=time_points,
        Pv=met0[j, :, 0],
        S_pyr_obs=X0[j, :, 0],
        S_lac_obs=X0[j, :, 1],
        TR=SCAN_TR,
        flips_pyr_deg=PYR_FA_SCHEDULE,
        flips_lac_deg=LAC_FA_SCHEDULE,
        R1p=1/30,
        R1l=1/25,
        p0=p0_mid,
        bounds=bounds_same,
        sigma_pyr=None,
        sigma_lac=None,
    )
    pred0.append([res["kpl"], res["kve"], res["vb"]])

pred0 = np.array(pred0)

print("Zero-noise oracle-Pv NLLS max abs error:")
print("kPL:", np.max(np.abs(pred0[:, 0] - y0[:, 0])))
print("kVE:", np.max(np.abs(pred0[:, 1] - y0[:, 1])))
print("vB :", np.max(np.abs(pred0[:, 2] - y0[:, 2])))

#select only 50 values from idx_test for faster NLLS testing, since it's much slower than the NN predictions
idx_test_subset = np.random.choice(idx_test, size=min(50, len(idx_test)), replace=False)

#test NLLS fitting with known Pv on the noisy test set to confirm it works and to compare against NN predictions

start_time = time.time()

# Use ORIGINAL raw noisy curves, not the prepared NN branch
X_test_tc = X[idx_test_subset]              # shape: (N_test, T, 2)
Pv_test = met_pools[idx_test_subset, :, 0]  # shape: (N_test, T), channel 0 is Pv
y_true = y[idx_test_subset]

nlls_preds = []
nlls_failures = 0

# Use same parameter bounds as the NN/generator
nlls_bounds = (
    (KPL_MIN, KVE_MIN, VB_MIN),
    (KPL_MAX, KVE_MAX, VB_MAX)
)

nlls_p0 = (
    0.5 * (KPL_MIN + KPL_MAX),
    0.5 * (KVE_MIN + KVE_MAX),
    0.5 * (VB_MIN + VB_MAX),
)

for j in range(len(idx_test_subset)):
    S_pyr_test = X_test_tc[j, :, 0]
    S_lac_test = X_test_tc[j, :, 1]
    Pv_val = Pv_test[j, :]

    try:
        res = fit_nlls_known_Pv(
            time_points=time_points,
            Pv=Pv_val,
            S_pyr_obs=S_pyr_test,
            S_lac_obs=S_lac_test,
            TR=SCAN_TR,
            flips_pyr_deg=PYR_FA_SCHEDULE,
            flips_lac_deg=LAC_FA_SCHEDULE,
            R1p=1/30,
            R1l=1/25,
            p0=nlls_p0,
            bounds=nlls_bounds,

            # For debugging, first try peak-normalized residuals:
            sigma_pyr=None,
            sigma_lac=None,

            # Later, for statistical fitting with known noise, use:
            # sigma_pyr=noise_level,
            # sigma_lac=noise_level,
        )

        if res["success"]:
            nlls_preds.append([res["kpl"], res["kve"], res["vb"]])
        else:
            print(f"NLLS did not converge for sample {j}: {res['message']}")
            nlls_preds.append([np.nan, np.nan, np.nan])
            nlls_failures += 1

    except Exception as e:
        print(f"NLLS failed for sample {j}: {e}")
        nlls_preds.append([np.nan, np.nan, np.nan])
        nlls_failures += 1

tradfit_time = time.time() - start_time
print(f"NLLS fitting time: {tradfit_time:.2f} seconds")
print(f"NLLS failed or non-converged fits: {nlls_failures} / {len(idx_test_subset)}")

y_pred_fit = np.array(nlls_preds)

plot_true_vs_pred(
    y_true,
    y_pred_fit,
    f"{SNR_MIN}_{SNR_MAX}_NLLS_OraclePv"
)
    

print("Using GAMMA AIF NLLS...")
start_time = time.time()

# Use ORIGINAL raw noisy curves, not the prepared NN branch
X_test_tc = X[idx_test_subset]              # shape: (N_test, T, 2)
Pv_test = met_pools[idx_test_subset, :, 0]  # shape: (N_test, T), channel 0 is Pv
y_true = y[idx_test_subset]

nlls_preds = []
nlls_failures = 0

# Use same parameter bounds as the NN/generator
nlls_bounds = (
    (KPL_MIN, KVE_MIN, VB_MIN),
    (KPL_MAX, KVE_MAX, VB_MAX)
)

nlls_p0 = (
    0.5 * (KPL_MIN + KPL_MAX),
    0.5 * (KVE_MIN + KVE_MAX),
    0.5 * (VB_MIN + VB_MAX),
)

for j in range(len(idx_test_subset)):
    S_pyr_test = X_test_tc[j, :, 0]
    S_lac_test = X_test_tc[j, :, 1]

    try:
        res = fit_nlls_gamma_AIF(
            time_points=time_points,
            S_pyr_obs=S_pyr_test,
            S_lac_obs=S_lac_test,
            TR=SCAN_TR,
            flips_pyr_deg=PYR_FA_SCHEDULE,
            flips_lac_deg=LAC_FA_SCHEDULE,
            R1p=1/30,
            R1l=1/25,
            p0=nlls_p0,
            bounds=nlls_bounds,

            # For debugging, first try peak-normalized residuals:
            sigma_pyr=None,
            sigma_lac=None,

            # Later, for statistical fitting with known noise, use:
            # sigma_pyr=noise_level,
            # sigma_lac=noise_level,
        )

        if res["success"]:
            nlls_preds.append([res["kpl"], res["kve"], res["vb"]])
        else:
            print(f"NLLS did not converge for sample {j}: {res['message']}")
            nlls_preds.append([np.nan, np.nan, np.nan])
            nlls_failures += 1

    except Exception as e:
        print(f"NLLS failed for sample {j}: {e}")
        nlls_preds.append([np.nan, np.nan, np.nan])
        nlls_failures += 1

tradfit_time = time.time() - start_time
print(f"NLLS fitting time: {tradfit_time:.2f} seconds")
print(f"NLLS failed or non-converged fits: {nlls_failures} / {len(idx_test_subset)}")

y_pred_fit = np.array(nlls_preds)

plot_true_vs_pred(
    y_true,
    y_pred_fit,
    f"{SNR_MIN}_{SNR_MAX}_NLLS_Gamma_AIF"
)







# Step 12: Generate a summary report
# Estimate SNR
pyr_signals = X[:, :, 0]
lac_signals = X[:, :, 1]

# Compute SNR as mean amplitude divided by noise std
# snr_pyr = np.max(np.abs(pyr_signals), axis=1) / noise_level
# snr_lac = np.max(np.abs(lac_signals), axis=1) / noise_level

# If X_raw_ntc is your RAW training signals before flattening, shape (N, T, 2)
# Prefer using TRAIN subset only to avoid leakage of validation/test amplitude statistics
X_train_raw_ntc = X[idx_train]  # assuming X is raw (N, T, 2) before prepare_hybrid_inputs

P_train = compute_peak_percentile(
    X_train_raw_ntc,
    percentile=99.9,
    pyr_channel=0,
    min_peak=1e-6
)

#test neural network on newly generated data with same params to confirm peak-based normalization is working as intended
# Test neural network on newly generated data
X2, y2, _, _ = generator.generate_dataset(n_samples=50, snr_range=(SNR_MIN, SNR_MAX))  # Generate new test set with same parameters    

P_clin = compute_peak_percentile(
    X2,
    percentile=99.9,
    pyr_channel=0,
    min_peak=1e-6
)

alpha2 = P_train / max(P_clin, 1e-8)

# IMPORTANT: same order as training
X_norm2, X_raw2, clin_meta = prepare_hybrid_inputs(
    X2,
    alpha=alpha2,
    pyr_channel=0,
    flatten=True
)

Xr2_tensor = torch.tensor(X_raw2, dtype=torch.float32)
Xn2_tensor = torch.tensor(X_norm2, dtype=torch.float32)
y2_tensor = torch.tensor(y2, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_pred = model(Xn2_tensor, Xr2_tensor).numpy()
    y_true = y2_tensor.numpy()

    plot_true_vs_pred(
        y_true,
        y_pred,
        f"{SNR_MIN}_{SNR_MAX}_Neural_Network_Newly_Generated_Data"
    )





calibration_meta = {
    "P_train": float(P_train),
    "percentile": 99.9,
    "pyr_channel": 0,
    "min_peak": 1e-6,
    "protocol_name": protocol_name,  
    "AIF_TYPE": AIF_TYPE
}

print("Training calibration:", calibration_meta)

print("\n--- Step 12: Generate Summary Report ---")
with open(os.path.join(run_dir, 'training_report.md'), 'w') as f:
    f.write(f"# Hyperpolarized 13C MRI Analysis Demo Summary\n\n")
    f.write(f"**Generator Name:** {generator.__class__.__name__}\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"## Dataset\n")
    f.write(f"Total samples: {n_samples}\n")
    f.write(f"Training samples: {Xr_train.shape[0]}\n")
    f.write(f"Validation samples: {Xr_val.shape[0]}\n")
    f.write(f"Test samples: {Xr_test.shape[0]}\n")
    f.write(f"SNR Range: {SNR_MIN} - {SNR_MAX}\n")

    f.write(f"## Configuration\n")
    f.write(f"NUM_TIME_POINTS: {n_timepoints}\n")
    f.write(f"SCAN_TR={SCAN_TR}\n")
    f.write(f"PYR_FA_SCHEDULE: {PYR_FA_SCHEDULE}\n")
    f.write(f"LAC_FA_SCHEDULE: {LAC_FA_SCHEDULE}\n")
    f.write(f"TRAINING_PEAK: {np.max(X):.6f}\n")# include the training peak value used for normalization
    f.write(f"## Parameter Ranges\n")
    f.write(f"kPL Range: {KPL_MIN:.3f} - {KPL_MAX:.3f} s^-1\n")
    f.write(f"kVE Range: {KVE_MIN:.3f} - {KVE_MAX:.3f} s^-1\n")
    f.write(f"vB Range: {VB_MIN:.3f} - {VB_MAX:.3f}\n\n")
    f.write(f"T1p Range: {1/R1P_MAX:.3f} - {1/R1P_MIN:.3f} s\n")
    f.write(f"T1l Range: {1/R1L_MAX:.3f} - {1/R1L_MIN:.3f} s\n\n")
    f.write(f"## Calibration Meta\n")
    for k, v in calibration_meta.items():
        f.write(f"{k}: {v}\n")
    
    f.write(f"## Training\n")
    f.write(f"Max epochs: {n_epochs}\n")
    f.write(f"Actual epochs trained: {len(train_losses)}\n")
    f.write(f"Early stopping patience: {patience}\n")
    f.write(f"Early stopping triggered: {'Yes' if patience_counter >= patience else 'No'}\n")
    f.write(f"Best validation loss: {best_val_loss:.6f}\n")
    f.write(f"Final training loss: {train_losses[-1]:.6f}\n")
    f.write(f"Initial learning rate: 0.001\n")
    f.write(f"Training time: {training_duration:.2f} seconds\n")
    # f.write(f"- Traditional fitting time: {tradfit_time:.2f} seconds\n\n")
    
    f.write(f"Test R² for kpl: {r2_kpl:.3f}")
    f.write(f"Test R² for kve: {r2_kve:.3f}")
    f.write(f"Test R² for vb:  {r2_vb:.3f}")
    
        # Display statistics and plot comparison
    f.write("\n=== X_raw Statistics ===\n")
    f.write(f"Maximum: {np.max(X_raw):.6f}\n")
    f.write(f"Minimum: {np.min(X_raw):.6f}\n")
    f.write(f"Mean: {np.mean(X_raw):.6f}\n")
    f.write(f"Shape: {X_raw.shape}\n")

    f.write("\n=== X_norm Statistics ===\n")
    f.write(f"Maximum: {np.max(X_norm):.6f}\n")
    f.write(f"Minimum: {np.min(X_norm):.6f}\n")
    f.write(f"Mean: {np.mean(X_norm):.6f}\n")
    f.write(f"Shape: {X_norm.shape}\n")


cmd = [
        sys.executable,
        'sim_inference_plot_agreement_gammaAIF.py',
        "--weights-dir",
        os.path.abspath(run_dir),
        "--n-samples",
        "100",
    ]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

cmd = [
        sys.executable,
        'sim_inference_R1variation_gammaAIF.py',
        "--weights-dir",
        os.path.abspath(run_dir),
        "--n-samples",
        "100",
    ]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

cmd = [
        sys.executable,
        'sim_inference_Flipvariation_gammaAIF.py',
        "--weights-dir",
        os.path.abspath(run_dir),
        "--n-samples",
        "100",
    ]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)


cmd = [
        sys.executable,
        'sim_inference_SNRvariation_gammaAIF.py',
        "--weights-dir",
        os.path.abspath(run_dir),
        "--n-samples",
        "100",
    ]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)


    
print("\n--- Launching in-vivo inference ---")
invivo_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invivo_inference.py")
invivo_data_dir = os.path.dirname(os.path.abspath(invivo_data_path))

if not os.path.isfile(invivo_script):
    print(f"Skipping in-vivo inference: script not found at {invivo_script}")
elif not os.path.isdir(invivo_data_dir):
    print(f"Skipping in-vivo inference: INVIVO_DATA_PATH is not a directory: {invivo_data_dir}")
else:
    cmd = [
        sys.executable,
        invivo_script,
        "--training_info_dir",
        os.path.abspath(run_dir),
        "--data_path",
        invivo_data_path
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    
    


    

    

    

    
    



