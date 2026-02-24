import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from two_compartment_generator import TwoCompartmentHPDataGenerator
import os
from datetime import datetime
import pandas as pd
from hybrid_model_utils import HybridMultiHead, plot_prediction_scatter2

n_samples = 50


# 1. Generate data
time_points=np.arange(0, 32, 2) # 16 time points from 0 to 30s with TR=2s
n_timepoints = len(time_points)
noise = 0.05  # Example noise level
generator = TwoCompartmentHPDataGenerator(time_points=time_points)
X, y = generator.generate_dataset(n_samples=n_samples, noise_std=noise)

# 2. Split channels
X_raw = X.copy()  # raw data for vb
X_norm = X 
n_output_params = 3

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
run_dir = os.path.join(output_dir, "noiselevel_0.05_20251020-022950")
weights_path = os.path.join(run_dir, "trained_hybrid_positive.pth")
# run_dir = os.path.join(output_dir, "noiselevel_0.05_20250912-081050")
# run_dir = os.path.join(output_dir, "noiselevel_0.05_20250804-133211")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(run_dir, f"simulatedData_{timestamp}")
os.makedirs(run_dir, exist_ok=True)




# Plot comparison of X_raw vs X_norm
import matplotlib.pyplot as plt

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
    axes[row, col].plot(time_points, X_raw[sample_idx, :, 0], 'b-', label='Pyruvate (Raw)', linewidth=2)
    axes[row, col].plot(time_points, X_raw[sample_idx, :, 1], 'r-', label='Lactate (Raw)', linewidth=2)
    
    # Plot normalized on secondary y-axis
    ax2 = axes[row, col].twinx()
    ax2.plot(time_points, X_norm[sample_idx, :, 0], 'b--', label='Pyruvate (Norm)', alpha=0.7)
    ax2.plot(time_points, X_norm[sample_idx, :, 1], 'r--', label='Lactate (Norm)', alpha=0.7)
    
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
Xr_test = X_raw.reshape(X.shape[0], -1)
Xn_test = X_norm.reshape(X.shape[0], -1)
Xr_test_tensor = torch.tensor(Xr_test, dtype=torch.float32)
Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.float32)

# 1. Load trained model
input_dim_raw = 16*2  # e.g., 12 timepoints x 2 channels flattened
input_dim_norm = 16*2
model = HybridMultiHead(input_dim_raw=input_dim_raw, input_dim_norm=input_dim_norm)
model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
model.eval()


#noise_levels = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.50, 1.0]
# noise_levels = [0.001,0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
t1pyrs = [20, 30.0, 40, 50, 60]  # Example T1 values for pyruvate
t1lacs = [15, 25.0, 35, 45, 55]  # Example T1 values for lactate
noise_levels = [0.05] * len(t1pyrs)  # Keep noise levels consistent with T1 variations

# cov shape: (n_noise_levels, model_type(0=NN,1=Traditional), param_index(kPL=0,kVE=1,vB=2))
cov = np.zeros((len(noise_levels), 2, 3))
T1s = np.zeros((len(noise_levels), 2))
# Arrays to hold mean predicted parameters per noise level (NN and Traditional)
mean_pred_nn = np.full((len(noise_levels), 3), np.nan)  # kPL,kVE,vB
mean_pred_tr = np.full((len(noise_levels), 3), np.nan)
# Arrays to hold coefficient of determination (R²) per noise level
r2_nn = np.full((len(noise_levels), 3), np.nan)
r2_tr = np.full((len(noise_levels), 3), np.nan)
# Bootstrap std estimates for R² (for shading)
r2_nn_std = np.full((len(noise_levels), 3), np.nan)
r2_tr_std = np.full((len(noise_levels), 3), np.nan)

# for noise_idx, noise in enumerate(noise_levels):
for noise_idx, (t1pyr, t1lac) in enumerate(zip(t1pyrs, t1lacs)):
    r2_kpl = 0
    r2_kve = 0
    r2_vb  = 0

    noise = 0.05 
    generator = TwoCompartmentHPDataGenerator(time_points=time_points, r1p_range=(1/t1pyr, 1/t1pyr), r1l_range=(1/t1lac, 1/t1lac))
    X, y = generator.generate_dataset(n_samples=n_samples, noise_std=noise)
    X_raw = X.copy()  # raw data for vb
    X_norm = X 
    # 3. Flatten inputs
    Xr_test = X_raw.reshape(X.shape[0], -1)
    Xn_test = X_norm.reshape(X.shape[0], -1)
    Xr_test_tensor = torch.tensor(Xr_test, dtype=torch.float32)
    Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y, dtype=torch.float32)   

    pyr_signals = X[:, :, 0]
    lac_signals = X[:, :, 1]

    # Compute SNR as mean amplitude divided by noise std
    snr_pyr = np.max(np.abs(pyr_signals)) / noise
    snr_lac = np.max(np.abs(lac_signals)) / noise
    T1s[noise_idx, 0] = t1pyr
    T1s[noise_idx, 1] = t1lac


    with torch.no_grad():
        y_pred = model(Xn_test_tensor, Xr_test_tensor).numpy()
        y_true = y_test_tensor.numpy()

        r2_kpl = 1 - np.sum((y_pred[:, 0] - y_true[:, 0])**2) / np.sum((y_true[:, 0] - np.mean(y_true[:, 0]))**2)
        r2_kve = 1 - np.sum((y_pred[:, 1] - y_true[:, 1])**2) / np.sum((y_true[:, 1] - np.mean(y_true[:, 1]))**2)
        r2_vb  = 1 - np.sum((y_pred[:, 2] - y_true[:, 2])**2) / np.sum((y_true[:, 2] - np.mean(y_true[:, 2]))**2)

        # Plot true vs. predicted for neural network
        import matplotlib.pyplot as plt
        def plot_true_vs_pred(y_true, y_pred, title_suffix):
            import matplotlib.pyplot as plt
            import numpy as np

            labels = ['kPL', 'kVE', 'vB']
            plt.figure(figsize=(15, 4))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
                plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                        [y_true[:, i].min(), y_true[:, i].max()], 'r--')

                # Compute R²
                ss_res = np.sum((y_pred[:, i] - y_true[:, i]) ** 2)
                ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
                r2 = 1 - ss_res / ss_tot

                plt.xlabel("True")
                plt.ylabel("Predicted")
                plt.title(f"{labels[i]}: R²={r2:.3f}\n{title_suffix}")

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"true_vs_pred_{title_suffix}.png"))
            plt.close()
            import pandas as pd
            df = pd.DataFrame({
                'kPL_true': y_true[:, 0],
                'kVE_true': y_true[:, 1],
                'vB_true': y_true[:, 2],
                'kPL_pred': y_pred[:, 0],
                'kVE_pred': y_pred[:, 1],
                'vB_pred': y_pred[:, 2],
            })
            df.to_csv(os.path.join(run_dir, f"predictions_vs_truth_{title_suffix}.csv"), index=False)

        plot_true_vs_pred(y_true, y_pred, f"{t1pyr}__{t1lac}_Neural Network")
        
        
    # # === Plotting ===
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = os.path.join(run_dir, f'NeuralNetwork_R1_PL_{int(t1pyr)}_{int(t1lac)}')
    plot_prediction_scatter2(y_true, y_pred, saveName=fname)

    # Compute CoV (coefficient of variation) robustly for neural-network predictions
    # per-parameter (kPL=0,kVE=1,vB=2). Use nan-aware stats and guard against small means.
    y_pred_mean = np.nanmean(y_pred, axis=0)
    y_pred_std = np.nanstd(y_pred, axis=0)
    eps = 1e-12
    for p_idx in range(3):
        mean_p = y_pred_mean[p_idx]
        std_p = y_pred_std[p_idx]
        if np.isfinite(mean_p) and np.abs(mean_p) > eps:
            cov[noise_idx, 0, p_idx] = std_p / np.abs(mean_p)
        else:
            cov[noise_idx, 0, p_idx] = np.nan
    # Store mean NN predictions (per-parameter) for this noise level
    if y_pred.shape[1] >= 3:
        mean_pred_nn[noise_idx, :] = np.nanmean(y_pred, axis=0)[:3]
    else:
        tmp = np.nanmean(y_pred, axis=0)
        mean_pred_nn[noise_idx, :tmp.shape[0]] = tmp
    # Compute per-parameter R^2 (NN)
    for p_idx in range(3):
        num = np.nansum((y_pred[:, p_idx] - y_true[:, p_idx])**2)
        den = np.nansum((y_true[:, p_idx] - np.nanmean(y_true[:, p_idx]))**2)
        if den > 0:
            r2_nn[noise_idx, p_idx] = 1 - num/den
        else:
            r2_nn[noise_idx, p_idx] = np.nan

    # Bootstrap estimate of R^2 std (NN)
    try:
        B = 200
        r2b = np.full((B, 3), np.nan)
        n_samples_local = y_true.shape[0]
        if n_samples_local > 1:
            for b in range(B):
                idxs = np.random.randint(0, n_samples_local, size=n_samples_local)
                y_t_b = y_true[idxs]
                y_p_b = y_pred[idxs]
                for p_idx in range(3):
                    den_b = np.nansum((y_t_b[:, p_idx] - np.nanmean(y_t_b[:, p_idx]))**2)
                    if den_b > 0:
                        num_b = np.nansum((y_p_b[:, p_idx] - y_t_b[:, p_idx])**2)
                        r2b[b, p_idx] = 1 - num_b/den_b
            r2_nn_std[noise_idx, :] = np.nanstd(r2b, axis=0)
    except Exception:
        pass

    # # Traditional fit comparison
    traditional_preds = []


    
    from fit_two_compartment import fit_traditional_2c_model


    X_test = X

    print("Using traditional fittings...")
    import time
    start_time = time.time()
    for j in range(len(X_test)):
        pyr = X_test[j, :, 0]
        lac = X_test[j, :, 1]
        try:
            params, _, success = fit_traditional_2c_model(
                time_points, pyr, lac, estimate_r1=False, use_3state=(n_output_params==4))
            if n_output_params == 4:
                traditional_preds.append(params[:4])  # [kpl, kve, vb, Ktrans]
            else:
                traditional_preds.append(params[:3])  # [kpl, kve, vb]
        except:
            traditional_preds.append([np.nan] * n_output_params)
    tradfit_time = time.time() - start_time
    print(f"Traditional model training time: {tradfit_time:.2f} seconds")
    y_pred_fit = np.array(traditional_preds)
    fname = os.path.join(run_dir, f'TraditionalFitOfGeneratorData_R1_PL_{int(t1pyr)}_{int(t1lac)}')
    plot_prediction_scatter2(y_true, y_pred_fit, saveName=fname)
    # CoV for traditional (NLLS) fits: per-parameter using nan-aware statistics
    y_pred_mean = np.nanmean(y_pred_fit, axis=0)
    y_pred_std = np.nanstd(y_pred_fit, axis=0)
    eps = 1e-12
    # If y_pred_fit has fewer than 3 columns (rare), pad with NaN
    if y_pred_mean.shape[0] < 3:
        y_pred_mean = np.pad(y_pred_mean, (0, 3 - y_pred_mean.shape[0]), constant_values=np.nan)
        y_pred_std = np.pad(y_pred_std, (0, 3 - y_pred_std.shape[0]), constant_values=np.nan)
    for p_idx in range(3):
        mean_p = y_pred_mean[p_idx]
        std_p = y_pred_std[p_idx]
        if np.isfinite(mean_p) and np.abs(mean_p) > eps:
            cov[noise_idx, 1, p_idx] = std_p / np.abs(mean_p)
        else:
            cov[noise_idx, 1, p_idx] = np.nan
    # Store mean Traditional predictions (per-parameter) for this noise level
    if y_pred_fit.ndim == 2 and y_pred_fit.shape[1] >= 3:
        mean_pred_tr[noise_idx, :] = np.nanmean(y_pred_fit, axis=0)[:3]
    else:
        # if fits returned fewer params, pad with NaN
        tmp = np.nanmean(y_pred_fit, axis=0)
        for ii in range(min(len(tmp), 3)):
            mean_pred_tr[noise_idx, ii] = tmp[ii]
    # Compute per-parameter R^2 (Traditional)
    if y_pred_fit.ndim == 2:
        for p_idx in range(min(3, y_pred_fit.shape[1])):
            num = np.nansum((y_pred_fit[:, p_idx] - y_true[:, p_idx])**2)
            den = np.nansum((y_true[:, p_idx] - np.nanmean(y_true[:, p_idx]))**2)
            if den > 0:
                r2_tr[noise_idx, p_idx] = 1 - num/den
            else:
                r2_tr[noise_idx, p_idx] = np.nan

    # Bootstrap estimate of R^2 std (Traditional)
    try:
        if y_pred_fit.ndim == 2 and y_pred_fit.shape[0] > 1:
            B = 200
            r2b = np.full((B, 3), np.nan)
            n_samples_local = y_true.shape[0]
            for b in range(B):
                idxs = np.random.randint(0, n_samples_local, size=n_samples_local)
                y_t_b = y_true[idxs]
                y_pf_b = y_pred_fit[idxs]
                for p_idx in range(min(3, y_pred_fit.shape[1])):
                    den_b = np.nansum((y_t_b[:, p_idx] - np.nanmean(y_t_b[:, p_idx]))**2)
                    if den_b > 0 and y_pf_b.shape[0] > 0:
                        num_b = np.nansum((y_pf_b[:, p_idx] - y_t_b[:, p_idx])**2)
                        r2b[b, p_idx] = 1 - num_b/den_b
            r2_tr_std[noise_idx, :] = np.nanstd(r2b, axis=0)
    except Exception:
        pass

    
## Save T1s and per-parameter CoV to Excel file
# Expand cov into separate 1-D columns so pandas DataFrame is happy
data_dict = {
    'Noise_Level': noise_levels,
    'T1_Pyruvate': T1s[:, 0],
    'T1_Lactate': T1s[:, 1],
    # Neural network CoV per parameter
    'COV_NN_kPL': cov[:, 0, 0],
    'COV_NN_kVE': cov[:, 0, 1],
    'COV_NN_vB':  cov[:, 0, 2],
    # Traditional fit CoV per parameter
    'COV_TR_kPL': cov[:, 1, 0],
    'COV_TR_kVE': cov[:, 1, 1],
    'COV_TR_vB':  cov[:, 1, 2],
}

df = pd.DataFrame(data_dict)
excel_filename = os.path.join(run_dir, f'SNR_COV_Analysis_{timestamp}.xlsx')
df.to_excel(excel_filename, index=False)
print(f"Data saved to {excel_filename}")

# plot coefficient of variation vs SNR (kPL only)
import matplotlib.pyplot as plt

plt.figure()
ax = plt.gca()
# kPL CoV is parameter index 0
plt.plot(T1s[:, 0], cov[:, 0, 0], label='NN CoV (kPL)', marker='o', color='blue')
plt.plot(T1s[:, 0], cov[:, 1, 0], label='NLLS CoV (kPL)', marker='x', color='orange')
plt.xlabel('T1 Pyruvate (s)')
plt.ylabel('Coefficient of Variation (kPL)')
plt.title('Coefficient of Variation (kPL) vs T1 Pyruvate (s)')
plt.legend()

# Enable minor ticks on y-axis
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))  # Adjust spacing as needed
ax.tick_params(axis='y', which='minor', length=3, width=0.5)
ax.grid(True, which='minor', alpha=0.3, linestyle=':')
ax.grid(True, which='major', alpha=0.7)

plt.savefig(os.path.join(run_dir, 'Cov_vs_T1P_kPL.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Plot SNR vs R² (true vs predicted) for each parameter ---
params = ['kPL', 'kVE', 'vB']
for p_idx, pname in enumerate(params):
    plt.figure()
    ax = plt.gca()
    x = T1s[:, 0]
    y_nn = r2_nn[:, p_idx]
    y_tr = r2_tr[:, p_idx]

    plt.plot(x, y_nn, label=f'NN R² {pname}', marker='o', color='blue')
    plt.plot(x, y_tr, label=f'NLLS R² {pname}', marker='x', color='orange')
    plt.xlabel('Pyruvate T1')
    plt.ylabel('Coefficient of Determination (R²)')
    plt.title(f'Pyruvate T1 vs R² ({pname})')
    plt.legend()
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, which='major', alpha=0.7)
    ax.grid(True, which='minor', alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f'T1P_vs_R2_{pname}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- Combined plot: SNR vs R² for all parameters (NN and Traditional) ---
# plt.figure(figsize=(8, 6))
# ax = plt.gca()
# colors = {'kPL': 'C0', 'kVE': 'C1', 'vB': 'C2'}
# markers = {'NN': 'o', 'TR': 'x'}
# for p_idx, pname in enumerate(params):
#     x = T1s[:, 0]
#     plt.plot(x, r2_nn[:, p_idx], label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], linestyle='-')
#     plt.plot(x, r2_tr[:, p_idx], label=f'TR R² {pname}', marker=markers['TR'], color=colors[pname], linestyle='--')
#     # Add shaded error bands if bootstrap std available
#     if np.any(np.isfinite(r2_nn_std[:, p_idx])):
#         plt.fill_between(x, r2_nn[:, p_idx] - r2_nn_std[:, p_idx], r2_nn[:, p_idx] + r2_nn_std[:, p_idx],
#                          color=colors[pname], alpha=0.15)
#     if np.any(np.isfinite(r2_tr_std[:, p_idx])):
#         plt.fill_between(x, r2_tr[:, p_idx] - r2_tr_std[:, p_idx], r2_tr[:, p_idx] + r2_tr_std[:, p_idx],
#                          color=colors[pname], alpha=0.08)

# plt.xlabel('Pyruvate T1 (s)')
# plt.ylabel('Coefficient of Determination (R²)')
# plt.title('Pyruvate T1 vs R² (all parameters)')
# plt.legend(ncol=2)
# ax.set_ylim(-0.5, 1.05)
# ax.grid(True, which='major', alpha=0.7)
# ax.grid(True, which='minor', alpha=0.3, linestyle=':')
# plt.tight_layout()
# plt.savefig(os.path.join(run_dir, 'T1P_vs_R2_all_params.png'), dpi=300, bbox_inches='tight')
# plt.close()


plt.figure(figsize=(8, 6))
# ax = plt.gca()
fig, ax1 = plt.subplots()
colors = {'kPL': 'C0', 'kVE': 'C1', 'vB': 'C2'}
markers = {'NN': 'o', 'TR': 'x'}
for p_idx, pname in enumerate(params):
    x = T1s[:, 0]
    ax1.plot(x, r2_nn[:, p_idx], label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], linestyle='-')
    ax1.plot(x, r2_tr[:, p_idx], label=f'TR R² {pname}', marker=markers['TR'], color=colors[pname], linestyle='--')
    # Add shaded error bands if bootstrap std available
    if np.any(np.isfinite(r2_nn_std[:, p_idx])):
        ax1.fill_between(x, r2_nn[:, p_idx] - r2_nn_std[:, p_idx], r2_nn[:, p_idx] + r2_nn_std[:, p_idx],
                         color=colors[pname], alpha=0.15)
    if np.any(np.isfinite(r2_tr_std[:, p_idx])):
        ax1.fill_between(x, r2_tr[:, p_idx] - r2_tr_std[:, p_idx], r2_tr[:, p_idx] + r2_tr_std[:, p_idx],
                         color=colors[pname], alpha=0.08)

ax1.set_xlabel('Pyruvate T1 (s)')
ax1.set_ylabel('Coefficient of Determination (R²)')
ax1.set_title('Pyruvate T1 vs R² (all parameters)')
# ax1.legend(ncol=2)
ax1.set_ylim(-0.5, 1.05)
ax1.grid(True, which='major', alpha=0.7)
ax1.grid(True, which='minor', alpha=0.3, linestyle=':')

ax2 = ax1.twiny()
ax2.set_xlabel('Lactate T1 (s)')
# ax2.set_xlim(ax1.get_xlim())
for p_idx, pname in enumerate(params):
    x = T1s[:, 1]
    ax2.plot(x, r2_nn[:, p_idx], label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], linestyle='-')
    ax2.plot(x, r2_tr[:, p_idx], label=f'TR R² {pname}', marker=markers['TR'], color=colors[pname], linestyle='--')
    # Add shaded error bands if bootstrap std available
    if np.any(np.isfinite(r2_nn_std[:, p_idx])):
        ax2.fill_between(x, r2_nn[:, p_idx] - r2_nn_std[:, p_idx], r2_nn[:, p_idx] + r2_nn_std[:, p_idx],
                         color=colors[pname], alpha=0.15)
    if np.any(np.isfinite(r2_tr_std[:, p_idx])):
        ax2.fill_between(x, r2_tr[:, p_idx] - r2_tr_std[:, p_idx], r2_tr[:, p_idx] + r2_tr_std[:, p_idx],
                         color=colors[pname], alpha=0.08)


plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'T1_vs_R2_all_params.png'), dpi=300, bbox_inches='tight')
plt.close()




# import matplotlib.pyplot as plt
# import numpy as np
# # Create some mock data
# t = np.arange(0.01, 20.0, 0.001)
# data1 = np.exp(t)
# data2 = np.sin(0.3 * np.pi * t)
# fig, ax1 = plt.subplots()
# # Plot the first dataset
# color = 'tab:blue'
# ax1.set_xlabel('exp', color=color)
# ax1.set_ylabel('time (s)')
# ax1.plot(data1, t, color=color)
# ax1.tick_params(axis='x', labelcolor=color)
# # Create a twin Axes sharing the y-axis
# ax2 = ax1.twiny()
# # Plot the second dataset
# color = 'tab:green'
# ax2.set_xlabel('sin', color=color)
# ax2.plot(data2, t, color=color)
# ax2.tick_params(axis='x', labelcolor=color)
# fig.suptitle('matplotlib.pyplot.twiny() function Example', fontweight="bold")
# plt.show()

# plt.figure(figsize=(8, 6))
# ax = plt.gca()
# colors = {'kPL': 'C0', 'kVE': 'C1', 'vB': 'C2'}
# markers = {'NN': 'o', 'TR': 'x'}
# for p_idx, pname in enumerate(params):
#     x = T1s[:, 1]
#     plt.plot(x, r2_nn[:, p_idx], label=f'NN R² {pname}', marker=markers['NN'], color=colors[pname], linestyle='-')
#     plt.plot(x, r2_tr[:, p_idx], label=f'TR R² {pname}', marker=markers['TR'], color=colors[pname], linestyle='--')
#     # Add shaded error bands if bootstrap std available
#     if np.any(np.isfinite(r2_nn_std[:, p_idx])):
#         plt.fill_between(x, r2_nn[:, p_idx] - r2_nn_std[:, p_idx], r2_nn[:, p_idx] + r2_nn_std[:, p_idx],
#                          color=colors[pname], alpha=0.15)
#     if np.any(np.isfinite(r2_tr_std[:, p_idx])):
#         plt.fill_between(x, r2_tr[:, p_idx] - r2_tr_std[:, p_idx], r2_tr[:, p_idx] + r2_tr_std[:, p_idx],
#                          color=colors[pname], alpha=0.08)

# plt.xlabel('Lactate T1 (s)')
# plt.ylabel('Coefficient of Determination (R²)')
# plt.title('Lactate T1 vs R² (all parameters)')
# plt.legend(ncol=2)
# ax.set_ylim(-0.5, 1.05)
# ax.grid(True, which='major', alpha=0.7)
# ax.grid(True, which='minor', alpha=0.3, linestyle=':')




# plt.tight_layout()
# plt.savefig(os.path.join(run_dir, 'T1L_vs_R2_all_params.png'), dpi=300, bbox_inches='tight')
# plt.close()






print("\n--- Step 12: Generate Summary Report ---")
with open(os.path.join(run_dir, 'summary_report.md'), 'w') as f:
    f.write(f"# Hyperpolarized 13C MRI Analysis Demo Summary\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"## Dataset\n")
    f.write(f"- Total samples: {n_samples}\n")
    f.write(f"- Test samples: {Xr_test.shape[0]}\n")
    f.write(f"- Time points: {n_timepoints} (TR=5s, 0-{(n_timepoints-1)*5}s)\n\n")
    
    f.write(f"## Parameter Ranges\n")
    f.write(f"- kPL: {generator.kpl_range[0]:.3f} - {generator.kpl_range[1]:.3f} s^-1\n")
    
    f.write(f"## Training\n")
    f.write(f"- Initial learning rate: 0.001\n")
    # f.write(f"- Traditional fitting time: {tradfit_time:.2f} seconds\n\n")
    
    f.write(f"Test R² for kpl: {r2_kpl:.3f}")
    f.write(f"Test R² for kve: {r2_kve:.3f}")
    f.write(f"Test R² for vb:  {r2_vb:.3f}")
    f.write(f"\n\n## SNR Analysis\n")
    f.write(f"- Mean SNR for Pyruvate: {np.mean(snr_pyr):.2f}\n")
    f.write(f"- Mean SNR for Lactate: {np.mean(snr_lac):.2f}\n")
    f.write(f"- Minimum SNR for Pyruvate: {np.min(snr_pyr):.2f}\n")
    f.write(f"- Minimum SNR for Lactate: {np.min(snr_lac):.2f}\n")
    f.write(f"- Maximum SNR for Pyruvate: {np.max(snr_pyr):.2f}\n")
    f.write(f"- Maximum SNR for Lactate: {np.max(snr_lac):.2f}\n")
    
        # Display statistics and plot comparison
    f.write("\n=== X_raw Statistics ===")
    f.write(f"Maximum: {np.max(X):.6f}\n")
    f.write(f"Minimum: {np.min(X):.6f}\n")
    f.write(f"Mean: {np.mean(X):.6f}\n")
    f.write(f"Shape: {X.shape}\n")

    f.write("\n=== X_norm Statistics ===")
    f.write(f"Maximum: {np.max(X_norm):.6f}\n")
    f.write(f"Minimum: {np.min(X_norm):.6f}\n")
    f.write(f"Mean: {np.mean(X_norm):.6f}\n")
    f.write(f"Shape: {X_norm.shape}\n")
    
    



